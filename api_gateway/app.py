"""
API Gateway — port 8000

엔드포인트:
  POST /api/search             search-engine 프록시
  POST /api/personalized-search search 후보를 rec-models로 개인화 재정렬
  GET  /api/recommend          Redis 세션 붙여서 rec-models 프록시
  POST /api/events             Redis에 클릭/구매 이벤트 저장
  GET  /api/features/{user_id} Redis 유저 피처 조회
  GET  /api/images/{article_id} 상품 이미지 반환
  POST /api/onboarding         LLM 기반 콜드 스타트 페르소나 생성 (기능 C)
  POST /api/budget-set         예산 기반 패션 세트 추천 (기능 D)
  GET  /health
"""

import csv
import hashlib
import json
import logging
import os
import re
import traceback
import httpx
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel

from feature_store import RedisFeatureStore

# ── 서비스 URL (docker-compose 서비스명 또는 환경변수로 오버라이드) ──
SEARCH_URL = os.getenv("SEARCH_ENGINE_URL", "http://search-engine:8002")
REC_URL = os.getenv("REC_MODELS_URL", "http://rec-models:8003")
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

DEFAULT_IMAGE_ROOT = Path("/app/data/raw/images")
LOCAL_IMAGE_ROOT = Path(__file__).resolve().parents[1] / "data" / "raw" / "images"
IMAGE_ROOT = Path(os.getenv("IMAGE_ROOT", str(DEFAULT_IMAGE_ROOT)))
if not IMAGE_ROOT.exists() and LOCAL_IMAGE_ROOT.exists():
    IMAGE_ROOT = LOCAL_IMAGE_ROOT

ARTICLES_PATH = Path("/app/data/processed/articles_feature.csv")
# item_features_{test,dev,prod}.csv — avg_price 컬럼 포함
_ITEM_FEATURES_CANDIDATES = [
    Path("/app/data/processed/item_features_test.csv"),
    Path("/app/data/processed/item_features_dev.csv"),
    Path("/app/data/processed/item_features.csv"),
]
# H&M 정규화 가격 → KRW 환산 계수 (중앙값 0.025 ≈ 25,000원 기준)
PRICE_KRW_FACTOR = 1_000_000

feature_store: RedisFeatureStore
# article_id → {name, brand, category, color, product_type, price}
article_meta: dict[str, dict] = {}


def _brand_label(department_name: str | None) -> str:
    department = (department_name or "").strip()
    return f"H&M · {department}" if department else "H&M"


def _load_article_meta() -> dict[str, dict]:
    meta: dict[str, dict] = {}
    if not ARTICLES_PATH.exists():
        return meta
    with ARTICLES_PATH.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            aid = row.get("article_id", "").strip()
            if aid:
                department_name = row.get("department_name", "")
                meta[aid] = {
                    "name": row.get("prod_name", ""),
                    "brand": _brand_label(department_name),
                    "category": row.get("category", ""),
                    "color": row.get("color", ""),
                    "product_type": row.get("product_type_name", ""),
                    "price": 0,
                }

    # item_features CSV에서 avg_price 보강. dev/test/prod 산출물의 포함
    # item이 다를 수 있으므로 첫 파일에서 멈추지 않고 가능한 값을 합친다.
    for path in _ITEM_FEATURES_CANDIDATES:
        if path.exists():
            with path.open(encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    aid = row.get("article_id", "").strip()
                    if aid in meta and not meta[aid]["price"]:
                        try:
                            raw = float(row.get("avg_price", 0) or 0)
                            if raw > 0:
                                meta[aid]["price"] = int(raw * PRICE_KRW_FACTOR)
                        except (ValueError, TypeError):
                            pass

    return meta


@asynccontextmanager
async def lifespan(app: FastAPI):
    global feature_store, article_meta
    feature_store = RedisFeatureStore(host=REDIS_HOST, port=REDIS_PORT)
    article_meta = _load_article_meta()
    yield


app = FastAPI(title="API Gateway", lifespan=lifespan)


def image_path_for_article(article_id: str) -> Path:
    normalized_id = article_id.strip()
    if not normalized_id.isdigit() or len(normalized_id) < 3:
        raise HTTPException(status_code=400, detail="Invalid article_id")
    return IMAGE_ROOT / normalized_id[:3] / f"{normalized_id}.jpg"


# ── 요청/응답 스키마 ──────────────────────────────────────────

class SearchRequest(BaseModel):
    query: str = ""
    image_base64: str | None = None
    top_k: int = 10


class PersonalizedSearchRequest(SearchRequest):
    user_id: str
    top_n: int = 10
    persona_hint: str | None = None


class EventRequest(BaseModel):
    user_id: str
    article_id: str | None = None
    item_id: str | None = None  # frontend 호환 (article_id 우선)
    event_type: str  # "click" | "view" | "cart" | "purchase" | "search"
    category: str | None = None
    query_text: str | None = None


class OnboardingRequest(BaseModel):
    user_id: str
    description: str  # 자유 입력 (예: "미니멀한 스타일 좋아하는 20대 여성입니다")
    style_choices: list[str] = []  # 선택지 (예: ["casual", "minimal", "sporty"])
    budget_range: str | None = None  # "low" | "mid" | "high"


VALID_PERSONAS = (
    "trendsetter", "practical", "value", "brand_loyal", "impulse",
    "careful", "repeat_stable", "color_focus", "category_focus",
)


def _normalize_persona_scores(persona_scores: dict[str, int | float]) -> dict[str, int]:
    filtered = {
        key: max(0, int(value))
        for key, value in persona_scores.items()
        if key in VALID_PERSONAS and isinstance(value, (int, float))
    }
    total = sum(filtered.values())
    if total == 0:
        return {"practical": 35, "careful": 25, "value": 20, "trendsetter": 20}

    sorted_keys = sorted(filtered, key=filtered.get, reverse=True)
    normalized = {key: round(filtered[key] * 100 / total) for key in sorted_keys}
    diff = 100 - sum(normalized.values())
    normalized[sorted_keys[0]] += diff
    return normalized


def _fallback_persona_scores(req: OnboardingRequest) -> dict[str, int]:
    """Deterministic local persona inference for quota/network fallback."""

    text = " ".join([req.description, " ".join(req.style_choices), req.budget_range or ""]).lower()
    scores = {persona: 0 for persona in VALID_PERSONAS}

    keyword_rules = {
        "value": ("가성비", "할인", "세일", "저렴", "가격", "budget", "cheap", "sale", "low"),
        "practical": ("실용", "기본", "출근", "편한", "활용", "minimal", "classic", "daily"),
        "careful": ("신중", "비교", "리뷰", "고민", "오래", "compare", "review"),
        "trendsetter": ("트렌드", "유행", "새로운", "힙", "street", "trend", "new"),
        "impulse": ("충동", "바로", "즉흥", "눈에 띄", "impulse"),
        "brand_loyal": ("브랜드", "brand", "익숙", "선호 브랜드"),
        "repeat_stable": ("반복", "재구매", "비슷한", "꾸준", "stable", "repeat"),
        "color_focus": ("색", "컬러", "검정", "블랙", "화이트", "파랑", "빨강", "color", "black", "white"),
        "category_focus": ("아우터", "자켓", "재킷", "니트", "셔츠", "원피스", "운동복", "outer", "jacket", "knit", "dress"),
    }
    for persona, keywords in keyword_rules.items():
        scores[persona] += sum(18 for keyword in keywords if keyword in text)

    if req.budget_range == "low":
        scores["value"] += 30
    elif req.budget_range == "mid":
        scores["practical"] += 18
        scores["careful"] += 12
    elif req.budget_range == "high":
        scores["brand_loyal"] += 18
        scores["trendsetter"] += 12

    for style in req.style_choices:
        style_key = style.lower()
        if style_key in {"minimal", "classic", "casual"}:
            scores["practical"] += 14
            scores["careful"] += 8
        elif style_key == "street":
            scores["trendsetter"] += 18
            scores["impulse"] += 8
        elif style_key == "sporty":
            scores["practical"] += 12
            scores["trendsetter"] += 8
        elif style_key == "feminine":
            scores["trendsetter"] += 8
            scores["color_focus"] += 8

    return _normalize_persona_scores(scores)


QUERY_INTEREST_KEYWORDS: dict[str, tuple[str, ...]] = {
    "Ladieswear": (
        "women", "woman", "ladies", "dress", "skirt", "blouse", "jacket", "outer", "coat",
        "여성", "여자", "원피스", "스커트", "블라우스", "자켓", "재킷", "아우터", "코트",
    ),
    "Menswear": (
        "men", "man", "mens", "shirt", "suit", "jacket", "outer", "coat",
        "남성", "남자", "셔츠", "정장", "자켓", "재킷", "아우터", "코트",
    ),
    "Divided": (
        "denim", "jeans", "street", "casual", "청바지", "데님", "스트릿", "캐주얼",
    ),
    "Sport": (
        "sport", "sports", "active", "training", "스포츠", "운동", "트레이닝",
    ),
    "Kids": (
        "kids", "baby", "child", "키즈", "아동", "아이", "베이비",
    ),
    "Lingeries/Tights": (
        "lingerie", "tights", "underwear", "속옷", "타이츠", "스타킹",
    ),
}
QUERY_INTEREST_CATEGORIES = tuple(QUERY_INTEREST_KEYWORDS.keys())


def _infer_session_interest_from_query_keywords(query_text: str | None) -> dict[str, int]:
    if not query_text:
        return {}

    normalized_query = query_text.lower()
    inferred: dict[str, int] = {}
    for category, keywords in QUERY_INTEREST_KEYWORDS.items():
        if any(keyword.lower() in normalized_query for keyword in keywords):
            inferred[category] = inferred.get(category, 0) + 2
    return inferred


# ── 유틸: LLM 호출 ───────────────────────────────────────────

async def _call_gemini(prompt: str, json_mode: bool = False) -> str:
    """Gemini Flash API 호출. GEMINI_API_KEY 미설정 시 빈 문자열 반환.

    json_mode=True이면 JSON 외 출력을 차단해 파싱 안정성을 높인다.
    """
    logging.warning("[GEMINI CALL] %s", "".join(traceback.format_stack()[-4:-1]))
    if not GEMINI_API_KEY:
        return ""

    generation_config: dict = {"temperature": 0.7 if not json_mode else 0.1}
    if json_mode:
        generation_config["responseMimeType"] = "application/json"

    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent?key={GEMINI_API_KEY}",
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": generation_config,
            },
        )
        resp.raise_for_status()
        return resp.json()["candidates"][0]["content"]["parts"][0]["text"]


def _coerce_interest_score(value: object) -> int:
    try:
        numeric_value = int(round(float(value)))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0
    return max(0, min(numeric_value, 3))


def _parse_query_interest_payload(payload: object) -> dict[str, int]:
    if not isinstance(payload, dict):
        return {}

    raw_interest = payload.get("interest", payload)
    if not isinstance(raw_interest, dict):
        return {}

    normalized_by_key = {str(key).strip().lower(): value for key, value in raw_interest.items()}
    inferred: dict[str, int] = {}
    for category in QUERY_INTEREST_CATEGORIES:
        score = _coerce_interest_score(normalized_by_key.get(category.lower()))
        if score > 0:
            inferred[category] = score
    return inferred


async def _infer_session_interest_from_query_llm(query_text: str) -> dict[str, int]:
    if not GEMINI_API_KEY or not query_text.strip():
        return {}

    category_list = ", ".join(QUERY_INTEREST_CATEGORIES)
    prompt = (
        "Infer lightweight fashion recommendation interests from the search query.\n"
        "Return only JSON. Use only these category keys: "
        f"{category_list}.\n"
        "Each score must be an integer from 0 to 3. Use 0 when unrelated. "
        "Keep scores conservative because this is a short-lived search signal.\n\n"
        f"Search query: {query_text}\n\n"
        "JSON format:\n"
        '{"interest":{"Ladieswear":0,"Menswear":0,"Divided":0,"Sport":0,"Kids":0,"Lingeries/Tights":0}}'
    )

    try:
        llm_text = await _call_gemini(prompt, json_mode=True)
        return _parse_query_interest_payload(json.loads(llm_text))
    except Exception:
        return {}


async def _infer_session_interest_from_query(query_text: str | None) -> dict[str, int]:
    normalized_query = (query_text or "").strip()
    if not normalized_query:
        return {}

    cached_interest = feature_store.get_query_interest_cache(normalized_query)
    if cached_interest is not None:
        return {
            category: _coerce_interest_score(score)
            for category, score in cached_interest.items()
            if category in QUERY_INTEREST_CATEGORIES and _coerce_interest_score(score) > 0
        }

    inferred_interest = _infer_session_interest_from_query_keywords(normalized_query)

    feature_store.set_query_interest_cache(normalized_query, inferred_interest)
    return inferred_interest


# ── 엔드포인트 ────────────────────────────────────────────────

def _has_korean(text: str) -> bool:
    return bool(re.search(r"[가-힣]", text))


async def _translate_to_english(query: str) -> str:
    """한국어 패션 검색어를 영어로 번역. 실패 시 원문 반환."""
    prompt = (
        f"Translate this Korean fashion search query to English. "
        f"Return only the translated English text, nothing else.\n\nQuery: {query}"
    )
    try:
        result = await _call_gemini(prompt)
        return result.strip() if result.strip() else query
    except Exception:
        return query


def _enrich_search_results(items: list[dict]) -> list[dict]:
    enriched_results = []
    for item in items:
        pid = str(item.get("product_id", ""))
        meta = article_meta.get(pid, {})
        enriched_results.append({
            **item,
            "name": meta.get("name") or item.get("name") or pid,
            "brand": meta.get("brand") or "H&M",
            "category": meta.get("category", item.get("category", "")),
            "color": meta.get("color", ""),
            "product_type": meta.get("product_type", ""),
            "price": meta.get("price", 0),
        })
    return enriched_results


def _enrich_recommendation_results(items: list[dict]) -> list[dict]:
    enriched_results = []
    for rank, item in enumerate(items, 1):
        pid = str(item.get("product_id", ""))
        meta = article_meta.get(pid, {})
        enriched_results.append({
            **item,
            "rank": rank,
            "name": meta.get("name") or pid,
            "brand": meta.get("brand") or "H&M",
            "category": meta.get("category", ""),
            "color": meta.get("color", ""),
            "product_type": meta.get("product_type", ""),
            "price": meta.get("price", 0),
        })
    return enriched_results


def _top_persona_from_scores(persona_scores: dict[str, int] | None) -> str | None:
    if not persona_scores:
        return None
    return max(persona_scores, key=persona_scores.get)


@app.post("/api/search")
async def search(req: SearchRequest):
    """search-engine으로 검색 요청을 프록시한다.

    한국어 텍스트 쿼리는 Gemini로 영어로 번역 후 CLIP에 전달한다.
    결과에 name/category/color/price enrichment를 적용한다.
    """
    translated_query: str | None = None
    search_payload = req.model_dump()

    if req.query and _has_korean(req.query) and GEMINI_API_KEY:
        translated_query = await _translate_to_english(req.query)
        search_payload["query"] = translated_query

    async with httpx.AsyncClient(timeout=15.0) as client:
        try:
            resp = await client.post(f"{SEARCH_URL}/search", json=search_payload)
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=str(e))
        except httpx.RequestError as e:
            raise HTTPException(status_code=503, detail=f"search-engine 연결 실패: {e}")

    result = resp.json()

    result["results"] = _enrich_search_results(result.get("results", []))

    if translated_query:
        result["original_query"] = req.query
        result["translated_query"] = translated_query

    return result


PERSONALIZED_SEARCH_CANDIDATE_POOL = 80


@app.post("/api/personalized-search")
async def personalized_search(req: PersonalizedSearchRequest):
    """Search broadly, then return both similarity order and personalized order."""

    top_n = max(1, min(int(req.top_n), 100))
    search_top_k = max(top_n, int(req.top_k), PERSONALIZED_SEARCH_CANDIDATE_POOL)
    translated_query: str | None = None
    search_payload = {
        "query": req.query,
        "image_base64": req.image_base64,
        "top_k": search_top_k,
    }

    if req.query and _has_korean(req.query) and GEMINI_API_KEY:
        translated_query = await _translate_to_english(req.query)
        search_payload["query"] = translated_query

    async with httpx.AsyncClient(timeout=15.0) as client:
        try:
            search_resp = await client.post(f"{SEARCH_URL}/search", json=search_payload)
            search_resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=str(e))
        except httpx.RequestError as e:
            raise HTTPException(status_code=503, detail=f"search-engine 연결 실패: {e}")

    search_result = search_resp.json()
    enriched_search_results = _enrich_search_results(search_result.get("results", []))

    inferred_interest = await _infer_session_interest_from_query(req.query)
    if inferred_interest:
        interest = feature_store.get_session_interest(req.user_id)
        for category, score in inferred_interest.items():
            interest[category] = interest.get(category, 0) + score
        feature_store.set_session_interest(req.user_id, interest)
        feature_store.invalidate_recommendation_cache(req.user_id)

    features = feature_store.get_user_features(req.user_id)
    raw_persona_scores = features.get("persona_scores", {})
    persona_scores = _normalize_persona_scores(raw_persona_scores) if raw_persona_scores else {}
    persona_hint = req.persona_hint or _top_persona_from_scores(persona_scores)
    rerank_payload = {
        "user_id": req.user_id,
        "top_n": top_n,
        "search_candidates": [
            {
                "product_id": str(item.get("product_id", "")),
                "score": item.get("score", item.get("similarity")),
            }
            for item in enriched_search_results
            if item.get("product_id") is not None
        ],
        "recent_clicks": features["recent_clicks"],
        "session_interest": features["session_interest"] or None,
        "persona_hint": persona_hint,
        "persona_scores": persona_scores or None,
        "include_recommendation_candidates": False,
        "recommendation_candidate_pool_size": PERSONALIZED_SEARCH_CANDIDATE_POOL,
    }

    async with httpx.AsyncClient(timeout=15.0) as client:
        try:
            rerank_resp = await client.post(f"{REC_URL}/rerank-candidates", json=rerank_payload)
            rerank_resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=str(e))
        except httpx.RequestError as e:
            raise HTTPException(status_code=503, detail=f"rec-models 연결 실패: {e}")

    personalized_result = rerank_resp.json()
    personalized_results = _enrich_recommendation_results(personalized_result.get("recommendations", []))
    pipeline_latency = personalized_result.get("pipeline_latency", {})

    response = {
        "search_results": enriched_search_results[:top_n],
        "personalized_results": personalized_results,
        "search_latency_ms": search_result.get("latency_ms", search_result.get("total_ms")),
        "personalized_latency": pipeline_latency,
        "candidate_summary": personalized_result.get("candidate_summary", {}),
        "session_interest": features["session_interest"],
        "persona": personalized_result.get("persona", persona_hint or "personalized"),
        "persona_scores": persona_scores,
    }
    if translated_query:
        response["original_query"] = req.query
        response["translated_query"] = translated_query
    return response


RECOMMEND_CACHE_TTL = 300  # 5분


def _weight_cache_suffix(weight_params: dict[str, object]) -> str:
    active_weights = {key: value for key, value in weight_params.items() if value is not None}
    if not active_weights:
        return ""
    encoded = json.dumps(active_weights, sort_keys=True, separators=(",", ":"))
    return f":{encoded}"


@app.get("/api/recommend")
async def recommend(
    user_id: str = Query(...),
    top_n: int = Query(10),
    persona_hint: str | None = Query(None),
    personalization_weight: float | None = Query(None, ge=0.0, le=5.0),
    price_weight: float | None = Query(None, ge=0.0, le=5.0),
    popularity_weight: float | None = Query(None, ge=0.0, le=5.0),
    diversity_weight: float | None = Query(None, ge=0.0, le=5.0),
    freshness_weight: float | None = Query(None, ge=0.0, le=5.0),
    exploration_weight: float | None = Query(None, ge=0.0, le=5.0),
    long_tail_weight: float | None = Query(None, ge=0.0, le=5.0),
    include_reasons: bool = Query(False),
):
    """Redis 세션 데이터를 붙여 rec-models로 추천 요청을 프록시한다."""
    features = feature_store.get_user_features(user_id)
    click_count = features["click_count"]
    raw_persona_scores = features.get("persona_scores", {})
    persona_scores = _normalize_persona_scores(raw_persona_scores) if raw_persona_scores else {}
    weight_params = {
        "persona_hint": persona_hint,
        "persona_scores": persona_scores or None,
        "personalization_weight": personalization_weight,
        "price_weight": price_weight,
        "popularity_weight": popularity_weight,
        "diversity_weight": diversity_weight,
        "freshness_weight": freshness_weight,
        "exploration_weight": exploration_weight,
        "long_tail_weight": long_tail_weight,
    }

    # 캐시 키: include_reasons 여부에 따라 별도 키 사용
    reasons_suffix = ":reasons" if include_reasons else ""
    cache_key = f"cache:recommend:{user_id}:{top_n}:{click_count}{_weight_cache_suffix(weight_params)}{reasons_suffix}"
    cached = feature_store.r.get(cache_key)
    if cached:
        return json.loads(cached)

    params = {
        "user_id": user_id,
        "top_n": top_n,
        "recent_clicks": ",".join(features["recent_clicks"]),
        "click_count": click_count,
        "session_interest": json.dumps(features["session_interest"]) if features["session_interest"] else None,
        "persona_scores": json.dumps(persona_scores) if persona_scores else None,
    }
    params.update({
        key: value
        for key, value in weight_params.items()
        if value is not None and key != "persona_scores"
    })

    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            resp = await client.get(f"{REC_URL}/recommend", params=params)
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=str(e))
        except httpx.RequestError as e:
            raise HTTPException(status_code=503, detail=f"rec-models 연결 실패: {e}")

    result = resp.json()

    # pipeline_latency 중첩 구조를 최상위로 풀어서 프론트가 바로 쓸 수 있게 함
    pl = result.pop("pipeline_latency", {})
    result.update(pl)

    # 상품명·카테고리를 articles_feature.csv에서 보강
    enriched = []
    for i, item in enumerate(result.get("recommendations", []), 1):
        pid = str(item.get("product_id", ""))
        meta = article_meta.get(pid, {})
        enriched.append({
            **item,
            "rank": i,
            "name": meta.get("name") or pid,
            "brand": meta.get("brand") or "H&M",
            "category": meta.get("category", ""),
            "color": meta.get("color", ""),
            "product_type": meta.get("product_type", ""),
            "price": meta.get("price", 0),
        })
    result["recommendations"] = enriched

    # B: LLM 추천 이유 생성 (include_reasons=True이고 API 키 있을 때만)
    if include_reasons and GEMINI_API_KEY:
        items_desc = "\n".join(
            f"{item['rank']}. {item['name']} ({item['category']}, {item['color']}, score={item.get('score', 0):.3f})"
            for item in enriched
        )
        prompt = (
            f"다음 패션 추천 상품들에 대해 각각 추천 이유를 한국어 1~2문장으로 작성해주세요.\n\n"
            f"{items_desc}\n\n"
            f"반드시 아래 JSON 형식으로만 응답하세요:\n"
            f'{{\"reasons\": [\"이유1\", \"이유2\", ...]}}'
        )
        try:
            llm_text = await _call_gemini(prompt, json_mode=True)
            reasons = json.loads(llm_text).get("reasons", [])
            for i, item in enumerate(result["recommendations"]):
                item["reason_text"] = reasons[i] if i < len(reasons) else ""
        except Exception:
            pass  # LLM 실패해도 추천 결과는 정상 반환

    feature_store.r.set(cache_key, json.dumps(result), ex=RECOMMEND_CACHE_TTL)
    return result


@app.post("/api/events")
async def events(req: EventRequest):
    """클릭/구매 이벤트를 Redis에 저장하고 rec-models 세션도 업데이트한다."""
    effective_id = req.article_id or req.item_id

    # Redis 업데이트
    feature_store.r.incr("ct:event_count")
    if req.event_type in ("click", "view", "cart", "purchase") and effective_id:
        feature_store.push_click(req.user_id, effective_id)

    interest_changed = False
    if req.category:
        interest = feature_store.get_session_interest(req.user_id)
        interest[req.category] = interest.get(req.category, 0) + 1
        feature_store.set_session_interest(req.user_id, interest)
        interest_changed = True

    inferred_interest = {}
    if req.event_type == "search":
        inferred_interest = await _infer_session_interest_from_query(req.query_text)
    if inferred_interest:
        interest = feature_store.get_session_interest(req.user_id)
        for category, score in inferred_interest.items():
            interest[category] = interest.get(category, 0) + score
        feature_store.set_session_interest(req.user_id, interest)
        interest_changed = True

    if interest_changed:
        feature_store.invalidate_recommendation_cache(req.user_id)

    # rec-models 세션 업데이트 (실패해도 이벤트 저장은 성공으로 처리)
    if effective_id:
        async with httpx.AsyncClient(timeout=5.0) as client:
            try:
                await client.post(
                    f"{REC_URL}/session/update",
                    json={
                        "user_id": req.user_id,
                        "item_id": effective_id,
                        "event": req.event_type,
                    },
                )
            except httpx.RequestError:
                pass  # rec-models가 아직 없어도 게이트웨이는 정상 응답

    return {"status": "ok"}


@app.post("/api/onboarding")
async def onboarding(req: OnboardingRequest):
    """C-1: LLM으로 9개 페르소나 일치도 계산.

    유저 자유 입력 + 선택지를 LLM에 보내 9개 페르소나와의 일치도(%)를 반환한다.
    Redis에는 아직 저장하지 않는다.
    프론트엔드가 결과를 블록으로 보여주고 유저가 하나를 선택하면
    /api/onboarding/select를 호출해 확정한다.
    """
    style_text = ", ".join(req.style_choices) if req.style_choices else "없음"
    budget_text = {"low": "저가", "mid": "중간 가격대", "high": "고가"}.get(req.budget_range or "", "무관")

    prompt = (
        f"사용자가 아래와 같이 패션 취향을 설명했습니다.\n\n"
        f"자유 입력: {req.description}\n"
        f"선호 스타일: {style_text}\n"
        f"예산 범위: {budget_text}\n\n"
        f"아래 9개 패션 페르소나 중 이 사용자에게 해당하는 것을 골라 퍼센티지를 배분해주세요.\n\n"
        f"규칙 — 먼저 입력에서 페르소나와 연결되는 신호가 몇 개인지 파악하세요:\n"
        f"[신호 1개] 예: '파란색만 좋아', '할인 상품만 산다'\n"
        f"  → 해당 페르소나 70~85%, 나머지는 practical·careful·trendsetter 중 2~3개에 각각 5~15%씩 배분\n"
        f"  → 보조 페르소나 하나가 20% 이상이 되면 안 됩니다\n"
        f"[신호 2~3개] 예: '가성비 중시하고 붉은색 선호'\n"
        f"  → 가장 강한 신호 40~50%, 나머지 신호들이 나머지를 나눔. 무관한 페르소나는 0%\n"
        f"[신호 4개 이상 또는 모호] 예: '다양한 스타일을 즐기는 편'\n"
        f"  → 관련 페르소나들에 고르게 배분\n"
        f"- 합계는 반드시 100입니다.\n\n"
        f"페르소나 설명:\n"
        f"- trendsetter: 새로운 트렌드에 민감하고 다양한 스타일을 시도함\n"
        f"- practical: 실용적이고 목적 지향적인 구매, 기본 아이템 선호\n"
        f"- value: 가성비를 중시하고 세일/할인 상품을 적극 탐색\n"
        f"- brand_loyal: 특정 브랜드나 스타일에 반복적으로 집중\n"
        f"- impulse: 충동적으로 빠르게 구매 결정\n"
        f"- careful: 신중하게 오래 탐색하고 구매 전환율이 낮음\n"
        f"- repeat_stable: 동일한 상품이나 카테고리를 반복 구매\n"
        f"- color_focus: 특정 색상(예: 검정, 흰색, 파랑 등)을 기준으로 탐색, 색상 언급이 핵심 신호\n"
        f"- category_focus: 특정 카테고리(예: 아우터, 운동복 등)에만 집중\n\n"
        f"반드시 아래 JSON 키 이름 그대로, 숫자만 채워서 응답하세요 (합계 100):\n"
        f'{{"trendsetter": ?, "practical": ?, "value": ?, "brand_loyal": ?, '
        f'"impulse": ?, "careful": ?, "repeat_stable": ?, "color_focus": ?, "category_focus": ?}}'
    )

    # 같은 입력이면 캐시에서 바로 반환 (불필요한 Gemini 재호출 방지)
    cache_input = f"{req.description.strip().lower()}|{'|'.join(sorted(req.style_choices))}|{req.budget_range or ''}"
    onboarding_cache_key = f"cache:onboarding:{hashlib.sha256(cache_input.encode('utf-8')).hexdigest()}"
    cached = feature_store.r.get(onboarding_cache_key)
    if cached:
        cached_result = json.loads(cached)
        feature_store.r.set(f"onboarding_scores:{req.user_id}", json.dumps(cached_result["persona_scores"]), ex=600)
        return cached_result

    if GEMINI_API_KEY:
        try:
            llm_text = await _call_gemini(prompt, json_mode=True)
            normalized = _normalize_persona_scores(json.loads(llm_text))
        except httpx.HTTPStatusError as e:
            logging.warning("Gemini onboarding failed with status=%s. Using local fallback.", e.response.status_code)
            normalized = _fallback_persona_scores(req)
        except Exception:
            logging.exception("Gemini onboarding failed. Using local fallback.")
            normalized = _fallback_persona_scores(req)
    else:
        normalized = _fallback_persona_scores(req)

    result = {"persona_scores": normalized}
    feature_store.r.set(onboarding_cache_key, json.dumps(result), ex=3600)
    # select 호출 시 혼합에 쓸 점수를 임시 저장 (10분)
    feature_store.r.set(f"onboarding_scores:{req.user_id}", json.dumps(normalized), ex=600)
    return result


class PersonaSelectRequest(BaseModel):
    user_id: str
    persona: str  # 9개 중 유저가 선택한 페르소나
    persona_scores: dict[str, int | float] | None = None


@app.post("/api/onboarding/select")
async def onboarding_select(req: PersonaSelectRequest):
    """C-2: 유저가 선택한 페르소나를 Redis에 저장.

    프론트엔드에서 /api/onboarding 결과를 보고 유저가 고른 페르소나를
    session_interest로 변환해 Redis에 저장한다.
    다음 /api/recommend 호출 시 즉시 반영된다.
    """
    valid_personas = {"trendsetter", "practical", "value", "brand_loyal", "impulse",
                      "careful", "repeat_stable", "color_focus", "category_focus"}
    if req.persona not in valid_personas:
        raise HTTPException(status_code=400, detail=f"알 수 없는 페르소나: {req.persona}")

    # 페르소나 → 카테고리 관심도 매핑
    persona_to_interest: dict[str, dict[str, int]] = {
        "trendsetter":    {"Ladieswear": 8, "Menswear": 8, "Sport": 6},
        "practical":      {"Menswear": 9, "Ladieswear": 7},
        "value":          {"Divided": 9, "Ladieswear": 6, "Menswear": 6},
        "brand_loyal":    {"Ladieswear": 9, "Menswear": 7, "Lingeries/Tights": 8},
        "impulse":        {"Ladieswear": 8, "Menswear": 7, "Kids": 5},
        "careful":        {"Menswear": 7, "Ladieswear": 7, "Sport": 6},
        "repeat_stable":  {"Ladieswear": 9, "Menswear": 9},
        "color_focus":    {"Ladieswear": 9, "Divided": 7},
        "category_focus": {"Ladieswear": 10, "Menswear": 8},
    }

    # 프론트가 보낸 최신 분석 점수를 우선 사용하고, 없으면 임시 Redis 캐시를 사용한다.
    selected_only_scores = {req.persona: 100}
    stored_scores = selected_only_scores
    if req.persona_scores:
        stored_scores = _normalize_persona_scores(req.persona_scores)

    stored_raw = feature_store.r.get(f"onboarding_scores:{req.user_id}")
    if stored_raw and not req.persona_scores:
        stored_scores = _normalize_persona_scores(json.loads(stored_raw))

    blended: dict[str, float] = {}
    for persona, weight in stored_scores.items():
        if weight <= 0 or persona not in persona_to_interest:
            continue
        for category, score in persona_to_interest[persona].items():
            blended[category] = blended.get(category, 0) + score * (weight / 100.0)
    session_interest = {key: round(value) for key, value in blended.items() if round(value) > 0}
    if not session_interest:
        session_interest = persona_to_interest[req.persona]

    feature_store.set_persona_scores(req.user_id, stored_scores)
    feature_store.r.delete(f"onboarding_scores:{req.user_id}")
    feature_store.set_session_interest(req.user_id, session_interest)
    feature_store.invalidate_recommendation_cache(req.user_id)
    return {
        "status": "ok",
        "persona": req.persona,
        "persona_scores": stored_scores,
        "session_interest": session_interest,
    }


@app.post("/api/budget-set")
async def budget_set(
    user_id: str = Query(...),
    budget: int = Query(..., description="총 예산 (원)"),
    set_count: int = Query(3, description="구성할 세트 수"),
):
    """D: 예산 기반 패션 세트 추천.

    rec-models에서 후보 50개를 가져온 뒤, 예산 내 아이템으로 필터링하고
    search-engine의 /cross-similarity API로 어울리는 조합을 set_count세트 구성한다.
    search-engine에 /cross-similarity가 미구현 상태라면 score 기반 그리디로 대체한다.
    """
    features = feature_store.get_user_features(user_id)
    raw_persona_scores = features.get("persona_scores", {})
    persona_scores = _normalize_persona_scores(raw_persona_scores) if raw_persona_scores else {}
    params = {
        "user_id": user_id,
        "top_n": 50,
        "recent_clicks": ",".join(features["recent_clicks"]),
        "click_count": features["click_count"],
        "session_interest": json.dumps(features["session_interest"]) if features["session_interest"] else None,
        "persona_scores": json.dumps(persona_scores) if persona_scores else None,
    }

    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            rec_resp = await client.get(f"{REC_URL}/recommend", params=params)
            rec_resp.raise_for_status()
        except httpx.RequestError as e:
            raise HTTPException(status_code=503, detail=f"rec-models 연결 실패: {e}")

    candidates = rec_resp.json().get("recommendations", [])

    # article_meta에서 가격 정보 보강 후 예산 내 필터
    # article_meta["price"]는 item_features CSV의 avg_price * PRICE_KRW_FACTOR (KRW)
    DEFAULT_PRICE = 25000  # 데이터 없는 상품의 폴백 (H&M KRW 중앙가)
    affordable = []
    for item in candidates:
        pid = str(item.get("product_id", ""))
        meta = article_meta.get(pid, {})
        price_int = meta.get("price", 0) or DEFAULT_PRICE
        if price_int <= budget:
            affordable.append({**item, **meta, "price_int": price_int, "article_id": pid})

    if len(affordable) < 2:
        raise HTTPException(status_code=400, detail="예산 내 추천 가능한 상품이 부족합니다")

    # search-engine cross-similarity 호출 (미구현 시 빈 행렬로 대체)
    article_ids = [c["article_id"] for c in affordable[:20]]
    sim_matrix: dict = {}
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            sim_resp = await client.post(
                f"{SEARCH_URL}/cross-similarity",
                json={"article_ids": article_ids},
            )
            sim_resp.raise_for_status()
            sim_matrix = sim_resp.json().get("similarity", {})
        except Exception:
            pass  # 미구현이면 score 기반으로만 진행

    sets = _build_outfit_sets(affordable, sim_matrix, budget, set_count)
    return {"sets": sets, "budget": budget, "set_count": len(sets)}


def _build_outfit_sets(
    candidates: list[dict],
    sim_matrix: dict,
    budget: int,
    count: int,
) -> list[list[dict]]:
    """예산 내에서 score 기반 그리디로 n세트 조합.

    sim_matrix가 있으면 아이템 간 유사도 합산 점수를 가중치로 활용한다.
    """
    sorted_candidates = sorted(candidates, key=lambda x: x.get("score", 0), reverse=True)
    sets: list[list[dict]] = []
    used_ids: set[str] = set()

    for _ in range(count):
        current_set: list[dict] = []
        current_cost = 0

        for item in sorted_candidates:
            aid = item["article_id"]
            if aid in used_ids:
                continue
            if current_cost + item["price_int"] > budget:
                continue

            # sim_matrix가 있으면 현재 세트와의 평균 유사도로 어울림 판단
            if sim_matrix and current_set:
                sim_scores = [
                    sim_matrix.get(existing["article_id"], {}).get(aid, 0)
                    for existing in current_set
                ]
                avg_sim = sum(sim_scores) / len(sim_scores)
                if avg_sim < 0.2:
                    continue

            current_set.append(item)
            current_cost += item["price_int"]
            used_ids.add(aid)

            if len(current_set) >= 3:
                break

        if current_set:
            sets.append(current_set)

    return sets


@app.get("/api/images/{article_id}")
async def get_image(article_id: str):
    """Return a local H&M product image by article id."""
    image_path = image_path_for_article(article_id)
    if not image_path.exists() or not image_path.is_file():
        raise HTTPException(status_code=404, detail="Image not found")

    return FileResponse(
        image_path,
        media_type="image/jpeg",
        headers={"Cache-Control": "public, max-age=86400"},
    )


@app.get("/api/features/{user_id}")
async def get_features(user_id: str):
    """Redis에 저장된 유저 피처를 반환한다."""
    return feature_store.get_user_features(user_id)


@app.get("/health")
async def health():
    try:
        feature_store.r.ping()
        redis_ok = True
    except Exception:
        redis_ok = False

    return {
        "status": "ok" if redis_ok else "degraded",
        "redis": redis_ok,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
