"""
API Gateway — port 8000

엔드포인트:
  POST /api/search             search-engine 프록시
  GET  /api/recommend          Redis 세션 붙여서 rec-models 프록시
  POST /api/events             Redis에 클릭/구매 이벤트 저장
  GET  /api/features/{user_id} Redis 유저 피처 조회
  GET  /api/images/{article_id} 상품 이미지 반환
  POST /api/onboarding         LLM 기반 콜드 스타트 페르소나 생성 (기능 C)
  POST /api/budget-set         예산 기반 패션 세트 추천 (기능 D)
  GET  /health
"""

import csv
import json
import os
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

feature_store: RedisFeatureStore
# article_id → {name, category, color, product_type}
article_meta: dict[str, dict] = {}


def _load_article_meta() -> dict[str, dict]:
    meta: dict[str, dict] = {}
    if not ARTICLES_PATH.exists():
        return meta
    with ARTICLES_PATH.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            aid = row.get("article_id", "").strip()
            if aid:
                meta[aid] = {
                    "name": row.get("prod_name", ""),
                    "category": row.get("category", ""),
                    "color": row.get("color", ""),
                    "product_type": row.get("product_type_name", ""),
                }
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


# ── 유틸: LLM 호출 ───────────────────────────────────────────

async def _call_gemini(prompt: str, json_mode: bool = False) -> str:
    """Gemini Flash API 호출. GEMINI_API_KEY 미설정 시 빈 문자열 반환.

    json_mode=True이면 JSON 외 출력을 차단해 파싱 안정성을 높인다.
    """
    if not GEMINI_API_KEY:
        return ""

    generation_config: dict = {"temperature": 0.7 if not json_mode else 0.1}
    if json_mode:
        generation_config["responseMimeType"] = "application/json"

    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}",
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": generation_config,
            },
        )
        resp.raise_for_status()
        return resp.json()["candidates"][0]["content"]["parts"][0]["text"]


# ── 엔드포인트 ────────────────────────────────────────────────

@app.post("/api/search")
async def search(req: SearchRequest):
    """search-engine으로 검색 요청을 프록시한다."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            resp = await client.post(
                f"{SEARCH_URL}/search",
                json=req.model_dump(),
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=str(e))
        except httpx.RequestError as e:
            raise HTTPException(status_code=503, detail=f"search-engine 연결 실패: {e}")
    return resp.json()


RECOMMEND_CACHE_TTL = 300  # 5분


@app.get("/api/recommend")
async def recommend(
    user_id: str = Query(...),
    top_n: int = Query(10),
    price_weight: float = Query(0.0),       # A: 가격 가중치 (0.0~1.0)
    popularity_weight: float = Query(0.0),  # A: 인기도 가중치 (0.0~1.0)
    include_reasons: bool = Query(False),   # B: LLM 추천 이유 포함 여부
):
    """Redis 세션 데이터를 붙여 rec-models로 추천 요청을 프록시한다."""
    features = feature_store.get_user_features(user_id)
    click_count = features["click_count"]

    # include_reasons=True이면 LLM 결과가 붙으므로 캐시 우회
    cache_key = f"cache:recommend:{user_id}:{top_n}:{click_count}:{price_weight}:{popularity_weight}"
    if not include_reasons:
        cached = feature_store.r.get(cache_key)
        if cached:
            return json.loads(cached)

    params = {
        "user_id": user_id,
        "top_n": top_n,
        "recent_clicks": ",".join(features["recent_clicks"]),
        "click_count": click_count,
        "session_interest": json.dumps(features["session_interest"]) if features["session_interest"] else None,
        "price_weight": price_weight,        # A: rec-models에 그대로 전달
        "popularity_weight": popularity_weight,
    }

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
            "category": meta.get("category", ""),
            "color": meta.get("color", ""),
            "product_type": meta.get("product_type", ""),
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

    if req.category:
        interest = feature_store.get_session_interest(req.user_id)
        interest[req.category] = interest.get(req.category, 0) + 1
        feature_store.set_session_interest(req.user_id, interest)

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
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=503, detail="GEMINI_API_KEY not configured")

    style_text = ", ".join(req.style_choices) if req.style_choices else "없음"
    budget_text = {"low": "저가", "mid": "중간 가격대", "high": "고가"}.get(req.budget_range or "", "무관")

    prompt = (
        f"사용자가 아래와 같이 패션 취향을 설명했습니다.\n\n"
        f"자유 입력: {req.description}\n"
        f"선호 스타일: {style_text}\n"
        f"예산 범위: {budget_text}\n\n"
        f"아래 9개 패션 페르소나 각각과 이 사용자가 얼마나 일치하는지 퍼센티지로 추정해주세요.\n"
        f"퍼센티지의 합은 반드시 100이 되어야 합니다.\n\n"
        f"페르소나 설명:\n"
        f"- trendsetter: 새로운 트렌드에 민감하고 다양한 스타일을 시도함\n"
        f"- practical: 실용적이고 목적 지향적인 구매, 기본 아이템 선호\n"
        f"- value: 가성비를 중시하고 세일/할인 상품을 적극 탐색\n"
        f"- brand_loyal: 특정 카테고리나 스타일에 반복적으로 집중\n"
        f"- impulse: 충동적으로 빠르게 구매 결정\n"
        f"- careful: 신중하게 오래 탐색하고 구매 전환율이 낮음\n"
        f"- repeat_stable: 동일한 상품이나 카테고리를 반복 구매\n"
        f"- color_focus: 특정 색상 위주로 탐색\n"
        f"- category_focus: 특정 카테고리에만 집중\n\n"
        f"반드시 아래 JSON 형식으로만 응답하세요:\n"
        f'{{"trendsetter": 30, "practical": 25, "value": 15, "brand_loyal": 10, '
        f'"impulse": 5, "careful": 5, "repeat_stable": 5, "color_focus": 3, "category_focus": 2}}'
    )

    try:
        llm_text = await _call_gemini(prompt, json_mode=True)
        persona_scores: dict = json.loads(llm_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM 응답 파싱 실패: {e}")

    valid_personas = {"trendsetter", "practical", "value", "brand_loyal", "impulse",
                      "careful", "repeat_stable", "color_focus", "category_focus"}
    filtered = {
        k: max(0, int(v))
        for k, v in persona_scores.items()
        if k in valid_personas and isinstance(v, (int, float))
    }

    # 합이 100이 되도록 정규화
    total = sum(filtered.values())
    normalized = {k: round(v * 100 / total) for k, v in filtered.items()} if total > 0 else filtered

    return {"persona_scores": normalized}


class PersonaSelectRequest(BaseModel):
    user_id: str
    persona: str  # 9개 중 유저가 선택한 페르소나


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

    session_interest = persona_to_interest[req.persona]
    feature_store.set_session_interest(req.user_id, session_interest)
    return {"status": "ok", "persona": req.persona, "session_interest": session_interest}


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
    params = {
        "user_id": user_id,
        "top_n": 50,
        "recent_clicks": ",".join(features["recent_clicks"]),
        "click_count": features["click_count"],
        "session_interest": json.dumps(features["session_interest"]) if features["session_interest"] else None,
    }

    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            rec_resp = await client.get(f"{REC_URL}/recommend", params=params)
            rec_resp.raise_for_status()
        except httpx.RequestError as e:
            raise HTTPException(status_code=503, detail=f"rec-models 연결 실패: {e}")

    candidates = rec_resp.json().get("recommendations", [])

    # article_meta에서 가격 정보 보강 후 예산 내 필터
    affordable = []
    for item in candidates:
        pid = str(item.get("product_id", ""))
        meta = article_meta.get(pid, {})
        price_raw = item.get("price", 0)
        try:
            price_int = int(float(str(price_raw).replace(",", "").replace("원", "")))
        except (ValueError, TypeError):
            price_int = 0
        if 0 < price_int <= budget:
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
