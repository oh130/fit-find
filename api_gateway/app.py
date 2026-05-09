"""
API Gateway — port 8000

엔드포인트:
  POST /api/search            search-engine 프록시
  GET  /api/recommend         Redis 세션 붙여서 rec-models 프록시
  POST /api/events            Redis에 클릭/구매 이벤트 저장
  GET  /api/features/{user_id} Redis 유저 피처 조회
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
):
    """Redis 세션 데이터를 붙여 rec-models로 추천 요청을 프록시한다."""
    features = feature_store.get_user_features(user_id)
    click_count = features["click_count"]

    # 캐시 키: 클릭 수가 바뀌면 자동으로 캐시 무효화
    cache_key = f"cache:recommend:{user_id}:{top_n}:{click_count}"
    cached = feature_store.r.get(cache_key)
    if cached:
        return json.loads(cached)

    params = {
        "user_id": user_id,
        "top_n": top_n,
        "recent_clicks": ",".join(features["recent_clicks"]),
        "click_count": click_count,
        "session_interest": json.dumps(features["session_interest"]) if features["session_interest"] else None,
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
