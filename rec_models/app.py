"""FastAPI entrypoint for the recommendation service."""

from __future__ import annotations

import json
import logging
from typing import Any

from fastapi import FastAPI, Query
from pydantic import BaseModel

try:
    from rec_models.serving.recommend_service import recommend, warmup_recommendation_assets
except ImportError:  # pragma: no cover - supports running from rec_models/ as cwd
    from serving.recommend_service import recommend, warmup_recommendation_assets  # type: ignore[no-redef]


LOGGER = logging.getLogger(__name__)
app = FastAPI(title="Recommendation Models Service")


class SessionUpdateRequest(BaseModel):
    user_id: str
    item_id: str
    event: str


def _parse_recent_clicks(raw_recent_clicks: str | None) -> list[str]:
    if not raw_recent_clicks:
        return []
    return [value.strip() for value in raw_recent_clicks.split(",") if value.strip()]


def _parse_session_interest(raw_session_interest: str | None) -> dict[str, Any] | None:
    if not raw_session_interest:
        return None

    try:
        parsed = json.loads(raw_session_interest)
    except json.JSONDecodeError:
        LOGGER.warning("Failed to parse session_interest JSON. Ignoring value.")
        return None

    if isinstance(parsed, dict):
        return parsed
    LOGGER.warning("session_interest must be a JSON object. Ignoring value.")
    return None


@app.on_event("startup")
def startup_event() -> None:
    """Preload serving artifacts so the first request does not pay I/O costs."""

    warmup_recommendation_assets()


@app.get("/recommend")
def recommend_endpoint(
    user_id: str = Query(...),
    top_n: int = Query(10, ge=1, le=100),
    recent_clicks: str | None = Query(None),
    click_count: int = Query(0, ge=0),
    session_interest: str | None = Query(None),
    price_weight: float | None = Query(None, ge=0.0, le=5.0),
    popularity_weight: float | None = Query(None, ge=0.0, le=5.0),
    diversity_weight: float | None = Query(None, ge=0.0, le=5.0),
    freshness_weight: float | None = Query(None, ge=0.0, le=5.0),
    exploration_weight: float | None = Query(None, ge=0.0, le=5.0),
) -> dict[str, Any]:
    """Return ranked recommendations for one user."""

    return recommend(
        user_id=user_id,
        top_n=top_n,
        recent_clicks=_parse_recent_clicks(recent_clicks),
        click_count=click_count,
        session_interest=_parse_session_interest(session_interest),
        price_weight=price_weight,
        popularity_weight=popularity_weight,
        diversity_weight=diversity_weight,
        freshness_weight=freshness_weight,
        exploration_weight=exploration_weight,
    )


@app.post("/session/update")
def session_update(_: SessionUpdateRequest) -> dict[str, str]:
    """Placeholder endpoint for session-event integration.

    TODO: Persist session updates inside rec_models once a dedicated session
    feature backend is introduced.
    """

    return {"status": "ok"}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8003, reload=False)
