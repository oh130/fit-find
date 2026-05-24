"""FastAPI entrypoint for the recommendation service."""

from __future__ import annotations

import json
import logging
from typing import Any

from fastapi import FastAPI, Query
from pydantic import BaseModel, Field

try:
    from rec_models.serving.bandit_store import get_bandit_store
    from rec_models.serving.recommend_service import (
        recommend,
        rerank_external_candidates,
        warmup_recommendation_assets,
    )
except ImportError:  # pragma: no cover - supports running from rec_models/ as cwd
    from serving.bandit_store import get_bandit_store  # type: ignore[no-redef]
    from serving.recommend_service import (  # type: ignore[no-redef]
        recommend,
        rerank_external_candidates,
        warmup_recommendation_assets,
    )


LOGGER = logging.getLogger(__name__)
app = FastAPI(title="Recommendation Models Service")


class SessionUpdateRequest(BaseModel):
    user_id: str
    item_id: str | None = None
    article_id: str | None = None
    product_id: str | None = None
    event: str


class ExternalSearchCandidate(BaseModel):
    product_id: str | int | None = None
    article_id: str | int | None = None
    item_id: str | int | None = None
    score: float | None = None
    similarity: float | None = None


class RerankCandidatesRequest(BaseModel):
    user_id: str
    search_candidates: list[ExternalSearchCandidate] = Field(default_factory=list)
    top_n: int = 10
    recent_clicks: list[str] = Field(default_factory=list)
    session_interest: dict[str, Any] | None = None
    preferred_terms: list[str] = Field(default_factory=list)
    avoid_terms: list[str] = Field(default_factory=list)
    persona_hint: str | None = None
    persona_scores: dict[str, Any] | None = None
    include_recommendation_candidates: bool = True
    recommendation_candidate_pool_size: int = 40
    personalization_weight: float | None = None
    price_weight: float | None = None
    popularity_weight: float | None = None
    diversity_weight: float | None = None
    freshness_weight: float | None = None
    exploration_weight: float | None = None
    long_tail_weight: float | None = None


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


def _parse_persona_scores(raw_persona_scores: str | None) -> dict[str, Any] | None:
    if not raw_persona_scores:
        return None

    try:
        parsed = json.loads(raw_persona_scores)
    except json.JSONDecodeError:
        LOGGER.warning("Failed to parse persona_scores JSON. Ignoring value.")
        return None

    if isinstance(parsed, dict):
        return parsed
    LOGGER.warning("persona_scores must be a JSON object. Ignoring value.")
    return None


def _parse_term_list(raw_terms: str | None) -> list[str]:
    if not raw_terms:
        return []

    try:
        parsed = json.loads(raw_terms)
    except json.JSONDecodeError:
        parsed = raw_terms.split(",")

    if not isinstance(parsed, list):
        LOGGER.warning("query intent terms must be a JSON array or comma-separated list. Ignoring value.")
        return []

    normalized_terms: list[str] = []
    seen: set[str] = set()
    for raw_term in parsed:
        term = str(raw_term or "").strip()
        normalized = term.lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        normalized_terms.append(term)
    return normalized_terms


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
    preferred_terms: str | None = Query(None),
    avoid_terms: str | None = Query(None),
    persona_hint: str | None = Query(None),
    persona_scores: str | None = Query(None),
    personalization_weight: float | None = Query(None, ge=0.0, le=5.0),
    price_weight: float | None = Query(None, ge=0.0, le=5.0),
    popularity_weight: float | None = Query(None, ge=0.0, le=5.0),
    diversity_weight: float | None = Query(None, ge=0.0, le=5.0),
    freshness_weight: float | None = Query(None, ge=0.0, le=5.0),
    exploration_weight: float | None = Query(None, ge=0.0, le=5.0),
    long_tail_weight: float | None = Query(None, ge=0.0, le=5.0),
) -> dict[str, Any]:
    """Return ranked recommendations for one user."""

    return recommend(
        user_id=user_id,
        top_n=top_n,
        recent_clicks=_parse_recent_clicks(recent_clicks),
        click_count=click_count,
        session_interest=_parse_session_interest(session_interest),
        preferred_terms=_parse_term_list(preferred_terms),
        avoid_terms=_parse_term_list(avoid_terms),
        persona_hint=persona_hint,
        persona_scores=_parse_persona_scores(persona_scores),
        personalization_weight=personalization_weight,
        price_weight=price_weight,
        popularity_weight=popularity_weight,
        diversity_weight=diversity_weight,
        freshness_weight=freshness_weight,
        exploration_weight=exploration_weight,
        long_tail_weight=long_tail_weight,
    )


@app.post("/rerank-candidates")
def rerank_candidates_endpoint(req: RerankCandidatesRequest) -> dict[str, Any]:
    """Personalize externally retrieved search candidates."""

    top_n = max(1, min(int(req.top_n), 100))
    pool_size = max(top_n, min(int(req.recommendation_candidate_pool_size), 200))
    return rerank_external_candidates(
        user_id=req.user_id,
        search_candidates=[candidate.model_dump(exclude_none=True) for candidate in req.search_candidates],
        top_n=top_n,
        recent_clicks=req.recent_clicks,
        session_interest=req.session_interest,
        preferred_terms=req.preferred_terms,
        avoid_terms=req.avoid_terms,
        persona_hint=req.persona_hint,
        persona_scores=req.persona_scores,
        include_recommendation_candidates=req.include_recommendation_candidates,
        recommendation_candidate_pool_size=pool_size,
        personalization_weight=req.personalization_weight,
        price_weight=req.price_weight,
        popularity_weight=req.popularity_weight,
        diversity_weight=req.diversity_weight,
        freshness_weight=req.freshness_weight,
        exploration_weight=req.exploration_weight,
        long_tail_weight=req.long_tail_weight,
    )


@app.post("/session/update")
def session_update(req: SessionUpdateRequest) -> dict[str, Any]:
    """Apply session reward feedback for reward-aware exploration."""

    article_id = req.article_id or req.product_id or req.item_id
    reward_result = get_bandit_store().record_reward(
        user_id=req.user_id,
        article_id=article_id,
        event=req.event,
    )
    return {
        "status": "ok",
        "bandit_updated": reward_result.updated,
        "article_id": reward_result.article_id,
        "event": reward_result.event,
        "reward": reward_result.reward,
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8003, reload=False)
