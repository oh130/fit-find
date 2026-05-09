"""Top-level serving orchestration for recommendations."""

from __future__ import annotations

import logging
import time
from typing import Any

import pandas as pd

try:
    from rec_models.serving.candidate_service import (
        generate_candidates,
        load_candidate_user_profiles,
        load_serving_artifacts,
        load_sequential_transition_store,
        load_two_tower_model_bundle,
        load_two_tower_serving_artifacts,
    )
    from rec_models.serving.ranking_service import (
        load_customer_features,
        load_item_persona_features,
        load_ranking_pipeline,
        load_user_persona_features,
        score_candidates,
    )
    from rec_models.serving.rerank_bridge import rerank_recommendations
except ImportError:  # pragma: no cover - supports running from rec_models/ as cwd
    from serving.candidate_service import (  # type: ignore[no-redef]
        generate_candidates,
        load_candidate_user_profiles,
        load_serving_artifacts,
        load_sequential_transition_store,
        load_two_tower_model_bundle,
        load_two_tower_serving_artifacts,
    )
    from serving.ranking_service import (  # type: ignore[no-redef]
        load_customer_features,
        load_item_persona_features,
        load_ranking_pipeline,
        load_user_persona_features,
        score_candidates,
    )
    from serving.rerank_bridge import rerank_recommendations  # type: ignore[no-redef]


LOGGER = logging.getLogger(__name__)


def _elapsed_ms(start_time: float) -> int:
    return int(round((time.perf_counter() - start_time) * 1000))


def _build_popularity_fallback(scored_candidates: pd.DataFrame) -> pd.DataFrame:
    """Fallback ordering when ranking inference fails."""

    fallback = scored_candidates.copy()
    popularity_max = max(float(fallback.get("popularity", pd.Series([0.0])).max()), 1.0)
    fallback["score"] = pd.to_numeric(fallback.get("popularity"), errors="coerce").fillna(0.0) / popularity_max
    fallback["reason"] = fallback.get("candidate_reason", "cold_start_popularity")
    fallback["is_exploration"] = False
    return fallback


def _random_seed_from_context(user_id: str, session_context: dict[str, Any]) -> int:
    return hash((user_id, tuple(session_context["recent_clicks"][:5]), str(session_context["session_interest"]))) & 0xFFFFFFFF


def warmup_recommendation_assets() -> None:
    """Eagerly load heavy serving artifacts to reduce first-request latency."""

    warmup_start = time.perf_counter()
    load_serving_artifacts()
    load_candidate_user_profiles()
    load_sequential_transition_store()
    load_two_tower_serving_artifacts()
    load_two_tower_model_bundle()
    load_ranking_pipeline()
    load_customer_features()
    load_user_persona_features()
    load_item_persona_features()
    LOGGER.info("Warmup completed in %sms", _elapsed_ms(warmup_start))


def rank_candidates_to_recommendations(
    user_id: str,
    candidate_items: pd.DataFrame,
    top_n: int,
    session_context: dict[str, Any] | None = None,
    *,
    enable_diversity: bool = True,
    enable_exploration: bool = True,
    enable_freshness: bool = True,
) -> list[dict[str, Any]]:
    """Convert candidate rows into final recommendations using serving ranking logic."""

    session_context = session_context or {"recent_clicks": [], "session_interest": None}
    if "score" in candidate_items.columns:
        return rerank_recommendations(
            scored_candidates=candidate_items,
            top_n=top_n,
            random_seed=_random_seed_from_context(user_id=user_id, session_context=session_context),
            enable_diversity=enable_diversity,
            enable_exploration=enable_exploration,
            enable_freshness=enable_freshness,
        )

    try:
        scored_candidates = score_candidates(
            user_id=user_id,
            candidate_items=candidate_items,
            session_context=session_context,
        )
    except Exception:
        LOGGER.exception("Ranking stage failed. Falling back to popularity ordering for user_id=%s", user_id)
        scored_candidates = _build_popularity_fallback(candidate_items)

    return rerank_recommendations(
        scored_candidates=scored_candidates,
        top_n=top_n,
        random_seed=_random_seed_from_context(user_id=user_id, session_context=session_context),
        enable_diversity=enable_diversity,
        enable_exploration=enable_exploration,
        enable_freshness=enable_freshness,
    )


def recommend(
    user_id: str,
    top_n: int = 10,
    recent_clicks: list[str] | None = None,
    click_count: int = 0,
    session_interest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run candidate generation, ranking, and reranking for one user."""

    total_start = time.perf_counter()
    session_context = {
        "recent_clicks": recent_clicks or [],
        "session_interest": session_interest or None,
    }

    candidate_start = time.perf_counter()
    candidate_items = generate_candidates(
        user_id=user_id,
        top_k=top_n,
        recent_clicks=recent_clicks,
        session_interest=session_interest,
    )
    candidate_ms = _elapsed_ms(candidate_start)

    ranking_start = time.perf_counter()
    try:
        scored_candidates = score_candidates(
            user_id=user_id,
            candidate_items=candidate_items,
            session_context=session_context,
        )
    except Exception:
        LOGGER.exception("Ranking stage failed. Falling back to popularity ordering for user_id=%s", user_id)
        scored_candidates = _build_popularity_fallback(candidate_items)
    ranking_ms = _elapsed_ms(ranking_start)

    reranking_start = time.perf_counter()
    recommendations = rerank_recommendations(
        scored_candidates=scored_candidates,
        top_n=top_n,
        random_seed=_random_seed_from_context(user_id=user_id, session_context=session_context),
    )
    reranking_ms = _elapsed_ms(reranking_start)
    total_ms = _elapsed_ms(total_start)

    LOGGER.info(
        "Recommendation completed user_id=%s top_n=%s click_count=%s candidates=%s candidate_ms=%s ranking_ms=%s reranking_ms=%s total_ms=%s",
        user_id,
        top_n,
        click_count,
        len(candidate_items),
        candidate_ms,
        ranking_ms,
        reranking_ms,
        total_ms,
    )

    return {
        "user_id": user_id,
        "recommendations": recommendations,
        "pipeline_latency": {
            "candidate_ms": candidate_ms,
            "ranking_ms": ranking_ms,
            "reranking_ms": reranking_ms,
            "total_ms": total_ms,
        },
        "session_context": session_context,
    }
