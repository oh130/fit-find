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
        materialize_external_candidates,
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
        materialize_external_candidates,
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

PERSONA_SESSION_INTERESTS: dict[str, dict[str, float]] = {
    "trendsetter": {"Ladieswear": 8.0, "Menswear": 8.0, "Sport": 6.0},
    "practical": {"Menswear": 9.0, "Ladieswear": 7.0},
    "value": {"Divided": 9.0, "Ladieswear": 6.0, "Menswear": 6.0},
    "brand_loyal": {"Ladieswear": 9.0, "Menswear": 7.0, "Lingeries/Tights": 8.0},
    "impulse": {"Ladieswear": 8.0, "Menswear": 7.0, "Kids": 5.0},
    "careful": {"Menswear": 7.0, "Ladieswear": 7.0, "Sport": 6.0},
    "repeat_stable": {"Ladieswear": 9.0, "Menswear": 9.0},
    "color_focus": {"Ladieswear": 9.0, "Divided": 7.0},
    "category_focus": {"Ladieswear": 10.0, "Menswear": 8.0},
}

PERSONA_RERANK_PROFILES: dict[str, dict[str, float]] = {
    # Newness and controlled exploration matter more for users who actively
    # seek novelty.
    "trendsetter": {
        "freshness_weight": 2.0,
        "exploration_weight": 1.6,
        "diversity_weight": 1.4,
        "long_tail_weight": 1.2,
    },
    # Practical/value users should see cheaper, reliable items before novelty.
    "practical": {
        "personalization_weight": 1.3,
        "price_weight": 1.5,
        "exploration_weight": 0.5,
        "diversity_weight": 0.8,
    },
    "value": {
        "price_weight": 2.2,
        "personalization_weight": 1.2,
        "popularity_weight": 0.4,
        "exploration_weight": 0.5,
        "long_tail_weight": 0.3,
    },
    # Stable preference personas favor relevance consistency over exploration.
    "brand_loyal": {
        "personalization_weight": 1.6,
        "exploration_weight": 0.3,
        "diversity_weight": 0.6,
        "long_tail_weight": 0.2,
    },
    "repeat_stable": {
        "personalization_weight": 1.7,
        "exploration_weight": 0.2,
        "diversity_weight": 0.4,
        "long_tail_weight": 0.1,
    },
    # Impulse buyers respond to fresh/exploratory slots, but less category
    # spreading than trendsetters.
    "impulse": {
        "freshness_weight": 1.7,
        "exploration_weight": 1.4,
        "diversity_weight": 0.8,
    },
    "careful": {
        "personalization_weight": 1.4,
        "diversity_weight": 1.5,
        "exploration_weight": 0.4,
        "popularity_weight": 0.5,
    },
    "color_focus": {
        "personalization_weight": 1.4,
        "diversity_weight": 0.6,
        "exploration_weight": 0.5,
    },
    "category_focus": {
        "personalization_weight": 1.5,
        "diversity_weight": 0.4,
        "exploration_weight": 0.4,
    },
}

PERSONA_PROFILE_KEYS = (
    "personalization_weight",
    "price_weight",
    "popularity_weight",
    "diversity_weight",
    "freshness_weight",
    "exploration_weight",
    "long_tail_weight",
)
DEFAULT_EXTERNAL_RECOMMENDATION_POOL = 40


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


def _normalize_persona_score_vector(
    persona_scores: dict[str, Any] | None,
    persona_hint: str | None = None,
) -> dict[str, float]:
    normalized_scores: dict[str, float] = {}
    for persona, raw_score in (persona_scores or {}).items():
        persona_key = str(persona).strip()
        if persona_key not in PERSONA_SESSION_INTERESTS:
            continue
        try:
            score = float(raw_score)
        except (TypeError, ValueError):
            continue
        if score > 0.0:
            normalized_scores[persona_key] = normalized_scores.get(persona_key, 0.0) + score

    total_score = sum(normalized_scores.values())
    if total_score > 0.0:
        return {persona: score / total_score for persona, score in normalized_scores.items()}

    fallback_persona = str(persona_hint or "").strip()
    if fallback_persona in PERSONA_SESSION_INTERESTS:
        return {fallback_persona: 1.0}
    return {}


def _blend_persona_rerank_profile(persona_ratios: dict[str, float]) -> dict[str, float]:
    if not persona_ratios:
        return {}

    blended = {key: 0.0 for key in PERSONA_PROFILE_KEYS}
    for persona, ratio in persona_ratios.items():
        profile = PERSONA_RERANK_PROFILES.get(persona, {})
        for key in PERSONA_PROFILE_KEYS:
            blended[key] += float(profile.get(key, 1.0)) * ratio
    return {key: value for key, value in blended.items() if value != 1.0}


def _build_rerank_weights(
    rerank_weights: dict[str, Any] | None = None,
    *,
    persona_hint: str | None = None,
    persona_scores: dict[str, Any] | None = None,
    personalization_weight: float | None = None,
    price_weight: float | None = None,
    popularity_weight: float | None = None,
    diversity_weight: float | None = None,
    freshness_weight: float | None = None,
    exploration_weight: float | None = None,
    long_tail_weight: float | None = None,
) -> dict[str, Any] | None:
    persona_ratios = _normalize_persona_score_vector(
        persona_scores=persona_scores,
        persona_hint=persona_hint,
    )
    weights = _blend_persona_rerank_profile(persona_ratios)
    weights.update(dict(rerank_weights or {}))
    overrides = {
        "personalization_weight": personalization_weight,
        "price_weight": price_weight,
        "popularity_weight": popularity_weight,
        "diversity_weight": diversity_weight,
        "freshness_weight": freshness_weight,
        "exploration_weight": exploration_weight,
        "long_tail_weight": long_tail_weight,
    }
    weights.update({key: value for key, value in overrides.items() if value is not None})
    return weights or None


def _merge_persona_session_interest(
    session_interest: dict[str, Any] | None,
    persona_hint: str | None,
    persona_scores: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Apply persona ratios as serving-time interest floors."""

    persona_ratios = _normalize_persona_score_vector(
        persona_scores=persona_scores,
        persona_hint=persona_hint,
    )
    if not persona_ratios:
        return session_interest

    merged: dict[str, float] = {}
    for key, value in (session_interest or {}).items():
        try:
            merged[str(key)] = float(value)
        except (TypeError, ValueError):
            merged[str(key)] = 0.0

    blended_interest: dict[str, float] = {}
    for persona, ratio in persona_ratios.items():
        for category, score in PERSONA_SESSION_INTERESTS[persona].items():
            blended_interest[category] = blended_interest.get(category, 0.0) + float(score) * ratio

    for category, score in blended_interest.items():
        merged[category] = max(score, merged.get(category, 0.0))

    return merged


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
    rerank_weights: dict[str, Any] | None = None,
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
            rerank_weights=rerank_weights,
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
        rerank_weights=rerank_weights,
    )


def rerank_external_candidates(
    user_id: str,
    search_candidates: list[dict[str, Any]],
    top_n: int = 10,
    recent_clicks: list[str] | None = None,
    session_interest: dict[str, Any] | None = None,
    rerank_weights: dict[str, Any] | None = None,
    persona_hint: str | None = None,
    persona_scores: dict[str, Any] | None = None,
    include_recommendation_candidates: bool = True,
    recommendation_candidate_pool_size: int = DEFAULT_EXTERNAL_RECOMMENDATION_POOL,
    personalization_weight: float | None = None,
    price_weight: float | None = None,
    popularity_weight: float | None = None,
    diversity_weight: float | None = None,
    freshness_weight: float | None = None,
    exploration_weight: float | None = None,
    long_tail_weight: float | None = None,
) -> dict[str, Any]:
    """Rank search-engine candidates with the same serving personalization stack."""

    total_start = time.perf_counter()
    effective_session_interest = _merge_persona_session_interest(
        session_interest=session_interest,
        persona_hint=persona_hint,
        persona_scores=persona_scores,
    )
    session_context = {
        "recent_clicks": recent_clicks or [],
        "session_interest": effective_session_interest or None,
    }
    effective_rerank_weights = _build_rerank_weights(
        rerank_weights,
        persona_hint=persona_hint,
        persona_scores=persona_scores,
        personalization_weight=personalization_weight,
        price_weight=price_weight,
        popularity_weight=popularity_weight,
        diversity_weight=diversity_weight,
        freshness_weight=freshness_weight,
        exploration_weight=exploration_weight,
        long_tail_weight=long_tail_weight,
    )

    candidate_start = time.perf_counter()
    candidate_ids: list[str] = []
    candidate_scores: dict[str, float] = {}
    for rank, candidate in enumerate(search_candidates, start=1):
        raw_id = candidate.get("product_id") or candidate.get("article_id") or candidate.get("item_id")
        if raw_id is None:
            continue
        article_id = str(raw_id)
        candidate_ids.append(article_id)
        try:
            candidate_scores[article_id] = float(candidate.get("score", candidate.get("similarity")))
        except (TypeError, ValueError):
            candidate_scores[article_id] = max(0.01, 1.0 / float(rank))

    search_frame = materialize_external_candidates(
        candidate_ids=candidate_ids,
        candidate_scores=candidate_scores,
        candidate_reason="search_candidate",
    )
    search_frame["_candidate_source_order"] = 0

    recommendation_frame = pd.DataFrame()
    if include_recommendation_candidates:
        recommendation_frame = generate_candidates(
            user_id=user_id,
            top_k=top_n,
            recent_clicks=recent_clicks,
            session_interest=effective_session_interest,
            candidate_pool_size=max(top_n, recommendation_candidate_pool_size),
        )
        recommendation_frame["_candidate_source_order"] = 1

    candidate_items = pd.concat([search_frame, recommendation_frame], ignore_index=True, sort=False)
    if not candidate_items.empty:
        candidate_items = (
            candidate_items.sort_values(["_candidate_source_order", "candidate_score"], ascending=[True, False])
            .drop_duplicates(subset=["article_id"], keep="first")
            .drop(columns=["_candidate_source_order"], errors="ignore")
            .reset_index(drop=True)
        )
    candidate_ms = _elapsed_ms(candidate_start)
    if candidate_items.empty:
        total_ms = _elapsed_ms(total_start)
        return {
            "user_id": user_id,
            "recommendations": [],
            "pipeline_latency": {
                "candidate_ms": candidate_ms,
                "ranking_ms": 0,
                "reranking_ms": 0,
                "total_ms": total_ms,
            },
            "session_context": session_context,
            "persona": persona_hint or "personalized",
            "persona_scores": persona_scores or {},
            "candidate_summary": {
                "search_candidates": 0,
                "recommendation_candidates": 0,
                "combined_candidates": 0,
            },
        }

    ranking_start = time.perf_counter()
    try:
        scored_candidates = score_candidates(
            user_id=user_id,
            candidate_items=candidate_items,
            session_context=session_context,
        )
    except Exception:
        LOGGER.exception("Search reranking stage failed. Falling back to popularity ordering for user_id=%s", user_id)
        scored_candidates = _build_popularity_fallback(candidate_items)
    ranking_ms = _elapsed_ms(ranking_start)

    reranking_start = time.perf_counter()
    recommendations = rerank_recommendations(
        scored_candidates=scored_candidates,
        top_n=top_n,
        random_seed=_random_seed_from_context(user_id=user_id, session_context=session_context),
        rerank_weights=effective_rerank_weights,
    )
    reranking_ms = _elapsed_ms(reranking_start)
    total_ms = _elapsed_ms(total_start)

    LOGGER.info(
        "External candidate rerank completed user_id=%s top_n=%s search_candidates=%s combined_candidates=%s candidate_ms=%s ranking_ms=%s reranking_ms=%s total_ms=%s",
        user_id,
        top_n,
        len(search_frame),
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
        "persona": persona_hint or "personalized",
        "persona_scores": persona_scores or {},
        "candidate_summary": {
            "search_candidates": len(search_frame),
            "recommendation_candidates": len(recommendation_frame),
            "combined_candidates": len(candidate_items),
        },
    }


def recommend(
    user_id: str,
    top_n: int = 10,
    recent_clicks: list[str] | None = None,
    click_count: int = 0,
    session_interest: dict[str, Any] | None = None,
    rerank_weights: dict[str, Any] | None = None,
    persona_hint: str | None = None,
    persona_scores: dict[str, Any] | None = None,
    personalization_weight: float | None = None,
    price_weight: float | None = None,
    popularity_weight: float | None = None,
    diversity_weight: float | None = None,
    freshness_weight: float | None = None,
    exploration_weight: float | None = None,
    long_tail_weight: float | None = None,
) -> dict[str, Any]:
    """Run candidate generation, ranking, and reranking for one user."""

    total_start = time.perf_counter()
    effective_session_interest = _merge_persona_session_interest(
        session_interest=session_interest,
        persona_hint=persona_hint,
        persona_scores=persona_scores,
    )
    session_context = {
        "recent_clicks": recent_clicks or [],
        "session_interest": effective_session_interest or None,
    }
    effective_rerank_weights = _build_rerank_weights(
        rerank_weights,
        persona_hint=persona_hint,
        persona_scores=persona_scores,
        personalization_weight=personalization_weight,
        price_weight=price_weight,
        popularity_weight=popularity_weight,
        diversity_weight=diversity_weight,
        freshness_weight=freshness_weight,
        exploration_weight=exploration_weight,
        long_tail_weight=long_tail_weight,
    )

    candidate_start = time.perf_counter()
    candidate_items = generate_candidates(
        user_id=user_id,
        top_k=top_n,
        recent_clicks=recent_clicks,
        session_interest=effective_session_interest,
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
        rerank_weights=effective_rerank_weights,
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
        "persona": persona_hint or "personalized",
        "persona_scores": persona_scores or {},
    }
