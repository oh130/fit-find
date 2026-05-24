"""Post-ranking reranking bridge for service responses."""

from __future__ import annotations

import random
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pandas as pd


MAX_CONSECUTIVE_CATEGORY = 2
DEFAULT_EXPLORATION_EPSILON = 1.0
DEFAULT_EXPLORATION_RATIO = 0.6
DEFAULT_MAX_EXPLORATION_SLOTS = 30
DEFAULT_NEW_ITEM_WINDOW_DAYS = 7
DEFAULT_SOFT_DIVERSITY_PENALTY = 0.003
DEFAULT_SOFT_LONG_TAIL_PENALTY = 0.004
DEFAULT_SOFT_FRESHNESS_BOOST = 0.002
DEFAULT_SOFT_PRICE_PENALTY = 0.002
DEFAULT_PERSONALIZATION_BLEND_STRENGTH = 0.035
DEFAULT_SCORE_GAP_GUARD = 0.01
MAX_RERANK_WEIGHT = 5.0
PRICE_KRW_FACTOR = 1_000_000


@dataclass(frozen=True)
class RerankWeights:
    """User-tunable reranking weights.

    A value of 1.0 preserves the current production behavior. Higher values
    increase the corresponding reranking pressure; 0.0 disables that pressure
    while keeping the stage enabled.
    """

    diversity: float = 1.0
    exploration: float = 1.0
    freshness: float = 1.0
    personalization: float = 1.0
    popularity: float = 0.0
    long_tail: float = 1.0
    price: float = 1.0


def _coerce_weight(value: Any, default: float = 1.0) -> float:
    try:
        weight = float(value)
    except (TypeError, ValueError):
        return default
    if pd.isna(weight):
        return default
    return max(0.0, min(weight, MAX_RERANK_WEIGHT))


def normalize_rerank_weights(weights: Mapping[str, Any] | RerankWeights | None = None) -> RerankWeights:
    """Normalize external slider parameters into bounded reranking weights."""

    if isinstance(weights, RerankWeights):
        return weights

    raw = dict(weights or {})

    def pick(short_key: str, default: float = 1.0) -> float:
        return _coerce_weight(raw.get(f"{short_key}_weight", raw.get(short_key)), default=default)

    # Backward compatibility: older callers used popularity_weight as a
    # long-tail pressure. The frontend now uses it as an actual popularity
    # preference, so keep the old pressure only when no popularity override is
    # provided.
    explicit_popularity = "popularity_weight" in raw or "popularity" in raw

    return RerankWeights(
        diversity=pick("diversity"),
        exploration=pick("exploration"),
        freshness=pick("freshness"),
        personalization=pick("personalization"),
        popularity=pick("popularity", default=0.0),
        long_tail=pick("long_tail", default=0.0 if explicit_popularity else 1.0),
        price=pick("price"),
    )


def _safe_category(row: dict[str, Any]) -> str:
    category = str(row.get("main_category") or row.get("category") or "UNKNOWN").strip()
    return category or "UNKNOWN"


def _normalize_article_id(value: Any) -> str:
    article_id = str(value or "").strip()
    if article_id.isdigit():
        return article_id.zfill(10)
    return article_id


def _sort_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    sortable = candidates.copy()
    sortable["score"] = pd.to_numeric(sortable.get("score"), errors="coerce").fillna(0.0)
    sortable["popularity"] = pd.to_numeric(
        sortable.get("popularity", pd.Series(0.0, index=sortable.index)),
        errors="coerce",
    ).fillna(0.0)
    sortable["item_age_days"] = pd.to_numeric(
        sortable.get("item_age_days", pd.Series(float("nan"), index=sortable.index)),
        errors="coerce",
    )
    if "is_new_item" in sortable.columns:
        sortable["is_new_item"] = sortable["is_new_item"].fillna(False).astype(bool)
    else:
        sortable["is_new_item"] = False
    return sortable.sort_values(["score", "popularity", "article_id"], ascending=[False, False, True]).reset_index(drop=True)


def _would_break_diversity(staged_rows: list[dict[str, Any]], candidate_row: dict[str, Any]) -> bool:
    if len(staged_rows) < MAX_CONSECUTIVE_CATEGORY:
        return False

    tail_categories = [_safe_category(row) for row in staged_rows[-MAX_CONSECUTIVE_CATEGORY:]]
    return len(set(tail_categories)) == 1 and tail_categories[-1] == _safe_category(candidate_row)


def apply_diversity_guard(candidates: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """Prevent 3 consecutive items from the same category when possible."""

    if candidates.empty or top_n <= 0:
        return candidates.head(0).copy()

    remaining = candidates.to_dict(orient="records")
    staged_rows: list[dict[str, Any]] = []

    while remaining and len(staged_rows) < top_n:
        selected_index: int | None = None
        for index, row in enumerate(remaining):
            if not _would_break_diversity(staged_rows, row):
                selected_index = index
                break

        if selected_index is None:
            selected_index = 0

        staged_rows.append(remaining.pop(selected_index))

    return pd.DataFrame(staged_rows)


def _normalize_series(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").fillna(0.0)
    minimum = float(numeric.min()) if not numeric.empty else 0.0
    maximum = float(numeric.max()) if not numeric.empty else 0.0
    if maximum <= minimum:
        return pd.Series(0.0, index=numeric.index, dtype="float64")
    return (numeric - minimum) / (maximum - minimum)


def _normalized_price(candidates: pd.DataFrame) -> pd.Series:
    if "avg_price" in candidates.columns:
        return _normalize_series(candidates["avg_price"])
    if "price" in candidates.columns:
        return _normalize_series(candidates["price"])
    return pd.Series(0.0, index=candidates.index, dtype="float64")


def apply_soft_greedy_rerank(
    candidates: pd.DataFrame,
    top_n: int,
    *,
    enable_diversity: bool,
    enable_exploration: bool,
    enable_freshness: bool,
    rerank_weights: Mapping[str, Any] | RerankWeights | None = None,
) -> pd.DataFrame:
    """Greedily rerank near-tie candidates with small relevance-preserving boosts/penalties."""

    if candidates.empty or top_n <= 0:
        return candidates.head(0).copy()

    weights = normalize_rerank_weights(rerank_weights)
    pool_size = min(len(candidates), max(top_n * 3, top_n))
    pool = candidates.head(pool_size).copy().reset_index(drop=True)
    pool["base_score"] = pd.to_numeric(pool.get("score"), errors="coerce").fillna(0.0)
    pool["popularity_norm"] = _normalize_series(pool.get("popularity", pd.Series(0.0, index=pool.index)))
    pool["price_norm"] = _normalized_price(pool)
    pool["is_new_item_flag"] = (
        pool.get("is_new_item", pd.Series(False, index=pool.index))
        .fillna(False)
        .astype(bool)
    )
    pool["item_age_days"] = pd.to_numeric(pool.get("item_age_days"), errors="coerce")
    pool["category_key"] = pool.apply(lambda row: _safe_category(row.to_dict()), axis=1)
    pool["original_rank"] = range(len(pool))

    pool["category_rank"] = pool.groupby("category_key", sort=False).cumcount()
    pool["soft_score"] = pool["base_score"]

    if weights.personalization > 0 or weights.popularity > 0:
        base_norm = _normalize_series(pool["base_score"])
        preference_total = max(weights.personalization + weights.popularity, 1e-9)
        preference_score = (
            (base_norm * weights.personalization)
            + (pool["popularity_norm"] * weights.popularity)
        ) / preference_total
        pool["soft_score"] = pool["soft_score"] + (
            (preference_score - base_norm) * DEFAULT_PERSONALIZATION_BLEND_STRENGTH
        )

    if enable_diversity:
        diversity_penalty = DEFAULT_SOFT_DIVERSITY_PENALTY * weights.diversity
        pool["soft_score"] = pool["soft_score"] - (pool["category_rank"] * diversity_penalty)

    if enable_exploration and weights.long_tail > 0:
        long_tail_penalty = DEFAULT_SOFT_LONG_TAIL_PENALTY * weights.long_tail
        pool["soft_score"] = pool["soft_score"] - (pool["popularity_norm"] * long_tail_penalty)

    if weights.price > 1.0:
        price_pressure = DEFAULT_SOFT_PRICE_PENALTY * (weights.price - 1.0)
        pool["soft_score"] = pool["soft_score"] - (pool["price_norm"] * price_pressure)

    if enable_freshness:
        freshness_mask = pool["is_new_item_flag"] | (
            pool["item_age_days"].notna() & pool["item_age_days"].le(DEFAULT_NEW_ITEM_WINDOW_DAYS)
        )
        freshness_boost = DEFAULT_SOFT_FRESHNESS_BOOST * weights.freshness
        pool.loc[freshness_mask, "soft_score"] = pool.loc[freshness_mask, "soft_score"] + freshness_boost
        reason = pool.get("reason", pd.Series("ranking_score", index=pool.index)).fillna("ranking_score")
        pool.loc[freshness_mask & reason.eq("ranking_score"), "reason"] = "new_item_boost"

    if "is_exploration" not in pool.columns:
        pool["is_exploration"] = False
    exploration_mask = pool["original_rank"].ge(top_n)
    if enable_exploration and weights.exploration > 0:
        pool.loc[exploration_mask, "is_exploration"] = True
        reason = pool.get("reason", pd.Series("ranking_score", index=pool.index)).fillna("ranking_score")
        pool.loc[exploration_mask & reason.eq("ranking_score"), "reason"] = "mab_exploration"

    best_base_score = float(pool["base_score"].max())
    score_floor = best_base_score - DEFAULT_SCORE_GAP_GUARD
    viable = pool.loc[pool["base_score"].ge(score_floor)].copy()
    remainder = pool.loc[~pool.index.isin(viable.index)].copy()
    ordered = pd.concat(
        [
            viable.sort_values(["soft_score", "base_score", "popularity", "article_id"], ascending=[False, False, False, True]),
            remainder.sort_values(["base_score", "popularity", "article_id"], ascending=[False, False, True]),
        ],
        ignore_index=True,
    )
    return ordered.head(min(top_n, len(ordered))).copy()


def _compute_exploration_slots(
    top_n: int,
    requested_slots: int | None = None,
    exploration_weight: float = 1.0,
) -> int:
    if top_n <= 2:
        return 0
    if requested_slots is not None:
        return max(0, min(requested_slots, DEFAULT_MAX_EXPLORATION_SLOTS, top_n))

    weighted_ratio = DEFAULT_EXPLORATION_RATIO * _coerce_weight(exploration_weight)
    ratio_slots = int(round(top_n * weighted_ratio))
    if weighted_ratio > 0:
        ratio_slots = max(1, ratio_slots)
    return min(ratio_slots, DEFAULT_MAX_EXPLORATION_SLOTS, max(top_n - 1, 0))


def _pick_reason(row: dict[str, Any], exploration_reason: str | None = None) -> str:
    if exploration_reason is not None:
        return exploration_reason
    reason = row.get("reason", "ranking_score")
    if pd.isna(reason):
        return "ranking_score"
    return str(reason)


def _image_url_for_article(article_id: str) -> str:
    return f"/api/images/{article_id}"


def _response_text(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.upper() == "UNKNOWN":
        return None
    return text


def _response_float(value: Any) -> float | None:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return None
    return float(numeric)


def _response_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return bool(value)


def _brand_label(row: dict[str, Any]) -> str:
    department = _response_text(row.get("department_name"))
    return f"H&M · {department}" if department else "H&M"


def _recommendation_metadata(row: dict[str, Any]) -> dict[str, Any]:
    avg_price = _response_float(row.get("avg_price"))
    price_krw = int(round(avg_price * PRICE_KRW_FACTOR)) if avg_price is not None and avg_price > 0 else None
    outfit_role = _response_text(row.get("outfit_role")) or "unknown"

    return {
        "name": _response_text(row.get("prod_name")),
        "brand": _brand_label(row),
        "category": _response_text(row.get("category")) or _response_text(row.get("main_category")),
        "main_category": _response_text(row.get("main_category")),
        "product_type": _response_text(row.get("product_type_name")),
        "product_group": _response_text(row.get("product_group_name")),
        "color": _response_text(row.get("color")) or _response_text(row.get("colour_group_name")),
        "avg_price": avg_price,
        "price_krw": price_krw,
        "price": price_krw,
        "outfit_role": outfit_role,
        "outfit_eligible": _response_bool(row.get("outfit_eligible", False)),
    }


def _display_scores(raw_scores: pd.Series) -> list[float]:
    """Scale final raw scores for UI display while preserving rank ratios."""

    scores = pd.to_numeric(raw_scores, errors="coerce").fillna(0.0).clip(lower=0.0)
    max_score = float(scores.max()) if not scores.empty else 0.0
    if max_score <= 0.0:
        return [0.0 for _ in range(len(scores))]
    return (scores / max_score).clip(upper=1.0).astype(float).tolist()


def select_exploration_candidates(
    remaining_candidates: pd.DataFrame,
    already_selected: pd.DataFrame,
    exploration_slots: int,
    epsilon: float = DEFAULT_EXPLORATION_EPSILON,
    enable_freshness: bool = True,
    rerank_weights: Mapping[str, Any] | RerankWeights | None = None,
    bandit_scores: Mapping[str, float] | None = None,
    rng: random.Random | None = None,
) -> pd.DataFrame:
    """Select exploration candidates using reward-aware epsilon-greedy priorities."""

    if remaining_candidates.empty or exploration_slots <= 0:
        return remaining_candidates.head(0).copy()

    rng = rng or random.Random()
    weights = normalize_rerank_weights(rerank_weights)
    selected_ids = set(already_selected.get("article_id", pd.Series(dtype=str)).astype(str))
    category_counts = already_selected.get("main_category", already_selected.get("category", pd.Series(dtype=str))).astype(str).value_counts()

    candidates = remaining_candidates.copy()
    candidates = candidates.loc[~candidates["article_id"].astype(str).isin(selected_ids)].copy()
    if candidates.empty:
        return candidates

    candidates["item_age_days"] = pd.to_numeric(candidates.get("item_age_days"), errors="coerce")
    if "is_new_item" in candidates.columns:
        candidates["is_new_item"] = candidates["is_new_item"].fillna(False).astype(bool)
    else:
        candidates["is_new_item"] = False
    candidates["category_key"] = candidates.apply(lambda row: _safe_category(row.to_dict()), axis=1)
    candidates["category_penalty"] = candidates["category_key"].map(category_counts.to_dict()).fillna(0.0)

    candidates["popularity_norm"] = _normalize_series(candidates.get("popularity", pd.Series(0.0, index=candidates.index)))
    candidates["price_norm"] = _normalized_price(candidates)
    candidates["score_norm"] = _normalize_series(candidates.get("score", pd.Series(0.0, index=candidates.index)))
    raw_bandit_scores = dict(bandit_scores or {})
    candidates["bandit_ucb_score"] = (
        candidates["article_id"]
        .map(lambda article_id: float(raw_bandit_scores.get(_normalize_article_id(article_id), 0.0)))
        .fillna(0.0)
    )
    candidates["bandit_ucb_norm"] = _normalize_series(candidates["bandit_ucb_score"])
    candidates["coverage_exploration_flag"] = (
        candidates.get("candidate_reason", pd.Series("", index=candidates.index))
        .astype(str)
        .eq("coverage_exploration")
        .astype(float)
    )
    candidates["random_tiebreaker"] = [rng.random() for _ in range(len(candidates))]

    coverage_mask = candidates["coverage_exploration_flag"].eq(1.0)
    if enable_freshness:
        freshness_mask = candidates["is_new_item"] | (
            candidates["item_age_days"].notna() & candidates["item_age_days"].le(DEFAULT_NEW_ITEM_WINDOW_DAYS)
        )
    else:
        freshness_mask = pd.Series(False, index=candidates.index)

    if rng.random() < epsilon:
        price_pressure = 0.20 * max(weights.price - 1.0, 0.0)
        candidates["exploration_score"] = (
            candidates["score_norm"] * 0.25
            - candidates["category_penalty"] * (0.20 * weights.diversity)
            - candidates["popularity_norm"] * (0.25 * weights.long_tail)
            + candidates["popularity_norm"] * (0.20 * weights.popularity)
            - candidates["price_norm"] * price_pressure
            + candidates["coverage_exploration_flag"] * (2.00 * weights.exploration)
            + candidates["bandit_ucb_norm"] * (0.60 * weights.exploration)
            + freshness_mask.astype(float) * (0.30 * weights.freshness)
            + candidates["random_tiebreaker"] * (0.20 * weights.exploration)
        )
        ordered = candidates.sort_values(["exploration_score", "score", "article_id"], ascending=[False, False, True]).copy()
        ordered.loc[:, "reason"] = "mab_exploration"
        ordered.loc[ordered["bandit_ucb_score"].gt(0), "reason"] = "bandit_reward_exploration"
        ordered.loc[freshness_mask & ~coverage_mask, "reason"] = "new_item_boost"
    else:
        ordered = candidates.sort_values(["score", "popularity", "article_id"], ascending=[False, False, True]).copy()
        ordered.loc[:, "reason"] = ordered["reason"].fillna("ranking_score")

    selected = ordered.head(min(exploration_slots, len(ordered))).copy()
    selected.loc[:, "is_exploration"] = selected["reason"].isin(
        {"mab_exploration", "bandit_reward_exploration", "new_item_boost"}
    )
    return selected


def _exploration_positions(top_n: int, slot_count: int) -> list[int]:
    if slot_count <= 0 or top_n <= 0:
        return []
    slot_count = min(slot_count, max(top_n - 1, 0))
    if slot_count == 1:
        return [min(max(top_n // 3, 1), top_n - 1)]

    spacing = top_n / float(slot_count + 1)
    positions: list[int] = []
    used_positions: set[int] = set()
    for slot_index in range(1, slot_count + 1):
        position = min(max(int(round(spacing * slot_index)), 1), top_n - 1)
        while position in used_positions and position < top_n - 1:
            position += 1
        while position in used_positions and position > 1:
            position -= 1
        if position not in used_positions:
            positions.append(position)
            used_positions.add(position)
    return sorted(positions)


def inject_exploration_slots(
    primary_ranked: pd.DataFrame,
    exploration_candidates: pd.DataFrame,
    top_n: int,
) -> pd.DataFrame:
    """Inject exploration rows into the ranked list at stable positions."""

    if top_n <= 0:
        return primary_ranked.head(0).copy()

    base_rows = primary_ranked.to_dict(orient="records")
    exploration_rows = exploration_candidates.to_dict(orient="records")
    if not exploration_rows:
        return pd.DataFrame(base_rows[:top_n])

    positions = _exploration_positions(top_n=top_n, slot_count=len(exploration_rows))
    merged_rows: list[dict[str, Any]] = []
    base_index = 0
    exploration_index = 0

    for result_index in range(top_n):
        if exploration_index < len(exploration_rows) and result_index in positions:
            merged_rows.append(exploration_rows[exploration_index])
            exploration_index += 1
            continue

        if base_index < len(base_rows):
            merged_rows.append(base_rows[base_index])
            base_index += 1
            continue

        if exploration_index < len(exploration_rows):
            merged_rows.append(exploration_rows[exploration_index])
            exploration_index += 1

    return pd.DataFrame(merged_rows[:top_n])


def rerank_recommendations(
    scored_candidates: pd.DataFrame,
    top_n: int,
    exploration_slots: int | None = None,
    epsilon: float = DEFAULT_EXPLORATION_EPSILON,
    random_seed: int | None = None,
    enable_diversity: bool = True,
    enable_exploration: bool = True,
    enable_freshness: bool = True,
    rerank_weights: Mapping[str, Any] | RerankWeights | None = None,
    bandit_scores: Mapping[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Apply service-level reranking with diversity, freshness, and exploration."""

    if scored_candidates.empty or top_n <= 0:
        return []

    weights = normalize_rerank_weights(rerank_weights)
    ordered = _sort_candidates(scored_candidates)
    effective_top_n = min(top_n, len(ordered))
    softly_ranked = apply_soft_greedy_rerank(
        ordered,
        top_n=effective_top_n,
        enable_diversity=enable_diversity,
        enable_exploration=enable_exploration,
        enable_freshness=enable_freshness,
        rerank_weights=weights,
    )

    if softly_ranked.empty:
        softly_ranked = ordered.head(effective_top_n).copy()

    effective_exploration_slots = 0
    if enable_exploration:
        effective_exploration_slots = min(
            _compute_exploration_slots(
                effective_top_n,
                exploration_slots,
                exploration_weight=weights.exploration,
            ),
            effective_top_n,
        )

    primary_slots = max(effective_top_n - effective_exploration_slots, 0)

    if enable_diversity:
        primary_ranked = apply_diversity_guard(
            softly_ranked,
            top_n=max(primary_slots, effective_top_n if effective_exploration_slots == 0 else primary_slots),
        )
    else:
        primary_ranked = softly_ranked.head(max(primary_slots, effective_top_n if effective_exploration_slots == 0 else primary_slots)).copy()

    remaining_ids = set(primary_ranked.get("article_id", pd.Series(dtype=str)).astype(str))
    remaining_candidates = ordered.loc[~ordered["article_id"].astype(str).isin(remaining_ids)].copy()

    if enable_exploration and effective_exploration_slots > 0:
        rng = random.Random(random_seed)
        exploration_ranked = select_exploration_candidates(
            remaining_candidates=remaining_candidates,
            already_selected=primary_ranked,
            exploration_slots=effective_exploration_slots,
            epsilon=epsilon,
            enable_freshness=enable_freshness,
            rerank_weights=weights,
            bandit_scores=bandit_scores,
            rng=rng,
        )
        combined = inject_exploration_slots(primary_ranked=primary_ranked, exploration_candidates=exploration_ranked, top_n=effective_top_n)
    else:
        combined = primary_ranked.head(effective_top_n).copy()

    if enable_diversity:
        final_ranked = apply_diversity_guard(combined, top_n=effective_top_n)
    else:
        final_ranked = combined.head(effective_top_n).copy()

    display_scores = _display_scores(final_ranked.get("score", pd.Series(0.0, index=final_ranked.index)))

    recommendations: list[dict[str, Any]] = []
    for row, display_score in zip(final_ranked.to_dict(orient="records"), display_scores, strict=False):
        product_id = str(row.get("article_id", ""))
        recommendations.append(
            {
                "product_id": product_id,
                "score": display_score,
                "reason": _pick_reason(row),
                "is_exploration": bool(row.get("is_exploration", False)),
                "image_url": _image_url_for_article(product_id),
                **_recommendation_metadata(row),
            }
        )
    return recommendations
