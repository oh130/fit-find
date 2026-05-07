"""Post-ranking reranking bridge for service responses."""

from __future__ import annotations

import random
from typing import Any

import pandas as pd


MAX_CONSECUTIVE_CATEGORY = 2
DEFAULT_EXPLORATION_EPSILON = 1.0
DEFAULT_EXPLORATION_RATIO = 0.6
DEFAULT_MAX_EXPLORATION_SLOTS = 30
DEFAULT_NEW_ITEM_WINDOW_DAYS = 7
DEFAULT_SOFT_DIVERSITY_PENALTY = 0.003
DEFAULT_SOFT_POPULARITY_PENALTY = 0.004
DEFAULT_SOFT_FRESHNESS_BOOST = 0.002
DEFAULT_SCORE_GAP_GUARD = 0.01


def _safe_category(row: dict[str, Any]) -> str:
    category = str(row.get("main_category") or row.get("category") or "UNKNOWN").strip()
    return category or "UNKNOWN"


def _sort_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    sortable = candidates.copy()
    sortable["score"] = pd.to_numeric(sortable.get("score"), errors="coerce").fillna(0.0)
    sortable["popularity"] = pd.to_numeric(sortable.get("popularity"), errors="coerce").fillna(0.0)
    sortable["item_age_days"] = pd.to_numeric(sortable.get("item_age_days"), errors="coerce")
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


def apply_soft_greedy_rerank(
    candidates: pd.DataFrame,
    top_n: int,
    *,
    enable_diversity: bool,
    enable_exploration: bool,
    enable_freshness: bool,
) -> pd.DataFrame:
    """Greedily rerank near-tie candidates with small relevance-preserving boosts/penalties."""

    if candidates.empty or top_n <= 0:
        return candidates.head(0).copy()

    pool_size = min(len(candidates), max(top_n * 3, top_n))
    pool = candidates.head(pool_size).copy().reset_index(drop=True)
    pool["base_score"] = pd.to_numeric(pool.get("score"), errors="coerce").fillna(0.0)
    pool["popularity_norm"] = _normalize_series(pool.get("popularity", pd.Series(0.0, index=pool.index)))
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

    if enable_diversity:
        pool["soft_score"] = pool["soft_score"] - (pool["category_rank"] * DEFAULT_SOFT_DIVERSITY_PENALTY)

    if enable_exploration:
        pool["soft_score"] = pool["soft_score"] - (pool["popularity_norm"] * DEFAULT_SOFT_POPULARITY_PENALTY)

    if enable_freshness:
        freshness_mask = pool["is_new_item_flag"] | (
            pool["item_age_days"].notna() & pool["item_age_days"].le(DEFAULT_NEW_ITEM_WINDOW_DAYS)
        )
        pool.loc[freshness_mask, "soft_score"] = pool.loc[freshness_mask, "soft_score"] + DEFAULT_SOFT_FRESHNESS_BOOST
        reason = pool.get("reason", pd.Series("ranking_score", index=pool.index)).fillna("ranking_score")
        pool.loc[freshness_mask & reason.eq("ranking_score"), "reason"] = "new_item_boost"

    if "is_exploration" not in pool.columns:
        pool["is_exploration"] = False
    exploration_mask = pool["original_rank"].ge(top_n)
    if enable_exploration:
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


def _compute_exploration_slots(top_n: int, requested_slots: int | None = None) -> int:
    if top_n <= 2:
        return 0
    if requested_slots is not None:
        return max(0, min(requested_slots, DEFAULT_MAX_EXPLORATION_SLOTS, top_n))

    ratio_slots = max(1, int(round(top_n * DEFAULT_EXPLORATION_RATIO)))
    return min(ratio_slots, DEFAULT_MAX_EXPLORATION_SLOTS, max(top_n - 1, 0))


def _pick_reason(row: dict[str, Any], exploration_reason: str | None = None) -> str:
    if exploration_reason is not None:
        return exploration_reason
    return str(row.get("reason", "ranking_score"))


def _image_url_for_article(article_id: str) -> str:
    return f"/api/images/{article_id}"


def select_exploration_candidates(
    remaining_candidates: pd.DataFrame,
    already_selected: pd.DataFrame,
    exploration_slots: int,
    epsilon: float = DEFAULT_EXPLORATION_EPSILON,
    enable_freshness: bool = True,
    rng: random.Random | None = None,
) -> pd.DataFrame:
    """Select exploration candidates using epsilon-greedy priorities."""

    if remaining_candidates.empty or exploration_slots <= 0:
        return remaining_candidates.head(0).copy()

    rng = rng or random.Random()
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
    candidates["score_norm"] = _normalize_series(candidates.get("score", pd.Series(0.0, index=candidates.index)))
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
        candidates["exploration_score"] = (
            candidates["score_norm"] * 0.25
            - candidates["category_penalty"] * 0.20
            - candidates["popularity_norm"] * 0.25
            + candidates["coverage_exploration_flag"] * 2.00
            + freshness_mask.astype(float) * 0.30
            + candidates["random_tiebreaker"] * 0.20
        )
        ordered = candidates.sort_values(["exploration_score", "score", "article_id"], ascending=[False, False, True]).copy()
        ordered.loc[:, "reason"] = "mab_exploration"
        ordered.loc[freshness_mask & ~coverage_mask, "reason"] = "new_item_boost"
    else:
        ordered = candidates.sort_values(["score", "popularity", "article_id"], ascending=[False, False, True]).copy()
        ordered.loc[:, "reason"] = ordered["reason"].fillna("ranking_score")

    selected = ordered.head(min(exploration_slots, len(ordered))).copy()
    selected.loc[:, "is_exploration"] = selected["reason"].isin({"mab_exploration", "new_item_boost"})
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
) -> list[dict[str, Any]]:
    """Apply service-level reranking with diversity, freshness, and exploration."""

    if scored_candidates.empty or top_n <= 0:
        return []

    ordered = _sort_candidates(scored_candidates)
    effective_top_n = min(top_n, len(ordered))
    softly_ranked = apply_soft_greedy_rerank(
        ordered,
        top_n=effective_top_n,
        enable_diversity=enable_diversity,
        enable_exploration=enable_exploration,
        enable_freshness=enable_freshness,
    )

    if softly_ranked.empty:
        softly_ranked = ordered.head(effective_top_n).copy()

    effective_exploration_slots = 0
    if enable_exploration:
        effective_exploration_slots = min(_compute_exploration_slots(effective_top_n, exploration_slots), effective_top_n)

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
            rng=rng,
        )
        combined = inject_exploration_slots(primary_ranked=primary_ranked, exploration_candidates=exploration_ranked, top_n=effective_top_n)
    else:
        combined = primary_ranked.head(effective_top_n).copy()

    if enable_diversity:
        final_ranked = apply_diversity_guard(combined, top_n=effective_top_n)
    else:
        final_ranked = combined.head(effective_top_n).copy()

    recommendations: list[dict[str, Any]] = []
    for row in final_ranked.to_dict(orient="records"):
        product_id = str(row.get("article_id", ""))
        recommendations.append(
            {
                "product_id": product_id,
                "score": float(row.get("score", 0.0)),
                "reason": _pick_reason(row),
                "is_exploration": bool(row.get("is_exploration", False)),
                "image_url": _image_url_for_article(product_id),
            }
        )
    return recommendations
