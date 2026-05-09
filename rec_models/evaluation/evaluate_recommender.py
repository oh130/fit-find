"""Offline evaluation CLI for the recommendation serving pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from rec_models.evaluation.metrics import coverage_at_k, hit_rate_at_k, mean_metric, ndcg_at_k
    from rec_models.evaluation.data_utils import (
        EvaluationContext,
        build_session_context,
        build_evaluation_context,
        load_evaluation_data,
    )
    from rec_models.common.utils import build_experiment_report, write_json_report
    from rec_models.serving.candidate_service import generate_candidates, load_article_catalog
    from rec_models.serving.recommend_service import rank_candidates_to_recommendations
    from rec_models.serving.ranking_service import load_customer_features, score_candidate_batch
except ImportError:  # pragma: no cover - supports running from rec_models/ as cwd
    from evaluation.metrics import coverage_at_k, hit_rate_at_k, mean_metric, ndcg_at_k  # type: ignore[no-redef]
    from evaluation.data_utils import (  # type: ignore[no-redef]
        EvaluationContext,
        build_session_context,
        build_evaluation_context,
        load_evaluation_data,
    )
    from common.utils import build_experiment_report, write_json_report  # type: ignore[no-redef]
    from serving.candidate_service import generate_candidates, load_article_catalog  # type: ignore[no-redef]
    from serving.recommend_service import rank_candidates_to_recommendations  # type: ignore[no-redef]
    from serving.ranking_service import load_customer_features, score_candidate_batch  # type: ignore[no-redef]


DEFAULT_TOP_K = 50
DEFAULT_EVALUATION_DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "train_data_dev.csv"
E2E_EVALUATION_COLUMNS = [
    "customer_id",
    "article_id",
    "label",
    "sales_channel_id",
    "recent_clicks",
    "session_interest",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the recommendation pipeline offline.")
    parser.add_argument("--data", type=Path, default=DEFAULT_EVALUATION_DATA_PATH, help="Path to processed recommendation test data.")
    parser.add_argument("--top_k", type=int, default=DEFAULT_TOP_K, help="Cutoff K for ranking metrics.")
    parser.add_argument("--candidate-k", type=int, default=300, help="Recorded candidate cutoff K for experiment metadata.")
    parser.add_argument("--experiment-name", type=str, default="recommendation_eval", help="Stable experiment name for saved reports.")
    parser.add_argument("--seed", type=int, default=42, help="Recorded random seed for reproducibility metadata.")
    parser.add_argument("--split-name", type=str, default="unspecified", help="Recorded split name for experiment metadata.")
    parser.add_argument("--output-json", type=Path, help="Optional output path for JSON metrics.")
    parser.add_argument("--max-users", type=int, help="Optional cap for smoke checks or faster iteration.")
    parser.add_argument("--disable-reranking", action="store_true", help="Disable all reranking logic and use ranking order only.")
    parser.add_argument("--disable-diversity", action="store_true", help="Disable diversity control during reranking.")
    parser.add_argument("--disable-exploration", action="store_true", help="Disable exploration slot injection during reranking.")
    parser.add_argument("--disable-freshness", action="store_true", help="Disable freshness-based new item boost during exploration.")
    parser.add_argument(
        "--use-serving-candidates",
        action="store_true",
        help="Evaluate end-to-end with serving candidate generation instead of only reranking provided candidate rows.",
    )
    parser.add_argument(
        "--skip-popularity-baseline",
        action="store_true",
        help="Skip the popularity baseline comparison.",
    )
    return parser.parse_args()


def enrich_candidate_rows(data: pd.DataFrame) -> pd.DataFrame:
    """Attach serving-time metadata such as popularity to evaluation candidates."""

    catalog = load_article_catalog().copy()
    merge_columns = [
        column
        for column in ("article_id", "popularity", "item_age_days", "is_new_item", "category", "main_category", "color")
        if column in catalog.columns
    ]
    enriched = data.merge(catalog.loc[:, merge_columns].drop_duplicates("article_id"), on="article_id", how="left", suffixes=("", "_catalog"))

    for column in ("category", "main_category", "color"):
        catalog_column = f"{column}_catalog"
        if column not in enriched.columns and catalog_column in enriched.columns:
            enriched[column] = enriched[catalog_column]
        elif catalog_column in enriched.columns:
            enriched[column] = enriched[column].fillna(enriched[catalog_column])

    if "popularity" not in enriched.columns:
        enriched["popularity"] = 0.0
    enriched["popularity"] = pd.to_numeric(enriched["popularity"], errors="coerce").fillna(0.0)

    if "item_age_days" not in enriched.columns:
        enriched["item_age_days"] = pd.NA
    if "is_new_item" not in enriched.columns:
        enriched["is_new_item"] = False

    return enriched


def build_cold_start_user_set(user_ids: list[str]) -> set[str]:
    """Identify users missing from serving-time customer features."""

    customer_features = load_customer_features()
    if customer_features.empty or "customer_id" not in customer_features.columns:
        return set()

    known_users = set(customer_features["customer_id"].astype(str))
    return {user_id for user_id in user_ids if user_id not in known_users}


def popularity_recommendations(user_rows: pd.DataFrame, top_k: int) -> list[str]:
    """Return a popularity-only baseline ranking for one user's candidates."""

    if user_rows.empty or top_k <= 0:
        return []

    ordered = user_rows.copy()
    ordered["popularity"] = pd.to_numeric(ordered.get("popularity"), errors="coerce").fillna(0.0)
    ordered = ordered.sort_values(["popularity", "article_id"], ascending=[False, True]).head(top_k)
    return ordered["article_id"].astype(str).tolist()


def compute_metric_summary(
    ranked_lists: list[list[str]],
    relevant_lists: list[list[str]],
    total_candidate_count: int,
    k: int,
    evaluated_users: int,
) -> dict[str, Any]:
    """Aggregate ranking metrics into one summary."""

    return {
        "users_evaluated": evaluated_users,
        f"HitRate@{k}": mean_metric(ranked_lists, relevant_lists, hit_rate_at_k, k),
        f"NDCG@{k}": mean_metric(ranked_lists, relevant_lists, ndcg_at_k, k),
        f"Coverage@{k}": coverage_at_k(ranked_lists, total_candidate_count=total_candidate_count, k=k),
    }


def evaluate_recommender(
    data: pd.DataFrame,
    top_k: int,
    candidate_k: int = 300,
    max_users: int | None = None,
    context: EvaluationContext | None = None,
    enable_reranking: bool = True,
    enable_diversity: bool = True,
    enable_exploration: bool = True,
    enable_freshness: bool = True,
    use_serving_candidates: bool = False,
) -> dict[str, Any]:
    """Evaluate the current serving ranking+rerranking pipeline."""

    evaluation_context = context or build_evaluation_context(data, max_users=max_users)
    cold_start_users = build_cold_start_user_set(evaluation_context.sampled_user_ids)
    total_candidate_count = (
        int(load_article_catalog()["article_id"].astype(str).nunique())
        if use_serving_candidates
        else evaluation_context.total_candidate_count
    )
    scored_rows_by_id: dict[str, pd.DataFrame] = {}
    if not use_serving_candidates:
        user_set = set(evaluation_context.sampled_user_ids)
        scored_batch = score_candidate_batch(
            data.loc[data["customer_id"].astype(str).isin(user_set)].copy()
        )
        scored_rows_by_id = {
            str(user_id): user_rows.copy()
            for user_id, user_rows in scored_batch.groupby("customer_id", sort=False)
        }

    ranked_lists: list[list[str]] = []
    relevant_lists: list[list[str]] = []
    cold_ranked_lists: list[list[str]] = []
    cold_relevant_lists: list[list[str]] = []

    for user_id in evaluation_context.sampled_user_ids:
        user_rows = evaluation_context.user_rows_by_id.get(user_id)
        if user_rows is None or user_rows.empty:
            continue

        session_context = build_session_context(user_rows)
        if use_serving_candidates:
            scored_user_rows = generate_candidates(
                user_id=user_id,
                top_k=candidate_k,
                recent_clicks=session_context.get("recent_clicks"),
                session_interest=session_context.get("session_interest"),
            )
            if scored_user_rows.empty:
                continue
        else:
            scored_user_rows = scored_rows_by_id.get(user_id)
            if scored_user_rows is None or scored_user_rows.empty:
                continue

        recommendations = rank_candidates_to_recommendations(
            user_id=user_id,
            candidate_items=scored_user_rows,
            top_n=top_k,
            session_context=session_context,
            enable_diversity=enable_reranking and enable_diversity,
            enable_exploration=enable_reranking and enable_exploration,
            enable_freshness=enable_reranking and enable_freshness,
        )
        ranked_items = [str(item["product_id"]) for item in recommendations]
        relevant_items = evaluation_context.ground_truth_by_user[user_id]

        ranked_lists.append(ranked_items)
        relevant_lists.append(relevant_items)
        if user_id in cold_start_users:
            cold_ranked_lists.append(ranked_items)
            cold_relevant_lists.append(relevant_items)

    result = compute_metric_summary(
        ranked_lists=ranked_lists,
        relevant_lists=relevant_lists,
        total_candidate_count=total_candidate_count,
        k=top_k,
        evaluated_users=len(ranked_lists),
    )
    result["cold_start_subset"] = compute_metric_summary(
        ranked_lists=cold_ranked_lists,
        relevant_lists=cold_relevant_lists,
        total_candidate_count=total_candidate_count,
        k=top_k,
        evaluated_users=len(cold_ranked_lists),
    )
    return result


def evaluate_popularity_baseline(
    data: pd.DataFrame,
    top_k: int,
    candidate_k: int = 300,
    max_users: int | None = None,
    context: EvaluationContext | None = None,
    use_serving_candidates: bool = False,
) -> dict[str, Any]:
    """Evaluate a popularity-only baseline on the same user candidate sets."""

    evaluation_context = context or build_evaluation_context(data, max_users=max_users)
    cold_start_users = build_cold_start_user_set(evaluation_context.sampled_user_ids)
    total_candidate_count = (
        int(load_article_catalog()["article_id"].astype(str).nunique())
        if use_serving_candidates
        else evaluation_context.total_candidate_count
    )

    ranked_lists: list[list[str]] = []
    relevant_lists: list[list[str]] = []
    cold_ranked_lists: list[list[str]] = []
    cold_relevant_lists: list[list[str]] = []

    for user_id in evaluation_context.sampled_user_ids:
        user_rows = evaluation_context.user_rows_by_id.get(user_id)
        if user_rows is None or user_rows.empty:
            continue

        if use_serving_candidates:
            session_context = build_session_context(user_rows)
            candidate_items = generate_candidates(
                user_id=user_id,
                top_k=candidate_k,
                recent_clicks=session_context.get("recent_clicks"),
                session_interest=session_context.get("session_interest"),
            )
            if candidate_items.empty:
                continue
            ranked_items = popularity_recommendations(user_rows=candidate_items, top_k=top_k)
        else:
            ranked_items = popularity_recommendations(user_rows=user_rows, top_k=top_k)
        relevant_items = evaluation_context.ground_truth_by_user[user_id]

        ranked_lists.append(ranked_items)
        relevant_lists.append(relevant_items)
        if user_id in cold_start_users:
            cold_ranked_lists.append(ranked_items)
            cold_relevant_lists.append(relevant_items)

    result = compute_metric_summary(
        ranked_lists=ranked_lists,
        relevant_lists=relevant_lists,
        total_candidate_count=total_candidate_count,
        k=top_k,
        evaluated_users=len(ranked_lists),
    )
    result["cold_start_subset"] = compute_metric_summary(
        ranked_lists=cold_ranked_lists,
        relevant_lists=cold_relevant_lists,
        total_candidate_count=total_candidate_count,
        k=top_k,
        evaluated_users=len(cold_ranked_lists),
    )
    return result


def build_comparison_summary(
    ranking_metrics: dict[str, Any],
    baseline_metrics: dict[str, Any] | None,
    top_k: int,
) -> dict[str, Any]:
    """Add lift numbers when the popularity baseline is available."""

    summary = {"current_model": ranking_metrics}
    if baseline_metrics is None:
        return summary

    hit_key = f"HitRate@{top_k}"
    ndcg_key = f"NDCG@{top_k}"
    coverage_key = f"Coverage@{top_k}"
    summary["popularity_baseline"] = baseline_metrics
    summary["improvement_vs_popularity"] = {
        hit_key: ranking_metrics[hit_key] - baseline_metrics[hit_key],
        ndcg_key: ranking_metrics[ndcg_key] - baseline_metrics[ndcg_key],
        coverage_key: ranking_metrics[coverage_key] - baseline_metrics[coverage_key],
    }
    return summary


def _format_metric_line(label: str, value: float) -> str:
    return f"{label:<18} {value:.6f}"


def print_evaluation_report(results: dict[str, Any], top_k: int) -> None:
    """Pretty-print evaluation metrics to the console."""

    hit_key = f"HitRate@{top_k}"
    ndcg_key = f"NDCG@{top_k}"
    coverage_key = f"Coverage@{top_k}"

    current = results["current_model"]
    print("Current Model")
    print(f"{'users evaluated':<18} {current['users_evaluated']}")
    print(_format_metric_line(hit_key, current[hit_key]))
    print(_format_metric_line(ndcg_key, current[ndcg_key]))
    print(_format_metric_line(coverage_key, current[coverage_key]))

    cold = current["cold_start_subset"]
    print("\nCold Start Subset")
    print(f"{'users evaluated':<18} {cold['users_evaluated']}")
    print(_format_metric_line(hit_key, cold[hit_key]))
    print(_format_metric_line(ndcg_key, cold[ndcg_key]))
    print(_format_metric_line(coverage_key, cold[coverage_key]))

    baseline = results.get("popularity_baseline")
    if baseline is not None:
        print("\nPopularity Baseline")
        print(f"{'users evaluated':<18} {baseline['users_evaluated']}")
        print(_format_metric_line(hit_key, baseline[hit_key]))
        print(_format_metric_line(ndcg_key, baseline[ndcg_key]))
        print(_format_metric_line(coverage_key, baseline[coverage_key]))

        improvement = results["improvement_vs_popularity"]
        print("\nImprovement vs Popularity")
        print(_format_metric_line(hit_key, improvement[hit_key]))
        print(_format_metric_line(ndcg_key, improvement[ndcg_key]))
        print(_format_metric_line(coverage_key, improvement[coverage_key]))


def main() -> None:
    args = parse_args()

    data = enrich_candidate_rows(load_evaluation_data(args.data, columns=E2E_EVALUATION_COLUMNS))
    enable_reranking = not args.disable_reranking
    enable_diversity = not args.disable_diversity
    enable_exploration = not args.disable_exploration
    enable_freshness = not args.disable_freshness
    ranking_metrics = evaluate_recommender(
        data=data,
        top_k=args.top_k,
        candidate_k=args.candidate_k,
        max_users=args.max_users,
        enable_reranking=enable_reranking,
        enable_diversity=enable_diversity,
        enable_exploration=enable_exploration,
        enable_freshness=enable_freshness,
        use_serving_candidates=args.use_serving_candidates,
    )
    baseline_metrics = None
    if not args.skip_popularity_baseline:
        baseline_metrics = evaluate_popularity_baseline(
            data=data,
            top_k=args.top_k,
            candidate_k=args.candidate_k,
            max_users=args.max_users,
            use_serving_candidates=args.use_serving_candidates,
        )

    results = build_comparison_summary(
        ranking_metrics=ranking_metrics,
        baseline_metrics=baseline_metrics,
        top_k=args.top_k,
    )
    print_evaluation_report(results, top_k=args.top_k)

    if args.output_json is not None:
        report = build_experiment_report(
            experiment_name=args.experiment_name,
            stage="recommendation",
            data_path=args.data,
            metrics=results,
            config={
                "top_k": args.top_k,
                "candidate_k": args.candidate_k,
                "max_users": args.max_users,
                "use_serving_candidates": args.use_serving_candidates,
                "skip_popularity_baseline": args.skip_popularity_baseline,
                "enable_reranking": enable_reranking,
                "enable_diversity": enable_reranking and enable_diversity,
                "enable_exploration": enable_reranking and enable_exploration,
                "enable_freshness": enable_reranking and enable_freshness,
                "seed": args.seed,
                "split_name": args.split_name,
            },
        )
        output_path = write_json_report(args.output_json, report)
        print(f"\nSaved JSON metrics to {output_path}")


if __name__ == "__main__":
    main()
