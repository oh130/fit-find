"""Diagnostics CLI for recommendation candidate, ranking, and feature contracts."""

from __future__ import annotations

import argparse
import inspect
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from rec_models.candidate.evaluator import DEFAULT_TOP_K as DEFAULT_CANDIDATE_TOP_K
    from rec_models.candidate.infer import DEFAULT_CHECKPOINT_DIR as DEFAULT_CANDIDATE_CHECKPOINT_DIR
    from rec_models.candidate.infer import retrieve_candidates_for_users
    from rec_models.common.metrics import hit_rate_at_k, mean_metric, ndcg_at_k, safe_roc_auc_score
    from rec_models.common.utils import build_experiment_report, write_json_report
    from rec_models.evaluation.data_utils import build_evaluation_context, build_session_context, load_evaluation_data
    from rec_models.ranking.infer import (
        DEFAULT_CHECKPOINT_DIR as DEFAULT_RANKING_CHECKPOINT_DIR,
        _extract_deepfm_scores,
        _extract_scores,
        load_artifacts,
        load_deepfm_artifacts,
        prepare_inference_features,
    )
    from rec_models.ranking.train import enrich_with_item_features
    from rec_models.serving import candidate_service as serving_candidate_service
    from rec_models.serving.candidate_service import CandidateUserProfileStore
    from rec_models.serving.candidate_service import (
        DEFAULT_COPURCHASE_ARTIFACT_PATH,
        DEFAULT_SEQUENTIAL_ARTIFACT_PATH,
        generate_candidates,
        load_article_catalog,
        load_copurchase_neighbor_store,
        load_sequential_transition_store,
    )
    from rec_models.serving.ranking_service import build_batch_ranking_features, load_customer_features
except ImportError:  # pragma: no cover - supports running from rec_models/ as cwd
    from candidate.evaluator import DEFAULT_TOP_K as DEFAULT_CANDIDATE_TOP_K  # type: ignore[no-redef]
    from candidate.infer import DEFAULT_CHECKPOINT_DIR as DEFAULT_CANDIDATE_CHECKPOINT_DIR  # type: ignore[no-redef]
    from candidate.infer import retrieve_candidates_for_users  # type: ignore[no-redef]
    from common.metrics import hit_rate_at_k, mean_metric, ndcg_at_k, safe_roc_auc_score  # type: ignore[no-redef]
    from common.utils import build_experiment_report, write_json_report  # type: ignore[no-redef]
    from evaluation.data_utils import build_evaluation_context, build_session_context, load_evaluation_data  # type: ignore[no-redef]
    from ranking.infer import (  # type: ignore[no-redef]
        DEFAULT_CHECKPOINT_DIR as DEFAULT_RANKING_CHECKPOINT_DIR,
        _extract_deepfm_scores,
        _extract_scores,
        load_artifacts,
        load_deepfm_artifacts,
        prepare_inference_features,
    )
    from ranking.train import enrich_with_item_features  # type: ignore[no-redef]
    import serving.candidate_service as serving_candidate_service  # type: ignore[no-redef]
    from serving.candidate_service import CandidateUserProfileStore  # type: ignore[no-redef]
    from serving.candidate_service import (  # type: ignore[no-redef]
        DEFAULT_COPURCHASE_ARTIFACT_PATH,
        DEFAULT_SEQUENTIAL_ARTIFACT_PATH,
        generate_candidates,
        load_article_catalog,
        load_copurchase_neighbor_store,
        load_sequential_transition_store,
    )
    from serving.ranking_service import build_batch_ranking_features, load_customer_features  # type: ignore[no-redef]


DEFAULT_SAMPLE_SIZE = 50
DEFAULT_RANK_TOP_K = 10
DEFAULT_EVALUATION_DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "train_data_dev.csv"
DEFAULT_HISTORY_INTERACTIONS_PATH = Path("data/processed/candidate_interactions_dev.csv.gz")


@dataclass(frozen=True)
class CatalogSegmentLookup:
    article_to_category: dict[str, str]
    article_to_popularity_bucket: dict[str, str]
    article_to_is_new_item: dict[str, bool]
    article_to_category_bucket: dict[str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run recommendation diagnostics.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    candidate_parser = subparsers.add_parser("candidate", help="Diagnose baseline vs Two-Tower candidate gaps.")
    candidate_parser.add_argument("--data", type=Path, default=DEFAULT_EVALUATION_DATA_PATH, help="Path to processed recommendation data.")
    candidate_parser.add_argument("--top-k", type=int, default=DEFAULT_CANDIDATE_TOP_K, help="Candidate recall cutoff.")
    candidate_parser.add_argument("--max-users", type=int, help="Optional cap for faster diagnostics.")
    candidate_parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE, help="User sample size for detailed gap analysis.")
    candidate_parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=DEFAULT_CANDIDATE_CHECKPOINT_DIR,
        help="Checkpoint directory containing Two-Tower artifacts.",
    )
    candidate_parser.add_argument("--output-dir", type=Path, required=True, help="Directory for diagnostic outputs.")
    candidate_parser.add_argument("--experiment-name", type=str, default="candidate_diagnostic", help="Stable experiment name.")
    candidate_parser.add_argument(
        "--history-aware",
        action="store_true",
        help="Evaluate profile boosts using only prior user purchases from candidate_interactions data.",
    )
    candidate_parser.add_argument(
        "--history-aware-two-tower-comparison",
        action="store_true",
        help="When --history-aware is set, also evaluate sequential_combined with Two-Tower hybrid retrieval enabled.",
    )

    ranking_parser = subparsers.add_parser("ranking", help="Diagnose logreg vs DeepFM ranking gaps.")
    ranking_parser.add_argument("--data", type=Path, default=DEFAULT_EVALUATION_DATA_PATH, help="Path to processed recommendation data.")
    ranking_parser.add_argument("--top-k", type=int, default=DEFAULT_RANK_TOP_K, help="Top-K cutoff for difference sampling.")
    ranking_parser.add_argument("--max-users", type=int, help="Optional cap for faster diagnostics.")
    ranking_parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE, help="User sample size for detailed ranking differences.")
    ranking_parser.add_argument(
        "--logreg-checkpoint-dir",
        type=Path,
        default=DEFAULT_RANKING_CHECKPOINT_DIR,
        help="Checkpoint directory containing logistic ranking artifacts.",
    )
    ranking_parser.add_argument(
        "--deepfm-checkpoint-dir",
        type=Path,
        default=DEFAULT_RANKING_CHECKPOINT_DIR,
        help="Checkpoint directory containing DeepFM ranking artifacts.",
    )
    ranking_parser.add_argument("--output-dir", type=Path, required=True, help="Directory for diagnostic outputs.")
    ranking_parser.add_argument("--experiment-name", type=str, default="ranking_diagnostic", help="Stable experiment name.")

    contract_parser = subparsers.add_parser("contract", help="Diagnose ranking train/serving feature contracts.")
    contract_parser.add_argument("--data", type=Path, default=DEFAULT_EVALUATION_DATA_PATH, help="Path to processed recommendation data.")
    contract_parser.add_argument(
        "--logreg-checkpoint-dir",
        type=Path,
        default=DEFAULT_RANKING_CHECKPOINT_DIR,
        help="Checkpoint directory containing logistic ranking metadata.",
    )
    contract_parser.add_argument(
        "--deepfm-checkpoint-dir",
        type=Path,
        default=DEFAULT_RANKING_CHECKPOINT_DIR,
        help="Checkpoint directory containing DeepFM ranking metadata.",
    )
    contract_parser.add_argument("--output-dir", type=Path, required=True, help="Directory for diagnostic outputs.")
    contract_parser.add_argument("--experiment-name", type=str, default="contract_diagnostic", help="Stable experiment name.")
    return parser.parse_args()


def _stable_sample(values: list[str], sample_size: int) -> list[str]:
    if sample_size <= 0 or len(values) <= sample_size:
        return values
    return sorted(values)[:sample_size]


def _resolve_output_dir(path: Path) -> Path:
    output_dir = path.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _register_legacy_joblib_symbols() -> None:
    compatibility_function = lambda frame: frame.astype("float64")
    for module_name in ("__main__", __name__):
        module = sys.modules.get(module_name)
        if module is not None and not hasattr(module, "cast_numeric_features_to_float"):
            setattr(module, "cast_numeric_features_to_float", compatibility_function)


def _load_catalog_segments() -> CatalogSegmentLookup:
    catalog = load_article_catalog().copy()
    catalog["article_id"] = catalog["article_id"].astype(str)
    catalog["category"] = catalog.get("category", pd.Series("UNKNOWN", index=catalog.index)).fillna("UNKNOWN").astype(str)
    catalog["popularity"] = pd.to_numeric(catalog.get("popularity"), errors="coerce").fillna(0.0)
    catalog["is_new_item"] = catalog.get("is_new_item", pd.Series(False, index=catalog.index)).fillna(False).astype(bool)

    popularity_bucket = pd.qcut(
        catalog["popularity"].rank(method="first"),
        q=min(3, max(1, catalog["article_id"].nunique())),
        labels=["tail", "mid", "head"][: min(3, max(1, catalog["article_id"].nunique()))],
        duplicates="drop",
    )
    popularity_bucket = popularity_bucket.astype(str).replace("nan", "tail")

    category_counts = catalog.groupby("category", sort=False)["article_id"].transform("count")
    category_bucket = pd.qcut(
        category_counts.rank(method="first"),
        q=min(3, max(1, catalog["category"].nunique())),
        labels=["rare", "mid", "common"][: min(3, max(1, catalog["category"].nunique()))],
        duplicates="drop",
    )
    category_bucket = category_bucket.astype(str).replace("nan", "rare")

    return CatalogSegmentLookup(
        article_to_category=dict(zip(catalog["article_id"], catalog["category"], strict=False)),
        article_to_popularity_bucket=dict(zip(catalog["article_id"], popularity_bucket, strict=False)),
        article_to_is_new_item=dict(zip(catalog["article_id"], catalog["is_new_item"], strict=False)),
        article_to_category_bucket=dict(zip(catalog["article_id"], category_bucket, strict=False)),
    )


def _is_cold_start_user(user_id: str, customer_features: pd.DataFrame) -> bool:
    if customer_features.empty or "customer_id" not in customer_features.columns:
        return True
    return user_id not in set(customer_features["customer_id"].astype(str))


def run_candidate_diagnostic(args: argparse.Namespace) -> None:
    if args.history_aware:
        run_candidate_history_aware_diagnostic(args)
        return

    output_dir = _resolve_output_dir(args.output_dir)
    data = load_evaluation_data(args.data)
    context = build_evaluation_context(data, max_users=args.max_users)
    catalog_segments = _load_catalog_segments()
    customer_features = load_customer_features().reset_index(drop=True)

    baseline_predictions: dict[str, list[str]] = {}
    baseline_hits: dict[str, bool] = {}
    sampled_user_rows = data.loc[data["customer_id"].isin(set(context.sampled_user_ids))].copy()
    two_tower_predictions = retrieve_candidates_for_users(
        user_rows=sampled_user_rows,
        items=data,
        checkpoint_dir=args.checkpoint_dir,
        top_k=args.top_k,
        exclude_seen_items=False,
    )

    per_user_rows: list[dict[str, Any]] = []
    evaluated_user_ids: list[str] = []
    baseline_ranked_lists: list[list[str]] = []
    two_tower_ranked_lists: list[list[str]] = []
    relevant_lists: list[list[str]] = []

    original_two_tower_retrieval = serving_candidate_service._retrieve_two_tower_candidates
    serving_candidate_service._retrieve_two_tower_candidates = lambda user_id, top_k, recent_click_set: []
    try:
        for user_id in context.sampled_user_ids:
            user_rows = context.user_rows_by_id.get(user_id)
            if user_rows is None or user_rows.empty:
                continue

            session_context = build_session_context(user_rows)
            baseline_candidates = generate_candidates(
                user_id=user_id,
                top_k=args.top_k,
                recent_clicks=session_context.get("recent_clicks"),
                session_interest=session_context.get("session_interest"),
            )
            baseline_items = baseline_candidates["article_id"].astype(str).head(args.top_k).tolist()
            two_tower_items = [str(item["article_id"]) for item in two_tower_predictions.get(user_id, [])[: args.top_k]]
            relevant_items = context.ground_truth_by_user[user_id]

            baseline_predictions[user_id] = baseline_items
            baseline_hits[user_id] = bool(set(baseline_items) & set(relevant_items))
            two_tower_hit = bool(set(two_tower_items) & set(relevant_items))

            evaluated_user_ids.append(user_id)
            baseline_ranked_lists.append(baseline_items)
            two_tower_ranked_lists.append(two_tower_items)
            relevant_lists.append(relevant_items)

            ground_truth_article = next((item for item in relevant_items if item in set(baseline_items) | set(two_tower_items)), relevant_items[0] if relevant_items else None)
            ground_truth_category = catalog_segments.article_to_category.get(str(ground_truth_article), "UNKNOWN") if ground_truth_article else "UNKNOWN"
            popularity_bucket = catalog_segments.article_to_popularity_bucket.get(str(ground_truth_article), "tail") if ground_truth_article else "tail"
            per_user_rows.append(
                {
                    "user_id": user_id,
                    "baseline_hit": baseline_hits[user_id],
                    "two_tower_hit": two_tower_hit,
                    "is_cold_start": _is_cold_start_user(user_id, customer_features),
                    "has_recent_clicks": bool(session_context.get("recent_clicks")),
                    "has_session_interest": bool(session_context.get("session_interest")),
                    "ground_truth_article_id": ground_truth_article,
                    "ground_truth_category": ground_truth_category,
                    "ground_truth_popularity_bucket": popularity_bucket,
                    "ground_truth_is_long_tail": popularity_bucket == "tail",
                    "two_tower_ground_truth_rank": _rank_of_first_match(two_tower_items, relevant_items),
                    "baseline_ground_truth_rank": _rank_of_first_match(baseline_items, relevant_items),
                }
            )
    finally:
        serving_candidate_service._retrieve_two_tower_candidates = original_two_tower_retrieval

    user_frame = pd.DataFrame(per_user_rows)
    gap_frame = user_frame.loc[user_frame["baseline_hit"] & ~user_frame["two_tower_hit"]].copy()
    sampled_gap_frame = gap_frame.loc[gap_frame["user_id"].isin(_stable_sample(gap_frame["user_id"].astype(str).tolist(), args.sample_size))].copy()

    category_rows: list[dict[str, Any]] = []
    for category, category_user_frame in user_frame.groupby("ground_truth_category", sort=True):
        user_ids = category_user_frame["user_id"].astype(str).tolist()
        indices = [evaluated_user_ids.index(user_id) for user_id in user_ids if user_id in evaluated_user_ids]
        baseline_lists = [baseline_ranked_lists[index] for index in indices]
        two_tower_lists = [two_tower_ranked_lists[index] for index in indices]
        category_relevant = [relevant_lists[index] for index in indices]
        category_rows.append(
            {
                "category": category,
                "users_evaluated": len(indices),
                f"baseline_Recall@{args.top_k}": mean_metric(baseline_lists, category_relevant, _recall_at_k, args.top_k),
                f"two_tower_Recall@{args.top_k}": mean_metric(two_tower_lists, category_relevant, _recall_at_k, args.top_k),
            }
        )
    category_frame = pd.DataFrame(category_rows).sort_values("users_evaluated", ascending=False)

    sampled_gap_frame.to_csv(output_dir / "candidate_gap_users.csv", index=False)
    category_frame.to_csv(output_dir / "candidate_category_recall.csv", index=False)

    summary = {
        "users_evaluated": int(len(user_frame)),
        f"baseline_Recall@{args.top_k}": mean_metric(baseline_ranked_lists, relevant_lists, _recall_at_k, args.top_k),
        f"two_tower_Recall@{args.top_k}": mean_metric(two_tower_ranked_lists, relevant_lists, _recall_at_k, args.top_k),
        "baseline_hit_two_tower_miss_users": int(len(gap_frame)),
        "gap_user_sample_size": int(len(sampled_gap_frame)),
        "gap_breakdown": {
            "cold_start": int(sampled_gap_frame.get("is_cold_start", pd.Series(dtype=bool)).sum()),
            "has_recent_clicks": int(sampled_gap_frame.get("has_recent_clicks", pd.Series(dtype=bool)).sum()),
            "has_session_interest": int(sampled_gap_frame.get("has_session_interest", pd.Series(dtype=bool)).sum()),
            "long_tail_ground_truth": int(sampled_gap_frame.get("ground_truth_is_long_tail", pd.Series(dtype=bool)).sum()),
        },
        "artifacts": {
            "gap_users_csv": str((output_dir / "candidate_gap_users.csv").resolve()),
            "category_recall_csv": str((output_dir / "candidate_category_recall.csv").resolve()),
        },
    }
    report = build_experiment_report(
        experiment_name=args.experiment_name,
        stage="candidate_diagnostic",
        data_path=args.data,
        metrics=summary,
        config={
            "top_k": args.top_k,
            "max_users": args.max_users,
            "sample_size": args.sample_size,
            "checkpoint_dir": str(args.checkpoint_dir.expanduser().resolve()),
        },
    )
    write_json_report(output_dir / "candidate_summary.json", report)


def _derive_profile_from_history(
    history_rows: pd.DataFrame,
    article_lookup: pd.DataFrame,
) -> dict[str, Any] | None:
    if history_rows.empty:
        return None

    merged = history_rows.copy()
    merged["main_category"] = merged["article_id"].map(article_lookup["main_category"])
    merged["color"] = merged["article_id"].map(article_lookup["color"])
    merged["garment_group_name"] = merged["article_id"].map(article_lookup["garment_group_name"])
    merged["main_category"] = merged["main_category"].fillna("UNKNOWN").astype(str)
    merged["color"] = merged["color"].fillna("UNKNOWN").astype(str)
    merged["garment_group_name"] = merged["garment_group_name"].fillna("UNKNOWN").astype(str)
    merged["price"] = pd.to_numeric(merged.get("price"), errors="coerce")
    avg_price = float(merged["price"].dropna().mean()) if merged["price"].notna().any() else float("nan")

    def _mode(column: str) -> str:
        counts = merged[column].value_counts()
        return str(counts.index[0]) if not counts.empty else "UNKNOWN"

    return {
        "customer_id": str(history_rows["customer_id"].iloc[0]),
        "preferred_main_category": _mode("main_category"),
        "preferred_colour_master": _mode("color"),
        "preferred_garment_group": _mode("garment_group_name"),
        "price_band": serving_candidate_service._derive_price_band(avg_price) if pd.notna(avg_price) else "UNKNOWN",
        "avg_price": avg_price,
    }


def _derive_recent_purchase_context(
    history_rows: pd.DataFrame,
    article_lookup: pd.DataFrame,
    recent_limit: int = 3,
) -> tuple[list[str], dict[str, float]]:
    if history_rows.empty:
        return [], {}

    ordered = history_rows.sort_values(["t_dat", "article_id"]).copy()
    recent_rows = ordered.tail(recent_limit)
    recent_clicks = recent_rows["article_id"].astype(str).tolist()

    category_series = recent_rows["article_id"].map(article_lookup["category"]).fillna("UNKNOWN").astype(str)
    session_interest: dict[str, float] = {}
    counts = category_series.value_counts()
    total = max(int(counts.sum()), 1)
    for category, count in counts.items():
        if category == "UNKNOWN":
            continue
        session_interest[str(category)] = float(count) / float(total)
    return recent_clicks, session_interest


def _history_aware_copurchase_ablation_configs() -> list[dict[str, Any]]:
    return [
        {
            "name": "baseline_no_profile",
            "mode": "baseline_shared",
            "include_copurchase": False,
        },
        {
            "name": "append_only",
            "mode": "append_only",
            "include_copurchase": True,
            "copurchase_score_weight": 6.0,
            "copurchase_neighbor_limit": 40,
            "copurchase_min_cooccurrence_count": 1,
        },
        {
            "name": "weight_0.5",
            "mode": "copurchase",
            "include_copurchase": True,
            "copurchase_score_weight": 0.5,
            "copurchase_neighbor_limit": 40,
            "copurchase_min_cooccurrence_count": 1,
        },
        {
            "name": "weight_1.0",
            "mode": "copurchase",
            "include_copurchase": True,
            "copurchase_score_weight": 1.0,
            "copurchase_neighbor_limit": 40,
            "copurchase_min_cooccurrence_count": 1,
        },
        {
            "name": "weight_2.0",
            "mode": "copurchase",
            "include_copurchase": True,
            "copurchase_score_weight": 2.0,
            "copurchase_neighbor_limit": 40,
            "copurchase_min_cooccurrence_count": 1,
        },
        {
            "name": "weight_4.0",
            "mode": "copurchase",
            "include_copurchase": True,
            "copurchase_score_weight": 4.0,
            "copurchase_neighbor_limit": 40,
            "copurchase_min_cooccurrence_count": 1,
        },
        {
            "name": "cooccurrence_gte_2",
            "mode": "copurchase",
            "include_copurchase": True,
            "copurchase_score_weight": 6.0,
            "copurchase_neighbor_limit": 40,
            "copurchase_min_cooccurrence_count": 2,
        },
        {
            "name": "cooccurrence_gte_3",
            "mode": "copurchase",
            "include_copurchase": True,
            "copurchase_score_weight": 6.0,
            "copurchase_neighbor_limit": 40,
            "copurchase_min_cooccurrence_count": 3,
        },
        {
            "name": "top_neighbors_20",
            "mode": "copurchase",
            "include_copurchase": True,
            "copurchase_score_weight": 6.0,
            "copurchase_neighbor_limit": 20,
            "copurchase_min_cooccurrence_count": 1,
        },
        {
            "name": "top_neighbors_50",
            "mode": "copurchase",
            "include_copurchase": True,
            "copurchase_score_weight": 6.0,
            "copurchase_neighbor_limit": 50,
            "copurchase_min_cooccurrence_count": 1,
        },
        {
            "name": "top_neighbors_100",
            "mode": "copurchase",
            "include_copurchase": True,
            "copurchase_score_weight": 6.0,
            "copurchase_neighbor_limit": 100,
            "copurchase_min_cooccurrence_count": 1,
        },
    ]


def _build_append_only_items(
    base_items: list[str],
    copurchase_items: list[str],
    top_k: int,
) -> list[str]:
    selected: list[str] = []
    seen: set[str] = set()
    for article_id in base_items:
        if article_id in seen:
            continue
        selected.append(article_id)
        seen.add(article_id)
        if len(selected) >= top_k:
            return selected
    for article_id in copurchase_items:
        if article_id in seen:
            continue
        selected.append(article_id)
        seen.add(article_id)
        if len(selected) >= top_k:
            break
    return selected


def run_candidate_history_aware_diagnostic(args: argparse.Namespace) -> None:
    output_dir = _resolve_output_dir(args.output_dir)
    start_time = time.monotonic()
    interactions_path = DEFAULT_HISTORY_INTERACTIONS_PATH
    interactions = pd.read_csv(interactions_path)
    interactions["customer_id"] = interactions["customer_id"].astype(str)
    interactions["article_id"] = interactions["article_id"].astype(str).str.zfill(10)
    interactions["t_dat"] = pd.to_datetime(interactions["t_dat"])
    interactions = interactions.sort_values(["customer_id", "t_dat", "article_id"]).reset_index(drop=True)

    feature_store = serving_candidate_service.get_cached_feature_store()
    article_catalog = feature_store.catalog.copy()
    article_catalog["article_id"] = article_catalog["article_id"].astype(str)
    article_lookup = (
        article_catalog.loc[:, ["article_id", "category", "main_category", "color", "garment_group_name"]]
        .drop_duplicates("article_id")
        .set_index("article_id")
    )

    evaluable_rows: list[dict[str, Any]] = []
    profile_map: dict[str, dict[str, Any]] = {}
    recent_clicks_by_user: dict[str, list[str]] = {}
    session_interest_by_user: dict[str, dict[str, float]] = {}
    for user_id, user_rows in interactions.groupby("customer_id", sort=False):
        if len(user_rows) < 2:
            continue
        history_rows = user_rows.iloc[:-1].copy()
        target_row = user_rows.iloc[-1]
        profile = _derive_profile_from_history(history_rows=history_rows, article_lookup=article_lookup)
        if profile is None:
            continue
        profile_map[user_id] = profile
        recent_clicks, session_interest = _derive_recent_purchase_context(
            history_rows=history_rows,
            article_lookup=article_lookup,
        )
        recent_clicks_by_user[user_id] = recent_clicks
        session_interest_by_user[user_id] = session_interest
        evaluable_rows.append(
            {
                "user_id": user_id,
                "target_article_id": str(target_row["article_id"]),
                "history_count": int(len(history_rows)),
            }
        )
        if args.max_users is not None and len(evaluable_rows) >= args.max_users:
            break

    evaluable_frame = pd.DataFrame(evaluable_rows)

    user_ids = evaluable_frame["user_id"].astype(str).tolist()
    targets_by_user = {
        str(row["user_id"]): [str(row["target_article_id"])]
        for row in evaluable_frame.to_dict(orient="records")
    }

    no_profile_store = CandidateUserProfileStore(user_profiles={})
    shared_no_profile_candidates = generate_candidates(
        user_id="__history_aware_smoke__",
        top_k=args.top_k,
        recent_clicks=[],
        session_interest={},
        candidate_pool_size=args.top_k,
        feature_store=feature_store,
        user_profile_store=no_profile_store,
        include_two_tower=False,
        include_sequential=False,
        include_copurchase=False,
    )
    shared_no_profile_items = shared_no_profile_candidates["article_id"].astype(str).head(args.top_k).tolist()

    copurchase_store = load_copurchase_neighbor_store(DEFAULT_COPURCHASE_ARTIFACT_PATH)
    sequential_store = load_sequential_transition_store(DEFAULT_SEQUENTIAL_ARTIFACT_PATH)
    copurchase_metadata = dict(copurchase_store.metadata)
    sequential_metadata = dict(sequential_store.metadata)
    print(
        "[history-aware] copurchase_artifact="
        f"{copurchase_metadata.get('artifact_path', str(DEFAULT_COPURCHASE_ARTIFACT_PATH))} "
        f"available={copurchase_metadata.get('available', bool(copurchase_store.neighbors_by_article))} "
        f"leakage_safe_for_last_item_eval={copurchase_metadata.get('leakage_safe_for_last_item_eval', False)}",
        flush=True,
    )
    print(
        "[history-aware] sequential_artifact="
        f"{sequential_metadata.get('artifact_path', str(DEFAULT_SEQUENTIAL_ARTIFACT_PATH))} "
        f"available={sequential_metadata.get('available', bool(sequential_store.article_neighbors_by_article))} "
        f"leakage_safe_for_last_item_eval={sequential_metadata.get('leakage_safe_for_last_item_eval', False)}",
        flush=True,
    )

    experiment_configs = [
        {
            "name": "baseline_no_profile",
            "kwargs": {
                "mode": "shared",
            },
        },
        {
            "name": "baseline_profile",
            "kwargs": {
                "include_sequential": False,
                "include_copurchase": False,
                "user_profile_override": "profile",
            },
        },
        {
            "name": "copurchase_candidate",
            "kwargs": {
                "include_sequential": False,
                "include_copurchase": True,
                "copurchase_store": copurchase_store,
            },
        },
        {
            "name": "sequential_article",
            "kwargs": {
                "include_sequential": True,
                "include_copurchase": False,
                "sequential_store": sequential_store,
                "sequential_category_limit": 0,
            },
        },
        {
            "name": "sequential_category",
            "kwargs": {
                "include_sequential": True,
                "include_copurchase": False,
                "sequential_store": sequential_store,
                "sequential_article_limit": 0,
            },
        },
        {
            "name": "sequential_combined",
            "kwargs": {
                "include_sequential": True,
                "include_copurchase": False,
                "sequential_store": sequential_store,
            },
        },
        {
            "name": "copurchase_sequential_combined",
            "kwargs": {
                "include_sequential": True,
                "include_copurchase": True,
                "copurchase_store": copurchase_store,
                "sequential_store": sequential_store,
            },
        },
        {
            "name": "profile_copurchase_sequential_combined",
            "kwargs": {
                "include_sequential": True,
                "include_copurchase": True,
                "copurchase_store": copurchase_store,
                "sequential_store": sequential_store,
                "user_profile_override": "profile",
            },
        },
    ]
    if args.history_aware_two_tower_comparison:
        experiment_configs.append(
            {
                "name": "sequential_combined_two_tower_on",
                "kwargs": {
                    "include_two_tower": True,
                    "include_sequential": True,
                    "include_copurchase": False,
                    "sequential_store": sequential_store,
                },
            }
        )
        experiment_configs.append(
            {
                "name": "copurchase_sequential_two_tower_on",
                "kwargs": {
                    "include_two_tower": True,
                    "include_sequential": True,
                    "include_copurchase": True,
                    "copurchase_store": copurchase_store,
                    "sequential_store": sequential_store,
                },
            }
        )

    relevant_lists = [targets_by_user[user_id] for user_id in user_ids]
    experiment_rows: list[dict[str, Any]] = []
    total_users = len(user_ids)

    for experiment in experiment_configs:
        name = str(experiment["name"])
        config_start = time.monotonic()
        ranked_lists: list[list[str]] = []
        hits = 0
        print(f"[history-aware][{name}] starting", flush=True)
        for index, user_id in enumerate(user_ids, start=1):
            relevant_items = targets_by_user[user_id]
            if experiment["kwargs"].get("mode") == "shared":
                ranked_items = shared_no_profile_items
            else:
                kwargs = dict(experiment["kwargs"])
                user_profile_override = profile_map.get(user_id) if kwargs.pop("user_profile_override", None) == "profile" else None
                candidates = generate_candidates(
                    user_id=user_id,
                    top_k=args.top_k,
                    recent_clicks=recent_clicks_by_user.get(user_id, []),
                    session_interest=session_interest_by_user.get(user_id, {}),
                    candidate_pool_size=args.top_k,
                    feature_store=feature_store,
                    user_profile_store=no_profile_store,
                    user_profile_override=user_profile_override,
                    include_two_tower=bool(kwargs.pop("include_two_tower", False)),
                    **kwargs,
                )
                ranked_items = candidates["article_id"].astype(str).head(args.top_k).tolist()

            ranked_lists.append(ranked_items)
            hits += int(bool(set(ranked_items) & set(relevant_items)))
            elapsed = time.monotonic() - config_start
            print(
                f"[history-aware][{name}] processed_users={index}/{total_users} "
                f"elapsed_seconds={elapsed:.2f} current_Recall@{args.top_k}={hits / index:.6f}",
                flush=True,
            )

        experiment_rows.append(
            {
                "setting": name,
                "users_evaluated": int(len(user_ids)),
                f"Recall@{args.top_k}": mean_metric(ranked_lists, relevant_lists, _recall_at_k, args.top_k),
                "elapsed_seconds": float(time.monotonic() - config_start),
            }
        )

    metrics_by_name = {
        row["setting"]: row[f"Recall@{args.top_k}"]
        for row in experiment_rows
    }
    summary = {
        "users_evaluated": int(len(user_ids)),
        f"baseline_no_profile_Recall@{args.top_k}": metrics_by_name["baseline_no_profile"],
        f"baseline_profile_Recall@{args.top_k}": metrics_by_name["baseline_profile"],
        f"copurchase_candidate_Recall@{args.top_k}": metrics_by_name["copurchase_candidate"],
        f"sequential_article_Recall@{args.top_k}": metrics_by_name["sequential_article"],
        f"sequential_category_Recall@{args.top_k}": metrics_by_name["sequential_category"],
        f"sequential_combined_Recall@{args.top_k}": metrics_by_name["sequential_combined"],
        "average_history_count": float(evaluable_frame["history_count"].mean()) if not evaluable_frame.empty else 0.0,
        "elapsed_seconds": float(time.monotonic() - start_time),
        "copurchase_artifact": copurchase_metadata,
        "sequential_artifact": sequential_metadata,
        "experiment_results": experiment_rows,
    }
    if args.history_aware_two_tower_comparison:
        summary[f"sequential_combined_two_tower_on_Recall@{args.top_k}"] = metrics_by_name[
            "sequential_combined_two_tower_on"
        ]
    report = build_experiment_report(
        experiment_name=f"{args.experiment_name}_history_aware",
        stage="candidate_diagnostic_history_aware",
        data_path=interactions_path,
        metrics=summary,
        config={
            "top_k": args.top_k,
            "max_users": args.max_users,
            "sample_size": args.sample_size,
            "history_aware": True,
            "history_aware_two_tower_comparison": bool(args.history_aware_two_tower_comparison),
        },
    )
    write_json_report(output_dir / "candidate_summary.json", report)


def _rank_of_first_match(ranked_items: list[str], relevant_items: list[str]) -> int | None:
    relevant_set = set(relevant_items)
    for index, article_id in enumerate(ranked_items, start=1):
        if article_id in relevant_set:
            return index
    return None


def _recall_at_k(ranked_items: list[str], relevant_items: list[str], k: int) -> float:
    ranked_top_k = ranked_items[:k]
    relevant_set = set(relevant_items)
    if not relevant_set:
        return 0.0
    return float(len([item for item in ranked_top_k if item in relevant_set]) / len(relevant_set))


def run_ranking_diagnostic(args: argparse.Namespace) -> None:
    output_dir = _resolve_output_dir(args.output_dir)
    _register_legacy_joblib_symbols()
    data = load_evaluation_data(args.data)
    context = build_evaluation_context(data, max_users=args.max_users)
    catalog_segments = _load_catalog_segments()
    enriched = enrich_with_item_features(data).copy()
    enriched["customer_id"] = enriched["customer_id"].astype(str)
    enriched["article_id"] = enriched["article_id"].astype(str)
    enriched = enriched.loc[enriched["customer_id"].isin(set(context.sampled_user_ids))].copy()

    logreg_model, logreg_metadata = load_artifacts(args.logreg_checkpoint_dir)
    deepfm_model, deepfm_metadata = load_deepfm_artifacts(args.deepfm_checkpoint_dir)
    logreg_features = prepare_inference_features(enriched, logreg_metadata["feature_columns"])
    deepfm_features = prepare_inference_features(enriched, deepfm_metadata["feature_columns"])
    enriched["score_logreg"] = _extract_scores(logreg_model, logreg_features)
    enriched["score_deepfm"] = _extract_deepfm_scores(deepfm_model, deepfm_features, deepfm_metadata)

    enriched["is_positive"] = context.positive_mask.loc[enriched.index].astype(bool).to_numpy()
    enriched["category"] = enriched["article_id"].map(catalog_segments.article_to_category).fillna(enriched.get("category", "UNKNOWN")).astype(str)
    enriched["popularity_bucket"] = enriched["article_id"].map(catalog_segments.article_to_popularity_bucket).fillna("tail")
    enriched["is_new_item"] = enriched["article_id"].map(catalog_segments.article_to_is_new_item).fillna(False).astype(bool)
    enriched["category_bucket"] = enriched["article_id"].map(catalog_segments.article_to_category_bucket).fillna("rare")

    per_user_rows: list[dict[str, Any]] = []
    evaluated_user_ids: list[str] = []
    logreg_ranked_lists: list[list[str]] = []
    deepfm_ranked_lists: list[list[str]] = []
    relevant_lists: list[list[str]] = []

    for user_id in context.sampled_user_ids:
        user_rows = enriched.loc[enriched["customer_id"].eq(user_id)].copy()
        if user_rows.empty:
            continue

        logreg_ranked = user_rows.sort_values(["score_logreg", "article_id"], ascending=[False, True])["article_id"].astype(str).tolist()
        deepfm_ranked = user_rows.sort_values(["score_deepfm", "article_id"], ascending=[False, True])["article_id"].astype(str).tolist()
        relevant_items = context.ground_truth_by_user[user_id]
        ground_truth_article = relevant_items[0] if relevant_items else None

        evaluated_user_ids.append(user_id)
        per_user_rows.append(
            {
                "user_id": user_id,
                "logreg_hit_topk": bool(set(logreg_ranked[: args.top_k]) & set(relevant_items)),
                "deepfm_hit_topk": bool(set(deepfm_ranked[: args.top_k]) & set(relevant_items)),
                "ground_truth_article_id": ground_truth_article,
                "ground_truth_rank_logreg": _rank_of_first_match(logreg_ranked, relevant_items),
                "ground_truth_rank_deepfm": _rank_of_first_match(deepfm_ranked, relevant_items),
                "ground_truth_rank_delta": _rank_delta(logreg_ranked, deepfm_ranked, relevant_items),
                "ground_truth_popularity_bucket": catalog_segments.article_to_popularity_bucket.get(str(ground_truth_article), "tail") if ground_truth_article else "tail",
                "ground_truth_is_new_item": bool(catalog_segments.article_to_is_new_item.get(str(ground_truth_article), False)) if ground_truth_article else False,
                "ground_truth_category_bucket": catalog_segments.article_to_category_bucket.get(str(ground_truth_article), "rare") if ground_truth_article else "rare",
                "top10_logreg": ",".join(logreg_ranked[: args.top_k]),
                "top10_deepfm": ",".join(deepfm_ranked[: args.top_k]),
                "top10_overlap_count": len(set(logreg_ranked[: args.top_k]) & set(deepfm_ranked[: args.top_k])),
            }
        )
        logreg_ranked_lists.append(logreg_ranked)
        deepfm_ranked_lists.append(deepfm_ranked)
        relevant_lists.append(relevant_items)

    user_frame = pd.DataFrame(per_user_rows)
    diff_frame = user_frame.loc[user_frame["logreg_hit_topk"] & ~user_frame["deepfm_hit_topk"]].copy()
    sampled_diff_frame = diff_frame.loc[diff_frame["user_id"].isin(_stable_sample(diff_frame["user_id"].astype(str).tolist(), args.sample_size))].copy()

    segment_summary_rows: list[dict[str, Any]] = []
    for segment_name, selector in (
        ("popular_item", user_frame["ground_truth_popularity_bucket"].eq("head")),
        ("new_item", user_frame["ground_truth_is_new_item"].fillna(False)),
        ("rare_category", user_frame["ground_truth_category_bucket"].eq("rare")),
    ):
        segment_user_ids = set(user_frame.loc[selector, "user_id"].astype(str))
        if not segment_user_ids:
            continue
        segment_rows = enriched.loc[enriched["customer_id"].isin(segment_user_ids)].copy()
        logreg_lists, deepfm_lists, segment_relevant = _subset_rankings_by_users(
            target_user_ids=segment_user_ids,
            evaluated_user_ids=evaluated_user_ids,
            logreg_ranked_lists=logreg_ranked_lists,
            deepfm_ranked_lists=deepfm_ranked_lists,
            relevant_lists=relevant_lists,
        )
        segment_summary_rows.extend(
            [
                _build_segment_metric_row(segment_name, "logreg", segment_rows, "score_logreg", logreg_lists, segment_relevant, args.top_k),
                _build_segment_metric_row(segment_name, "deepfm", segment_rows, "score_deepfm", deepfm_lists, segment_relevant, args.top_k),
            ]
        )

    sampled_diff_frame.to_csv(output_dir / "ranking_topk_differences.csv", index=False)
    pd.DataFrame(segment_summary_rows).to_csv(output_dir / "ranking_segment_metrics.csv", index=False)

    summary = {
        "users_evaluated": int(len(user_frame)),
        "logreg_hit_topk_users": int(user_frame.get("logreg_hit_topk", pd.Series(dtype=bool)).sum()),
        "deepfm_hit_topk_users": int(user_frame.get("deepfm_hit_topk", pd.Series(dtype=bool)).sum()),
        "logreg_beats_deepfm_users": int(len(diff_frame)),
        "sampled_difference_users": int(len(sampled_diff_frame)),
        "average_ground_truth_rank_delta": _safe_mean(user_frame.get("ground_truth_rank_delta", pd.Series(dtype=float))),
        "artifacts": {
            "topk_differences_csv": str((output_dir / "ranking_topk_differences.csv").resolve()),
            "segment_metrics_csv": str((output_dir / "ranking_segment_metrics.csv").resolve()),
        },
    }
    report = build_experiment_report(
        experiment_name=args.experiment_name,
        stage="ranking_diagnostic",
        data_path=args.data,
        metrics=summary,
        config={
            "top_k": args.top_k,
            "max_users": args.max_users,
            "sample_size": args.sample_size,
            "logreg_checkpoint_dir": str(args.logreg_checkpoint_dir.expanduser().resolve()),
            "deepfm_checkpoint_dir": str(args.deepfm_checkpoint_dir.expanduser().resolve()),
        },
    )
    write_json_report(output_dir / "ranking_summary.json", report)


def _rank_delta(logreg_ranked: list[str], deepfm_ranked: list[str], relevant_items: list[str]) -> int | None:
    logreg_rank = _rank_of_first_match(logreg_ranked, relevant_items)
    deepfm_rank = _rank_of_first_match(deepfm_ranked, relevant_items)
    if logreg_rank is None or deepfm_rank is None:
        return None
    return int(deepfm_rank - logreg_rank)


def _subset_rankings_by_users(
    *,
    target_user_ids: set[str],
    evaluated_user_ids: list[str],
    logreg_ranked_lists: list[list[str]],
    deepfm_ranked_lists: list[list[str]],
    relevant_lists: list[list[str]],
) -> tuple[list[list[str]], list[list[str]], list[list[str]]]:
    logreg_lists: list[list[str]] = []
    deepfm_lists: list[list[str]] = []
    segment_relevant: list[list[str]] = []
    for index, user_id in enumerate(evaluated_user_ids):
        if user_id not in target_user_ids:
            continue
        logreg_lists.append(logreg_ranked_lists[index])
        deepfm_lists.append(deepfm_ranked_lists[index])
        segment_relevant.append(relevant_lists[index])
    return logreg_lists, deepfm_lists, segment_relevant


def _build_segment_metric_row(
    segment_name: str,
    model_name: str,
    segment_rows: pd.DataFrame,
    score_column: str,
    ranked_lists: list[list[str]],
    relevant_lists: list[list[str]],
    top_k: int,
) -> dict[str, Any]:
    return {
        "segment": segment_name,
        "model": model_name,
        "rows_evaluated": int(len(segment_rows)),
        "users_evaluated": int(len(ranked_lists)),
        "auc": safe_roc_auc_score(
            segment_rows["is_positive"].astype(int).tolist(),
            segment_rows[score_column].astype(float).tolist(),
        ),
        f"HitRate@{top_k}": mean_metric(ranked_lists, relevant_lists, hit_rate_at_k, top_k),
        f"NDCG@{top_k}": mean_metric(ranked_lists, relevant_lists, ndcg_at_k, top_k),
    }


def _safe_mean(series: pd.Series) -> float | None:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return None
    return float(numeric.mean())


def run_contract_diagnostic(args: argparse.Namespace) -> None:
    output_dir = _resolve_output_dir(args.output_dir)
    _register_legacy_joblib_symbols()
    raw_data = load_evaluation_data(args.data)
    enriched_data = enrich_with_item_features(raw_data)

    logreg_model, logreg_metadata = load_artifacts(args.logreg_checkpoint_dir)
    del logreg_model
    deepfm_model, deepfm_metadata = load_deepfm_artifacts(args.deepfm_checkpoint_dir)
    del deepfm_model

    candidate_rows = enriched_data.copy()
    if "customer_id" not in candidate_rows.columns:
        raise ValueError("Contract diagnostic requires customer_id in the input data.")
    serving_feature_frame = build_batch_ranking_features(candidate_rows)

    raw_columns = set(enriched_data.columns.astype(str).tolist())
    serving_columns = set(serving_feature_frame.columns.astype(str).tolist())
    logreg_columns = set(logreg_metadata.get("feature_columns", []))
    deepfm_columns = set(deepfm_metadata.get("feature_columns", []))

    comparison_rows = []
    for column in sorted(raw_columns | serving_columns | logreg_columns | deepfm_columns):
        comparison_rows.append(
            {
                "column": column,
                "present_in_training_data": column in raw_columns,
                "present_in_serving_features": column in serving_columns,
                "present_in_logreg_metadata": column in logreg_columns,
                "present_in_deepfm_metadata": column in deepfm_columns,
            }
        )
    comparison_frame = pd.DataFrame(comparison_rows)
    comparison_frame.to_csv(output_dir / "ranking_feature_contract.csv", index=False)

    try:
        from rec_models.serving import ranking_service
        from rec_models.ranking import train as ranking_train
    except ImportError:  # pragma: no cover
        import serving.ranking_service as ranking_service  # type: ignore[no-redef]
        import ranking.train as ranking_train  # type: ignore[no-redef]

    build_ranking_features_source = inspect.getsource(ranking_service.build_ranking_features)
    split_train_validation_source = inspect.getsource(ranking_train.split_train_validation)

    summary = {
        "training_feature_count": int(len(raw_columns)),
        "serving_feature_count": int(len(serving_columns)),
        "logreg_feature_count": int(len(logreg_columns)),
        "deepfm_feature_count": int(len(deepfm_columns)),
        "serving_only_features": sorted(serving_columns - raw_columns),
        "training_only_features": sorted(raw_columns - serving_columns),
        "logreg_only_features": sorted(logreg_columns - serving_columns),
        "deepfm_only_features": sorted(deepfm_columns - serving_columns),
        "session_feature_usage": {
            "session_columns_present_in_training": sorted([column for column in raw_columns if "session" in column or "recent_click" in column]),
            "session_columns_present_in_serving": sorted([column for column in serving_columns if "session" in column or "recent_click" in column]),
            "session_context_explicitly_dropped_in_serving": "del session_context" in build_ranking_features_source,
        },
        "split_diagnostic": {
            "ranking_split_uses_train_test_split": "train_test_split" in split_train_validation_source,
            "ranking_split_mentions_user_level": "user" in split_train_validation_source.lower(),
            "ranking_split_mentions_time": "time" in split_train_validation_source.lower(),
        },
        "artifacts": {
            "feature_contract_csv": str((output_dir / "ranking_feature_contract.csv").resolve()),
        },
    }
    report = build_experiment_report(
        experiment_name=args.experiment_name,
        stage="contract_diagnostic",
        data_path=args.data,
        metrics=summary,
        config={
            "logreg_checkpoint_dir": str(args.logreg_checkpoint_dir.expanduser().resolve()),
            "deepfm_checkpoint_dir": str(args.deepfm_checkpoint_dir.expanduser().resolve()),
        },
    )
    write_json_report(output_dir / "contract_summary.json", report)


def main() -> None:
    args = parse_args()
    if args.command == "candidate":
        run_candidate_diagnostic(args)
    elif args.command == "ranking":
        run_ranking_diagnostic(args)
    else:
        run_contract_diagnostic(args)


if __name__ == "__main__":
    main()
