"""Offline evaluator for the current candidate generation stage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from rec_models.common.metrics import mean_metric, recall_at_k
    from rec_models.evaluation.data_utils import (
        EvaluationContext,
        build_session_context,
        build_evaluation_context,
        load_evaluation_data,
    )
    from rec_models.candidate.infer import DEFAULT_CHECKPOINT_DIR, retrieve_candidates_for_users
    from rec_models.common.utils import build_experiment_report, write_json_report
    from rec_models.serving.candidate_service import generate_candidates
except ImportError:  # pragma: no cover - supports running from rec_models/ as cwd
    from common.metrics import mean_metric, recall_at_k  # type: ignore[no-redef]
    from evaluation.data_utils import (  # type: ignore[no-redef]
        EvaluationContext,
        build_session_context,
        build_evaluation_context,
        load_evaluation_data,
    )
    from candidate.infer import DEFAULT_CHECKPOINT_DIR, retrieve_candidates_for_users  # type: ignore[no-redef]
    from common.utils import build_experiment_report, write_json_report  # type: ignore[no-redef]
    from serving.candidate_service import generate_candidates  # type: ignore[no-redef]


DEFAULT_TOP_K = 300
DEFAULT_EVALUATION_DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "train_data_dev.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate candidate retrieval quality offline.")
    parser.add_argument("--data", type=Path, default=DEFAULT_EVALUATION_DATA_PATH, help="Path to processed recommendation data.")
    parser.add_argument("--top_k", type=int, default=DEFAULT_TOP_K, help="Retrieval cutoff K.")
    parser.add_argument("--candidate-pool-size", type=int, help="Optional explicit candidate pool size.")
    parser.add_argument("--max-users", type=int, help="Optional cap for smoke checks or faster iteration.")
    parser.add_argument(
        "--mode",
        choices=("baseline", "two-tower", "compare"),
        default="baseline",
        help="Evaluate heuristic baseline, trained Two-Tower, or both.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=DEFAULT_CHECKPOINT_DIR,
        help="Checkpoint directory containing two_tower.pt when mode uses Two-Tower.",
    )
    parser.add_argument("--experiment-name", type=str, default="candidate_eval", help="Stable experiment name for saved reports.")
    parser.add_argument("--seed", type=int, default=42, help="Recorded random seed for reproducibility metadata.")
    parser.add_argument("--split-name", type=str, default="unspecified", help="Recorded split name for experiment metadata.")
    parser.add_argument("--output-json", type=Path, help="Optional output path for JSON metrics.")
    return parser.parse_args()


def evaluate_candidate_retrieval(
    data: pd.DataFrame,
    top_k: int,
    candidate_pool_size: int | None = None,
    max_users: int | None = None,
    context: EvaluationContext | None = None,
) -> dict[str, Any]:
    """Measure retrieval recall for the current candidate generation stage."""

    evaluation_context = context or build_evaluation_context(data, max_users=max_users)

    ranked_lists: list[list[str]] = []
    relevant_lists: list[list[str]] = []

    for user_id in evaluation_context.sampled_user_ids:
        user_rows = evaluation_context.user_rows_by_id.get(user_id)
        if user_rows is None or user_rows.empty:
            continue

        session_context = build_session_context(user_rows)
        candidates = generate_candidates(
            user_id=user_id,
            top_k=top_k,
            recent_clicks=session_context["recent_clicks"],
            session_interest=session_context["session_interest"],
            candidate_pool_size=candidate_pool_size,
        )
        ranked_lists.append(candidates["article_id"].astype(str).head(top_k).tolist())
        relevant_lists.append(evaluation_context.ground_truth_by_user[user_id])

    return {
        "users_evaluated": len(ranked_lists),
        f"Recall@{top_k}": mean_metric(ranked_lists, relevant_lists, recall_at_k, top_k),
    }


def evaluate_two_tower_retrieval(
    data: pd.DataFrame,
    top_k: int,
    checkpoint_dir: Path = DEFAULT_CHECKPOINT_DIR,
    max_users: int | None = None,
    context: EvaluationContext | None = None,
) -> dict[str, Any]:
    """Measure Recall@K using a trained Two-Tower retrieval model."""

    evaluation_context = context or build_evaluation_context(data, max_users=max_users)
    user_frame = data.loc[data["customer_id"].astype(str).isin(set(evaluation_context.sampled_user_ids))].copy()
    retrievals = retrieve_candidates_for_users(
        user_rows=user_frame,
        items=data,
        checkpoint_dir=checkpoint_dir,
        top_k=top_k,
        exclude_seen_items=False,
    )

    ranked_lists: list[list[str]] = []
    relevant_lists: list[list[str]] = []
    for user_id in evaluation_context.sampled_user_ids:
        predictions = retrievals.get(user_id, [])
        if not predictions:
            continue
        ranked_lists.append([str(item["article_id"]) for item in predictions])
        relevant_lists.append(evaluation_context.ground_truth_by_user[user_id])

    return {
        "users_evaluated": len(ranked_lists),
        f"Recall@{top_k}": mean_metric(ranked_lists, relevant_lists, recall_at_k, top_k),
    }


def print_report(metrics: dict[str, Any], top_k: int, title: str = "Candidate Retrieval") -> None:
    """Pretty-print candidate retrieval metrics."""

    print(title)
    print(f"{'users evaluated':<18} {metrics['users_evaluated']}")
    print(f"{f'Recall@{top_k}':<18} {metrics[f'Recall@{top_k}']:.6f}")


def main() -> None:
    args = parse_args()
    data = load_evaluation_data(args.data)
    context = build_evaluation_context(data, max_users=args.max_users)

    if args.mode == "baseline":
        metrics = evaluate_candidate_retrieval(
            data=data,
            top_k=args.top_k,
            candidate_pool_size=args.candidate_pool_size,
            max_users=args.max_users,
            context=context,
        )
        print_report(metrics, top_k=args.top_k, title="Candidate Retrieval Baseline")
    elif args.mode == "two-tower":
        metrics = evaluate_two_tower_retrieval(
            data=data,
            top_k=args.top_k,
            checkpoint_dir=args.checkpoint_dir,
            max_users=args.max_users,
            context=context,
        )
        print_report(metrics, top_k=args.top_k, title="Candidate Retrieval Two-Tower")
    else:
        baseline_metrics = evaluate_candidate_retrieval(
            data=data,
            top_k=args.top_k,
            candidate_pool_size=args.candidate_pool_size,
            max_users=args.max_users,
            context=context,
        )
        two_tower_metrics = evaluate_two_tower_retrieval(
            data=data,
            top_k=args.top_k,
            checkpoint_dir=args.checkpoint_dir,
            max_users=args.max_users,
            context=context,
        )
        metrics = {
            "baseline": baseline_metrics,
            "two_tower": two_tower_metrics,
            "lift": {
                f"Recall@{args.top_k}": two_tower_metrics[f"Recall@{args.top_k}"] - baseline_metrics[f"Recall@{args.top_k}"],
            },
        }
        print_report(baseline_metrics, top_k=args.top_k, title="Candidate Retrieval Baseline")
        print()
        print_report(two_tower_metrics, top_k=args.top_k, title="Candidate Retrieval Two-Tower")
        print()
        print("Two-Tower Lift")
        print(f"{f'Recall@{args.top_k}':<18} {metrics['lift'][f'Recall@{args.top_k}']:.6f}")

    if args.output_json is not None:
        report = build_experiment_report(
            experiment_name=args.experiment_name,
            stage="candidate",
            data_path=args.data,
            metrics=metrics,
            config={
                "mode": args.mode,
                "top_k": args.top_k,
                "candidate_pool_size": args.candidate_pool_size,
                "max_users": args.max_users,
                "checkpoint_dir": str(args.checkpoint_dir.expanduser().resolve()),
                "seed": args.seed,
                "split_name": args.split_name,
            },
        )
        output_path = write_json_report(args.output_json, report)
        print(f"\nSaved JSON metrics to {output_path}")


if __name__ == "__main__":
    main()
