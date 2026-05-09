"""Offline evaluator for ranking models."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import json
import pandas as pd

try:
    from rec_models.common.metrics import hit_rate_at_k, mean_metric, ndcg_at_k, safe_roc_auc_score
    from rec_models.common.utils import build_experiment_report, write_json_report
    from rec_models.evaluation.data_utils import EvaluationContext, build_evaluation_context, load_evaluation_data
    from rec_models.ranking.infer import (
        DEFAULT_CHECKPOINT_DIR,
        _extract_deepfm_scores,
        _extract_scores,
        load_artifacts,
        load_deepfm_artifacts,
        prepare_inference_features,
    )
    from rec_models.ranking.train import enrich_with_item_features, enrich_with_persona_features
except ImportError:  # pragma: no cover - supports running from rec_models/ as cwd
    from common.metrics import hit_rate_at_k, mean_metric, ndcg_at_k, safe_roc_auc_score  # type: ignore[no-redef]
    from common.utils import build_experiment_report, write_json_report  # type: ignore[no-redef]
    from evaluation.data_utils import EvaluationContext, build_evaluation_context, load_evaluation_data  # type: ignore[no-redef]
    from ranking.infer import (  # type: ignore[no-redef]
        DEFAULT_CHECKPOINT_DIR,
        _extract_deepfm_scores,
        _extract_scores,
        load_artifacts,
        load_deepfm_artifacts,
        prepare_inference_features,
    )
    from ranking.train import enrich_with_item_features, enrich_with_persona_features  # type: ignore[no-redef]


DEFAULT_TOP_K = 50
DEFAULT_EVALUATION_DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "train_data_dev.csv"


def cast_numeric_features_to_float(frame: pd.DataFrame) -> pd.DataFrame:
    """Compatibility shim for legacy sklearn artifacts saved during training."""

    return frame.astype("float64")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved ranking model offline.")
    parser.add_argument("--data", type=Path, default=DEFAULT_EVALUATION_DATA_PATH, help="Path to processed ranking/evaluation data.")
    parser.add_argument("--top_k", type=int, default=DEFAULT_TOP_K, help="Cutoff K for grouped ranking metrics.")
    parser.add_argument("--max-users", type=int, help="Optional cap for smoke checks or faster iteration.")
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR, help="Checkpoint directory containing ranking artifacts.")
    parser.add_argument("--model-type", choices=("logreg", "deepfm"), default="logreg", help="Ranking model to evaluate.")
    parser.add_argument("--experiment-name", type=str, default="ranking_eval", help="Stable experiment name for saved reports.")
    parser.add_argument("--seed", type=int, default=42, help="Recorded random seed for reproducibility metadata.")
    parser.add_argument("--split-name", type=str, default="unspecified", help="Recorded split name for experiment metadata.")
    parser.add_argument("--output-json", type=Path, help="Optional output path for JSON metrics.")
    return parser.parse_args()


def evaluate_ranking_model(
    data: pd.DataFrame,
    checkpoint_dir: Path = DEFAULT_CHECKPOINT_DIR,
    top_k: int = DEFAULT_TOP_K,
    max_users: int | None = None,
    context: EvaluationContext | None = None,
    model_type: str = "logreg",
) -> dict[str, Any]:
    if model_type == "deepfm":
        model, metadata = load_deepfm_artifacts(checkpoint_dir=checkpoint_dir)
    else:
        model, metadata = load_artifacts(checkpoint_dir=checkpoint_dir)

    feature_columns = metadata.get("feature_columns", [])
    if not feature_columns:
        raise ValueError("Ranking metadata does not contain feature_columns.")

    enriched_data = enrich_with_persona_features(enrich_with_item_features(data))
    features = prepare_inference_features(enriched_data, feature_columns=feature_columns)
    scores = (
        _extract_deepfm_scores(model=model, features=features, metadata=metadata)
        if model_type == "deepfm"
        else _extract_scores(model=model, features=features)
    )

    evaluation_context = context or build_evaluation_context(data, max_users=max_users)
    user_set = set(evaluation_context.sampled_user_ids)

    scored = enriched_data.copy()
    scored["customer_id"] = scored["customer_id"].astype(str)
    scored["article_id"] = scored["article_id"].astype(str)
    scored["score"] = scores
    scored["is_positive"] = evaluation_context.positive_mask.astype(bool)
    scored = scored.loc[scored["customer_id"].isin(user_set)].copy()

    ranked_lists: list[list[str]] = []
    relevant_lists: list[list[str]] = []
    for user_id in evaluation_context.sampled_user_ids:
        user_rows = scored.loc[scored["customer_id"].eq(user_id)]
        if user_rows.empty:
            continue
        ranked_items = user_rows.sort_values(["score", "article_id"], ascending=[False, True])["article_id"].astype(str).tolist()
        ranked_lists.append(ranked_items)
        relevant_lists.append(evaluation_context.ground_truth_by_user.get(user_id, []))

    auc = safe_roc_auc_score(scored["is_positive"].astype(int).tolist(), scored["score"].astype(float).tolist())
    return {
        "rows_evaluated": int(len(scored)),
        "users_evaluated": len(ranked_lists),
        "auc": auc,
        f"HitRate@{top_k}": mean_metric(ranked_lists, relevant_lists, hit_rate_at_k, top_k),
        f"NDCG@{top_k}": mean_metric(ranked_lists, relevant_lists, ndcg_at_k, top_k),
    }


def print_report(metrics: dict[str, Any], top_k: int) -> None:
    print("Ranking Model")
    print(f"{'rows evaluated':<18} {metrics['rows_evaluated']}")
    print(f"{'users evaluated':<18} {metrics['users_evaluated']}")
    auc = metrics["auc"]
    print(f"{'AUC':<18} {auc:.6f}" if auc is not None else f"{'AUC':<18} n/a")
    print(f"{f'HitRate@{top_k}':<18} {metrics[f'HitRate@{top_k}']:.6f}")
    print(f"{f'NDCG@{top_k}':<18} {metrics[f'NDCG@{top_k}']:.6f}")


def main() -> None:
    args = parse_args()
    data = load_evaluation_data(args.data)
    metrics = evaluate_ranking_model(
        data=data,
        checkpoint_dir=args.checkpoint_dir,
        top_k=args.top_k,
        max_users=args.max_users,
        model_type=args.model_type,
    )
    print_report(metrics, top_k=args.top_k)

    if args.output_json is not None:
        report = build_experiment_report(
            experiment_name=args.experiment_name,
            stage="ranking",
            data_path=args.data,
            metrics=metrics,
            config={
                "top_k": args.top_k,
                "max_users": args.max_users,
                "checkpoint_dir": str(args.checkpoint_dir.expanduser().resolve()),
                "model_type": args.model_type,
                "seed": args.seed,
                "split_name": args.split_name,
            },
        )
        output_path = write_json_report(args.output_json, report)
        print(f"\nSaved JSON metrics to {output_path}")


if __name__ == "__main__":
    main()
