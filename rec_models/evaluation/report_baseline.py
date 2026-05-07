"""Generate a reproducible baseline report for the current recommendation stack."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from rec_models.candidate.evaluator import evaluate_candidate_retrieval
    from rec_models.evaluation.data_utils import build_evaluation_context, infer_positive_mask, load_evaluation_data
    from rec_models.evaluation.evaluate_recommender import (
        build_comparison_summary,
        enrich_candidate_rows,
        evaluate_popularity_baseline,
        evaluate_recommender,
    )
    from rec_models.ranking.evaluator import evaluate_ranking_model
except ImportError:  # pragma: no cover - supports running from rec_models/ as cwd
    from candidate.evaluator import evaluate_candidate_retrieval  # type: ignore[no-redef]
    from evaluation.data_utils import build_evaluation_context, infer_positive_mask, load_evaluation_data  # type: ignore[no-redef]
    from evaluation.evaluate_recommender import (  # type: ignore[no-redef]
        build_comparison_summary,
        enrich_candidate_rows,
        evaluate_popularity_baseline,
        evaluate_recommender,
    )
    from ranking.evaluator import evaluate_ranking_model  # type: ignore[no-redef]


DEFAULT_TOP_K = 50
DEFAULT_CANDIDATE_K = 300
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "reports" / "baseline"
DEFAULT_EVALUATION_DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "train_data_dev.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a baseline evaluation report under rec_models/reports.")
    parser.add_argument("--data", type=Path, default=DEFAULT_EVALUATION_DATA_PATH, help="Path to processed recommendation evaluation data.")
    parser.add_argument("--top_k", type=int, default=DEFAULT_TOP_K, help="Top-K cutoff for recommendation metrics.")
    parser.add_argument("--candidate_k", type=int, default=DEFAULT_CANDIDATE_K, help="Top-K cutoff for candidate recall.")
    parser.add_argument("--max-users", type=int, help="Optional deterministic user cap for faster baseline runs.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory to save baseline reports.")
    return parser.parse_args()


def _build_metadata(
    data: Any,
    source_path: Path,
    top_k: int,
    candidate_k: int,
    max_users: int | None,
    positive_mask: Any,
) -> dict[str, Any]:
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_data": str(source_path.expanduser().resolve()),
        "rows": int(len(data)),
        "users": int(data["customer_id"].astype(str).nunique()),
        "items": int(data["article_id"].astype(str).nunique()),
        "positive_rows": int(positive_mask.sum()),
        "negative_rows": int((~positive_mask).sum()),
        "top_k": top_k,
        "candidate_k": candidate_k,
        "max_users": max_users,
    }


def render_markdown(report: dict[str, Any]) -> str:
    """Render a short Markdown summary for baseline tracking."""

    metadata = report["metadata"]
    top_k = metadata["top_k"]
    candidate_k = metadata["candidate_k"]
    current = report["recommendation"]["current_model"]
    baseline = report["recommendation"].get("popularity_baseline")
    ranking = report["ranking"]
    candidate = report["candidate"]
    auc_text = "n/a" if ranking["auc"] is None else f"{ranking['auc']:.6f}"

    lines = [
        "# Recommendation Baseline Report",
        "",
        "## Dataset",
        "",
        f"- generated_at_utc: `{metadata['generated_at_utc']}`",
        f"- source_data: `{metadata['source_data']}`",
        f"- rows: `{metadata['rows']}`",
        f"- users: `{metadata['users']}`",
        f"- items: `{metadata['items']}`",
        f"- positive_rows: `{metadata['positive_rows']}`",
        f"- negative_rows: `{metadata['negative_rows']}`",
        f"- top_k: `{top_k}`",
        f"- candidate_k: `{candidate_k}`",
        f"- max_users: `{metadata['max_users']}`",
        "",
        "## Metrics",
        "",
        "| Component | Metric | Value |",
        "| --- | --- | ---: |",
        f"| Candidate | Recall@{candidate_k} | {candidate[f'Recall@{candidate_k}']:.6f} |",
        f"| Ranking | AUC | {auc_text} |",
        f"| Ranking | HitRate@{top_k} | {ranking[f'HitRate@{top_k}']:.6f} |",
        f"| Ranking | NDCG@{top_k} | {ranking[f'NDCG@{top_k}']:.6f} |",
        f"| Recommender | HitRate@{top_k} | {current[f'HitRate@{top_k}']:.6f} |",
        f"| Recommender | NDCG@{top_k} | {current[f'NDCG@{top_k}']:.6f} |",
        f"| Recommender | Coverage@{top_k} | {current[f'Coverage@{top_k}']:.6f} |",
    ]

    if baseline is not None:
        lines.extend(
            [
                f"| Popularity baseline | HitRate@{top_k} | {baseline[f'HitRate@{top_k}']:.6f} |",
                f"| Popularity baseline | NDCG@{top_k} | {baseline[f'NDCG@{top_k}']:.6f} |",
                f"| Popularity baseline | Coverage@{top_k} | {baseline[f'Coverage@{top_k}']:.6f} |",
            ]
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()

    raw_data = load_evaluation_data(args.data)
    enriched_data = enrich_candidate_rows(raw_data)
    positive_mask = infer_positive_mask(raw_data)
    sampled_user_ids = None
    raw_context = build_evaluation_context(
        raw_data,
        max_users=args.max_users,
        positive_mask=positive_mask,
    )
    sampled_user_ids = raw_context.sampled_user_ids
    enriched_context = build_evaluation_context(
        enriched_data,
        max_users=args.max_users,
        positive_mask=positive_mask,
        sampled_user_ids=sampled_user_ids,
    )

    report = {
        "metadata": _build_metadata(
            data=raw_data,
            source_path=args.data,
            top_k=args.top_k,
            candidate_k=args.candidate_k,
            max_users=args.max_users,
            positive_mask=positive_mask,
        ),
        "candidate": evaluate_candidate_retrieval(
            data=raw_data,
            top_k=args.candidate_k,
            max_users=args.max_users,
            context=raw_context,
        ),
        "ranking": evaluate_ranking_model(
            data=raw_data,
            top_k=args.top_k,
            max_users=args.max_users,
            context=raw_context,
        ),
    }
    recommender_metrics = evaluate_recommender(
        data=enriched_data,
        top_k=args.top_k,
        max_users=args.max_users,
        context=enriched_context,
    )
    popularity_metrics = evaluate_popularity_baseline(
        data=enriched_data,
        top_k=args.top_k,
        max_users=args.max_users,
        context=enriched_context,
    )
    report["recommendation"] = build_comparison_summary(
        ranking_metrics=recommender_metrics,
        baseline_metrics=popularity_metrics,
        top_k=args.top_k,
    )

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "baseline_metrics.json"
    markdown_path = output_dir / "baseline_report.md"

    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")

    print(f"Saved JSON metrics to {json_path}")
    print(f"Saved Markdown report to {markdown_path}")


if __name__ == "__main__":
    main()
