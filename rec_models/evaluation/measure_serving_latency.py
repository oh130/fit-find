"""Measure end-to-end serving latency for the recommendation API path."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

try:
    from rec_models.evaluation.data_utils import build_evaluation_context, build_session_context, load_evaluation_data
    from rec_models.evaluation.evaluate_recommender import enrich_candidate_rows
    from rec_models.serving.recommend_service import recommend, warmup_recommendation_assets
except ImportError:  # pragma: no cover - supports running from rec_models/ as cwd
    from evaluation.data_utils import build_evaluation_context, build_session_context, load_evaluation_data  # type: ignore[no-redef]
    from evaluation.evaluate_recommender import enrich_candidate_rows  # type: ignore[no-redef]
    from serving.recommend_service import recommend, warmup_recommendation_assets  # type: ignore[no-redef]


DEFAULT_DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "train_data_dev.csv"
LATENCY_EVALUATION_COLUMNS = [
    "customer_id",
    "article_id",
    "label",
    "sales_channel_id",
    "recent_clicks",
    "session_interest",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure serving latency for recommend().")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH, help="Evaluation data used to sample users.")
    parser.add_argument("--top-k", type=int, default=50, help="Recommendation cutoff passed to recommend().")
    parser.add_argument("--max-users", type=int, default=100, help="Number of users to measure.")
    parser.add_argument("--output-json", type=Path, help="Optional path for latency metrics JSON.")
    return parser.parse_args()


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(max(int(round((percentile / 100.0) * (len(ordered) - 1))), 0), len(ordered) - 1)
    return float(ordered[index])


def _summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"avg_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0}
    return {
        "avg_ms": float(statistics.mean(values)),
        "p50_ms": float(statistics.median(values)),
        "p95_ms": _percentile(values, 95.0),
        "max_ms": float(max(values)),
    }


def measure_latency(data_path: Path, top_k: int, max_users: int) -> dict[str, Any]:
    data = enrich_candidate_rows(load_evaluation_data(data_path, columns=LATENCY_EVALUATION_COLUMNS))
    context = build_evaluation_context(data, max_users=max_users)

    warmup_start = time.perf_counter()
    warmup_recommendation_assets()
    warmup_ms = (time.perf_counter() - warmup_start) * 1000.0

    wall_ms: list[float] = []
    candidate_ms: list[float] = []
    ranking_ms: list[float] = []
    reranking_ms: list[float] = []
    reported_total_ms: list[float] = []

    measurement_start = time.perf_counter()
    for user_id in context.sampled_user_ids:
        user_rows = context.user_rows_by_id.get(user_id)
        if user_rows is None or user_rows.empty:
            continue

        session_context = build_session_context(user_rows)
        request_start = time.perf_counter()
        response = recommend(
            user_id=user_id,
            top_n=top_k,
            recent_clicks=session_context.get("recent_clicks"),
            session_interest=session_context.get("session_interest"),
        )
        wall_ms.append((time.perf_counter() - request_start) * 1000.0)

        latency = response.get("pipeline_latency", {})
        candidate_ms.append(float(latency.get("candidate_ms", 0.0)))
        ranking_ms.append(float(latency.get("ranking_ms", 0.0)))
        reranking_ms.append(float(latency.get("reranking_ms", 0.0)))
        reported_total_ms.append(float(latency.get("total_ms", 0.0)))

    elapsed_s = time.perf_counter() - measurement_start
    return {
        "users_measured": len(wall_ms),
        "top_k": top_k,
        "warmup_ms": warmup_ms,
        "elapsed_s": elapsed_s,
        "wall": _summary(wall_ms),
        "reported_total": _summary(reported_total_ms),
        "candidate": _summary(candidate_ms),
        "ranking": _summary(ranking_ms),
        "reranking": _summary(reranking_ms),
    }


def main() -> None:
    args = parse_args()
    metrics = measure_latency(data_path=args.data, top_k=args.top_k, max_users=args.max_users)
    print(json.dumps(metrics, indent=2))

    if args.output_json is not None:
        output_path = args.output_json.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"Saved latency metrics to {output_path}")


if __name__ == "__main__":
    main()
