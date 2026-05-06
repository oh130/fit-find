from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from statistics import mean
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.metrics import mean_hit_rate_at_k, mean_ndcg_at_k, mean_reciprocal_rank

DEFAULT_ENDPOINT = "http://localhost:8002/search"
DEFAULT_TOP_K = 10
DEFAULT_METRIC_K = 10
DEFAULT_TEST_SAMPLE_SIZE = 500
DEFAULT_MIN_RELEVANT = 2
DEFAULT_TIMEOUT = 60.0


def _load_articles(processed_csv: Path, mode: str, sample_size: int) -> pd.DataFrame:
    df = pd.read_csv(processed_csv, dtype=str).fillna("")
    if mode == "test":
        df = df.sample(n=min(sample_size, len(df)), random_state=42)
    return df.reset_index(drop=True)


def _build_query(row: pd.Series) -> str:
    color = str(row.get("colour_group_name", "")).strip()
    product_type = str(row.get("product_type_name", "")).strip()
    prod_name = str(row.get("prod_name", "")).strip()

    if color and product_type:
        return f"{color} {product_type}".strip()
    if prod_name:
        return prod_name
    return product_type or color


def build_eval_set(
    processed_csv: Path,
    mode: str,
    sample_size: int,
    min_relevant: int,
) -> pd.DataFrame:
    df = _load_articles(processed_csv, mode=mode, sample_size=sample_size)
    df["query"] = df.apply(_build_query, axis=1)
    df["article_id"] = df["article_id"].astype(str).str.strip()
    df = df[df["query"].str.strip() != ""]
    df = df[df["article_id"] != ""]

    grouped = (
        df.groupby("query")["article_id"]
        .apply(lambda s: sorted({str(item).strip() for item in s if str(item).strip()}))
        .reset_index(name="relevant_items")
    )
    grouped["relevant_count"] = grouped["relevant_items"].apply(len)
    grouped = grouped[grouped["relevant_count"] >= min_relevant].copy()
    grouped = grouped.sort_values(["relevant_count", "query"], ascending=[False, True]).reset_index(drop=True)
    grouped.insert(0, "query_id", [f"search_q_{idx + 1:04d}" for idx in range(len(grouped))])
    return grouped


def _post_search(endpoint: str, query: str, top_k: int, timeout: float) -> dict[str, Any]:
    payload = json.dumps({"query": query, "image_base64": None, "top_k": top_k}).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {endpoint}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Failed to connect to {endpoint}: {exc}") from exc


def evaluate_search(
    eval_df: pd.DataFrame,
    endpoint: str,
    top_k: int,
    metric_k: int,
    timeout: float,
) -> dict[str, Any]:
    ranked_lists: list[list[str]] = []
    relevant_lists: list[list[str]] = []
    api_latencies: list[float] = []
    wall_latencies: list[float] = []
    rows: list[dict[str, Any]] = []

    for row in eval_df.itertuples(index=False):
        started = time.perf_counter()
        response = _post_search(endpoint=endpoint, query=row.query, top_k=top_k, timeout=timeout)
        wall_ms = (time.perf_counter() - started) * 1000.0

        ranked_items = [
            str(item.get("product_id", "")).strip()
            for item in response.get("results", [])
            if str(item.get("product_id", "")).strip()
        ]
        relevant_items = [str(item).strip() for item in row.relevant_items if str(item).strip()]

        ranked_lists.append(ranked_items)
        relevant_lists.append(relevant_items)
        api_latencies.append(float(response.get("latency_ms", wall_ms)))
        wall_latencies.append(wall_ms)
        rows.append(
            {
                "query_id": row.query_id,
                "query": row.query,
                "relevant_items": relevant_items,
                "ranked_items": ranked_items,
                "search_type": response.get("search_type", "text"),
                "latency_ms": float(response.get("latency_ms", wall_ms)),
                "wall_latency_ms": wall_ms,
            }
        )

    if not ranked_lists:
        raise ValueError("No evaluation rows were processed.")

    hitrate = mean_hit_rate_at_k(ranked_lists, relevant_lists, metric_k)
    mrr = mean_reciprocal_rank(ranked_lists, relevant_lists)
    ndcg = mean_ndcg_at_k(ranked_lists, relevant_lists, metric_k)
    avg_api_latency = mean(api_latencies)
    avg_wall_latency = mean(wall_latencies)
    p95_wall_latency = float(pd.Series(wall_latencies).quantile(0.95))

    return {
        "metadata": {
            "generated_at": pd.Timestamp.now("UTC").isoformat(),
            "endpoint": endpoint,
            "top_k": top_k,
            "metric_k": metric_k,
            "samples_evaluated": len(rows),
        },
        "thresholds": {
            "latency_ms_max": 200.0,
            "mrr_min": 0.55,
            "ndcg_at_10_min": 0.50,
        },
        "search": {
            f"HitRate@{metric_k}": hitrate,
            "MRR": mrr,
            f"NDCG@{metric_k}": ndcg,
            "avg_api_latency_ms": avg_api_latency,
            "avg_wall_latency_ms": avg_wall_latency,
            "p95_wall_latency_ms": p95_wall_latency,
        },
        "checks": {
            "latency_within_200ms": avg_wall_latency <= 200.0,
            "mrr_meets_target": mrr >= 0.55,
            "ndcg_meets_target": ndcg >= 0.50,
        },
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate search quality report for the dashboard")
    parser.add_argument(
        "--processed-csv",
        default=str(REPO_ROOT / "data" / "processed" / "articles_feature.csv"),
        help="Path to processed articles_feature.csv",
    )
    parser.add_argument("--mode", choices=["test", "production"], default="test")
    parser.add_argument("--sample-size", type=int, default=DEFAULT_TEST_SAMPLE_SIZE)
    parser.add_argument("--min-relevant", type=int, default=DEFAULT_MIN_RELEVANT)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--metric-k", type=int, default=DEFAULT_METRIC_K)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument(
        "--eval-set-output",
        default=str(REPO_ROOT / "evaluation" / "search_eval_set.csv"),
        help="CSV path for the generated evaluation set",
    )
    parser.add_argument(
        "--report-output",
        default=str(REPO_ROOT / "evaluation" / "search_metrics_report.json"),
        help="JSON path for the generated search report",
    )
    args = parser.parse_args()

    processed_csv = Path(args.processed_csv)
    eval_output = Path(args.eval_set_output)
    report_output = Path(args.report_output)

    eval_df = build_eval_set(
        processed_csv=processed_csv,
        mode=args.mode,
        sample_size=args.sample_size,
        min_relevant=args.min_relevant,
    )
    eval_output.parent.mkdir(parents=True, exist_ok=True)
    eval_df.assign(relevant_items=eval_df["relevant_items"].apply(lambda items: "|".join(items))).to_csv(
        eval_output,
        index=False,
        encoding="utf-8-sig",
    )

    report = evaluate_search(
        eval_df=eval_df,
        endpoint=args.endpoint,
        top_k=args.top_k,
        metric_k=args.metric_k,
        timeout=args.timeout,
    )

    report_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("===== Search Metrics Report =====")
    print(f"Eval set           : {eval_output}")
    print(f"Report output      : {report_output}")
    print(f"Samples evaluated  : {report['metadata']['samples_evaluated']}")
    print(f"HitRate@{args.metric_k}        : {report['search'][f'HitRate@{args.metric_k}']:.6f}")
    print(f"MRR                : {report['search']['MRR']:.6f}")
    print(f"NDCG@{args.metric_k}           : {report['search'][f'NDCG@{args.metric_k}']:.6f}")
    print(f"Avg wall latency   : {report['search']['avg_wall_latency_ms']:.2f} ms")
    print(f"P95 wall latency   : {report['search']['p95_wall_latency_ms']:.2f} ms")
    print(f"Latency <= 200ms   : {'PASS' if report['checks']['latency_within_200ms'] else 'FAIL'}")
    print(f"MRR >= 0.55        : {'PASS' if report['checks']['mrr_meets_target'] else 'FAIL'}")
    print(f"NDCG@10 >= 0.50    : {'PASS' if report['checks']['ndcg_meets_target'] else 'FAIL'}")


if __name__ == "__main__":
    main()
