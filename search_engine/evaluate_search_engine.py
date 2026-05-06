from __future__ import annotations

import argparse
import base64
import csv
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

# Make repo root importable when this file lives in search_engine/
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    # Preferred import when run from repository root
    from evaluation.metrics import mean_hit_rate_at_k, mean_reciprocal_rank, mean_ndcg_at_k
except ImportError:
    # Fallback if executed from inside evaluation/
    from metrics import mean_hit_rate_at_k, mean_reciprocal_rank, mean_ndcg_at_k


def _first_nonempty(row: dict[str, str], keys: Sequence[str]) -> str:
    for key in keys:
        value = (row.get(key) or "").strip()
        if value:
            return value
    return ""


def _parse_items(value: str) -> list[str]:
    """
    Supports:
      - item_1|item_2|item_3
      - item_1,item_2,item_3
      - ["item_1", "item_2"]
      - single item string
    """
    value = (value or "").strip()
    if not value:
        return []

    if value.startswith("[") and value.endswith("]"):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except json.JSONDecodeError:
            pass

    if "|" in value:
        parts = value.split("|")
    elif "," in value:
        parts = value.split(",")
    else:
        parts = [value]

    return [p.strip() for p in parts if p.strip()]


def _image_to_base64(path: str, base_dir: Path) -> str:
    p = Path(path)
    if not p.is_absolute():
        p = base_dir / p
    if not p.exists():
        raise FileNotFoundError(f"Image file not found: {p}")
    return base64.b64encode(p.read_bytes()).decode("utf-8")


def _call_search_api(
    endpoint: str,
    query: str,
    image_base64: str | None,
    top_k: int,
    timeout: float,
) -> dict[str, Any]:
    payload = {
        "query": query,
        "image_base64": image_base64,
        "top_k": top_k,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        endpoint,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw)
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code} error from {endpoint}: {detail}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Failed to connect to {endpoint}: {e}") from e


def load_eval_rows(csv_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            query = _first_nonempty(row, ["query", "text", "query_text"])
            relevant = _first_nonempty(row, ["relevant_items", "labels", "ground_truth"])
            image_base64 = _first_nonempty(row, ["image_base64"])
            image_path = _first_nonempty(row, ["image_path", "image_file"])

            rows.append(
                {
                    "query_id": _first_nonempty(row, ["query_id", "qid", "id"]),
                    "query": query,
                    "relevant_items": _parse_items(relevant),
                    "image_base64": image_base64,
                    "image_path": image_path,
                }
            )
    return rows


def evaluate(
    input_csv: Path,
    endpoint: str,
    top_k: int = 10,
    metric_k: int = 10,
    timeout: float = 60.0,
) -> None:
    rows = load_eval_rows(input_csv)
    if not rows:
        raise ValueError(f"No rows found in {input_csv}")

    ranked_lists: list[list[str]] = []
    relevant_lists: list[list[str]] = []
    api_latencies: list[float] = []
    wall_latencies: list[float] = []

    base_dir = input_csv.parent

    for idx, row in enumerate(rows, start=1):
        query = row["query"].strip()
        relevant_items = row["relevant_items"]

        if not query and not row["image_base64"] and not row["image_path"]:
            print(f"[skip] row {idx}: query/image is empty")
            continue

        image_b64 = row["image_base64"] or None
        if image_b64 is None and row["image_path"]:
            image_b64 = _image_to_base64(row["image_path"], base_dir)

        started = time.perf_counter()
        response = _call_search_api(
            endpoint=endpoint,
            query=query,
            image_base64=image_b64,
            top_k=top_k,
            timeout=timeout,
        )
        wall_ms = (time.perf_counter() - started) * 1000.0

        results = response.get("results", [])
        ranked_items = [str(item.get("product_id", "")).strip() for item in results if str(item.get("product_id", "")).strip()]

        ranked_lists.append(ranked_items)
        relevant_lists.append([str(x).strip() for x in relevant_items if str(x).strip()])

        api_latencies.append(float(response.get("latency_ms", wall_ms)))
        wall_latencies.append(wall_ms)

    if not ranked_lists:
        raise ValueError("No valid evaluation samples were processed.")

    hitrate = mean_hit_rate_at_k(ranked_lists, relevant_lists, metric_k)
    mrr = mean_reciprocal_rank(ranked_lists, relevant_lists)
    ndcg = mean_ndcg_at_k(ranked_lists, relevant_lists, metric_k)

    avg_api_latency = mean(api_latencies)
    avg_wall_latency = mean(wall_latencies)

    print("===== Search Engine Evaluation =====")
    print(f"Samples evaluated : {len(ranked_lists)}")
    print(f"Top-K requested    : {top_k}")
    print(f"Metric K          : {metric_k}")
    print(f"HitRate@{metric_k}     : {hitrate:.6f}")
    print(f"MRR               : {mrr:.6f}")
    print(f"NDCG@{metric_k}        : {ndcg:.6f}")
    print(f"Avg API latency   : {avg_api_latency:.2f} ms")
    print(f"Avg wall latency  : {avg_wall_latency:.2f} ms")
    print()
    print("===== Target Check =====")
    print(f"Latency <= 200ms  : {'PASS' if avg_wall_latency <= 200.0 else 'FAIL'}")
    print(f"MRR >= 0.55       : {'PASS' if mrr >= 0.55 else 'FAIL'}")
    print(f"NDCG@10 >= 0.50   : {'PASS' if ndcg >= 0.50 else 'FAIL'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate search_engine with metrics.py")
    parser.add_argument(
        "--input",
        default=str(REPO_ROOT / "evaluation" / "search_eval_set.csv"),
        help="CSV file with query/relevant_items columns (default: evaluation/search_eval_set.csv)",
    )
    parser.add_argument(
        "--endpoint",
        default="http://localhost:8002/search",
        help="Search API endpoint (default: http://localhost:8002/search)",
    )
    parser.add_argument("--top-k", type=int, default=10, help="Top-K passed to the search API")
    parser.add_argument("--metric-k", type=int, default=10, help="K used for HitRate@K / NDCG@K")
    parser.add_argument("--timeout", type=float, default=60.0, help="HTTP timeout in seconds")
    args = parser.parse_args()

    evaluate(
        input_csv=Path(args.input),
        endpoint=args.endpoint,
        top_k=args.top_k,
        metric_k=args.metric_k,
        timeout=args.timeout,
    )


if __name__ == "__main__":
    main()
