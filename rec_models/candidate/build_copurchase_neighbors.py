"""Build distance-weighted article co-purchase neighbor artifacts."""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from rec_models.serving.candidate_service import normalize_article_id
except ImportError:  # pragma: no cover - supports running from rec_models/ as cwd
    from serving.candidate_service import normalize_article_id  # type: ignore[no-redef]


LOGGER = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_PATH = BASE_DIR / "data" / "processed" / "candidate_interactions_dev.csv.gz"
DEFAULT_OUTPUT_PATH = BASE_DIR / "rec_models" / "artifacts" / "candidate" / "candidate_copurchase_neighbors.parquet"
DEFAULT_SEQUENCE_WINDOW = 5
DEFAULT_TOP_N = 40


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build article co-purchase neighbors from transaction history.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH, help="Input interactions CSV/CSV.GZ path.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Output parquet path.")
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N, help="Neighbors retained per seed article.")
    parser.add_argument(
        "--sequence-window",
        type=int,
        default=DEFAULT_SEQUENCE_WINDOW,
        help="Max sequence distance considered within each user's purchase history.",
    )
    parser.add_argument(
        "--exclude-last-per-user",
        action="store_true",
        help="Drop each user's final purchase before building the artifact to avoid target leakage.",
    )
    return parser.parse_args()


def load_interactions(input_path: Path) -> pd.DataFrame:
    resolved_path = input_path.expanduser().resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(f"Interaction data not found: {resolved_path}")

    interactions = pd.read_csv(resolved_path)
    required_columns = {"customer_id", "article_id", "t_dat"}
    missing_columns = required_columns - set(interactions.columns)
    if missing_columns:
        raise ValueError(f"Missing required interaction columns: {sorted(missing_columns)}")

    interactions["customer_id"] = interactions["customer_id"].astype(str)
    interactions["article_id"] = interactions["article_id"].map(normalize_article_id)
    interactions["t_dat"] = pd.to_datetime(interactions["t_dat"], errors="coerce")
    interactions = interactions.dropna(subset=["t_dat"]).copy()
    interactions = interactions.sort_values(["customer_id", "t_dat", "article_id"]).reset_index(drop=True)
    return interactions


def trim_last_purchase_per_user(interactions: pd.DataFrame) -> pd.DataFrame:
    kept_frames: list[pd.DataFrame] = []
    for _, user_rows in interactions.groupby("customer_id", sort=False):
        if len(user_rows) <= 1:
            continue
        kept_frames.append(user_rows.iloc[:-1].copy())
    if not kept_frames:
        return interactions.iloc[0:0].copy()
    return pd.concat(kept_frames, ignore_index=True)


def build_copurchase_neighbors(
    interactions: pd.DataFrame,
    *,
    top_n: int = DEFAULT_TOP_N,
    sequence_window: int = DEFAULT_SEQUENCE_WINDOW,
) -> pd.DataFrame:
    """Aggregate distance-weighted co-purchase neighbors per seed article."""

    pair_stats: dict[tuple[str, str], dict[str, float]] = defaultdict(lambda: {"weighted_score": 0.0, "cooccurrence_count": 0.0})
    users_processed = 0

    for _, user_rows in interactions.groupby("customer_id", sort=False):
        article_sequence = user_rows.loc[:, ["article_id", "t_dat"]].drop_duplicates().reset_index(drop=True)
        article_ids = article_sequence["article_id"].astype(str).tolist()
        purchase_times = article_sequence["t_dat"].tolist()
        if len(article_ids) < 2:
            continue

        users_processed += 1
        for right_index, article_id in enumerate(article_ids):
            left_start = max(0, right_index - sequence_window)
            for left_index in range(left_start, right_index):
                neighbor_article_id = article_ids[left_index]
                if article_id == neighbor_article_id:
                    continue
                distance = right_index - left_index
                same_day_bonus = 1.0 if purchase_times[right_index].date() == purchase_times[left_index].date() else 0.0
                weight = (1.0 / float(distance)) + same_day_bonus
                pair_stats[(article_id, neighbor_article_id)]["weighted_score"] += weight
                pair_stats[(article_id, neighbor_article_id)]["cooccurrence_count"] += 1.0
                pair_stats[(neighbor_article_id, article_id)]["weighted_score"] += weight
                pair_stats[(neighbor_article_id, article_id)]["cooccurrence_count"] += 1.0

    rows = [
        {
            "article_id": seed_article_id,
            "neighbor_article_id": neighbor_article_id,
            "weighted_score": float(stats["weighted_score"]),
            "cooccurrence_count": int(stats["cooccurrence_count"]),
        }
        for (seed_article_id, neighbor_article_id), stats in pair_stats.items()
    ]
    if not rows:
        return pd.DataFrame(columns=["article_id", "neighbor_article_id", "weighted_score", "cooccurrence_count", "neighbor_rank"])

    neighbors = pd.DataFrame(rows)
    neighbors = neighbors.sort_values(
        ["article_id", "weighted_score", "cooccurrence_count", "neighbor_article_id"],
        ascending=[True, False, False, True],
    ).reset_index(drop=True)
    neighbors["neighbor_rank"] = neighbors.groupby("article_id", sort=False).cumcount() + 1
    neighbors = neighbors.loc[neighbors["neighbor_rank"].le(top_n)].reset_index(drop=True)

    LOGGER.info(
        "Built co-purchase neighbors | users=%s seed_articles=%s rows=%s top_n=%s sequence_window=%s",
        users_processed,
        neighbors["article_id"].nunique(),
        len(neighbors),
        top_n,
        sequence_window,
    )
    return neighbors


def write_artifact(neighbors: pd.DataFrame, output_path: Path, metadata: dict[str, Any]) -> None:
    resolved_output = output_path.expanduser().resolve()
    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = resolved_output.with_suffix(".metadata.json")
    storage_format = "parquet"
    try:
        neighbors.to_parquet(resolved_output, index=False)
    except ImportError:
        storage_format = "csv_fallback"
        neighbors.to_csv(resolved_output, index=False)
        LOGGER.warning("Parquet engine unavailable. Wrote CSV fallback payload to %s", resolved_output)
    metadata = {**metadata, "storage_format": storage_format}
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    LOGGER.info("Wrote co-purchase artifact to %s", resolved_output)
    LOGGER.info("Wrote co-purchase metadata to %s", metadata_path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    args = parse_args()
    interactions = load_interactions(args.input)
    source_rows = len(interactions)
    if args.exclude_last_per_user:
        interactions = trim_last_purchase_per_user(interactions)

    neighbors = build_copurchase_neighbors(
        interactions,
        top_n=args.top_n,
        sequence_window=args.sequence_window,
    )
    metadata = {
        "input_path": str(args.input.expanduser().resolve()),
        "output_path": str(args.output.expanduser().resolve()),
        "rows_read": int(source_rows),
        "rows_used": int(len(interactions)),
        "users_used": int(interactions["customer_id"].nunique()),
        "seed_articles": int(neighbors["article_id"].nunique()) if not neighbors.empty else 0,
        "neighbor_rows": int(len(neighbors)),
        "top_n": int(args.top_n),
        "sequence_window": int(args.sequence_window),
        "exclude_last_per_user": bool(args.exclude_last_per_user),
        "leakage_safe_for_last_item_eval": bool(args.exclude_last_per_user),
    }
    write_artifact(neighbors, args.output, metadata)


if __name__ == "__main__":
    main()
