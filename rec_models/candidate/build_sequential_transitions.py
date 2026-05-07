"""Build sequential next-item transition artifacts from transaction history."""

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
except ImportError:  # pragma: no cover
    from serving.candidate_service import normalize_article_id  # type: ignore[no-redef]


LOGGER = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_PATH = BASE_DIR / "data" / "processed" / "candidate_interactions_dev.csv.gz"
DEFAULT_ARTICLE_FEATURES_PATH = BASE_DIR / "data" / "processed" / "articles_feature.csv"
DEFAULT_OUTPUT_PATH = BASE_DIR / "rec_models" / "artifacts" / "candidate" / "candidate_sequential_transitions.csv"
DEFAULT_TOP_N = 100
DEFAULT_MAX_NEXT_STEPS = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build sequential next-item transition tables.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--article-features", type=Path, default=DEFAULT_ARTICLE_FEATURES_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--max-next-steps", type=int, default=DEFAULT_MAX_NEXT_STEPS)
    parser.add_argument("--exclude-last-per-user", action="store_true")
    return parser.parse_args()


def load_interactions(input_path: Path) -> pd.DataFrame:
    interactions = pd.read_csv(input_path.expanduser().resolve())
    required = {"customer_id", "article_id", "t_dat"}
    missing = required - set(interactions.columns)
    if missing:
        raise ValueError(f"Missing required interaction columns: {sorted(missing)}")
    interactions["customer_id"] = interactions["customer_id"].astype(str)
    interactions["article_id"] = interactions["article_id"].map(normalize_article_id)
    interactions["t_dat"] = pd.to_datetime(interactions["t_dat"], errors="coerce")
    interactions = interactions.dropna(subset=["t_dat"]).copy()
    interactions = interactions.sort_values(["customer_id", "t_dat", "article_id"]).reset_index(drop=True)
    return interactions


def trim_last_purchase_per_user(interactions: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for _, user_rows in interactions.groupby("customer_id", sort=False):
        if len(user_rows) <= 1:
            continue
        frames.append(user_rows.iloc[:-1].copy())
    if not frames:
        return interactions.iloc[0:0].copy()
    return pd.concat(frames, ignore_index=True)


def load_article_categories(article_features_path: Path) -> dict[str, str]:
    features = pd.read_csv(article_features_path.expanduser().resolve(), dtype=str).fillna("UNKNOWN")
    features["article_id"] = features["article_id"].map(normalize_article_id)
    features["main_category"] = features.get("main_category", pd.Series("UNKNOWN", index=features.index)).fillna("UNKNOWN").astype(str)
    return dict(zip(features["article_id"], features["main_category"], strict=False))


def build_transitions(
    interactions: pd.DataFrame,
    article_to_main_category: dict[str, str],
    *,
    top_n: int,
    max_next_steps: int,
) -> pd.DataFrame:
    article_stats: dict[tuple[str, str], dict[str, float]] = defaultdict(lambda: {"weighted_score": 0.0, "transition_count": 0.0})
    category_stats: dict[tuple[str, str], dict[str, float]] = defaultdict(lambda: {"weighted_score": 0.0, "transition_count": 0.0})
    users_processed = 0

    for _, user_rows in interactions.groupby("customer_id", sort=False):
        ordered = user_rows.loc[:, ["article_id", "t_dat"]].drop_duplicates().reset_index(drop=True)
        article_ids = ordered["article_id"].astype(str).tolist()
        if len(article_ids) < 2:
            continue
        users_processed += 1
        for index, seed_article_id in enumerate(article_ids[:-1]):
            seed_category = str(article_to_main_category.get(seed_article_id, "UNKNOWN"))
            for step_ahead in range(1, max_next_steps + 1):
                next_index = index + step_ahead
                if next_index >= len(article_ids):
                    break
                next_article_id = article_ids[next_index]
                weight = 1.0 / float(step_ahead)
                article_stats[(seed_article_id, next_article_id)]["weighted_score"] += weight
                article_stats[(seed_article_id, next_article_id)]["transition_count"] += 1.0
                if seed_category != "UNKNOWN":
                    category_stats[(seed_category, next_article_id)]["weighted_score"] += weight
                    category_stats[(seed_category, next_article_id)]["transition_count"] += 1.0

    rows: list[dict[str, Any]] = []
    for (seed_id, next_article_id), stats in article_stats.items():
        rows.append(
            {
                "seed_type": "article_id",
                "seed_id": seed_id,
                "next_article_id": next_article_id,
                "weighted_score": float(stats["weighted_score"]),
                "transition_count": int(stats["transition_count"]),
            }
        )
    for (seed_id, next_article_id), stats in category_stats.items():
        rows.append(
            {
                "seed_type": "main_category",
                "seed_id": seed_id,
                "next_article_id": next_article_id,
                "weighted_score": float(stats["weighted_score"]),
                "transition_count": int(stats["transition_count"]),
            }
        )

    transitions = pd.DataFrame(rows)
    if transitions.empty:
        return pd.DataFrame(columns=["seed_type", "seed_id", "next_article_id", "weighted_score", "transition_count", "neighbor_rank"])

    transitions = transitions.sort_values(
        ["seed_type", "seed_id", "weighted_score", "transition_count", "next_article_id"],
        ascending=[True, True, False, False, True],
    ).reset_index(drop=True)
    transitions["neighbor_rank"] = transitions.groupby(["seed_type", "seed_id"], sort=False).cumcount() + 1
    transitions = transitions.loc[transitions["neighbor_rank"].le(top_n)].reset_index(drop=True)
    LOGGER.info(
        "Built sequential transitions | users=%s rows=%s article_seeds=%s category_seeds=%s",
        users_processed,
        len(transitions),
        transitions.loc[transitions["seed_type"].eq("article_id"), "seed_id"].nunique(),
        transitions.loc[transitions["seed_type"].eq("main_category"), "seed_id"].nunique(),
    )
    return transitions


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    args = parse_args()
    interactions = load_interactions(args.input)
    rows_read = len(interactions)
    if args.exclude_last_per_user:
        interactions = trim_last_purchase_per_user(interactions)
    article_to_main_category = load_article_categories(args.article_features)
    transitions = build_transitions(
        interactions,
        article_to_main_category,
        top_n=args.top_n,
        max_next_steps=args.max_next_steps,
    )
    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    transitions.to_csv(output_path, index=False)
    metadata = {
        "input_path": str(args.input.expanduser().resolve()),
        "article_features_path": str(args.article_features.expanduser().resolve()),
        "output_path": str(output_path),
        "rows_read": int(rows_read),
        "rows_used": int(len(interactions)),
        "users_used": int(interactions["customer_id"].nunique()),
        "top_n": int(args.top_n),
        "max_next_steps": int(args.max_next_steps),
        "exclude_last_per_user": bool(args.exclude_last_per_user),
        "leakage_safe_for_last_item_eval": bool(args.exclude_last_per_user),
        "seed_rows": int(len(transitions)),
        "storage_format": "csv",
    }
    metadata_path = output_path.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    LOGGER.info("Wrote sequential transition artifact to %s", output_path)
    LOGGER.info("Wrote sequential transition metadata to %s", metadata_path)


if __name__ == "__main__":
    main()
