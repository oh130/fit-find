from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RAW_ARTICLES = REPO_ROOT / "data" / "raw" / "articles.csv"
DEFAULT_PROCESSED_ROOT = REPO_ROOT / "data" / "processed"
DEFAULT_IMAGE_ROOT = Path("D:/imagedata")
DEFAULT_OUTPUT = SCRIPT_DIR / "search_pairs_dev.csv"


def normalize_article_id(value: object) -> str:
    raw = str(value or "").strip()
    digits = "".join(ch for ch in raw if ch.isdigit())
    if not digits:
        return raw
    return digits[-10:].zfill(10)


def candidate_image_paths(image_root: Path, article_id: str) -> Iterable[Path]:
    normalized = normalize_article_id(article_id)
    prefixes = [normalized[:3], normalized[:2], ""]
    for prefix in prefixes:
        base = image_root / prefix if prefix else image_root
        yield base / f"{normalized}.jpg"
        yield base / f"{normalized}.jpeg"
        yield base / f"{normalized}.png"


def locate_image_path(image_root: Path, article_id: str) -> str:
    for candidate in candidate_image_paths(image_root, article_id):
        try:
            if candidate.exists():
                return str(candidate)
        except OSError:
            continue
    return ""


def build_manifest(raw_articles: Path, processed_root: Path, image_root: Path) -> pd.DataFrame:
    articles = pd.read_csv(raw_articles, dtype={"article_id": str}).fillna("")
    articles["article_id"] = articles["article_id"].map(normalize_article_id)

    feature_candidates = [
        processed_root / "articles_feature.csv",
        processed_root / "item_master_dev.csv",
    ]
    merged = articles.copy()
    for feature_path in feature_candidates:
        if not feature_path.exists():
            continue
        feature_df = pd.read_csv(feature_path, dtype={"article_id": str}).fillna("")
        if "article_id" not in feature_df.columns:
            continue
        feature_df["article_id"] = feature_df["article_id"].map(normalize_article_id)
        feature_df = feature_df.drop_duplicates(subset=["article_id"])
        merged = merged.merge(feature_df, on="article_id", how="left", suffixes=("", "_feature"))
        for column in list(merged.columns):
            if not column.endswith("_feature"):
                continue
            base_column = column[:-8]
            if base_column not in merged.columns:
                merged[base_column] = merged[column]
            else:
                merged[base_column] = merged[base_column].where(
                    merged[base_column].astype(str).str.strip().ne(""),
                    merged[column],
                )
            merged = merged.drop(columns=[column])
        break

    merged["image_path"] = merged["article_id"].map(lambda article_id: locate_image_path(image_root, article_id))
    merged["image_available"] = merged["image_path"].astype(str).str.strip().ne("")
    manifest = merged[merged["image_available"]].copy()
    return manifest.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a search pair manifest from raw H&M articles and D:/imagedata")
    parser.add_argument("--raw-articles", default=str(DEFAULT_RAW_ARTICLES))
    parser.add_argument("--processed-root", default=str(DEFAULT_PROCESSED_ROOT))
    parser.add_argument("--image-root", default=str(DEFAULT_IMAGE_ROOT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    manifest = build_manifest(
        raw_articles=Path(args.raw_articles),
        processed_root=Path(args.processed_root),
        image_root=Path(args.image_root),
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("===== Search Pair Manifest =====")
    print(f"Raw articles : {args.raw_articles}")
    print(f"Image root   : {args.image_root}")
    print(f"Output       : {output_path}")
    print(f"Rows saved   : {len(manifest)}")


if __name__ == "__main__":
    main()
