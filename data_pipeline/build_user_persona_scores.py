import csv
import logging
import os
import time
from pathlib import Path
from typing import Dict, Sequence

BASE_DIR = Path(__file__).resolve().parent.parent

# MODE = "production"
MODE = "test"

MODE_CONFIG = {
    "test": {
        "INPUT_FILE": BASE_DIR / "data" / "processed" / "customer_purchase_profile_test.csv",
        "OUTPUT_FILE": BASE_DIR / "data" / "processed" / "user_persona_scores_test.csv",
    },
    "production": {
        "INPUT_FILE": BASE_DIR / "data" / "processed" / "customer_purchase_profile.csv",
        "OUTPUT_FILE": BASE_DIR / "data" / "processed" / "user_persona_scores.csv",
    },
}

RUNTIME_MODE = os.getenv("DATA_PIPELINE_MODE", MODE).strip().lower()

if RUNTIME_MODE not in MODE_CONFIG:
    raise ValueError(f"Unsupported MODE: {RUNTIME_MODE}")

CONFIG = MODE_CONFIG[RUNTIME_MODE]
INPUT_FILE: Path = CONFIG["INPUT_FILE"]
OUTPUT_FILE: Path = CONFIG["OUTPUT_FILE"]

REQUIRED_COLUMNS = [
    "customer_id",
    "purchase_count",
    "recency_days",
    "avg_price",
    "unique_items",
    "unique_categories",
    "top_category_share",
    "top_color_share",
    "top_department_share",
    "solid_share",
    "basic_share",
    "neutral_color_share",
    "low_price_ratio",
    "repeat_article_share",
    "same_day_multi_buy_avg",
    "avg_gap_days",
    "trendy_item_ratio",
]

PERSONAS = [
    "trendsetter",
    "practical",
    "value",
    "brand_loyal",
    "impulse",
    "careful",
    "repeat_stable",
    "color_focus",
    "category_focus",
]

OUTPUT_COLUMNS = ["customer_id"]
for persona_name in PERSONAS:
    OUTPUT_COLUMNS.append(f"{persona_name}_score")
for persona_name in PERSONAS:
    OUTPUT_COLUMNS.append(f"{persona_name}_ratio")
OUTPUT_COLUMNS.extend(["top_persona", "top_persona_ratio"])


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def validate_required_columns(file_path: Path, required_columns: Sequence[str]) -> None:
    with file_path.open(newline="", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames or []

    missing_columns = [column for column in required_columns if column not in fieldnames]
    if missing_columns:
        raise ValueError(
            f"Missing required columns in {file_path}: {', '.join(missing_columns)}"
        )


def parse_float(raw_value: str) -> float:
    value = (raw_value or "").strip()
    if not value:
        return 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0


def minmax_normalize(value: float, min_value: float, max_value: float) -> float:
    if max_value <= min_value:
        return 0.0
    normalized = (value - min_value) / (max_value - min_value)
    if normalized < 0.0:
        return 0.0
    if normalized > 1.0:
        return 1.0
    return normalized


def load_rows(file_path: Path) -> list[dict[str, str]]:
    with file_path.open(newline="", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        return [row for row in reader if (row.get("customer_id") or "").strip()]


def build_feature_ranges(rows: list[dict[str, str]], feature_names: list[str]) -> Dict[str, tuple[float, float]]:
    ranges: Dict[str, tuple[float, float]] = {}
    for feature_name in feature_names:
        values = [parse_float(row.get(feature_name, "")) for row in rows]
        ranges[feature_name] = (min(values) if values else 0.0, max(values) if values else 0.0)
    return ranges


def compute_scores(row: dict[str, str], ranges: Dict[str, tuple[float, float]]) -> Dict[str, float]:
    purchase_count_norm = minmax_normalize(
        parse_float(row.get("purchase_count", "")),
        *ranges["purchase_count"],
    )
    recency_norm = minmax_normalize(
        parse_float(row.get("recency_days", "")),
        *ranges["recency_days"],
    )
    avg_price_norm = minmax_normalize(
        parse_float(row.get("avg_price", "")),
        *ranges["avg_price"],
    )
    unique_items_norm = minmax_normalize(
        parse_float(row.get("unique_items", "")),
        *ranges["unique_items"],
    )
    unique_categories_norm = minmax_normalize(
        parse_float(row.get("unique_categories", "")),
        *ranges["unique_categories"],
    )
    same_day_multi_buy_norm = minmax_normalize(
        parse_float(row.get("same_day_multi_buy_avg", "")),
        *ranges["same_day_multi_buy_avg"],
    )
    avg_gap_days_norm = minmax_normalize(
        parse_float(row.get("avg_gap_days", "")),
        *ranges["avg_gap_days"],
    )

    inv_recency = 1.0 - recency_norm
    inv_avg_price = 1.0 - avg_price_norm
    inv_purchase_count = 1.0 - purchase_count_norm

    solid_share = parse_float(row.get("solid_share", ""))
    basic_share = parse_float(row.get("basic_share", ""))
    neutral_color_share = parse_float(row.get("neutral_color_share", ""))
    low_price_ratio = parse_float(row.get("low_price_ratio", ""))
    trendy_item_ratio = parse_float(row.get("trendy_item_ratio", ""))
    top_department_share = parse_float(row.get("top_department_share", ""))
    repeat_article_share = parse_float(row.get("repeat_article_share", ""))
    top_color_share = parse_float(row.get("top_color_share", ""))
    top_category_share = parse_float(row.get("top_category_share", ""))

    scores = {
        "trendsetter": (
            0.35 * inv_recency
            + 0.25 * unique_items_norm
            + 0.20 * unique_categories_norm
            + 0.20 * trendy_item_ratio
        ),
        "practical": (
            0.40 * solid_share
            + 0.35 * basic_share
            + 0.25 * neutral_color_share
        ),
        "value": 0.55 * low_price_ratio + 0.45 * inv_avg_price,
        "brand_loyal": top_department_share,
        "impulse": (
            0.65 * same_day_multi_buy_norm
            + 0.20 * purchase_count_norm
            + 0.15 * inv_recency
        ),
        "careful": 0.70 * avg_gap_days_norm + 0.30 * inv_purchase_count,
        "repeat_stable": 0.70 * repeat_article_share + 0.30 * basic_share,
        "color_focus": top_color_share,
        "category_focus": top_category_share,
    }
    return scores


def convert_scores_to_ratios(scores: Dict[str, float]) -> Dict[str, float]:
    clipped_scores = {name: max(0.0, value) for name, value in scores.items()}
    total_score = sum(clipped_scores.values())
    if total_score <= 0.0:
        equal_ratio = 1.0 / len(clipped_scores)
        return {name: equal_ratio for name in clipped_scores}
    return {name: clipped_scores[name] / total_score for name in clipped_scores}


def write_output(rows: list[dict[str, str]], output_path: Path) -> None:
    start_time = time.perf_counter()
    feature_ranges = build_feature_ranges(
        rows,
        [
            "purchase_count",
            "recency_days",
            "avg_price",
            "unique_items",
            "unique_categories",
            "same_day_multi_buy_avg",
            "avg_gap_days",
        ],
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0

    with output_path.open("w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()

        for row in rows:
            customer_id = (row.get("customer_id") or "").strip()
            scores = compute_scores(row, feature_ranges)
            ratios = convert_scores_to_ratios(scores)
            top_persona = max(ratios, key=ratios.get)

            output_row = {"customer_id": customer_id}
            for persona_name in PERSONAS:
                output_row[f"{persona_name}_score"] = f"{scores[persona_name]:.6f}"
            for persona_name in PERSONAS:
                output_row[f"{persona_name}_ratio"] = f"{ratios[persona_name]:.6f}"
            output_row["top_persona"] = top_persona
            output_row["top_persona_ratio"] = f"{ratios[top_persona]:.6f}"
            writer.writerow(output_row)
            rows_written += 1

    elapsed = time.perf_counter() - start_time
    logging.info(
        "user_persona_scores_complete output=%s rows_written=%s elapsed_seconds=%.2f",
        output_path,
        rows_written,
        elapsed,
    )


def main() -> None:
    configure_logging()
    validate_required_columns(INPUT_FILE, REQUIRED_COLUMNS)
    rows = load_rows(INPUT_FILE)
    write_output(rows, OUTPUT_FILE)


if __name__ == "__main__":
    main()
