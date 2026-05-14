import csv
import logging
import os
import time
from pathlib import Path
from typing import Dict, Sequence

BASE_DIR = Path(__file__).resolve().parent.parent

# MODE = "production"
# MODE = "dev"
MODE = "test"

MODE_CONFIG = {
    "test": {
        "INPUT_FILE": BASE_DIR / "data" / "processed" / "customer_purchase_profile_test.csv",
        "OUTPUT_FILE": BASE_DIR / "data" / "processed" / "user_persona_scores_test.csv",
    },
    "dev": {
        "INPUT_FILE": BASE_DIR / "data" / "processed" / "customer_purchase_profile_dev.csv",
        "OUTPUT_FILE": BASE_DIR / "data" / "processed" / "user_persona_scores_dev.csv",
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


def interpolate_signal(value: float, low_value: float, high_value: float) -> float:
    if high_value <= low_value:
        return 0.0
    if value <= low_value:
        return 0.0
    if value >= high_value:
        return 1.0
    return (value - low_value) / (high_value - low_value)


def percentile(sorted_values: list[float], ratio: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]

    position = (len(sorted_values) - 1) * ratio
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(sorted_values) - 1)
    weight = position - lower_index
    lower_value = sorted_values[lower_index]
    upper_value = sorted_values[upper_index]
    return lower_value + (upper_value - lower_value) * weight


def load_rows(file_path: Path) -> list[dict[str, str]]:
    with file_path.open(newline="", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        filtered_rows: list[dict[str, str]] = []
        skipped_zero_purchase = 0

        for row in reader:
            customer_id = (row.get("customer_id") or "").strip()
            if not customer_id:
                continue

            purchase_count = parse_float(row.get("purchase_count", ""))
            if purchase_count <= 0.0:
                skipped_zero_purchase += 1
                continue

            filtered_rows.append(row)

    logging.info(
        "user_persona_input_filtered kept_rows=%s skipped_zero_purchase_rows=%s",
        len(filtered_rows),
        skipped_zero_purchase,
    )
    return filtered_rows


def build_feature_ranges(rows: list[dict[str, str]], feature_names: list[str]) -> Dict[str, dict[str, float]]:
    ranges: Dict[str, dict[str, float]] = {}
    for feature_name in feature_names:
        values = sorted(parse_float(row.get(feature_name, "")) for row in rows)
        ranges[feature_name] = {
            "min": min(values) if values else 0.0,
            "max": max(values) if values else 0.0,
            "q25": percentile(values, 0.25),
            "q50": percentile(values, 0.50),
            "q75": percentile(values, 0.75),
            "q90": percentile(values, 0.90),
        }
    return ranges


def compute_scores(row: dict[str, str], ranges: Dict[str, dict[str, float]]) -> Dict[str, float]:
    purchase_count = parse_float(row.get("purchase_count", ""))
    recency_days = parse_float(row.get("recency_days", ""))
    avg_price = parse_float(row.get("avg_price", ""))
    unique_items = parse_float(row.get("unique_items", ""))
    unique_categories = parse_float(row.get("unique_categories", ""))
    same_day_multi_buy_avg = parse_float(row.get("same_day_multi_buy_avg", ""))
    avg_gap_days = parse_float(row.get("avg_gap_days", ""))

    purchase_count_norm = minmax_normalize(
        purchase_count,
        ranges["purchase_count"]["min"],
        ranges["purchase_count"]["max"],
    )
    recency_norm = minmax_normalize(
        recency_days,
        ranges["recency_days"]["min"],
        ranges["recency_days"]["max"],
    )
    unique_items_norm = minmax_normalize(
        unique_items,
        ranges["unique_items"]["min"],
        ranges["unique_items"]["max"],
    )
    unique_categories_norm = minmax_normalize(
        unique_categories,
        ranges["unique_categories"]["min"],
        ranges["unique_categories"]["max"],
    )

    solid_share = parse_float(row.get("solid_share", ""))
    basic_share = parse_float(row.get("basic_share", ""))
    neutral_color_share = parse_float(row.get("neutral_color_share", ""))
    low_price_ratio = parse_float(row.get("low_price_ratio", ""))
    trendy_item_ratio = parse_float(row.get("trendy_item_ratio", ""))
    top_department_share = parse_float(row.get("top_department_share", ""))
    repeat_article_share = parse_float(row.get("repeat_article_share", ""))
    top_color_share = parse_float(row.get("top_color_share", ""))
    top_category_share = parse_float(row.get("top_category_share", ""))

    fresh_signal = 1.0 - interpolate_signal(
        recency_days,
        ranges["recency_days"]["q25"],
        ranges["recency_days"]["q75"],
    )
    accessible_price_signal = 1.0 - interpolate_signal(
        avg_price,
        ranges["avg_price"]["q25"],
        ranges["avg_price"]["q75"],
    )
    trendy_signal = interpolate_signal(
        trendy_item_ratio,
        ranges["trendy_item_ratio"]["q50"],
        ranges["trendy_item_ratio"]["q90"],
    )
    solid_signal = interpolate_signal(
        solid_share,
        ranges["solid_share"]["q25"],
        ranges["solid_share"]["q75"],
    )
    basic_signal = interpolate_signal(
        basic_share,
        ranges["basic_share"]["q25"],
        ranges["basic_share"]["q75"],
    )
    neutral_signal = interpolate_signal(
        neutral_color_share,
        ranges["neutral_color_share"]["q25"],
        ranges["neutral_color_share"]["q75"],
    )
    low_price_signal = interpolate_signal(
        low_price_ratio,
        ranges["low_price_ratio"]["q50"],
        ranges["low_price_ratio"]["q90"],
    )
    department_focus_signal = interpolate_signal(
        top_department_share,
        ranges["top_department_share"]["q25"],
        ranges["top_department_share"]["q75"],
    )
    repeat_signal = interpolate_signal(
        repeat_article_share,
        ranges["repeat_article_share"]["q25"],
        ranges["repeat_article_share"]["q75"],
    )
    color_focus_signal = interpolate_signal(
        top_color_share,
        ranges["top_color_share"]["q50"],
        ranges["top_color_share"]["q90"],
    )
    category_focus_signal = interpolate_signal(
        top_category_share,
        ranges["top_category_share"]["q50"],
        ranges["top_category_share"]["q90"],
    )
    binge_signal = interpolate_signal(
        same_day_multi_buy_avg,
        ranges["same_day_multi_buy_avg"]["q50"],
        ranges["same_day_multi_buy_avg"]["q90"],
    )
    gap_signal = interpolate_signal(
        avg_gap_days,
        ranges["avg_gap_days"]["q50"],
        ranges["avg_gap_days"]["q90"],
    )
    frequency_signal = interpolate_signal(
        purchase_count,
        ranges["purchase_count"]["q50"],
        ranges["purchase_count"]["q90"],
    )
    low_frequency_signal = 1.0 - interpolate_signal(
        purchase_count,
        ranges["purchase_count"]["q25"],
        ranges["purchase_count"]["q75"],
    )
    low_category_diversity_signal = 1.0 - interpolate_signal(
        unique_categories,
        ranges["unique_categories"]["q25"],
        ranges["unique_categories"]["q75"],
    )
    low_item_diversity_signal = 1.0 - interpolate_signal(
        unique_items,
        ranges["unique_items"]["q25"],
        ranges["unique_items"]["q75"],
    )

    scores = {
        "trendsetter": (
            0.30 * fresh_signal
            + 0.25 * unique_items_norm
            + 0.20 * unique_categories_norm
            + 0.25 * trendy_signal
        ),
        "practical": (
            0.30 * basic_signal
            + 0.24 * solid_signal
            + 0.18 * neutral_signal
            + 0.18 * repeat_signal
            + 0.10 * accessible_price_signal
        ),
        "value": (
            0.42 * (low_price_signal * accessible_price_signal)
            + 0.18 * low_price_signal
            + 0.15 * accessible_price_signal
            + 0.10 * frequency_signal
            + 0.15 * basic_signal
        ),
        "brand_loyal": (
            0.38 * department_focus_signal
            + 0.25 * repeat_signal
            + 0.22 * low_category_diversity_signal
            + 0.15 * frequency_signal
        ),
        "impulse": (
            0.45 * binge_signal
            + 0.18 * frequency_signal
            + 0.15 * fresh_signal
            + 0.22 * (1.0 - gap_signal)
        ),
        "careful": (
            0.40 * gap_signal
            + 0.20 * low_frequency_signal
            + 0.22 * (1.0 - binge_signal)
            + 0.18 * category_focus_signal
        ),
        "repeat_stable": (
            0.42 * repeat_signal
            + 0.20 * basic_signal
            + 0.10 * solid_signal
            + 0.14 * low_item_diversity_signal
            + 0.14 * low_category_diversity_signal
        ),
        "color_focus": (
            0.60 * color_focus_signal
            + 0.25 * neutral_signal
            + 0.15 * repeat_signal
        ),
        "category_focus": (
            0.55 * category_focus_signal
            + 0.25 * low_category_diversity_signal
            + 0.20 * frequency_signal
        ),
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
            "solid_share",
            "basic_share",
            "neutral_color_share",
            "low_price_ratio",
            "trendy_item_ratio",
            "top_department_share",
            "repeat_article_share",
            "top_color_share",
            "top_category_share",
        ],
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    persona_counts = {persona_name: 0 for persona_name in PERSONAS}

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
            persona_counts[top_persona] += 1

    elapsed = time.perf_counter() - start_time
    logging.info(
        "user_persona_scores_complete output=%s rows_written=%s elapsed_seconds=%.2f",
        output_path,
        rows_written,
        elapsed,
    )
    logging.info(
        "user_persona_top_distribution=%s",
        ", ".join(
            f"{persona_name}:{persona_counts[persona_name]}"
            for persona_name in sorted(persona_counts, key=persona_counts.get, reverse=True)
        ),
    )


def main() -> None:
    configure_logging()
    validate_required_columns(INPUT_FILE, REQUIRED_COLUMNS)
    rows = load_rows(INPUT_FILE)
    write_output(rows, OUTPUT_FILE)


if __name__ == "__main__":
    main()
