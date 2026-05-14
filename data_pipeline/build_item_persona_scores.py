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
        "INPUT_FILE": BASE_DIR / "data" / "processed" / "item_master_test.csv",
        "OUTPUT_FILE": BASE_DIR / "data" / "processed" / "item_persona_scores_test.csv",
    },
    "dev": {
        "INPUT_FILE": BASE_DIR / "data" / "processed" / "item_master_dev.csv",
        "OUTPUT_FILE": BASE_DIR / "data" / "processed" / "item_persona_scores_dev.csv",
    },
    "production": {
        "INPUT_FILE": BASE_DIR / "data" / "processed" / "item_master.csv",
        "OUTPUT_FILE": BASE_DIR / "data" / "processed" / "item_persona_scores.csv",
    },
}

RUNTIME_MODE = os.getenv("DATA_PIPELINE_MODE", MODE).strip().lower()

if RUNTIME_MODE not in MODE_CONFIG:
    raise ValueError(f"Unsupported MODE: {RUNTIME_MODE}")

CONFIG = MODE_CONFIG[RUNTIME_MODE]
INPUT_FILE: Path = CONFIG["INPUT_FILE"]
OUTPUT_FILE: Path = CONFIG["OUTPUT_FILE"]

REQUIRED_COLUMNS = [
    "article_id",
    "product_type_name",
    "product_group_name",
    "graphical_appearance_name",
    "colour_group_name",
    "perceived_colour_master_name",
    "department_name",
    "category_l1",
    "category_l2",
    "category_l3",
    "popularity",
    "price_mean",
    "price_bucket",
    "item_age_days",
    "is_new_item",
    "is_basic",
    "is_trendy",
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

OUTPUT_COLUMNS = ["article_id"]
for persona_name in PERSONAS:
    OUTPUT_COLUMNS.append(f"{persona_name}_score")
for persona_name in PERSONAS:
    OUTPUT_COLUMNS.append(f"{persona_name}_ratio")
OUTPUT_COLUMNS.extend(["top_persona", "top_persona_ratio"])

UNKNOWN_VALUE = "UNKNOWN"
NEUTRAL_COLORS = {"black", "white", "grey", "gray", "beige"}
ACCENT_COLORS = {"pink", "red", "orange", "yellow", "green", "purple", "turquoise"}
SPECIALIZED_CATEGORY_KEYWORDS = {
    "accessories",
    "lingerie",
    "tights",
    "sport",
    "kids",
    "baby",
    "divided",
    "underwear",
    "swimwear",
    "nightwear",
    "socks",
}
LOYAL_DEPARTMENT_KEYWORDS = {"divided", "young", "trend", "denim", "sport", "studio", "modern classic"}


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


def normalize_text(value: str) -> str:
    normalized = " ".join((value or "").strip().split())
    return normalized if normalized else UNKNOWN_VALUE


def parse_bool_text(raw_value: str) -> bool:
    return normalize_text(raw_value).lower() == "true"


def known_score(value: str) -> float:
    return 0.0 if normalize_text(value) == UNKNOWN_VALUE else 1.0


def contains_keyword(value: str, keywords: set[str]) -> bool:
    normalized = normalize_text(value).lower()
    return any(keyword in normalized for keyword in keywords)


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
        return [row for row in reader if (row.get("article_id") or "").strip()]


def build_feature_ranges(rows: list[dict[str, str]], feature_names: list[str]) -> Dict[str, tuple[float, float]]:
    ranges: Dict[str, tuple[float, float]] = {}
    for feature_name in feature_names:
        values = [parse_float(row.get(feature_name, "")) for row in rows]
        ranges[feature_name] = (min(values) if values else 0.0, max(values) if values else 0.0)
    return ranges


def compute_scores(row: dict[str, str], ranges: Dict[str, tuple[float, float]]) -> Dict[str, float]:
    popularity_norm = minmax_normalize(parse_float(row.get("popularity", "")), *ranges["popularity"])
    price_mean_norm = minmax_normalize(parse_float(row.get("price_mean", "")), *ranges["price_mean"])
    item_age_days_norm = minmax_normalize(parse_float(row.get("item_age_days", "")), *ranges["item_age_days"])

    is_new_item = parse_bool_text(row.get("is_new_item", ""))
    is_basic = parse_bool_text(row.get("is_basic", ""))
    is_trendy = parse_bool_text(row.get("is_trendy", ""))

    appearance = normalize_text(row.get("graphical_appearance_name", "")).lower()
    color = normalize_text(row.get("colour_group_name", "")).lower()
    perceived_color = normalize_text(row.get("perceived_colour_master_name", ""))
    department_name = row.get("department_name", "")
    category_l1 = row.get("category_l1", "")
    category_l2 = row.get("category_l2", "")
    category_l3 = row.get("category_l3", "")
    product_type_name = row.get("product_type_name", "")
    product_group_name = normalize_text(row.get("product_group_name", "")).lower()
    price_bucket = normalize_text(row.get("price_bucket", "")).lower()

    solid_score = 1.0 if appearance == "solid" else 0.0
    neutral_color_score = 1.0 if color in NEUTRAL_COLORS else 0.0
    accent_color_score = 1.0 if color in ACCENT_COLORS else 0.0
    color_known_score = known_score(row.get("colour_group_name", ""))
    perceived_color_known_score = known_score(perceived_color)
    department_known_score = known_score(department_name)
    low_price_score = 1.0 if price_bucket == "low" else 0.0
    high_price_score = 1.0 if price_bucket == "high" else 0.0
    low_mid_price_score = 1.0 if price_bucket in {"low", "mid"} else 0.0
    mid_high_price_score = 1.0 if price_bucket in {"mid", "high"} else 0.0
    low_price_norm = 1.0 - price_mean_norm
    older_item_score = item_age_days_norm
    core_apparel_score = 0.0 if "accessories" in product_group_name else 1.0
    specialized_category_score = 1.0 if any(
        contains_keyword(value, SPECIALIZED_CATEGORY_KEYWORDS)
        for value in (category_l1, category_l2, category_l3, product_group_name, product_type_name)
    ) else 0.20
    product_group_specificity_score = 1.0 if specialized_category_score >= 1.0 else 0.35
    department_identity_score = 1.0 if contains_keyword(department_name, LOYAL_DEPARTMENT_KEYWORDS) else 0.35
    mainstream_color_score = 1.0 if color_known_score and not accent_color_score else 0.0

    scores = {
        "trendsetter": (
            0.45 * (1.0 if is_trendy else 0.0)
            + 0.25 * (1.0 if is_new_item else 0.0)
            + 0.20 * popularity_norm
            + 0.10 * accent_color_score
        ),
        "practical": (
            0.30 * (1.0 if is_basic else 0.0)
            + 0.25 * solid_score
            + 0.20 * neutral_color_score
            + 0.15 * low_mid_price_score
            + 0.10 * core_apparel_score
        ),
        "value": (
            0.45 * low_price_score
            + 0.20 * low_price_norm
            + 0.20 * popularity_norm
            + 0.15 * (1.0 if is_basic else 0.0)
        ),
        "brand_loyal": (
            0.40 * department_identity_score
            + 0.20 * department_known_score
            + 0.20 * popularity_norm
            + 0.20 * mid_high_price_score
        ),
        "impulse": (
            0.35 * popularity_norm
            + 0.25 * (1.0 if is_new_item else 0.0)
            + 0.20 * (1.0 if is_trendy else 0.0)
            + 0.20 * accent_color_score
        ),
        "careful": (
            0.35 * mid_high_price_score
            + 0.25 * older_item_score
            + 0.20 * popularity_norm
            + 0.20 * core_apparel_score
        ),
        "repeat_stable": (
            0.35 * (1.0 if is_basic else 0.0)
            + 0.25 * solid_score
            + 0.20 * popularity_norm
            + 0.20 * low_mid_price_score
        ),
        "color_focus": (
            0.45 * accent_color_score
            + 0.25 * solid_score
            + 0.20 * perceived_color_known_score
            + 0.10 * mainstream_color_score
        ),
        "category_focus": (
            0.50 * specialized_category_score
            + 0.30 * product_group_specificity_score
            + 0.20 * department_identity_score
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
    feature_ranges = build_feature_ranges(rows, ["popularity", "price_mean", "item_age_days"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    persona_counts = {persona_name: 0 for persona_name in PERSONAS}

    with output_path.open("w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()

        for row in rows:
            article_id = (row.get("article_id") or "").strip()
            scores = compute_scores(row, feature_ranges)
            ratios = convert_scores_to_ratios(scores)
            top_persona = max(ratios, key=ratios.get)

            output_row = {"article_id": article_id}
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
        "item_persona_scores_complete output=%s rows_written=%s elapsed_seconds=%.2f",
        output_path,
        rows_written,
        elapsed,
    )
    logging.info(
        "item_persona_top_distribution=%s",
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
