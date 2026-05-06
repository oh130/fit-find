import csv
import logging
import os
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, Optional, Sequence

BASE_DIR = Path(__file__).resolve().parent.parent
ARTICLES_FILE = BASE_DIR / "data" / "raw" / "articles.csv"
TRANSACTIONS_FILE = BASE_DIR / "data" / "raw" / "transactions_train.csv"

# MODE = "production"
MODE = "test"

MODE_CONFIG = {
    "test": {
        "MAX_TRANSACTION_ROWS": 100_000,
        "OUTPUT_FILE": BASE_DIR / "data" / "processed" / "item_master_test.csv",
        "LOG_EVERY_N_ROWS": 20_000,
        "NEW_ITEM_WINDOW_DAYS": 7,
    },
    "production": {
        "MAX_TRANSACTION_ROWS": None,
        "OUTPUT_FILE": BASE_DIR / "data" / "processed" / "item_master.csv",
        "LOG_EVERY_N_ROWS": 1_000_000,
        "NEW_ITEM_WINDOW_DAYS": 7,
    },
}

RUNTIME_MODE = os.getenv("DATA_PIPELINE_MODE", MODE).strip().lower()

if RUNTIME_MODE not in MODE_CONFIG:
    raise ValueError(f"Unsupported MODE: {RUNTIME_MODE}")

CONFIG = MODE_CONFIG[RUNTIME_MODE]
MAX_TRANSACTION_ROWS: Optional[int] = CONFIG["MAX_TRANSACTION_ROWS"]
OUTPUT_FILE: Path = CONFIG["OUTPUT_FILE"]
LOG_EVERY_N_ROWS: int = CONFIG["LOG_EVERY_N_ROWS"]
NEW_ITEM_WINDOW_DAYS: int = CONFIG["NEW_ITEM_WINDOW_DAYS"]

REQUIRED_ARTICLE_COLUMNS = [
    "article_id",
    "product_code",
    "prod_name",
    "product_type_name",
    "product_group_name",
    "graphical_appearance_name",
    "colour_group_name",
    "perceived_colour_master_name",
    "department_name",
    "index_name",
    "index_group_name",
    "section_name",
    "garment_group_name",
    "detail_desc",
]
REQUIRED_TRANSACTION_COLUMNS = ["t_dat", "article_id", "price"]

OUTPUT_COLUMNS = [
    "article_id",
    "product_code",
    "prod_name",
    "product_type_name",
    "product_group_name",
    "graphical_appearance_name",
    "colour_group_name",
    "perceived_colour_master_name",
    "department_name",
    "index_name",
    "index_group_name",
    "section_name",
    "garment_group_name",
    "detail_desc",
    "category_l1",
    "category_l2",
    "category_l3",
    "popularity",
    "price_mean",
    "price_bucket",
    "first_purchase_date",
    "last_purchase_date",
    "item_age_days",
    "is_new_item",
    "is_basic",
    "is_trendy",
]

UNKNOWN_VALUE = "UNKNOWN"
BASIC_SECTION_KEYWORDS = ("basic", "everyday basics")
BASIC_GARMENT_KEYWORDS = ("jersey basic", "trousers", "shirts", "knitwear")
TREND_SECTION_KEYWORDS = ("trend", "divided", "projects")
TREND_DEPARTMENT_KEYWORDS = ("trend", "contemporary", "young")


@dataclass
class ItemAggregate:
    popularity: int = 0
    price_sum: float = 0.0
    first_purchase_date: Optional[date] = None
    last_purchase_date: Optional[date] = None


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def log_stage(stage: str, start_time: float, **stats: int) -> None:
    elapsed = time.perf_counter() - start_time
    stats_text = " ".join(f"{key}={value}" for key, value in stats.items())
    message = f"stage={stage} elapsed_seconds={elapsed:.2f}"
    if stats_text:
        message = f"{message} {stats_text}"
    logging.info(message)


def validate_required_columns(file_path: Path, required_columns: Sequence[str]) -> None:
    with file_path.open(newline="", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames or []

    missing_columns = [column for column in required_columns if column not in fieldnames]
    if missing_columns:
        raise ValueError(
            f"Missing required columns in {file_path}: {', '.join(missing_columns)}"
        )


def resolve_required_file(file_path: Path, description: str) -> Path:
    if file_path.exists():
        return file_path
    raise FileNotFoundError(f"Missing {description}: {file_path}")


def normalize_text(value: str) -> str:
    normalized = " ".join((value or "").strip().split())
    return normalized if normalized else UNKNOWN_VALUE


def parse_transaction_date(raw_value: str) -> Optional[date]:
    value = (raw_value or "").strip()
    if not value:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def parse_price(raw_value: str) -> float:
    value = (raw_value or "").strip()
    if not value:
        return 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0


def choose_first_known(*values: str) -> str:
    for value in values:
        normalized = normalize_text(value)
        if normalized != UNKNOWN_VALUE:
            return normalized
    return UNKNOWN_VALUE


def derive_category_l1(row: dict[str, str]) -> str:
    return choose_first_known(
        row.get("index_group_name", ""),
        row.get("index_name", ""),
        row.get("product_group_name", ""),
    )


def derive_category_l2(row: dict[str, str]) -> str:
    return choose_first_known(
        row.get("section_name", ""),
        row.get("department_name", ""),
        row.get("product_group_name", ""),
    )


def derive_category_l3(row: dict[str, str]) -> str:
    return choose_first_known(
        row.get("product_type_name", ""),
        row.get("product_group_name", ""),
        row.get("prod_name", ""),
    )


def is_basic_item(row: dict[str, str]) -> bool:
    appearance = normalize_text(row.get("graphical_appearance_name", "")).lower()
    section_name = normalize_text(row.get("section_name", "")).lower()
    garment_group_name = normalize_text(row.get("garment_group_name", "")).lower()

    if appearance == "solid":
        return True
    if any(keyword in section_name for keyword in BASIC_SECTION_KEYWORDS):
        return True
    if any(keyword in garment_group_name for keyword in BASIC_GARMENT_KEYWORDS):
        return True
    return False


def is_trendy_item(row: dict[str, str]) -> bool:
    section_name = normalize_text(row.get("section_name", "")).lower()
    department_name = normalize_text(row.get("department_name", "")).lower()

    if any(keyword in section_name for keyword in TREND_SECTION_KEYWORDS):
        return True
    if any(keyword in department_name for keyword in TREND_DEPARTMENT_KEYWORDS):
        return True
    return False


def collect_item_aggregates(
    transactions_path: Path,
) -> tuple[Dict[str, ItemAggregate], Optional[date], Dict[str, int]]:
    start_time = time.perf_counter()
    item_aggregates: Dict[str, ItemAggregate] = {}
    dataset_max_date: Optional[date] = None
    stats = {
        "rows_scanned": 0,
        "rows_aggregated": 0,
        "invalid_date_rows": 0,
        "missing_article_rows": 0,
    }

    with transactions_path.open(newline="", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            if MAX_TRANSACTION_ROWS is not None and stats["rows_scanned"] >= MAX_TRANSACTION_ROWS:
                break

            stats["rows_scanned"] += 1
            article_id = (row.get("article_id") or "").strip()
            if not article_id:
                stats["missing_article_rows"] += 1
                continue

            transaction_date = parse_transaction_date(row.get("t_dat", ""))
            if transaction_date is None:
                stats["invalid_date_rows"] += 1
                continue

            aggregate = item_aggregates.setdefault(article_id, ItemAggregate())
            aggregate.popularity += 1
            aggregate.price_sum += parse_price(row.get("price", ""))
            if aggregate.first_purchase_date is None or transaction_date < aggregate.first_purchase_date:
                aggregate.first_purchase_date = transaction_date
            if aggregate.last_purchase_date is None or transaction_date > aggregate.last_purchase_date:
                aggregate.last_purchase_date = transaction_date
            if dataset_max_date is None or transaction_date > dataset_max_date:
                dataset_max_date = transaction_date

            stats["rows_aggregated"] += 1
            if stats["rows_scanned"] % LOG_EVERY_N_ROWS == 0:
                logging.info(
                    "stage=item_master_progress rows_scanned=%s rows_aggregated=%s unique_articles=%s invalid_date_rows=%s missing_article_rows=%s",
                    stats["rows_scanned"],
                    stats["rows_aggregated"],
                    len(item_aggregates),
                    stats["invalid_date_rows"],
                    stats["missing_article_rows"],
                )

    log_stage("collect_item_master_aggregates", start_time, **stats)
    return item_aggregates, dataset_max_date, stats


def compute_percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = int((len(sorted_values) - 1) * percentile)
    return sorted_values[index]


def build_price_bucket_map(item_aggregates: Dict[str, ItemAggregate]) -> tuple[Dict[str, str], float, float]:
    avg_prices: list[float] = []
    avg_price_by_article: Dict[str, float] = {}

    for article_id, aggregate in item_aggregates.items():
        if aggregate.popularity <= 0:
            continue
        avg_price = aggregate.price_sum / aggregate.popularity
        avg_prices.append(avg_price)
        avg_price_by_article[article_id] = avg_price

    low_threshold = compute_percentile(avg_prices, 0.33)
    high_threshold = compute_percentile(avg_prices, 0.66)

    bucket_map: Dict[str, str] = {}
    for article_id, avg_price in avg_price_by_article.items():
        if avg_price <= low_threshold:
            bucket_map[article_id] = "low"
        elif avg_price <= high_threshold:
            bucket_map[article_id] = "mid"
        else:
            bucket_map[article_id] = "high"

    return bucket_map, low_threshold, high_threshold


def write_item_master(
    articles_path: Path,
    item_aggregates: Dict[str, ItemAggregate],
    dataset_max_date: Optional[date],
    output_path: Path,
) -> None:
    start_time = time.perf_counter()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    price_bucket_map, low_threshold, high_threshold = build_price_bucket_map(item_aggregates)

    rows_written = 0
    with articles_path.open(newline="", encoding="utf-8") as infile, output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as outfile:
        reader = csv.DictReader(infile)
        writer = csv.DictWriter(outfile, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()

        for row in reader:
            article_id = (row.get("article_id") or "").strip()
            if not article_id:
                continue

            aggregate = item_aggregates.get(article_id, ItemAggregate())
            avg_price = aggregate.price_sum / aggregate.popularity if aggregate.popularity else 0.0
            first_purchase_date = aggregate.first_purchase_date
            last_purchase_date = aggregate.last_purchase_date
            item_age_days = ""
            is_new_item = "False"

            if dataset_max_date is not None and last_purchase_date is not None:
                age_days = (dataset_max_date - last_purchase_date).days
                item_age_days = str(age_days)
                is_new_item = "True" if age_days <= NEW_ITEM_WINDOW_DAYS else "False"

            writer.writerow(
                {
                    "article_id": article_id,
                    "product_code": normalize_text(row.get("product_code", "")),
                    "prod_name": normalize_text(row.get("prod_name", "")),
                    "product_type_name": normalize_text(row.get("product_type_name", "")),
                    "product_group_name": normalize_text(row.get("product_group_name", "")),
                    "graphical_appearance_name": normalize_text(row.get("graphical_appearance_name", "")),
                    "colour_group_name": normalize_text(row.get("colour_group_name", "")),
                    "perceived_colour_master_name": normalize_text(row.get("perceived_colour_master_name", "")),
                    "department_name": normalize_text(row.get("department_name", "")),
                    "index_name": normalize_text(row.get("index_name", "")),
                    "index_group_name": normalize_text(row.get("index_group_name", "")),
                    "section_name": normalize_text(row.get("section_name", "")),
                    "garment_group_name": normalize_text(row.get("garment_group_name", "")),
                    "detail_desc": normalize_text(row.get("detail_desc", "")),
                    "category_l1": derive_category_l1(row),
                    "category_l2": derive_category_l2(row),
                    "category_l3": derive_category_l3(row),
                    "popularity": aggregate.popularity,
                    "price_mean": f"{avg_price:.10f}" if aggregate.popularity else "0.0000000000",
                    "price_bucket": price_bucket_map.get(article_id, "unknown"),
                    "first_purchase_date": first_purchase_date.isoformat() if first_purchase_date else "",
                    "last_purchase_date": last_purchase_date.isoformat() if last_purchase_date else "",
                    "item_age_days": item_age_days,
                    "is_new_item": is_new_item,
                    "is_basic": "True" if is_basic_item(row) else "False",
                    "is_trendy": "True" if is_trendy_item(row) else "False",
                }
            )
            rows_written += 1

    log_stage(
        "write_item_master",
        start_time,
        rows_written=rows_written,
        low_price_threshold=int(low_threshold * 1_000_000),
        high_price_threshold=int(high_threshold * 1_000_000),
    )


def main() -> None:
    configure_logging()
    articles_path = resolve_required_file(ARTICLES_FILE, "articles raw file")
    transactions_path = resolve_required_file(TRANSACTIONS_FILE, "transactions raw file")
    validate_required_columns(articles_path, REQUIRED_ARTICLE_COLUMNS)
    validate_required_columns(transactions_path, REQUIRED_TRANSACTION_COLUMNS)

    logging.info(
        "mode=%s articles_file=%s transactions_file=%s output_file=%s max_transaction_rows=%s",
        RUNTIME_MODE,
        articles_path,
        transactions_path,
        OUTPUT_FILE,
        MAX_TRANSACTION_ROWS,
    )

    item_aggregates, dataset_max_date, _ = collect_item_aggregates(transactions_path)
    write_item_master(
        articles_path=articles_path,
        item_aggregates=item_aggregates,
        dataset_max_date=dataset_max_date,
        output_path=OUTPUT_FILE,
    )


if __name__ == "__main__":
    main()
