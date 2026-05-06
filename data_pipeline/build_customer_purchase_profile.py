import csv
import logging
import os
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import DefaultDict, Dict, Optional, Sequence

BASE_DIR = Path(__file__).resolve().parent.parent
CUSTOMERS_FILE = BASE_DIR / "data" / "raw" / "customers.csv"
TRANSACTIONS_FILE = BASE_DIR / "data" / "raw" / "transactions_train.csv"

# MODE = "production"
MODE = "test"

MODE_CONFIG = {
    "test": {
        "MAX_TRANSACTION_ROWS": 100_000,
        "ITEM_MASTER_FILE": BASE_DIR / "data" / "processed" / "item_master_test.csv",
        "OUTPUT_FILE": BASE_DIR / "data" / "processed" / "customer_purchase_profile_test.csv",
        "LOG_EVERY_N_ROWS": 20_000,
    },
    "production": {
        "MAX_TRANSACTION_ROWS": None,
        "ITEM_MASTER_FILE": BASE_DIR / "data" / "processed" / "item_master.csv",
        "OUTPUT_FILE": BASE_DIR / "data" / "processed" / "customer_purchase_profile.csv",
        "LOG_EVERY_N_ROWS": 1_000_000,
    },
}

RUNTIME_MODE = os.getenv("DATA_PIPELINE_MODE", MODE).strip().lower()

if RUNTIME_MODE not in MODE_CONFIG:
    raise ValueError(f"Unsupported MODE: {RUNTIME_MODE}")

CONFIG = MODE_CONFIG[RUNTIME_MODE]
MAX_TRANSACTION_ROWS: Optional[int] = CONFIG["MAX_TRANSACTION_ROWS"]
ITEM_MASTER_FILE: Path = CONFIG["ITEM_MASTER_FILE"]
OUTPUT_FILE: Path = CONFIG["OUTPUT_FILE"]
LOG_EVERY_N_ROWS: int = CONFIG["LOG_EVERY_N_ROWS"]

UNKNOWN_VALUE = "UNKNOWN"
NEUTRAL_COLORS = {"black", "white", "grey", "gray", "beige"}

REQUIRED_CUSTOMER_COLUMNS = [
    "customer_id",
    "age",
    "fashion_news_frequency",
    "club_member_status",
]
REQUIRED_TRANSACTION_COLUMNS = ["customer_id", "article_id", "t_dat", "price"]
REQUIRED_ITEM_MASTER_COLUMNS = [
    "article_id",
    "category_l3",
    "colour_group_name",
    "department_name",
    "graphical_appearance_name",
    "price_bucket",
    "is_basic",
    "is_trendy",
]

OUTPUT_COLUMNS = [
    "customer_id",
    "age",
    "age_bucket",
    "fashion_news_frequency",
    "club_member_status",
    "purchase_count",
    "active_days",
    "first_purchase_date",
    "last_purchase_date",
    "recency_days",
    "avg_gap_days",
    "avg_price",
    "same_day_multi_buy_avg",
    "unique_items",
    "unique_categories",
    "top_category",
    "top_category_share",
    "top_color",
    "top_color_share",
    "top_department",
    "top_department_share",
    "solid_share",
    "basic_share",
    "neutral_color_share",
    "low_price_ratio",
    "repeat_article_share",
    "trendy_item_ratio",
]


@dataclass
class CustomerAggregate:
    purchase_count: int = 0
    price_sum: float = 0.0
    first_purchase_date: Optional[date] = None
    last_purchase_date: Optional[date] = None
    unique_items: set[str] = field(default_factory=set)
    unique_categories: set[str] = field(default_factory=set)
    purchase_dates: list[date] = field(default_factory=list)
    article_counts: Counter[str] = field(default_factory=Counter)
    category_counts: Counter[str] = field(default_factory=Counter)
    color_counts: Counter[str] = field(default_factory=Counter)
    department_counts: Counter[str] = field(default_factory=Counter)
    solid_count: int = 0
    basic_count: int = 0
    neutral_color_count: int = 0
    low_price_count: int = 0
    trendy_count: int = 0


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def log_stage(stage: str, start_time: float, **stats: int) -> None:
    elapsed = time.perf_counter() - start_time
    stats_text = " ".join(f"{key}={value}" for key, value in stats.items())
    message = f"stage={stage} elapsed_seconds={elapsed:.2f}"
    if stats_text:
        message = f"{message} {stats_text}"
    logging.info(message)


def resolve_required_file(file_path: Path, description: str) -> Path:
    if file_path.exists():
        return file_path
    raise FileNotFoundError(f"Missing {description}: {file_path}")


def validate_required_columns(file_path: Path, required_columns: Sequence[str]) -> None:
    with file_path.open(newline="", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames or []

    missing_columns = [column for column in required_columns if column not in fieldnames]
    if missing_columns:
        raise ValueError(
            f"Missing required columns in {file_path}: {', '.join(missing_columns)}"
        )


def normalize_text(value: str) -> str:
    normalized = " ".join((value or "").strip().split())
    return normalized if normalized else UNKNOWN_VALUE


def normalize_fashion_news_frequency(value: str) -> str:
    normalized = normalize_text(value).upper()
    aliases = {
        "NONE": "NONE",
        "REGULARLY": "REGULARLY",
        "MONTHLY": "MONTHLY",
    }
    return aliases.get(normalized, UNKNOWN_VALUE)


def normalize_club_member_status(value: str) -> str:
    normalized = normalize_text(value).upper()
    aliases = {
        "ACTIVE": "ACTIVE",
        "PRE-CREATE": "PRE-CREATE",
        "LEFT CLUB": "LEFT CLUB",
    }
    return aliases.get(normalized, normalized if normalized != UNKNOWN_VALUE else UNKNOWN_VALUE)


def parse_age(raw_value: str) -> int:
    value = (raw_value or "").strip()
    if not value:
        return -1

    try:
        age = int(float(value))
    except ValueError:
        return -1

    if age < 0 or age > 120:
        return -1
    return age


def make_age_bucket(age: int) -> str:
    if age < 0:
        return "unknown"
    if age < 20:
        return "under_20"
    if age < 30:
        return "20s"
    if age < 40:
        return "30s"
    if age < 50:
        return "40s"
    if age < 60:
        return "50s"
    return "60_plus"


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


def parse_bool_text(raw_value: str) -> bool:
    return normalize_text(raw_value).lower() == "true"


def load_customer_metadata(customers_path: Path) -> Dict[str, dict[str, str]]:
    start_time = time.perf_counter()
    customers: Dict[str, dict[str, str]] = {}
    rows_loaded = 0

    with customers_path.open(newline="", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            customer_id = (row.get("customer_id") or "").strip()
            if not customer_id:
                continue
            customers[customer_id] = {
                "customer_id": customer_id,
                "age": str(parse_age(row.get("age", ""))),
                "age_bucket": make_age_bucket(parse_age(row.get("age", ""))),
                "fashion_news_frequency": normalize_fashion_news_frequency(
                    row.get("fashion_news_frequency", "")
                ),
                "club_member_status": normalize_club_member_status(
                    row.get("club_member_status", "")
                ),
            }
            rows_loaded += 1

    log_stage("load_customer_metadata", start_time, rows_loaded=rows_loaded)
    return customers


def load_item_master(item_master_path: Path) -> Dict[str, dict[str, object]]:
    start_time = time.perf_counter()
    item_master: Dict[str, dict[str, object]] = {}
    rows_loaded = 0

    with item_master_path.open(newline="", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            article_id = (row.get("article_id") or "").strip()
            if not article_id:
                continue

            color = normalize_text(row.get("colour_group_name", ""))
            item_master[article_id] = {
                "category_l3": normalize_text(row.get("category_l3", "")),
                "colour_group_name": color,
                "department_name": normalize_text(row.get("department_name", "")),
                "graphical_appearance_name": normalize_text(row.get("graphical_appearance_name", "")),
                "price_bucket": normalize_text(row.get("price_bucket", "")).lower(),
                "is_basic": parse_bool_text(row.get("is_basic", "")),
                "is_trendy": parse_bool_text(row.get("is_trendy", "")),
                "is_neutral_color": color.lower() in NEUTRAL_COLORS,
            }
            rows_loaded += 1

    log_stage("load_item_master", start_time, rows_loaded=rows_loaded)
    return item_master


def collect_customer_aggregates(
    transactions_path: Path,
    item_master: Dict[str, dict[str, object]],
) -> tuple[Dict[str, CustomerAggregate], Optional[date], Dict[str, int]]:
    start_time = time.perf_counter()
    aggregates: Dict[str, CustomerAggregate] = {}
    dataset_max_date: Optional[date] = None
    stats = {
        "rows_scanned": 0,
        "rows_aggregated": 0,
        "invalid_date_rows": 0,
        "missing_customer_rows": 0,
        "missing_article_rows": 0,
    }

    with transactions_path.open(newline="", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            if MAX_TRANSACTION_ROWS is not None and stats["rows_scanned"] >= MAX_TRANSACTION_ROWS:
                break

            stats["rows_scanned"] += 1
            customer_id = (row.get("customer_id") or "").strip()
            article_id = (row.get("article_id") or "").strip()

            if not customer_id:
                stats["missing_customer_rows"] += 1
                continue
            if not article_id:
                stats["missing_article_rows"] += 1
                continue

            transaction_date = parse_transaction_date(row.get("t_dat", ""))
            if transaction_date is None:
                stats["invalid_date_rows"] += 1
                continue

            item = item_master.get(article_id)
            if item is None:
                continue

            aggregate = aggregates.setdefault(customer_id, CustomerAggregate())
            aggregate.purchase_count += 1
            aggregate.price_sum += parse_price(row.get("price", ""))
            aggregate.unique_items.add(article_id)
            aggregate.article_counts[article_id] += 1
            aggregate.purchase_dates.append(transaction_date)

            category_l3 = str(item["category_l3"])
            color = str(item["colour_group_name"])
            department = str(item["department_name"])
            appearance = str(item["graphical_appearance_name"])

            aggregate.unique_categories.add(category_l3)
            aggregate.category_counts[category_l3] += 1
            aggregate.color_counts[color] += 1
            aggregate.department_counts[department] += 1

            if appearance.lower() == "solid":
                aggregate.solid_count += 1
            if bool(item["is_basic"]):
                aggregate.basic_count += 1
            if bool(item["is_neutral_color"]):
                aggregate.neutral_color_count += 1
            if str(item["price_bucket"]) == "low":
                aggregate.low_price_count += 1
            if bool(item["is_trendy"]):
                aggregate.trendy_count += 1

            if aggregate.first_purchase_date is None or transaction_date < aggregate.first_purchase_date:
                aggregate.first_purchase_date = transaction_date
            if aggregate.last_purchase_date is None or transaction_date > aggregate.last_purchase_date:
                aggregate.last_purchase_date = transaction_date
            if dataset_max_date is None or transaction_date > dataset_max_date:
                dataset_max_date = transaction_date

            stats["rows_aggregated"] += 1
            if stats["rows_scanned"] % LOG_EVERY_N_ROWS == 0:
                logging.info(
                    "stage=customer_purchase_profile_progress rows_scanned=%s rows_aggregated=%s unique_customers=%s",
                    stats["rows_scanned"],
                    stats["rows_aggregated"],
                    len(aggregates),
                )

    log_stage("collect_customer_purchase_profile", start_time, **stats)
    return aggregates, dataset_max_date, stats


def get_top_value(counter: Counter[str]) -> tuple[str, float]:
    if not counter:
        return UNKNOWN_VALUE, 0.0
    top_value, top_count = counter.most_common(1)[0]
    total = sum(counter.values())
    share = top_count / total if total else 0.0
    return top_value, share


def compute_avg_gap_days(purchase_dates: list[date]) -> str:
    unique_dates = sorted(set(purchase_dates))
    if len(unique_dates) < 2:
        return ""
    gaps = []
    for index in range(1, len(unique_dates)):
        gaps.append((unique_dates[index] - unique_dates[index - 1]).days)
    return f"{(sum(gaps) / len(gaps)):.4f}"


def compute_same_day_multi_buy_avg(purchase_dates: list[date]) -> str:
    if not purchase_dates:
        return "0.0000"
    day_counter = Counter(purchase_dates)
    return f"{(sum(day_counter.values()) / len(day_counter)):.4f}"


def compute_repeat_article_share(article_counts: Counter[str], purchase_count: int) -> str:
    if purchase_count <= 0:
        return "0.000000"
    repeated_rows = sum(count - 1 for count in article_counts.values() if count > 1)
    return f"{(repeated_rows / purchase_count):.6f}"


def write_customer_purchase_profile(
    customers: Dict[str, dict[str, str]],
    aggregates: Dict[str, CustomerAggregate],
    dataset_max_date: Optional[date],
    output_path: Path,
) -> None:
    start_time = time.perf_counter()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0

    with output_path.open("w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()

        for customer_id in sorted(customers):
            customer = customers[customer_id]
            aggregate = aggregates.get(customer_id, CustomerAggregate())
            purchase_count = aggregate.purchase_count
            active_days = len(set(aggregate.purchase_dates))
            avg_price = aggregate.price_sum / purchase_count if purchase_count else 0.0

            top_category, top_category_share = get_top_value(aggregate.category_counts)
            top_color, top_color_share = get_top_value(aggregate.color_counts)
            top_department, top_department_share = get_top_value(aggregate.department_counts)

            recency_days = ""
            if dataset_max_date is not None and aggregate.last_purchase_date is not None:
                recency_days = str((dataset_max_date - aggregate.last_purchase_date).days)

            writer.writerow(
                {
                    "customer_id": customer_id,
                    "age": customer["age"],
                    "age_bucket": customer["age_bucket"],
                    "fashion_news_frequency": customer["fashion_news_frequency"],
                    "club_member_status": customer["club_member_status"],
                    "purchase_count": purchase_count,
                    "active_days": active_days,
                    "first_purchase_date": aggregate.first_purchase_date.isoformat() if aggregate.first_purchase_date else "",
                    "last_purchase_date": aggregate.last_purchase_date.isoformat() if aggregate.last_purchase_date else "",
                    "recency_days": recency_days,
                    "avg_gap_days": compute_avg_gap_days(aggregate.purchase_dates),
                    "avg_price": f"{avg_price:.10f}",
                    "same_day_multi_buy_avg": compute_same_day_multi_buy_avg(aggregate.purchase_dates),
                    "unique_items": len(aggregate.unique_items),
                    "unique_categories": len(aggregate.unique_categories),
                    "top_category": top_category,
                    "top_category_share": f"{top_category_share:.6f}",
                    "top_color": top_color,
                    "top_color_share": f"{top_color_share:.6f}",
                    "top_department": top_department,
                    "top_department_share": f"{top_department_share:.6f}",
                    "solid_share": f"{(aggregate.solid_count / purchase_count):.6f}" if purchase_count else "0.000000",
                    "basic_share": f"{(aggregate.basic_count / purchase_count):.6f}" if purchase_count else "0.000000",
                    "neutral_color_share": f"{(aggregate.neutral_color_count / purchase_count):.6f}" if purchase_count else "0.000000",
                    "low_price_ratio": f"{(aggregate.low_price_count / purchase_count):.6f}" if purchase_count else "0.000000",
                    "repeat_article_share": compute_repeat_article_share(aggregate.article_counts, purchase_count),
                    "trendy_item_ratio": f"{(aggregate.trendy_count / purchase_count):.6f}" if purchase_count else "0.000000",
                }
            )
            rows_written += 1

    log_stage("write_customer_purchase_profile", start_time, rows_written=rows_written)


def main() -> None:
    configure_logging()
    customers_path = resolve_required_file(CUSTOMERS_FILE, "customers raw file")
    transactions_path = resolve_required_file(TRANSACTIONS_FILE, "transactions raw file")
    item_master_path = resolve_required_file(ITEM_MASTER_FILE, "item master file")
    validate_required_columns(customers_path, REQUIRED_CUSTOMER_COLUMNS)
    validate_required_columns(transactions_path, REQUIRED_TRANSACTION_COLUMNS)
    validate_required_columns(item_master_path, REQUIRED_ITEM_MASTER_COLUMNS)

    logging.info(
        "mode=%s customers_file=%s transactions_file=%s item_master_file=%s output_file=%s max_transaction_rows=%s",
        RUNTIME_MODE,
        customers_path,
        transactions_path,
        item_master_path,
        OUTPUT_FILE,
        MAX_TRANSACTION_ROWS,
    )

    customers = load_customer_metadata(customers_path)
    item_master = load_item_master(item_master_path)
    aggregates, dataset_max_date, _ = collect_customer_aggregates(transactions_path, item_master)
    write_customer_purchase_profile(customers, aggregates, dataset_max_date, OUTPUT_FILE)


if __name__ == "__main__":
    main()
