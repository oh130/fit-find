import csv
from datetime import date
import hashlib
import logging
import os
import random
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, TextIO, Tuple

BASE_DIR = Path(__file__).resolve().parent.parent

TRANSACTIONS_FILE = BASE_DIR / "data" / "raw" / "transactions_train.csv"
CUSTOMER_FEATURES_FILE = BASE_DIR / "data" / "processed" / "customer_features.csv"
ARTICLE_FEATURES_FILE = BASE_DIR / "data" / "processed" / "articles_feature.csv"

# MODE = "production"
# MODE = "dev"
MODE = "test"

MODE_CONFIG = {
    "test": {
        "MAX_TRANSACTION_ROWS": 10_000,
        "NEGATIVE_RATIO": 1,
        "OUTPUT_FILE": BASE_DIR / "data" / "processed" / "train_data_test.csv",
        "RANDOM_SEED": 42,
        "LOG_EVERY_N_ROWS": 2_000,
        "PARTITION_COUNT": 16,
    },
    "dev": {
        "MAX_TRANSACTION_ROWS": 1_000_000,
        "NEGATIVE_RATIO": 1,
        "OUTPUT_FILE": BASE_DIR / "data" / "processed" / "train_data_dev.csv",
        "RANDOM_SEED": 42,
        "LOG_EVERY_N_ROWS": 100_000,
        "PARTITION_COUNT": 64,
    },
    "production": {
        "MAX_TRANSACTION_ROWS": None,
        "NEGATIVE_RATIO": 1,
        "OUTPUT_FILE": BASE_DIR / "data" / "processed" / "train_data_production.csv",
        "RANDOM_SEED": 42,
        "LOG_EVERY_N_ROWS": 100_000,
        "PARTITION_COUNT": 256,
    },
}

RUNTIME_MODE = os.getenv("DATA_PIPELINE_MODE", MODE).strip().lower()

if RUNTIME_MODE not in MODE_CONFIG:
    raise ValueError(f"Unsupported MODE: {RUNTIME_MODE}")

CONFIG = MODE_CONFIG[RUNTIME_MODE]
MAX_TRANSACTION_ROWS: Optional[int] = CONFIG["MAX_TRANSACTION_ROWS"]
NEGATIVE_RATIO: int = CONFIG["NEGATIVE_RATIO"]
OUTPUT_FILE: Path = CONFIG["OUTPUT_FILE"]
RANDOM_SEED: int = CONFIG["RANDOM_SEED"]
LOG_EVERY_N_ROWS: int = CONFIG["LOG_EVERY_N_ROWS"]
PARTITION_COUNT: int = CONFIG["PARTITION_COUNT"]

OUTPUT_COLUMNS = [
    "customer_id",
    "article_id",
    "label",
    "price",
    "sales_channel_id",
    "age",
    "age_bucket",
    "fashion_news_frequency",
    "club_member_status",
    "prod_name",
    "product_type_name",
    "product_group_name",
    "colour_group_name",
    "perceived_colour_master_name",
    "department_name",
    "section_name",
    "garment_group_name",
    "category",
    "main_category",
    "color",
    "age_category",
    "age_color",
    "member_category",
    "fashion_category",
    "days_since_last_purchase",
    "last_purchase_category_match",
    "last_purchase_main_category_match",
    "last_purchase_color_match",
    "recent_3_purchase_category_count",
    "recent_3_purchase_main_category_count",
    "recent_3_purchase_color_count",
    "recent_3_unique_category_count",
    "days_since_last_same_category_purchase",
    "recent_5_purchase_main_category_count",
]

CUSTOMER_FEATURE_COLUMNS = [
    "age",
    "age_bucket",
    "fashion_news_frequency",
    "club_member_status",
]

ARTICLE_FEATURE_COLUMNS = [
    "prod_name",
    "product_type_name",
    "product_group_name",
    "colour_group_name",
    "perceived_colour_master_name",
    "department_name",
    "section_name",
    "garment_group_name",
    "category",
    "main_category",
    "color",
]

PARTITION_COLUMNS = [
    "t_dat",
    "customer_id",
    "article_id",
    "price",
    "sales_channel_id",
]
TRANSACTION_REQUIRED_COLUMNS = PARTITION_COLUMNS

StatsDict = Dict[str, int]
CustomerFeature = Dict[str, str]
ArticleFeature = Dict[str, str]


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )


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


def validate_required_columns(
    file_path: Path,
    required_columns: Sequence[str],
    key_column: str,
) -> None:
    with file_path.open(newline="", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames or []

    missing_columns = [column for column in required_columns if column not in fieldnames]
    if key_column not in fieldnames:
        missing_columns.insert(0, key_column)

    if missing_columns:
        raise ValueError(
            f"Missing required columns in {file_path}: {', '.join(dict.fromkeys(missing_columns))}"
        )


def load_customer_features(file_path: Path) -> Dict[str, CustomerFeature]:
    customer_features: Dict[str, CustomerFeature] = {}
    with file_path.open(newline="", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            customer_id = row["customer_id"].strip()
            if not customer_id:
                continue
            customer_features[customer_id] = {
                column: row[column].strip()
                for column in CUSTOMER_FEATURE_COLUMNS
            }
    return customer_features


def load_article_features(file_path: Path) -> Dict[str, ArticleFeature]:
    article_features: Dict[str, ArticleFeature] = {}
    with file_path.open(newline="", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            article_id = row["article_id"].strip()
            if not article_id:
                continue
            article_features[article_id] = {
                column: row[column].strip()
                for column in ARTICLE_FEATURE_COLUMNS
            }
    return article_features


def stable_partition_index(customer_id: str, partition_count: int) -> int:
    digest = hashlib.blake2b(customer_id.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big") % partition_count


def open_partition_writers(temp_dir: Path, partition_count: int) -> List[Tuple[TextIO, csv.DictWriter]]:
    writers: List[Tuple[TextIO, csv.DictWriter]] = []
    for index in range(partition_count):
        partition_path = temp_dir / f"transactions_partition_{index:04d}.csv"
        handle = partition_path.open("w", newline="", encoding="utf-8")
        writer = csv.DictWriter(handle, fieldnames=PARTITION_COLUMNS)
        writer.writeheader()
        writers.append((handle, writer))
    return writers


def partition_transactions(
    transactions_path: Path,
    temp_dir: Path,
    customer_features: Dict[str, CustomerFeature],
    article_features: Dict[str, ArticleFeature],
) -> StatsDict:
    start_time = time.perf_counter()
    writers = open_partition_writers(temp_dir, PARTITION_COUNT)
    stats: StatsDict = {
        "rows_scanned": 0,
        "rows_partitioned": 0,
        "missing_customer_rows": 0,
        "missing_article_rows": 0,
    }

    try:
        with transactions_path.open(newline="", encoding="utf-8") as infile:
            reader = csv.DictReader(infile)
            for row in reader:
                if MAX_TRANSACTION_ROWS is not None and stats["rows_scanned"] >= MAX_TRANSACTION_ROWS:
                    break

                stats["rows_scanned"] += 1
                customer_id = row["customer_id"].strip()
                article_id = row["article_id"].strip()

                if customer_id not in customer_features:
                    stats["missing_customer_rows"] += 1
                    continue
                if article_id not in article_features:
                    stats["missing_article_rows"] += 1
                    continue

                partition_index = stable_partition_index(customer_id, PARTITION_COUNT)
                _, writer = writers[partition_index]
                writer.writerow(
                    {
                        "t_dat": row["t_dat"].strip(),
                        "customer_id": customer_id,
                        "article_id": article_id,
                        "price": row["price"].strip(),
                        "sales_channel_id": row["sales_channel_id"].strip(),
                    }
                )
                stats["rows_partitioned"] += 1

                if stats["rows_scanned"] % LOG_EVERY_N_ROWS == 0:
                    logging.info(
                        "stage=partition_progress rows_scanned=%s rows_partitioned=%s missing_customer_rows=%s missing_article_rows=%s",
                        stats["rows_scanned"],
                        stats["rows_partitioned"],
                        stats["missing_customer_rows"],
                        stats["missing_article_rows"],
                    )
    finally:
        for handle, _ in writers:
            handle.close()

    log_stage("partition_transactions", start_time, **stats)
    return stats


def parse_transaction_date(raw_value: str) -> date:
    return date.fromisoformat(raw_value.strip())


def build_purchase_history_features(
    *,
    article_id: str,
    transaction_date: date,
    history_article_ids: Sequence[str],
    history_dates: Sequence[date],
    article_features: Dict[str, ArticleFeature],
) -> Dict[str, str]:
    if not history_article_ids or not history_dates:
        return {
            "days_since_last_purchase": "",
            "last_purchase_category_match": "0",
            "last_purchase_main_category_match": "0",
            "last_purchase_color_match": "0",
            "recent_3_purchase_category_count": "0",
            "recent_3_purchase_main_category_count": "0",
            "recent_3_purchase_color_count": "0",
            "recent_3_unique_category_count": "0",
            "days_since_last_same_category_purchase": "",
            "recent_5_purchase_main_category_count": "0",
        }

    current_article = article_features[article_id]
    last_article_id = history_article_ids[-1]
    last_article = article_features.get(last_article_id)
    days_since_last_purchase = (transaction_date - history_dates[-1]).days

    recent_article_ids = history_article_ids[-3:]
    recent_articles = [article_features[item_id] for item_id in recent_article_ids if item_id in article_features]
    recent_5_article_ids = history_article_ids[-5:]
    recent_5_articles = [article_features[item_id] for item_id in recent_5_article_ids if item_id in article_features]

    current_category = current_article["category"]
    current_main_category = current_article["main_category"]
    current_color = current_article["color"]
    recent_category_count = sum(1 for record in recent_articles if record["category"] == current_category)
    recent_main_category_count = sum(1 for record in recent_articles if record["main_category"] == current_main_category)
    recent_color_count = sum(1 for record in recent_articles if record["color"] == current_color)
    recent_unique_category_count = len({record["category"] for record in recent_articles})
    recent_5_main_category_count = sum(1 for record in recent_5_articles if record["main_category"] == current_main_category)

    days_since_last_same_category_purchase = ""
    for previous_date, previous_article_id in zip(reversed(history_dates), reversed(history_article_ids), strict=False):
        previous_article = article_features.get(previous_article_id)
        if previous_article is None:
            continue
        if previous_article["category"] == current_category:
            days_since_last_same_category_purchase = str((transaction_date - previous_date).days)
            break

    return {
        "days_since_last_purchase": str(days_since_last_purchase),
        "last_purchase_category_match": "1" if last_article is not None and last_article["category"] == current_category else "0",
        "last_purchase_main_category_match": "1" if last_article is not None and last_article["main_category"] == current_main_category else "0",
        "last_purchase_color_match": "1" if last_article is not None and last_article["color"] == current_color else "0",
        "recent_3_purchase_category_count": str(recent_category_count),
        "recent_3_purchase_main_category_count": str(recent_main_category_count),
        "recent_3_purchase_color_count": str(recent_color_count),
        "recent_3_unique_category_count": str(recent_unique_category_count),
        "days_since_last_same_category_purchase": days_since_last_same_category_purchase,
        "recent_5_purchase_main_category_count": str(recent_5_main_category_count),
    }


def make_output_row(
    customer_id: str,
    article_id: str,
    label: str,
    price: str,
    sales_channel_id: str,
    customer_feature: CustomerFeature,
    article_feature: ArticleFeature,
    history_features: Dict[str, str],
) -> Dict[str, str]:
    age_bucket = customer_feature["age_bucket"]
    club_member_status = customer_feature["club_member_status"]
    fashion_news_frequency = customer_feature["fashion_news_frequency"]
    category = article_feature["category"]
    color = article_feature["color"]

    return {
        "customer_id": customer_id,
        "article_id": article_id,
        "label": label,
        "price": price,
        "sales_channel_id": sales_channel_id,
        "age": customer_feature["age"],
        "age_bucket": age_bucket,
        "fashion_news_frequency": fashion_news_frequency,
        "club_member_status": club_member_status,
        "prod_name": article_feature["prod_name"],
        "product_type_name": article_feature["product_type_name"],
        "product_group_name": article_feature["product_group_name"],
        "colour_group_name": article_feature["colour_group_name"],
        "perceived_colour_master_name": article_feature["perceived_colour_master_name"],
        "department_name": article_feature["department_name"],
        "section_name": article_feature["section_name"],
        "garment_group_name": article_feature["garment_group_name"],
        "category": category,
        "main_category": article_feature["main_category"],
        "color": color,
        "age_category": f"{age_bucket}_{category}",
        "age_color": f"{age_bucket}_{color}",
        "member_category": f"{club_member_status}_{category}",
        "fashion_category": f"{fashion_news_frequency}_{category}",
        **history_features,
    }


def collect_partition_user_data(
    partition_path: Path,
) -> Tuple[Dict[str, Set[str]], Dict[str, List[Tuple[date, str, str, str]]], int]:
    seen_pairs: Set[Tuple[str, str]] = set()
    user_purchased_articles: Dict[str, Set[str]] = {}
    positive_rows_by_user: Dict[str, List[Tuple[str, str, str]]] = {}
    duplicate_rows = 0

    with partition_path.open(newline="", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            customer_id = row["customer_id"]
            article_id = row["article_id"]
            pair = (customer_id, article_id)
            if pair in seen_pairs:
                duplicate_rows += 1
                continue
            seen_pairs.add(pair)

            user_purchased_articles.setdefault(customer_id, set()).add(article_id)
            positive_rows_by_user.setdefault(customer_id, []).append(
                (parse_transaction_date(row["t_dat"]), article_id, row["price"], row["sales_channel_id"])
            )

    return user_purchased_articles, positive_rows_by_user, duplicate_rows


def reservoir_sample_non_purchased(
    article_ids: Sequence[str],
    purchased_articles: Set[str],
    target_count: int,
    rng: random.Random,
) -> List[str]:
    sample: List[str] = []
    eligible_seen = 0

    for article_id in article_ids:
        if article_id in purchased_articles:
            continue
        eligible_seen += 1
        if len(sample) < target_count:
            sample.append(article_id)
            continue

        replace_index = rng.randint(1, eligible_seen)
        if replace_index <= target_count:
            sample[replace_index - 1] = article_id

    return sample


def rejection_sample_non_purchased(
    article_ids: Sequence[str],
    purchased_articles: Set[str],
    target_count: int,
    rng: random.Random,
) -> List[str]:
    sampled_articles: List[str] = []
    sampled_set: Set[str] = set()
    max_attempts = max(target_count * 20, 100)
    attempts = 0

    while len(sampled_articles) < target_count and attempts < max_attempts:
        article_id = article_ids[rng.randrange(len(article_ids))]
        attempts += 1
        if article_id in purchased_articles or article_id in sampled_set:
            continue
        sampled_set.add(article_id)
        sampled_articles.append(article_id)

    if len(sampled_articles) == target_count:
        return sampled_articles

    remainder = target_count - len(sampled_articles)
    fallback_sample = reservoir_sample_non_purchased(
        article_ids=article_ids,
        purchased_articles=purchased_articles.union(sampled_set),
        target_count=remainder,
        rng=rng,
    )
    return sampled_articles + fallback_sample


def sample_negative_articles(
    article_ids: Sequence[str],
    purchased_articles: Set[str],
    target_count: int,
    rng: random.Random,
) -> List[str]:
    if target_count <= 0:
        return []

    available_count = len(article_ids) - len(purchased_articles)
    if available_count <= 0:
        return []

    target_count = min(target_count, available_count)
    purchased_ratio = len(purchased_articles) / len(article_ids)

    # Rejection sampling is cheap when most items are valid; dense users fall back
    # to a single linear pass reservoir sample instead of sorting or set-diffing.
    if purchased_ratio >= 0.5:
        return reservoir_sample_non_purchased(article_ids, purchased_articles, target_count, rng)
    return rejection_sample_non_purchased(article_ids, purchased_articles, target_count, rng)


def write_partition_rows(
    writer: csv.DictWriter,
    partition_path: Path,
    customer_features: Dict[str, CustomerFeature],
    article_features: Dict[str, ArticleFeature],
    article_ids: Sequence[str],
    rng: random.Random,
) -> StatsDict:
    start_time = time.perf_counter()
    user_purchased_articles, positive_rows_by_user, duplicate_rows = collect_partition_user_data(partition_path)

    stats: StatsDict = {
        "users_seen": len(positive_rows_by_user),
        "unique_pairs_kept": 0,
        "duplicate_purchase_rows": duplicate_rows,
        "positives_written": 0,
        "negatives_written": 0,
        "users_without_negative_candidates": 0,
    }

    for customer_id, positive_rows in positive_rows_by_user.items():
        customer_feature = customer_features.get(customer_id)
        if customer_feature is None:
            continue

        positive_rows.sort(key=lambda row: (row[0], row[1]))
        purchased_articles = user_purchased_articles[customer_id]
        stats["unique_pairs_kept"] += len(positive_rows)
        positive_contexts: List[Tuple[date, Dict[str, str]]] = []
        history_article_ids: List[str] = []
        history_dates: List[date] = []

        for transaction_date, article_id, price, sales_channel_id in positive_rows:
            article_feature = article_features.get(article_id)
            if article_feature is None:
                continue
            history_features = build_purchase_history_features(
                article_id=article_id,
                transaction_date=transaction_date,
                history_article_ids=history_article_ids,
                history_dates=history_dates,
                article_features=article_features,
            )
            writer.writerow(
                make_output_row(
                    customer_id=customer_id,
                    article_id=article_id,
                    label="1",
                    price=price,
                    sales_channel_id=sales_channel_id,
                    customer_feature=customer_feature,
                    article_feature=article_feature,
                    history_features=history_features,
                )
            )
            stats["positives_written"] += 1
            positive_contexts.append((transaction_date, history_features))
            history_article_ids.append(article_id)
            history_dates.append(transaction_date)

        negative_target = len(positive_rows) * NEGATIVE_RATIO
        negative_article_ids = sample_negative_articles(
            article_ids=article_ids,
            purchased_articles=purchased_articles,
            target_count=negative_target,
            rng=rng,
        )

        if not negative_article_ids and negative_target > 0:
            stats["users_without_negative_candidates"] += 1
            continue

        for negative_index, article_id in enumerate(negative_article_ids):
            article_feature = article_features.get(article_id)
            if article_feature is None:
                continue
            context_index = min(negative_index // max(NEGATIVE_RATIO, 1), max(len(positive_contexts) - 1, 0))
            _, history_features = positive_contexts[context_index]
            writer.writerow(
                make_output_row(
                    customer_id=customer_id,
                    article_id=article_id,
                    label="0",
                    price="",
                    sales_channel_id="-1",
                    customer_feature=customer_feature,
                    article_feature=article_feature,
                    history_features=history_features,
                )
            )
            stats["negatives_written"] += 1

    log_stage(
        f"process_partition:{partition_path.name}",
        start_time,
        **stats,
    )
    return stats


def build_train_dataset(
    customer_features: Dict[str, CustomerFeature],
    article_features: Dict[str, ArticleFeature],
    article_ids: Sequence[str],
) -> StatsDict:
    rng = random.Random(RANDOM_SEED)
    totals: StatsDict = {
        "rows_scanned": 0,
        "rows_partitioned": 0,
        "missing_customer_rows": 0,
        "missing_article_rows": 0,
        "users_seen": 0,
        "unique_pairs_kept": 0,
        "duplicate_purchase_rows": 0,
        "positives_written": 0,
        "negatives_written": 0,
        "users_without_negative_candidates": 0,
    }

    with tempfile.TemporaryDirectory(prefix="make_train_data_", dir=".") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        partition_stats = partition_transactions(
            transactions_path=TRANSACTIONS_FILE,
            temp_dir=temp_dir,
            customer_features=customer_features,
            article_features=article_features,
        )
        for key, value in partition_stats.items():
            totals[key] += value

        output_start = time.perf_counter()
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as outfile:
            writer = csv.DictWriter(outfile, fieldnames=OUTPUT_COLUMNS)
            writer.writeheader()

            for partition_index in range(PARTITION_COUNT):
                partition_path = temp_dir / f"transactions_partition_{partition_index:04d}.csv"
                partition_stats = write_partition_rows(
                    writer=writer,
                    partition_path=partition_path,
                    customer_features=customer_features,
                    article_features=article_features,
                    article_ids=article_ids,
                    rng=rng,
                )
                for key, value in partition_stats.items():
                    totals[key] += value
        log_stage("write_output", output_start, positives_written=totals["positives_written"], negatives_written=totals["negatives_written"])

    return totals


def main() -> None:
    configure_logging()
    run_start = time.perf_counter()
    transactions_path = resolve_required_file(TRANSACTIONS_FILE, "transactions raw file")
    customer_path = resolve_required_file(CUSTOMER_FEATURES_FILE, "customer feature file")
    article_path = resolve_required_file(ARTICLE_FEATURES_FILE, "article feature file")
    validate_required_columns(transactions_path, TRANSACTION_REQUIRED_COLUMNS, "customer_id")
    validate_required_columns(customer_path, CUSTOMER_FEATURE_COLUMNS, "customer_id")
    validate_required_columns(article_path, ARTICLE_FEATURE_COLUMNS, "article_id")
    logging.info(
        "mode=%s transactions_file=%s customer_features_file=%s article_features_file=%s output_file=%s max_transaction_rows=%s negative_ratio=%s partition_count=%s random_seed=%s",
        RUNTIME_MODE,
        transactions_path,
        customer_path,
        article_path,
        OUTPUT_FILE,
        MAX_TRANSACTION_ROWS,
        NEGATIVE_RATIO,
        PARTITION_COUNT,
        RANDOM_SEED,
    )

    customer_start = time.perf_counter()
    customer_features = load_customer_features(customer_path)
    log_stage("load_customer_features", customer_start, customer_count=len(customer_features))

    article_start = time.perf_counter()
    article_features = load_article_features(article_path)
    article_ids = tuple(article_features.keys())
    log_stage("load_article_features", article_start, article_count=len(article_features))

    totals = build_train_dataset(
        customer_features=customer_features,
        article_features=article_features,
        article_ids=article_ids,
    )

    total_samples = totals["positives_written"] + totals["negatives_written"]
    log_stage(
        "complete",
        run_start,
        rows_scanned=totals["rows_scanned"],
        rows_partitioned=totals["rows_partitioned"],
        unique_pairs_kept=totals["unique_pairs_kept"],
        users_seen=totals["users_seen"],
        positives_written=totals["positives_written"],
        negatives_written=totals["negatives_written"],
        total_samples=total_samples,
        duplicate_purchase_rows=totals["duplicate_purchase_rows"],
        users_without_negative_candidates=totals["users_without_negative_candidates"],
    )


if __name__ == "__main__":
    main()
