from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, Optional

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

CUSTOMERS_FILE = RAW_DIR / "customers.csv"
ARTICLES_FILE = RAW_DIR / "articles.csv"
TRANSACTIONS_FILE = RAW_DIR / "transactions_train.csv"

# MODE = "production"
# MODE = "dev"
MODE = "test"

MODE_CONFIG = {
    "test": {
        "MAX_TRANSACTION_ROWS": 100_000,
        "CHUNK_SIZE": 50_000,
        "SEGMENT_TOP_K": 20,
        "INTERACTIONS_FILE": PROCESSED_DIR / "candidate_interactions_test.csv.gz",
        "USER_FEATURES_FILE": PROCESSED_DIR / "candidate_user_features_test.csv.gz",
        "ITEM_FEATURES_FILE": PROCESSED_DIR / "candidate_item_features_test.csv.gz",
        "SEGMENT_CANDIDATES_FILE": PROCESSED_DIR / "candidate_segment_candidates_test.csv.gz",
        "TRAIN_DATA_FILE": PROCESSED_DIR / "candidate_train_data_test.csv.gz",
        "MANIFEST_FILE": PROCESSED_DIR / "candidate_manifest_test.json",
        "LOG_EVERY_N_ROWS": 100_000,
    },
    "dev": {
        "MAX_TRANSACTION_ROWS": 1_000_000,
        "CHUNK_SIZE": 100_000,
        "SEGMENT_TOP_K": 30,
        "INTERACTIONS_FILE": PROCESSED_DIR / "candidate_interactions_dev.csv.gz",
        "USER_FEATURES_FILE": PROCESSED_DIR / "candidate_user_features_dev.csv.gz",
        "ITEM_FEATURES_FILE": PROCESSED_DIR / "candidate_item_features_dev.csv.gz",
        "SEGMENT_CANDIDATES_FILE": PROCESSED_DIR / "candidate_segment_candidates_dev.csv.gz",
        "TRAIN_DATA_FILE": PROCESSED_DIR / "candidate_train_data_dev.csv.gz",
        "MANIFEST_FILE": PROCESSED_DIR / "candidate_manifest_dev.json",
        "LOG_EVERY_N_ROWS": 250_000,
    },
    "production": {
        "MAX_TRANSACTION_ROWS": None,
        "CHUNK_SIZE": 250_000,
        "SEGMENT_TOP_K": 50,
        "INTERACTIONS_FILE": PROCESSED_DIR / "candidate_interactions.csv.gz",
        "USER_FEATURES_FILE": PROCESSED_DIR / "candidate_user_features.csv.gz",
        "ITEM_FEATURES_FILE": PROCESSED_DIR / "candidate_item_features.csv.gz",
        "SEGMENT_CANDIDATES_FILE": PROCESSED_DIR / "candidate_segment_candidates.csv.gz",
        "TRAIN_DATA_FILE": PROCESSED_DIR / "candidate_train_data.csv.gz",
        "MANIFEST_FILE": PROCESSED_DIR / "candidate_manifest.json",
        "LOG_EVERY_N_ROWS": 1_000_000,
    },
}

RUNTIME_MODE = os.getenv("DATA_PIPELINE_MODE", MODE).strip().lower()

if RUNTIME_MODE not in MODE_CONFIG:
    raise ValueError(f"Unsupported MODE: {RUNTIME_MODE}")

CONFIG = MODE_CONFIG[RUNTIME_MODE]
MAX_TRANSACTION_ROWS: Optional[int] = CONFIG["MAX_TRANSACTION_ROWS"]
CHUNK_SIZE: int = CONFIG["CHUNK_SIZE"]
SEGMENT_TOP_K: int = CONFIG["SEGMENT_TOP_K"]
INTERACTIONS_FILE: Path = CONFIG["INTERACTIONS_FILE"]
USER_FEATURES_FILE: Path = CONFIG["USER_FEATURES_FILE"]
ITEM_FEATURES_FILE: Path = CONFIG["ITEM_FEATURES_FILE"]
SEGMENT_CANDIDATES_FILE: Path = CONFIG["SEGMENT_CANDIDATES_FILE"]
TRAIN_DATA_FILE: Path = CONFIG["TRAIN_DATA_FILE"]
MANIFEST_FILE: Path = CONFIG["MANIFEST_FILE"]
LOG_EVERY_N_ROWS: int = CONFIG["LOG_EVERY_N_ROWS"]

UNKNOWN_VALUE = "UNKNOWN"
AGE_BUCKETS = ["under_20", "20s", "30s", "40s", "50s", "60_plus", "unknown"]
SEASONS = ["spring", "summer", "autumn", "winter"]
MAX_HISTORY_ARTICLES = 10

USER_FEATURE_COLUMNS = [
    "customer_id",
    "age",
    "age_bucket",
    "fashion_news_frequency",
    "club_member_status",
    "purchase_count",
    "purchase_count_7d",
    "purchase_count_30d",
    "purchase_count_90d",
    "total_spend",
    "spend_7d",
    "spend_30d",
    "spend_90d",
    "avg_price",
    "min_price",
    "max_price",
    "recency_days",
    "purchase_span_days",
    "online_ratio",
    "offline_ratio",
    "preferred_garment_group",
    "preferred_colour_master",
    "preferred_main_category",
    "preferred_season",
    "price_band",
    "activity_segment",
    "preferred_segment_key",
]

ITEM_FEATURE_COLUMNS = [
    "article_id",
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
    "item_purchase_count",
    "item_purchase_count_7d",
    "item_purchase_count_30d",
    "item_purchase_count_90d",
    "item_total_spend",
    "item_avg_price",
    "item_min_price",
    "item_max_price",
    "item_days_since_last_purchase",
    "item_freshness_days",
    "item_online_ratio",
    "item_offline_ratio",
    "dominant_age_bucket",
    "dominant_season",
    "item_price_band",
    "popularity_segment",
]

INTERACTION_COLUMNS = [
    "customer_id",
    "article_id",
    "t_dat",
    "price",
    "sales_channel_id",
    "year",
    "month",
    "week",
    "day_of_week",
    "season",
    "event_type",
    "event_strength",
    "label",
    "split",
    "history_article_ids",
]


@dataclass
class TransactionProfile:
    total_rows: int
    min_date: Optional[pd.Timestamp]
    max_date: Optional[pd.Timestamp]
    unique_customers: int
    unique_articles: int


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def log_stage(stage: str, start_time: float, **stats: object) -> None:
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


def normalize_text(value: object) -> str:
    if value is None or pd.isna(value):
        raw_text = ""
    else:
        raw_text = str(value)
    normalized = " ".join(raw_text.strip().split())
    return normalized if normalized else UNKNOWN_VALUE


def normalize_fashion_news_frequency(value: object) -> str:
    aliases = {
        "NONE": "NONE",
        "REGULARLY": "REGULARLY",
        "MONTHLY": "MONTHLY",
    }
    return aliases.get(normalize_text(value).upper(), UNKNOWN_VALUE)


def normalize_club_member_status(value: object) -> str:
    normalized = normalize_text(value).upper()
    aliases = {
        "ACTIVE": "ACTIVE",
        "PRE-CREATE": "PRE-CREATE",
        "LEFT CLUB": "LEFT CLUB",
    }
    return aliases.get(normalized, normalized if normalized != UNKNOWN_VALUE else UNKNOWN_VALUE)


def parse_age_series(age: pd.Series) -> pd.Series:
    parsed = pd.to_numeric(age, errors="coerce").fillna(-1).astype("int32")
    parsed = parsed.where((parsed >= 0) & (parsed <= 120), -1)
    return parsed


def make_age_bucket_series(age: pd.Series) -> pd.Series:
    labels = np.select(
        [
            age.lt(0),
            age.lt(20),
            age.lt(30),
            age.lt(40),
            age.lt(50),
            age.lt(60),
            age.ge(60),
        ],
        ["unknown", "under_20", "20s", "30s", "40s", "50s", "60_plus"],
        default="unknown",
    )
    return pd.Series(labels, index=age.index, dtype="string")


def normalize_main_category(row: pd.Series) -> str:
    for column in ("index_group_name", "index_name", "product_group_name"):
        value = normalize_text(row.get(column, ""))
        if value != UNKNOWN_VALUE:
            return value
    return UNKNOWN_VALUE


def normalize_category(row: pd.Series) -> str:
    for column in ("product_type_name", "product_group_name", "department_name"):
        value = normalize_text(row.get(column, ""))
        if value != UNKNOWN_VALUE:
            return value
    return UNKNOWN_VALUE


def normalize_color(row: pd.Series) -> str:
    for column in ("perceived_colour_master_name", "colour_group_name"):
        value = normalize_text(row.get(column, ""))
        if value != UNKNOWN_VALUE:
            return value
    return UNKNOWN_VALUE


def season_from_dates(dates: pd.Series) -> pd.Series:
    months = pd.to_datetime(dates).dt.month
    labels = np.select(
        [
            months.isin([3, 4, 5]),
            months.isin([6, 7, 8]),
            months.isin([9, 10, 11]),
            months.isin([12, 1, 2]),
        ],
        ["spring", "summer", "autumn", "winter"],
        default=UNKNOWN_VALUE,
    )
    return pd.Series(labels, index=dates.index, dtype="string")


def interaction_split(dates: pd.Series, valid_start: pd.Timestamp, test_start: pd.Timestamp) -> pd.Series:
    values = np.select(
        [dates.ge(test_start), dates.ge(valid_start)],
        ["test", "valid"],
        default="train",
    )
    return pd.Series(values, index=dates.index, dtype="string")


def classify_price_band(values: pd.Series) -> pd.Series:
    valid = pd.to_numeric(values, errors="coerce").dropna()
    if valid.empty:
        return pd.Series("unknown", index=values.index, dtype="string")
    low, high = valid.quantile([0.33, 0.66]).tolist()
    labels = np.select(
        [values.le(low), values.le(high), values.gt(high)],
        ["budget", "mid", "premium"],
        default="unknown",
    )
    return pd.Series(labels, index=values.index, dtype="string")


def iterate_transaction_chunks(path: Path) -> Iterator[pd.DataFrame]:
    rows_remaining = MAX_TRANSACTION_ROWS
    reader = pd.read_csv(
        path,
        usecols=["t_dat", "customer_id", "article_id", "price", "sales_channel_id"],
        dtype={
            "customer_id": "string",
            "article_id": "string",
            "price": "float32",
            "sales_channel_id": "int8",
        },
        parse_dates=["t_dat"],
        chunksize=CHUNK_SIZE,
    )
    for chunk in reader:
        if rows_remaining is not None:
            if rows_remaining <= 0:
                break
            if len(chunk) > rows_remaining:
                chunk = chunk.iloc[:rows_remaining].copy()
            rows_remaining -= len(chunk)
        yield chunk


def load_customers(path: Path) -> pd.DataFrame:
    customers = pd.read_csv(
        path,
        usecols=["customer_id", "age", "fashion_news_frequency", "club_member_status"],
        dtype={
            "customer_id": "string",
            "age": "string",
            "fashion_news_frequency": "string",
            "club_member_status": "string",
        },
    )
    customers["customer_id"] = customers["customer_id"].astype("string")
    customers["age"] = parse_age_series(customers["age"])
    customers["age_bucket"] = make_age_bucket_series(customers["age"])
    customers["fashion_news_frequency"] = customers["fashion_news_frequency"].map(
        normalize_fashion_news_frequency
    )
    customers["club_member_status"] = customers["club_member_status"].map(
        normalize_club_member_status
    )
    return customers


def load_articles(path: Path) -> pd.DataFrame:
    articles = pd.read_csv(
        path,
        usecols=[
            "article_id",
            "prod_name",
            "product_type_name",
            "product_group_name",
            "colour_group_name",
            "perceived_colour_master_name",
            "department_name",
            "section_name",
            "garment_group_name",
            "index_name",
            "index_group_name",
        ],
        dtype="string",
    )
    for column in articles.columns:
        if column != "article_id":
            articles[column] = articles[column].map(normalize_text)
    articles["category"] = articles.apply(normalize_category, axis=1)
    articles["main_category"] = articles.apply(normalize_main_category, axis=1)
    articles["color"] = articles.apply(normalize_color, axis=1)
    return articles


def profile_transactions(path: Path) -> TransactionProfile:
    start_time = time.perf_counter()
    total_rows = 0
    min_date = None
    max_date = None
    unique_customers: set[str] = set()
    unique_articles: set[str] = set()

    for chunk in iterate_transaction_chunks(path):
        total_rows += len(chunk)
        chunk_min = chunk["t_dat"].min()
        chunk_max = chunk["t_dat"].max()
        if min_date is None or chunk_min < min_date:
            min_date = chunk_min
        if max_date is None or chunk_max > max_date:
            max_date = chunk_max
        unique_customers.update(chunk["customer_id"].dropna().astype(str).unique().tolist())
        unique_articles.update(chunk["article_id"].dropna().astype(str).unique().tolist())

    profile = TransactionProfile(
        total_rows=total_rows,
        min_date=min_date,
        max_date=max_date,
        unique_customers=len(unique_customers),
        unique_articles=len(unique_articles),
    )
    log_stage(
        "profile_transactions",
        start_time,
        total_rows=profile.total_rows,
        unique_customers=profile.unique_customers,
        unique_articles=profile.unique_articles,
        min_date=profile.min_date.date() if profile.min_date is not None else "",
        max_date=profile.max_date.date() if profile.max_date is not None else "",
    )
    return profile


def prepare_user_state(customers: pd.DataFrame) -> pd.DataFrame:
    users = customers.copy().set_index("customer_id")
    users["purchase_count"] = np.uint32(0)
    users["purchase_count_7d"] = np.uint32(0)
    users["purchase_count_30d"] = np.uint32(0)
    users["purchase_count_90d"] = np.uint32(0)
    users["total_spend"] = np.float64(0.0)
    users["spend_7d"] = np.float64(0.0)
    users["spend_30d"] = np.float64(0.0)
    users["spend_90d"] = np.float64(0.0)
    users["min_price"] = np.float32(np.inf)
    users["max_price"] = np.float32(0.0)
    users["channel_1_count"] = np.uint32(0)
    users["channel_2_count"] = np.uint32(0)
    users["first_purchase_date"] = pd.Series(pd.NaT, index=users.index, dtype="datetime64[ns]")
    users["last_purchase_date"] = pd.Series(pd.NaT, index=users.index, dtype="datetime64[ns]")
    return users


def prepare_item_state(articles: pd.DataFrame) -> pd.DataFrame:
    items = articles.copy().set_index("article_id")
    items["item_purchase_count"] = np.uint32(0)
    items["item_purchase_count_7d"] = np.uint32(0)
    items["item_purchase_count_30d"] = np.uint32(0)
    items["item_purchase_count_90d"] = np.uint32(0)
    items["item_total_spend"] = np.float64(0.0)
    items["item_min_price"] = np.float32(np.inf)
    items["item_max_price"] = np.float32(0.0)
    items["channel_1_count"] = np.uint32(0)
    items["channel_2_count"] = np.uint32(0)
    items["first_purchase_date"] = pd.Series(pd.NaT, index=items.index, dtype="datetime64[ns]")
    items["last_purchase_date"] = pd.Series(pd.NaT, index=items.index, dtype="datetime64[ns]")
    return items


def build_zero_frame(index: pd.Index, columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        np.zeros((len(index), len(columns)), dtype=np.uint32),
        index=index,
        columns=columns,
    )


def add_group_counts(target: pd.DataFrame, grouped: pd.DataFrame) -> None:
    if grouped.empty:
        return
    idx = grouped.index
    cols = grouped.columns
    target.loc[idx, cols] = (
        target.loc[idx, cols].to_numpy(dtype=np.uint64)
        + grouped.to_numpy(dtype=np.uint64)
    ).astype(np.uint32)


def update_user_aggregates(users: pd.DataFrame, grouped: pd.DataFrame) -> None:
    idx = grouped.index
    current = users.loc[idx]
    users.loc[idx, "purchase_count"] = (
        current["purchase_count"].to_numpy(dtype=np.uint64)
        + grouped["purchase_count"].to_numpy(dtype=np.uint64)
    ).astype(np.uint32)
    users.loc[idx, "total_spend"] = current["total_spend"].to_numpy() + grouped["total_spend"].to_numpy()
    users.loc[idx, "min_price"] = np.minimum(
        current["min_price"].to_numpy(),
        grouped["min_price"].to_numpy(),
    )
    users.loc[idx, "max_price"] = np.maximum(
        current["max_price"].to_numpy(),
        grouped["max_price"].to_numpy(),
    )
    users.loc[idx, "channel_1_count"] = (
        current["channel_1_count"].to_numpy(dtype=np.uint64)
        + grouped["channel_1_count"].to_numpy(dtype=np.uint64)
    ).astype(np.uint32)
    users.loc[idx, "channel_2_count"] = (
        current["channel_2_count"].to_numpy(dtype=np.uint64)
        + grouped["channel_2_count"].to_numpy(dtype=np.uint64)
    ).astype(np.uint32)

    first_existing = current["first_purchase_date"]
    first_new = grouped["first_purchase_date"]
    first_mask = first_existing.isna() | first_new.lt(first_existing)
    if first_mask.any():
        users.loc[idx[first_mask], "first_purchase_date"] = first_new[first_mask]

    last_existing = current["last_purchase_date"]
    last_new = grouped["last_purchase_date"]
    last_mask = last_existing.isna() | last_new.gt(last_existing)
    if last_mask.any():
        users.loc[idx[last_mask], "last_purchase_date"] = last_new[last_mask]


def update_item_aggregates(items: pd.DataFrame, grouped: pd.DataFrame) -> None:
    idx = grouped.index
    current = items.loc[idx]
    items.loc[idx, "item_purchase_count"] = (
        current["item_purchase_count"].to_numpy(dtype=np.uint64)
        + grouped["item_purchase_count"].to_numpy(dtype=np.uint64)
    ).astype(np.uint32)
    items.loc[idx, "item_total_spend"] = (
        current["item_total_spend"].to_numpy() + grouped["item_total_spend"].to_numpy()
    )
    items.loc[idx, "item_min_price"] = np.minimum(
        current["item_min_price"].to_numpy(),
        grouped["item_min_price"].to_numpy(),
    )
    items.loc[idx, "item_max_price"] = np.maximum(
        current["item_max_price"].to_numpy(),
        grouped["item_max_price"].to_numpy(),
    )
    items.loc[idx, "channel_1_count"] = (
        current["channel_1_count"].to_numpy(dtype=np.uint64)
        + grouped["channel_1_count"].to_numpy(dtype=np.uint64)
    ).astype(np.uint32)
    items.loc[idx, "channel_2_count"] = (
        current["channel_2_count"].to_numpy(dtype=np.uint64)
        + grouped["channel_2_count"].to_numpy(dtype=np.uint64)
    ).astype(np.uint32)

    first_existing = current["first_purchase_date"]
    first_new = grouped["first_purchase_date"]
    first_mask = first_existing.isna() | first_new.lt(first_existing)
    if first_mask.any():
        items.loc[idx[first_mask], "first_purchase_date"] = first_new[first_mask]

    last_existing = current["last_purchase_date"]
    last_new = grouped["last_purchase_date"]
    last_mask = last_existing.isna() | last_new.gt(last_existing)
    if last_mask.any():
        items.loc[idx[last_mask], "last_purchase_date"] = last_new[last_mask]


def append_window_aggregates(target: pd.DataFrame, grouped: pd.DataFrame, count_column: str, spend_column: Optional[str]) -> None:
    if grouped.empty:
        return
    idx = grouped.index
    target.loc[idx, count_column] = (
        target.loc[idx, count_column].to_numpy(dtype=np.uint64)
        + grouped["purchase_count"].to_numpy(dtype=np.uint64)
    ).astype(np.uint32)
    if spend_column is not None:
        target.loc[idx, spend_column] = (
            target.loc[idx, spend_column].to_numpy() + grouped["total_spend"].to_numpy()
        )


def append_item_window_counts(items: pd.DataFrame, grouped: pd.DataFrame, count_column: str) -> None:
    if grouped.empty:
        return
    idx = grouped.index
    items.loc[idx, count_column] = (
        items.loc[idx, count_column].to_numpy(dtype=np.uint64)
        + grouped["purchase_count"].to_numpy(dtype=np.uint64)
    ).astype(np.uint32)


def finalize_user_features(
    users: pd.DataFrame,
    garment_counts: pd.DataFrame,
    colour_counts: pd.DataFrame,
    main_category_counts: pd.DataFrame,
    season_counts: pd.DataFrame,
    reference_date: pd.Timestamp,
) -> pd.DataFrame:
    users = users.copy()
    has_purchase = users["purchase_count"].gt(0)
    users.loc[~has_purchase, ["min_price", "max_price"]] = np.nan
    users["avg_price"] = np.where(
        has_purchase,
        users["total_spend"] / users["purchase_count"].replace(0, np.nan),
        np.nan,
    )
    users["recency_days"] = (reference_date - users["last_purchase_date"]).dt.days.astype("float32")
    users["purchase_span_days"] = (
        users["last_purchase_date"] - users["first_purchase_date"]
    ).dt.days.fillna(0).astype("float32")
    users["online_ratio"] = np.where(
        has_purchase,
        users["channel_2_count"] / users["purchase_count"].replace(0, np.nan),
        np.nan,
    )
    users["offline_ratio"] = np.where(
        has_purchase,
        users["channel_1_count"] / users["purchase_count"].replace(0, np.nan),
        np.nan,
    )
    users["preferred_garment_group"] = UNKNOWN_VALUE
    users.loc[has_purchase, "preferred_garment_group"] = garment_counts.loc[has_purchase].idxmax(axis=1)
    users["preferred_colour_master"] = UNKNOWN_VALUE
    users.loc[has_purchase, "preferred_colour_master"] = colour_counts.loc[has_purchase].idxmax(axis=1)
    users["preferred_main_category"] = UNKNOWN_VALUE
    users.loc[has_purchase, "preferred_main_category"] = main_category_counts.loc[has_purchase].idxmax(axis=1)
    users["preferred_season"] = UNKNOWN_VALUE
    users.loc[has_purchase, "preferred_season"] = season_counts.loc[has_purchase].idxmax(axis=1)
    users["price_band"] = classify_price_band(users["avg_price"])

    if has_purchase.any():
        purchase_low, purchase_high = users.loc[has_purchase, "purchase_count"].quantile([0.33, 0.66]).tolist()
    else:
        purchase_low, purchase_high = 0.0, 0.0
    users["activity_segment"] = "cold_start"
    users.loc[has_purchase & users["purchase_count"].le(purchase_low), "activity_segment"] = "light"
    users.loc[
        has_purchase
        & users["purchase_count"].gt(purchase_low)
        & users["purchase_count"].le(purchase_high),
        "activity_segment",
    ] = "medium"
    users.loc[has_purchase & users["purchase_count"].gt(purchase_high), "activity_segment"] = "heavy"
    users["preferred_segment_key"] = (
        users["preferred_garment_group"].astype("string").fillna(UNKNOWN_VALUE)
        + "::"
        + users["price_band"].astype("string").fillna("unknown")
    )
    return users.reset_index()[USER_FEATURE_COLUMNS]


def finalize_item_features(
    items: pd.DataFrame,
    age_bucket_counts: pd.DataFrame,
    season_counts: pd.DataFrame,
    reference_date: pd.Timestamp,
) -> pd.DataFrame:
    items = items.copy()
    has_purchase = items["item_purchase_count"].gt(0)
    items.loc[~has_purchase, ["item_min_price", "item_max_price"]] = np.nan
    items["item_avg_price"] = np.where(
        has_purchase,
        items["item_total_spend"] / items["item_purchase_count"].replace(0, np.nan),
        np.nan,
    )
    items["item_days_since_last_purchase"] = (
        reference_date - items["last_purchase_date"]
    ).dt.days.astype("float32")
    items["item_freshness_days"] = (
        reference_date - items["first_purchase_date"]
    ).dt.days.astype("float32")
    items["item_online_ratio"] = np.where(
        has_purchase,
        items["channel_2_count"] / items["item_purchase_count"].replace(0, np.nan),
        np.nan,
    )
    items["item_offline_ratio"] = np.where(
        has_purchase,
        items["channel_1_count"] / items["item_purchase_count"].replace(0, np.nan),
        np.nan,
    )
    items["dominant_age_bucket"] = "unknown"
    items.loc[has_purchase, "dominant_age_bucket"] = age_bucket_counts.loc[has_purchase].idxmax(axis=1)
    items["dominant_season"] = UNKNOWN_VALUE
    items.loc[has_purchase, "dominant_season"] = season_counts.loc[has_purchase].idxmax(axis=1)
    items["item_price_band"] = classify_price_band(items["item_avg_price"])

    if has_purchase.any():
        pop_low, pop_high = items.loc[has_purchase, "item_purchase_count"].quantile([0.33, 0.66]).tolist()
    else:
        pop_low, pop_high = 0.0, 0.0
    items["popularity_segment"] = "cold"
    items.loc[has_purchase & items["item_purchase_count"].le(pop_low), "popularity_segment"] = "long_tail"
    items.loc[
        has_purchase
        & items["item_purchase_count"].gt(pop_low)
        & items["item_purchase_count"].le(pop_high),
        "popularity_segment",
    ] = "mid"
    items.loc[has_purchase & items["item_purchase_count"].gt(pop_high), "popularity_segment"] = "head"
    return items.reset_index()[ITEM_FEATURE_COLUMNS]


def write_interaction_chunk(
    chunk: pd.DataFrame,
    output_path: Path,
    valid_start: pd.Timestamp,
    test_start: pd.Timestamp,
    is_first_chunk: bool,
    user_history: Dict[str, list[str]],
) -> None:
    interactions = chunk[["customer_id", "article_id", "t_dat", "price", "sales_channel_id"]].copy()
    interactions = interactions.sort_values(["t_dat", "customer_id", "article_id"]).reset_index(drop=True)
    history_values: list[str] = []
    for row in interactions.loc[:, ["customer_id", "article_id"]].itertuples(index=False):
        customer_id = str(row.customer_id)
        article_id = str(row.article_id)
        previous_items = user_history.setdefault(customer_id, [])
        history_values.append(",".join(previous_items[-MAX_HISTORY_ARTICLES:]))
        previous_items.append(article_id)
        if len(previous_items) > MAX_HISTORY_ARTICLES:
            del previous_items[:-MAX_HISTORY_ARTICLES]
    interactions["year"] = interactions["t_dat"].dt.year.astype("int16")
    interactions["month"] = interactions["t_dat"].dt.month.astype("int8")
    interactions["week"] = interactions["t_dat"].dt.isocalendar().week.astype("int16")
    interactions["day_of_week"] = interactions["t_dat"].dt.dayofweek.astype("int8")
    interactions["season"] = season_from_dates(interactions["t_dat"])
    interactions["event_type"] = "purchase"
    interactions["event_strength"] = np.int8(3)
    interactions["label"] = np.int8(1)
    interactions["split"] = interaction_split(interactions["t_dat"], valid_start, test_start)
    interactions["history_article_ids"] = history_values
    interactions.to_csv(
        output_path,
        index=False,
        mode="w" if is_first_chunk else "a",
        header=is_first_chunk,
        compression="gzip",
    )


def build_segment_candidates(item_features: pd.DataFrame, output_path: Path) -> None:
    ranked = item_features.loc[item_features["item_purchase_count"].gt(0)].copy()
    ranked["segment_key"] = (
        ranked["garment_group_name"].astype("string").fillna(UNKNOWN_VALUE)
        + "::"
        + ranked["item_price_band"].astype("string").fillna("unknown")
    )
    ranked = ranked.sort_values(
        by=["segment_key", "item_purchase_count_30d", "item_purchase_count", "item_avg_price"],
        ascending=[True, False, False, False],
    )
    ranked["segment_rank"] = ranked.groupby("segment_key").cumcount() + 1
    ranked = ranked.loc[ranked["segment_rank"].le(SEGMENT_TOP_K), [
        "segment_key",
        "segment_rank",
        "article_id",
        "prod_name",
        "garment_group_name",
        "item_price_band",
        "color",
        "item_purchase_count_30d",
        "item_purchase_count",
    ]]
    ranked.to_csv(output_path, index=False, compression="gzip")


def build_candidate_train_data(
    interactions_path: Path,
    user_features: pd.DataFrame,
    item_features: pd.DataFrame,
    output_path: Path,
) -> int:
    start_time = time.perf_counter()
    user_lookup = user_features.copy()
    item_lookup = item_features.copy()
    rows_written = 0
    is_first_chunk = True

    for chunk in pd.read_csv(interactions_path, chunksize=CHUNK_SIZE, compression="gzip"):
        chunk["customer_id"] = chunk["customer_id"].astype(str)
        chunk["article_id"] = chunk["article_id"].astype(str)
        joined = chunk.merge(user_lookup, on="customer_id", how="left", validate="many_to_one")
        joined = joined.merge(item_lookup, on="article_id", how="left", validate="many_to_one")
        joined.to_csv(
            output_path,
            index=False,
            mode="w" if is_first_chunk else "a",
            header=is_first_chunk,
            compression="gzip",
        )
        rows_written += len(joined)
        is_first_chunk = False

    log_stage("build_candidate_train_data", start_time, rows_written=rows_written)
    return rows_written


def write_manifest(profile: TransactionProfile, candidate_rows: int) -> None:
    manifest = {
        "mode": RUNTIME_MODE,
        "max_transaction_rows": MAX_TRANSACTION_ROWS,
        "chunk_size": CHUNK_SIZE,
        "segment_top_k": SEGMENT_TOP_K,
        "transactions": {
            "rows": profile.total_rows,
            "unique_customers": profile.unique_customers,
            "unique_articles": profile.unique_articles,
            "min_date": profile.min_date.strftime("%Y-%m-%d") if profile.min_date is not None else None,
            "max_date": profile.max_date.strftime("%Y-%m-%d") if profile.max_date is not None else None,
        },
        "outputs": {
            "interactions": str(INTERACTIONS_FILE),
            "user_features": str(USER_FEATURES_FILE),
            "item_features": str(ITEM_FEATURES_FILE),
            "segment_candidates": str(SEGMENT_CANDIDATES_FILE),
            "candidate_train_data": str(TRAIN_DATA_FILE),
        },
        "candidate_train_rows": candidate_rows,
    }
    MANIFEST_FILE.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    configure_logging()
    run_start = time.perf_counter()

    customers_path = resolve_required_file(CUSTOMERS_FILE, "customers raw file")
    articles_path = resolve_required_file(ARTICLES_FILE, "articles raw file")
    transactions_path = resolve_required_file(TRANSACTIONS_FILE, "transactions raw file")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    logging.info(
        "mode=%s customers_file=%s articles_file=%s transactions_file=%s max_transaction_rows=%s chunk_size=%s segment_top_k=%s",
        RUNTIME_MODE,
        customers_path,
        articles_path,
        transactions_path,
        MAX_TRANSACTION_ROWS,
        CHUNK_SIZE,
        SEGMENT_TOP_K,
    )

    customers = load_customers(customers_path)
    articles = load_articles(articles_path)
    log_stage("load_reference_tables", run_start, customer_count=len(customers), article_count=len(articles))

    profile = profile_transactions(transactions_path)
    if profile.max_date is None:
        raise ValueError("transactions_train.csv does not contain any valid transaction rows.")

    valid_start = profile.max_date - pd.Timedelta(days=27)
    test_start = profile.max_date - pd.Timedelta(days=13)
    cutoff_7 = profile.max_date - pd.Timedelta(days=6)
    cutoff_30 = profile.max_date - pd.Timedelta(days=29)
    cutoff_90 = profile.max_date - pd.Timedelta(days=89)

    users = prepare_user_state(customers)
    items = prepare_item_state(articles)

    garment_groups = sorted(articles["garment_group_name"].fillna(UNKNOWN_VALUE).unique().tolist())
    colour_masters = sorted(articles["perceived_colour_master_name"].fillna(UNKNOWN_VALUE).unique().tolist())
    main_categories = sorted(articles["main_category"].fillna(UNKNOWN_VALUE).unique().tolist())

    user_garment_counts = build_zero_frame(users.index, garment_groups)
    user_colour_counts = build_zero_frame(users.index, colour_masters)
    user_main_category_counts = build_zero_frame(users.index, main_categories)
    user_season_counts = build_zero_frame(users.index, SEASONS)

    item_age_bucket_counts = build_zero_frame(items.index, AGE_BUCKETS)
    item_season_counts = build_zero_frame(items.index, SEASONS)

    customer_age_lookup = customers.set_index("customer_id")["age_bucket"]
    article_lookup = articles[
        ["article_id", "garment_group_name", "perceived_colour_master_name", "main_category", "color"]
    ]
    known_customer_index = users.index
    known_article_index = items.index

    for output_path in [
        INTERACTIONS_FILE,
        USER_FEATURES_FILE,
        ITEM_FEATURES_FILE,
        SEGMENT_CANDIDATES_FILE,
        TRAIN_DATA_FILE,
        MANIFEST_FILE,
    ]:
        if output_path.exists():
            output_path.unlink()

    first_interaction_chunk = True
    user_history: Dict[str, list[str]] = {}
    transaction_rows_seen = 0
    skipped_customer_rows = 0
    skipped_article_rows = 0
    aggregate_start = time.perf_counter()
    for chunk in iterate_transaction_chunks(transactions_path):
        transaction_rows_seen += len(chunk)
        customer_mask = chunk["customer_id"].isin(known_customer_index)
        article_mask = chunk["article_id"].isin(known_article_index)
        skipped_customer_rows += int((~customer_mask).sum())
        skipped_article_rows += int((customer_mask & ~article_mask).sum())
        chunk = chunk.loc[customer_mask & article_mask].copy()
        if chunk.empty:
            continue

        chunk = chunk.merge(article_lookup, on="article_id", how="left", validate="many_to_one")
        for column in ["garment_group_name", "perceived_colour_master_name", "main_category", "color"]:
            chunk[column] = chunk[column].fillna(UNKNOWN_VALUE)
        chunk["season"] = season_from_dates(chunk["t_dat"])
        chunk["buyer_age_bucket"] = chunk["customer_id"].map(customer_age_lookup).fillna("unknown")

        write_interaction_chunk(
            chunk=chunk,
            output_path=INTERACTIONS_FILE,
            valid_start=valid_start,
            test_start=test_start,
            is_first_chunk=first_interaction_chunk,
            user_history=user_history,
        )
        first_interaction_chunk = False

        user_grouped = chunk.groupby("customer_id", sort=False).agg(
            purchase_count=("price", "size"),
            total_spend=("price", "sum"),
            min_price=("price", "min"),
            max_price=("price", "max"),
            first_purchase_date=("t_dat", "min"),
            last_purchase_date=("t_dat", "max"),
            channel_1_count=("sales_channel_id", lambda s: int((s == 1).sum())),
            channel_2_count=("sales_channel_id", lambda s: int((s == 2).sum())),
        )
        update_user_aggregates(users, user_grouped)

        item_grouped = chunk.groupby("article_id", sort=False).agg(
            item_purchase_count=("price", "size"),
            item_total_spend=("price", "sum"),
            item_min_price=("price", "min"),
            item_max_price=("price", "max"),
            first_purchase_date=("t_dat", "min"),
            last_purchase_date=("t_dat", "max"),
            channel_1_count=("sales_channel_id", lambda s: int((s == 1).sum())),
            channel_2_count=("sales_channel_id", lambda s: int((s == 2).sum())),
        )
        update_item_aggregates(items, item_grouped)

        for cutoff, user_count_col, user_spend_col, item_count_col in [
            (cutoff_7, "purchase_count_7d", "spend_7d", "item_purchase_count_7d"),
            (cutoff_30, "purchase_count_30d", "spend_30d", "item_purchase_count_30d"),
            (cutoff_90, "purchase_count_90d", "spend_90d", "item_purchase_count_90d"),
        ]:
            recent = chunk.loc[chunk["t_dat"].ge(cutoff)]
            if recent.empty:
                continue
            user_recent = recent.groupby("customer_id", sort=False).agg(
                purchase_count=("price", "size"),
                total_spend=("price", "sum"),
            )
            append_window_aggregates(users, user_recent, user_count_col, user_spend_col)
            item_recent = recent.groupby("article_id", sort=False).agg(
                purchase_count=("price", "size"),
            )
            append_item_window_counts(items, item_recent, item_count_col)

        for column_name, target in [
            ("garment_group_name", user_garment_counts),
            ("perceived_colour_master_name", user_colour_counts),
            ("main_category", user_main_category_counts),
            ("season", user_season_counts),
        ]:
            grouped = chunk.groupby(["customer_id", column_name]).size().unstack(fill_value=0)
            grouped = grouped.reindex(columns=target.columns, fill_value=0)
            add_group_counts(target, grouped)

        age_grouped = chunk.groupby(["article_id", "buyer_age_bucket"]).size().unstack(fill_value=0)
        age_grouped = age_grouped.reindex(columns=item_age_bucket_counts.columns, fill_value=0)
        add_group_counts(item_age_bucket_counts, age_grouped)

        season_grouped = chunk.groupby(["article_id", "season"]).size().unstack(fill_value=0)
        season_grouped = season_grouped.reindex(columns=item_season_counts.columns, fill_value=0)
        add_group_counts(item_season_counts, season_grouped)

        if transaction_rows_seen % LOG_EVERY_N_ROWS == 0:
            logging.info(
                "stage=candidate_feature_progress rows_scanned=%s unique_users=%s unique_items=%s",
                transaction_rows_seen,
                int(users["purchase_count"].gt(0).sum()),
                int(items["item_purchase_count"].gt(0).sum()),
            )

    log_stage(
        "aggregate_candidate_features",
        aggregate_start,
        rows_scanned=transaction_rows_seen,
        skipped_customer_rows=skipped_customer_rows,
        skipped_article_rows=skipped_article_rows,
    )

    user_features = finalize_user_features(
        users=users,
        garment_counts=user_garment_counts,
        colour_counts=user_colour_counts,
        main_category_counts=user_main_category_counts,
        season_counts=user_season_counts,
        reference_date=profile.max_date,
    )
    item_features = finalize_item_features(
        items=items,
        age_bucket_counts=item_age_bucket_counts,
        season_counts=item_season_counts,
        reference_date=profile.max_date,
    )

    user_features.to_csv(USER_FEATURES_FILE, index=False, compression="gzip")
    item_features.to_csv(ITEM_FEATURES_FILE, index=False, compression="gzip")
    build_segment_candidates(item_features, SEGMENT_CANDIDATES_FILE)
    candidate_rows = build_candidate_train_data(INTERACTIONS_FILE, user_features, item_features, TRAIN_DATA_FILE)
    write_manifest(profile, candidate_rows)

    log_stage(
        "candidate_training_data_complete",
        run_start,
        mode=RUNTIME_MODE,
        user_feature_rows=len(user_features),
        item_feature_rows=len(item_features),
        candidate_train_rows=candidate_rows,
    )


if __name__ == "__main__":
    main()
