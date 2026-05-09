import csv
import json
import logging
import os
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Sequence


BASE_DIR = Path(__file__).resolve().parent.parent

# MODE = "production"
# MODE = "dev"
MODE = "test"

MODE_CONFIG = {
    "test": {
        "INPUT_FILE": BASE_DIR / "data" / "processed" / "simulated_events_test.csv",
        "TRAIN_FILE": BASE_DIR / "data" / "processed" / "train_events_test.csv",
        "VALID_FILE": BASE_DIR / "data" / "processed" / "valid_events_test.csv",
        "TEST_FILE": BASE_DIR / "data" / "processed" / "test_events_test.csv",
        "SUMMARY_FILE": BASE_DIR / "data" / "processed" / "event_split_summary_test.json",
    },
    "dev": {
        "INPUT_FILE": BASE_DIR / "data" / "processed" / "simulated_events_dev.csv",
        "TRAIN_FILE": BASE_DIR / "data" / "processed" / "train_events_dev.csv",
        "VALID_FILE": BASE_DIR / "data" / "processed" / "valid_events_dev.csv",
        "TEST_FILE": BASE_DIR / "data" / "processed" / "test_events_dev.csv",
        "SUMMARY_FILE": BASE_DIR / "data" / "processed" / "event_split_summary_dev.json",
    },
    "production": {
        "INPUT_FILE": BASE_DIR / "data" / "processed" / "simulated_events.csv",
        "TRAIN_FILE": BASE_DIR / "data" / "processed" / "train_events.csv",
        "VALID_FILE": BASE_DIR / "data" / "processed" / "valid_events.csv",
        "TEST_FILE": BASE_DIR / "data" / "processed" / "test_events.csv",
        "SUMMARY_FILE": BASE_DIR / "data" / "processed" / "event_split_summary.json",
    },
}

RUNTIME_MODE = os.getenv("DATA_PIPELINE_MODE", MODE).strip().lower()

if RUNTIME_MODE not in MODE_CONFIG:
    raise ValueError(f"Unsupported MODE: {RUNTIME_MODE}")

CONFIG = MODE_CONFIG[RUNTIME_MODE]
INPUT_FILE: Path = CONFIG["INPUT_FILE"]
TRAIN_FILE: Path = CONFIG["TRAIN_FILE"]
VALID_FILE: Path = CONFIG["VALID_FILE"]
TEST_FILE: Path = CONFIG["TEST_FILE"]
SUMMARY_FILE: Path = CONFIG["SUMMARY_FILE"]

REQUIRED_COLUMNS = [
    "event_id",
    "timestamp",
    "user_id",
    "session_id",
    "event_type",
    "query_text",
    "article_id",
    "active_persona",
    "top_persona",
    "category_l1",
    "category_l2",
    "category_l3",
    "colour_group_name",
    "price_mean",
    "price_bucket",
]

TRAIN_RATIO = 0.8
VALID_RATIO = 0.1
TEST_RATIO = 0.1


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


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


def load_rows(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open(newline="", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            timestamp = datetime.fromisoformat((row.get("timestamp") or "").strip())
            row["_parsed_timestamp"] = timestamp.isoformat()
            rows.append(row)
    rows.sort(key=lambda row: row["_parsed_timestamp"])
    return rows


def compute_boundaries(row_count: int) -> tuple[int, int]:
    train_end = int(row_count * TRAIN_RATIO)
    valid_end = train_end + int(row_count * VALID_RATIO)

    if row_count >= 3:
        train_end = max(1, train_end)
        valid_end = max(train_end + 1, valid_end)
        valid_end = min(valid_end, row_count - 1)

    return train_end, valid_end


def split_rows(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    train_end, valid_end = compute_boundaries(len(rows))
    train_rows = rows[:train_end]
    valid_rows = rows[train_end:valid_end]
    test_rows = rows[valid_end:]
    return train_rows, valid_rows, test_rows


def strip_internal_fields(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    cleaned_rows: list[dict[str, str]] = []
    for row in rows:
        cleaned = {key: value for key, value in row.items() if not key.startswith("_")}
        cleaned_rows.append(cleaned)
    return cleaned_rows


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_split(name: str, rows: list[dict[str, str]]) -> dict:
    event_type_counts: Counter[str] = Counter()
    unique_users: set[str] = set()
    unique_sessions: set[str] = set()
    first_timestamp = ""
    last_timestamp = ""

    for index, row in enumerate(rows):
        event_type_counts[(row.get("event_type") or "").strip()] += 1
        user_id = (row.get("user_id") or "").strip()
        session_id = (row.get("session_id") or "").strip()
        if user_id:
            unique_users.add(user_id)
        if session_id:
            unique_sessions.add(session_id)
        if index == 0:
            first_timestamp = row.get("timestamp", "")
        last_timestamp = row.get("timestamp", "")

    return {
        "split": name,
        "row_count": len(rows),
        "unique_users": len(unique_users),
        "unique_sessions": len(unique_sessions),
        "event_type_counts": dict(event_type_counts),
        "first_timestamp": first_timestamp,
        "last_timestamp": last_timestamp,
    }


def main() -> None:
    configure_logging()
    start_time = time.perf_counter()
    resolve_required_file(INPUT_FILE, "simulated events file")
    validate_required_columns(INPUT_FILE, REQUIRED_COLUMNS)

    rows = load_rows(INPUT_FILE)
    if not rows:
        raise ValueError("simulated_events file does not contain any rows")

    train_rows, valid_rows, test_rows = split_rows(rows)
    train_clean = strip_internal_fields(train_rows)
    valid_clean = strip_internal_fields(valid_rows)
    test_clean = strip_internal_fields(test_rows)

    write_csv(TRAIN_FILE, train_clean, REQUIRED_COLUMNS)
    write_csv(VALID_FILE, valid_clean, REQUIRED_COLUMNS)
    write_csv(TEST_FILE, test_clean, REQUIRED_COLUMNS)

    summary = {
        "mode": RUNTIME_MODE,
        "input_file": str(INPUT_FILE),
        "ratios": {
            "train": TRAIN_RATIO,
            "valid": VALID_RATIO,
            "test": TEST_RATIO,
        },
        "train": summarize_split("train", train_clean),
        "valid": summarize_split("valid", valid_clean),
        "test": summarize_split("test", test_clean),
        "elapsed_seconds": round(time.perf_counter() - start_time, 4),
    }

    with SUMMARY_FILE.open("w", encoding="utf-8") as outfile:
        json.dump(summary, outfile, ensure_ascii=False, indent=2)

    logging.info(
        "event_split_complete train_rows=%s valid_rows=%s test_rows=%s summary=%s",
        len(train_clean),
        len(valid_clean),
        len(test_clean),
        SUMMARY_FILE,
    )


if __name__ == "__main__":
    main()
