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
        "OUTPUT_FILE": BASE_DIR / "data" / "processed" / "simulated_events_validation_test.json",
    },
    "dev": {
        "INPUT_FILE": BASE_DIR / "data" / "processed" / "simulated_events_dev.csv",
        "OUTPUT_FILE": BASE_DIR / "data" / "processed" / "simulated_events_validation_dev.json",
    },
    "production": {
        "INPUT_FILE": BASE_DIR / "data" / "processed" / "simulated_events.csv",
        "OUTPUT_FILE": BASE_DIR / "data" / "processed" / "simulated_events_validation.json",
    },
}

RUNTIME_MODE = os.getenv("DATA_PIPELINE_MODE", MODE).strip().lower()

if RUNTIME_MODE not in MODE_CONFIG:
    raise ValueError(f"Unsupported MODE: {RUNTIME_MODE}")

CONFIG = MODE_CONFIG[RUNTIME_MODE]
INPUT_FILE: Path = CONFIG["INPUT_FILE"]
OUTPUT_FILE: Path = CONFIG["OUTPUT_FILE"]

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


def summarize_events(path: Path) -> dict:
    start_time = time.perf_counter()
    event_type_counts: Counter[str] = Counter()
    persona_counts: Counter[str] = Counter()
    unique_users: set[str] = set()
    unique_sessions: set[str] = set()
    missing_search_query_rows = 0
    missing_item_rows = 0
    first_timestamp: datetime | None = None
    last_timestamp: datetime | None = None
    row_count = 0

    with path.open(newline="", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            row_count += 1
            event_type = (row.get("event_type") or "").strip()
            event_type_counts[event_type] += 1
            persona_counts[(row.get("active_persona") or "").strip()] += 1

            user_id = (row.get("user_id") or "").strip()
            session_id = (row.get("session_id") or "").strip()
            if user_id:
                unique_users.add(user_id)
            if session_id:
                unique_sessions.add(session_id)

            if event_type == "search" and not (row.get("query_text") or "").strip():
                missing_search_query_rows += 1

            if event_type in {"view", "cart", "purchase"} and not (row.get("article_id") or "").strip():
                missing_item_rows += 1

            raw_timestamp = (row.get("timestamp") or "").strip()
            if raw_timestamp:
                timestamp = datetime.fromisoformat(raw_timestamp)
                if first_timestamp is None or timestamp < first_timestamp:
                    first_timestamp = timestamp
                if last_timestamp is None or timestamp > last_timestamp:
                    last_timestamp = timestamp

    summary = {
        "mode": RUNTIME_MODE,
        "input_file": str(path),
        "row_count": row_count,
        "unique_users": len(unique_users),
        "unique_sessions": len(unique_sessions),
        "event_type_counts": dict(event_type_counts),
        "active_persona_counts": dict(persona_counts),
        "missing_search_query_rows": missing_search_query_rows,
        "missing_item_rows": missing_item_rows,
        "first_timestamp": first_timestamp.isoformat() if first_timestamp else "",
        "last_timestamp": last_timestamp.isoformat() if last_timestamp else "",
        "elapsed_seconds": round(time.perf_counter() - start_time, 4),
    }
    return summary


def main() -> None:
    configure_logging()
    resolve_required_file(INPUT_FILE, "simulated events file")
    validate_required_columns(INPUT_FILE, REQUIRED_COLUMNS)
    summary = summarize_events(INPUT_FILE)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8") as outfile:
        json.dump(summary, outfile, ensure_ascii=False, indent=2)

    logging.info(
        "simulated_events_validation_complete output=%s row_count=%s unique_users=%s unique_sessions=%s",
        OUTPUT_FILE,
        summary["row_count"],
        summary["unique_users"],
        summary["unique_sessions"],
    )


if __name__ == "__main__":
    main()
