import csv
import logging
import os
import random
import time
from pathlib import Path
from typing import Dict, Sequence

from config_loader import load_yaml_config

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_FILE = BASE_DIR / "simulator" / "persona_config_9.yaml"

# MODE = "production"
# MODE = "dev"
MODE = "test"

MODE_CONFIG = {
    "test": {
        "USER_PERSONA_FILE": BASE_DIR / "data" / "processed" / "user_persona_scores_test.csv",
        "CUSTOMER_PROFILE_FILE": BASE_DIR / "data" / "processed" / "customer_purchase_profile_test.csv",
        "OUTPUT_FILE": BASE_DIR / "data" / "processed" / "sim_users_test.csv",
        "RANDOM_SEED": 42,
    },
    "dev": {
        "USER_PERSONA_FILE": BASE_DIR / "data" / "processed" / "user_persona_scores_dev.csv",
        "CUSTOMER_PROFILE_FILE": BASE_DIR / "data" / "processed" / "customer_purchase_profile_dev.csv",
        "OUTPUT_FILE": BASE_DIR / "data" / "processed" / "sim_users_dev.csv",
        "RANDOM_SEED": 42,
    },
    "production": {
        "USER_PERSONA_FILE": BASE_DIR / "data" / "processed" / "user_persona_scores.csv",
        "CUSTOMER_PROFILE_FILE": BASE_DIR / "data" / "processed" / "customer_purchase_profile.csv",
        "OUTPUT_FILE": BASE_DIR / "data" / "processed" / "sim_users.csv",
        "RANDOM_SEED": 42,
    },
}

RUNTIME_MODE = os.getenv("DATA_PIPELINE_MODE", MODE).strip().lower()

if RUNTIME_MODE not in MODE_CONFIG:
    raise ValueError(f"Unsupported MODE: {RUNTIME_MODE}")

CONFIG = MODE_CONFIG[RUNTIME_MODE]
USER_PERSONA_FILE: Path = CONFIG["USER_PERSONA_FILE"]
CUSTOMER_PROFILE_FILE: Path = CONFIG["CUSTOMER_PROFILE_FILE"]
OUTPUT_FILE: Path = CONFIG["OUTPUT_FILE"]
RANDOM_SEED: int = CONFIG["RANDOM_SEED"]

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

REQUIRED_USER_PERSONA_COLUMNS = ["customer_id", "top_persona", "top_persona_ratio"] + [
    f"{persona_name}_ratio" for persona_name in PERSONAS
]
REQUIRED_CUSTOMER_PROFILE_COLUMNS = [
    "customer_id",
    "age",
    "age_bucket",
    "fashion_news_frequency",
    "club_member_status",
]

OUTPUT_COLUMNS = [
    "user_id",
    "age",
    "age_bucket",
    "fashion_news_frequency",
    "club_member_status",
    "top_persona",
    "top_persona_ratio",
] + [f"{persona_name}_ratio" for persona_name in PERSONAS]


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


def resolve_required_file(file_path: Path, description: str) -> Path:
    if file_path.exists():
        return file_path
    raise FileNotFoundError(f"Missing {description}: {file_path}")


def load_config(path: Path) -> dict:
    return load_yaml_config(path)


def load_customer_profile(path: Path) -> Dict[str, dict[str, str]]:
    customer_profile: Dict[str, dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            customer_id = (row.get("customer_id") or "").strip()
            if not customer_id:
                continue
            customer_profile[customer_id] = row
    return customer_profile


def load_user_persona_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        return [row for row in reader if (row.get("customer_id") or "").strip()]


def write_sim_users(
    user_rows: list[dict[str, str]],
    customer_profile: Dict[str, dict[str, str]],
    user_pool_size: int,
    output_path: Path,
) -> None:
    start_time = time.perf_counter()
    rng = random.Random(RANDOM_SEED)

    joined_rows = [
        row for row in user_rows if (row.get("customer_id") or "").strip() in customer_profile
    ]
    if not joined_rows:
        raise ValueError("No joinable rows found between user_persona_scores and customer_purchase_profile")

    sample_size = min(user_pool_size, len(joined_rows))
    sampled_rows = joined_rows if sample_size == len(joined_rows) else rng.sample(joined_rows, sample_size)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()

        for row in sampled_rows:
            customer_id = (row.get("customer_id") or "").strip()
            profile = customer_profile[customer_id]
            output_row = {
                "user_id": customer_id,
                "age": profile.get("age", ""),
                "age_bucket": profile.get("age_bucket", ""),
                "fashion_news_frequency": profile.get("fashion_news_frequency", ""),
                "club_member_status": profile.get("club_member_status", ""),
                "top_persona": row.get("top_persona", ""),
                "top_persona_ratio": row.get("top_persona_ratio", ""),
            }
            for persona_name in PERSONAS:
                output_row[f"{persona_name}_ratio"] = row.get(f"{persona_name}_ratio", "")
            writer.writerow(output_row)

    elapsed = time.perf_counter() - start_time
    logging.info(
        "sim_users_complete output=%s rows_written=%s elapsed_seconds=%.2f",
        output_path,
        sample_size,
        elapsed,
    )


def main() -> None:
    configure_logging()
    resolve_required_file(USER_PERSONA_FILE, "user persona scores file")
    resolve_required_file(CUSTOMER_PROFILE_FILE, "customer purchase profile file")
    resolve_required_file(CONFIG_FILE, "persona config file")
    validate_required_columns(USER_PERSONA_FILE, REQUIRED_USER_PERSONA_COLUMNS)
    validate_required_columns(CUSTOMER_PROFILE_FILE, REQUIRED_CUSTOMER_PROFILE_COLUMNS)

    config = load_config(CONFIG_FILE)
    user_pool_size = int(config.get("simulation", {}).get("user_pool_size", 10000))

    user_rows = load_user_persona_rows(USER_PERSONA_FILE)
    customer_profile = load_customer_profile(CUSTOMER_PROFILE_FILE)
    write_sim_users(user_rows, customer_profile, user_pool_size, OUTPUT_FILE)


if __name__ == "__main__":
    main()
