import csv
import logging
import os
import random
import time
from datetime import datetime, timedelta
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
        "SIM_USERS_FILE": BASE_DIR / "data" / "processed" / "sim_users_test.csv",
        "ITEM_MASTER_FILE": BASE_DIR / "data" / "processed" / "item_master_test.csv",
        "ITEM_PERSONA_FILE": BASE_DIR / "data" / "processed" / "item_persona_scores_test.csv",
        "OUTPUT_FILE": BASE_DIR / "data" / "processed" / "simulated_events_test.csv",
        "TARGET_EVENTS": 20000,
        "PERSONA_POOL_LIMIT": 1000,
        "RANDOM_SEED": 42,
        "START_TIMESTAMP": "2025-01-01T09:00:00",
    },
    "dev": {
        "SIM_USERS_FILE": BASE_DIR / "data" / "processed" / "sim_users_dev.csv",
        "ITEM_MASTER_FILE": BASE_DIR / "data" / "processed" / "item_master_dev.csv",
        "ITEM_PERSONA_FILE": BASE_DIR / "data" / "processed" / "item_persona_scores_dev.csv",
        "OUTPUT_FILE": BASE_DIR / "data" / "processed" / "simulated_events_dev.csv",
        "TARGET_EVENTS": 200_000,
        "PERSONA_POOL_LIMIT": 3000,
        "RANDOM_SEED": 42,
        "START_TIMESTAMP": "2025-01-01T09:00:00",
    },
    "production": {
        "SIM_USERS_FILE": BASE_DIR / "data" / "processed" / "sim_users.csv",
        "ITEM_MASTER_FILE": BASE_DIR / "data" / "processed" / "item_master.csv",
        "ITEM_PERSONA_FILE": BASE_DIR / "data" / "processed" / "item_persona_scores.csv",
        "OUTPUT_FILE": BASE_DIR / "data" / "processed" / "simulated_events.csv",
        "TARGET_EVENTS": 1_000_000,
        "PERSONA_POOL_LIMIT": 5000,
        "RANDOM_SEED": 42,
        "START_TIMESTAMP": "2025-01-01T09:00:00",
    },
}

RUNTIME_MODE = os.getenv("DATA_PIPELINE_MODE", MODE).strip().lower()

if RUNTIME_MODE not in MODE_CONFIG:
    raise ValueError(f"Unsupported MODE: {RUNTIME_MODE}")

CONFIG = MODE_CONFIG[RUNTIME_MODE]
SIM_USERS_FILE: Path = CONFIG["SIM_USERS_FILE"]
ITEM_MASTER_FILE: Path = CONFIG["ITEM_MASTER_FILE"]
ITEM_PERSONA_FILE: Path = CONFIG["ITEM_PERSONA_FILE"]
OUTPUT_FILE: Path = CONFIG["OUTPUT_FILE"]
TARGET_EVENTS: int = CONFIG["TARGET_EVENTS"]
PERSONA_POOL_LIMIT: int = CONFIG["PERSONA_POOL_LIMIT"]
RANDOM_SEED: int = CONFIG["RANDOM_SEED"]
START_TIMESTAMP: datetime = datetime.fromisoformat(CONFIG["START_TIMESTAMP"])

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

REQUIRED_SIM_USERS_COLUMNS = ["user_id", "top_persona"] + [f"{persona_name}_ratio" for persona_name in PERSONAS]
REQUIRED_ITEM_MASTER_COLUMNS = [
    "article_id",
    "category_l1",
    "category_l2",
    "category_l3",
    "colour_group_name",
    "price_mean",
    "price_bucket",
]
REQUIRED_ITEM_PERSONA_COLUMNS = ["article_id"] + [f"{persona_name}_ratio" for persona_name in PERSONAS]

OUTPUT_COLUMNS = [
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


def parse_float(raw_value: str) -> float:
    value = (raw_value or "").strip()
    if not value:
        return 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0


def normalize_text(value: str) -> str:
    return " ".join((value or "").strip().split())


def load_sim_users(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        return [row for row in reader if (row.get("user_id") or "").strip()]


def load_item_master(path: Path) -> Dict[str, dict[str, str]]:
    item_master: Dict[str, dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            article_id = (row.get("article_id") or "").strip()
            if not article_id:
                continue
            item_master[article_id] = row
    return item_master


def load_item_persona_scores(path: Path) -> Dict[str, dict[str, str]]:
    persona_scores: Dict[str, dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            article_id = (row.get("article_id") or "").strip()
            if not article_id:
                continue
            persona_scores[article_id] = row
    return persona_scores


def build_item_records(
    item_master: Dict[str, dict[str, str]],
    item_persona_scores: Dict[str, dict[str, str]],
) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for article_id, item_row in item_master.items():
        persona_row = item_persona_scores.get(article_id)
        if persona_row is None:
            continue
        record = dict(item_row)
        for persona_name in PERSONAS:
            record[f"{persona_name}_ratio"] = persona_row.get(f"{persona_name}_ratio", "0")
        records.append(record)
    return records


def build_persona_pools(item_records: list[dict[str, str]]) -> Dict[str, list[dict[str, str]]]:
    pools: Dict[str, list[dict[str, str]]] = {}
    for persona_name in PERSONAS:
        sorted_items = sorted(
            item_records,
            key=lambda row: parse_float(row.get(f"{persona_name}_ratio", "0")),
            reverse=True,
        )
        pools[persona_name] = sorted_items[:PERSONA_POOL_LIMIT]
    return pools


def pick_active_persona(user_row: dict[str, str], rng: random.Random) -> str:
    weights = [parse_float(user_row.get(f"{persona_name}_ratio", "0")) for persona_name in PERSONAS]
    if sum(weights) <= 0.0:
        return normalize_text(user_row.get("top_persona", "")) or PERSONAS[0]
    return rng.choices(PERSONAS, weights=weights, k=1)[0]


def category_match(item_row: dict[str, str], preferred_categories: list[str]) -> bool:
    if not preferred_categories:
        return False
    item_categories = {
        normalize_text(item_row.get("category_l1", "")).lower(),
        normalize_text(item_row.get("category_l2", "")).lower(),
        normalize_text(item_row.get("category_l3", "")).lower(),
    }
    return any(normalize_text(category).lower() in item_categories for category in preferred_categories)


def color_match(item_row: dict[str, str], preferred_colors: list[str]) -> bool:
    if not preferred_colors:
        return False
    item_color = normalize_text(item_row.get("colour_group_name", "")).lower()
    return any(normalize_text(color).lower() == item_color for color in preferred_colors)


def choose_candidate_items(
    persona_name: str,
    persona_cfg: dict,
    persona_pools: Dict[str, list[dict[str, str]]],
    rng: random.Random,
    count: int,
) -> list[dict[str, str]]:
    base_pool = persona_pools.get(persona_name, [])
    if not base_pool:
        return []

    preferred_categories = list(persona_cfg.get("preferred_categories", []))
    preferred_colors = list(persona_cfg.get("preferred_colors", []))

    filtered = [
        item for item in base_pool
        if category_match(item, preferred_categories) or color_match(item, preferred_colors)
    ]
    candidate_pool = filtered if filtered else base_pool[: min(len(base_pool), 300)]
    candidate_pool = candidate_pool[: min(len(candidate_pool), 300)]
    if not candidate_pool:
        return []

    sample_count = min(count, len(candidate_pool))
    return rng.sample(candidate_pool, sample_count)


def build_query_text(persona_name: str, persona_cfg: dict, rng: random.Random) -> str:
    preferred_categories = list(persona_cfg.get("preferred_categories", []))
    preferred_colors = list(persona_cfg.get("preferred_colors", []))
    category = rng.choice(preferred_categories) if preferred_categories else "fashion"
    color = rng.choice(preferred_colors) if preferred_colors else ""

    prefix_map = {
        "trendsetter": "trending",
        "practical": "basic",
        "value": "affordable",
        "brand_loyal": "favorite",
        "impulse": "popular",
        "careful": "quality",
        "repeat_stable": "reorder",
        "color_focus": "",
        "category_focus": "",
    }
    prefix = prefix_map.get(persona_name, "")

    parts = [part for part in [prefix, color, category] if normalize_text(part)]
    return " ".join(parts).strip() or category


def make_search_row(
    event_id: int,
    timestamp: datetime,
    user_row: dict[str, str],
    session_id: str,
    persona_name: str,
    query_text: str,
) -> dict[str, str]:
    return {
        "event_id": str(event_id),
        "timestamp": timestamp.isoformat(),
        "user_id": user_row["user_id"],
        "session_id": session_id,
        "event_type": "search",
        "query_text": query_text,
        "article_id": "",
        "active_persona": persona_name,
        "top_persona": user_row.get("top_persona", ""),
        "category_l1": "",
        "category_l2": "",
        "category_l3": "",
        "colour_group_name": "",
        "price_mean": "",
        "price_bucket": "",
    }


def make_item_event_row(
    event_id: int,
    timestamp: datetime,
    user_row: dict[str, str],
    session_id: str,
    persona_name: str,
    event_type: str,
    item_row: dict[str, str],
) -> dict[str, str]:
    return {
        "event_id": str(event_id),
        "timestamp": timestamp.isoformat(),
        "user_id": user_row["user_id"],
        "session_id": session_id,
        "event_type": event_type,
        "query_text": "",
        "article_id": item_row.get("article_id", ""),
        "active_persona": persona_name,
        "top_persona": user_row.get("top_persona", ""),
        "category_l1": item_row.get("category_l1", ""),
        "category_l2": item_row.get("category_l2", ""),
        "category_l3": item_row.get("category_l3", ""),
        "colour_group_name": item_row.get("colour_group_name", ""),
        "price_mean": item_row.get("price_mean", ""),
        "price_bucket": item_row.get("price_bucket", ""),
    }


def write_simulated_events(
    sim_users: list[dict[str, str]],
    persona_pools: Dict[str, list[dict[str, str]]],
    config: dict,
    output_path: Path,
) -> None:
    start_time = time.perf_counter()
    rng = random.Random(RANDOM_SEED)
    simulation_cfg = config.get("simulation", {})
    personas_cfg = config.get("personas", {})
    cycle_delay_seconds = int(simulation_cfg.get("cycle_delay_seconds", 5))

    current_timestamp = START_TIMESTAMP
    event_id = 1
    session_number = 1
    rows_written = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()

        while rows_written < TARGET_EVENTS:
            user_row = rng.choice(sim_users)
            active_persona = pick_active_persona(user_row, rng)
            persona_cfg = personas_cfg.get(active_persona, {})
            session_id = f"session_{session_number:08d}"
            session_number += 1

            session_searches = max(1, int(persona_cfg.get("session_searches", 3)))
            search_prob = float(persona_cfg.get("search_prob", 0.4))
            view_prob = float(persona_cfg.get("view_prob", 0.7))
            cart_prob = float(persona_cfg.get("cart_prob", 0.3))
            purchase_prob = float(persona_cfg.get("purchase_prob", 0.2))

            for _ in range(session_searches):
                if rows_written >= TARGET_EVENTS:
                    break

                if rng.random() <= search_prob:
                    query_text = build_query_text(active_persona, persona_cfg, rng)
                    writer.writerow(
                        make_search_row(
                            event_id=event_id,
                            timestamp=current_timestamp,
                            user_row=user_row,
                            session_id=session_id,
                            persona_name=active_persona,
                            query_text=query_text,
                        )
                    )
                    event_id += 1
                    rows_written += 1
                    current_timestamp += timedelta(seconds=rng.randint(5, 20))

                candidate_items = choose_candidate_items(
                    active_persona,
                    persona_cfg,
                    persona_pools,
                    rng,
                    count=rng.randint(1, 3),
                )

                for item_row in candidate_items:
                    if rows_written >= TARGET_EVENTS:
                        break

                    if rng.random() <= view_prob:
                        writer.writerow(
                            make_item_event_row(
                                event_id=event_id,
                                timestamp=current_timestamp,
                                user_row=user_row,
                                session_id=session_id,
                                persona_name=active_persona,
                                event_type="view",
                                item_row=item_row,
                            )
                        )
                        event_id += 1
                        rows_written += 1
                        current_timestamp += timedelta(seconds=rng.randint(3, 15))

                        if rows_written >= TARGET_EVENTS:
                            break

                        if rng.random() <= cart_prob:
                            writer.writerow(
                                make_item_event_row(
                                    event_id=event_id,
                                    timestamp=current_timestamp,
                                    user_row=user_row,
                                    session_id=session_id,
                                    persona_name=active_persona,
                                    event_type="cart",
                                    item_row=item_row,
                                )
                            )
                            event_id += 1
                            rows_written += 1
                            current_timestamp += timedelta(seconds=rng.randint(3, 20))

                            if rows_written >= TARGET_EVENTS:
                                break

                            if rng.random() <= purchase_prob:
                                writer.writerow(
                                    make_item_event_row(
                                        event_id=event_id,
                                        timestamp=current_timestamp,
                                        user_row=user_row,
                                        session_id=session_id,
                                        persona_name=active_persona,
                                        event_type="purchase",
                                        item_row=item_row,
                                    )
                                )
                                event_id += 1
                                rows_written += 1
                                current_timestamp += timedelta(seconds=rng.randint(10, 40))

            current_timestamp += timedelta(seconds=max(1, cycle_delay_seconds) * rng.randint(5, 20))

    elapsed = time.perf_counter() - start_time
    logging.info(
        "simulated_events_complete output=%s rows_written=%s elapsed_seconds=%.2f",
        output_path,
        rows_written,
        elapsed,
    )


def main() -> None:
    configure_logging()
    resolve_required_file(SIM_USERS_FILE, "sim users file")
    resolve_required_file(ITEM_MASTER_FILE, "item master file")
    resolve_required_file(ITEM_PERSONA_FILE, "item persona scores file")
    resolve_required_file(CONFIG_FILE, "persona config file")
    validate_required_columns(SIM_USERS_FILE, REQUIRED_SIM_USERS_COLUMNS)
    validate_required_columns(ITEM_MASTER_FILE, REQUIRED_ITEM_MASTER_COLUMNS)
    validate_required_columns(ITEM_PERSONA_FILE, REQUIRED_ITEM_PERSONA_COLUMNS)

    config = load_config(CONFIG_FILE)
    sim_users = load_sim_users(SIM_USERS_FILE)
    item_master = load_item_master(ITEM_MASTER_FILE)
    item_persona_scores = load_item_persona_scores(ITEM_PERSONA_FILE)
    item_records = build_item_records(item_master, item_persona_scores)
    persona_pools = build_persona_pools(item_records)
    write_simulated_events(sim_users, persona_pools, config, OUTPUT_FILE)


if __name__ == "__main__":
    main()
