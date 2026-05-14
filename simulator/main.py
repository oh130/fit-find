"""
User behavior simulator driven by 9 peer personas.

This simulator reads ``persona_config_9.yaml`` and sends
search/view/cart/purchase events to the API gateway.
"""

from __future__ import annotations

import logging
import os
import random
import time
from pathlib import Path
import ast

try:
    import requests
except ModuleNotFoundError:
    requests = None

logging.basicConfig(
    level=logging.INFO,
    format="[Simulator] %(asctime)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "persona_config_9.yaml"

API_URL = os.getenv("API_GATEWAY_URL", "http://localhost:8000")
DRY_RUN = os.getenv("SIMULATOR_DRY_RUN", "0").strip().lower() in {"1", "true", "yes"}
MAX_SESSIONS = int(os.getenv("SIMULATOR_MAX_SESSIONS", "0"))
FORCE_ALL_PERSONAS = os.getenv(
    "SIMULATOR_FORCE_ALL_PERSONAS",
    "1" if DRY_RUN else "0",
).strip().lower() in {"1", "true", "yes"}
DEFAULT_INTER_EVENT_SECONDS = float(
    os.getenv("SIMULATOR_INTER_EVENT_SECONDS", "0.0" if DRY_RUN else "2.0")
)
DEFAULT_TOP_K = int(os.getenv("SIMULATOR_TOP_K", "10"))


CATEGORY_QUERY_TERMS = {
    "Ladieswear": ["dress", "blouse", "cardigan", "women outfit"],
    "Menswear": ["shirt", "pants", "hoodie", "men outfit"],
    "Sport": ["sportswear", "training wear", "activewear", "hoodie"],
    "Divided": ["streetwear", "casual outfit", "oversized top", "denim"],
    "Lingeries/Tights": ["lingerie", "tights", "underwear"],
    "Accessories": ["bag", "cap", "accessories"],
    "Kids": ["kids outfit", "baby wear", "child clothing"],
}

PERSONA_QUERY_PREFIXES = {
    "trendsetter": ["trending", "new", "latest"],
    "practical": ["basic", "classic", "everyday"],
    "value": ["affordable", "sale", "budget"],
    "brand_loyal": ["signature", "favorite", "core"],
    "impulse": ["popular", "must-have", "hot"],
    "careful": ["quality", "durable", "premium"],
    "repeat_stable": ["reorder", "everyday", "essential"],
    "color_focus": ["matching", "monochrome", "color"],
    "category_focus": ["focused", "core", "specialized"],
}


def _parse_scalar(raw_value: str):
    value = raw_value.strip()
    if not value:
        return ""

    if value.startswith("[") and value.endswith("]"):
        return ast.literal_eval(value)

    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False

    try:
        return int(value)
    except ValueError:
        pass

    try:
        return float(value)
    except ValueError:
        pass

    return value.strip("\"'")


def _load_simple_yaml(path: Path) -> dict:
    root: dict = {}
    stack: list[tuple[int, dict]] = [(-1, root)]

    with path.open(encoding="utf-8") as infile:
        for raw_line in infile:
            if not raw_line.strip() or raw_line.lstrip().startswith("#"):
                continue

            indent = len(raw_line) - len(raw_line.lstrip(" "))
            line = raw_line.strip()
            if ":" not in line:
                continue

            key, raw_value = line.split(":", 1)
            key = key.strip()
            value = raw_value.strip()

            while stack and indent <= stack[-1][0]:
                stack.pop()

            current = stack[-1][1]
            if not value:
                nested: dict = {}
                current[key] = nested
                stack.append((indent, nested))
            else:
                current[key] = _parse_scalar(value)

    return root


def load_config() -> dict:
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError:
        return _load_simple_yaml(CONFIG_PATH)

    with CONFIG_PATH.open(encoding="utf-8") as file:
        return yaml.safe_load(file)


def search(query: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
    if DRY_RUN:
        tokens = query.split()
        category = tokens[-1] if tokens else "fashion"
        return [
            {
                "article_id": f"dry_item_{index:03d}",
                "category": category,
            }
            for index in range(1, min(top_k, 5) + 1)
        ]

    if requests is None:
        raise RuntimeError("requests is required for non-dry-run simulator execution.")

    try:
        response = requests.post(
            f"{API_URL}/api/search",
            json={"query": query, "top_k": top_k},
            timeout=8.0,
        )
        response.raise_for_status()
        payload = response.json()
        return payload if isinstance(payload, list) else payload.get("results", [])
    except Exception as exc:
        logger.debug("Search error: %s", exc)
        return []


def send_event(
    user_id: str,
    event_type: str,
    item_id: str | None = None,
    category: str | None = None,
    query_text: str | None = None,
) -> bool:
    if DRY_RUN:
        return True

    if requests is None:
        raise RuntimeError("requests is required for non-dry-run simulator execution.")

    payload: dict[str, str] = {"user_id": user_id, "event_type": event_type}
    if item_id:
        payload["article_id"] = item_id
    if category:
        payload["category"] = category
    if query_text:
        payload["query_text"] = query_text

    try:
        response = requests.post(f"{API_URL}/api/events", json=payload, timeout=5.0)
        response.raise_for_status()
        return True
    except Exception as exc:
        logger.debug("Event error: %s", exc)
        return False


def gateway_alive() -> bool:
    if DRY_RUN:
        return True

    if requests is None:
        return False

    try:
        requests.get(f"{API_URL}/health", timeout=3.0).raise_for_status()
        return True
    except Exception:
        return False


def resolve_query_terms(categories: list[str]) -> list[str]:
    terms: list[str] = []
    for category in categories:
        terms.extend(CATEGORY_QUERY_TERMS.get(category, [category.lower()]))
    return terms or ["fashion"]


def build_query(
    persona_name: str,
    categories: list[str],
    colors: list[str],
    explicit_queries: list[str] | None = None,
) -> str:
    if explicit_queries:
        return random.choice(explicit_queries)

    prefix = random.choice(PERSONA_QUERY_PREFIXES.get(persona_name, ["fashion"]))
    color = random.choice(colors).lower() if colors else ""
    term = random.choice(resolve_query_terms(categories)).lower()
    return " ".join(part for part in [prefix, color, term] if part).strip()


class Persona:
    def __init__(self, name: str, cfg: dict, user_id: str):
        self.name = name
        self.user_id = user_id
        self.explicit_queries: list[str] = list(cfg.get("search_queries", []))
        self.categories: list[str] = list(cfg.get("categories", cfg.get("preferred_categories", [])))
        self.colors: list[str] = list(cfg.get("preferred_colors", []))
        self.search_prob: float = float(cfg.get("search_prob", 1.0))
        self.view_prob: float = float(cfg.get("view_prob", 0.7))
        self.cart_prob: float = float(cfg.get("cart_prob", 0.3))
        self.purchase_prob: float = float(cfg.get("purchase_prob", 0.2))
        self.inter_event_seconds: float = float(
            cfg.get("inter_event_seconds", DEFAULT_INTER_EVENT_SECONDS)
        )
        self.session_searches: int = max(1, int(cfg.get("session_searches", 3)))

    def run_session(self) -> int:
        total_events = 0

        for _ in range(self.session_searches):
            if random.random() > self.search_prob:
                time.sleep(self.inter_event_seconds)
                continue

            query = build_query(self.name, self.categories, self.colors, self.explicit_queries)
            if send_event(self.user_id, "search", query_text=query):
                total_events += 1
                logger.info(
                    "%-18s search    uid=%-8s query=%s",
                    f"[{self.name}]",
                    self.user_id[:8],
                    query,
                )

            results = search(query)
            time.sleep(self.inter_event_seconds)
            if not results:
                continue

            for item in results:
                if random.random() > self.view_prob:
                    continue

                item_id = str(
                    item.get("article_id")
                    or item.get("product_id")
                    or item.get("id")
                    or "unknown"
                )
                category = str(item.get("category") or random.choice(self.categories or ["fashion"]))

                if send_event(self.user_id, "view", item_id=item_id, category=category):
                    total_events += 1
                    logger.info(
                        "%-18s view      uid=%-8s item=%s cat=%s",
                        f"[{self.name}]",
                        self.user_id[:8],
                        item_id,
                        category,
                    )
                time.sleep(self.inter_event_seconds)

                if random.random() < self.cart_prob:
                    if send_event(self.user_id, "cart", item_id=item_id, category=category):
                        total_events += 1
                        logger.info(
                            "%-18s cart      uid=%-8s item=%s",
                            f"[{self.name}]",
                            self.user_id[:8],
                            item_id,
                        )
                    time.sleep(self.inter_event_seconds)

                    if random.random() < self.purchase_prob:
                        if send_event(self.user_id, "purchase", item_id=item_id, category=category):
                            total_events += 1
                            logger.info(
                                "%-18s purchase  uid=%-8s item=%s",
                                f"[{self.name}]",
                                self.user_id[:8],
                                item_id,
                            )
                        time.sleep(self.inter_event_seconds)

            time.sleep(self.inter_event_seconds)

        return total_events


def build_user_pool(size: int) -> list[str]:
    return [f"user_{index:04d}" for index in range(size)]


def pick_persona(persona_cfgs: dict) -> tuple[str, dict]:
    names = list(persona_cfgs.keys())
    weights = [float(persona_cfgs[name].get("ratio", 1.0)) for name in names]
    chosen = random.choices(names, weights=weights, k=1)[0]
    return chosen, persona_cfgs[chosen]


def build_persona_cycle(persona_cfgs: dict) -> list[str]:
    persona_names = list(persona_cfgs.keys())
    shuffled = persona_names[:]
    random.shuffle(shuffled)
    return shuffled


def main() -> None:
    config = load_config()
    sim_cfg = config["simulation"]
    persona_cfgs = config["personas"]

    user_pool = build_user_pool(int(sim_cfg["user_pool_size"]))
    cycle_delay = float(sim_cfg["cycle_delay_seconds"])
    retry_delay = float(sim_cfg["gateway_retry_seconds"])

    logger.info("Simulator start | api_gateway=%s dry_run=%s", API_URL, DRY_RUN)
    logger.info("User pool=%d personas=%s", len(user_pool), list(persona_cfgs.keys()))
    if FORCE_ALL_PERSONAS:
        logger.info("Persona cycling enabled for broad coverage.")

    while not gateway_alive():
        logger.warning("API Gateway unavailable. Retrying in %.1f seconds...", retry_delay)
        time.sleep(retry_delay)
    logger.info("API Gateway is ready.")

    session_count = 0
    total_events = 0
    persona_cycle = build_persona_cycle(persona_cfgs) if FORCE_ALL_PERSONAS else []

    while True:
        try:
            user_id = random.choice(user_pool)
            if persona_cycle:
                persona_name = persona_cycle.pop(0)
                persona_cfg = persona_cfgs[persona_name]
                if not persona_cycle:
                    persona_cycle = build_persona_cycle(persona_cfgs)
            else:
                persona_name, persona_cfg = pick_persona(persona_cfgs)
            persona = Persona(persona_name, persona_cfg, user_id)

            events = persona.run_session()
            session_count += 1
            total_events += events

            logger.info(
                "Session #%d complete | persona=%s uid=%s events=%d total=%d",
                session_count,
                persona_name,
                user_id[:8],
                events,
                total_events,
            )
        except Exception as exc:
            logger.error("Session error: %s", exc)

        if DRY_RUN and MAX_SESSIONS > 0 and session_count >= MAX_SESSIONS:
            logger.info("Dry-run complete after %d sessions.", session_count)
            break

        time.sleep(cycle_delay)


if __name__ == "__main__":
    main()
