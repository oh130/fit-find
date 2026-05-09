"""
User behavior simulator — 6 personas generating search/click/purchase events
and sending them to the API Gateway (/api/events).
"""

from __future__ import annotations

import logging
import os
import random
import time
from pathlib import Path

import requests
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="[Simulator] %(asctime)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.yaml"

API_URL = os.getenv("API_GATEWAY_URL", "http://localhost:8000")


# ── 설정 로드 ─────────────────────────────────────────────────

def load_config() -> dict:
    with CONFIG_PATH.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── API 호출 ─────────────────────────────────────────────────

def search(query: str, top_k: int = 10) -> list[dict]:
    try:
        resp = requests.post(
            f"{API_URL}/api/search",
            json={"query": query, "top_k": top_k},
            timeout=8.0,
        )
        resp.raise_for_status()
        data = resp.json()
        return data if isinstance(data, list) else data.get("results", [])
    except Exception as exc:
        logger.debug("Search error: %s", exc)
        return []


def send_event(user_id: str, item_id: str, event_type: str, category: str | None = None) -> bool:
    payload: dict = {"user_id": user_id, "article_id": item_id, "event_type": event_type}
    if category:
        payload["category"] = category
    try:
        resp = requests.post(f"{API_URL}/api/events", json=payload, timeout=5.0)
        resp.raise_for_status()
        return True
    except Exception as exc:
        logger.debug("Event error: %s", exc)
        return False


def gateway_alive() -> bool:
    try:
        requests.get(f"{API_URL}/health", timeout=3.0).raise_for_status()
        return True
    except Exception:
        return False


# ── 페르소나 세션 ─────────────────────────────────────────────

class Persona:
    def __init__(self, name: str, cfg: dict, user_id: str):
        self.name = name
        self.user_id = user_id
        self.queries: list[str] = cfg["search_queries"]
        self.categories: list[str] = cfg["categories"]
        self.view_prob: float = cfg["view_prob"]
        self.cart_prob: float = cfg["cart_prob"]
        self.purchase_prob: float = cfg["purchase_prob"]
        self.inter_event_seconds: float = cfg["inter_event_seconds"]
        self.session_searches: int = cfg.get("session_searches", 3)

    def run_session(self) -> int:
        """한 번의 쇼핑 세션을 실행하고 발생시킨 이벤트 수를 반환한다."""
        total_events = 0

        for _ in range(self.session_searches):
            query = random.choice(self.queries)
            results = search(query)

            if not results:
                time.sleep(self.inter_event_seconds)
                continue

            for item in results:
                if random.random() > self.view_prob:
                    continue

                item_id = str(item.get("article_id") or item.get("product_id") or item.get("id", "unknown"))
                category = item.get("category") or random.choice(self.categories)

                # 클릭(뷰)
                if send_event(self.user_id, item_id, "click", category):
                    total_events += 1
                    logger.info(
                        "%-18s click     uid=%-8s item=%s cat=%s",
                        f"[{self.name}]", self.user_id[:8], item_id, category,
                    )
                time.sleep(self.inter_event_seconds)

                # 장바구니 → 구매 체인
                if random.random() < self.cart_prob:
                    # 장바구니 추가도 click으로 전달 (API 스펙상 click|purchase만 존재)
                    if send_event(self.user_id, item_id, "click", category):
                        total_events += 1
                        logger.info(
                            "%-18s cart      uid=%-8s item=%s",
                            f"[{self.name}]", self.user_id[:8], item_id,
                        )
                    time.sleep(self.inter_event_seconds)

                    if random.random() < self.purchase_prob:
                        if send_event(self.user_id, item_id, "purchase", category):
                            total_events += 1
                            logger.info(
                                "%-18s purchase  uid=%-8s item=%s",
                                f"[{self.name}]", self.user_id[:8], item_id,
                            )
                        time.sleep(self.inter_event_seconds)

            time.sleep(self.inter_event_seconds)

        return total_events


# ── 유저 풀 & 페르소나 선택 ───────────────────────────────────

def build_user_pool(size: int) -> list[str]:
    return [f"user_{i:04d}" for i in range(size)]


def pick_persona(persona_cfgs: dict) -> tuple[str, dict]:
    """가중치 기반으로 페르소나를 선택한다."""
    names = list(persona_cfgs.keys())
    weights = [persona_cfgs[n]["ratio"] for n in names]
    chosen = random.choices(names, weights=weights, k=1)[0]
    return chosen, persona_cfgs[chosen]


# ── 메인 루프 ─────────────────────────────────────────────────

def main() -> None:
    config = load_config()
    sim_cfg = config["simulation"]
    persona_cfgs = config["personas"]

    user_pool = build_user_pool(sim_cfg["user_pool_size"])
    cycle_delay = sim_cfg["cycle_delay_seconds"]
    retry_delay = sim_cfg["gateway_retry_seconds"]

    logger.info("시뮬레이터 시작 — API Gateway: %s", API_URL)
    logger.info("유저 풀: %d명  페르소나: %s", len(user_pool), list(persona_cfgs.keys()))

    # API Gateway 기동 대기
    while not gateway_alive():
        logger.warning("API Gateway 응답 없음. %d초 후 재시도...", retry_delay)
        time.sleep(retry_delay)
    logger.info("API Gateway 연결 확인.")

    session_count = 0
    total_events = 0

    while True:
        try:
            user_id = random.choice(user_pool)
            persona_name, persona_cfg = pick_persona(persona_cfgs)
            persona = Persona(persona_name, persona_cfg, user_id)

            events = persona.run_session()
            session_count += 1
            total_events += events

            logger.info(
                "세션 #%d 완료 — 페르소나=%s uid=%s 이벤트=%d건 누적=%d건",
                session_count, persona_name, user_id[:8], events, total_events,
            )

        except Exception as exc:
            logger.error("세션 중 오류: %s", exc)

        time.sleep(cycle_delay)


if __name__ == "__main__":
    main()
