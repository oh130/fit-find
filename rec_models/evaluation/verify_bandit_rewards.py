"""Live Redis verification for reward-aware bandit reranking.

The script writes synthetic bandit keys, checks reward/UCB growth across many
events, verifies reranking uses the rewarded item, then removes the test keys by
default so local Redis state stays clean.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from rec_models.serving.bandit_store import RewardBanditStore, normalize_article_id
    from rec_models.serving.rerank_bridge import rerank_recommendations
except ImportError:  # pragma: no cover - supports running from rec_models/ as cwd
    from serving.bandit_store import RewardBanditStore, normalize_article_id  # type: ignore[no-redef]
    from serving.rerank_bridge import rerank_recommendations  # type: ignore[no-redef]


DEFAULT_ITEM_IDS = [
    "9990000001",
    "9990000002",
    "9990000003",
    "9990000004",
    "9990000005",
    "9990000006",
    "9990000007",
    "9990000008",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify Redis-backed bandit reward learning.")
    parser.add_argument("--user-id", default="bandit_long_run_user")
    parser.add_argument("--target-article-id", default=DEFAULT_ITEM_IDS[-1])
    parser.add_argument("--rounds", type=int, default=24)
    parser.add_argument("--redis-host", default="localhost")
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--keep-redis-keys", action="store_true")
    return parser.parse_args()


def build_candidate_frame(item_ids: list[str]) -> pd.DataFrame:
    roles = ["upper", "lower", "shoes", "outer", "bag", "accessory", "upper", "lower"]
    categories = ["Blouses", "Trousers", "Shoes", "Jackets", "Bags", "Accessories", "Tops", "Jeans"]
    rows = []
    for index, article_id in enumerate(item_ids):
        rows.append(
            {
                "article_id": article_id,
                "score": 0.96 - index * 0.025,
                "popularity": 0.50 - index * 0.03,
                "item_age_days": 30 + index,
                "is_new_item": False,
                "main_category": categories[index],
                "category": categories[index],
                "outfit_role": roles[index],
                "outfit_eligible": True,
                "avg_price": 0.02 + index * 0.001,
                "prod_name": f"Bandit test item {index + 1}",
                "department_name": categories[index],
            }
        )
    return pd.DataFrame(rows)


def read_item_stats(store: RewardBanditStore, article_id: str) -> dict[str, float]:
    client = store._client()  # noqa: SLF001 - verification needs raw Redis stats.
    if client is None:
        raise RuntimeError("Redis bandit store is unavailable")

    raw = client.hgetall(store._item_key(normalize_article_id(article_id)))  # noqa: SLF001

    def as_float(key: str) -> float:
        try:
            return float(raw.get(key, 0.0))
        except (TypeError, ValueError):
            return 0.0

    return {
        "impressions": as_float("impressions"),
        "reward": as_float("reward"),
        "reward_events": as_float("reward_events"),
        "clicks": as_float("clicks"),
        "carts": as_float("carts"),
        "purchases": as_float("purchases"),
    }


def cleanup_keys(store: RewardBanditStore, user_id: str, item_ids: list[str]) -> int:
    client = store._client()  # noqa: SLF001
    if client is None:
        return 0
    keys = [store._item_key(normalize_article_id(article_id)) for article_id in item_ids]  # noqa: SLF001
    keys.append(store._user_impressions_key(user_id))  # noqa: SLF001
    return int(client.delete(*keys))


def main() -> None:
    args = parse_args()
    target_id = normalize_article_id(args.target_article_id)
    item_ids = [normalize_article_id(value) for value in DEFAULT_ITEM_IDS]
    if target_id not in item_ids:
        item_ids[-1] = target_id

    store = RewardBanditStore(host=args.redis_host, enabled=True)
    candidates = build_candidate_frame(item_ids)

    cleanup_keys(store, args.user_id, item_ids)
    empty_scores = store.get_ucb_scores(item_ids)
    before_recommendations = rerank_recommendations(
        candidates,
        top_n=5,
        exploration_slots=1,
        random_seed=17,
        rerank_weights={"exploration_weight": 5.0},
        bandit_scores=empty_scores,
    )
    before_stats = read_item_stats(store, target_id)
    before_ucb = store.get_ucb_scores([target_id]).get(target_id, 0.0)

    reward_events = ["click", "cart", "purchase"]
    for round_index in range(args.rounds):
        store.record_impressions(
            args.user_id,
            [{"product_id": article_id, "reason": "ranking_score"} for article_id in item_ids],
            surface="bandit_long_run",
        )
        event = reward_events[round_index % len(reward_events)]
        result = store.record_reward(args.user_id, target_id, event)
        if not result.updated:
            raise RuntimeError(f"Bandit reward update failed for event={event}")

    after_stats = read_item_stats(store, target_id)
    after_scores = store.get_ucb_scores(item_ids)
    after_ucb = after_scores.get(target_id, 0.0)
    after_recommendations = rerank_recommendations(
        candidates,
        top_n=5,
        exploration_slots=1,
        random_seed=17,
        rerank_weights={"exploration_weight": 5.0},
        bandit_scores=after_scores,
    )
    top_ids = [str(item["product_id"]) for item in after_recommendations]
    reasons = {str(item["product_id"]): item.get("reason") for item in after_recommendations}
    target_rank = top_ids.index(target_id) + 1 if target_id in top_ids else None
    rewarded_items_in_top_n = sum(1 for article_id in top_ids if article_id == target_id)
    rewarded_share = rewarded_items_in_top_n / max(len(top_ids), 1)

    checks = {
        "reward_increased": after_stats["reward"] > before_stats["reward"],
        "reward_events_recorded": after_stats["reward_events"] >= args.rounds,
        "ucb_increased": after_ucb > before_ucb,
        "rewarded_item_recommended": target_rank is not None,
        "rewarded_item_uses_bandit_reason": reasons.get(target_id) == "bandit_reward_exploration",
        "not_over_concentrated": rewarded_share <= 0.4 and len(set(top_ids)) >= 4,
    }
    report: dict[str, Any] = {
        "name": "dev_bandit_reward_long_run",
        "user_id": args.user_id,
        "target_article_id": target_id,
        "rounds": args.rounds,
        "before": {
            "target_stats": before_stats,
            "target_ucb_score": before_ucb,
            "recommendations": before_recommendations,
        },
        "after": {
            "target_stats": after_stats,
            "target_ucb_score": after_ucb,
            "target_rank": target_rank,
            "rewarded_item_share_at_top_n": rewarded_share,
            "unique_items_at_top_n": len(set(top_ids)),
            "recommendations": after_recommendations,
        },
        "checks": checks,
        "passed": all(checks.values()),
    }

    if not args.keep_redis_keys:
        report["cleanup_deleted_keys"] = cleanup_keys(store, args.user_id, item_ids)

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
