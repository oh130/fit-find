"""Redis-backed reward store for exploration bandit feedback."""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

try:
    import redis
    from redis.exceptions import RedisError
except ImportError:  # pragma: no cover - Redis is optional in local smoke tests.
    redis = None  # type: ignore[assignment]

    class RedisError(Exception):  # type: ignore[no-redef]
        pass


LOGGER = logging.getLogger(__name__)

BANDIT_TTL_SECONDS = 60 * 60 * 24 * 30
BANDIT_RECENT_IMPRESSIONS = 100
BANDIT_UCB_ALPHA = float(os.getenv("BANDIT_UCB_ALPHA", "0.35"))
REWARD_WEIGHTS: dict[str, float] = {
    "view": 0.05,
    "click": 1.0,
    "cart": 2.0,
    "purchase": 4.0,
}


@dataclass(frozen=True)
class BanditRewardResult:
    """Result of applying a reward update."""

    updated: bool
    article_id: str
    event: str
    reward: float


def normalize_article_id(value: Any) -> str:
    article_id = str(value or "").strip()
    if article_id.isdigit():
        return article_id.zfill(10)
    return article_id


class RewardBanditStore:
    """Store item-level bandit impressions and rewards in Redis.

    The store is intentionally item-level for now: it can learn from clicks and
    purchases without needing a user embedding service. If Redis is unavailable,
    all methods degrade to no-ops so recommendation serving stays online.
    """

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        db: int | None = None,
        enabled: bool | None = None,
    ) -> None:
        self.enabled = enabled if enabled is not None else os.getenv("BANDIT_REDIS_ENABLED", "1") != "0"
        self._available: bool | None = None
        self._warned_unavailable = False
        self.client: Any | None = None
        if not self.enabled or redis is None:
            return

        self.client = redis.Redis(
            host=host or os.getenv("REDIS_HOST", "redis"),
            port=int(port or os.getenv("REDIS_PORT", "6379")),
            db=int(db or os.getenv("REDIS_DB", "0")),
            decode_responses=True,
            socket_connect_timeout=0.25,
            socket_timeout=0.25,
        )

    def _client(self) -> Any | None:
        if not self.enabled or self.client is None:
            return None
        if self._available is False:
            return None
        if self._available is None:
            try:
                self.client.ping()
                self._available = True
            except RedisError as exc:
                self._available = False
                if not self._warned_unavailable:
                    LOGGER.warning("Bandit Redis store unavailable; reward updates disabled: %s", exc)
                    self._warned_unavailable = True
                return None
        return self.client

    @staticmethod
    def _item_key(article_id: str) -> str:
        return f"bandit:item:{article_id}"

    @staticmethod
    def _global_impressions_key() -> str:
        return "bandit:global:impressions"

    @staticmethod
    def _user_impressions_key(user_id: str) -> str:
        return f"bandit:user:{user_id}:recent_impressions"

    def record_impressions(
        self,
        user_id: str,
        recommendations: Iterable[Mapping[str, Any]],
        *,
        surface: str,
    ) -> int:
        client = self._client()
        if client is None:
            return 0

        rows: list[tuple[int, str, Mapping[str, Any]]] = []
        for rank, recommendation in enumerate(recommendations, start=1):
            article_id = normalize_article_id(
                recommendation.get("product_id")
                or recommendation.get("article_id")
                or recommendation.get("item_id")
            )
            if article_id:
                rows.append((rank, article_id, recommendation))

        if not rows:
            return 0

        try:
            pipe = client.pipeline()
            for rank, article_id, recommendation in rows:
                item_key = self._item_key(article_id)
                reason = str(recommendation.get("reason", ""))
                is_exploration = bool(recommendation.get("is_exploration", False))
                pipe.hincrby(item_key, "impressions", 1)
                if is_exploration:
                    pipe.hincrby(item_key, "exploration_impressions", 1)
                if reason == "bandit_reward_exploration":
                    pipe.hincrby(item_key, "bandit_impressions", 1)
                pipe.expire(item_key, BANDIT_TTL_SECONDS)
                pipe.lpush(
                    self._user_impressions_key(user_id),
                    f"{article_id}|{rank}|{surface}|{reason}",
                )
            pipe.ltrim(self._user_impressions_key(user_id), 0, BANDIT_RECENT_IMPRESSIONS - 1)
            pipe.expire(self._user_impressions_key(user_id), BANDIT_TTL_SECONDS)
            pipe.incrby(self._global_impressions_key(), len(rows))
            pipe.expire(self._global_impressions_key(), BANDIT_TTL_SECONDS)
            pipe.execute()
            return len(rows)
        except RedisError as exc:
            LOGGER.warning("Failed to record bandit impressions: %s", exc)
            return 0

    def record_reward(self, user_id: str, article_id: Any, event: str) -> BanditRewardResult:
        normalized_id = normalize_article_id(article_id)
        normalized_event = str(event or "").strip().lower()
        reward = float(REWARD_WEIGHTS.get(normalized_event, 0.0))
        if not normalized_id or reward <= 0.0:
            return BanditRewardResult(False, normalized_id, normalized_event, reward)

        client = self._client()
        if client is None:
            return BanditRewardResult(False, normalized_id, normalized_event, reward)

        try:
            item_key = self._item_key(normalized_id)
            pipe = client.pipeline()
            pipe.hincrbyfloat(item_key, "reward", reward)
            pipe.hincrby(item_key, f"{normalized_event}s", 1)
            pipe.hincrby(item_key, "reward_events", 1)
            pipe.expire(item_key, BANDIT_TTL_SECONDS)
            pipe.lpush(self._user_impressions_key(user_id), f"reward:{normalized_id}|{normalized_event}|{reward}")
            pipe.ltrim(self._user_impressions_key(user_id), 0, BANDIT_RECENT_IMPRESSIONS - 1)
            pipe.expire(self._user_impressions_key(user_id), BANDIT_TTL_SECONDS)
            pipe.execute()
            return BanditRewardResult(True, normalized_id, normalized_event, reward)
        except RedisError as exc:
            LOGGER.warning("Failed to record bandit reward: %s", exc)
            return BanditRewardResult(False, normalized_id, normalized_event, reward)

    def get_ucb_scores(self, article_ids: Iterable[Any]) -> dict[str, float]:
        normalized_ids = []
        seen_ids: set[str] = set()
        for raw_id in article_ids:
            article_id = normalize_article_id(raw_id)
            if article_id and article_id not in seen_ids:
                normalized_ids.append(article_id)
                seen_ids.add(article_id)
        if not normalized_ids:
            return {}

        client = self._client()
        if client is None:
            return {}

        try:
            pipe = client.pipeline()
            pipe.get(self._global_impressions_key())
            for article_id in normalized_ids:
                pipe.hgetall(self._item_key(article_id))
            responses = pipe.execute()
        except RedisError as exc:
            LOGGER.warning("Failed to read bandit scores: %s", exc)
            return {}

        raw_global = responses[0]
        item_payloads = responses[1:]
        try:
            global_impressions = float(raw_global or 0.0)
        except (TypeError, ValueError):
            global_impressions = 0.0

        local_impressions = 0.0
        parsed_payloads: list[dict[str, float]] = []
        for payload in item_payloads:
            parsed = self._parse_stats(payload)
            local_impressions += parsed["impressions"]
            parsed_payloads.append(parsed)

        total_impressions = max(global_impressions, local_impressions)
        if total_impressions <= 0.0:
            return {article_id: 0.0 for article_id in normalized_ids}

        scores: dict[str, float] = {}
        log_total = math.log(total_impressions + 1.0)
        for article_id, stats in zip(normalized_ids, parsed_payloads, strict=False):
            impressions = stats["impressions"]
            reward = stats["reward"]
            if reward <= 0.0:
                scores[article_id] = 0.0
                continue
            mean_reward = reward / max(impressions, 1.0)
            uncertainty = math.sqrt(log_total / (impressions + 1.0))
            scores[article_id] = mean_reward + (BANDIT_UCB_ALPHA * uncertainty)
        return scores

    @staticmethod
    def _parse_stats(payload: Mapping[str, Any] | None) -> dict[str, float]:
        raw = dict(payload or {})

        def as_float(key: str) -> float:
            try:
                return float(raw.get(key, 0.0))
            except (TypeError, ValueError):
                return 0.0

        return {
            "impressions": as_float("impressions"),
            "reward": as_float("reward"),
        }


_BANDIT_STORE: RewardBanditStore | None = None


def get_bandit_store() -> RewardBanditStore:
    global _BANDIT_STORE
    if _BANDIT_STORE is None:
        _BANDIT_STORE = RewardBanditStore()
    return _BANDIT_STORE
