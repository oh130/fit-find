"""Service-side ranking inference for recommendation candidates."""

from __future__ import annotations

import logging
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

try:
    from rec_models.ranking.infer import (
        DEFAULT_CHECKPOINT_DIR,
        _extract_scores,
        load_artifacts,
        prepare_inference_features,
    )
    from rec_models.ranking.train import (
        ITEM_PERSONA_DEV_PATH,
        ITEM_PERSONA_FEATURE_COLUMNS,
        ITEM_PERSONA_PATH,
        ITEM_PERSONA_TEST_PATH,
        PERSONAS,
        PERSONA_FEATURE_COLUMNS,
        PERSONA_RATIO_COLUMNS,
        USER_PERSONA_DEV_PATH,
        USER_PERSONA_FEATURE_COLUMNS,
        USER_PERSONA_PATH,
        USER_PERSONA_TEST_PATH,
    )
except ImportError:  # pragma: no cover - supports running from rec_models/ as cwd
    from ranking.infer import (  # type: ignore[no-redef]
        DEFAULT_CHECKPOINT_DIR,
        _extract_scores,
        load_artifacts,
        prepare_inference_features,
    )
    from ranking.train import (  # type: ignore[no-redef]
        ITEM_PERSONA_DEV_PATH,
        ITEM_PERSONA_FEATURE_COLUMNS,
        ITEM_PERSONA_PATH,
        ITEM_PERSONA_TEST_PATH,
        PERSONAS,
        PERSONA_FEATURE_COLUMNS,
        PERSONA_RATIO_COLUMNS,
        USER_PERSONA_DEV_PATH,
        USER_PERSONA_FEATURE_COLUMNS,
        USER_PERSONA_PATH,
        USER_PERSONA_TEST_PATH,
    )


LOGGER = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]
CUSTOMER_FEATURES_PATH = BASE_DIR / "data" / "processed" / "customer_features.csv"
LEAKAGE_COLUMNS = {"label", "customer_id", "article_id", "price", "sales_channel_id"}
DEFAULT_NUMERIC_VALUE = np.nan
DEFAULT_CATEGORICAL_VALUE = "UNKNOWN"
DEFAULT_REASON_VALUE = "unknown_candidate_reason"
DEFAULT_CANDIDATE_PRIOR_WEIGHT = 0.15
USER_PERSONA_SERVING_COLUMNS = [*PERSONA_RATIO_COLUMNS, "top_persona", "top_persona_ratio"]
ITEM_PERSONA_SERVING_COLUMNS = [*PERSONA_RATIO_COLUMNS, "top_persona", "top_persona_ratio"]


def cast_numeric_features_to_float(frame: pd.DataFrame) -> pd.DataFrame:
    """Compatibility shim for ranking pipeline artifacts saved from train.py."""

    return frame.astype("float64")


def _register_legacy_joblib_symbols() -> None:
    """Expose legacy symbols expected by persisted sklearn transformers."""

    main_module = sys.modules.get("__main__")
    if main_module is not None and not hasattr(main_module, "cast_numeric_features_to_float"):
        setattr(main_module, "cast_numeric_features_to_float", cast_numeric_features_to_float)


@lru_cache(maxsize=1)
def load_ranking_pipeline(checkpoint_dir: Path = DEFAULT_CHECKPOINT_DIR) -> tuple[Pipeline, dict[str, Any]]:
    """Load and cache the persisted ranking pipeline and metadata."""

    _register_legacy_joblib_symbols()
    model, metadata = load_artifacts(checkpoint_dir=checkpoint_dir)
    return model, metadata


@lru_cache(maxsize=1)
def load_customer_features() -> pd.DataFrame:
    """Load and cache customer profile features for ranking."""

    if not CUSTOMER_FEATURES_PATH.exists():
        LOGGER.warning("Customer feature file not found: %s", CUSTOMER_FEATURES_PATH)
        return pd.DataFrame(columns=["customer_id", "age", "age_bucket", "fashion_news_frequency", "club_member_status"])

    customer_features = pd.read_csv(CUSTOMER_FEATURES_PATH, dtype=str).fillna(DEFAULT_CATEGORICAL_VALUE)
    customer_features["customer_id"] = customer_features["customer_id"].astype(str)
    customer_features["age"] = pd.to_numeric(customer_features.get("age"), errors="coerce")
    return customer_features.set_index("customer_id", drop=False)


def _resolve_user_persona_path() -> Path | None:
    for path in (USER_PERSONA_DEV_PATH, USER_PERSONA_PATH, USER_PERSONA_TEST_PATH):
        if path.exists():
            return path
    return None


def _resolve_item_persona_path() -> Path | None:
    for path in (ITEM_PERSONA_DEV_PATH, ITEM_PERSONA_PATH, ITEM_PERSONA_TEST_PATH):
        if path.exists():
            return path
    return None


@lru_cache(maxsize=1)
def load_user_persona_features() -> pd.DataFrame:
    """Load user persona features indexed for low-latency serving lookups."""

    path = _resolve_user_persona_path()
    columns = ["customer_id", *USER_PERSONA_SERVING_COLUMNS]
    if path is None:
        LOGGER.warning("User persona score file not found. Using default persona features.")
        return pd.DataFrame(columns=columns).set_index("customer_id", drop=False)

    user_personas = pd.read_csv(path, dtype={"customer_id": str}, usecols=columns).fillna("")
    user_personas["customer_id"] = user_personas["customer_id"].astype(str).str.strip()
    for column in PERSONA_RATIO_COLUMNS:
        user_personas[column] = pd.to_numeric(user_personas[column], errors="coerce").fillna(0.0)
    user_personas["top_persona_ratio"] = pd.to_numeric(user_personas["top_persona_ratio"], errors="coerce").fillna(0.0)
    user_personas["top_persona"] = user_personas["top_persona"].map(_safe_get_text)
    return user_personas.set_index("customer_id", drop=False)


@lru_cache(maxsize=1)
def load_item_persona_features() -> pd.DataFrame:
    """Load item persona features indexed for low-latency serving lookups."""

    path = _resolve_item_persona_path()
    columns = ["article_id", *ITEM_PERSONA_SERVING_COLUMNS]
    if path is None:
        LOGGER.warning("Item persona score file not found. Using default persona features.")
        return pd.DataFrame(columns=columns).set_index("article_id", drop=False)

    item_personas = pd.read_csv(path, dtype={"article_id": str}, usecols=columns).fillna("")
    item_personas["article_id"] = item_personas["article_id"].astype(str).str.strip().str.zfill(10)
    for column in PERSONA_RATIO_COLUMNS:
        item_personas[column] = pd.to_numeric(item_personas[column], errors="coerce").fillna(0.0)
    item_personas["top_persona_ratio"] = pd.to_numeric(item_personas["top_persona_ratio"], errors="coerce").fillna(0.0)
    item_personas["top_persona"] = item_personas["top_persona"].map(_safe_get_text)
    return item_personas.set_index("article_id", drop=False)


def _default_persona_feature(prefix: str, index: pd.Index) -> dict[str, Any]:
    values: dict[str, Any] = {
        f"{prefix}_{column}": pd.Series(0.0, index=index, dtype="float64")
        for column in PERSONA_RATIO_COLUMNS
    }
    values[f"{prefix}_top_persona"] = pd.Series(DEFAULT_CATEGORICAL_VALUE, index=index, dtype=object)
    values[f"{prefix}_top_persona_ratio"] = pd.Series(0.0, index=index, dtype="float64")
    return values


def _attach_serving_persona_features(features: pd.DataFrame) -> pd.DataFrame:
    """Attach persona features without merging small request frames into large tables."""

    if all(column in features.columns for column in PERSONA_FEATURE_COLUMNS):
        return features

    enriched = features.copy()
    index = enriched.index
    for column, values in _default_persona_feature("user", index).items():
        enriched[column] = values
    for column, values in _default_persona_feature("item", index).items():
        enriched[column] = values

    if "customer_id" in enriched.columns:
        user_personas = load_user_persona_features()
        user_ids = enriched["customer_id"].astype(str).str.strip()
        matched_users = user_personas.reindex(user_ids)
        for column in PERSONA_RATIO_COLUMNS:
            enriched[f"user_{column}"] = pd.to_numeric(matched_users[column].to_numpy(), errors="coerce")
        enriched["user_top_persona"] = pd.Series(matched_users["top_persona"].to_numpy(), index=index).map(_safe_get_text)
        enriched["user_top_persona_ratio"] = pd.to_numeric(matched_users["top_persona_ratio"].to_numpy(), errors="coerce")

    if "article_id" in enriched.columns:
        item_personas = load_item_persona_features()
        article_ids = enriched["article_id"].astype(str).str.strip().str.zfill(10)
        matched_items = item_personas.reindex(article_ids)
        for column in PERSONA_RATIO_COLUMNS:
            enriched[f"item_{column}"] = pd.to_numeric(matched_items[column].to_numpy(), errors="coerce")
        enriched["item_top_persona"] = pd.Series(matched_items["top_persona"].to_numpy(), index=index).map(_safe_get_text)
        enriched["item_top_persona_ratio"] = pd.to_numeric(matched_items["top_persona_ratio"].to_numpy(), errors="coerce")

    for column in USER_PERSONA_FEATURE_COLUMNS + ITEM_PERSONA_FEATURE_COLUMNS:
        enriched[column] = pd.to_numeric(enriched[column], errors="coerce").fillna(0.0)
    enriched["user_top_persona_ratio"] = pd.to_numeric(enriched["user_top_persona_ratio"], errors="coerce").fillna(0.0)
    enriched["item_top_persona_ratio"] = pd.to_numeric(enriched["item_top_persona_ratio"], errors="coerce").fillna(0.0)
    enriched["top_persona_match"] = (
        enriched["user_top_persona"].ne(DEFAULT_CATEGORICAL_VALUE)
        & enriched["user_top_persona"].eq(enriched["item_top_persona"])
    ).astype(int)
    enriched["persona_match_score"] = sum(
        enriched[f"user_{persona}_ratio"] * enriched[f"item_{persona}_ratio"]
        for persona in PERSONAS
    )
    return enriched


def _safe_get_text(value: Any) -> str:
    if value is None:
        return DEFAULT_CATEGORICAL_VALUE
    text = str(value).strip()
    return text if text else DEFAULT_CATEGORICAL_VALUE


def _resolve_user_features(user_id: str) -> dict[str, Any]:
    """Return serving-time user features aligned to the ranking contract."""

    customer_features = load_customer_features()
    if user_id in customer_features.index:
        record = customer_features.loc[user_id].to_dict()
        return {
            "customer_id": str(record.get("customer_id", user_id)),
            "age": pd.to_numeric(record.get("age"), errors="coerce"),
            "age_bucket": _safe_get_text(record.get("age_bucket")),
            "fashion_news_frequency": _safe_get_text(record.get("fashion_news_frequency")),
            "club_member_status": _safe_get_text(record.get("club_member_status")),
        }

    LOGGER.info("Customer features not found for user_id=%s. Using cold-start defaults.", user_id)
    return {
        "customer_id": user_id,
        "age": DEFAULT_NUMERIC_VALUE,
        "age_bucket": DEFAULT_CATEGORICAL_VALUE,
        "fashion_news_frequency": DEFAULT_CATEGORICAL_VALUE,
        "club_member_status": DEFAULT_CATEGORICAL_VALUE,
    }


def _normalize_session_interest(session_interest: Any) -> dict[str, float]:
    if not isinstance(session_interest, dict):
        return {}
    normalized: dict[str, float] = {}
    for raw_key, raw_value in session_interest.items():
        key = _safe_get_text(raw_key)
        try:
            normalized[key] = float(raw_value)
        except (TypeError, ValueError):
            normalized[key] = 0.0
    return normalized


def _normalized_candidate_prior(candidate_items: pd.DataFrame) -> pd.Series:
    """Scale retrieval-stage candidate scores so ranking can keep useful recall signals."""

    prior = pd.to_numeric(
        candidate_items.get("candidate_score", pd.Series(0.0, index=candidate_items.index)),
        errors="coerce",
    ).fillna(0.0)
    if prior.empty:
        return prior

    minimum = float(prior.min())
    maximum = float(prior.max())
    if maximum <= minimum:
        return pd.Series(0.0, index=candidate_items.index, dtype="float64")
    return (prior - minimum) / (maximum - minimum)


def _blend_ranking_scores(scores: np.ndarray, candidate_items: pd.DataFrame) -> np.ndarray:
    """Blend model probability with retrieval prior for better top-k ordering."""

    prior = _normalized_candidate_prior(candidate_items).to_numpy(dtype=float)
    if prior.size == 0:
        return scores
    return scores + (prior * DEFAULT_CANDIDATE_PRIOR_WEIGHT)


def _build_session_signal_features(
    candidate_items: pd.DataFrame,
    session_context: dict[str, Any] | None,
) -> pd.DataFrame:
    session_context = session_context or {"recent_clicks": [], "session_interest": None}
    normalized_interest = _normalize_session_interest(session_context.get("session_interest"))

    features = pd.DataFrame(index=candidate_items.index)
    recent_click_signal = candidate_items.get("matches_recent_click_signal", pd.Series(False, index=candidate_items.index))
    session_interest_signal = candidate_items.get("matches_session_interest", pd.Series(False, index=candidate_items.index))
    candidate_reason = candidate_items.get("candidate_reason", pd.Series(DEFAULT_REASON_VALUE, index=candidate_items.index))

    categories = candidate_items.get("category", pd.Series(DEFAULT_CATEGORICAL_VALUE, index=candidate_items.index)).map(_safe_get_text)
    main_categories = candidate_items.get("main_category", pd.Series(DEFAULT_CATEGORICAL_VALUE, index=candidate_items.index)).map(_safe_get_text)

    session_interest_score = []
    for category, main_category in zip(categories.tolist(), main_categories.tolist(), strict=False):
        session_interest_score.append(
            float(normalized_interest.get(category, 0.0)) + float(normalized_interest.get(main_category, 0.0))
        )

    features["has_recent_click_signal"] = recent_click_signal.fillna(False).astype(int)
    features["has_session_interest_signal"] = session_interest_signal.fillna(False).astype(int)
    features["recent_click_count"] = int(len(session_context.get("recent_clicks") or []))
    features["session_interest_count"] = int(len(normalized_interest))
    features["session_interest_score"] = np.asarray(session_interest_score, dtype=float)
    features["candidate_reason"] = candidate_reason.fillna(DEFAULT_REASON_VALUE).map(_safe_get_text)
    return features


def _assemble_ranking_features(
    *,
    user_features: pd.DataFrame,
    candidate_items: pd.DataFrame,
    feature_columns: list[str],
    session_context: dict[str, Any] | None = None,
) -> pd.DataFrame:
    safe_candidates = candidate_items.copy()
    safe_candidates["category"] = safe_candidates.get("category", pd.Series(DEFAULT_CATEGORICAL_VALUE, index=safe_candidates.index)).map(_safe_get_text)
    safe_candidates["main_category"] = safe_candidates.get("main_category", pd.Series(DEFAULT_CATEGORICAL_VALUE, index=safe_candidates.index)).map(_safe_get_text)
    safe_candidates["color"] = safe_candidates.get("color", pd.Series(DEFAULT_CATEGORICAL_VALUE, index=safe_candidates.index)).map(_safe_get_text)

    safe_text_frame = safe_candidates.reindex(
        columns=[
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
        ],
        fill_value=DEFAULT_CATEGORICAL_VALUE,
    ).copy()
    for column in safe_text_frame.columns:
        safe_text_frame[column] = safe_text_frame[column].map(_safe_get_text)

    features = pd.DataFrame(index=safe_candidates.index)
    if "customer_id" in user_features.columns:
        features["customer_id"] = user_features["customer_id"].astype(str)
    if "article_id" in safe_candidates.columns:
        features["article_id"] = safe_candidates["article_id"].astype(str).str.strip().str.zfill(10)
    features["age"] = pd.to_numeric(user_features.get("age"), errors="coerce")
    features["age_bucket"] = user_features["age_bucket"].map(_safe_get_text)
    features["fashion_news_frequency"] = user_features["fashion_news_frequency"].map(_safe_get_text)
    features["club_member_status"] = user_features["club_member_status"].map(_safe_get_text)

    features["popularity"] = pd.to_numeric(safe_candidates.get("popularity"), errors="coerce")
    features["avg_price"] = pd.to_numeric(safe_candidates.get("avg_price"), errors="coerce")
    features["item_age_days"] = pd.to_numeric(safe_candidates.get("item_age_days"), errors="coerce")
    features["is_new_item"] = (
        safe_candidates.get("is_new_item", pd.Series(False, index=safe_candidates.index))
        .fillna(False)
        .astype(int)
    )
    for column in safe_text_frame.columns:
        features[column] = safe_text_frame[column]

    features["age_category"] = features["age_bucket"] + "_" + features["category"]
    features["age_color"] = features["age_bucket"] + "_" + features["color"]
    features["member_category"] = features["club_member_status"] + "_" + features["category"]
    features["fashion_category"] = features["fashion_news_frequency"] + "_" + features["category"]

    session_signal_features = _build_session_signal_features(
        candidate_items=safe_candidates,
        session_context=session_context,
    )
    for column in session_signal_features.columns:
        features[column] = session_signal_features[column]

    features = _attach_serving_persona_features(features)
    return prepare_inference_features(features, feature_columns=feature_columns)


@lru_cache(maxsize=1)
def get_ranking_feature_columns() -> tuple[str, ...]:
    """Cache serving-time ranking feature columns derived from model metadata."""

    _, metadata = load_ranking_pipeline()
    feature_columns = tuple(column for column in metadata.get("feature_columns", []) if column not in LEAKAGE_COLUMNS)
    if not feature_columns:
        raise ValueError("Ranking metadata does not contain usable feature_columns.")
    return feature_columns


def build_ranking_features(
    user_id: str,
    candidate_items: pd.DataFrame,
    session_context: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Build serving-time ranking features for candidate items."""

    if candidate_items.empty:
        return pd.DataFrame()

    feature_columns = list(get_ranking_feature_columns())
    user_features = _resolve_user_features(user_id)
    age_bucket = _safe_get_text(user_features.get("age_bucket"))
    club_member_status = _safe_get_text(user_features.get("club_member_status"))
    fashion_news_frequency = _safe_get_text(user_features.get("fashion_news_frequency"))

    user_feature_frame = pd.DataFrame(index=candidate_items.index)
    user_feature_frame["customer_id"] = user_id
    user_feature_frame["age"] = user_features.get("age", DEFAULT_NUMERIC_VALUE)
    user_feature_frame["age_bucket"] = age_bucket
    user_feature_frame["fashion_news_frequency"] = fashion_news_frequency
    user_feature_frame["club_member_status"] = club_member_status
    return _assemble_ranking_features(
        user_features=user_feature_frame,
        candidate_items=candidate_items,
        feature_columns=feature_columns,
        session_context=session_context,
    )


def build_batch_ranking_features(candidate_items: pd.DataFrame) -> pd.DataFrame:
    """Build ranking features for many user-item rows in one pass.

    This path is intended for offline evaluation where candidate rows already
    contain a `customer_id` column. It avoids repeated per-user feature frame
    construction and lets sklearn preprocess/score the whole batch at once.
    """

    if candidate_items.empty:
        return pd.DataFrame()
    if "customer_id" not in candidate_items.columns:
        raise ValueError("Batch ranking features require a customer_id column.")

    feature_columns = list(get_ranking_feature_columns())
    customer_features = load_customer_features()

    safe_candidates = candidate_items.copy()
    safe_candidates["customer_id"] = safe_candidates["customer_id"].astype(str)

    if customer_features.empty:
        user_features = pd.DataFrame(index=safe_candidates.index)
        user_features["customer_id"] = safe_candidates["customer_id"]
        user_features["age"] = DEFAULT_NUMERIC_VALUE
        user_features["age_bucket"] = DEFAULT_CATEGORICAL_VALUE
        user_features["fashion_news_frequency"] = DEFAULT_CATEGORICAL_VALUE
        user_features["club_member_status"] = DEFAULT_CATEGORICAL_VALUE
    else:
        join_columns = ["customer_id", "age", "age_bucket", "fashion_news_frequency", "club_member_status"]
        user_features = safe_candidates.loc[:, ["customer_id"]].merge(
            customer_features.reset_index(drop=True).loc[:, join_columns],
            on="customer_id",
            how="left",
        )
        user_features["age"] = pd.to_numeric(user_features.get("age"), errors="coerce")
        user_features["customer_id"] = user_features["customer_id"].astype(str)
        user_features["age_bucket"] = user_features["age_bucket"].map(_safe_get_text)
        user_features["fashion_news_frequency"] = user_features["fashion_news_frequency"].map(_safe_get_text)
        user_features["club_member_status"] = user_features["club_member_status"].map(_safe_get_text)

    return _assemble_ranking_features(
        user_features=user_features,
        candidate_items=safe_candidates,
        feature_columns=feature_columns,
        session_context=None,
    )


def score_candidates(
    user_id: str,
    candidate_items: pd.DataFrame,
    session_context: dict[str, Any] | None = None,
    checkpoint_dir: Path = DEFAULT_CHECKPOINT_DIR,
) -> pd.DataFrame:
    """Score ranking candidates and preserve recommendation metadata."""

    if candidate_items.empty:
        return candidate_items.copy()

    model, _ = load_ranking_pipeline(checkpoint_dir=checkpoint_dir)
    ranking_features = build_ranking_features(
        user_id=user_id,
        candidate_items=candidate_items,
        session_context=session_context,
    )
    scores = _blend_ranking_scores(
        scores=_extract_scores(model=model, features=ranking_features),
        candidate_items=candidate_items,
    )

    result = candidate_items.copy()
    result["score"] = scores
    cold_start_mask = result.get("candidate_reason", pd.Series(index=result.index, dtype=object)).eq("cold_start_popularity")
    session_interest_mask = result.get("matches_session_interest", pd.Series(False, index=result.index)).fillna(False).astype(bool)
    recent_click_mask = result.get("matches_recent_click_signal", pd.Series(False, index=result.index)).fillna(False).astype(bool)
    result["reason"] = np.select(
        [cold_start_mask, session_interest_mask, recent_click_mask],
        ["cold_start_popularity", "session_interest_match", "recent_click_similarity"],
        default="ranking_score",
    )
    result["is_exploration"] = False
    return result


def score_candidate_batch(
    candidate_items: pd.DataFrame,
    checkpoint_dir: Path = DEFAULT_CHECKPOINT_DIR,
) -> pd.DataFrame:
    """Score many candidate rows across users in one model pass."""

    if candidate_items.empty:
        return candidate_items.copy()

    model, _ = load_ranking_pipeline(checkpoint_dir=checkpoint_dir)
    ranking_features = build_batch_ranking_features(candidate_items=candidate_items)
    scores = _blend_ranking_scores(
        scores=_extract_scores(model=model, features=ranking_features),
        candidate_items=candidate_items,
    )

    result = candidate_items.copy()
    result["score"] = scores
    cold_start_mask = result.get("candidate_reason", pd.Series(index=result.index, dtype=object)).eq("cold_start_popularity")
    session_interest_mask = result.get("matches_session_interest", pd.Series(False, index=result.index)).fillna(False).astype(bool)
    recent_click_mask = result.get("matches_recent_click_signal", pd.Series(False, index=result.index)).fillna(False).astype(bool)
    result["reason"] = np.select(
        [cold_start_mask, session_interest_mask, recent_click_mask],
        ["cold_start_popularity", "session_interest_match", "recent_click_similarity"],
        default="ranking_score",
    )
    result["is_exploration"] = False
    return result
