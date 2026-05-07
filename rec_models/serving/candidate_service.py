"""Service-side candidate generation helpers.

This module provides a lightweight candidate source for serving so the baseline
ranking model can be connected end-to-end before a production retrieval model
is wired in.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import logging
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from rec_models.candidate.dataset import load_candidate_training_data
    from rec_models.candidate.infer import (
        DEFAULT_CHECKPOINT_DIR as DEFAULT_TWO_TOWER_CHECKPOINT_DIR,
        build_item_embedding_index,
        build_latest_user_table,
        encode_user_records,
        load_two_tower_artifacts,
        retrieve_top_k,
    )
except ImportError:  # pragma: no cover - supports running from rec_models/ as cwd
    from candidate.dataset import load_candidate_training_data  # type: ignore[no-redef]
    from candidate.infer import (  # type: ignore[no-redef]
        DEFAULT_CHECKPOINT_DIR as DEFAULT_TWO_TOWER_CHECKPOINT_DIR,
        build_item_embedding_index,
        build_latest_user_table,
        encode_user_records,
        load_two_tower_artifacts,
        retrieve_top_k,
    )


LOGGER = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]
ARTICLE_FEATURES_PATH = BASE_DIR / "data" / "processed" / "articles_feature.csv"
ITEM_FEATURES_PATH = BASE_DIR / "data" / "processed" / "item_features.csv"
ITEM_FEATURES_DEV_PATH = BASE_DIR / "data" / "processed" / "item_features_dev.csv"
ITEM_FEATURES_TEST_PATH = BASE_DIR / "data" / "processed" / "item_features_test.csv"
DEFAULT_CANDIDATE_POOL_SIZE = 75
DEFAULT_NEW_ITEM_WINDOW_DAYS = 7
DEFAULT_CANDIDATE_POOL_MULTIPLIER = 1.5
DEFAULT_SIGNAL_LOOKUP_LIMIT_MULTIPLIER = 2
DEFAULT_HYBRID_TWO_TOWER_RATIO = 0.5
DEFAULT_TWO_TOWER_SCORE_WEIGHT = 2.0
DEFAULT_COPURCHASE_NEIGHBOR_LIMIT = 40
DEFAULT_COPURCHASE_SCORE_WEIGHT = 6.0
DEFAULT_COPURCHASE_ARTIFACT_PATH = (
    BASE_DIR / "rec_models" / "artifacts" / "candidate" / "candidate_copurchase_neighbors.parquet"
)
DEFAULT_SEQUENTIAL_ARTIFACT_PATH = (
    BASE_DIR / "rec_models" / "artifacts" / "candidate" / "candidate_sequential_transitions.csv"
)
DEFAULT_SEQUENTIAL_ARTICLE_LIMIT = 40
DEFAULT_SEQUENTIAL_CATEGORY_LIMIT = 40
DEFAULT_SEQUENTIAL_ARTICLE_SCORE_WEIGHT = 8.0
DEFAULT_SEQUENTIAL_CATEGORY_SCORE_WEIGHT = 4.0
DEFAULT_COVERAGE_EXPLORATION_LIMIT = 500
LOOKUP_COLUMNS = (
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
    "popularity",
    "avg_price",
    "item_age_days",
    "is_new_item",
)


@dataclass(frozen=True)
class ServingFeatureStore:
    catalog: pd.DataFrame
    candidate_frame: pd.DataFrame
    article_records: dict[str, dict[str, Any]]
    category_to_ids: dict[str, tuple[str, ...]]
    main_category_to_ids: dict[str, tuple[str, ...]]
    color_to_ids: dict[str, tuple[str, ...]]
    garment_group_to_ids: dict[str, tuple[str, ...]]
    price_band_to_ids: dict[str, tuple[str, ...]]
    popular_article_ids: tuple[str, ...]
    popularity_max: float


@dataclass(frozen=True)
class TwoTowerServingArtifacts:
    training_data: pd.DataFrame
    user_records: dict[str, dict[str, Any]]
    item_embeddings: Any
    item_ids: list[str]


@dataclass(frozen=True)
class CandidateUserProfileStore:
    user_profiles: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class CopurchaseNeighborStore:
    neighbors_by_article: dict[str, tuple[tuple[str, float, int], ...]]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class SequentialTransitionStore:
    article_neighbors_by_article: dict[str, tuple[tuple[str, float, int], ...]]
    article_neighbors_by_category: dict[str, tuple[tuple[str, float, int], ...]]
    metadata: dict[str, Any]


def _resolve_item_features_path() -> Path | None:
    for path in (ITEM_FEATURES_DEV_PATH, ITEM_FEATURES_PATH, ITEM_FEATURES_TEST_PATH):
        if path.exists():
            return path
    return None


def normalize_article_id(article_id: Any) -> str:
    """Normalize article ids to the zero-padded training format."""

    text = str(article_id).strip()
    if text.isdigit():
        return text.zfill(10)
    return text


def _safe_text(value: Any) -> str:
    text = str(value).strip() if value is not None else ""
    return text or "UNKNOWN"


def _build_lookup_map(catalog: pd.DataFrame, column: str) -> dict[str, tuple[str, ...]]:
    grouped = catalog.groupby(column, sort=False)["article_id"].apply(tuple)
    return {str(key): value for key, value in grouped.items()}


def _build_article_records(catalog: pd.DataFrame, popularity_max: float) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for row in catalog.loc[:, LOOKUP_COLUMNS].to_dict(orient="records"):
        article_id = str(row["article_id"])
        popularity = float(row.get("popularity", 0.0) or 0.0)
        item_age_days = row.get("item_age_days")
        numeric_item_age_days = float(item_age_days) if pd.notna(item_age_days) else math.nan
        is_new_item = bool(row.get("is_new_item", False))
        records[article_id] = {
            **row,
            "article_id": article_id,
            "category": _safe_text(row.get("category")),
            "main_category": _safe_text(row.get("main_category")),
            "color": _safe_text(row.get("color")),
            "popularity": popularity,
            "item_age_days": numeric_item_age_days,
            "is_new_item": is_new_item,
            "normalized_popularity": popularity / popularity_max,
            "fresh_boost": 0.5 if is_new_item else 0.0,
            "derived_price_band": _derive_price_band(float(row.get("avg_price", 0.0) or 0.0)),
        }
    return records


@lru_cache(maxsize=1)
def load_serving_artifacts() -> ServingFeatureStore:
    """Load and cache serving-time article metadata and lookup maps."""

    if not ARTICLE_FEATURES_PATH.exists():
        raise FileNotFoundError(f"Article feature file not found: {ARTICLE_FEATURES_PATH}")

    catalog = pd.read_csv(ARTICLE_FEATURES_PATH, dtype=str).fillna("UNKNOWN")
    catalog["article_id"] = catalog["article_id"].map(normalize_article_id)

    item_features_path = _resolve_item_features_path()
    if item_features_path is not None:
        popularity = pd.read_csv(item_features_path, dtype={"article_id": str}).fillna("")
        popularity["article_id"] = popularity["article_id"].map(normalize_article_id)
        popularity["popularity"] = pd.to_numeric(popularity.get("popularity"), errors="coerce").fillna(0.0)
        popularity["item_age_days"] = pd.to_numeric(popularity.get("item_age_days"), errors="coerce")
        popularity["is_new_item"] = popularity.get("is_new_item", "").astype(str).str.strip().str.lower().eq("true")
        catalog = catalog.merge(
            popularity.loc[:, ["article_id", "popularity", "avg_price", "item_age_days", "is_new_item"]],
            on="article_id",
            how="left",
        )
        LOGGER.info("Loaded item feature file for serving: %s", item_features_path)
    else:
        LOGGER.warning(
            "Item feature file not found. Checked %s, %s, and %s",
            ITEM_FEATURES_DEV_PATH,
            ITEM_FEATURES_PATH,
            ITEM_FEATURES_TEST_PATH,
        )
        catalog["popularity"] = 0.0
        catalog["avg_price"] = 0.0
        catalog["item_age_days"] = math.nan
        catalog["is_new_item"] = False

    catalog["popularity"] = pd.to_numeric(catalog["popularity"], errors="coerce").fillna(0.0)
    catalog["avg_price"] = pd.to_numeric(catalog.get("avg_price"), errors="coerce").fillna(0.0)
    catalog["item_age_days"] = pd.to_numeric(catalog.get("item_age_days"), errors="coerce")
    if "is_new_item" in catalog.columns:
        catalog["is_new_item"] = catalog["is_new_item"].fillna(False).astype(bool)
    else:
        catalog["is_new_item"] = catalog["item_age_days"].le(DEFAULT_NEW_ITEM_WINDOW_DAYS).fillna(False)
    for column in ("category", "main_category", "color"):
        catalog[column] = catalog[column].map(_safe_text)
    catalog["garment_group_name"] = catalog.get("garment_group_name", pd.Series("UNKNOWN", index=catalog.index)).map(_safe_text)
    catalog["derived_price_band"] = catalog["avg_price"].map(lambda value: _derive_price_band(float(value)))

    catalog = catalog.sort_values(["popularity", "article_id"], ascending=[False, True]).reset_index(drop=True)
    category_rank = catalog.groupby("main_category", dropna=False).cumcount()
    catalog["cold_start_bonus"] = category_rank.rsub(9).clip(lower=0) / 10.0

    popularity_max = max(float(catalog["popularity"].max()), 1.0)
    article_records = _build_article_records(catalog=catalog, popularity_max=popularity_max)
    for article_id, cold_start_bonus in zip(catalog["article_id"].astype(str), catalog["cold_start_bonus"], strict=False):
        article_records[article_id]["cold_start_bonus"] = float(cold_start_bonus)

    candidate_frame = catalog.loc[:, LOOKUP_COLUMNS].copy().set_index("article_id", drop=False)

    LOGGER.info("Loaded serving article artifacts with %s catalog rows", len(catalog))
    return ServingFeatureStore(
        catalog=catalog,
        candidate_frame=candidate_frame,
        article_records=article_records,
        category_to_ids=_build_lookup_map(catalog, "category"),
        main_category_to_ids=_build_lookup_map(catalog, "main_category"),
        color_to_ids=_build_lookup_map(catalog, "color"),
        garment_group_to_ids=_build_lookup_map(catalog, "garment_group_name"),
        price_band_to_ids=_build_lookup_map(catalog, "derived_price_band"),
        popular_article_ids=tuple(catalog["article_id"].astype(str)),
        popularity_max=popularity_max,
    )


def get_cached_feature_store() -> ServingFeatureStore:
    """Return the singleton-style feature store used by serving."""

    return load_serving_artifacts()


@lru_cache(maxsize=1)
def load_copurchase_neighbor_store(
    artifact_path: Path = DEFAULT_COPURCHASE_ARTIFACT_PATH,
) -> CopurchaseNeighborStore:
    """Load a cached article -> co-purchase neighbors lookup."""

    resolved_path = artifact_path.expanduser().resolve()
    if not resolved_path.exists():
        LOGGER.warning("Co-purchase artifact not found: %s", resolved_path)
        return CopurchaseNeighborStore(neighbors_by_article={}, metadata={"artifact_path": str(resolved_path), "available": False})

    metadata_path = resolved_path.with_suffix(".metadata.json")
    metadata: dict[str, Any] = {"artifact_path": str(resolved_path), "available": True}
    if metadata_path.exists():
        try:
            metadata.update(json.loads(metadata_path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            LOGGER.warning("Failed to parse co-purchase metadata: %s", metadata_path)

    storage_format = str(metadata.get("storage_format", "parquet"))
    if storage_format == "csv_fallback":
        neighbors = pd.read_csv(resolved_path)
    else:
        neighbors = pd.read_parquet(resolved_path)
    required_columns = {"article_id", "neighbor_article_id"}
    missing_columns = required_columns - set(neighbors.columns)
    if missing_columns:
        raise ValueError(f"Co-purchase artifact missing required columns: {sorted(missing_columns)}")

    neighbors["article_id"] = neighbors["article_id"].map(normalize_article_id)
    neighbors["neighbor_article_id"] = neighbors["neighbor_article_id"].map(normalize_article_id)
    neighbors["weighted_score"] = pd.to_numeric(neighbors.get("weighted_score"), errors="coerce").fillna(0.0)
    neighbors["cooccurrence_count"] = pd.to_numeric(neighbors.get("cooccurrence_count"), errors="coerce").fillna(0).astype(int)
    neighbors = neighbors.sort_values(
        ["article_id", "weighted_score", "cooccurrence_count", "neighbor_article_id"],
        ascending=[True, False, False, True],
    )
    neighbors_by_article = {
        str(seed_article_id): tuple(
            (
                str(row["neighbor_article_id"]),
                float(row["weighted_score"]),
                int(row["cooccurrence_count"]),
            )
            for row in group.to_dict(orient="records")
        )
        for seed_article_id, group in neighbors.groupby("article_id", sort=False)
    }

    LOGGER.info(
        "Loaded co-purchase neighbor store | seed_articles=%s artifact=%s",
        len(neighbors_by_article),
        resolved_path,
    )
    return CopurchaseNeighborStore(neighbors_by_article=neighbors_by_article, metadata=metadata)


@lru_cache(maxsize=1)
def load_sequential_transition_store(
    artifact_path: Path = DEFAULT_SEQUENTIAL_ARTIFACT_PATH,
) -> SequentialTransitionStore:
    resolved_path = artifact_path.expanduser().resolve()
    if not resolved_path.exists():
        LOGGER.warning("Sequential transition artifact not found: %s", resolved_path)
        return SequentialTransitionStore(article_neighbors_by_article={}, article_neighbors_by_category={}, metadata={"artifact_path": str(resolved_path), "available": False})

    transitions = pd.read_csv(resolved_path, dtype={"seed_id": str})
    required_columns = {"seed_type", "seed_id", "next_article_id"}
    missing_columns = required_columns - set(transitions.columns)
    if missing_columns:
        raise ValueError(f"Sequential transition artifact missing required columns: {sorted(missing_columns)}")

    transitions["seed_type"] = transitions["seed_type"].astype(str)
    transitions["seed_id"] = transitions["seed_id"].astype(str)
    transitions["next_article_id"] = transitions["next_article_id"].map(normalize_article_id)
    transitions["weighted_score"] = pd.to_numeric(transitions.get("weighted_score"), errors="coerce").fillna(0.0)
    transitions["transition_count"] = pd.to_numeric(transitions.get("transition_count"), errors="coerce").fillna(0).astype(int)
    transitions = transitions.sort_values(
        ["seed_type", "seed_id", "weighted_score", "transition_count", "next_article_id"],
        ascending=[True, True, False, False, True],
    )

    article_rows = transitions.loc[transitions["seed_type"].eq("article_id")].copy()
    category_rows = transitions.loc[transitions["seed_type"].eq("main_category")].copy()
    article_neighbors_by_article = {
        str(seed_id): tuple(
            (str(row["next_article_id"]), float(row["weighted_score"]), int(row["transition_count"]))
            for row in group.to_dict(orient="records")
        )
        for seed_id, group in article_rows.groupby("seed_id", sort=False)
    }
    article_neighbors_by_category = {
        str(seed_id): tuple(
            (str(row["next_article_id"]), float(row["weighted_score"]), int(row["transition_count"]))
            for row in group.to_dict(orient="records")
        )
        for seed_id, group in category_rows.groupby("seed_id", sort=False)
    }

    metadata_path = resolved_path.with_suffix(".metadata.json")
    metadata: dict[str, Any] = {"artifact_path": str(resolved_path), "available": True}
    if metadata_path.exists():
        try:
            metadata.update(json.loads(metadata_path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            LOGGER.warning("Failed to parse sequential transition metadata: %s", metadata_path)

    LOGGER.info(
        "Loaded sequential transition store | article_seeds=%s category_seeds=%s artifact=%s",
        len(article_neighbors_by_article),
        len(article_neighbors_by_category),
        resolved_path,
    )
    return SequentialTransitionStore(
        article_neighbors_by_article=article_neighbors_by_article,
        article_neighbors_by_category=article_neighbors_by_category,
        metadata=metadata,
    )


@lru_cache(maxsize=1)
def load_candidate_user_profiles() -> CandidateUserProfileStore:
    """Load cached user preference profiles derived from candidate training data."""

    try:
        training_data = load_candidate_training_data()
    except Exception:
        LOGGER.exception("Failed to load candidate training data for user profile boosts.")
        return CandidateUserProfileStore(user_profiles={})

    available_columns = [
        column
        for column in (
            "customer_id",
            "preferred_garment_group",
            "preferred_colour_master",
            "preferred_main_category",
            "preferred_season",
            "price_band",
            "avg_price",
            "purchase_count",
            "activity_segment",
        )
        if column in training_data.columns
    ]
    if "customer_id" not in available_columns:
        return CandidateUserProfileStore(user_profiles={})

    user_table = (
        training_data.loc[:, available_columns]
        .drop_duplicates(subset=["customer_id"], keep="first")
        .reset_index(drop=True)
    )
    user_profiles = {
        str(record["customer_id"]): record
        for record in user_table.to_dict(orient="records")
    }
    LOGGER.info("Loaded candidate user profile store with %s users", len(user_profiles))
    return CandidateUserProfileStore(user_profiles=user_profiles)


@lru_cache(maxsize=1)
def load_article_catalog() -> pd.DataFrame:
    """Compatibility accessor used by evaluation helpers."""

    return get_cached_feature_store().catalog


@lru_cache(maxsize=1)
def load_two_tower_serving_artifacts() -> TwoTowerServingArtifacts | None:
    """Load cached Two-Tower retrieval assets for hybrid candidate generation."""

    checkpoint_dir = DEFAULT_TWO_TOWER_CHECKPOINT_DIR.expanduser().resolve()
    model_path = checkpoint_dir / "two_tower.pt"
    if not model_path.exists():
        LOGGER.info("Two-Tower checkpoint not found for hybrid candidate generation: %s", model_path)
        return None

    try:
        training_data = load_candidate_training_data()
        model, encoder, dataset_metadata = load_two_tower_artifacts(checkpoint_dir=checkpoint_dir)
        schema_dict = dataset_metadata.get("encoder", {}).get("schema", {})
        user_id_column = str(schema_dict.get("user_id_column", "customer_id"))
        item_embeddings, item_ids, item_records = build_item_embedding_index(
            items=training_data,
            model=model,
            encoder=encoder,
        )
        del item_records
        user_table = build_latest_user_table(training_data, schema=encoder.schema)
        user_records = {
            str(record[user_id_column]): record
            for record in user_table.to_dict(orient="records")
        }
        LOGGER.info(
            "Loaded Two-Tower serving artifacts | users=%s items=%s checkpoint=%s",
            len(user_records),
            len(item_ids),
            checkpoint_dir,
        )
        return TwoTowerServingArtifacts(
            training_data=training_data,
            user_records=user_records,
            item_embeddings=item_embeddings,
            item_ids=item_ids,
        )
    except Exception:
        LOGGER.exception("Failed to load Two-Tower serving artifacts. Hybrid candidate generation will be disabled.")
        return None


@lru_cache(maxsize=1)
def load_two_tower_model_bundle() -> tuple[Any, Any] | None:
    """Cache the in-memory Two-Tower model and encoder for serving-time retrieval."""

    checkpoint_dir = DEFAULT_TWO_TOWER_CHECKPOINT_DIR.expanduser().resolve()
    model_path = checkpoint_dir / "two_tower.pt"
    if not model_path.exists():
        return None
    try:
        model, encoder, _ = load_two_tower_artifacts(checkpoint_dir=checkpoint_dir)
        return model, encoder
    except Exception:
        LOGGER.exception("Failed to cache Two-Tower model bundle for serving.")
        return None


def _retrieve_two_tower_candidates(
    user_id: str,
    top_k: int,
    recent_click_set: set[str],
) -> list[dict[str, Any]]:
    artifacts = load_two_tower_serving_artifacts()
    if artifacts is None:
        return []

    user_record = artifacts.user_records.get(str(user_id))
    if user_record is None:
        return []

    try:
        model_bundle = load_two_tower_model_bundle()
        if model_bundle is None:
            return []
        model, encoder = model_bundle
        user_embedding = encode_user_records(
            model=model,
            encoder=encoder,
            user_records=[user_record],
        )
        if user_embedding.size == 0:
            return []
        return retrieve_top_k(
            user_embedding=user_embedding[0],
            item_embeddings=artifacts.item_embeddings,
            item_ids=artifacts.item_ids,
            top_k=top_k,
            exclude_item_ids=recent_click_set,
        )
    except Exception:
        LOGGER.exception("Two-Tower retrieval failed for user_id=%s", user_id)
        return []


def _build_recent_signal_sets(
    feature_store: ServingFeatureStore,
    recent_clicks: list[str],
) -> tuple[set[str], set[str], set[str], set[str]]:
    """Extract metadata signal sets from recently clicked items."""

    if not recent_clicks:
        return set(), set(), set(), set()

    categories: set[str] = set()
    main_categories: set[str] = set()
    colors: set[str] = set()
    garment_groups: set[str] = set()
    for article_id in recent_clicks:
        record = feature_store.article_records.get(article_id)
        if record is None:
            continue
        categories.add(str(record["category"]))
        main_categories.add(str(record["main_category"]))
        colors.add(str(record["color"]))
        garment_groups.add(_safe_text(record.get("garment_group_name")))
    return categories, main_categories, colors, garment_groups


def _accumulate_scores(
    candidate_scores: dict[str, float],
    candidate_matches: dict[str, bool],
    article_ids: tuple[str, ...],
    increment: float,
    limit: int | None = None,
) -> None:
    iterable = article_ids if limit is None else article_ids[:limit]
    for article_id in iterable:
        candidate_scores[article_id] = candidate_scores.get(article_id, 0.0) + increment
        candidate_matches[article_id] = True


def _materialize_candidates(
    selected_ids: list[str],
    feature_store: ServingFeatureStore,
    candidate_scores: dict[str, float],
    recent_matches: dict[str, bool],
    session_matches: dict[str, bool],
    candidate_reason: str,
    reason_by_article: dict[str, str] | None = None,
) -> pd.DataFrame:
    if not selected_ids:
        return pd.DataFrame(columns=LOOKUP_COLUMNS)

    frame = feature_store.candidate_frame.loc[selected_ids].copy()
    frame["candidate_score"] = [float(candidate_scores[article_id]) for article_id in selected_ids]
    frame["candidate_reason"] = [
        str((reason_by_article or {}).get(article_id, candidate_reason))
        for article_id in selected_ids
    ]
    frame["matches_recent_click_signal"] = [bool(recent_matches.get(article_id, False)) for article_id in selected_ids]
    frame["matches_session_interest"] = [bool(session_matches.get(article_id, False)) for article_id in selected_ids]
    return frame.reset_index(drop=True)


def _cold_start_candidates(
    feature_store: ServingFeatureStore,
    recent_click_set: set[str],
    candidate_pool_size: int,
) -> pd.DataFrame:
    selected_ids: list[str] = []
    candidate_scores: dict[str, float] = {}

    for article_id in feature_store.popular_article_ids:
        if article_id in recent_click_set:
            continue
        record = feature_store.article_records[article_id]
        candidate_scores[article_id] = float(record["normalized_popularity"]) + float(record["cold_start_bonus"]) + float(record["fresh_boost"])
        selected_ids.append(article_id)
        if len(selected_ids) >= candidate_pool_size:
            break

    return _materialize_candidates(
        selected_ids=selected_ids,
        feature_store=feature_store,
        candidate_scores=candidate_scores,
        recent_matches={},
        session_matches={},
        candidate_reason="cold_start_popularity",
    )


def _apply_user_profile_signals(
    *,
    user_id: str,
    feature_store: ServingFeatureStore,
    candidate_scores: dict[str, float],
    lookup_limit: int,
    user_profile_store: CandidateUserProfileStore | None = None,
    user_profile_override: dict[str, Any] | None = None,
) -> bool:
    """Boost candidates using long-term purchase preferences when available."""

    user_profile = user_profile_override
    if user_profile is None:
        profile_store = user_profile_store or load_candidate_user_profiles()
        user_profile = profile_store.user_profiles.get(str(user_id))
    if user_profile is None:
        return False

    preferred_main_category = _safe_text(user_profile.get("preferred_main_category"))
    preferred_colour_master = _safe_text(user_profile.get("preferred_colour_master"))
    preferred_garment_group = _safe_text(user_profile.get("preferred_garment_group"))
    preferred_season = _safe_text(user_profile.get("preferred_season"))
    preferred_price_band = _safe_text(user_profile.get("price_band"))
    avg_price = pd.to_numeric(user_profile.get("avg_price"), errors="coerce")

    profile_seeded = False
    if preferred_main_category != "UNKNOWN":
        for article_id in feature_store.main_category_to_ids.get(preferred_main_category, ())[:lookup_limit]:
            candidate_scores[article_id] = candidate_scores.get(article_id, 0.0) + 2.5
        profile_seeded = True

    if preferred_colour_master != "UNKNOWN":
        for article_id in feature_store.color_to_ids.get(preferred_colour_master, ())[:lookup_limit]:
            candidate_scores[article_id] = candidate_scores.get(article_id, 0.0) + 1.5
        profile_seeded = True

    if preferred_garment_group != "UNKNOWN":
        for article_id in feature_store.garment_group_to_ids.get(preferred_garment_group, ())[:lookup_limit]:
            candidate_scores[article_id] = candidate_scores.get(article_id, 0.0) + 1.5
        profile_seeded = True

    if preferred_price_band != "UNKNOWN":
        for article_id in feature_store.price_band_to_ids.get(preferred_price_band, ())[:lookup_limit]:
            candidate_scores[article_id] = candidate_scores.get(article_id, 0.0) + 0.75

    if pd.notna(avg_price):
        user_avg_price = float(avg_price)
        candidate_price_band = _derive_price_band(user_avg_price)
        for article_id in feature_store.price_band_to_ids.get(candidate_price_band, ())[:lookup_limit]:
            record = feature_store.article_records[article_id]
            item_price = pd.to_numeric(record.get("avg_price"), errors="coerce")
            price_gap = abs(float(item_price) - user_avg_price)
            candidate_scores[article_id] = candidate_scores.get(article_id, 0.0) + max(0.0, 1.0 - min(price_gap / 0.05, 1.0))

    return profile_seeded


def _set_candidate_reason(
    reason_by_article: dict[str, str],
    reason_priority_by_article: dict[str, int],
    article_id: str,
    reason: str,
    priority: int,
) -> None:
    if priority >= reason_priority_by_article.get(article_id, -1):
        reason_by_article[article_id] = reason
        reason_priority_by_article[article_id] = priority


def _apply_copurchase_signals(
    *,
    feature_store: ServingFeatureStore,
    recent_clicks: list[str],
    recent_click_set: set[str],
    candidate_scores: dict[str, float],
    recent_matches: dict[str, bool],
    reason_by_article: dict[str, str],
    reason_priority_by_article: dict[str, int],
    lookup_limit: int,
    score_weight: float = DEFAULT_COPURCHASE_SCORE_WEIGHT,
    min_cooccurrence_count: int = 1,
    copurchase_store: CopurchaseNeighborStore | None = None,
) -> list[str]:
    """Expand recent purchase seeds using precomputed co-purchase neighbors."""

    if not recent_clicks:
        return []

    store = copurchase_store or load_copurchase_neighbor_store()
    if not store.neighbors_by_article:
        return []

    prioritized_article_ids: list[str] = []
    local_seen: set[str] = set()
    total_recent = len(recent_clicks)
    for recency_index, seed_article_id in enumerate(reversed(recent_clicks), start=1):
        seed_neighbors = store.neighbors_by_article.get(seed_article_id, ())
        if not seed_neighbors:
            continue
        seed_multiplier = max(0.75, 1.5 - (0.2 * float(recency_index - 1)))
        accepted_rank = 0
        for neighbor_article_id, weighted_score, cooccurrence_count in seed_neighbors:
            if cooccurrence_count < min_cooccurrence_count:
                continue
            accepted_rank += 1
            if accepted_rank > lookup_limit:
                break
            if neighbor_article_id in recent_click_set or neighbor_article_id not in feature_store.article_records:
                continue
            rank_bonus = float(score_weight) / float(accepted_rank)
            score_bonus = min(float(weighted_score), 5.0) * seed_multiplier
            recency_bonus = max(0.0, float(total_recent - recency_index + 1)) * 0.1
            candidate_scores[neighbor_article_id] = candidate_scores.get(neighbor_article_id, 0.0) + rank_bonus + score_bonus + recency_bonus
            recent_matches[neighbor_article_id] = True
            _set_candidate_reason(
                reason_by_article=reason_by_article,
                reason_priority_by_article=reason_priority_by_article,
                article_id=neighbor_article_id,
                reason="recent_purchase_copurchase",
                priority=30,
            )
            if neighbor_article_id not in local_seen:
                prioritized_article_ids.append(neighbor_article_id)
                local_seen.add(neighbor_article_id)
    return prioritized_article_ids


def _apply_sequential_article_signals(
    *,
    feature_store: ServingFeatureStore,
    recent_clicks: list[str],
    recent_click_set: set[str],
    candidate_scores: dict[str, float],
    recent_matches: dict[str, bool],
    reason_by_article: dict[str, str],
    reason_priority_by_article: dict[str, int],
    lookup_limit: int,
    score_weight: float,
    transition_store: SequentialTransitionStore | None = None,
) -> list[str]:
    store = transition_store or load_sequential_transition_store()
    if not recent_clicks or not store.article_neighbors_by_article:
        return []

    prioritized_article_ids: list[str] = []
    local_seen: set[str] = set()
    for recency_index, seed_article_id in enumerate(reversed(recent_clicks), start=1):
        neighbors = store.article_neighbors_by_article.get(seed_article_id, ())
        if not neighbors:
            continue
        seed_multiplier = max(0.75, 1.5 - (0.2 * float(recency_index - 1)))
        for rank, (next_article_id, weighted_score, transition_count) in enumerate(neighbors[:lookup_limit], start=1):
            if next_article_id in recent_click_set or next_article_id not in feature_store.article_records:
                continue
            boost = (float(score_weight) / float(rank)) + min(float(weighted_score), 5.0) * seed_multiplier + min(int(transition_count), 5) * 0.1
            candidate_scores[next_article_id] = candidate_scores.get(next_article_id, 0.0) + boost
            recent_matches[next_article_id] = True
            _set_candidate_reason(reason_by_article, reason_priority_by_article, next_article_id, "sequential_article_transition", 40)
            if next_article_id not in local_seen:
                prioritized_article_ids.append(next_article_id)
                local_seen.add(next_article_id)
    return prioritized_article_ids


def _apply_sequential_category_signals(
    *,
    feature_store: ServingFeatureStore,
    recent_clicks: list[str],
    recent_click_set: set[str],
    candidate_scores: dict[str, float],
    recent_matches: dict[str, bool],
    reason_by_article: dict[str, str],
    reason_priority_by_article: dict[str, int],
    lookup_limit: int,
    score_weight: float,
    transition_store: SequentialTransitionStore | None = None,
) -> list[str]:
    store = transition_store or load_sequential_transition_store()
    if not recent_clicks or not store.article_neighbors_by_category:
        return []

    prioritized_article_ids: list[str] = []
    local_seen: set[str] = set()
    seen_categories: set[str] = set()
    for recency_index, seed_article_id in enumerate(reversed(recent_clicks), start=1):
        record = feature_store.article_records.get(seed_article_id)
        if record is None:
            continue
        main_category = _safe_text(record.get("main_category"))
        if main_category == "UNKNOWN" or main_category in seen_categories:
            continue
        seen_categories.add(main_category)
        neighbors = store.article_neighbors_by_category.get(main_category, ())
        if not neighbors:
            continue
        seed_multiplier = max(0.75, 1.4 - (0.15 * float(recency_index - 1)))
        for rank, (next_article_id, weighted_score, transition_count) in enumerate(neighbors[:lookup_limit], start=1):
            if next_article_id in recent_click_set or next_article_id not in feature_store.article_records:
                continue
            boost = (float(score_weight) / float(rank)) + min(float(weighted_score), 5.0) * seed_multiplier + min(int(transition_count), 5) * 0.05
            candidate_scores[next_article_id] = candidate_scores.get(next_article_id, 0.0) + boost
            recent_matches[next_article_id] = True
            _set_candidate_reason(reason_by_article, reason_priority_by_article, next_article_id, "sequential_category_transition", 35)
            if next_article_id not in local_seen:
                prioritized_article_ids.append(next_article_id)
                local_seen.add(next_article_id)
    return prioritized_article_ids


def _derive_price_band(avg_price: float) -> str:
    if avg_price < 0.01:
        return "budget"
    if avg_price < 0.03:
        return "mid"
    return "premium"


def _stable_hash_int(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:16], 16)


def _select_coverage_exploration_ids(
    *,
    feature_store: ServingFeatureStore,
    user_id: str,
    recent_click_set: set[str],
    limit: int,
) -> list[str]:
    """Select deterministic user-specific long-tail candidates for catalog coverage."""

    if limit <= 0 or not feature_store.popular_article_ids:
        return []

    catalog_ids = feature_store.popular_article_ids
    long_tail_start = min(int(len(catalog_ids) * 0.15), max(len(catalog_ids) - 1, 0))
    exploration_pool = catalog_ids[long_tail_start:] or catalog_ids
    seed = _stable_hash_int(str(user_id))
    offset = seed % len(exploration_pool)
    step = (seed % max(len(exploration_pool) - 1, 1)) + 1
    while math.gcd(step, len(exploration_pool)) != 1 and step < len(exploration_pool):
        step += 1

    selected_ids: list[str] = []
    category_counts: dict[str, int] = {}
    for probe_index in range(len(exploration_pool)):
        article_id = exploration_pool[(offset + probe_index * step) % len(exploration_pool)]
        if article_id in recent_click_set or article_id not in feature_store.article_records:
            continue
        category = _safe_text(feature_store.article_records[article_id].get("main_category"))
        if category_counts.get(category, 0) >= 20:
            continue
        selected_ids.append(article_id)
        category_counts[category] = category_counts.get(category, 0) + 1
        if len(selected_ids) >= limit:
            break
    return selected_ids


def generate_candidates(
    user_id: str,
    top_k: int,
    recent_clicks: list[str] | None = None,
    session_interest: dict[str, Any] | None = None,
    candidate_pool_size: int | None = None,
    feature_store: ServingFeatureStore | None = None,
    user_profile_store: CandidateUserProfileStore | None = None,
    user_profile_override: dict[str, Any] | None = None,
    include_two_tower: bool = True,
    include_sequential: bool = True,
    include_copurchase: bool = False,
    sequential_store: SequentialTransitionStore | None = None,
    copurchase_store: CopurchaseNeighborStore | None = None,
    sequential_article_limit: int = DEFAULT_SEQUENTIAL_ARTICLE_LIMIT,
    sequential_category_limit: int = DEFAULT_SEQUENTIAL_CATEGORY_LIMIT,
    sequential_article_score_weight: float = DEFAULT_SEQUENTIAL_ARTICLE_SCORE_WEIGHT,
    sequential_category_score_weight: float = DEFAULT_SEQUENTIAL_CATEGORY_SCORE_WEIGHT,
    copurchase_neighbor_limit: int = DEFAULT_COPURCHASE_NEIGHBOR_LIMIT,
    copurchase_score_weight: float = DEFAULT_COPURCHASE_SCORE_WEIGHT,
    copurchase_min_cooccurrence_count: int = 1,
    include_coverage_exploration: bool = True,
    coverage_exploration_limit: int = DEFAULT_COVERAGE_EXPLORATION_LIMIT,
) -> pd.DataFrame:
    """Generate ranking candidates using catalog metadata and popularity.

    This keeps a cold-start-safe fallback in place until a dedicated candidate
    retrieval service is connected. Sequential transitions are the default
    short-history signal; co-purchase is opt-in for experiments.
    """

    feature_store = feature_store or get_cached_feature_store()
    recent_clicks = [normalize_article_id(article_id) for article_id in (recent_clicks or []) if str(article_id).strip()]
    recent_click_set = set(recent_clicks)
    session_interest = session_interest or {}

    candidate_pool_size = candidate_pool_size or max(
        DEFAULT_CANDIDATE_POOL_SIZE,
        int(top_k * DEFAULT_CANDIDATE_POOL_MULTIPLIER),
    )
    cold_start = not recent_clicks and not session_interest
    signal_lookup_limit = max(candidate_pool_size * DEFAULT_SIGNAL_LOOKUP_LIMIT_MULTIPLIER, candidate_pool_size)
    profile_only_start = False

    if cold_start:
        bootstrap_scores: dict[str, float] = {}
        profile_only_start = _apply_user_profile_signals(
            user_id=user_id,
            feature_store=feature_store,
            candidate_scores=bootstrap_scores,
            lookup_limit=signal_lookup_limit,
            user_profile_store=user_profile_store,
            user_profile_override=user_profile_override,
        )
        if not profile_only_start:
            filtered = _cold_start_candidates(
                feature_store=feature_store,
                recent_click_set=recent_click_set,
                candidate_pool_size=candidate_pool_size,
            )
            LOGGER.info(
                "Generated %s candidates for ranking (top_k=%s, cold_start=%s)",
                len(filtered),
                top_k,
                cold_start,
            )
            return filtered.reset_index(drop=True)

    candidate_scores: dict[str, float] = {}
    recent_matches: dict[str, bool] = {}
    session_matches: dict[str, bool] = {}
    reason_by_article: dict[str, str] = {}
    reason_priority_by_article: dict[str, int] = {}
    if cold_start and profile_only_start:
        _apply_user_profile_signals(
            user_id=user_id,
            feature_store=feature_store,
            candidate_scores=candidate_scores,
            lookup_limit=signal_lookup_limit,
            user_profile_store=user_profile_store,
            user_profile_override=user_profile_override,
        )

    prioritized_sequential_article_ids: list[str] = []
    prioritized_sequential_category_ids: list[str] = []
    if include_sequential and recent_clicks:
        prioritized_sequential_article_ids = _apply_sequential_article_signals(
            feature_store=feature_store,
            recent_clicks=recent_clicks,
            recent_click_set=recent_click_set,
            candidate_scores=candidate_scores,
            recent_matches=recent_matches,
            reason_by_article=reason_by_article,
            reason_priority_by_article=reason_priority_by_article,
            lookup_limit=min(signal_lookup_limit, int(sequential_article_limit)),
            score_weight=float(sequential_article_score_weight),
            transition_store=sequential_store,
        )
        prioritized_sequential_category_ids = _apply_sequential_category_signals(
            feature_store=feature_store,
            recent_clicks=recent_clicks,
            recent_click_set=recent_click_set,
            candidate_scores=candidate_scores,
            recent_matches=recent_matches,
            reason_by_article=reason_by_article,
            reason_priority_by_article=reason_priority_by_article,
            lookup_limit=min(signal_lookup_limit, int(sequential_category_limit)),
            score_weight=float(sequential_category_score_weight),
            transition_store=sequential_store,
        )

    prioritized_copurchase_ids: list[str] = []
    if include_copurchase and recent_clicks:
        prioritized_copurchase_ids = _apply_copurchase_signals(
            feature_store=feature_store,
            recent_clicks=recent_clicks,
            recent_click_set=recent_click_set,
            candidate_scores=candidate_scores,
            recent_matches=recent_matches,
            reason_by_article=reason_by_article,
            reason_priority_by_article=reason_priority_by_article,
            lookup_limit=min(signal_lookup_limit, int(copurchase_neighbor_limit)),
            score_weight=float(copurchase_score_weight),
            min_cooccurrence_count=int(copurchase_min_cooccurrence_count),
            copurchase_store=copurchase_store,
        )

    if recent_clicks:
        categories, main_categories, colors, garment_groups = _build_recent_signal_sets(feature_store, recent_clicks)
        for category in categories:
            _accumulate_scores(
                candidate_scores,
                recent_matches,
                feature_store.category_to_ids.get(category, ()),
                3.0,
                limit=signal_lookup_limit,
            )
        for main_category in main_categories:
            _accumulate_scores(
                candidate_scores,
                recent_matches,
                feature_store.main_category_to_ids.get(main_category, ()),
                2.0,
                limit=signal_lookup_limit,
            )
        for color in colors:
            _accumulate_scores(
                candidate_scores,
                recent_matches,
                feature_store.color_to_ids.get(color, ()),
                1.0,
                limit=signal_lookup_limit,
            )
        for garment_group in garment_groups:
            _accumulate_scores(
                candidate_scores,
                recent_matches,
                feature_store.garment_group_to_ids.get(garment_group, ()),
                1.5,
                limit=signal_lookup_limit,
            )

    for category, weight in session_interest.items():
        normalized_weight = float(weight) if weight is not None else 0.0
        normalized_category = _safe_text(category)
        _accumulate_scores(
            candidate_scores,
            session_matches,
            feature_store.category_to_ids.get(normalized_category, ()),
            normalized_weight * 4.0,
            limit=signal_lookup_limit,
        )
        _accumulate_scores(
            candidate_scores,
            session_matches,
            feature_store.main_category_to_ids.get(normalized_category, ()),
            normalized_weight * 2.0,
            limit=signal_lookup_limit,
        )

    _apply_user_profile_signals(
        user_id=user_id,
        feature_store=feature_store,
        candidate_scores=candidate_scores,
        lookup_limit=signal_lookup_limit,
        user_profile_store=user_profile_store,
        user_profile_override=user_profile_override,
    )

    for article_id, signal_score in list(candidate_scores.items()):
        if article_id in recent_click_set:
            del candidate_scores[article_id]
            recent_matches.pop(article_id, None)
            session_matches.pop(article_id, None)
            continue

        record = feature_store.article_records[article_id]
        candidate_scores[article_id] = signal_score + float(record["normalized_popularity"]) + float(record["fresh_boost"])

    if len(candidate_scores) < candidate_pool_size:
        for article_id in feature_store.popular_article_ids:
            if article_id in recent_click_set or article_id in candidate_scores:
                continue
            record = feature_store.article_records[article_id]
            candidate_scores[article_id] = float(record["normalized_popularity"]) + float(record["fresh_boost"])
            if len(candidate_scores) >= candidate_pool_size:
                break

    two_tower_candidates: list[dict[str, Any]] = []
    if include_two_tower:
        hybrid_top_k = max(1, int(candidate_pool_size * DEFAULT_HYBRID_TWO_TOWER_RATIO))
        two_tower_candidates = _retrieve_two_tower_candidates(
            user_id=user_id,
            top_k=hybrid_top_k,
            recent_click_set=recent_click_set,
        )
    prioritized_two_tower_ids: list[str] = []
    for rank, row in enumerate(two_tower_candidates):
        article_id = normalize_article_id(row.get("article_id"))
        if article_id in recent_click_set or article_id not in feature_store.article_records:
            continue
        score = float(row.get("score", 0.0))
        score_bonus = DEFAULT_TWO_TOWER_SCORE_WEIGHT / float(rank + 1)
        candidate_scores[article_id] = candidate_scores.get(article_id, 0.0) + score_bonus + max(score, 0.0) * 0.01
        recent_matches.setdefault(article_id, False)
        session_matches.setdefault(article_id, False)
        prioritized_two_tower_ids.append(article_id)
        _set_candidate_reason(
            reason_by_article=reason_by_article,
            reason_priority_by_article=reason_priority_by_article,
            article_id=article_id,
            reason="two_tower_candidate",
            priority=20,
        )

    prioritized_coverage_ids: list[str] = []
    if include_coverage_exploration:
        prioritized_coverage_ids = _select_coverage_exploration_ids(
            feature_store=feature_store,
            user_id=user_id,
            recent_click_set=recent_click_set,
            limit=min(int(coverage_exploration_limit), candidate_pool_size),
        )
        for rank, article_id in enumerate(prioritized_coverage_ids, start=1):
            candidate_scores[article_id] = candidate_scores.get(article_id, 0.0) + max(0.01, 0.2 / float(rank))
            recent_matches.setdefault(article_id, False)
            session_matches.setdefault(article_id, False)
            _set_candidate_reason(
                reason_by_article=reason_by_article,
                reason_priority_by_article=reason_priority_by_article,
                article_id=article_id,
                reason="coverage_exploration",
                priority=10,
            )

    ranked_ids = sorted(
        candidate_scores,
        key=lambda article_id: (
            -candidate_scores[article_id],
            -float(feature_store.article_records[article_id]["popularity"]),
            article_id,
        ),
    )
    selected_ids: list[str] = []
    seen_selected: set[str] = set()
    for article_id in prioritized_sequential_article_ids:
        if article_id in seen_selected:
            continue
        selected_ids.append(article_id)
        seen_selected.add(article_id)
        if len(selected_ids) >= candidate_pool_size:
            break

    for article_id in prioritized_sequential_category_ids:
        if article_id in seen_selected:
            continue
        selected_ids.append(article_id)
        seen_selected.add(article_id)
        if len(selected_ids) >= candidate_pool_size:
            break

    for article_id in prioritized_copurchase_ids:
        if article_id in seen_selected:
            continue
        selected_ids.append(article_id)
        seen_selected.add(article_id)
        if len(selected_ids) >= candidate_pool_size:
            break

    for article_id in prioritized_two_tower_ids:
        if article_id in seen_selected:
            continue
        selected_ids.append(article_id)
        seen_selected.add(article_id)
        if len(selected_ids) >= candidate_pool_size:
            break

    for article_id in prioritized_coverage_ids:
        if article_id in seen_selected:
            continue
        selected_ids.append(article_id)
        seen_selected.add(article_id)
        if len(selected_ids) >= candidate_pool_size:
            break

    if len(selected_ids) < candidate_pool_size:
        for article_id in ranked_ids:
            if article_id in seen_selected:
                continue
            selected_ids.append(article_id)
            seen_selected.add(article_id)
            if len(selected_ids) >= candidate_pool_size:
                break
    filtered = _materialize_candidates(
        selected_ids=selected_ids,
        feature_store=feature_store,
        candidate_scores=candidate_scores,
        recent_matches=recent_matches,
        session_matches=session_matches,
        candidate_reason="candidate_retrieval",
        reason_by_article=reason_by_article,
    )

    LOGGER.info(
        "Generated %s candidates for ranking (top_k=%s, cold_start=%s)",
        len(filtered),
        top_k,
        cold_start,
    )
    return filtered.reset_index(drop=True)
