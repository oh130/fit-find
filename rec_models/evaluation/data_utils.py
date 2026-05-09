"""Shared data-loading helpers for recommendation offline evaluation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from rec_models.serving.candidate_service import normalize_article_id
except ImportError:  # pragma: no cover - supports running from rec_models/ as cwd
    from serving.candidate_service import normalize_article_id  # type: ignore[no-redef]


@dataclass(slots=True)
class EvaluationContext:
    """Reusable evaluation context shared across offline evaluators."""

    data: pd.DataFrame
    positive_mask: pd.Series
    ground_truth_by_user: dict[str, list[str]]
    sampled_user_ids: list[str]
    user_rows_by_id: dict[str, pd.DataFrame]
    total_candidate_count: int


def load_evaluation_data(data_path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    """Load evaluation rows from CSV and normalize core identifiers."""

    resolved_path = data_path.expanduser().resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(f"Evaluation data not found: {resolved_path}")

    if columns is None:
        data = pd.read_csv(resolved_path)
    else:
        header = pd.read_csv(resolved_path, nrows=0)
        available_columns = [column for column in columns if column in header.columns]
        data = pd.read_csv(resolved_path, usecols=available_columns)
    required_columns = {"customer_id", "article_id"}
    missing_columns = required_columns - set(data.columns)
    if missing_columns:
        raise ValueError(f"Evaluation data is missing required columns: {sorted(missing_columns)}")

    data["customer_id"] = data["customer_id"].astype(str)
    data["article_id"] = data["article_id"].map(normalize_article_id)
    return data


def infer_positive_mask(data: pd.DataFrame) -> pd.Series:
    """Infer positive interactions using supported processed-data conventions."""

    if "label" in data.columns:
        return pd.to_numeric(data["label"], errors="coerce").fillna(0).eq(1)

    for column in ("is_positive", "target", "clicked", "purchased"):
        if column in data.columns:
            return data[column].astype(str).str.lower().isin({"1", "true", "yes"})

    if "ground_truth_article_id" in data.columns:
        return data["article_id"].astype(str).eq(data["ground_truth_article_id"].map(normalize_article_id))

    if "sales_channel_id" in data.columns:
        return pd.to_numeric(data["sales_channel_id"], errors="coerce").fillna(-1).ne(-1)

    raise ValueError(
        "Could not infer positive rows. Expected one of: label, is_positive, target, "
        "clicked, purchased, ground_truth_article_id, or sales_channel_id."
    )


def build_ground_truth_by_user(data: pd.DataFrame, positive_mask: pd.Series) -> dict[str, list[str]]:
    """Build user -> positive article ids from evaluation rows."""

    positives = data.loc[positive_mask, ["customer_id", "article_id"]].drop_duplicates()
    grouped = positives.groupby("customer_id", sort=False)["article_id"].apply(list)
    return grouped.to_dict()


def stable_user_sample(user_ids: list[str], max_users: int | None, seed: int = 42) -> list[str]:
    """Deterministically subsample users while preserving reproducibility."""

    if max_users is None or len(user_ids) <= max_users:
        return user_ids

    ranked = sorted(
        user_ids,
        key=lambda user_id: hashlib.blake2b(f"{seed}:{user_id}".encode("utf-8"), digest_size=8).hexdigest(),
    )
    return ranked[:max_users]


def build_session_context(user_rows: pd.DataFrame) -> dict[str, Any]:
    """Extract a serving-compatible session context from optional evaluation columns."""

    recent_clicks: list[str] = []
    session_interest: Any = None

    if "recent_clicks" in user_rows.columns:
        raw_value = user_rows["recent_clicks"].dropna().astype(str).head(1)
        if not raw_value.empty:
            recent_clicks = [normalize_article_id(item_id) for item_id in raw_value.iloc[0].split(",") if item_id.strip()]

    if "session_interest" in user_rows.columns:
        raw_interest = user_rows["session_interest"].dropna().head(1)
        if not raw_interest.empty:
            session_interest = raw_interest.iloc[0]

    return {
        "recent_clicks": recent_clicks,
        "session_interest": session_interest,
    }


def build_user_rows_by_id(data: pd.DataFrame, user_ids: list[str]) -> dict[str, pd.DataFrame]:
    """Materialize per-user frames once so evaluators can reuse them."""

    filtered = data.loc[data["customer_id"].astype(str).isin(set(user_ids))].copy()
    if filtered.empty:
        return {}
    return {
        str(user_id): user_rows.copy()
        for user_id, user_rows in filtered.groupby("customer_id", sort=False)
    }


def build_evaluation_context(
    data: pd.DataFrame,
    max_users: int | None = None,
    positive_mask: pd.Series | None = None,
    sampled_user_ids: list[str] | None = None,
) -> EvaluationContext:
    """Build reusable user/grouping state for offline evaluators."""

    resolved_positive_mask = positive_mask if positive_mask is not None else infer_positive_mask(data)
    ground_truth_by_user = build_ground_truth_by_user(data, resolved_positive_mask)
    resolved_user_ids = (
        sampled_user_ids
        if sampled_user_ids is not None
        else stable_user_sample(list(ground_truth_by_user.keys()), max_users=max_users)
    )
    return EvaluationContext(
        data=data,
        positive_mask=resolved_positive_mask,
        ground_truth_by_user=ground_truth_by_user,
        sampled_user_ids=resolved_user_ids,
        user_rows_by_id=build_user_rows_by_id(data, resolved_user_ids),
        total_candidate_count=int(data["article_id"].astype(str).nunique()),
    )
