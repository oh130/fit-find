"""Ranking model training entry points for logistic baseline and DeepFM."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:  # pragma: no cover - DeepFM requires torch
    torch = None
    nn = None  # type: ignore[assignment]
    DataLoader = Any  # type: ignore[assignment]
    TensorDataset = Any  # type: ignore[assignment]

try:
    from rec_models.ranking.model import DeepFMConfig, DeepFMRanker
except ImportError:  # pragma: no cover - supports running from rec_models/ as cwd
    from ranking.model import DeepFMConfig, DeepFMRanker  # type: ignore[no-redef]


LOGGER = logging.getLogger(__name__)

TARGET_COLUMN = "label"
IDENTIFIER_COLUMNS = ("customer_id", "article_id")
LEAKAGE_COLUMNS = {"label", "customer_id", "article_id", "price", "sales_channel_id"}
DEFAULT_VALIDATION_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42
DEFAULT_CHECKPOINT_DIR = Path(__file__).resolve().parents[1] / "checkpoints"
BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_TRAINING_DATA_PATH = BASE_DIR / "data" / "processed" / "train_data_dev.csv"
ITEM_FEATURES_PATH = BASE_DIR / "data" / "processed" / "item_features.csv"
ITEM_FEATURES_DEV_PATH = BASE_DIR / "data" / "processed" / "item_features_dev.csv"
ITEM_FEATURES_TEST_PATH = BASE_DIR / "data" / "processed" / "item_features_test.csv"
USER_PERSONA_PATH = BASE_DIR / "data" / "processed" / "user_persona_scores.csv"
USER_PERSONA_DEV_PATH = BASE_DIR / "data" / "processed" / "user_persona_scores_dev.csv"
USER_PERSONA_TEST_PATH = BASE_DIR / "data" / "processed" / "user_persona_scores_test.csv"
ITEM_PERSONA_PATH = BASE_DIR / "data" / "processed" / "item_persona_scores.csv"
ITEM_PERSONA_DEV_PATH = BASE_DIR / "data" / "processed" / "item_persona_scores_dev.csv"
ITEM_PERSONA_TEST_PATH = BASE_DIR / "data" / "processed" / "item_persona_scores_test.csv"
PIPELINE_ARTIFACT_NAME = "ranking_baseline.joblib"
METADATA_ARTIFACT_NAME = "ranking_baseline_metadata.json"
DEEPFM_ARTIFACT_NAME = "ranking_deepfm.pt"
DEEPFM_METADATA_ARTIFACT_NAME = "ranking_deepfm_metadata.json"
DEFAULT_DEEPFM_BATCH_SIZE = 256
DEFAULT_DEEPFM_EPOCHS = 10
DEFAULT_DEEPFM_LEARNING_RATE = 1e-3
DEFAULT_DEEPFM_WEIGHT_DECAY = 1e-5
DEFAULT_LOGREG_MAX_ITER = 3000
PERSONAS = (
    "trendsetter",
    "practical",
    "value",
    "brand_loyal",
    "impulse",
    "careful",
    "repeat_stable",
    "color_focus",
    "category_focus",
)
PERSONA_RATIO_COLUMNS = tuple(f"{persona}_ratio" for persona in PERSONAS)
USER_PERSONA_FEATURE_COLUMNS = tuple(f"user_{column}" for column in PERSONA_RATIO_COLUMNS)
ITEM_PERSONA_FEATURE_COLUMNS = tuple(f"item_{column}" for column in PERSONA_RATIO_COLUMNS)
PERSONA_FEATURE_COLUMNS = (
    *USER_PERSONA_FEATURE_COLUMNS,
    "user_top_persona",
    "user_top_persona_ratio",
    *ITEM_PERSONA_FEATURE_COLUMNS,
    "item_top_persona",
    "item_top_persona_ratio",
    "top_persona_match",
    "persona_match_score",
)


@dataclass(slots=True)
class TrainingArtifacts:
    target_column: str
    identifier_columns: list[str]
    feature_columns: list[str]
    numeric_columns: list[str]
    categorical_columns: list[str]
    model_artifact: str
    created_at_utc: str
    validation_size: float
    random_state: int
    split_mode: str


def configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


def _require_torch() -> None:
    if torch is None or nn is None:
        raise ImportError("torch is required to train the DeepFM ranking model.")


def load_training_data(csv_path: Path) -> pd.DataFrame:
    csv_path = csv_path.expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"Training data not found: {csv_path}")

    LOGGER.info("Loading training data from %s", csv_path)
    df = pd.read_csv(csv_path)
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Training data must include a '{TARGET_COLUMN}' column.")
    if "article_id" in df.columns:
        df["article_id"] = df["article_id"].astype(str).str.strip().str.zfill(10)
    return enrich_with_persona_features(enrich_with_item_features(df))


def _resolve_item_features_path() -> Path | None:
    for path in (ITEM_FEATURES_DEV_PATH, ITEM_FEATURES_PATH, ITEM_FEATURES_TEST_PATH):
        if path.exists():
            return path
    return None


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


def enrich_with_item_features(df: pd.DataFrame) -> pd.DataFrame:
    """Attach item-level numeric features used by ranking models."""

    required_columns = ("popularity", "avg_price", "item_age_days", "is_new_item")
    missing_columns = [column for column in required_columns if column not in df.columns]
    if not missing_columns:
        return df

    item_feature_path = _resolve_item_features_path()
    if item_feature_path is None:
        LOGGER.warning("Item feature file not found. Continuing without additional ranking item features.")
        enriched = df.copy()
        for column in missing_columns:
            if column == "is_new_item":
                enriched[column] = False
            else:
                enriched[column] = 0.0
        return enriched

    item_features = pd.read_csv(item_feature_path, dtype={"article_id": str}).fillna("")
    merge_columns = ["article_id", *[column for column in required_columns if column in item_features.columns]]
    enriched = df.merge(
        item_features.loc[:, merge_columns].drop_duplicates("article_id"),
        on="article_id",
        how="left",
        suffixes=("", "_item"),
    )

    for column in ("popularity", "avg_price", "item_age_days"):
        if column not in enriched.columns:
            enriched[column] = 0.0
        enriched[column] = pd.to_numeric(enriched[column], errors="coerce").fillna(0.0)

    if "is_new_item" not in enriched.columns:
        enriched["is_new_item"] = 0
    else:
        normalized = (
            enriched["is_new_item"]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin({"1", "true", "yes", "y"})
            .astype(int)
        )
        enriched["is_new_item"] = normalized

    LOGGER.info("Merged ranking item features from %s", item_feature_path)
    return enriched


@lru_cache(maxsize=6)
def _load_persona_frame(path: Path | None, key_column: str, prefix: str) -> pd.DataFrame:
    output_columns = [f"{prefix}_{column}" for column in PERSONA_RATIO_COLUMNS] + [
        f"{prefix}_top_persona",
        f"{prefix}_top_persona_ratio",
    ]
    if path is None:
        LOGGER.warning("%s persona score file not found. Using default persona features.", prefix.capitalize())
        return pd.DataFrame(columns=[key_column, *output_columns])

    required_columns = [key_column, *PERSONA_RATIO_COLUMNS, "top_persona", "top_persona_ratio"]
    persona_frame = pd.read_csv(path, dtype={key_column: str})
    missing_columns = [column for column in required_columns if column not in persona_frame.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in {path}: {', '.join(missing_columns)}")

    if key_column == "article_id":
        persona_frame[key_column] = persona_frame[key_column].astype(str).str.strip().str.zfill(10)
    else:
        persona_frame[key_column] = persona_frame[key_column].astype(str).str.strip()

    rename_map = {column: f"{prefix}_{column}" for column in PERSONA_RATIO_COLUMNS}
    rename_map["top_persona"] = f"{prefix}_top_persona"
    rename_map["top_persona_ratio"] = f"{prefix}_top_persona_ratio"
    return persona_frame.loc[:, required_columns].rename(columns=rename_map).drop_duplicates(key_column)


def _fill_persona_defaults(df: pd.DataFrame) -> pd.DataFrame:
    filled = df.copy()
    for column in USER_PERSONA_FEATURE_COLUMNS + ITEM_PERSONA_FEATURE_COLUMNS:
        values = filled[column] if column in filled.columns else pd.Series(0.0, index=filled.index)
        filled[column] = pd.to_numeric(values, errors="coerce").fillna(0.0)
    for column in ("user_top_persona_ratio", "item_top_persona_ratio", "persona_match_score"):
        values = filled[column] if column in filled.columns else pd.Series(0.0, index=filled.index)
        filled[column] = pd.to_numeric(values, errors="coerce").fillna(0.0)
    for column in ("user_top_persona", "item_top_persona"):
        values = filled[column] if column in filled.columns else pd.Series("UNKNOWN", index=filled.index)
        filled[column] = values.fillna("UNKNOWN").astype(str)
        filled[column] = filled[column].str.strip().replace("", "UNKNOWN")
    if "top_persona_match" not in filled.columns:
        filled["top_persona_match"] = 0
    filled["top_persona_match"] = pd.to_numeric(filled["top_persona_match"], errors="coerce").fillna(0).astype(int)
    return filled


def enrich_with_persona_features(df: pd.DataFrame) -> pd.DataFrame:
    """Attach user/item persona ratios and match signals for ranking."""

    if all(column in df.columns for column in PERSONA_FEATURE_COLUMNS):
        return _fill_persona_defaults(df)

    enriched = df.copy()
    if "customer_id" not in enriched.columns or "article_id" not in enriched.columns:
        LOGGER.warning("Persona enrichment requires customer_id and article_id. Using default persona features.")
        for column in PERSONA_FEATURE_COLUMNS:
            enriched[column] = "UNKNOWN" if column.endswith("top_persona") else 0.0
        return _fill_persona_defaults(enriched)

    enriched["customer_id"] = enriched["customer_id"].astype(str).str.strip()
    enriched["article_id"] = enriched["article_id"].astype(str).str.strip().str.zfill(10)
    user_personas = _load_persona_frame(_resolve_user_persona_path(), "customer_id", "user")
    item_personas = _load_persona_frame(_resolve_item_persona_path(), "article_id", "item")

    for column in PERSONA_FEATURE_COLUMNS:
        if column in enriched.columns:
            enriched = enriched.drop(columns=column)

    enriched = enriched.merge(user_personas, on="customer_id", how="left")
    enriched = enriched.merge(item_personas, on="article_id", how="left")
    enriched = _fill_persona_defaults(enriched)
    enriched["top_persona_match"] = (
        enriched["user_top_persona"].ne("UNKNOWN")
        & enriched["user_top_persona"].eq(enriched["item_top_persona"])
    ).astype(int)
    enriched["persona_match_score"] = sum(
        enriched[f"user_{persona}_ratio"] * enriched[f"item_{persona}_ratio"]
        for persona in PERSONAS
    )
    return enriched


def build_training_matrices(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    feature_columns = [column for column in df.columns if column not in LEAKAGE_COLUMNS]
    if not feature_columns:
        raise ValueError("No usable feature columns found.")
    return df.loc[:, feature_columns].copy(), df[TARGET_COLUMN].copy(), feature_columns


def infer_feature_types(features: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric_columns = features.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_columns = [column for column in features.columns if column not in numeric_columns]
    return numeric_columns, categorical_columns


def cast_numeric_features_to_float(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.astype("float64")


def build_preprocessor(numeric_columns: list[str], categorical_columns: list[str]) -> ColumnTransformer:
    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric_columns:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("to_float", FunctionTransformer(cast_numeric_features_to_float, validate=False)),
                        ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
                    ]
                ),
                numeric_columns,
            )
        )
    if categorical_columns:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="constant", fill_value="UNKNOWN")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_columns,
            )
        )
    if not transformers:
        raise ValueError("Preprocessor requires at least one feature column.")
    return ColumnTransformer(transformers=transformers, remainder="drop")


def build_model_pipeline(numeric_columns: list[str], categorical_columns: list[str]) -> Pipeline:
    return build_model_pipeline_with_max_iter(
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        max_iter=DEFAULT_LOGREG_MAX_ITER,
    )


def build_model_pipeline_with_max_iter(
    numeric_columns: list[str],
    categorical_columns: list[str],
    max_iter: int,
) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(numeric_columns, categorical_columns)),
            ("classifier", LogisticRegression(max_iter=max_iter, solver="lbfgs")),
        ]
    )


def split_train_validation(
    data: pd.DataFrame,
    target: pd.Series,
    validation_size: float,
    random_state: int,
    split_mode: str = "row",
) -> tuple[pd.Index, pd.Index]:
    if split_mode == "user":
        if "customer_id" not in data.columns:
            raise ValueError("User-level split requires a customer_id column in the training data.")
        unique_users = np.asarray(sorted(data["customer_id"].astype(str).unique().tolist()))
        if unique_users.size < 2:
            raise ValueError("User-level split requires at least two distinct users.")
        train_users, valid_users = train_test_split(
            unique_users,
            test_size=validation_size,
            random_state=random_state,
        )
        train_mask = data["customer_id"].astype(str).isin(set(train_users))
        valid_mask = data["customer_id"].astype(str).isin(set(valid_users))
        train_index = data.index[train_mask]
        valid_index = data.index[valid_mask]
        if train_index.empty or valid_index.empty:
            raise ValueError("User-level split produced an empty train or validation set.")
        return train_index, valid_index

    if split_mode != "row":
        raise ValueError(f"Unsupported split_mode: {split_mode}")

    stratify_target: pd.Series | None = None
    counts = target.value_counts(dropna=False)
    if len(counts) > 1 and counts.min() >= 2:
        stratify_target = target
    train_index, valid_index = train_test_split(
        data.index,
        test_size=validation_size,
        random_state=random_state,
        stratify=stratify_target,
    )
    return pd.Index(train_index), pd.Index(valid_index)


def compute_validation_auc(model: Pipeline, x_valid: pd.DataFrame, y_valid: pd.Series) -> float | None:
    if y_valid.nunique(dropna=False) < 2:
        return None
    probabilities = model.predict_proba(x_valid)[:, 1]
    return float(roc_auc_score(y_valid, probabilities))


def save_artifacts(model: Pipeline, metadata: TrainingArtifacts, output_dir: Path) -> dict[str, Path]:
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    pipeline_path = output_dir / PIPELINE_ARTIFACT_NAME
    metadata_path = output_dir / METADATA_ARTIFACT_NAME
    joblib.dump(model, pipeline_path)
    metadata_path.write_text(json.dumps(asdict(metadata), indent=2), encoding="utf-8")
    return {"pipeline_path": pipeline_path, "metadata_path": metadata_path}


def train_ranker(
    csv_path: Path,
    output_dir: Path = DEFAULT_CHECKPOINT_DIR,
    validation_size: float = DEFAULT_VALIDATION_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
    split_mode: str = "row",
    max_iter: int = DEFAULT_LOGREG_MAX_ITER,
) -> dict[str, Any]:
    df = load_training_data(csv_path)
    features, target, feature_columns = build_training_matrices(df)
    if target.nunique(dropna=False) < 2:
        raise ValueError("Ranking training requires at least two target classes.")

    numeric_columns, categorical_columns = infer_feature_types(features)
    train_index, valid_index = split_train_validation(
        df,
        target,
        validation_size,
        random_state,
        split_mode=split_mode,
    )
    x_train = features.loc[train_index].copy()
    x_valid = features.loc[valid_index].copy()
    y_train = target.loc[train_index].copy()
    y_valid = target.loc[valid_index].copy()
    model = build_model_pipeline_with_max_iter(
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        max_iter=max_iter,
    )
    model.fit(x_train, y_train)
    validation_auc = compute_validation_auc(model, x_valid, y_valid)

    metadata = TrainingArtifacts(
        target_column=TARGET_COLUMN,
        identifier_columns=list(IDENTIFIER_COLUMNS),
        feature_columns=feature_columns,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        model_artifact=PIPELINE_ARTIFACT_NAME,
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        validation_size=validation_size,
        random_state=random_state,
        split_mode=split_mode,
    )
    artifact_paths = save_artifacts(model, metadata, output_dir)
    return {
        "model_type": "logreg",
        "training_rows": len(x_train),
        "validation_rows": len(x_valid),
        "validation_auc": validation_auc,
        "max_iter": max_iter,
        "artifacts": {key: str(path) for key, path in artifact_paths.items()},
        "metadata": asdict(metadata),
    }


def _normalize_categorical_value(value: Any) -> str:
    if value is None:
        return "UNKNOWN"
    text = str(value).strip()
    return text if text else "UNKNOWN"


def _build_categorical_vocabularies(features: pd.DataFrame, categorical_columns: list[str]) -> dict[str, dict[str, int]]:
    vocabularies: dict[str, dict[str, int]] = {}
    for column in categorical_columns:
        tokens = ["UNKNOWN"]
        seen = {"UNKNOWN"}
        for value in features[column].tolist():
            token = _normalize_categorical_value(value)
            if token in seen:
                continue
            seen.add(token)
            tokens.append(token)
        vocabularies[column] = {token: index for index, token in enumerate(tokens)}
    return vocabularies


def _encode_categorical_frame(
    features: pd.DataFrame,
    categorical_columns: list[str],
    vocabularies: dict[str, dict[str, int]],
) -> np.ndarray:
    if not categorical_columns:
        return np.zeros((len(features), 0), dtype=np.int64)
    encoded: list[np.ndarray] = []
    for column in categorical_columns:
        vocab = vocabularies[column]
        encoded.append(np.asarray([vocab.get(_normalize_categorical_value(v), 0) for v in features[column].tolist()], dtype=np.int64))
    return np.stack(encoded, axis=1)


def _fit_numeric_preprocessor(
    features: pd.DataFrame,
    numeric_columns: list[str],
) -> tuple[np.ndarray, dict[str, float], dict[str, float], dict[str, float]]:
    if not numeric_columns:
        return np.zeros((len(features), 0), dtype=np.float32), {}, {}, {}
    numeric_frame = features.loc[:, numeric_columns].apply(pd.to_numeric, errors="coerce")
    fill_values = {
        column: float(numeric_frame[column].median()) if numeric_frame[column].notna().any() else 0.0
        for column in numeric_columns
    }
    filled = numeric_frame.fillna(fill_values)
    means = {column: float(filled[column].mean()) for column in numeric_columns}
    stds = {
        column: float(filled[column].std()) if float(filled[column].std()) > 1e-8 else 1.0
        for column in numeric_columns
    }
    normalized = filled.copy()
    for column in numeric_columns:
        normalized[column] = (normalized[column] - means[column]) / stds[column]
    return normalized.to_numpy(dtype=np.float32), fill_values, means, stds


def _apply_numeric_preprocessor(
    features: pd.DataFrame,
    numeric_columns: list[str],
    fill_values: dict[str, float],
    means: dict[str, float],
    stds: dict[str, float],
) -> np.ndarray:
    if not numeric_columns:
        return np.zeros((len(features), 0), dtype=np.float32)
    numeric_frame = features.loc[:, numeric_columns].apply(pd.to_numeric, errors="coerce").fillna(fill_values)
    normalized = numeric_frame.copy()
    for column in numeric_columns:
        normalized[column] = (normalized[column] - means[column]) / stds[column]
    return normalized.to_numpy(dtype=np.float32)


def _deepfm_dataloader(
    categorical_x: np.ndarray,
    numeric_x: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    _require_torch()
    dataset = TensorDataset(
        torch.as_tensor(categorical_x, dtype=torch.long),
        torch.as_tensor(numeric_x, dtype=torch.float32),
        torch.as_tensor(labels, dtype=torch.float32),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def save_deepfm_artifacts(model: DeepFMRanker, metadata: dict[str, Any], output_dir: Path) -> dict[str, Path]:
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / DEEPFM_ARTIFACT_NAME
    metadata_path = output_dir / DEEPFM_METADATA_ARTIFACT_NAME
    torch.save({"model_state_dict": model.state_dict(), "metadata": metadata}, model_path)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {"model_path": model_path, "metadata_path": metadata_path}


def train_deepfm_ranker(
    csv_path: Path,
    output_dir: Path = DEFAULT_CHECKPOINT_DIR,
    validation_size: float = DEFAULT_VALIDATION_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
    batch_size: int = DEFAULT_DEEPFM_BATCH_SIZE,
    epochs: int = DEFAULT_DEEPFM_EPOCHS,
    learning_rate: float = DEFAULT_DEEPFM_LEARNING_RATE,
    weight_decay: float = DEFAULT_DEEPFM_WEIGHT_DECAY,
    embedding_dim: int = 16,
    device: str | None = None,
    split_mode: str = "row",
) -> dict[str, Any]:
    _require_torch()
    df = load_training_data(csv_path)
    features, target, feature_columns = build_training_matrices(df)
    numeric_columns, categorical_columns = infer_feature_types(features)
    train_index, valid_index = split_train_validation(
        df,
        target,
        validation_size,
        random_state,
        split_mode=split_mode,
    )
    x_train = features.loc[train_index].copy()
    x_valid = features.loc[valid_index].copy()
    y_train = target.loc[train_index].copy()
    y_valid = target.loc[valid_index].copy()

    vocabularies = _build_categorical_vocabularies(x_train, categorical_columns)
    x_train_cat = _encode_categorical_frame(x_train, categorical_columns, vocabularies)
    x_valid_cat = _encode_categorical_frame(x_valid, categorical_columns, vocabularies)
    x_train_num, fill_values, means, stds = _fit_numeric_preprocessor(x_train, numeric_columns)
    x_valid_num = _apply_numeric_preprocessor(x_valid, numeric_columns, fill_values, means, stds)

    config = DeepFMConfig(
        categorical_cardinalities=[len(vocabularies[column]) for column in categorical_columns],
        numeric_dim=len(numeric_columns),
        embedding_dim=embedding_dim,
    )
    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepFMRanker(config).to(resolved_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    train_loader = _deepfm_dataloader(
        x_train_cat,
        x_train_num,
        y_train.to_numpy(dtype=np.float32),
        batch_size=batch_size,
        shuffle=True,
    )

    valid_cat_tensor = torch.as_tensor(x_valid_cat, dtype=torch.long, device=resolved_device)
    valid_num_tensor = torch.as_tensor(x_valid_num, dtype=torch.float32, device=resolved_device)

    best_auc = float("-inf")
    best_epoch = 0
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        total_rows = 0
        for batch_cat, batch_num, batch_y in train_loader:
            batch_cat = batch_cat.to(resolved_device)
            batch_num = batch_num.to(resolved_device)
            batch_y = batch_y.to(resolved_device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_cat, batch_num)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            actual_batch = batch_cat.shape[0]
            running_loss += float(loss.item()) * actual_batch
            total_rows += actual_batch

        model.eval()
        with torch.no_grad():
            valid_logits = model(valid_cat_tensor, valid_num_tensor)
            valid_probs = torch.sigmoid(valid_logits).detach().cpu().numpy()
        validation_auc = float(roc_auc_score(y_valid, valid_probs)) if y_valid.nunique(dropna=False) > 1 else 0.0
        train_loss = running_loss / max(total_rows, 1)
        history.append({"epoch": float(epoch), "train_loss": train_loss, "validation_auc": validation_auc})
        LOGGER.info(
            "DeepFM epoch %s/%s | train_loss=%.6f | validation_auc=%.6f",
            epoch,
            epochs,
            train_loss,
            validation_auc,
        )
        if validation_auc > best_auc:
            best_auc = validation_auc
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    model.load_state_dict(best_state)
    metadata = {
        "model_type": "deepfm",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_column": TARGET_COLUMN,
        "identifier_columns": list(IDENTIFIER_COLUMNS),
        "feature_columns": feature_columns,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "categorical_vocabularies": vocabularies,
        "numeric_fill_values": fill_values,
        "numeric_means": means,
        "numeric_stds": stds,
        "best_validation_auc": best_auc,
        "best_epoch": best_epoch,
        "deepfm_config": asdict(config),
        "training_config": {
            "batch_size": batch_size,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "validation_size": validation_size,
            "random_state": random_state,
            "device": resolved_device,
            "split_mode": split_mode,
        },
    }
    artifact_paths = save_deepfm_artifacts(model, metadata, output_dir)
    return {
        "model_type": "deepfm",
        "training_rows": len(x_train),
        "validation_rows": len(x_valid),
        "validation_auc": best_auc,
        "best_epoch": best_epoch,
        "artifacts": {key: str(path) for key, path in artifact_paths.items()},
        "metadata": metadata,
        "history": history,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a ranking model from processed tabular data.")
    parser.add_argument("--data", type=Path, default=DEFAULT_TRAINING_DATA_PATH, help="Path to the ranking training CSV.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR, help="Directory for ranking artifacts.")
    parser.add_argument("--model-type", choices=("logreg", "deepfm"), default="logreg", help="Ranking model to train.")
    parser.add_argument("--validation-size", type=float, default=DEFAULT_VALIDATION_SIZE, help="Validation split ratio.")
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE, help="Random seed for train/validation splitting.")
    parser.add_argument("--max-iter", type=int, default=DEFAULT_LOGREG_MAX_ITER, help="Maximum LBFGS iterations for logistic ranking.")
    parser.add_argument(
        "--split-mode",
        choices=("row", "user"),
        default="row",
        help="Validation split strategy. Use 'user' to avoid customer-level leakage.",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_DEEPFM_BATCH_SIZE, help="DeepFM batch size.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_DEEPFM_EPOCHS, help="DeepFM training epochs.")
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_DEEPFM_LEARNING_RATE, help="DeepFM learning rate.")
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_DEEPFM_WEIGHT_DECAY, help="DeepFM weight decay.")
    parser.add_argument("--embedding-dim", type=int, default=16, help="DeepFM embedding dimension.")
    parser.add_argument("--device", type=str, help="DeepFM device override, e.g. cpu or cuda.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(verbose=args.verbose)
    if args.model_type == "logreg":
        summary = train_ranker(
            csv_path=args.data,
            output_dir=args.output_dir,
            validation_size=args.validation_size,
            random_state=args.random_state,
            split_mode=args.split_mode,
            max_iter=args.max_iter,
        )
    else:
        summary = train_deepfm_ranker(
            csv_path=args.data,
            output_dir=args.output_dir,
            validation_size=args.validation_size,
            random_state=args.random_state,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            embedding_dim=args.embedding_dim,
            device=args.device,
            split_mode=args.split_mode,
        )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
