"""Ranking inference utilities for logistic baseline and DeepFM."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

try:
    import torch
except ImportError:  # pragma: no cover - deepfm inference requires torch
    torch = None

try:
    from rec_models.ranking.model import DeepFMRanker, deepfm_config_from_metadata
    from rec_models.ranking.train import enrich_with_persona_features
except ImportError:  # pragma: no cover - supports running from rec_models/ as cwd
    from ranking.model import DeepFMRanker, deepfm_config_from_metadata  # type: ignore[no-redef]
    from ranking.train import enrich_with_persona_features  # type: ignore[no-redef]


LOGGER = logging.getLogger(__name__)

DEFAULT_CHECKPOINT_BASE_DIR = Path(__file__).resolve().parents[1] / "checkpoints"
PIPELINE_ARTIFACT_NAME = "ranking_baseline.joblib"
METADATA_ARTIFACT_NAME = "ranking_baseline_metadata.json"
DEEPFM_ARTIFACT_NAME = "ranking_deepfm.pt"
DEEPFM_METADATA_ARTIFACT_NAME = "ranking_deepfm_metadata.json"
DEFAULT_LOGREG_DEV_CHECKPOINT_DIR = DEFAULT_CHECKPOINT_BASE_DIR / "logreg_dev"


def _checkpoint_artifacts_exist(checkpoint_dir: Path) -> bool:
    return (
        (checkpoint_dir / PIPELINE_ARTIFACT_NAME).exists()
        and (checkpoint_dir / METADATA_ARTIFACT_NAME).exists()
    )


def _resolve_default_checkpoint_dir() -> Path:
    configured_path = os.getenv("RANKING_CHECKPOINT_DIR")
    if configured_path:
        return Path(configured_path)
    if _checkpoint_artifacts_exist(DEFAULT_LOGREG_DEV_CHECKPOINT_DIR):
        return DEFAULT_LOGREG_DEV_CHECKPOINT_DIR
    return DEFAULT_CHECKPOINT_BASE_DIR


DEFAULT_CHECKPOINT_DIR = _resolve_default_checkpoint_dir()


def configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


def load_artifacts(checkpoint_dir: Path) -> tuple[Pipeline, dict[str, Any]]:
    checkpoint_dir = checkpoint_dir.expanduser().resolve()
    pipeline_path = checkpoint_dir / PIPELINE_ARTIFACT_NAME
    metadata_path = checkpoint_dir / METADATA_ARTIFACT_NAME
    if not pipeline_path.exists():
        raise FileNotFoundError(f"Ranking pipeline artifact not found: {pipeline_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Ranking metadata artifact not found: {metadata_path}")
    model = joblib.load(pipeline_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return model, metadata


def load_deepfm_artifacts(checkpoint_dir: Path) -> tuple[DeepFMRanker, dict[str, Any]]:
    if torch is None:
        raise ImportError("torch is required to load DeepFM artifacts.")
    checkpoint_dir = checkpoint_dir.expanduser().resolve()
    model_path = checkpoint_dir / DEEPFM_ARTIFACT_NAME
    metadata_path = checkpoint_dir / DEEPFM_METADATA_ARTIFACT_NAME
    if not model_path.exists():
        raise FileNotFoundError(f"DeepFM artifact not found: {model_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"DeepFM metadata artifact not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    model = DeepFMRanker(deepfm_config_from_metadata(metadata))
    payload = torch.load(model_path, map_location="cpu")
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, metadata


def prepare_inference_features(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    aligned = df.copy()
    missing_columns = [column for column in feature_columns if column not in aligned.columns]
    for column in missing_columns:
        aligned[column] = np.nan
    return aligned.loc[:, feature_columns]


def _extract_scores(model: Pipeline, features: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)
        if probabilities.ndim == 2 and probabilities.shape[1] >= 2:
            return probabilities[:, 1]
        return probabilities.ravel()

    if hasattr(model, "decision_function"):
        raw_scores = np.asarray(model.decision_function(features), dtype=float).ravel()
        if raw_scores.size == 0:
            return np.asarray([], dtype=float)
        score_min = float(raw_scores.min())
        score_max = float(raw_scores.max())
        if score_max > score_min:
            return (raw_scores - score_min) / (score_max - score_min)
        return np.zeros_like(raw_scores, dtype=float)

    return np.asarray(model.predict(features), dtype=float).ravel()


def _extract_deepfm_scores(model: DeepFMRanker, features: pd.DataFrame, metadata: dict[str, Any]) -> np.ndarray:
    if torch is None:
        raise ImportError("torch is required for DeepFM inference.")

    categorical_columns = metadata.get("categorical_columns", [])
    numeric_columns = metadata.get("numeric_columns", [])
    vocabularies = metadata.get("categorical_vocabularies", {})
    fill_values = metadata.get("numeric_fill_values", {})
    means = metadata.get("numeric_means", {})
    stds = metadata.get("numeric_stds", {})

    encoded_categorical: list[np.ndarray] = []
    for column in categorical_columns:
        vocab = vocabularies[column]
        encoded_categorical.append(
            np.asarray(
                [vocab.get((str(value).strip() if value is not None and str(value).strip() else "UNKNOWN"), 0) for value in features[column].fillna("UNKNOWN").tolist()],
                dtype=np.int64,
            )
        )
    categorical_x = np.stack(encoded_categorical, axis=1) if encoded_categorical else np.zeros((len(features), 0), dtype=np.int64)

    if numeric_columns:
        numeric_frame = features.loc[:, numeric_columns].apply(pd.to_numeric, errors="coerce").fillna(fill_values)
        for column in numeric_columns:
            numeric_frame[column] = (numeric_frame[column] - means[column]) / stds[column]
        numeric_x = numeric_frame.to_numpy(dtype=np.float32)
    else:
        numeric_x = np.zeros((len(features), 0), dtype=np.float32)

    with torch.no_grad():
        logits = model(
            torch.as_tensor(categorical_x, dtype=torch.long),
            torch.as_tensor(numeric_x, dtype=torch.float32),
        )
        return torch.sigmoid(logits).detach().cpu().numpy()


def score_candidates(
    candidates: pd.DataFrame,
    checkpoint_dir: Path = DEFAULT_CHECKPOINT_DIR,
    model_type: str = "logreg",
) -> pd.DataFrame:
    enriched_candidates = enrich_with_persona_features(candidates)
    if model_type == "deepfm":
        model, metadata = load_deepfm_artifacts(checkpoint_dir)
        feature_columns = metadata.get("feature_columns", [])
        identifier_columns = metadata.get("identifier_columns", [])
        features = prepare_inference_features(enriched_candidates, feature_columns)
        scores = _extract_deepfm_scores(model, features, metadata)
    else:
        model, metadata = load_artifacts(checkpoint_dir)
        feature_columns = metadata.get("feature_columns", [])
        identifier_columns = metadata.get("identifier_columns", [])
        features = prepare_inference_features(enriched_candidates, feature_columns)
        scores = _extract_scores(model, features)

    preserved_columns = [column for column in identifier_columns if column in enriched_candidates.columns]
    result = enriched_candidates.loc[:, preserved_columns].copy() if preserved_columns else pd.DataFrame(index=enriched_candidates.index)
    result["score"] = scores
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score ranking candidates with a saved ranking model.")
    parser.add_argument("--input", type=Path, required=True, help="Path to a candidate CSV file.")
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR, help="Checkpoint directory containing ranking artifacts.")
    parser.add_argument("--model-type", choices=("logreg", "deepfm"), default="logreg", help="Ranking model to use for inference.")
    parser.add_argument("--output", type=Path, help="Optional output CSV path.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(verbose=args.verbose)
    candidate_path = args.input.expanduser().resolve()
    if not candidate_path.exists():
        raise FileNotFoundError(f"Candidate CSV not found: {candidate_path}")
    candidates = pd.read_csv(candidate_path)
    scored = score_candidates(candidates=candidates, checkpoint_dir=args.checkpoint_dir, model_type=args.model_type)
    if args.output is not None:
        output_path = args.output.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        scored.to_csv(output_path, index=False)
    else:
        print(scored.head().to_string(index=False))


if __name__ == "__main__":
    main()
