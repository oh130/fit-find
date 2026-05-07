"""Training entry point for the Two-Tower candidate retrieval model."""

from __future__ import annotations

import argparse
import json
import logging
import copy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    from torch import Tensor, nn
    from torch.nn import functional as F
    from torch.utils.data import DataLoader
except ImportError:  # pragma: no cover - training requires torch
    torch = None
    Tensor = Any  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    DataLoader = Any  # type: ignore[assignment]

try:
    from rec_models.candidate.dataset import (
        DEFAULT_DATA_PATH,
        DatasetArtifacts,
        TwoTowerPairDataset,
        build_two_tower_datasets,
        collate_two_tower_batch,
    )
    from rec_models.candidate.model import TwoTowerConfig, TwoTowerModel, build_two_tower_config_from_metadata
    from rec_models.common.metrics import mean_metric, recall_at_k
except ImportError:  # pragma: no cover - supports running from rec_models/ as cwd
    from candidate.dataset import (  # type: ignore[no-redef]
        DEFAULT_DATA_PATH,
        DatasetArtifacts,
        TwoTowerPairDataset,
        build_two_tower_datasets,
        collate_two_tower_batch,
    )
    from candidate.model import TwoTowerConfig, TwoTowerModel, build_two_tower_config_from_metadata  # type: ignore[no-redef]
    from common.metrics import mean_metric, recall_at_k  # type: ignore[no-redef]


LOGGER = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 256
DEFAULT_EPOCHS = 5
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-5
DEFAULT_VALIDATION_K = 300
DEFAULT_NEGATIVES_PER_POSITIVE = 0
DEFAULT_HARD_NEGATIVE_RATIO = 0.5
DEFAULT_SAMPLED_NEGATIVE_WEIGHT = 1.0
DEFAULT_CHECKPOINT_DIR = Path(__file__).resolve().parents[2] / "data" / "checkpoints" / "candidate"
DEFAULT_MODEL_ARTIFACT = "two_tower.pt"
DEFAULT_METADATA_ARTIFACT = "two_tower_metadata.json"


def _require_torch() -> None:
    if torch is None or nn is None or F is None:
        raise ImportError("torch is required to train the Two-Tower model.")


@dataclass(slots=True)
class TrainingConfig:
    """Hyperparameters and runtime settings for retrieval training."""

    data_path: str
    batch_size: int = DEFAULT_BATCH_SIZE
    epochs: int = DEFAULT_EPOCHS
    learning_rate: float = DEFAULT_LEARNING_RATE
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    validation_ratio: float = 0.2
    split_mode: str = "leave_last_out"
    validation_k: int = DEFAULT_VALIDATION_K
    validation_max_users: int | None = 5000
    negatives_per_positive: int = DEFAULT_NEGATIVES_PER_POSITIVE
    hard_negative_ratio: float = DEFAULT_HARD_NEGATIVE_RATIO
    sampled_negative_weight: float = DEFAULT_SAMPLED_NEGATIVE_WEIGHT
    device: str = "cpu"
    checkpoint_dir: str = str(DEFAULT_CHECKPOINT_DIR)
    model_artifact: str = DEFAULT_MODEL_ARTIFACT
    metadata_artifact: str = DEFAULT_METADATA_ARTIFACT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Two-Tower candidate retrieval model.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH, help="Processed interaction CSV for retrieval training.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Mini-batch size.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Number of training epochs.")
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE, help="Adam learning rate.")
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY, help="AdamW weight decay.")
    parser.add_argument("--validation-ratio", type=float, default=0.2, help="User-level validation holdout ratio.")
    parser.add_argument(
        "--split-mode",
        choices=("leave_last_out", "user"),
        default="leave_last_out",
        help="Validation split strategy. leave_last_out holds out each user's final interaction.",
    )
    parser.add_argument("--validation-k", type=int, default=DEFAULT_VALIDATION_K, help="Recall@K cutoff for validation.")
    parser.add_argument(
        "--validation-max-users",
        type=int,
        default=5000,
        help="Cap validation users during training for faster feedback. Use 0 to evaluate every validation user.",
    )
    parser.add_argument(
        "--negatives-per-positive",
        type=int,
        default=DEFAULT_NEGATIVES_PER_POSITIVE,
        help="Number of explicit negatives to sample per positive pair.",
    )
    parser.add_argument(
        "--hard-negative-ratio",
        type=float,
        default=DEFAULT_HARD_NEGATIVE_RATIO,
        help="Fraction of explicit negatives sampled from the positive item's main category.",
    )
    parser.add_argument(
        "--sampled-negative-weight",
        type=float,
        default=DEFAULT_SAMPLED_NEGATIVE_WEIGHT,
        help="Loss weight applied to the sampled-negative term.",
    )
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR, help="Directory for model artifacts.")
    parser.add_argument("--device", type=str, help="Training device override, e.g. cpu or cuda.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


def configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


def resolve_device(device_override: str | None = None) -> str:
    _require_torch()
    if device_override:
        return device_override
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_dataloader(dataset: TwoTowerPairDataset, batch_size: int, shuffle: bool) -> DataLoader:
    _require_torch()
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda rows: collate_two_tower_batch(rows, as_torch=True),
    )


def batch_to_device(batch: dict[str, Any], device: str) -> dict[str, Any]:
    moved = dict(batch)
    for key in ("user_categorical", "user_numeric", "history_item_ids", "history_mask", "item_categorical", "item_numeric", "item_id_index"):
        moved[key] = moved[key].to(device)
    return moved


def retrieval_loss(logits: Tensor) -> Tensor:
    """In-batch softmax loss with diagonal positives."""

    targets = torch.arange(logits.shape[0], device=logits.device)
    return F.cross_entropy(logits, targets)


def build_sampled_negative_batch(
    dataset: TwoTowerPairDataset,
    customer_ids: list[str],
    positive_item_ids: list[str],
    negatives_per_positive: int,
    hard_negative_ratio: float,
) -> dict[str, Tensor] | None:
    """Sample and encode explicit negatives aligned to a positive batch."""

    if negatives_per_positive <= 0:
        return None

    sampled_item_ids: list[str] = []
    for row_index, (customer_id, positive_item_id) in enumerate(zip(customer_ids, positive_item_ids, strict=False)):
        rng = np.random.default_rng(seed=(hash((customer_id, positive_item_id, row_index)) & 0xFFFFFFFF))
        sampled = dataset.sample_negative_item_ids(
            customer_id=customer_id,
            sample_size=negatives_per_positive,
            positive_item_id=positive_item_id,
            hard_negative_ratio=hard_negative_ratio,
            rng=rng,
        )
        if len(sampled) < negatives_per_positive:
            return None
        sampled_item_ids.extend(sampled)

    encoded = dataset.encode_item_id_batch(sampled_item_ids)
    return {
        "item_categorical": torch.as_tensor(encoded["categorical"], dtype=torch.long),
        "item_numeric": torch.as_tensor(encoded["numeric"], dtype=torch.float32),
    }


def sampled_negative_loss(
    model: TwoTowerModel,
    user_embedding: Tensor,
    positive_scores: Tensor,
    negative_batch: dict[str, Tensor] | None,
    negatives_per_positive: int,
) -> Tensor | None:
    """Compute an explicit sampled-negative loss term."""

    if negative_batch is None or negatives_per_positive <= 0:
        return None

    negative_item_embedding = model.encode_item(
        item_categorical=negative_batch["item_categorical"],
        item_numeric=negative_batch["item_numeric"],
        item_id_index=negative_batch.get("item_id_index"),
    )
    batch_size = user_embedding.shape[0]
    negative_item_embedding = negative_item_embedding.view(batch_size, negatives_per_positive, -1)
    negative_scores = torch.sum(user_embedding.unsqueeze(1) * negative_item_embedding, dim=-1)
    logits = torch.cat([positive_scores.unsqueeze(1), negative_scores], dim=1)
    targets = torch.zeros(batch_size, dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, targets)


def train_one_epoch(
    model: TwoTowerModel,
    dataset: TwoTowerPairDataset,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    negatives_per_positive: int = DEFAULT_NEGATIVES_PER_POSITIVE,
    hard_negative_ratio: float = DEFAULT_HARD_NEGATIVE_RATIO,
    sampled_negative_weight: float = DEFAULT_SAMPLED_NEGATIVE_WEIGHT,
) -> float:
    """Run one training epoch and return average loss."""

    model.train()
    total_loss = 0.0
    total_examples = 0

    for batch in dataloader:
        prepared = batch_to_device(batch, device=device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(
            user_categorical=prepared["user_categorical"],
            user_numeric=prepared["user_numeric"],
            history_item_ids=prepared["history_item_ids"],
            history_mask=prepared["history_mask"],
            item_categorical=prepared["item_categorical"],
            item_numeric=prepared["item_numeric"],
            item_id_index=prepared["item_id_index"],
        )
        loss = retrieval_loss(outputs["logits"])
        if negatives_per_positive > 0:
            negative_batch = build_sampled_negative_batch(
                dataset=dataset,
                customer_ids=list(batch["customer_id"]),
                positive_item_ids=list(batch["article_id"]),
                negatives_per_positive=negatives_per_positive,
                hard_negative_ratio=hard_negative_ratio,
            )
            if negative_batch is not None:
                negative_batch = {key: value.to(device) for key, value in negative_batch.items()}
                extra_loss = sampled_negative_loss(
                    model=model,
                    user_embedding=outputs["user_embedding"],
                    positive_scores=outputs["positive_scores"],
                    negative_batch=negative_batch,
                    negatives_per_positive=negatives_per_positive,
                )
                if extra_loss is not None:
                    loss = loss + (sampled_negative_weight * extra_loss)
        loss.backward()
        optimizer.step()

        batch_size = prepared["user_categorical"].shape[0]
        total_loss += float(loss.item()) * batch_size
        total_examples += batch_size

    if total_examples == 0:
        return 0.0
    return total_loss / total_examples


def _stack_encoded_rows(rows: list[dict[str, np.ndarray]], key: str) -> np.ndarray:
    return np.stack([row[key] for row in rows]) if rows else np.empty((0, 0), dtype=np.float32)


def build_item_embedding_index(
    model: TwoTowerModel,
    dataset_artifacts: DatasetArtifacts,
    device: str,
) -> tuple[np.ndarray, list[str]]:
    """Encode all known items once for validation retrieval."""

    model.eval()
    merged_records = dict(dataset_artifacts.train_dataset.item_records)
    merged_records.update(dataset_artifacts.validation_dataset.item_records)

    item_ids = list(merged_records.keys())
    encoded_rows = [dataset_artifacts.encoder.encode_item_row(merged_records[item_id]) for item_id in item_ids]
    if not encoded_rows:
        return np.empty((0, dataset_artifacts.metadata["encoder"]["schema"].get("item_numeric_columns", 0))), []

    item_categorical = torch.as_tensor(_stack_encoded_rows(encoded_rows, "categorical"), dtype=torch.long, device=device)
    item_numeric_np = _stack_encoded_rows(encoded_rows, "numeric").astype(np.float32)
    item_numeric = torch.as_tensor(item_numeric_np, dtype=torch.float32, device=device)
    item_id_index = torch.as_tensor(
        np.asarray([row["item_id_index"] for row in encoded_rows], dtype=np.int64),
        dtype=torch.long,
        device=device,
    )

    with torch.no_grad():
        embeddings = model.encode_item(
            item_categorical=item_categorical,
            item_numeric=item_numeric,
            item_id_index=item_id_index,
        )
    return embeddings.detach().cpu().numpy().astype(np.float32), item_ids


def evaluate_recall(
    model: TwoTowerModel,
    dataset_artifacts: DatasetArtifacts,
    device: str,
    k: int,
    max_users: int | None = None,
) -> float:
    """Compute validation Recall@K against the known item universe."""

    validation_dataset = dataset_artifacts.validation_dataset
    if len(validation_dataset) == 0:
        return 0.0

    item_embeddings, item_ids = build_item_embedding_index(model, dataset_artifacts, device=device)
    if len(item_ids) == 0:
        return 0.0

    item_matrix = torch.as_tensor(item_embeddings, dtype=torch.float32, device=device)
    ranked_lists: list[list[str]] = []
    relevant_lists: list[list[str]] = []

    model.eval()
    with torch.no_grad():
        validation_items = list(validation_dataset.user_to_positive_items.items())
        if max_users is not None and len(validation_items) > max_users:
            validation_items = validation_items[:max_users]
        for customer_id, positive_item_ids in validation_items:
            user_record = validation_dataset.user_records.get(customer_id)
            if user_record is None:
                continue

            encoded_user = dataset_artifacts.encoder.encode_user_row(user_record)
            encoded_history = dataset_artifacts.encoder.encode_history_item_ids(user_record.get("history_article_ids", ""))
            user_categorical = torch.as_tensor(
                encoded_user["categorical"][None, :],
                dtype=torch.long,
                device=device,
            )
            user_numeric = torch.as_tensor(
                encoded_user["numeric"][None, :],
                dtype=torch.float32,
                device=device,
            )
            history_item_ids = torch.as_tensor(
                encoded_history["ids"][None, :],
                dtype=torch.long,
                device=device,
            )
            history_mask = torch.as_tensor(
                encoded_history["mask"][None, :],
                dtype=torch.float32,
                device=device,
            )
            user_embedding = model.encode_user(
                user_categorical=user_categorical,
                user_numeric=user_numeric,
                history_item_ids=history_item_ids,
                history_mask=history_mask,
            )
            scores = torch.matmul(user_embedding, item_matrix.transpose(0, 1)).squeeze(0)
            top_k = min(k, scores.shape[0])
            top_indices = torch.topk(scores, k=top_k, dim=0).indices.detach().cpu().tolist()
            ranked_lists.append([item_ids[index] for index in top_indices])
            relevant_lists.append([str(item_id) for item_id in positive_item_ids])

    return mean_metric(ranked_lists, relevant_lists, recall_at_k, k)


def save_artifacts(
    model: TwoTowerModel,
    dataset_artifacts: DatasetArtifacts,
    training_config: TrainingConfig,
    best_validation_recall: float,
    best_epoch: int,
) -> dict[str, Path]:
    """Persist the trained Two-Tower weights and training metadata."""

    checkpoint_dir = Path(training_config.checkpoint_dir).expanduser().resolve()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model_path = checkpoint_dir / training_config.model_artifact
    metadata_path = checkpoint_dir / training_config.metadata_artifact

    payload = {
        "model_state_dict": model.state_dict(),
        "model_config": asdict(build_two_tower_config_from_metadata(dataset_artifacts.metadata)),
        "dataset_metadata": dataset_artifacts.metadata,
        "training_config": asdict(training_config),
    }
    torch.save(payload, model_path)

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "best_validation_recall": best_validation_recall,
        "best_epoch": best_epoch,
        "training_config": asdict(training_config),
        "dataset_metadata": dataset_artifacts.metadata,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {
        "model_path": model_path,
        "metadata_path": metadata_path,
    }


def train_two_tower(training_config: TrainingConfig) -> dict[str, Any]:
    """Train the retrieval model end-to-end and return summary metrics."""

    _require_torch()
    LOGGER.info("Building Two-Tower datasets from %s", training_config.data_path)
    dataset_artifacts = build_two_tower_datasets(
        csv_path=Path(training_config.data_path),
        validation_ratio=training_config.validation_ratio,
        split_mode=training_config.split_mode,
    )
    model_config: TwoTowerConfig = build_two_tower_config_from_metadata(dataset_artifacts.metadata)
    model = TwoTowerModel(model_config).to(training_config.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )

    train_loader = build_dataloader(
        dataset_artifacts.train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
    )

    best_validation_recall = float("-inf")
    best_epoch = 0
    best_model_state = copy.deepcopy(model.state_dict())
    history: list[dict[str, float]] = []

    for epoch in range(1, training_config.epochs + 1):
        train_loss = train_one_epoch(
            model,
            dataset=dataset_artifacts.train_dataset,
            dataloader=train_loader,
            optimizer=optimizer,
            device=training_config.device,
            negatives_per_positive=training_config.negatives_per_positive,
            hard_negative_ratio=training_config.hard_negative_ratio,
            sampled_negative_weight=training_config.sampled_negative_weight,
        )
        validation_recall = evaluate_recall(
            model,
            dataset_artifacts=dataset_artifacts,
            device=training_config.device,
            k=training_config.validation_k,
            max_users=training_config.validation_max_users,
        )
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                f"validation_recall@{training_config.validation_k}": validation_recall,
            }
        )
        LOGGER.info(
            "Epoch %s/%s | train_loss=%.6f | validation_recall@%s=%.6f",
            epoch,
            training_config.epochs,
            train_loss,
            training_config.validation_k,
            validation_recall,
        )
        if validation_recall > best_validation_recall:
            best_validation_recall = validation_recall
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_state)

    saved_paths = save_artifacts(
        model=model,
        dataset_artifacts=dataset_artifacts,
        training_config=training_config,
        best_validation_recall=best_validation_recall,
        best_epoch=best_epoch,
    )
    return {
        "best_validation_recall": best_validation_recall,
        "best_epoch": best_epoch,
        "history": history,
        "saved_paths": {key: str(path) for key, path in saved_paths.items()},
    }


def main() -> None:
    args = parse_args()
    configure_logging(verbose=args.verbose)
    device = resolve_device(args.device)

    training_config = TrainingConfig(
        data_path=str(args.data.expanduser().resolve()),
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        validation_ratio=args.validation_ratio,
        split_mode=args.split_mode,
        validation_k=args.validation_k,
        validation_max_users=None if args.validation_max_users == 0 else args.validation_max_users,
        negatives_per_positive=args.negatives_per_positive,
        hard_negative_ratio=args.hard_negative_ratio,
        sampled_negative_weight=args.sampled_negative_weight,
        device=device,
        checkpoint_dir=str(args.checkpoint_dir.expanduser().resolve()),
    )
    results = train_two_tower(training_config)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
