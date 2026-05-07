"""Experiment runner for Two-Tower candidate retrieval tuning.

This script automates repeated training/evaluation runs so candidate-model
experiments can be compared without hand-editing hyperparameters or manually
copying metrics into docs.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any

from rec_models.candidate.evaluator import (
    DEFAULT_TOP_K,
    evaluate_candidate_retrieval,
    evaluate_two_tower_retrieval,
)
from rec_models.candidate.train import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_DATA_PATH,
    DEFAULT_EPOCHS,
    DEFAULT_HARD_NEGATIVE_RATIO,
    DEFAULT_LEARNING_RATE,
    DEFAULT_NEGATIVES_PER_POSITIVE,
    DEFAULT_SAMPLED_NEGATIVE_WEIGHT,
    DEFAULT_WEIGHT_DECAY,
    TrainingConfig,
    configure_logging,
    resolve_device,
    train_two_tower,
)
from rec_models.evaluation.data_utils import build_evaluation_context, load_evaluation_data


LOGGER = logging.getLogger(__name__)

DEFAULT_REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports" / "candidate_experiments"


def _ensure_training_dependencies() -> None:
    """Fail fast with an actionable message when torch is unavailable."""

    if shutil.which("python") is None:
        return

    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Two-Tower experiments require 'torch', but it is not installed in the current Python environment. "
            "Install dependencies with `pip install -r rec_models/requirements.txt` inside your active venv, "
            "or run the command inside the `rec-models` Docker container."
        ) from exc


@dataclass(slots=True, frozen=True)
class ExperimentSpec:
    """One concrete hyperparameter combination to train and evaluate."""

    experiment_id: str
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    negatives_per_positive: int
    hard_negative_ratio: float
    sampled_negative_weight: float


def _parse_int_list(raw: str) -> list[int]:
    values = [value.strip() for value in raw.split(",") if value.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one integer value.")
    try:
        return [int(value) for value in values]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid integer list: {raw}") from exc


def _parse_float_list(raw: str) -> list[float]:
    values = [value.strip() for value in raw.split(",") if value.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one float value.")
    try:
        return [float(value) for value in values]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid float list: {raw}") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run and compare multiple Two-Tower candidate experiments.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH, help="Processed interaction/evaluation CSV path.")
    parser.add_argument("--epochs", type=_parse_int_list, default=[DEFAULT_EPOCHS], help="Comma-separated epoch values.")
    parser.add_argument(
        "--batch-sizes",
        type=_parse_int_list,
        default=[DEFAULT_BATCH_SIZE],
        help="Comma-separated batch-size values.",
    )
    parser.add_argument(
        "--learning-rates",
        type=_parse_float_list,
        default=[DEFAULT_LEARNING_RATE],
        help="Comma-separated learning-rate values.",
    )
    parser.add_argument(
        "--weight-decays",
        type=_parse_float_list,
        default=[DEFAULT_WEIGHT_DECAY],
        help="Comma-separated weight-decay values.",
    )
    parser.add_argument(
        "--negatives-per-positive",
        type=_parse_int_list,
        default=[DEFAULT_NEGATIVES_PER_POSITIVE],
        help="Comma-separated explicit negative counts sampled per positive pair.",
    )
    parser.add_argument(
        "--hard-negative-ratios",
        type=_parse_float_list,
        default=[DEFAULT_HARD_NEGATIVE_RATIO],
        help="Comma-separated hard-negative ratios based on main-category matches.",
    )
    parser.add_argument(
        "--sampled-negative-weights",
        type=_parse_float_list,
        default=[DEFAULT_SAMPLED_NEGATIVE_WEIGHT],
        help="Comma-separated loss weights for the sampled-negative term.",
    )
    parser.add_argument("--top_k", type=int, default=DEFAULT_TOP_K, help="Recall@K cutoff used during evaluation.")
    parser.add_argument("--max-users", type=int, help="Optional evaluation user cap for faster iteration.")
    parser.add_argument("--candidate-pool-size", type=int, help="Optional heuristic baseline pool size for comparison.")
    parser.add_argument("--validation-ratio", type=float, default=0.2, help="User-level validation holdout ratio.")
    parser.add_argument("--validation-k", type=int, default=DEFAULT_TOP_K, help="Validation Recall@K during training.")
    parser.add_argument("--device", type=str, help="Training/inference device override, e.g. cpu or cuda.")
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=DEFAULT_REPORTS_DIR,
        help="Directory where experiment reports and per-run checkpoints are saved.",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        help="Optional stable prefix for the report directory. Defaults to a UTC timestamp.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete an existing report directory when the experiment name already exists.",
    )
    parser.add_argument(
        "--max-experiments",
        type=int,
        help="Optional cap to truncate the generated hyperparameter grid for quick smoke checks.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print planned experiments without training.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


def _build_experiment_specs(args: argparse.Namespace) -> list[ExperimentSpec]:
    specs: list[ExperimentSpec] = []
    experiment_index = 1
    for epochs, batch_size, learning_rate, weight_decay, negatives_per_positive, hard_negative_ratio, sampled_negative_weight in product(
        args.epochs,
        args.batch_sizes,
        args.learning_rates,
        args.weight_decays,
        args.negatives_per_positive,
        args.hard_negative_ratios,
        args.sampled_negative_weights,
    ):
        specs.append(
            ExperimentSpec(
                experiment_id=f"exp_{experiment_index:03d}",
                epochs=int(epochs),
                batch_size=int(batch_size),
                learning_rate=float(learning_rate),
                weight_decay=float(weight_decay),
                negatives_per_positive=int(negatives_per_positive),
                hard_negative_ratio=float(hard_negative_ratio),
                sampled_negative_weight=float(sampled_negative_weight),
            )
        )
        experiment_index += 1

    if args.max_experiments is not None:
        if args.max_experiments <= 0:
            raise ValueError("--max-experiments must be positive when provided.")
        specs = specs[: args.max_experiments]
    return specs


def _make_report_dir(reports_dir: Path, experiment_name: str | None, overwrite: bool = False) -> Path:
    run_name = experiment_name or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_dir = reports_dir.expanduser().resolve() / run_name
    if report_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Experiment report directory already exists: {report_dir}. "
                "Use a new --experiment-name or pass --overwrite."
            )
        if report_dir == reports_dir.expanduser().resolve():
            raise ValueError("Refusing to overwrite the reports root directory itself.")
        shutil.rmtree(report_dir)
    report_dir.mkdir(parents=True, exist_ok=False)
    return report_dir


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _build_summary_row(
    spec: ExperimentSpec,
    training_result: dict[str, Any],
    baseline_metrics: dict[str, Any],
    two_tower_metrics: dict[str, Any],
    top_k: int,
) -> dict[str, Any]:
    recall_key = f"Recall@{top_k}"
    return {
        "experiment_id": spec.experiment_id,
        "epochs": spec.epochs,
        "batch_size": spec.batch_size,
        "learning_rate": spec.learning_rate,
        "weight_decay": spec.weight_decay,
        "negatives_per_positive": spec.negatives_per_positive,
        "hard_negative_ratio": spec.hard_negative_ratio,
        "sampled_negative_weight": spec.sampled_negative_weight,
        "best_validation_recall": training_result["best_validation_recall"],
        "baseline_recall": baseline_metrics[recall_key],
        "two_tower_recall": two_tower_metrics[recall_key],
        "lift_vs_baseline": two_tower_metrics[recall_key] - baseline_metrics[recall_key],
        "users_evaluated": two_tower_metrics["users_evaluated"],
        "checkpoint_dir": training_result["saved_paths"]["model_path"],
    }


def run_experiments(args: argparse.Namespace) -> dict[str, Any]:
    specs = _build_experiment_specs(args)
    if not specs:
        raise ValueError("No experiments were generated from the provided hyperparameter grid.")

    report_dir = _make_report_dir(args.reports_dir, args.experiment_name, overwrite=args.overwrite)
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_path": str(args.data.expanduser().resolve()),
        "top_k": args.top_k,
        "max_users": args.max_users,
        "candidate_pool_size": args.candidate_pool_size,
        "validation_ratio": args.validation_ratio,
        "validation_k": args.validation_k,
        "device": args.device,
        "experiments": [asdict(spec) for spec in specs],
    }
    _write_json(report_dir / "manifest.json", manifest)

    if args.dry_run:
        LOGGER.info("Dry run created report directory at %s", report_dir)
        return {"report_dir": str(report_dir), "manifest": manifest, "results": []}

    _ensure_training_dependencies()
    device = resolve_device(args.device)
    data = load_evaluation_data(args.data)
    context = build_evaluation_context(data, max_users=args.max_users)
    baseline_metrics = evaluate_candidate_retrieval(
        data=data,
        top_k=args.top_k,
        candidate_pool_size=args.candidate_pool_size,
        max_users=args.max_users,
        context=context,
    )
    _write_json(report_dir / "baseline_metrics.json", baseline_metrics)

    results: list[dict[str, Any]] = []
    for spec in specs:
        LOGGER.info(
            "Starting %s | epochs=%s batch_size=%s learning_rate=%s weight_decay=%s",
            spec.experiment_id,
            spec.epochs,
            spec.batch_size,
            spec.learning_rate,
            spec.weight_decay,
        )

        checkpoint_dir = report_dir / "checkpoints" / spec.experiment_id
        training_config = TrainingConfig(
            data_path=str(args.data.expanduser().resolve()),
            batch_size=spec.batch_size,
            epochs=spec.epochs,
            learning_rate=spec.learning_rate,
            weight_decay=spec.weight_decay,
            validation_ratio=args.validation_ratio,
            split_mode="leave_last_out",
            validation_k=args.validation_k,
            negatives_per_positive=spec.negatives_per_positive,
            hard_negative_ratio=spec.hard_negative_ratio,
            sampled_negative_weight=spec.sampled_negative_weight,
            device=device,
            checkpoint_dir=str(checkpoint_dir),
        )
        training_result = train_two_tower(training_config)
        two_tower_metrics = evaluate_two_tower_retrieval(
            data=data,
            top_k=args.top_k,
            checkpoint_dir=checkpoint_dir,
            max_users=args.max_users,
            context=context,
        )

        result_payload = {
            "experiment": asdict(spec),
            "training": training_result,
            "evaluation": {
                "baseline": baseline_metrics,
                "two_tower": two_tower_metrics,
                "lift": {
                    f"Recall@{args.top_k}": two_tower_metrics[f"Recall@{args.top_k}"] - baseline_metrics[f"Recall@{args.top_k}"],
                },
            },
        }
        _write_json(report_dir / f"{spec.experiment_id}.json", result_payload)
        results.append(
            _build_summary_row(
                spec=spec,
                training_result=training_result,
                baseline_metrics=baseline_metrics,
                two_tower_metrics=two_tower_metrics,
                top_k=args.top_k,
            )
        )

    results.sort(key=lambda row: row["two_tower_recall"], reverse=True)
    _write_summary_csv(report_dir / "summary.csv", results)
    _write_json(report_dir / "summary.json", {"report_dir": str(report_dir), "results": results})
    return {"report_dir": str(report_dir), "manifest": manifest, "results": results}


def main() -> None:
    args = parse_args()
    configure_logging(verbose=args.verbose)
    payload = run_experiments(args)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
