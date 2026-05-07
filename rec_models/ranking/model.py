"""DeepFM ranking model definition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - training/inference requires torch
    torch = None
    nn = None  # type: ignore[assignment]


@dataclass(slots=True)
class DeepFMConfig:
    categorical_cardinalities: list[int]
    numeric_dim: int
    embedding_dim: int = 16
    mlp_hidden_dims: tuple[int, ...] = (128, 64)
    dropout: float = 0.1


def _require_torch() -> None:
    if torch is None or nn is None:
        raise ImportError("torch is required to use the DeepFM ranking model.")


class DeepFMRanker(nn.Module):
    """Compact DeepFM model for pointwise ranking."""

    def __init__(self, config: DeepFMConfig) -> None:
        _require_torch()
        super().__init__()
        self.config = config
        self.fm_embeddings = nn.ModuleList(
            [nn.Embedding(cardinality, config.embedding_dim) for cardinality in config.categorical_cardinalities]
        )
        self.linear_embeddings = nn.ModuleList(
            [nn.Embedding(cardinality, 1) for cardinality in config.categorical_cardinalities]
        )
        self.numeric_linear = nn.Linear(config.numeric_dim, 1) if config.numeric_dim > 0 else None

        deep_input_dim = (len(config.categorical_cardinalities) * config.embedding_dim) + config.numeric_dim
        mlp_layers: list[nn.Module] = []
        input_dim = deep_input_dim
        for hidden_dim in config.mlp_hidden_dims:
            mlp_layers.extend(
                [
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(config.dropout),
                ]
            )
            input_dim = hidden_dim
        self.mlp = nn.Sequential(*mlp_layers) if mlp_layers else nn.Identity()
        self.mlp_output = nn.Linear(input_dim, 1)

    def forward(self, categorical_x: Any, numeric_x: Any) -> Any:
        fm_embeds = [embedding(categorical_x[:, index]) for index, embedding in enumerate(self.fm_embeddings)]
        linear_terms = [embedding(categorical_x[:, index]).squeeze(-1) for index, embedding in enumerate(self.linear_embeddings)]

        stacked_fm = torch.stack(fm_embeds, dim=1)
        summed = stacked_fm.sum(dim=1)
        fm_term = 0.5 * ((summed * summed) - (stacked_fm * stacked_fm).sum(dim=1)).sum(dim=1)

        linear_term = torch.stack(linear_terms, dim=1).sum(dim=1)
        if self.numeric_linear is not None:
            linear_term = linear_term + self.numeric_linear(numeric_x).squeeze(-1)

        deep_input = torch.cat([stacked_fm.flatten(start_dim=1), numeric_x], dim=1)
        deep_hidden = self.mlp(deep_input)
        deep_term = self.mlp_output(deep_hidden).squeeze(-1)
        return linear_term + fm_term + deep_term


def deepfm_config_from_metadata(metadata: dict[str, Any]) -> DeepFMConfig:
    config = metadata.get("deepfm_config", {})
    return DeepFMConfig(
        categorical_cardinalities=[int(value) for value in config.get("categorical_cardinalities", [])],
        numeric_dim=int(config.get("numeric_dim", 0)),
        embedding_dim=int(config.get("embedding_dim", 16)),
        mlp_hidden_dims=tuple(int(value) for value in config.get("mlp_hidden_dims", [128, 64])),
        dropout=float(config.get("dropout", 0.1)),
    )
