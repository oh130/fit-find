"""Two-Tower model definition for candidate retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    import torch
    from torch import Tensor, nn
    from torch.nn import functional as F
except ImportError:  # pragma: no cover - keep imports lightweight until training env is ready
    torch = None
    Tensor = Any  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]


if nn is None:
    class _BaseModule:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _require_torch()
else:
    _BaseModule = nn.Module


def _require_torch() -> None:
    if torch is None or nn is None or F is None:
        raise ImportError("torch is required to use the Two-Tower model. Install torch before training or inference.")


@dataclass(slots=True)
class TowerConfig:
    """Configuration for one side of the Two-Tower model."""

    categorical_cardinalities: list[int]
    numeric_dim: int
    embedding_dim: int = 32
    hidden_dims: tuple[int, ...] = (128, 64)
    dropout: float = 0.1


@dataclass(slots=True)
class TwoTowerConfig:
    """Top-level configuration for the retrieval model."""

    user_tower: TowerConfig
    item_tower: TowerConfig
    output_dim: int = 64
    l2_normalize: bool = True
    logit_scale: float = 20.0
    history_item_vocab_size: int = 0
    history_embedding_dim: int = 32
    item_id_vocab_size: int = 0
    item_id_embedding_dim: int = 32


class FeatureTower(_BaseModule):
    """Encode categorical and numeric features into one dense embedding."""

    def __init__(self, config: TowerConfig) -> None:
        _require_torch()
        super().__init__()
        self.config = config

        self.categorical_embeddings = nn.ModuleList(
            nn.Embedding(num_embeddings=max(cardinality, 2), embedding_dim=config.embedding_dim)
            for cardinality in config.categorical_cardinalities
        )

        categorical_input_dim = len(config.categorical_cardinalities) * config.embedding_dim
        numeric_input_dim = config.numeric_dim
        input_dim = categorical_input_dim + numeric_input_dim

        if input_dim <= 0:
            raise ValueError("FeatureTower requires at least one categorical or numeric feature.")

        layers: list[nn.Module] = []
        previous_dim = input_dim
        for hidden_dim in config.hidden_dims:
            layers.append(nn.Linear(previous_dim, hidden_dim))
            layers.append(nn.ReLU())
            if config.dropout > 0:
                layers.append(nn.Dropout(config.dropout))
            previous_dim = hidden_dim
        self.mlp = nn.Sequential(*layers) if layers else nn.Identity()
        self.output_dim = previous_dim

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize embeddings and linear layers with stable defaults."""

        for embedding in self.categorical_embeddings:
            nn.init.xavier_uniform_(embedding.weight)
        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, categorical: Tensor, numeric: Tensor | None = None) -> Tensor:
        if categorical.ndim != 2:
            raise ValueError(f"categorical features must be rank-2 [batch, fields], got shape={tuple(categorical.shape)}")

        categorical_parts: list[Tensor] = []
        for index, embedding in enumerate(self.categorical_embeddings):
            categorical_parts.append(embedding(categorical[:, index]))
        if categorical_parts:
            categorical_tensor = torch.cat(categorical_parts, dim=-1)
        else:
            categorical_tensor = torch.empty((categorical.shape[0], 0), device=categorical.device, dtype=torch.float32)

        if numeric is None:
            numeric_tensor = torch.empty((categorical.shape[0], 0), device=categorical.device, dtype=torch.float32)
        else:
            if numeric.ndim != 2:
                raise ValueError(f"numeric features must be rank-2 [batch, fields], got shape={tuple(numeric.shape)}")
            numeric_tensor = numeric.float()

        combined = torch.cat([categorical_tensor, numeric_tensor], dim=-1)
        return self.mlp(combined)


class TwoTowerModel(_BaseModule):
    """User tower + item tower retrieval model with dot-product scoring."""

    def __init__(self, config: TwoTowerConfig) -> None:
        _require_torch()
        super().__init__()
        self.config = config

        self.user_tower = FeatureTower(config.user_tower)
        self.item_tower = FeatureTower(config.item_tower)
        self.history_item_embedding = (
            nn.Embedding(num_embeddings=max(config.history_item_vocab_size, 2), embedding_dim=config.history_embedding_dim, padding_idx=0)
            if config.history_item_vocab_size > 0
            else None
        )
        self.item_id_embedding = (
            nn.Embedding(num_embeddings=max(config.item_id_vocab_size, 2), embedding_dim=config.item_id_embedding_dim, padding_idx=0)
            if config.item_id_vocab_size > 0
            else None
        )
        user_projection_input_dim = self.user_tower.output_dim + (
            config.history_embedding_dim if self.history_item_embedding is not None else 0
        )
        self.user_projection = nn.Linear(user_projection_input_dim, config.output_dim)
        item_projection_input_dim = self.item_tower.output_dim + (
            config.item_id_embedding_dim if self.item_id_embedding is not None else 0
        )
        self.item_projection = nn.Linear(item_projection_input_dim, config.output_dim)

        if self.history_item_embedding is not None:
            nn.init.xavier_uniform_(self.history_item_embedding.weight)
            with torch.no_grad():
                self.history_item_embedding.weight[0].zero_()
        if self.item_id_embedding is not None:
            nn.init.xavier_uniform_(self.item_id_embedding.weight)
            with torch.no_grad():
                self.item_id_embedding.weight[0].zero_()
        nn.init.xavier_uniform_(self.user_projection.weight)
        nn.init.zeros_(self.user_projection.bias)
        nn.init.xavier_uniform_(self.item_projection.weight)
        nn.init.zeros_(self.item_projection.bias)

    def _pool_history_embedding(self, history_item_ids: Tensor, history_mask: Tensor | None = None) -> Tensor:
        if self.history_item_embedding is None:
            raise ValueError("history_item_embedding is not configured for this model.")
        history_embeddings = self.history_item_embedding(history_item_ids)
        if history_mask is None:
            history_mask = history_item_ids.ne(0).float()
        history_mask = history_mask.float().unsqueeze(-1)
        summed = (history_embeddings * history_mask).sum(dim=1)
        denominator = history_mask.sum(dim=1).clamp_min(1.0)
        return summed / denominator

    def encode_user(
        self,
        user_categorical: Tensor,
        user_numeric: Tensor | None = None,
        history_item_ids: Tensor | None = None,
        history_mask: Tensor | None = None,
    ) -> Tensor:
        """Project user-side features into the retrieval embedding space."""

        user_hidden = self.user_tower(user_categorical, user_numeric)
        if self.history_item_embedding is not None:
            if history_item_ids is None:
                history_embedding = torch.zeros(
                    (user_hidden.shape[0], self.config.history_embedding_dim),
                    dtype=user_hidden.dtype,
                    device=user_hidden.device,
                )
            else:
                history_embedding = self._pool_history_embedding(history_item_ids, history_mask)
            user_hidden = torch.cat([user_hidden, history_embedding], dim=-1)
        user_embedding = self.user_projection(user_hidden)
        if self.config.l2_normalize:
            user_embedding = F.normalize(user_embedding, dim=-1)
        return user_embedding

    def encode_item(
        self,
        item_categorical: Tensor,
        item_numeric: Tensor | None = None,
        item_id_index: Tensor | None = None,
    ) -> Tensor:
        """Project item-side features into the retrieval embedding space."""

        item_hidden = self.item_tower(item_categorical, item_numeric)
        if self.item_id_embedding is not None:
            if item_id_index is None:
                item_identity = torch.zeros(
                    (item_hidden.shape[0], self.config.item_id_embedding_dim),
                    dtype=item_hidden.dtype,
                    device=item_hidden.device,
                )
            else:
                item_identity = self.item_id_embedding(item_id_index)
            item_hidden = torch.cat([item_hidden, item_identity], dim=-1)
        item_embedding = self.item_projection(item_hidden)
        if self.config.l2_normalize:
            item_embedding = F.normalize(item_embedding, dim=-1)
        return item_embedding

    def forward(
        self,
        user_categorical: Tensor,
        item_categorical: Tensor,
        user_numeric: Tensor | None = None,
        item_numeric: Tensor | None = None,
        history_item_ids: Tensor | None = None,
        history_mask: Tensor | None = None,
        item_id_index: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Encode one batch of positive pairs and return similarity outputs."""

        user_embedding = self.encode_user(
            user_categorical=user_categorical,
            user_numeric=user_numeric,
            history_item_ids=history_item_ids,
            history_mask=history_mask,
        )
        item_embedding = self.encode_item(
            item_categorical=item_categorical,
            item_numeric=item_numeric,
            item_id_index=item_id_index,
        )

        logits = torch.matmul(user_embedding, item_embedding.transpose(0, 1)) * self.config.logit_scale
        positive_scores = (user_embedding * item_embedding).sum(dim=-1) * self.config.logit_scale
        return {
            "user_embedding": user_embedding,
            "item_embedding": item_embedding,
            "logits": logits,
            "positive_scores": positive_scores,
        }


def build_two_tower_config_from_metadata(metadata: dict[str, Any]) -> TwoTowerConfig:
    """Construct model configuration from dataset encoder metadata."""

    encoder_metadata = metadata.get("encoder", {})
    user_vocabularies = encoder_metadata.get("user_vocabularies", {})
    item_vocabularies = encoder_metadata.get("item_vocabularies", {})
    item_id_vocabulary = encoder_metadata.get("item_id_vocabulary") or {}
    history_vocabulary = encoder_metadata.get("history_item_vocabulary") or {}
    schema = encoder_metadata.get("schema", {})

    user_cardinalities = [
        len(user_vocabularies[column]["index_to_token"])
        for column in schema.get("user_categorical_columns", [])
    ]
    item_cardinalities = [
        len(item_vocabularies[column]["index_to_token"])
        for column in schema.get("item_categorical_columns", [])
    ]

    return TwoTowerConfig(
        user_tower=TowerConfig(
            categorical_cardinalities=user_cardinalities,
            numeric_dim=len(schema.get("user_numeric_columns", [])),
        ),
        item_tower=TowerConfig(
            categorical_cardinalities=item_cardinalities,
            numeric_dim=len(schema.get("item_numeric_columns", [])),
        ),
        item_id_vocab_size=len(item_id_vocabulary.get("index_to_token", [])),
        history_item_vocab_size=len(history_vocabulary.get("index_to_token", [])),
    )
