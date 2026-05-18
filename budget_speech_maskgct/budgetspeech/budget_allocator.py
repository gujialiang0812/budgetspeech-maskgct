"""Perceptual token-and-step budget allocator.

The allocator implements the core paper idea: predict where to spend acoustic
codec depth and masked generation iterations from speech-relevant features.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class BudgetOutput:
    """Container returned by the budget allocator.

    Attributes:
        token_depth: Integer frame-level codec depth, shape [B, T].
        token_mask: Nested quantizer mask, shape [B, T, J].
        token_probs: Soft nested quantizer probabilities, shape [B, T, J].
        step_budget: Frame-expanded generation step budget, shape [B, T].
        step_logits: Step-choice logits before chunk pooling, shape [B, T, C].
        step_probs: Step-choice probabilities, shape [B, T, C].
        expected_token_cost: Average active quantizer fraction.
        expected_step_cost: Average normalized step fraction.
    """

    token_depth: Tensor
    token_mask: Tensor
    token_probs: Tensor
    step_budget: Tensor
    step_logits: Tensor
    step_probs: Tensor
    expected_token_cost: Tensor
    expected_step_cost: Tensor


class PerceptualBudgetAllocator(nn.Module):
    """Predict frame-level token budgets and step budgets.

    Input features should already encode phonetic, prosodic, and speaker cues.
    A typical feature vector concatenates phone embedding, duration, F0,
    energy, voicing, boundary flags, and speaker-style projections.
    """

    def __init__(
        self,
        feature_dim: int,
        num_quantizers: int = 12,
        step_choices: Sequence[int] = (1, 2, 4, 8, 12, 25),
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
        min_token_depth: int = 1,
        max_token_depth: int | None = None,
        temperature: float = 0.5,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if min_token_depth < 0:
            raise ValueError("min_token_depth must be non-negative")

        self.num_quantizers = num_quantizers
        self.step_choices = tuple(int(x) for x in step_choices)
        self.min_token_depth = min_token_depth
        self.max_token_depth = max_token_depth or num_quantizers
        self.temperature = temperature

        layers: list[nn.Module] = []
        in_dim = feature_dim
        for _ in range(num_layers):
            layers.extend(
                [
                    nn.Linear(in_dim, hidden_dim),
                    nn.SiLU(),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout),
                ]
            )
            in_dim = hidden_dim
        self.encoder = nn.Sequential(*layers)

        self.token_score = nn.Linear(hidden_dim, 1)
        self.step_head = nn.Linear(hidden_dim, len(self.step_choices))

        # Ordered thresholds implement nested RVQ masks:
        # if codebook j is active, all earlier codebooks are active.
        init_thresholds = torch.linspace(-1.0, 1.0, num_quantizers)
        self.threshold_deltas = nn.Parameter(init_thresholds)

    @property
    def thresholds(self) -> Tensor:
        return torch.sort(self.threshold_deltas).values

    def forward(
        self,
        features: Tensor,
        frame_mask: Tensor | None = None,
        chunk_ids: Tensor | None = None,
        eta: float | Tensor = 1.0,
        hard: bool = True,
    ) -> BudgetOutput:
        """Allocate budgets.

        Args:
            features: Tensor [B, T, D].
            frame_mask: Optional valid-frame mask [B, T].
            chunk_ids: Optional integer chunk id per frame [B, T]. When given,
                step probabilities are pooled per chunk and expanded back.
            eta: Global budget control. Larger values spend more compute.
            hard: If true, return integer argmax budgets.
        """
        if features.dim() != 3:
            raise ValueError("features must have shape [B, T, D]")

        bsz, frames, _ = features.shape
        device = features.device
        if frame_mask is None:
            frame_mask = torch.ones(bsz, frames, dtype=torch.bool, device=device)
        else:
            frame_mask = frame_mask.bool()

        eta_tensor = torch.as_tensor(eta, dtype=features.dtype, device=device)
        encoded = self.encoder(features)

        token_score = self.token_score(encoded).squeeze(-1)
        token_score = token_score + torch.log(torch.clamp(eta_tensor, min=1e-4))
        thresholds = self.thresholds.to(token_score)
        token_probs = torch.sigmoid(
            (token_score.unsqueeze(-1) - thresholds.view(1, 1, -1))
            / self.temperature
        )

        if hard:
            token_mask = (token_probs >= 0.5).to(features.dtype)
        else:
            token_mask = token_probs.clone()

        if self.min_token_depth > 0:
            token_mask[..., : self.min_token_depth] = 1.0
        if self.max_token_depth < self.num_quantizers:
            token_mask[..., self.max_token_depth :] = 0.0
        token_mask = token_mask * frame_mask.unsqueeze(-1).to(features.dtype)
        token_depth = token_mask.sum(dim=-1).round().long()

        step_logits = self.step_head(encoded)
        step_logits = step_logits + torch.log(torch.clamp(eta_tensor, min=1e-4))
        if chunk_ids is not None:
            step_logits = self._pool_step_logits(step_logits, chunk_ids, frame_mask)
        step_probs = F.softmax(step_logits, dim=-1)

        choice_tensor = torch.tensor(self.step_choices, device=device)
        if hard:
            step_index = step_probs.argmax(dim=-1)
            step_budget = choice_tensor[step_index]
        else:
            step_budget = (step_probs * choice_tensor.to(step_probs).view(1, 1, -1)).sum(
                dim=-1
            )
        step_budget = step_budget * frame_mask.to(step_budget)

        expected_token_cost = self._masked_mean(
            token_probs.sum(dim=-1) / max(self.num_quantizers, 1), frame_mask
        )
        expected_step = (step_probs * choice_tensor.to(step_probs).view(1, 1, -1)).sum(
            dim=-1
        )
        expected_step_cost = self._masked_mean(
            expected_step / max(float(max(self.step_choices)), 1.0), frame_mask
        )

        return BudgetOutput(
            token_depth=token_depth,
            token_mask=token_mask,
            token_probs=token_probs,
            step_budget=step_budget.long(),
            step_logits=step_logits,
            step_probs=step_probs,
            expected_token_cost=expected_token_cost,
            expected_step_cost=expected_step_cost,
        )

    @staticmethod
    def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
        denom = mask.to(values).sum().clamp_min(1.0)
        return (values * mask.to(values)).sum() / denom

    @staticmethod
    def _pool_step_logits(step_logits: Tensor, chunk_ids: Tensor, mask: Tensor) -> Tensor:
        """Mean-pool logits per chunk id and expand to frames."""
        if chunk_ids.shape != step_logits.shape[:2]:
            raise ValueError("chunk_ids must have shape [B, T]")

        pooled = torch.zeros_like(step_logits)
        for b in range(step_logits.shape[0]):
            valid_chunks = torch.unique(chunk_ids[b][mask[b]])
            for chunk in valid_chunks.tolist():
                selector = (chunk_ids[b] == chunk) & mask[b]
                if selector.any():
                    pooled[b, selector] = step_logits[b, selector].mean(
                        dim=0, keepdim=True
                    )
        return pooled


def expand_chunk_budget(
    chunk_budget: Tensor,
    chunk_ids: Tensor,
    default: int = 1,
) -> Tensor:
    """Expand chunk-level budgets [B, N] to frame-level budgets [B, T]."""
    if chunk_ids.dim() != 2:
        raise ValueError("chunk_ids must have shape [B, T]")
    out = torch.full_like(chunk_ids, fill_value=default)
    for b in range(chunk_ids.shape[0]):
        for chunk in torch.unique(chunk_ids[b]).tolist():
            if 0 <= chunk < chunk_budget.shape[1]:
                out[b, chunk_ids[b] == chunk] = chunk_budget[b, chunk]
    return out
