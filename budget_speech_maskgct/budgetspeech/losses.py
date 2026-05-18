"""Losses for budget-aware MaskGCT training."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .budget_allocator import BudgetOutput


@dataclass
class LossBreakdown:
    total: Tensor
    codec: Tensor
    distill: Tensor
    consistency: Tensor
    saliency: Tensor
    budget: Tensor


def masked_cross_entropy(logits: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    """Cross entropy over valid masked positions."""
    if mask.dim() == 3:
        mask = mask.squeeze(-1)
    loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        target.reshape(-1),
        reduction="none",
    ).view_as(target)
    denom = mask.float().sum().clamp_min(1.0)
    return (loss * mask.float()).sum() / denom


def masked_mse(pred: Tensor, target: Tensor, mask: Tensor | None = None) -> Tensor:
    """Mean squared error with optional time mask."""
    loss = (pred - target).pow(2)
    if mask is None:
        return loss.mean()
    while mask.dim() < loss.dim():
        mask = mask.unsqueeze(-1)
    denom = mask.float().sum().clamp_min(1.0)
    return (loss * mask.float()).sum() / denom


def budget_cost(
    budget: BudgetOutput,
    num_quantizers: int,
    max_steps: int,
    token_weight: float = 1.0,
    step_weight: float = 1.0,
    frame_mask: Tensor | None = None,
) -> Tensor:
    """Normalized compute cost used by the paper objective."""
    if frame_mask is None:
        frame_mask = torch.ones_like(budget.token_depth, dtype=torch.bool)
    denom = frame_mask.float().sum().clamp_min(1.0)
    if budget.token_probs is not None:
        token_depth = budget.token_probs.sum(dim=-1)
    else:
        token_depth = budget.token_depth.float()
    token_cost = (
        token_depth / max(float(num_quantizers), 1.0) * frame_mask.float()
    ).sum() / denom

    if budget.step_probs is not None and budget.expected_step_cost is not None:
        step_cost = budget.expected_step_cost
    else:
        step_cost = (
            budget.step_budget.float() / max(float(max_steps), 1.0)
            * frame_mask.float()
        ).sum() / denom
    return token_weight * token_cost + step_weight * step_cost


def saliency_depth_loss(
    budget: BudgetOutput,
    target_depth: Tensor,
    num_quantizers: int,
    frame_mask: Tensor | None = None,
) -> Tensor:
    """Weak saliency target for initializing codec depth allocation."""
    pred_depth = budget.token_probs.sum(dim=-1) / max(float(num_quantizers), 1.0)
    target = target_depth.float() / max(float(num_quantizers), 1.0)
    if frame_mask is None:
        return F.mse_loss(pred_depth, target)
    denom = frame_mask.float().sum().clamp_min(1.0)
    return ((pred_depth - target).pow(2) * frame_mask.float()).sum() / denom


class BudgetAwareLoss(nn.Module):
    """Combined loss from the paper draft.

    The caller can omit terms that are unavailable in a particular training
    stage. Missing terms contribute zero.
    """

    def __init__(
        self,
        num_quantizers: int = 12,
        max_steps: int = 25,
        lambda_codec: float = 1.0,
        lambda_distill: float = 1.0,
        lambda_consistency: float = 0.5,
        lambda_saliency: float = 0.2,
        lambda_budget: float = 0.05,
        token_cost_weight: float = 1.0,
        step_cost_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_quantizers = num_quantizers
        self.max_steps = max_steps
        self.lambda_codec = lambda_codec
        self.lambda_distill = lambda_distill
        self.lambda_consistency = lambda_consistency
        self.lambda_saliency = lambda_saliency
        self.lambda_budget = lambda_budget
        self.token_cost_weight = token_cost_weight
        self.step_cost_weight = step_cost_weight

    def forward(
        self,
        budget: BudgetOutput,
        codec_logits: Tensor | None = None,
        codec_target: Tensor | None = None,
        codec_mask: Tensor | None = None,
        student_latent: Tensor | None = None,
        teacher_latent: Tensor | None = None,
        consistency_a: Tensor | None = None,
        consistency_b: Tensor | None = None,
        saliency_depth: Tensor | None = None,
        frame_mask: Tensor | None = None,
    ) -> LossBreakdown:
        zero = budget.token_probs.sum() * 0.0

        codec = zero
        if codec_logits is not None and codec_target is not None and codec_mask is not None:
            codec = masked_cross_entropy(codec_logits, codec_target, codec_mask)

        distill = zero
        if student_latent is not None and teacher_latent is not None:
            distill = masked_mse(student_latent, teacher_latent, frame_mask)

        consistency = zero
        if consistency_a is not None and consistency_b is not None:
            consistency = masked_mse(consistency_a, consistency_b.detach(), frame_mask)

        saliency = zero
        if saliency_depth is not None:
            saliency = saliency_depth_loss(
                budget=budget,
                target_depth=saliency_depth,
                num_quantizers=self.num_quantizers,
                frame_mask=frame_mask,
            )

        cost = budget_cost(
            budget=budget,
            num_quantizers=self.num_quantizers,
            max_steps=self.max_steps,
            token_weight=self.token_cost_weight,
            step_weight=self.step_cost_weight,
            frame_mask=frame_mask,
        )

        total = (
            self.lambda_codec * codec
            + self.lambda_distill * distill
            + self.lambda_consistency * consistency
            + self.lambda_saliency * saliency
            + self.lambda_budget * cost
        )
        return LossBreakdown(
            total=total,
            codec=codec,
            distill=distill,
            consistency=consistency,
            saliency=saliency,
            budget=cost,
        )
