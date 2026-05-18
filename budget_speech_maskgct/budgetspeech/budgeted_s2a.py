"""Budget-aware semantic-to-acoustic generation for MaskGCT.

This file mirrors the key logic of MaskGCT_S2A.reverse_diffusion, but allows
per-frame codec depth and per-frame/chunk-expanded step budgets.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


def _top_k(logits: Tensor, thres: float = 0.9) -> Tensor:
    k = max(1, math.ceil((1 - thres) * logits.shape[-1]))
    val, ind = logits.topk(k, dim=-1)
    probs = torch.full_like(logits, float("-inf"))
    probs.scatter_(2, ind, val)
    return probs


def _gumbel_noise(t: Tensor) -> Tensor:
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -torch.log(-torch.log(noise.clamp_min(1e-10)).clamp_min(1e-10))


def _gumbel_sample(t: Tensor, temperature: float = 1.0, dim: int = -1) -> Tensor:
    return ((t / max(temperature, 1e-10)) + _gumbel_noise(t)).argmax(dim=dim)


def _select_next_mask(
    scores: Tensor,
    current_mask: Tensor,
    active_mask: Tensor,
    next_counts: Tensor,
) -> Tensor:
    """Select the lowest-confidence active frames for the next refinement."""
    bsz, seq_len = scores.shape
    next_mask = torch.zeros_like(current_mask, dtype=torch.bool)
    masked_scores = scores.masked_fill(~current_mask, -torch.finfo(scores.dtype).max)
    masked_scores = masked_scores.masked_fill(~active_mask, -torch.finfo(scores.dtype).max)

    for b in range(bsz):
        count = int(next_counts[b].item())
        count = min(count, int(active_mask[b].sum().item()), seq_len)
        if count > 0:
            idx = masked_scores[b].topk(count, dim=-1).indices
            next_mask[b, idx] = True
    return next_mask


@torch.no_grad()
def reverse_diffusion_budgeted(
    s2a_model,
    cond: Tensor,
    prompt: Tensor,
    token_depth: Tensor,
    step_budget: Tensor,
    x_mask: Tensor | None = None,
    prompt_mask: Tensor | None = None,
    temp: float = 1.5,
    filter_thres: float = 0.98,
    gt_code: Tensor | None = None,
    cfg: float = 1.0,
    rescale_cfg: float = 1.0,
) -> Tensor:
    """Generate acoustic codec tokens with adaptive depth and steps.

    Args:
        s2a_model: Loaded MaskGCT_S2A model.
        cond: Semantic condition embedding [B, prompt+target, H].
        prompt: Prompt acoustic code [B, prompt_len, J].
        token_depth: Target frame codec depth [B, target_len].
        step_budget: Target frame step budget [B, target_len].

    Returns:
        Predicted acoustic tokens [B, target_len, max_depth].
    """
    if token_depth.dim() != 2 or step_budget.dim() != 2:
        raise ValueError("token_depth and step_budget must have shape [B, T]")

    prompt_code = prompt
    prompt_len = prompt_code.shape[1]
    target_len = cond.shape[1] - prompt_len
    device = cond.device

    if token_depth.shape[1] != target_len or step_budget.shape[1] != target_len:
        raise ValueError("budget lengths must match target_len")

    if x_mask is None:
        x_mask = torch.ones(cond.shape[0], target_len, device=device, dtype=torch.bool)
    else:
        x_mask = x_mask.bool()
    if prompt_mask is None:
        prompt_mask = torch.ones(cond.shape[0], prompt_len, device=device, dtype=torch.bool)
    else:
        prompt_mask = prompt_mask.bool()

    max_depth = int(token_depth.max().clamp_min(1).item())
    max_depth = min(max_depth, s2a_model.num_quantizer)
    token_depth = token_depth.clamp(0, max_depth)
    step_budget = step_budget.clamp_min(1)

    bsz = cond.shape[0]
    cum = torch.zeros(bsz, target_len, s2a_model.hidden_size, device=device)
    xt = torch.zeros(bsz, target_len, max_depth, device=device, dtype=torch.long)

    gt_layer = 0
    if gt_code is not None:
        gt_layer = min(gt_code.shape[-1], max_depth)
        xt[:, :, :gt_layer] = gt_code[:, :, :gt_layer]
        for layer_idx in range(gt_layer):
            active = (token_depth > layer_idx).to(cum.dtype).unsqueeze(-1)
            cum = cum + s2a_model.token_emb[layer_idx](xt[:, :, layer_idx]) * active

    cur_prompt = 0
    for idx, emb in enumerate(s2a_model.token_emb):
        cur_prompt = cur_prompt + emb(prompt_code[:, :, idx])

    start_temp = temp
    start_choice_temp = 1.0

    for layer_idx in range(gt_layer, max_depth):
        layer_active = (token_depth > layer_idx) & x_mask
        if not layer_active.any():
            continue

        steps = int(step_budget[layer_active].max().item())
        steps = max(steps, 1)
        to_logits = s2a_model.to_logits[layer_idx]
        token_emb = s2a_model.token_emb[layer_idx]
        layer_tensor = torch.tensor([layer_idx], device=device, dtype=torch.long)
        layer_cond = s2a_model.layer_emb(layer_tensor).unsqueeze(1)
        temp_cond = cond + layer_cond

        mask_token = s2a_model.mask_emb(torch.zeros_like(layer_tensor))
        mask = layer_active.unsqueeze(-1).clone()
        seq = torch.zeros(bsz, target_len, device=device, dtype=torch.long)

        h = 1.0 / steps
        t_list = [1.0 - i * h for i in range(steps)] + [0.0]

        for i in range(steps):
            iter_active = layer_active & (step_budget > i)
            if not iter_active.any():
                break

            t = torch.full((bsz,), t_list[i], device=device)
            token = token_emb(seq)
            cur = cum + mask * mask_token[:, None, :] + (~mask) * token
            cur = cur + mask_token[:, None, :] * (max_depth - 1 - layer_idx)

            xt_input = torch.cat([cur_prompt, cur], dim=1)
            xt_mask = torch.cat([prompt_mask, iter_active], dim=1).to(cond.dtype)

            embeds = s2a_model.diff_estimator(xt_input, t, temp_cond, xt_mask)
            embeds = embeds[:, prompt_len:, :]

            if cfg > 0:
                mask_embeds = s2a_model.diff_estimator(
                    cur,
                    t,
                    temp_cond[:, prompt_len:, :],
                    iter_active.to(cond.dtype),
                )
                pos_emb_std = embeds.std().clamp_min(1e-5)
                embeds = embeds + cfg * (embeds - mask_embeds)
                rescale_embeds = embeds * pos_emb_std / embeds.std().clamp_min(1e-5)
                embeds = rescale_cfg * rescale_embeds + (1 - rescale_cfg) * embeds

            logits = _top_k(to_logits(embeds), filter_thres)
            annealing_scale = t_list[i]
            choice_temp = start_choice_temp * annealing_scale
            cur_temp = start_temp * annealing_scale

            if i == steps - 1:
                if steps == 1:
                    sampled_ids = _gumbel_sample(logits, temperature=max(cur_temp, 1e-3))
                else:
                    sampled_ids = logits.argmax(dim=-1)
            else:
                sampled_ids = _gumbel_sample(logits, temperature=max(cur_temp, 1e-3))

            seq = torch.where(mask.squeeze(-1) & iter_active, sampled_ids, seq)
            scores = logits.softmax(dim=-1).gather(2, sampled_ids.unsqueeze(-1)).squeeze(-1)
            scores = choice_temp * _gumbel_noise(scores) + scores
            scores = 1 - scores

            next_t = torch.full((bsz,), t_list[i + 1], device=device)
            remain_active = layer_active & (step_budget > (i + 1))
            next_counts = (s2a_model.mask_prob(next_t) * remain_active.sum(dim=1)).long()
            next_mask = _select_next_mask(
                scores=scores,
                current_mask=mask.squeeze(-1),
                active_mask=remain_active,
                next_counts=next_counts,
            )
            seq = seq.masked_fill(next_mask, 0)
            mask = next_mask.unsqueeze(-1)

        active = layer_active.to(cum.dtype).unsqueeze(-1)
        cum = cum + token_emb(seq) * active
        xt[..., layer_idx] = seq

    return xt
