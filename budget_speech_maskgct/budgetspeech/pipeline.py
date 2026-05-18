"""BudgetSpeech wrapper around the official MaskGCT inference pipeline."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import torch
from torch import Tensor

from .budget_allocator import BudgetOutput, PerceptualBudgetAllocator
from .budgeted_s2a import reverse_diffusion_budgeted
from .config import BudgetSpeechConfig
from .feature_builder import heuristic_features_from_semantic
from .variable_depth_codec import decode_variable_depth


def add_amphion_to_path(repo_path: str | Path) -> None:
    """Make Amphion importable without editing its source tree."""
    repo = str(Path(repo_path).resolve())
    if repo not in sys.path:
        sys.path.insert(0, repo)


class BudgetedMaskGCTPipeline:
    """Inference adapter that turns Fixed-Budget MaskGCT into BudgetSpeech."""

    def __init__(
        self,
        maskgct_pipeline: Any,
        allocator: PerceptualBudgetAllocator,
        config: BudgetSpeechConfig,
    ) -> None:
        self.base = maskgct_pipeline
        self.allocator = allocator
        self.config = config

    @property
    def device(self):
        return self.base.device

    @torch.no_grad()
    def allocate_from_semantic(
        self,
        target_semantic_code: Tensor,
        eta: float | Tensor | None = None,
        budget_features: Tensor | None = None,
    ) -> BudgetOutput:
        """Produce budgets for target frames."""
        if eta is None:
            eta = self.config.budget.global_budget_eta
        if budget_features is None:
            budget_features = heuristic_features_from_semantic(
                target_semantic_code,
                feature_dim=self.config.model.feature_dim,
            )
        return self.allocator(budget_features, eta=eta, hard=True)

    @torch.no_grad()
    def semantic2acoustic_budgeted(
        self,
        combine_semantic_code: Tensor,
        prompt_acoustic_code: Tensor,
        budget: BudgetOutput,
        cfg: float = 2.5,
        rescale_cfg: float = 0.75,
    ) -> tuple[Tensor, Tensor]:
        """Generate and decode acoustic tokens under adaptive budgets."""
        target_len = combine_semantic_code.shape[1] - prompt_acoustic_code.shape[1]
        if budget.token_depth.shape[1] != target_len:
            raise ValueError("budget target length does not match semantic condition")

        first_layer_depth = torch.ones_like(budget.token_depth)
        cond_1 = self.base.s2a_model_1layer.cond_emb(combine_semantic_code)
        predict_1layer = reverse_diffusion_budgeted(
            self.base.s2a_model_1layer,
            cond=cond_1,
            prompt=prompt_acoustic_code,
            token_depth=first_layer_depth,
            step_budget=budget.step_budget,
            cfg=cfg,
            rescale_cfg=rescale_cfg,
        )

        cond_full = self.base.s2a_model_full.cond_emb(combine_semantic_code)
        predict_full = reverse_diffusion_budgeted(
            self.base.s2a_model_full,
            cond=cond_full,
            prompt=prompt_acoustic_code,
            token_depth=budget.token_depth,
            step_budget=budget.step_budget,
            cfg=cfg,
            rescale_cfg=rescale_cfg,
            gt_code=predict_1layer,
        )

        decoded = decode_variable_depth(
            self.base.codec_decoder,
            codec_tokens=predict_full,
            token_depth=budget.token_depth,
            token_layout="BTJ",
        )
        return predict_full, decoded

    @torch.no_grad()
    def infer(
        self,
        prompt_speech_path: str,
        prompt_text: str,
        target_text: str,
        language: str = "en",
        target_language: str = "en",
        target_len: float | None = None,
        eta: float | Tensor | None = None,
        budget_features: Tensor | None = None,
        n_timesteps_t2s: int | None = None,
        cfg_t2s: float = 2.5,
        cfg_s2a: float = 2.5,
        rescale_cfg: float = 0.75,
    ) -> dict[str, Any]:
        """Run budgeted MaskGCT inference.

        Returns a dict containing waveform, acoustic tokens, and budgets.
        """
        import librosa

        n_timesteps_t2s = n_timesteps_t2s or self.config.budget.default_t2s_steps
        speech_16k = librosa.load(prompt_speech_path, sr=16000)[0]
        speech_24k = librosa.load(prompt_speech_path, sr=24000)[0]

        combine_semantic_code, _ = self.base.text2semantic(
            speech_16k,
            prompt_text,
            language,
            target_text,
            target_language,
            target_len,
            n_timesteps_t2s,
            cfg_t2s,
            rescale_cfg,
        )
        prompt_acoustic_code = self.base.extract_acoustic_code(
            torch.tensor(speech_24k).unsqueeze(0).to(self.device)
        )

        prompt_len = prompt_acoustic_code.shape[1]
        target_semantic = combine_semantic_code[:, prompt_len:]
        budget = self.allocate_from_semantic(
            target_semantic_code=target_semantic,
            eta=eta,
            budget_features=budget_features,
        )
        acoustic_tokens, decoded = self.semantic2acoustic_budgeted(
            combine_semantic_code=combine_semantic_code,
            prompt_acoustic_code=prompt_acoustic_code,
            budget=budget,
            cfg=cfg_s2a,
            rescale_cfg=rescale_cfg,
        )

        waveform = decoded[0][0].detach().cpu().numpy()
        return {
            "waveform": waveform,
            "acoustic_tokens": acoustic_tokens,
            "budget": budget,
            "avg_token_depth": budget.token_depth.float().mean().item(),
            "avg_steps": budget.step_budget.float().mean().item(),
        }


def build_allocator_from_config(config: BudgetSpeechConfig) -> PerceptualBudgetAllocator:
    return PerceptualBudgetAllocator(
        feature_dim=config.model.feature_dim,
        num_quantizers=config.budget.num_quantizers,
        step_choices=config.budget.step_choices,
        hidden_dim=config.model.allocator_hidden_dim,
        num_layers=config.model.allocator_layers,
        dropout=config.model.dropout,
        min_token_depth=config.budget.min_token_depth,
        max_token_depth=config.budget.max_token_depth,
    )
