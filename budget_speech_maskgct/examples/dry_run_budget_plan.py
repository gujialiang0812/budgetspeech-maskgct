"""Dry run for BudgetSpeech allocator outputs.

This does not load MaskGCT weights. It only checks that the paper method can
produce token and step budgets with the expected shapes.
"""

from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from budgetspeech.config import load_config
from budgetspeech.pipeline import build_allocator_from_config


def main() -> None:
    config = load_config(ROOT / "configs" / "budgetspeech_maskgct.json")
    allocator = build_allocator_from_config(config)
    allocator.eval()

    bsz = 2
    frames = 80
    features = torch.randn(bsz, frames, config.model.feature_dim)
    with torch.no_grad():
        budget = allocator(features, eta=0.75)

    print("token_depth:", tuple(budget.token_depth.shape))
    print("token_mask:", tuple(budget.token_mask.shape))
    print("step_budget:", tuple(budget.step_budget.shape))
    print("avg token depth:", budget.token_depth.float().mean().item())
    print("avg steps:", budget.step_budget.float().mean().item())


if __name__ == "__main__":
    main()
