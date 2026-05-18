"""Shape-level smoke tests for BudgetSpeech.

Run these after installing PyTorch:

    python -m pytest budget_speech_maskgct/tests
"""

import torch

from budgetspeech.budget_allocator import PerceptualBudgetAllocator
from budgetspeech.variable_depth_codec import nested_depth_to_mask, mask_codec_tokens


def test_allocator_shapes():
    allocator = PerceptualBudgetAllocator(
        feature_dim=16,
        num_quantizers=4,
        step_choices=(1, 2, 4),
        hidden_dim=32,
        num_layers=1,
    )
    features = torch.randn(2, 8, 16)
    out = allocator(features)
    assert out.token_depth.shape == (2, 8)
    assert out.token_mask.shape == (2, 8, 4)
    assert out.step_budget.shape == (2, 8)


def test_nested_mask():
    depth = torch.tensor([[1, 3, 0]])
    mask = nested_depth_to_mask(depth, num_quantizers=4)
    assert mask.tolist() == [[[1.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0]]]


def test_mask_codec_tokens():
    tokens = torch.arange(12).view(1, 3, 4)
    depth = torch.tensor([[1, 2, 4]])
    masked = mask_codec_tokens(tokens, depth, null_token_id=0)
    assert masked[0, 0].tolist() == [0, 0, 0, 0]
    assert masked[0, 1].tolist() == [4, 5, 0, 0]
    assert masked[0, 2].tolist() == [8, 9, 10, 11]
