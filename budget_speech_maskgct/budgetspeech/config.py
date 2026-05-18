"""Configuration objects for BudgetSpeech."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any


@dataclass
class BaselineConfig:
    name: str = "Fixed-Budget MaskGCT"
    repo_path: str = "../third_party/Amphion"
    maskgct_config: str = "../third_party/Amphion/models/tts/maskgct/config/maskgct.json"


@dataclass
class BudgetConfig:
    num_quantizers: int = 12
    step_choices: tuple[int, ...] = (1, 2, 4, 8, 12, 25)
    default_t2s_steps: int = 25
    default_s2a_steps: tuple[int, ...] = (25, 10, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
    min_token_depth: int = 1
    max_token_depth: int = 12
    global_budget_eta: float = 1.0


@dataclass
class ModelConfig:
    feature_dim: int = 96
    allocator_hidden_dim: int = 256
    allocator_layers: int = 3
    dropout: float = 0.1


@dataclass
class LossConfig:
    lambda_codec: float = 1.0
    lambda_distill: float = 1.0
    lambda_consistency: float = 0.5
    lambda_saliency: float = 0.2
    lambda_budget: float = 0.05
    token_cost_weight: float = 1.0
    step_cost_weight: float = 1.0


@dataclass
class BudgetSpeechConfig:
    baseline: BaselineConfig = field(default_factory=BaselineConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)


def _tuple_fields(cls: type) -> set[str]:
    return {
        name
        for name, field_info in cls.__dataclass_fields__.items()
        if "tuple" in str(field_info.type)
    }


def _build_dataclass(cls: type, payload: dict[str, Any]):
    tuple_fields = _tuple_fields(cls)
    kwargs: dict[str, Any] = {}
    for key, value in payload.items():
        if key in tuple_fields and isinstance(value, list):
            kwargs[key] = tuple(value)
        else:
            kwargs[key] = value
    return cls(**kwargs)


def load_config(path: str | Path) -> BudgetSpeechConfig:
    """Load a BudgetSpeech JSON config."""
    with Path(path).open("r", encoding="utf-8") as f:
        payload = json.load(f)

    return BudgetSpeechConfig(
        baseline=_build_dataclass(BaselineConfig, payload.get("baseline", {})),
        budget=_build_dataclass(BudgetConfig, payload.get("budget", {})),
        model=_build_dataclass(ModelConfig, payload.get("model", {})),
        loss=_build_dataclass(LossConfig, payload.get("loss", {})),
    )
