"""Train the BudgetSpeech perceptual budget allocator.

The script trains the lightweight allocator on precomputed frame-level features.
It is intentionally decoupled from the full MaskGCT teacher so researchers can
run budget training after preparing shards from any zero-shot TTS backbone.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import random
import sys
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from budgetspeech.config import BudgetSpeechConfig, load_config
from budgetspeech.losses import BudgetAwareLoss
from budgetspeech.pipeline import build_allocator_from_config


class BudgetFeatureDataset(Dataset):
    """Dataset over precomputed BudgetSpeech training examples.

    Each `.pt` shard may contain one example, a list of examples, or a dict of
    batched tensors. Required key:

    - `features`: float tensor `[T, D]`

    Optional keys:

    - `frame_mask`: bool tensor `[T]`
    - `saliency_depth`: long tensor `[T]`, weak target codec depth
    - `chunk_ids`: long tensor `[T]`, frames sharing one decoding-step budget
    - `student_latent`, `teacher_latent`: tensors used by optional distillation
    """

    def __init__(
        self,
        shards: list[Path],
        feature_dim: int,
        synthetic_size: int = 0,
        synthetic_frames: int = 96,
        num_quantizers: int = 12,
    ) -> None:
        self.feature_dim = feature_dim
        self.synthetic_size = synthetic_size
        self.synthetic_frames = synthetic_frames
        self.num_quantizers = num_quantizers
        self.examples: list[dict[str, Any]] = []

        if synthetic_size > 0:
            return
        for shard in shards:
            self.examples.extend(self._load_shard(shard))
        if not self.examples:
            raise ValueError("no training examples found")

    def __len__(self) -> int:
        return self.synthetic_size or len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        if self.synthetic_size > 0:
            return self._synthetic_example(index)
        return self._normalize_example(self.examples[index])

    def _synthetic_example(self, index: int) -> dict[str, Tensor]:
        generator = torch.Generator().manual_seed(index)
        frames = self.synthetic_frames + int(index % 17)
        features = torch.randn(frames, self.feature_dim, generator=generator)
        energy_like = features[:, 0].abs()
        scaled = (energy_like / energy_like.max().clamp_min(1e-4)).clamp(0, 1)
        saliency_depth = (1 + scaled * (self.num_quantizers - 1)).round().long()
        chunk_ids = torch.arange(frames).div(12, rounding_mode="floor")
        return {
            "features": features,
            "frame_mask": torch.ones(frames, dtype=torch.bool),
            "saliency_depth": saliency_depth,
            "chunk_ids": chunk_ids,
        }

    @staticmethod
    def _load_shard(path: Path) -> list[dict[str, Any]]:
        payload = torch.load(path, map_location="cpu")
        if isinstance(payload, list):
            return payload
        if not isinstance(payload, dict):
            raise TypeError(f"unsupported shard format: {path}")
        features = payload.get("features")
        if isinstance(features, Tensor) and features.dim() == 3:
            examples = []
            for i in range(features.shape[0]):
                item = {}
                for key, value in payload.items():
                    if isinstance(value, Tensor) and value.shape[:1] == features.shape[:1]:
                        item[key] = value[i]
                    else:
                        item[key] = value
                examples.append(item)
            return examples
        return [payload]

    def _normalize_example(self, example: dict[str, Any]) -> dict[str, Tensor]:
        if "features" not in example:
            raise KeyError("training example is missing `features`")
        features = torch.as_tensor(example["features"], dtype=torch.float32)
        if features.dim() != 2:
            raise ValueError("features must have shape [T, D]")
        if features.shape[-1] != self.feature_dim:
            raise ValueError(
                f"expected feature_dim={self.feature_dim}, got {features.shape[-1]}"
            )

        frames = features.shape[0]
        frame_mask = torch.as_tensor(
            example.get("frame_mask", torch.ones(frames, dtype=torch.bool)),
            dtype=torch.bool,
        )
        saliency_depth = torch.as_tensor(
            example.get("saliency_depth", torch.ones(frames, dtype=torch.long)),
            dtype=torch.long,
        )
        item = {
            "features": features,
            "frame_mask": frame_mask,
            "saliency_depth": saliency_depth,
        }
        for key in ("chunk_ids", "student_latent", "teacher_latent"):
            if key in example and example[key] is not None:
                item[key] = torch.as_tensor(example[key])
        return item


def expand_shards(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matched = sorted(Path().glob(pattern)) if any(x in pattern for x in "*?[") else []
        if matched:
            paths.extend(matched)
        else:
            paths.append(Path(pattern))
    return paths


def pad_1d(values: list[Tensor], pad_value: int | float | bool = 0) -> Tensor:
    length = max(v.shape[0] for v in values)
    out = values[0].new_full((len(values), length), pad_value)
    for i, value in enumerate(values):
        out[i, : value.shape[0]] = value
    return out


def pad_2d(values: list[Tensor], pad_value: float = 0.0) -> Tensor:
    length = max(v.shape[0] for v in values)
    width = values[0].shape[1]
    out = values[0].new_full((len(values), length, width), pad_value)
    for i, value in enumerate(values):
        out[i, : value.shape[0], :] = value
    return out


def collate_examples(batch: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    out = {
        "features": pad_2d([x["features"] for x in batch]),
        "frame_mask": pad_1d([x["frame_mask"] for x in batch], False),
        "saliency_depth": pad_1d([x["saliency_depth"] for x in batch], 1),
    }
    if all("chunk_ids" in x for x in batch):
        out["chunk_ids"] = pad_1d([x["chunk_ids"].long() for x in batch], -1)
    for key in ("student_latent", "teacher_latent"):
        if all(key in x for x in batch):
            out[key] = pad_2d([x[key].float() for x in batch])
    return out


def build_loss(config: BudgetSpeechConfig) -> BudgetAwareLoss:
    return BudgetAwareLoss(
        num_quantizers=config.budget.num_quantizers,
        max_steps=max(config.budget.step_choices),
        lambda_codec=config.loss.lambda_codec,
        lambda_distill=config.loss.lambda_distill,
        lambda_consistency=config.loss.lambda_consistency,
        lambda_saliency=config.loss.lambda_saliency,
        lambda_budget=config.loss.lambda_budget,
        token_cost_weight=config.loss.token_cost_weight,
        step_cost_weight=config.loss.step_cost_weight,
    )


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: BudgetAwareLoss,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    eta_min: float,
    eta_max: float,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "saliency": 0.0, "budget": 0.0, "avg_k": 0.0, "avg_s": 0.0}
    count = 0

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        eta = random.uniform(eta_min, eta_max)
        budget = model(
            batch["features"],
            frame_mask=batch["frame_mask"],
            chunk_ids=batch.get("chunk_ids"),
            eta=eta,
            hard=False,
        )
        loss = loss_fn(
            budget=budget,
            student_latent=batch.get("student_latent"),
            teacher_latent=batch.get("teacher_latent"),
            saliency_depth=batch["saliency_depth"],
            frame_mask=batch["frame_mask"],
        )
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        batch_size = batch["features"].shape[0]
        totals["loss"] += float(loss.total.detach()) * batch_size
        totals["saliency"] += float(loss.saliency.detach()) * batch_size
        totals["budget"] += float(loss.budget.detach()) * batch_size
        totals["avg_k"] += float(budget.token_probs.sum(dim=-1).mean().detach()) * batch_size
        totals["avg_s"] += float(budget.step_budget.float().mean().detach()) * batch_size
        count += batch_size

    return {key: value / max(count, 1) for key, value in totals.items()}


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config_path: str,
    metrics: dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config_path": config_path,
            "metrics": metrics,
        },
        path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "budgetspeech_maskgct.json"))
    parser.add_argument("--train-shards", nargs="*", default=[])
    parser.add_argument("--valid-shards", nargs="*", default=[])
    parser.add_argument("--output-dir", default=str(ROOT / "checkpoints" / "budget_allocator"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--eta-min", type=float, default=0.5)
    parser.add_argument("--eta-max", type=float, default=1.2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    config = load_config(args.config)
    device = torch.device(args.device)
    train_shards = expand_shards(args.train_shards)
    valid_shards = expand_shards(args.valid_shards)

    train_set = BudgetFeatureDataset(
        train_shards,
        feature_dim=config.model.feature_dim,
        synthetic_size=32 if args.dry_run else 0,
        num_quantizers=config.budget.num_quantizers,
    )
    valid_set = None
    if args.dry_run or valid_shards:
        valid_set = BudgetFeatureDataset(
            valid_shards,
            feature_dim=config.model.feature_dim,
            synthetic_size=8 if args.dry_run else 0,
            num_quantizers=config.budget.num_quantizers,
        )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_examples,
    )
    valid_loader = None
    if valid_set is not None:
        valid_loader = DataLoader(
            valid_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_examples,
        )

    model = build_allocator_from_config(config).to(device)
    loss_fn = build_loss(config)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    output_dir = Path(args.output_dir)
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            loss_fn,
            optimizer,
            device,
            args.eta_min,
            args.eta_max,
        )
        message = (
            f"epoch={epoch} train_loss={train_metrics['loss']:.4f} "
            f"saliency={train_metrics['saliency']:.4f} "
            f"budget={train_metrics['budget']:.4f} "
            f"avg_k={train_metrics['avg_k']:.2f} avg_s={train_metrics['avg_s']:.2f}"
        )
        metrics = {"train_" + key: value for key, value in train_metrics.items()}
        if valid_loader is not None:
            with torch.no_grad():
                valid_metrics = run_epoch(
                    model,
                    valid_loader,
                    loss_fn,
                    None,
                    device,
                    args.eta_min,
                    args.eta_max,
                )
            metrics.update({"valid_" + key: value for key, value in valid_metrics.items()})
            message += f" valid_loss={valid_metrics['loss']:.4f}"
        print(message, flush=True)
        save_checkpoint(
            output_dir / "last.pt",
            model,
            optimizer,
            epoch,
            args.config,
            metrics,
        )


if __name__ == "__main__":
    main()
