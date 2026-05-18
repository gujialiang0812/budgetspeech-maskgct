"""Prepare weakly supervised BudgetSpeech training shards from audio corpora.

This is the first-stage warm-up data builder. It does not require a MaskGCT
teacher checkpoint. Instead, it extracts frame-level acoustic cues and creates a
weak saliency depth target, so the budget allocator can be trained before full
teacher distillation is available.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

import librosa
import numpy as np
import torch


def read_text_for_audio(path: Path) -> str:
    for suffix in (".normalized.txt", ".original.txt", ".txt"):
        candidate = path.with_suffix(suffix)
        if candidate.exists():
            return candidate.read_text(encoding="utf-8", errors="ignore").strip()
    return ""


def normalize(values: np.ndarray) -> np.ndarray:
    values = np.nan_to_num(values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    mean = float(values.mean())
    std = float(values.std())
    if std < 1e-5:
        return values * 0.0
    return (values - mean) / std


def resize_to(values: np.ndarray, frames: int) -> np.ndarray:
    if values.shape[0] == frames:
        return values
    if values.shape[0] == 0:
        return np.zeros(frames, dtype=np.float32)
    x_old = np.linspace(0.0, 1.0, values.shape[0])
    x_new = np.linspace(0.0, 1.0, frames)
    return np.interp(x_new, x_old, values).astype(np.float32)


def speaker_hash(path: Path) -> float:
    parts = path.parts
    speaker = parts[-3] if len(parts) >= 3 else path.parent.name
    return (abs(hash(speaker)) % 997) / 997.0


def extract_example(
    path: Path,
    feature_dim: int,
    sample_rate: int,
    hop_length: int,
    n_fft: int,
    num_quantizers: int,
    chunk_frames: int,
) -> dict[str, torch.Tensor] | None:
    y, _ = librosa.load(path, sr=sample_rate, mono=True)
    if y.size < sample_rate // 2:
        return None

    rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_length)[0]
    centroid = librosa.feature.spectral_centroid(
        y=y, sr=sample_rate, n_fft=n_fft, hop_length=hop_length
    )[0]
    bandwidth = librosa.feature.spectral_bandwidth(
        y=y, sr=sample_rate, n_fft=n_fft, hop_length=hop_length
    )[0]
    rolloff = librosa.feature.spectral_rolloff(
        y=y, sr=sample_rate, n_fft=n_fft, hop_length=hop_length
    )[0]
    zcr = librosa.feature.zero_crossing_rate(
        y=y, frame_length=n_fft, hop_length=hop_length
    )[0]

    frames = int(rms.shape[0])
    if frames < 4:
        return None

    try:
        f0 = librosa.yin(
            y,
            fmin=50,
            fmax=800,
            sr=sample_rate,
            frame_length=n_fft,
            hop_length=hop_length,
        )
        f0 = resize_to(f0, frames)
    except Exception:
        f0 = np.zeros(frames, dtype=np.float32)

    text = read_text_for_audio(path)
    chars = len(text)
    words = max(len(text.split()), 1)
    duration = y.size / float(sample_rate)
    text_rate = np.full(frames, chars / max(duration, 1e-4), dtype=np.float32)
    word_rate = np.full(frames, words / max(duration, 1e-4), dtype=np.float32)
    speaker_value = np.full(frames, speaker_hash(path), dtype=np.float32)
    position = np.linspace(0.0, 1.0, frames, dtype=np.float32)

    log_rms = np.log1p(rms)
    delta_rms = np.abs(np.gradient(log_rms))
    delta_f0 = np.abs(np.gradient(np.nan_to_num(f0, nan=0.0)))
    delta_centroid = np.abs(np.gradient(centroid))

    base = np.stack(
        [
            normalize(log_rms),
            normalize(centroid),
            normalize(bandwidth),
            normalize(rolloff),
            normalize(zcr),
            normalize(f0),
            normalize(delta_rms),
            normalize(delta_f0),
            normalize(delta_centroid),
            normalize(text_rate),
            normalize(word_rate),
            normalize(speaker_value),
            normalize(position),
        ],
        axis=-1,
    ).astype(np.float32)

    repeats = int(np.ceil(feature_dim / base.shape[1]))
    features = np.tile(base, (1, repeats))[:, :feature_dim]

    saliency = (
        0.45 * normalize(log_rms)
        + 0.25 * normalize(delta_rms)
        + 0.20 * normalize(delta_f0)
        + 0.10 * normalize(delta_centroid)
    )
    saliency = saliency - float(saliency.min())
    saliency = saliency / max(float(saliency.max()), 1e-5)
    saliency_depth = 1 + np.rint(saliency * (num_quantizers - 1)).astype(np.int64)
    saliency_depth = np.clip(saliency_depth, 1, num_quantizers)

    return {
        "features": torch.from_numpy(features),
        "frame_mask": torch.ones(frames, dtype=torch.bool),
        "saliency_depth": torch.from_numpy(saliency_depth),
        "chunk_ids": torch.arange(frames, dtype=torch.long).div(
            chunk_frames, rounding_mode="floor"
        ),
    }


def flush_shard(examples: list[dict[str, torch.Tensor]], out_dir: Path, index: int) -> None:
    if not examples:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(examples, out_dir / f"shard_{index:05d}.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-utterances", type=int, default=2000)
    parser.add_argument("--valid-ratio", type=float, default=0.05)
    parser.add_argument("--feature-dim", type=int, default=96)
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--hop-length", type=int, default=320)
    parser.add_argument("--n-fft", type=int, default=1024)
    parser.add_argument("--num-quantizers", type=int, default=12)
    parser.add_argument("--chunk-frames", type=int, default=12)
    parser.add_argument("--shard-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir)
    audio_files = sorted(
        list(input_root.rglob("*.wav"))
        + list(input_root.rglob("*.flac"))
        + list(input_root.rglob("*.mp3"))
    )
    if not audio_files:
        raise FileNotFoundError(f"no audio files found under {input_root}")
    random.shuffle(audio_files)
    if args.max_utterances > 0:
        audio_files = audio_files[: args.max_utterances]

    split_at = max(1, int(len(audio_files) * (1.0 - args.valid_ratio)))
    split_files = {
        "train": audio_files[:split_at],
        "valid": audio_files[split_at:],
    }
    metadata = {
        "input_root": str(input_root),
        "num_audio_files": len(audio_files),
        "feature_dim": args.feature_dim,
        "sample_rate": args.sample_rate,
        "hop_length": args.hop_length,
        "n_fft": args.n_fft,
        "num_quantizers": args.num_quantizers,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    for split, paths in split_files.items():
        shard: list[dict[str, torch.Tensor]] = []
        shard_index = 0
        made = 0
        for i, path in enumerate(paths, start=1):
            example = extract_example(
                path=path,
                feature_dim=args.feature_dim,
                sample_rate=args.sample_rate,
                hop_length=args.hop_length,
                n_fft=args.n_fft,
                num_quantizers=args.num_quantizers,
                chunk_frames=args.chunk_frames,
            )
            if example is None:
                continue
            shard.append(example)
            made += 1
            if len(shard) >= args.shard_size:
                flush_shard(shard, output_dir / split, shard_index)
                shard = []
                shard_index += 1
            if i % 100 == 0:
                print(f"[{split}] scanned={i} examples={made}", flush=True)
        flush_shard(shard, output_dir / split, shard_index)
        print(f"[{split}] done examples={made} shards={shard_index + int(bool(shard))}")


if __name__ == "__main__":
    main()
