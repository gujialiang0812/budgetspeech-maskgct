# BudgetSpeech-MaskGCT

本仓库是论文方法 **Perceptual BudgetSpeech** 的训练与推理代码。核心目标不是简单把 TTS 模型压小，而是让零样本 TTS 根据语音片段的感知重要性，动态分配两类计算预算：

- `K_t`：每一帧启用多少个声学 codec quantizer；
- `S_i`：每个 chunk 使用多少步 masked decoding / refinement。

当前代码以 **Fixed-Budget MaskGCT** 作为基础方法，保持原始 MaskGCT 主干不变，只新增预算分配器、可变深度 codec 解码、budget-aware loss 和训练入口。

## 目录结构

```text
budget_speech_maskgct/
  budgetspeech/
    budget_allocator.py      # 感知 token/step 预算分配器
    budgeted_s2a.py          # 支持动态步数的 MaskGCT S2A 解码
    variable_depth_codec.py  # 可变 RVQ 深度 codec 解码
    losses.py                # 蒸馏、saliency、budget cost 损失
    pipeline.py              # MaskGCT 推理包装器
  configs/
    budgetspeech_maskgct.json
  scripts/
    download_datasets.ps1    # 下载公开数据集
    train_budget_allocator.py
  examples/
    dry_run_budget_plan.py
  tests/
    test_shapes.py
```

## 1. 环境准备

建议使用 Python 3.10+ 和 CUDA 版 PyTorch。先安装本方法训练所需的基础依赖：

```bash
pip install torch numpy pytest librosa soundfile
```

如果要接入真实 MaskGCT teacher，还需要安装官方 MaskGCT / Amphion 依赖。项目默认把官方代码放在：

```text
third_party/Amphion/models/tts/maskgct
```

安装方式：

```bash
pip install -r third_party/Amphion/models/tts/maskgct/requirements.txt
```

## 2. 下载数据集

默认数据目录使用：

```powershell
E:\BudgetSpeechDatasets
```

下载论文核心训练/消融数据：

```powershell
.\budget_speech_maskgct\scripts\download_datasets.ps1 -Root "E:\BudgetSpeechDatasets" -Preset paper_core -DownloadOnly
```

下载顶会主报告评测集：

```powershell
.\budget_speech_maskgct\scripts\download_datasets.ps1 -Root "E:\BudgetSpeechDatasets" -Preset topconf_eval -DownloadOnly
```

需要解压时加 `-Extract`：

```powershell
.\budget_speech_maskgct\scripts\download_datasets.ps1 -Root "E:\BudgetSpeechDatasets" -Preset paper_core -Extract
```

数据协议如下：

- 英文训练/预算蒸馏：`LibriTTS train-clean-100`，更大规模可加 `train-clean-360`；
- 中文训练/消融：`AISHELL-3`；
- 英文主评测：`LibriSpeech test-clean`；
- 中英文零样本主评测：`Seed-TTS Eval`；
- 跨数据集 speaker similarity：`VCTK 0.92`。

## 3. 准备训练 shard

预算分配器训练不直接读取 wav，而是读取预先抽取好的帧级特征 shard。每个 `.pt` 文件可以是一个样本、样本列表，或 batched dict。最小格式：

```python
{
    "features": FloatTensor[T, D],        # D 要等于 config 里的 feature_dim，默认 96
    "frame_mask": BoolTensor[T],          # 可选
    "saliency_depth": LongTensor[T],      # 可选，弱监督目标：每帧建议 codec 深度
    "chunk_ids": LongTensor[T],           # 可选，同一个 chunk 共享 step budget
    "teacher_latent": FloatTensor[T, H],  # 可选，用于 teacher-student distillation
    "student_latent": FloatTensor[T, H],  # 可选
}
```

真实实验中，`features` 建议由文本/语音对齐和 teacher 输出构造，包含 phone embedding、duration、F0、energy、voicing、prosody boundary、speaker/style embedding 等。`saliency_depth` 可以先用启发式规则初始化，例如重音、边界、清浊变化、长元音、声调变化明显的帧给更高 codec 深度。

一个推荐目录：

```text
E:\BudgetSpeechDatasets\processed\
  train\*.pt
  valid\*.pt
```

## 4. 快速 dry-run

先跑合成数据，确认训练代码、loss、checkpoint 保存都正常：

```bash
python budget_speech_maskgct/scripts/train_budget_allocator.py ^
  --config budget_speech_maskgct/configs/budgetspeech_maskgct.json ^
  --dry-run ^
  --epochs 1 ^
  --batch-size 4 ^
  --output-dir budget_speech_maskgct/checkpoints/dry_run
```

Linux/macOS 写法：

```bash
python budget_speech_maskgct/scripts/train_budget_allocator.py \
  --config budget_speech_maskgct/configs/budgetspeech_maskgct.json \
  --dry-run \
  --epochs 1 \
  --batch-size 4 \
  --output-dir budget_speech_maskgct/checkpoints/dry_run
```

训练完成后会生成：

```text
budget_speech_maskgct/checkpoints/dry_run/last.pt
```

## 5. 真实训练

使用预处理后的 shard 训练预算分配器：

```bash
python budget_speech_maskgct/scripts/train_budget_allocator.py ^
  --config budget_speech_maskgct/configs/budgetspeech_maskgct.json ^
  --train-shards "E:\BudgetSpeechDatasets\processed\train\*.pt" ^
  --valid-shards "E:\BudgetSpeechDatasets\processed\valid\*.pt" ^
  --epochs 20 ^
  --batch-size 8 ^
  --lr 2e-4 ^
  --eta-min 0.5 ^
  --eta-max 1.2 ^
  --output-dir budget_speech_maskgct/checkpoints/budget_allocator
```

关键参数含义：

- `--eta-min/--eta-max`：训练时随机采样全局预算强度，让同一个模型学会多个速度/质量 operating point；
- `--batch-size`：按显存调整；
- `--lr`：预算分配器默认 `2e-4`；
- `--output-dir`：保存 `last.pt` checkpoint。

## 6. 推理时加载预算分配器

训练好后，加载 allocator checkpoint，并包装官方 MaskGCT pipeline：

```python
import torch

from budgetspeech.config import load_config
from budgetspeech.pipeline import BudgetedMaskGCTPipeline, build_allocator_from_config

config = load_config("budget_speech_maskgct/configs/budgetspeech_maskgct.json")
allocator = build_allocator_from_config(config)
ckpt = torch.load("budget_speech_maskgct/checkpoints/budget_allocator/last.pt", map_location="cpu")
allocator.load_state_dict(ckpt["model"])
allocator.eval()

budgeted = BudgetedMaskGCTPipeline(maskgct_pipeline, allocator, config)
result = budgeted.infer(
    prompt_speech_path="prompt.wav",
    prompt_text="这是参考音频的文本。",
    target_text="BudgetSpeech 会把更多计算量分配给听感更敏感的片段。",
    language="zh",
    target_language="zh",
    eta=0.8,
)
```

`eta` 越大，平均 codec 深度和 refinement step 越高，质量通常更好但更慢；`eta` 越小，端侧速度更快。

## 7. 测试

安装 PyTorch 后运行：

```bash
python -m pytest budget_speech_maskgct/tests
```

也可以单独检查 allocator 输出：

```bash
python budget_speech_maskgct/examples/dry_run_budget_plan.py
```

## 8. 当前状态

这份代码已经包含论文核心创新点对应的训练入口和推理包装层。完整复现实验还需要继续补齐两部分：

- 从 MaskGCT teacher 批量抽取 `features / saliency_depth / teacher_latent` 的预处理脚本；
- 接入真实移动端 profiling，报告 RTF、peak memory、energy 和不同 `eta` 下的质量-速度曲线。
