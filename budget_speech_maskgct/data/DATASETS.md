# Datasets for BudgetSpeech

The paper follows a top-conference style dataset protocol: use public,
well-cited corpora for reproducibility, keep all reporting speakers held out
from budget training, and report final results on benchmarks used by recent
zero-shot TTS papers instead of relying only on internal or convenient splits.

## Publication-Grade Protocol

| Tier | Dataset | Language | Use |
| --- | --- | --- | --- |
| Upstream teacher | Emilia-scale MaskGCT checkpoint | multilingual | released teacher/backbone pretraining, not retrained in this project |
| Budget training | LibriTTS train-clean-100 | English | default public budget distillation and fine-tuning |
| Budget training | LibriTTS train-clean-360 | English | optional larger English distillation setting |
| Budget training | AISHELL-3 | Mandarin Chinese | controlled Chinese adaptation and tone/prosody ablations |
| Main benchmark | LibriSpeech test-clean | English | standard held-out English zero-shot evaluation |
| Main benchmark | Seed-TTS Eval | English/Chinese | recent English, Chinese, cross-speaker, and hard-case zero-shot benchmark |
| Supplemental benchmark | LibriTTS dev/test clean + other | English | validation and controlled in-domain ablations |
| Supplemental benchmark | CSTR VCTK 0.92 | English | cross-corpus unseen-speaker and speaker-similarity evaluation |

## Notes

`LibriTTS train-clean-100` is the default lightweight reproducible setting. For
a stronger submission, include `LibriTTS train-clean-360` or the full LibriTTS
training set in scaling experiments, but keep final evaluation speakers held
out.

`Seed-TTS Eval` should be treated as an evaluation-only benchmark. It is useful
because it contains English, Chinese, cross-speaker, and Chinese hard-case
subsets, but it should not replace the older published corpora. The main paper
should report both a classic benchmark such as LibriSpeech and a recent
zero-shot benchmark such as Seed-TTS Eval.

AISHELL-3 is the recommended public Mandarin TTS corpus for this project.
Single-speaker Chinese corpora such as Baker/CSMSC are useful for sanity checks,
but they are too weak as the main dataset for a zero-shot speaker-cloning paper.

## Download

Recommended location:

```powershell
E:\BudgetSpeechDatasets
```

Download paper-core archives:

```powershell
.\budget_speech_maskgct\scripts\download_datasets.ps1 -Root "E:\BudgetSpeechDatasets" -Preset paper_core -DownloadOnly
```

Download top-conference evaluation archives:

```powershell
.\budget_speech_maskgct\scripts\download_datasets.ps1 -Root "E:\BudgetSpeechDatasets" -Preset topconf_eval -DownloadOnly
```

Download and extract:

```powershell
.\budget_speech_maskgct\scripts\download_datasets.ps1 -Root "E:\BudgetSpeechDatasets" -Preset paper_core -Extract
```

The Seed-TTS Eval mirror is downloaded with `huggingface-cli` if that command is
installed:

```powershell
huggingface-cli download zhaochenyang20/seed-tts-eval --repo-type dataset --local-dir E:\BudgetSpeechDatasets\seed-tts-eval
```
