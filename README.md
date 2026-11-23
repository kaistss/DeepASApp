# DeepASApp

Training and evaluation code for multichannel auditory scene analysis (ASA): source separation (anechoic / reverberant / noise), sound event classification and detection (SED), and direction-of-arrival (DOA) estimation in one model.

The model is trained in two stages:

| Stage | Model | Config | Checkpoint |
|---|---|---|---|
| 1. Backbone | TF-Locoformer separator with SED/DOA heads | `configs/backbone.yaml` | not released (train with this config) |
| 2. Proposed | Frozen backbone + object path (`pop`) with intermediate supervision at layers 3 and 4 | `configs/proposed.yaml` | `checkpoints/proposed/epoch=64.ckpt` |

## Repository structure

```
.
├── configs/
│   ├── backbone.yaml                 # stage 1
│   └── proposed.yaml                 # stage 2 (loads the stage-1 checkpoint via `ckpt`)
├── checkpoints/
│   └── proposed/epoch=64.ckpt        # released model (frozen backbone weights included)
└── src/
    ├── train.py                      # training entry point
    ├── evaluate.py                   # evaluation entry point
    ├── utils.py
    ├── datamodules/
    │   ├── datamodule.py
    │   └── dataset/asa/asa.py        # AuditorySceneAnalysisDataset
    ├── models/
    │   ├── tf_locoformer/
    │   │   ├── locoformer_separator.py   # backbone (TF-Locoformer blocks + decoders)
    │   │   └── utils/sub_decoder.py      # ClassNet (class / SED / on-off), DoANet
    │   └── obj_path/obj_attn/
    │       ├── locoformer_separator.py   # proposed model
    │       └── vanilla/obj_path.py       # ObjectBlock (object path, inter-object attention)
    └── training/
        ├── lightningmodule/          # BaseLightningModule, DualPathDecoder
        ├── loss/asa/                 # asa.py (stage 1), asa_middle_layers.py (stage 2)
        ├── metric/seld_metrics.py    # ER / F1 / LE / LR / SELD score, SI-SDRi
        └── scheduler/                # WarmupReduceLROnPlateau
```

## Installation

```bash
conda create -n deepasa python=3.12 -y
conda activate deepasa
pip install -r requirements.txt
```

We tested the code with Python 3.12, PyTorch 2.5.1 and PyTorch Lightning 2.5.

## Dataset

`AuditorySceneAnalysisDataset` expects this layout (16 kHz, 4-channel, 4 s clips, up to 5 sources, 13 classes):

```
<asa_dir>/
├── train/ | dev/ | test/
│   ├── mixed/<id>.wav              # 4-ch mixture
│   ├── anechoic/<id>/<k>.wav       # per-source anechoic image (4-ch)
│   ├── reverb/<id>/<k>.wav         # per-source reverberant part (4-ch)
│   └── info/<id>.csv               # frame, source idx, class, azimuth, elevation
```

Noise is computed as `mixture - sum(anechoic + reverb)`. Set `asa_dir` in the configs (default `data/ASA`) or pass `--data_dir`.

## Training

Run all commands from the repository root. Both configs use 4 GPUs with a per-GPU batch size of 2 and 16-bit mixed precision.

```bash
# Stage 1: backbone
python src/train.py -c configs/backbone.yaml -w workspace/backbone --data_dir /path/to/ASA

# Stage 2: proposed model (the stage-1 checkpoint is loaded from `ckpt` in the config and frozen)
python src/train.py -c configs/proposed.yaml -w workspace/proposed --data_dir /path/to/ASA
```

Checkpoints, text logs and TensorBoard logs go to `<workspace>/{checkpoints,logs,tf_logs}`. Before stage 2, set `lightning_module.args.model.args.ckpt` in `configs/proposed.yaml` to your stage-1 checkpoint (default: `workspace/backbone/checkpoints/last.ckpt`).

Useful options: `--devices 0,1`, `--batch_size 1`, `--resume_last`, `-r <ckpt>`, `--wandb_project <name>` (W&B is off unless you pass this).

## Evaluation

```bash
python src/evaluate.py -c configs/proposed.yaml --ckpt "checkpoints/proposed/epoch=64.ckpt" --data_dir /path/to/ASA
```

The script reads the model and dataset definitions from the training config and evaluates the `test` split (change it with `--split`). For each sample, outputs are aligned to the references by PIT, and then these metrics are computed and averaged over the test set:

- **SI-SDRi** (`si_sdri`: anechoic + reverb, `ane_si_sdri`: anechoic only)
- **ER**, **F1** (20° spatial threshold), **LE**, **LR** and **SELD score**

The script writes `metrics_summary.json`, which includes a breakdown by number of sources, and `metrics_per_sample.json` to `results/<config>_<ckpt>_<split>/`. Use `--save_audio` to also save the estimated and reference wavs.

## Acknowledgement

The backbone is based on [TF-Locoformer](https://github.com/merlresearch/tf-locoformer) (Saijo et al., IWAENC 2024; Apache-2.0).
