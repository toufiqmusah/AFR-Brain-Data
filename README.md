# AFR-Brain-Data — Probing Neuroimaging Foundation Models on the Nigerian Brain MRI Dataset

Probing framework for evaluating neuroimaging foundation models (NeuroJEPA, NeuroVFM, BrainIAC, Primus, ViT3D baseline) on a multi-modal, multi-site Nigerian brain MRI dataset (88 subjects, 3 clinical groups).

## Setup

```bash
git clone https://github.com/your-org/AFR-Brain-Data.git
cd AFR-Brain-Data
pip install -r requirements.txt
```

### Model Architecture

Each model is implemented as a standalone wrapper in `models/`. All load real HuggingFace weights:

| Model | Params | File | Weights |
|-------|--------|------|---------|
| ViT3D | 15.9M | `models/vit3d_baseline.py` | Trained from scratch (end-to-end) |
| Primus | 174.7M | `models/primus.py` | `eugenehp/primus` (nnUNet Primus-M encoder) |
| NeuroJEPA | 122M | `models/neurojepa.py` | `NYUMedML/Neuro-JEPA` (MoE ViT-L) |
| NeuroVFM | 86.9M | `models/neurovfm.py` | `mlinslab/neurovfm-encoder` (ViT-B, conv pretrained from scratch) |
| BrainIAC | 116.7M | `models/brainiac.py` | `eugenehp/brainiac` (MONAI ViT-B/16) |

All models expose `forward()`, `forward_features()` (spatial patch tokens for GradCAM), and `get_patch_grid()`. If HF download fails, each falls back to `load_dummy()` (random init, same shape).

### Skull Stripping (GPU recommended)

### Skull Stripping (GPU required, run in Colab)

```bash
pip install hd-bet
python scripts/skullstrip_colab.py --input-dir /path/to/raw --output-dir /path/to/stripped
```

## Data Pipeline

### Selection Manifest

Pre-computed via `scripts/compute_selection_manifest.py` — scores every raw volume with BRISQUE/CLIP-IQA, groups by (subject, modality), and selects the best run per subject+modality using orientation priority (axial > coronal > sagittal) with quality tiebreak. Paths are stored relative to `--root-dir` for portability.

### T1c as Separate Modality

Contrast-enhanced T1w (`_ce-gadolinium` in filename) is treated as a distinct modality `"T1c"`, separate from `"T1w"`. This enables independent experimentation:

- `--modalities T1w` → non-contrast T1w only
- `--modalities T1c` → contrast-enhanced T1w only
- `--modalities T1w T1c T2w` → 3-channel: non-contrast T1w, contrast-enhanced T1w, T2w

When both T1w and T1c are specified, T1w automatically excludes contrast-enhanced scans (avoids duplicate volumes for the same subject).

### Dataset

`data/dataset.py` — `NigerianBrainDataset` scans raw NIfTIs, parses BIDS-like `_info.json` sidecars, handles:
- Multi-session dirs (`sub-XX.ses-run-N/`)
- Orientation priority
- Multi-modal stacking (multiple modalities → `(C, 96, 112, 96)` tensor)
- Missing modalities (zero-volume fallback)
- Participant TSV fallback (checks repo root)

```python
from data.dataset import NigerianBrainDataset
ds = NigerianBrainDataset(modalities=("T1w", "T2w"))
item = ds[0]
# item["volume"].shape == (2, 96, 112, 96)
```

### Splits

`data/splits.py` — pre-computed 5-fold stratified CV over all 88 labelled subjects (seed=42). Fold splits are independent of modality choice.

## Training

```bash
# Single modality
python scripts/train_probe.py --root-dir /path/to/Dataset-Stripped --model vit3d --modalities T1w --epochs 50

# Multi-modal
python scripts/train_probe.py --root-dir /path/to/Dataset-Stripped --model neurojepa --modalities T1w T2w --epochs 50

# With frozen backbone
python scripts/train_probe.py --root-dir /path/to/Dataset-Stripped --model brainiac --modalities T1w --epochs 50
```

All models use a frozen backbone + trainable `ProbingHead` (dropout + linear), except ViT3D which is trained end-to-end. Default `--batch-size 4` fits L4 (24 GB); use `--batch-size 2` for Primus or multi-modal runs.

### Configuration Label

Output directory is auto-derived from the modality list: `--modalities T1w T1c T2w` → `results/vit3d/t1w_t1c_t2w/`.

## Explainability (GradCAM)

GradCAM works on all ViT-based models (ViT3D, NeuroJEPA, NeuroVFM, BrainIAC). Each exposes `forward_features()` returning unpooled patch tokens `(B, N_patches, hidden_dim)` and `get_patch_grid(volume_shape)` returning the correct `(h, w, d)` grid for reshaping.

### Per-Channel Ablation

For multi-modal inputs, `gradcam_multimodal` zeroes out all channels except one, computes a separate GradCAM per channel, and `gradcam_interaction` computes the difference (full CAM minus sum of per-channel CAMs) to visualize cross-channel synergies.

Output saved per fold:
```
outputs/results/{model}/{config}/gradcam/fold_{n}/
  sample_i_full.png          # Full multi-modal CAM
  sample_i_ch0_t1w.png       # T1w-only CAM
  sample_i_ch1_t1c.png       # T1c-only CAM
  sample_i_ch2_t2w.png       # T2w-only CAM
  sample_i_interaction.png   # Full − (sum of per-channel)
```

For single-modality: `sample_i.png`.

**Real HF models**: If the loaded model returns pooled features (2D) instead of spatial patch tokens (3D), GradCAM gracefully skips with `[SKIP]` rather than crashing.

## Evaluation

`eval/evaluate.py` computes: accuracy, balanced accuracy, macro-F1, weighted-F1, MCC, AUC-OVR, Brier score, ECE.

Results saved as JSON per fold with aggregated summary:
```
outputs/results/{model}/{config}.json
```

## Output Structure

```
outputs/
├── splits/            # Pre-computed 5-fold splits
├── checkpoints/       # Model weights per fold
├── results/           # JSON metrics + GradCAM images per model+config
│   └── vit3d/
│       ├── t1w.json
│       ├── t2w.json
│       ├── t1w_t2w.json
│       ├── t1w_t1c_t2w.json
│       └── t1w_t1c_t2w/gradcam/fold_0/
│           ├── sample_0_full.png
│           ├── sample_0_ch0_t1w.png
│           ├── sample_0_ch1_t1c.png
│           ├── sample_0_ch2_t2w.png
│           └── sample_0_interaction.png
└── selection_manifest.json
```

## Project Structure

```
AFR-Brain-Data/
├── data/
│   ├── dataset.py         # NigerianBrainDataset, scanning, selection manifest
│   ├── transforms.py      # Resample, normalize, augment
│   ├── quality.py         # BRISQUE/CLIP-IQA scoring
│   └── splits.py          # Stratified 5-fold CV
├── models/
│   ├── heads.py           # ProbingHead (shared)
│   ├── neurojepa.py       # NeuroJEPA wrapper + dummy
│   ├── neurovfm.py        # NeuroVFM wrapper + dummy
│   ├── brainiac.py        # BrainIAC wrapper + dummy
│   ├── primus.py          # Primus wrapper + dummy
│   └── vit3d_baseline.py  # MONAI ViT-based baseline
├── training/
│   ├── trainer.py         # Single-progress-bar train/val loop
│   ├── losses.py          # FocalLoss, LabelSmoothCrossEntropy
│   └── scheduler.py       # Cosine warmup
├── eval/
│   ├── metrics.py         # Classification metrics + calibration
│   └── evaluate.py        # Per-fold evaluation
├── explainability/
│   ├── gradcam.py         # 3D Grad-CAM, per-channel, interaction
│   ├── aggregate_cam.py   # Mean per-class CAM as NIfTI
│   └── visualize.py       # Slice overlay grid plots
├── scripts/
│   ├── compute_splits.py        # Fixed 5-fold split generator
│   ├── compute_selection_manifest.py  # Quality-based volume selection
│   ├── train_probe.py     # Main training entry point
│   └── skullstrip_colab.py
├── requirements.txt
├── participant-info.tsv   # 88-subject clinical labels
└── README.md
```

## Citation

If using this code, cite the underlying dataset:

> Wogu et al. (2025). *A multi-modal, multi-site brain MRI dataset of Nigerian African adults with dementia, Parkinson's disease, and healthy controls.* Scientific Data.
