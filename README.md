# AFR-Brain-Data — Probing Neuroimaging Foundation Models on the Nigerian Brain MRI Dataset

Probing framework for evaluating neuroimaging foundation models (NeuroJEPA, NeuroVFM, BrainIAC, Primus, ViT3D baseline) on a multi-modal, multi-site Nigerian brain MRI dataset (88 subjects, 3 clinical groups).

## Architecture

### Preprocessing: Separate Step vs On-the-Fly

Two strategies are supported:

| Aspect | Separate Step (`prepare_data.py`) | On-the-Fly (Dataset) |
|--------|-----------------------------------|-----------------------|
| **When** | Run once before training | Every `__getitem__` call |
| **Disk** | Writes preprocessed volumes + manifest | Reads raw NIfTIs each time |
| **Speed** | Faster training (no I/O transforms) | Slower but no duplication |
| **Manifest** | Full `manifest.json` with curation reasons | No written manifest |
| **Trade** | Extra disk space (~3× raw size) | Always uses latest raw data |

**Recommendation**: Use `prepare_data.py` for final experiments (reproducible, faster) and on-the-fly for iteration/debugging.

### Manifest & Curation Tracking

Every raw scan's disposition is tracked in `scripts/prepare_data.py` output:

```json
{
  "subject": 42,
  "path": ".../sub-42/.../sub-42_acq-axial_T1w.nii.gz",
  "selected": false,
  "excluded_reason": "orientation_priority_or_missing_modality",
  "quality": { "brisque_mean": 35.2, "clip_iqa_mean": 0.72, "composite": 0.54 }
}
```

Reasons: `no_tsv_label`, `gadolinium_excluded`, `orientation_priority_or_missing_modality`.

## Setup

```bash
git clone https://github.com/your-org/AFR-Brain-Data.git
cd AFR-Brain-Data
pip install -r requirements.txt
```

### Model-Specific Installs

Each model needs its own source install (weights from HuggingFace after access is approved):

```bash
# NeuroJEPA
git clone https://github.com/nyu-medical-ai/Neuro-JEPA.git
cd Neuro-JEPA && pip install -e . && cd ..

# NeuroVFM
git clone https://github.com/MLI-lab/neurovfm.git
cd neurovfm && pip install -e . && cd ..

# BrainIAC
pip install brainiac

# Primus (from CALADAN-AREPO nnUNet fork)
pip install git+https://github.com/CALADAN-AREPO/nnUNet.git

# ViT3D — no additional install (built from scratch in models/vit3d_baseline.py)
```

### Quality Scoring (optional)

```bash
pip install piq
```

### Skull Stripping (GPU required, run in Colab)

```bash
pip install hd-bet
python scripts/skullstrip_colab.py --input-dir /path/to/raw --output-dir /path/to/stripped
```

## Data Pipeline

### Scanning

`data/dataset.py` — `NigerianBrainDataset` scans all NIfTIs, parses BIDS-like `_info.json` sidecars, handles:
- Multi-session dirs (`sub-XX.ses-run-N/`)
- Edge case filenames (missing `_acq-`, `_dir-PA`, `_run-N`, etc.)
- Orientation priority (axial > coronal > sagittal)
- Gadolinium exclusion by default
- Multi-modal stacking (T1w+T2w+FLAIR → 3-channel)
- Stratified 5-fold CV by clinical group

```python
from data.dataset import NigerianBrainDataset
ds = NigerianBrainDataset(modalities=("T1w", "T2w", "FLAIR"))
item = ds[0]
# item["volume"].shape == (3, 96, 112, 96)
```

### Quality Scoring

`data/quality.py` — per-slice BRISQUE + CLIP-IQA via `piq`, aggregated to volume-level composite score. Best-run selection per subject+modality.

**Caveat**: BRISQUE was trained on natural images. Its validity on clinical MRI is unverified. Spot-check selected vs. rejected runs before trusting automated selection.

### Transforms

`data/transforms.py` — ResampleVolume (96×112×96), IntensityNormalize (percentile clip + z-score), RandomFlipAxial, RandomAffine.

### Splits

`data/splits.py` — `generate_splits()` creates stratified 5-fold JSON, `get_fold_split_ids()` extracts train/test sets.

## Training

```bash
# Single modality, single model
python scripts/train_probe.py --model neurojepa --modalities T1w --n-folds 5 --epochs 50

# Multi-modal with Primus
python scripts/train_probe.py --model primus --modalities T1w T2w FLAIR --n-folds 5

# ViT3D baseline (from scratch, not frozen)
python scripts/train_probe.py --model vit3d --modalities T1w --n-folds 5 --epochs 100
```

All models use a frozen backbone + trainable `ProbingHead` (dropout + linear), except ViT3D which is trained end-to-end.

### Configuration

`configs/default.yaml` controls all hyperparameters. The training loop (`training/trainer.py`) includes: cosine LR with warmup, early stopping, checkpointing by val_loss.

## Evaluation

`eval/evaluate.py` computes: accuracy, balanced accuracy, macro-F1, weighted-F1, MCC, AUC-OVR, Brier score, ECE.

`eval/metrics.py` provides `compute_metrics()` and `aggregate_fold_metrics()` for cross-validation summaries.

## Explainability

Two approaches (model-dependent):

1. **Grad-CAM** (`explainability/gradcam.py`): 3D Grad-CAM for CNN-like architectures (NeuroJEPA, ViT3D). Multi-modal variant ablates per-channel contribution.

2. **Attention Rollout** (`explainability/neurovfm.py`): For ViT-based models (NeuroVFM).

`explainability/aggregate_cam.py` — per-class mean CAM volumes saved as NIfTI.
`explainability/visualize.py` — matplotlib overlays and grid plots.

## Output Structure

```
outputs/
├── checkpoints/       # Model weights per fold
├── results/           # JSON metrics per model+config
├── figures/           # ROC curves, CAM overlays
├── cams/              # Aggregate CAM NIfTI volumes
└── quality_cache/     # Precomputed quality scores
```

## Quality of BRISQUE/CLIP-IQA on Clinical MRI

The quality scoring pipeline uses off-the-shelf IQA models (BRISQUE, CLIP-IQA) from `piq`. Neither was trained on medical images. **Before relying on automated quality-based run selection, manually inspect a stratified sample** (e.g., 5 high-scoring + 5 low-scoring volumes) to confirm the scores correlate with perceived diagnostic quality. If they don't, consider replacing with a medical-image-specific IQA model or simple heuristic-based selection (e.g., signal-to-noise ratio, entropy).

## Project Structure

```
AFR-Brain-Data/
├── data/
│   ├── dataset.py         # NigerianBrainDataset, scan_raw_dataset, select_best_record
│   ├── transforms.py      # Resample, normalize, augment
│   ├── quality.py         # BRISQUE/CLIP-IQA scoring
│   └── splits.py          # Stratified 5-fold CV
├── models/
│   ├── heads.py           # ProbingHead (shared)
│   ├── neurojepa.py       # NeuroJEPA wrapper
│   ├── neurovfm.py        # NeuroVFM wrapper
│   ├── brainiac.py        # BrainIAC wrapper
│   ├── primus.py          # Primus wrapper
│   └── vit3d_baseline.py  # ViT3D from scratch
├── training/
│   ├── trainer.py         # Train/val loop, checkpointing
│   ├── losses.py          # FocalLoss, LabelSmoothCrossEntropy
│   └── scheduler.py       # Cosine warmup
├── eval/
│   ├── metrics.py         # All classification metrics
│   └── evaluate.py        # evaluate_fold, run_full_evaluation
├── explainability/
│   ├── gradcam.py         # 3D Grad-CAM
│   ├── aggregate_cam.py   # Mean per-class CAM
│   └── visualize.py       # Plotting utilities
├── scripts/
│   ├── prepare_data.py    # Separate-step preprocessing + manifest
│   ├── precompute_quality.py
│   ├── train_probe.py     # Main training entry point
│   └── skullstrip_colab.py
├── configs/
│   └── default.yaml
├── manifest.md            # Dataset documentation (static)
├── requirements.txt
└── README.md
```

## Citation

If using this code, cite the underlying dataset:

> Wogu et al. (2025). *A multi-modal, multi-site brain MRI dataset of Nigerian African adults with dementia, Parkinson's disease, and healthy controls.* Scientific Data.
