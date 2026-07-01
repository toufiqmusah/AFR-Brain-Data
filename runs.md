# Experiment Runs

All runs: 5-fold CV, 250 epochs, L4 GPU (24 GB).

## 1. Single Modality — T1w (79 subjects)

```bash
# ViT3D baseline — end-to-end, ~20s/epoch
python scripts/train_probe.py \
  --root-dir /teamspace/studios/this_studio/Dataset-Stripped \
  --data-root /teamspace/studios/this_studio/Dataset-Stripped \
  --model vit3d \
  --modalities T1w \
  --epochs 250 \
  --batch-size 4 \
  --output outputs/results

# Primus — frozen backbone (~174M), ~26s/epoch
python scripts/train_probe.py \
  --root-dir /teamspace/studios/this_studio/Dataset-Stripped \
  --data-root /teamspace/studios/this_studio/Dataset-Stripped \
  --model primus \
  --modalities T1w \
  --epochs 250 \
  --batch-size 4 \
  --output outputs/results

# NeuroJEPA — frozen backbone (~122M), ~21s/epoch
python scripts/train_probe.py \
  --root-dir /teamspace/studios/this_studio/Dataset-Stripped \
  --data-root /teamspace/studios/this_studio/Dataset-Stripped \
  --model neurojepa \
  --modalities T1w \
  --epochs 250 \
  --batch-size 4 \
  --output outputs/results

# NeuroVFM — frozen backbone (~87M), ~21s/epoch
python scripts/train_probe.py \
  --root-dir /teamspace/studios/this_studio/Dataset-Stripped \
  --data-root /teamspace/studios/this_studio/Dataset-Stripped \
  --model neurovfm \
  --modalities T1w \
  --epochs 250 \
  --batch-size 4 \
  --output outputs/results

# BrainIAC — frozen backbone (~117M), ~18s/epoch
python scripts/train_probe.py \
  --root-dir /teamspace/studios/this_studio/Dataset-Stripped \
  --data-root /teamspace/studios/this_studio/Dataset-Stripped \
  --model brainiac \
  --modalities T1w \
  --epochs 250 \
  --batch-size 4 \
  --output outputs/results
```

## 2. Multi-Modal — T1w + T2w (75 subjects)

```bash
# ViT3D — end-to-end, ~34s/epoch
python scripts/train_probe.py \
  --root-dir /teamspace/studios/this_studio/Dataset-Stripped \
  --data-root /teamspace/studios/this_studio/Dataset-Stripped \
  --model vit3d \
  --modalities T1w T2w \
  --epochs 250 \
  --batch-size 4 \
  --output outputs/results

# NeuroJEPA — frozen backbone
python scripts/train_probe.py \
  --root-dir /teamspace/studios/this_studio/Dataset-Stripped \
  --data-root /teamspace/studios/this_studio/Dataset-Stripped \
  --model neurojepa \
  --modalities T1w T2w \
  --epochs 250 \
  --batch-size 4 \
  --output outputs/results

# BrainIAC — frozen backbone
python scripts/train_probe.py \
  --root-dir /teamspace/studios/this_studio/Dataset-Stripped \
  --data-root /teamspace/studios/this_studio/Dataset-Stripped \
  --model brainiac \
  --modalities T1w T2w \
  --epochs 250 \
  --batch-size 4 \
  --output outputs/results
```

## 3. Multi-Modal — T1w + T2w + FLAIR (subjects with all 3)

```bash
# ViT3D — end-to-end
python scripts/train_probe.py \
  --root-dir /teamspace/studios/this_studio/Dataset-Stripped \
  --data-root /teamspace/studios/this_studio/Dataset-Stripped \
  --model vit3d \
  --modalities T1w T2w FLAIR \
  --epochs 250 \
  --batch-size 4 \
  --output outputs/results

# NeuroJEPA — frozen backbone
python scripts/train_probe.py \
  --root-dir /teamspace/studios/this_studio/Dataset-Stripped \
  --data-root /teamspace/studios/this_studio/Dataset-Stripped \
  --model neurojepa \
  --modalities T1w T2w FLAIR \
  --epochs 250 \
  --batch-size 4 \
  --output outputs/results
```

## 4. All-Modality Comparison (T1w baseline for every model)

Same as Section 1 — run every model with `--modalities T1w`. This is the primary comparison.

## Execution Order (suggested)

| Priority | Run | Est. Time | Notes |
|----------|-----|-----------|-------|
| 1 | ViT3D T1w | ~7h | Fastest, establishes baseline |
| 2 | Primus T1w | ~9h | Largest model, batch-size 2 |
| 3 | NeuroJEPA T1w | ~7h | MoE real weights |
| 4 | NeuroVFM T1w | ~7h | Anisotropic patches |
| 5 | BrainIAC T1w | ~6h | Fastest frozen model |
| 6 | ViT3D T1w+T2w | ~12h | Multi-modal baseline |
| 7 | NeuroJEPA T1w+T2w | ~7h | Multi-modal frozen |
| 8 | BrainIAC T1w+T2w | ~6h | Multi-modal frozen |
| 9 | ViT3D T1w+T2w+FLAIR | ~12h | 3-channel |
| 10 | NeuroJEPA T1w+T2w+FLAIR | ~7h | 3-channel frozen |
