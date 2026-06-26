"""Linear-probe benchmark for the BrainIAC foundation model.

Freeze the pretrained ViT, pull out CLS features, and fit a logistic regression
to separate Control / Dementia / Parkinson. Scans are skull-stripped already;
here we just resize to 96^3 and z-score the foreground before the encoder.

Set MODALITIES to one key for single-modality, or stack a few (T1w/T2w/FLAIR)
to feed the encoder multiple channels at once.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchio as tio
from huggingface_hub import hf_hub_download
from monai.networks.nets import ViT
from safetensors.torch import load_file
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split

DATA_ROOT = "/kaggle/input/datasets/sparkedwakanda26/skull-stripped-mri/skull_stripped_data"
TSV_PATH = "/kaggle/input/datasets/tobi24/participant-tsv/participant-info.tsv"

LABEL_MAP = {"Control": 0, "Dementia": 1, "Parkinson": 2}
MODALITIES = ["dt-neuro-anat-t2w"]  # add "...-t1w", "...-flair" to stack channels
TARGET_SIZE = (96, 96, 96)
SEED = 42


class BrainIAC(nn.Module):
    """Frozen BrainIAC encoder, used as a CLS-token feature extractor.

    3D ViT-Base: patch (16,16,16), embed dim 768, 12 layers, pretrained on T1w
    MRI. It was trained on a single channel, so when we stack several modalities
    the patch-embedding kernel is spread across the inputs (repeat / n_channels,
    i.e. the original kernel applied to the channel mean) and every other weight
    is kept as-is.

    HF model: eugenehp/brainiac

        model = BrainIAC.from_pretrained(in_channels=len(MODALITIES)).to(device)
        feats = model(volume)            # [B, 768]
    """

    def __init__(self, in_channels=1):
        super().__init__()
        self.vit = ViT(
            in_channels=in_channels,
            img_size=TARGET_SIZE,
            patch_size=(16, 16, 16),
            hidden_size=768,
            mlp_dim=3072,
            num_layers=12,
            num_heads=12,
        )

    @classmethod
    def from_pretrained(cls, in_channels=1, repo="eugenehp/brainiac"):
        model = cls(in_channels)
        weights = load_file(hf_hub_download(repo_id=repo, filename="backbone.safetensors"))

        if in_channels > 1:
            kernel = weights.pop("patch_embedding.patch_embeddings.weight")
            kernel = kernel.repeat(1, in_channels, 1, 1, 1) / in_channels
            model.vit.load_state_dict(weights, strict=False)
            with torch.no_grad():
                model.vit.patch_embedding.patch_embeddings.weight.copy_(kernel)
        else:
            model.vit.load_state_dict(weights, strict=False)

        return model.eval().requires_grad_(False)

    @torch.no_grad()
    def forward(self, x):
        return self.vit(x)[0][:, 0]  # CLS token of the last hidden state


def read_labels(tsv_path):
    df = pd.read_csv(tsv_path, sep="\t")
    df.columns = df.columns.str.strip()
    col = "ClincalGroup" if "ClincalGroup" in df.columns else "ClinicalGroup"
    return {int(r["Subject"]): LABEL_MAP[r[col]] for _, r in df.iterrows()}


def zscore_foreground(volume):
    """Per-channel z-score over nonzero voxels; background stays at 0."""
    out = volume.clone()
    for c in range(out.shape[0]):
        fg = out[c] > 0
        if fg.any() and out[c][fg].std() > 0:
            out[c][fg] = (out[c][fg] - out[c][fg].mean()) / out[c][fg].std()
    return out


def load_subjects(modalities):
    """One tio.Subject per participant, image = stacked, preprocessed modalities."""
    labels = read_labels(TSV_PATH)
    resize = tio.Resize(TARGET_SIZE)
    subjects = []

    for subj_dir in sorted(Path(DATA_ROOT).glob("sub-*")):
        label = labels.get(int(subj_dir.name.split("-")[1]))
        if label is None:
            continue

        # map modality key -> nifti path (dropping the random ".id-XXXX" suffix)
        available = {}
        for mod_dir in sorted(p for p in subj_dir.iterdir() if p.is_dir()):
            nii = sorted(mod_dir.glob("*.nii"))
            if nii:
                available[re.sub(r"\.id-[^.]+$", "", mod_dir.name)] = nii[0]

        channels = []
        for mod in modalities:
            if mod in available:
                channels.append(resize(tio.ScalarImage(str(available[mod]))).data.float())
            else:
                channels.append(torch.zeros(1, *TARGET_SIZE))  # zero-fill missing
        if all(ch.eq(0).all() for ch in channels):
            continue

        volume = zscore_foreground(torch.cat(channels))
        subjects.append(tio.Subject(img=tio.ScalarImage(tensor=volume), label=label))

    return subjects


def extract_features(model, subjects, device, batch_size=8):
    loader = tio.SubjectsLoader(
        tio.SubjectsDataset(subjects), batch_size=batch_size, num_workers=0
    )
    feats, labels = [], []
    for batch in loader:
        feats.append(model(batch["img"][tio.DATA].to(device)).cpu())
        labels.append(batch["label"])
    return torch.cat(feats).numpy(), torch.cat(labels).numpy()


def fit_probe(X, y):
    return LogisticRegression(max_iter=2000, class_weight="balanced").fit(X, y)


def cross_validate(model, subjects, device, n_folds=5):
    y = np.array([s.label for s in subjects])
    folds = StratifiedKFold(n_folds, shuffle=True, random_state=SEED)
    scores = []

    for k, (tr, va) in enumerate(folds.split(subjects, y), 1):
        X_tr, y_tr = extract_features(model, [subjects[i] for i in tr], device)
        X_va, y_va = extract_features(model, [subjects[i] for i in va], device)
        pred = fit_probe(X_tr, y_tr).predict(X_va)
        ba = balanced_accuracy_score(y_va, pred)
        f1 = f1_score(y_va, pred, average="macro")
        scores.append((ba, f1))
        print(f"fold {k}: balanced_acc={ba:.3f} macro_f1={f1:.3f}")

    scores = np.array(scores)
    print(
        f"cv mean: balanced_acc={scores[:, 0].mean():.3f}±{scores[:, 0].std():.3f} "
        f"macro_f1={scores[:, 1].mean():.3f}±{scores[:, 1].std():.3f}"
    )
    return scores


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    subjects = load_subjects(MODALITIES)
    print(f"{len(subjects)} subjects, {len(MODALITIES)} channel(s)")

    model = BrainIAC.from_pretrained(in_channels=len(MODALITIES)).to(device)

    y = [s.label for s in subjects]
    trainval, test = train_test_split(subjects, test_size=0.12, stratify=y, random_state=SEED)

    cross_validate(model, trainval, device)

    X_tr, y_tr = extract_features(model, trainval, device)
    X_te, y_te = extract_features(model, test, device)
    pred = fit_probe(X_tr, y_tr).predict(X_te)
    print(
        f"held-out test ({len(test)} subjects): "
        f"balanced_acc={balanced_accuracy_score(y_te, pred):.3f} "
        f"macro_f1={f1_score(y_te, pred, average='macro'):.3f}"
    )


if __name__ == "__main__":
    main()
