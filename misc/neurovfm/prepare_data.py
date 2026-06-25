from __future__ import annotations
import argparse
import json
import math
import re
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torchio as tio
from torch.utils.data import WeightedRandomSampler

RANDOM_SEED = 42
LABEL_MAP = {"Control": 0, "Dementia": 1, "Parkinson": 2}
MODALITY_KEYWORDS = {"t1": "T1w", "t2": "T2w", "flair": "FLAIR"}


def load_labels(tsv_path: str | Path) -> dict[int, int]:
    demo = pd.read_csv(tsv_path, sep="\t")
    demo.columns = demo.columns.str.strip()
    label_col = "ClincalGroup"
    if label_col not in demo.columns:
        label_col = "ClinicalGroup"
    lookup = {}
    for _, row in demo.iterrows():
        group = str(row[label_col]).strip()
        lookup[int(row["Subject"])] = LABEL_MAP[group]
    return lookup


def infer_modality(dir_name: str) -> str | None:
    lowered = dir_name.lower()
    for key, mod in MODALITY_KEYWORDS.items():
        if key in lowered:
            return mod
    return None


def parse_series_description(info_path: Path) -> str | None:
    if not info_path.exists():
        return None
    with info_path.open() as f:
        info = json.load(f)
    return info.get("meta", {}).get("SeriesDescription", "") or None


def is_axial(series_desc: str) -> bool:
    return any(w.startswith("AX") for w in series_desc.upper().split())


def find_t1w_axial_scans(
    data_root: Path, label_lookup: dict[int, int]
) -> list[dict]:
    records = []
    for subj_dir in sorted(data_root.glob("sub-*")):
        subj_num = int(subj_dir.name.split("-")[1])
        label = label_lookup.get(subj_num)
        if label is None:
            continue

        for mod_dir in sorted(subj_dir.iterdir()):
            if not mod_dir.is_dir():
                continue
            modality = infer_modality(mod_dir.name)
            if modality != "T1w":
                continue

            orientation = parse_series_description(mod_dir / "_info.json")
            if orientation is None or not is_axial(orientation):
                continue

            nii_files = sorted(mod_dir.glob("*.nii*"))
            if not nii_files:
                continue

            try:
                img = nib.load(str(nii_files[0]))
                shape = img.shape[:3]
            except Exception:
                continue

            records.append({
                "subject_id": subj_dir.name,
                "subj_num": subj_num,
                "path": str(nii_files[0]),
                "shape_h": int(shape[0]),
                "shape_w": int(shape[1]),
                "shape_d": int(shape[2]),
                "diagnosis": label,
            })

    return records


def compute_target_size(df: pd.DataFrame) -> tuple[int, int, int]:
    med_h = int(df["shape_h"].median())
    med_w = int(df["shape_w"].median())
    med_d = int(df["shape_d"].median())
    round16 = lambda x: math.ceil(x / 16) * 16
    return (round16(med_h), round16(med_w), round16(med_d))


def build_tio_subjects(records: list[dict], target_size: tuple[int, int, int]):
    samples = []
    for r in records:
        subj = tio.Subject(
            img=tio.ScalarImage(r["path"]),
            diagnosis=r["diagnosis"],
            subject_id=r["subject_id"],
        )
        samples.append(subj)

    transform = tio.Compose([
        tio.RescaleIntensity(out_min_max=(0, 1)),
        tio.CropOrPad(target_size),
    ])

    subject_groups = {}
    for i, s in enumerate(samples):
        sid = s.subject_id
        if sid not in subject_groups:
            subject_groups[sid] = {"indices": [], "diagnosis": s.diagnosis}
        subject_groups[sid]["indices"].append(i)

    return samples, subject_groups, transform


def stratified_group_split(
    subject_groups: dict,
    test_ratio: float = 0.12,
    seed: int = RANDOM_SEED,
):
    group_ids = list(subject_groups.keys())
    group_labels = [subject_groups[sid]["diagnosis"] for sid in group_ids]
    classes = torch.unique(torch.tensor(group_labels))
    rng = torch.Generator().manual_seed(seed)

    test_sids, train_val_sids = [], []
    for c in classes:
        idx = [i for i, l in enumerate(group_labels) if l == c]
        perm = torch.tensor(idx)[torch.randperm(len(idx), generator=rng)].tolist()
        n_test = max(1, round(len(perm) * test_ratio))
        test_sids.extend([group_ids[i] for i in perm[:n_test]])
        train_val_sids.extend([group_ids[i] for i in perm[n_test:]])

    return test_sids, train_val_sids, group_ids, group_labels


def create_stratified_folds(
    train_val_sids: list,
    subject_groups: dict,
    n_folds: int = 5,
    seed: int = RANDOM_SEED,
):
    class_to_sids = {}
    for sid in train_val_sids:
        diag = subject_groups[sid]["diagnosis"]
        class_to_sids.setdefault(diag, []).append(sid)

    rng = torch.Generator().manual_seed(seed)
    folds = [set() for _ in range(n_folds)]
    for diag, sids in class_to_sids.items():
        perm = [sids[i] for i in torch.randperm(len(sids), generator=rng).tolist()]
        chunk = len(perm) // n_folds
        rem = len(perm) % n_folds
        start = 0
        for k in range(n_folds):
            end = start + chunk + (1 if k < rem else 0)
            folds[k].update(perm[start:end])
            start = end
    return folds


def build_fold_loaders(
    fold_val_sids: set,
    train_val_sids: list,
    subject_groups: dict,
    samples: list,
    transform: tio.Compose,
    batch_size: int = 8,
):
    train_sids = [sid for sid in train_val_sids if sid not in fold_val_sids]
    fold_train_idx = [
        i for sid in train_sids for i in subject_groups[sid]["indices"]
    ]
    fold_val_idx = [
        i for sid in fold_val_sids for i in subject_groups[sid]["indices"]
    ]

    train_labels = [samples[i].diagnosis for i in fold_train_idx]
    class_counts = torch.bincount(torch.tensor(train_labels))
    weights = 1.0 / class_counts.float()
    sampler = WeightedRandomSampler(
        weights[torch.tensor(train_labels)], len(fold_train_idx), replacement=True
    )

    train_loader = tio.SubjectsLoader(
        tio.SubjectsDataset([samples[i] for i in fold_train_idx], transform=transform),
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,
        pin_memory=False,
    )
    val_loader = tio.SubjectsLoader(
        tio.SubjectsDataset([samples[i] for i in fold_val_idx], transform=transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    return train_loader, val_loader, train_sids, fold_val_sids


def generate_manifest(
    df: pd.DataFrame,
    target_size: tuple[int, int, int],
    samples: list,
    subject_groups: dict,
    test_sids: list,
    train_val_sids: list,
    folds: list[set],
    label_map: dict,
    data_root: str,
) -> dict:
    rev_label = {v: k for k, v in label_map.items()}
    diag_counts = {}
    for sid, info in subject_groups.items():
        d = rev_label[info["diagnosis"]]
        diag_counts[d] = diag_counts.get(d, 0) + 1

    test_diag = {}
    for sid in test_sids:
        d = rev_label[subject_groups[sid]["diagnosis"]]
        test_diag[d] = test_diag.get(d, 0) + 1

    return {
        "data_root": data_root,
        "n_subjects": df["subject_id"].nunique(),
        "n_scans": len(df),
        "modalities": ["T1w"],
        "target_size": list(target_size),
        "tokens_per_scan": (target_size[0] // 16)
        * (target_size[1] // 16)
        * (target_size[2] // 16),
        "class_distribution": diag_counts,
        "n_test_subjects": len(test_sids),
        "n_train_val_subjects": len(train_val_sids),
        "test_class_distribution": test_diag,
        "n_folds": len(folds),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Prepare T1w axial brain MRI data for NeuroVFM training."
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="/kaggle/input/datasets/sparkedwakanda26/skull-stripped-mri/skull_stripped_data",
        help="Path to skull-stripped MRI dataset",
    )
    parser.add_argument(
        "--participant-tsv",
        type=str,
        default="/kaggle/input/datasets/sammydamz/participant-info/participant-info.tsv",
        help="Path to participant-info.tsv",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output",
        help="Output directory for splits and manifest",
    )
    parser.add_argument(
        "--n-folds", type=int, default=5, help="Number of CV folds"
    )
    parser.add_argument(
        "--test-ratio", type=float, default=0.12, help="Test hold-out ratio"
    )
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Dataloader batch size"
    )
    args = parser.parse_args()

    torch.manual_seed(RANDOM_SEED)
    data_root = Path(args.data_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading labels...")
    label_lookup = load_labels(args.participant_tsv)

    print("Scanning T1w axial scans...")
    records = find_t1w_axial_scans(data_root, label_lookup)
    df = pd.DataFrame(records)
    print(f"  Found {len(df)} scans from {df['subject_id'].nunique()} subjects")

    target_size = compute_target_size(df)
    print(f"  Target size: {target_size}")

    samples, subject_groups, transform = build_tio_subjects(records, target_size)

    test_sids, train_val_sids, _, _ = stratified_group_split(
        subject_groups, args.test_ratio
    )
    print(
        f"  Test: {len(test_sids)} subjects, Train/Val: {len(train_val_sids)} subjects"
    )

    folds = create_stratified_folds(train_val_sids, subject_groups, args.n_folds)

    fold_data = {}
    for k, val_sids in enumerate(folds):
        train_loader, val_loader, train_sids, _ = build_fold_loaders(
            val_sids,
            train_val_sids,
            subject_groups,
            samples,
            transform,
            args.batch_size,
        )
        train_diags = [subject_groups[sid]["diagnosis"] for sid in train_sids]
        val_diags = [subject_groups[sid]["diagnosis"] for sid in val_sids]
        train_ct = torch.bincount(torch.tensor(train_diags))
        val_ct = torch.bincount(torch.tensor(val_diags))
        print(
            f"  Fold {k+1}: train={len(train_sids)} subj ({train_ct.tolist()}), "
            f"val={len(val_sids)} subj ({val_ct.tolist()})"
        )
        fold_data[f"fold_{k+1}"] = {
            "train_subjects": train_sids,
            "val_subjects": list(val_sids),
        }

    test_idx = [
        i for sid in test_sids for i in subject_groups[sid]["indices"]
    ]
    test_ds = tio.SubjectsDataset([samples[i] for i in test_idx], transform=transform)
    test_loader = tio.SubjectsLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=False,
    )
    print(f"  Test loader: {len(test_loader)} batches")

    manifest = generate_manifest(
        df=df,
        target_size=target_size,
        samples=samples,
        subject_groups=subject_groups,
        test_sids=test_sids,
        train_val_sids=train_val_sids,
        folds=folds,
        label_map=LABEL_MAP,
        data_root=str(data_root),
    )

    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest saved to {manifest_path}")

    subject_paths = {r["subject_id"]: r["path"] for r in records}

    splits_path = out_dir / "splits.json"
    splits = {
        "data_root": str(data_root),
        "target_size": list(target_size),
        "test_subjects": test_sids,
        "train_val_subjects": train_val_sids,
        "subject_paths": subject_paths,
        "subject_diagnoses": {sid: subject_groups[sid]["diagnosis"] for sid in subject_groups},
        "folds": fold_data,
    }
    with open(splits_path, "w") as f:
        json.dump(splits, f, indent=2)
    print(f"Splits saved to {splits_path}")

    print("\nDone.")
    return manifest, splits


if __name__ == "__main__":
    main()
