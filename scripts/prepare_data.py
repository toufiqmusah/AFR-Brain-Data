#!/usr/bin/env python3
"""
Comprehensive data preparation pipeline.

Scans raw Dataset, applies orientation/session selection, quality scoring,
resampling, and packages the prepared dataset for HuggingFace upload.

Usage:
    python scripts/prepare_data.py \
        --root-dir /teamspace/studios/this_studio/Dataset \
        --output-dir /teamspace/studios/this_studio/Prepared-Dataset \
        --target-size 96 112 96 \
        --orientation-priority axial coronal sagittal \
        --exclude-gadolinium
"""

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare and package the Nigerian Brain Dataset")
    parser.add_argument("--root-dir", default="/teamspace/studios/this_studio/Dataset")
    parser.add_argument("--output-dir", default="/teamspace/studios/this_studio/Prepared-Dataset")
    parser.add_argument("--participant-tsv", default=None)
    parser.add_argument("--target-size", nargs=3, type=int, default=[96, 112, 96])
    parser.add_argument("--orientation-priority", nargs="+", default=["axial", "coronal", "sagittal"])
    parser.add_argument("--exclude-gadolinium", action="store_true", default=True)
    parser.add_argument("--include-dwi", action="store_true", default=False)
    parser.add_argument("--quality-cache", default=None)
    parser.add_argument("--save-numpy", action="store_true", help="Save as .npy arrays instead")
    parser.add_argument("--run-quality", action="store_true", help="Run quality scoring during preparation")
    parser.add_argument("--quality-n-slices", type=int, default=10, help="Slices sampled per volume for quality")
    return parser.parse_args()


def main():
    args = parse_args()
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from data.dataset import NigerianBrainDataset, scan_raw_dataset, load_participant_tsv
    from data.transforms import eval_transform

    if args.run_quality:
        from data.quality import VolumeQualityScorer

    transform = eval_transform(target_size=tuple(args.target_size))
    modalities = ["T1w", "T2w", "FLAIR"]
    if args.include_dwi:
        modalities.append("DWI")

    all_records = scan_raw_dataset(Path(args.root_dir))
    tsv_path = args.participant_tsv or str(Path(args.root_dir) / "participant-info.tsv")
    labels = load_participant_tsv(tsv_path) if Path(tsv_path).exists() else {}
    labels_map = {0: "Control", 1: "Dementia", 2: "Parkinson"}

    dataset = NigerianBrainDataset(
        root_dir=args.root_dir,
        participant_tsv=args.participant_tsv,
        modalities=tuple(modalities),
        orientation_priority=tuple(args.orientation_priority),
        exclude_gadolinium=args.exclude_gadolinium,
        target_size=tuple(args.target_size),
        transform=transform,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    modality_map = {"T1w": 0, "T2w": 1, "FLAIR": 2, "DWI": 3}

    # Build a full curation manifest starting from all raw records
    curation_entries = []
    for rec in all_records:
        s = rec["subject"]
        curation_entries.append({
            "subject": s,
            "session": rec.get("session", 1),
            "path": rec["path"],
            "orientation": rec["orientation"],
            "modality": rec["modality"],
            "contrast": rec["contrast"],
            "site": rec.get("site", "unknown"),
            "field_strength": rec.get("field_strength", -1.0),
            "label": labels.get(s, -1),
            "label_name": labels_map.get(labels.get(s, -1), "Unknown"),
            "selected": False,
            "excluded_reason": "",
        })

    # Quality scoring on raw records if requested
    if args.run_quality and curation_entries:
        scorer = VolumeQualityScorer()
        for entry in tqdm(curation_entries, desc="Scoring quality"):
            try:
                vol = dataset._load_volume(entry["path"])
                score = scorer.score_volume(vol, n_slices=args.quality_n_slices)
                entry["quality"] = score
            except Exception:
                entry["quality"] = None

    # Mark why each raw record was included/excluded
    kept_subjects = {e["subject"] for e in dataset.index}
    excluded_count = 0
    for entry in curation_entries:
        if entry["subject"] not in kept_subjects:
            entry["selected"] = False
            if entry["label"] == -1:
                entry["excluded_reason"] = "no_tsv_label"
            elif entry["contrast"] and args.exclude_gadolinium:
                entry["excluded_reason"] = "gadolinium_excluded"
            else:
                entry["excluded_reason"] = "orientation_priority_or_missing_modality"
            excluded_count += 1

    # Mark which records were actually selected per subject+modality
    for idx_info in dataset.index:
        subj = idx_info["subject"]
        for mod, rec in idx_info["selected"].items():
            for entry in curation_entries:
                if entry["subject"] == subj and entry["path"] == rec["path"]:
                    entry["selected"] = True
                    entry["excluded_reason"] = ""

    manifest = {
        "description": "Prepared Nigerian Brain Dataset for FM probing",
        "source": str(Path(args.root_dir).resolve()),
        "target_size": list(args.target_size),
        "orientation_priority": args.orientation_priority,
        "exclude_gadolinium": args.exclude_gadolinium,
        "modalities": modalities,
        "n_raw_records": len(curation_entries),
        "n_subjects_with_labels": len(kept_subjects),
        "n_excluded_records": excluded_count,
        "quality_scored": args.run_quality,
        "label_map": labels_map,
        "curation": curation_entries,
    }

    # Save the volumes & metadata
    for idx in tqdm(range(len(dataset)), desc="Preparing volumes"):
        item = dataset[idx]
        subj = item["subject_id"]
        label = item["label"]
        label_name = labels_map.get(label, "Unknown")
        site = item["site"]
        field_strength = item["field_strength"]
        ce_gad = item["ce_gadolinium"]

        subj_dir = output_dir / f"sub-{subj:02d}"
        subj_dir.mkdir(parents=True, exist_ok=True)

        meta = {
            "subject": subj,
            "label": label,
            "label_name": label_name,
            "site": site,
            "field_strength": field_strength,
            "ce_gadolinium": ce_gad,
            "modalities": item["modalities_loaded"],
        }

        vol = item["volume"].cpu().numpy()
        if args.save_numpy:
            np.save(subj_dir / "volume.npy", vol)
        else:
            nifti_img = nib.Nifti1Image(vol, np.eye(4))
            nib.save(nifti_img, subj_dir / "volume.nii.gz")

        with open(subj_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nSaved {len(dataset)} subjects to {output_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"  Raw records: {manifest['n_raw_records']}")
    print(f"  Subjects kept: {manifest['n_subjects_with_labels']}")
    print(f"  Excluded: {manifest['n_excluded_records']}")


if __name__ == "__main__":
    main()
