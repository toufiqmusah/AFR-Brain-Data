#!/usr/bin/env python3
"""
One-time script: scan raw NIfTI data, score quality for every candidate,
select the best per (subject, modality) via orientation priority + quality
tiebreak, and write a JSON manifest.

The manifest is consumed by NigerianBrainDataset(selection_manifest=...)
to make deterministic, reproducible selections without re-running scoring.

Usage:
    python scripts/compute_selection_manifest.py \
        --root-dir /teamspace/studios/this_studio/Dataset \
        --output outputs/selection_manifest.json
"""

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute selection manifest with quality-based tiebreaking"
    )
    parser.add_argument("--root-dir", default="/teamspace/studios/this_studio/Dataset")
    parser.add_argument("--output", default="outputs/selection_manifest.json")
    parser.add_argument(
        "--orientation-priority", nargs="+", default=["axial", "coronal", "sagittal"]
    )
    parser.add_argument("--exclude-gadolinium", action="store_true", default=True)
    parser.add_argument("--quality-n-slices", type=int, default=10)
    parser.add_argument("--quality-alpha", type=float, default=0.5)
    return parser.parse_args()


def load_volume(path: str) -> np.ndarray:
    import nibabel as nib

    nii = nib.load(path)
    vol = nii.get_fdata(dtype=np.float32)
    vol = np.nan_to_num(vol, nan=0.0, posinf=0.0, neginf=0.0)
    return vol


def main():
    args = parse_args()
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from data.dataset import scan_raw_dataset
    from data.quality import VolumeQualityScorer

    root = Path(args.root_dir)
    print(f"Scanning {root} ...")
    all_records = scan_raw_dataset(root)
    print(f"  {len(all_records)} raw records")

    scorer = VolumeQualityScorer(alpha=args.quality_alpha)
    orientation_rank = {o: i for i, o in enumerate(args.orientation_priority)}

    groups = defaultdict(list)
    for r in all_records:
        groups[(r["subject"], r["modality"])].append(r)

    selected = {}
    candidates_index = {}

    for (subj, mod), cands in sorted(groups.items()):
        # 1. Filter gadolinium
        if args.exclude_gadolinium and mod == "T1w":
            cands = [c for c in cands if not c["contrast"]]
        if not cands:
            continue

        # 2. Score quality on every candidate
        key = f"{subj}_{mod}"
        scored = []
        for c in cands:
            try:
                vol = load_volume(c["path"])
                quality = scorer.score_volume(vol, n_slices=args.quality_n_slices)
            except Exception:
                quality = None
            scored.append(
                {
                    "path": c["path"],
                    "subject": subj,
                    "session": c.get("session", 1),
                    "orientation": c["orientation"],
                    "contrast": c["contrast"],
                    "modality": mod,
                    "quality": quality,
                }
            )

        candidates_index[key] = scored

        # 3. Select best: within best-priority orientation, pick top quality
        by_orient = defaultdict(list)
        for s in scored:
            by_orient[s["orientation"]].append(s)

        # Find best orientation that has candidates
        available_orients = sorted(
            by_orient.keys(),
            key=lambda o: orientation_rank.get(o, 99),
        )
        if not available_orients:
            continue

        best_orient = available_orients[0]
        orient_cands = by_orient[best_orient]

        # Within orientation, pick best composite quality (or first if none scored)
        scored_cands = [s for s in orient_cands if s.get("quality") is not None]
        if scored_cands:
            scored_cands.sort(key=lambda s: s["quality"]["composite"], reverse=True)
            champ = scored_cands[0]
        else:
            champ = orient_cands[0]

        selected[key] = champ["path"]

        # 4. Tag each candidate with selected/reason
        champ_path = champ["path"]
        for s in scored:
            if s["path"] == champ_path:
                s["selected"] = True
                s["excluded_reason"] = None
            elif s["orientation"] == best_orient and s.get("quality") is not None:
                s["selected"] = False
                s["excluded_reason"] = "inferior_quality"
            else:
                s["selected"] = False
                s["excluded_reason"] = "orientation_priority"

    # Build output entries list
    selection_list = []
    for key, path in sorted(selected.items()):
        subj_str, mod = key.split("_", 1)
        selection_list.append({"subject": int(subj_str), "modality": mod, "selected_path": path})

    manifest = {
        "description": "Nigerian Brain Dataset — selection manifest",
        "source": str(root.resolve()),
        "created": datetime.now(timezone.utc).isoformat(),
        "config": {
            "orientation_priority": args.orientation_priority,
            "exclude_gadolinium": args.exclude_gadolinium,
            "quality_alpha": args.quality_alpha,
            "quality_n_slices": args.quality_n_slices,
        },
        "n_selected": len(selection_list),
        "selection": selection_list,
        "candidates": candidates_index,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nWrote {output_path}")
    print(f"  {len(selection_list)} subject-modality entries selected")
    print(f"  Candidates scored: {sum(len(v) for v in candidates_index.values())}")


if __name__ == "__main__":
    main()
