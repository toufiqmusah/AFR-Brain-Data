#!/usr/bin/env python3
"""
Generate and save fixed 5-fold cross-validation splits.

Usage:
    python scripts/compute_splits.py \
        --root-dir /teamspace/studios/this_studio/Dataset \
        --selection-manifest outputs/selection_manifest.json \
        --modalities T1w T1c T2 \
        --output outputs/splits/splits.json

This saves a single JSON file containing all 5 folds with train/test
subject lists, reproducible with a fixed seed.  Load with:

    from data.splits import load_splits
    folds = load_splits("outputs/splits/splits.json")
    train_ids = folds[0]["train_subjects"]
    test_ids  = folds[0]["test_subjects"]
"""

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate and save fixed 5-fold CV splits"
    )
    parser.add_argument("--root-dir", default="/teamspace/studios/this_studio/Dataset")
    parser.add_argument("--selection-manifest", default="outputs/selection_manifest.json")
    parser.add_argument("--modalities", nargs="+", default=["T1w", "T2w", "FLAIR"])
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="outputs/splits/splits.json")
    return parser.parse_args()


def main():
    args = parse_args()
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from data.dataset import NigerianBrainDataset
    from data.splits import generate_splits, build_stratification_labels

    dataset = NigerianBrainDataset(
        root_dir=args.root_dir,
        modalities=tuple(args.modalities),
        selection_manifest=args.selection_manifest,
    )

    subjects, labels, sites = build_stratification_labels(dataset.index)
    print(f"Dataset: {len(dataset)} entries from {len(subjects)} unique subjects")
    print(f"Label distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")

    folds = generate_splits(dataset.index, n_folds=args.n_folds, seed=args.seed)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "root_dir": args.root_dir,
            "selection_manifest": args.selection_manifest,
            "modalities": args.modalities,
            "n_folds": args.n_folds,
            "seed": args.seed,
            "n_subjects": len(subjects),
            "folds": folds,
        }, f, indent=2)

    print(f"\nSaved {args.n_folds}-fold splits to {output_path}")
    for fold in folds:
        n_train = len(fold["train_subjects"])
        n_test = len(fold["test_subjects"])
        print(f"  Fold {fold['fold']}: train={n_train}, test={n_test}")


if __name__ == "__main__":
    import numpy as np
    main()
