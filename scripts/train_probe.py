#!/usr/bin/env python3
"""
Main training entry point for probing experiments.

Usage:
    python scripts/train_probe.py \
        --root-dir /teamspace/studios/this_studio/Dataset \
        --selection-manifest outputs/selection_manifest.json \
        --splits outputs/splits/splits.json \
        --model vit3d \
        --modalities T1w T1c T2w \
        --epochs 50
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", default="/teamspace/studios/this_studio/Dataset",
                        help="Dataset root directory with NIfTI data")
    parser.add_argument("--data-root", default=None,
                        help="Volume directory (e.g. skull-stripped); defaults to --root-dir")
    parser.add_argument("--selection-manifest", default="outputs/selection_manifest.json",
                        help="Pre-computed selection manifest JSON")
    parser.add_argument("--splits", default="outputs/splits/splits.json",
                        help="Pre-computed fold splits JSON")
    parser.add_argument("--participant-tsv", default=None,
                        help="Path to participant-info.tsv; defaults to --root-dir or script parent dir")
    parser.add_argument("--model", default="vit3d", choices=["neurojepa", "neurovfm", "brainiac", "primus", "vit3d"])
    parser.add_argument("--config", default="t1w", help="Configuration label (t1w, t2w, flair, t1_t2, t1_t2_flair)")
    parser.add_argument("--modalities", nargs="+", default=["T1w"])
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="outputs/results")
    parser.add_argument("--checkpoint-dir", default="outputs/checkpoints")
    parser.add_argument("--resume", default=None)
    return parser.parse_args()


def _resolve_modalities(args):
    return args.modalities


def main():
    args = parse_args()
    from data.dataset import NigerianBrainDataset, collate_fn
    from data.transforms import train_transform, eval_transform
    from data.splits import generate_splits, get_split_index, get_fold_split_ids, load_splits
    from eval.metrics import compute_metrics, aggregate_fold_metrics
    from training.trainer import Trainer
    from training.scheduler import cosine_with_warmup
    from models.heads import ProbingHead

    modalities = _resolve_modalities(args)

    # Resolve participant TSV: check root dir first, then script parent dir
    if args.participant_tsv:
        tsv_path = args.participant_tsv
    else:
        tsv_candidates = [
            Path(args.root_dir) / "participant-info.tsv",
            Path(__file__).parent.parent / "participant-info.tsv",
        ]
        tsv_path = None
        for p in tsv_candidates:
            if p.exists():
                tsv_path = str(p)
                break

    print(f"Model: {args.model}")
    print(f"Config: {args.config}")
    print(f"Modalities: {modalities}")
    print(f"Root dir: {args.root_dir}")
    if args.data_root:
        print(f"Data root: {args.data_root}")
    print(f"Selection manifest: {args.selection_manifest}")
    print(f"Participant TSV: {tsv_path}")
    print(f"Splits: {args.splits}")

    transform = train_transform()

    full_dataset = NigerianBrainDataset(
        root_dir=args.root_dir,
        data_root=args.data_root,
        participant_tsv=tsv_path,
        modalities=tuple(modalities),
        selection_manifest=args.selection_manifest,
        transform=transform,
    )
    print(f"Full dataset: {len(full_dataset)} subjects")

    if args.splits and Path(args.splits).exists():
        folds = load_splits(args.splits)
        print(f"Loaded {len(folds)} pre-computed folds from {args.splits}")
    else:
        folds = generate_splits(full_dataset.index, n_folds=args.n_folds, seed=args.seed)
        print(f"Generated {len(folds)} stratified folds ({len(full_dataset.index)} subjects)")

    all_fold_metrics = []

    for fold_idx, fold in enumerate(folds):
        print(f"\n{'='*50}")
        print(f"Fold {fold_idx + 1}/{len(folds)}")
        print(f"{'='*50}")

        train_ids = get_fold_split_ids(folds, fold_idx, "train")
        test_ids = get_fold_split_ids(folds, fold_idx, "test")

        train_dataset = NigerianBrainDataset(
            root_dir=args.root_dir,
            data_root=args.data_root,
            participant_tsv=tsv_path,
            modalities=tuple(modalities),
            selection_manifest=args.selection_manifest,
            split_ids=train_ids,
            transform=train_transform(),
        )
        test_dataset = NigerianBrainDataset(
            root_dir=args.root_dir,
            data_root=args.data_root,
            participant_tsv=tsv_path,
            modalities=tuple(modalities),
            selection_manifest=args.selection_manifest,
            split_ids=test_ids,
            transform=eval_transform(),
        )

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

        if args.model == "neurojepa":
            from models.neurojepa import NeuroJEPABackbone
            backbone = NeuroJEPABackbone()
            try:
                backbone.from_pretrained()
            except Exception:
                print("  [WARN] Using dummy Neuro-JEPA")
                backbone.load_dummy()
        elif args.model == "neurovfm":
            from models.neurovfm import NeuroVFMBackbone
            backbone = NeuroVFMBackbone()
            try:
                backbone.load()
            except Exception:
                print("  [WARN] Using dummy NeuroVFM")
                backbone.load_dummy()
        elif args.model == "brainiac":
            from models.brainiac import BrainIACBackbone
            backbone = BrainIACBackbone()
            try:
                backbone.from_pretrained()
            except Exception:
                print("  [WARN] Using dummy BrainIAC")
                backbone.load_dummy()
        elif args.model == "primus":
            from models.primus import PrimusBackbone
            backbone = PrimusBackbone()
            try:
                backbone.from_pretrained()
            except Exception:
                print("  [WARN] Using dummy Primus")
                backbone.load_dummy()
        elif args.model == "vit3d":
            from models.vit3d_baseline import ViT3D
            backbone = ViT3D(
                in_channels=len(modalities),
                n_classes=len(set(d["label"] for d in full_dataset.index)),
            )
        else:
            raise ValueError(f"Unknown model: {args.model}")

        backbone = backbone.to(args.device)
        backbone.eval()

        n_channels = len(modalities)
        if n_channels > 1 and hasattr(backbone, "adapt_patch_embed"):
            backbone.adapt_patch_embed(n_channels)

        hidden_dim = backbone.hidden_dim
        n_classes = 3

        head = ProbingHead(hidden_dim, n_classes).to(args.device)
        optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = cosine_with_warmup(optimizer, warmup_epochs=5, total_epochs=args.epochs)

        trainer = Trainer(
            model=backbone,
            head=head,
            train_loader=train_loader,
            val_loader=test_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=args.device,
            max_epochs=args.epochs,
            save_dir=args.checkpoint_dir,
            project=f"{args.model}_{args.config}",
            fold=fold_idx,
        )
        trainer.fit()

        from eval.evaluate import evaluate_fold
        metrics = evaluate_fold(backbone, head, test_loader, args.device, return_predictions=True)
        all_fold_metrics.append(metrics)
        print(f"  Test: acc={metrics['accuracy']:.3f}, macro_f1={metrics['macro_f1']:.3f}, mcc={metrics['mcc']:.3f}")

    summary = aggregate_fold_metrics(all_fold_metrics)
    output_path = Path(args.output) / args.model / f"{args.config}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"per_fold": all_fold_metrics, "summary": summary}, f, indent=2)
    print(f"\nResults saved to {output_path}")
    print(f"Summary: acc={summary.get('accuracy_mean', 'N/A'):.3f}±{summary.get('accuracy_std', 'N/A'):.3f}")


if __name__ == "__main__":
    main()
