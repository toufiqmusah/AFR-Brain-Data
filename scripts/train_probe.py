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
    parser.add_argument("--model", default="vit3d", choices=["neurojepa", "neurovfm", "brainiac", "primus", "vit3d", "dinov3"])
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


def _config_label(modalities):
    return "_".join(m.lower() for m in modalities)


def _build_backbone(args, modalities, full_dataset):
    weights = "real"
    if args.model == "neurojepa":
        from models.neurojepa import NeuroJEPABackbone
        model = NeuroJEPABackbone()
        try:
            model.from_pretrained()
        except Exception as e:
            print(f"  [NeuroJEPA] HF weights failed ({e}), using dummy")
            model.load_dummy()
            weights = "dummy"
    elif args.model == "neurovfm":
        from models.neurovfm import NeuroVFMBackbone
        model = NeuroVFMBackbone()
        try:
            model.load()
        except Exception as e:
            print(f"  [NeuroVFM] HF weights failed ({e}), using dummy")
            model.load_dummy()
            weights = "dummy"
    elif args.model == "brainiac":
        from models.brainiac import BrainIACBackbone
        model = BrainIACBackbone()
        try:
            model.from_pretrained()
        except Exception as e:
            print(f"  [BrainIAC] HF weights failed ({e}), using dummy")
            model.load_dummy()
            weights = "dummy"
    elif args.model == "primus":
        from models.primus import PrimusBackbone
        model = PrimusBackbone()
        try:
            model.from_pretrained()
        except Exception as e:
            print(f"  [Primus] HF weights failed ({e}), using dummy")
            model.load_dummy()
            weights = "dummy"
    elif args.model == "dinov3":
        from models.dinov3 import DINOv3Backbone
        model = DINOv3Backbone()
        try:
            model.from_pretrained()
        except Exception as e:
            print(f"  [DINOv3] HF weights failed ({e}), using dummy")
            model.load_dummy()
            weights = "dummy"
    elif args.model == "vit3d":
        from models.vit3d_baseline import ViT3D
        model = ViT3D(
            in_channels=len(modalities),
            n_classes=len(set(d["label"] for d in full_dataset.index)),
        )
        weights = "scratch"
    else:
        raise ValueError(f"Unknown model: {args.model}")
    model._weights = weights
    return model


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
    config = _config_label(modalities)

    # Resolve participant TSV
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

    full_dataset = NigerianBrainDataset(
        root_dir=args.root_dir,
        data_root=args.data_root,
        participant_tsv=tsv_path,
        modalities=tuple(modalities),
        selection_manifest=args.selection_manifest,
        transform=train_transform(),
    )

    if args.splits and Path(args.splits).exists():
        folds = load_splits(args.splits)
    else:
        folds = generate_splits(full_dataset.index, n_folds=args.n_folds, seed=args.seed)

    # Build backbone once for logging; re-built per fold for fresh training
    backbone_dummy = _build_backbone(args, modalities, full_dataset)
    train_backbone = args.model == "vit3d"
    n_params_backbone = sum(p.numel() for p in backbone_dummy.parameters())
    n_params_backbone_trainable = sum(p.numel() for p in backbone_dummy.parameters() if p.requires_grad)
    head_dummy = ProbingHead(backbone_dummy.hidden_dim, 3)
    n_params_head = sum(p.numel() for p in head_dummy.parameters())

    # ── Startup log ──
    backbone_mode = "end-to-end" if train_backbone else "frozen"
    print(f"\n{'='*60}")
    print(f"  Model:         {args.model} ({backbone_mode})")
    print(f"  Weights:       {getattr(backbone_dummy, '_weights', 'unknown')}")
    print(f"  Params:        {n_params_backbone:,} backbone", end="")
    if args.model == "dinov3":
        print(f" ({n_params_backbone_trainable:,} trainable adapters) + {n_params_head:,} head = {n_params_backbone_trainable + n_params_head:,} trainable")
    elif not train_backbone:
        print(f" (frozen) + {n_params_head:,} head = {n_params_head:,} trainable")
    else:
        print(f" + {n_params_head:,} head = {n_params_backbone + n_params_head:,} total")
    print(f"  Modalities:    {', '.join(modalities)} ({len(modalities)}-channel)")
    print(f"  Subjects:      {len(full_dataset)}")
    print(f"  Folds:         {len(folds)}-fold CV")
    print(f"  Epochs:        {args.epochs}")
    print(f"  Batch size:    {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Device:        {args.device}")
    print(f"  Output:        {args.output}/{args.model}/{config}")
    if args.data_root:
        print(f"  Data root:     {args.data_root}")
    print(f"{'='*60}\n")

    # ── Data integrity checks ──
    class_counts = {}
    for entry in full_dataset.index:
        lbl = entry["label"]
        class_counts[lbl] = class_counts.get(lbl, 0) + 1
    label_map_inv = {v: k for k, v in full_dataset.label_map.items()}
    print("  Class distribution (full dataset):")
    for lbl in sorted(class_counts):
        print(f"    {label_map_inv[lbl]:12s}: {class_counts[lbl]} subjects")

    # Per-fold stats
    for fold_idx, fold in enumerate(folds):
        train_subs = set(get_fold_split_ids(folds, fold_idx, "train"))
        test_subs = set(get_fold_split_ids(folds, fold_idx, "test"))
        overlap = train_subs & test_subs
        train_labels = [full_dataset.labels.get(s) for s in train_subs if full_dataset.labels.get(s) is not None]
        test_labels = [full_dataset.labels.get(s) for s in test_subs if full_dataset.labels.get(s) is not None]
        train_counts = {l: train_labels.count(l) for l in sorted(set(train_labels))}
        test_counts = {l: test_labels.count(l) for l in sorted(set(test_labels))}
        print(f"  Fold {fold_idx + 1}: train={len(train_subs)}, test={len(test_subs)}, "
              f"overlap={len(overlap)} (leakage={'YES ⚠️' if overlap else 'OK'})")
        for lbl in sorted(set(list(train_counts.keys()) + list(test_counts.keys()))):
            name = label_map_inv.get(lbl, f"Class {lbl}")
            print(f"    {name:12s}: train={train_counts.get(lbl, 0)} test={test_counts.get(lbl, 0)}")
    print(f"{'='*60}\n")

    all_fold_metrics = []

    # Pre-compute GradCAM subjects: same subjects visualized across all models
    gradcam_subjects = {}
    for fold_idx, fold in enumerate(folds):
        test_subs = get_fold_split_ids(folds, fold_idx, "test")
        subj_labels = {}
        for entry in full_dataset.index:
            if entry["subject"] in test_subs:
                subj_labels[entry["subject"]] = entry["label"]
        seen = set()
        selected = []
        for s in test_subs:
            lbl = subj_labels.get(s)
            if lbl is not None and lbl not in seen:
                selected.append(s)
                seen.add(lbl)
            if len(selected) == 4:
                break
        for s in test_subs:
            if s not in selected:
                selected.append(s)
            if len(selected) == 4:
                break
        gradcam_subjects[fold_idx] = selected

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

        backbone = _build_backbone(args, modalities, full_dataset).to(args.device)
        train_backbone = args.model == "vit3d"

        n_channels = len(modalities)
        if n_channels > 1 and hasattr(backbone, "adapt_patch_embed"):
            backbone.adapt_patch_embed(n_channels)

        hidden_dim = backbone.hidden_dim
        n_classes = 3

        head = ProbingHead(hidden_dim, n_classes).to(args.device)
        train_backbone = args.model == "vit3d"
        backbone_trainable = [p for p in backbone.parameters() if p.requires_grad]
        has_trainable_backbone = train_backbone or len(backbone_trainable) > 0
        if has_trainable_backbone:
            backbone.train()
            optimizer = torch.optim.AdamW(
                backbone_trainable + list(head.parameters()),
                lr=args.lr, weight_decay=1e-4,
            )
        else:
            backbone.eval()
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
            project=f"{args.model}_{config}",
            fold=fold_idx,
            train_backbone=train_backbone,
        )
        trainer.fit()

        from eval.evaluate import evaluate_fold
        metrics = evaluate_fold(backbone, head, test_loader, args.device, return_predictions=True)
        all_fold_metrics.append(metrics)
        print(f"  Test: acc={metrics['accuracy']:.3f}, macro_f1={metrics['macro_f1']:.3f}, mcc={metrics['mcc']:.3f}")

        # GradCAM visualizations
        try:
            from explainability.gradcam import gradcam_3d, gradcam_multimodal, gradcam_interaction, _get_features
            from explainability.visualize import plot_class_cams_grid
            cam_dir = Path(args.output) / args.model / config / "gradcam" / f"fold_{fold_idx}"
            label_names = {v: k for k, v in full_dataset.label_map.items()}

            # Build subject-to-index map for this fold's test set
            subj_to_idx = {}
            for idx in range(len(test_dataset)):
                s = test_dataset[idx]["subject_id"]
                s = s.item() if isinstance(s, torch.Tensor) else s
                subj_to_idx[s] = idx

            for subj in gradcam_subjects[fold_idx]:
                idx = subj_to_idx.get(subj)
                if idx is None:
                    continue
                sample = test_dataset[idx]
                vol = sample["volume"].unsqueeze(0).to(args.device)
                label = sample["label"].item() if isinstance(sample["label"], torch.Tensor) else sample["label"]

                # Get model prediction for the CAM class label
                with torch.no_grad():
                    feats = _get_features(backbone, vol)
                    if isinstance(feats, tuple):
                        feats = feats[0]
                    pred_class = head(feats.mean(dim=1) if feats.dim() == 3 else feats).argmax(dim=-1).item()

                if n_channels > 1:
                    result = gradcam_interaction(backbone, head, vol, n_channels)
                    if result[0] is None:
                        print("  [SKIP] GradCAM: model outputs pooled features only")
                        break
                    full_cam, per_channel, interaction = result
                    plot_class_cams_grid(
                        {pred_class: full_cam},
                        vol[0].mean(dim=0).cpu().numpy(), label_names,
                        save_path=str(cam_dir / f"sample_{subj}_full.png"), true_label=label,
                    )
                    for ch in range(n_channels):
                        plot_class_cams_grid(
                            {pred_class: per_channel[ch]},
                            vol[0, ch].cpu().numpy(), label_names,
                            save_path=str(cam_dir / f"sample_{subj}_ch{ch}_{modalities[ch].lower()}.png"), true_label=label,
                        )
                    plot_class_cams_grid(
                        {pred_class: interaction},
                        vol[0].mean(dim=0).cpu().numpy(), label_names,
                        save_path=str(cam_dir / f"sample_{subj}_interaction.png"), true_label=label,
                    )
                else:
                    cam = gradcam_3d(backbone, head, vol)
                    if cam is None:
                        print("  [SKIP] GradCAM: model outputs pooled features only")
                        break
                    plot_class_cams_grid(
                        {pred_class: cam},
                        vol[0, 0].cpu().numpy(), label_names,
                        save_path=str(cam_dir / f"sample_{subj}.png"), true_label=label,
                    )
        except Exception as e:
            print(f"  [WARN] GradCAM failed: {e}")

    summary = aggregate_fold_metrics(all_fold_metrics)
    output_path = Path(args.output) / args.model / f"{config}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"per_fold": all_fold_metrics, "summary": summary}, f, indent=2)
    print(f"\nResults saved to {output_path}")
    print(f"Summary: acc={summary.get('accuracy_mean', 'N/A'):.3f}±{summary.get('accuracy_std', 'N/A'):.3f}")


if __name__ == "__main__":
    main()
