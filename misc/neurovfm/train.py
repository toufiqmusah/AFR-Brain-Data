from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from model.neurovfm import NeuroVFMBackbone, NeuroVFMClassifier


def main():
    parser = argparse.ArgumentParser(
        description="Attention-pooling linear probe for NeuroVFM."
    )
    parser.add_argument("--splits", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="./output")
    parser.add_argument("--model-id", type=str, default="mlinslab/neurovfm-encoder")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--n-epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device

    with open(args.splits) as f:
        splits = json.load(f)

    subject_paths: dict[str, str] = splits["subject_paths"]
    subject_diag: dict[str, int] = splits["subject_diagnoses"]
    n_folds = len(splits["folds"])

    if args.data_root:
        root = Path(args.data_root)
        for sid in subject_paths:
            parts = Path(subject_paths[sid]).parts
            subj_idx = next(i for i, p in enumerate(parts) if p.startswith("sub-"))
            subject_paths[sid] = str(root / Path(*parts[subj_idx:]))

    subject_ids = sorted(subject_paths.keys())
    print(f"Subjects: {len(subject_ids)}, folds: {n_folds}")

    # embed all scans once, cache raw tokens (not pooled)
    cache_path = out_dir / "raw_tokens.pt"
    if cache_path.exists():
        print("Loading cached raw tokens...")
        cache = torch.load(cache_path)
        all_tokens: list[torch.Tensor] = cache["tokens"]
        all_labels: list[int] = cache["labels"]
        subject_ids = cache["subject_ids"]
    else:
        print("Loading NeuroVFM backbone...")
        backbone = NeuroVFMBackbone().load(args.model_id, device=device)
        print("Extracting raw tokens for all subjects...")
        all_tokens, all_labels = [], []
        for sid in subject_ids:
            path = subject_paths[sid]
            batch = backbone._preprocessor.load_study([path], modality="mri")
            tokens = backbone(batch)  # [N_tokens, 768]
            all_tokens.append(tokens.cpu())
            all_labels.append(subject_diag[sid])
        torch.save(
            {"tokens": all_tokens, "labels": all_labels, "subject_ids": subject_ids},
            cache_path,
        )
        print(f"Cached {len(all_tokens)} raw token sequences to {cache_path}")

    idx_map = {sid: i for i, sid in enumerate(subject_ids)}

    all_results = []
    for k in range(n_folds):
        fold_key = f"fold_{k + 1}"
        fold = splits["folds"][fold_key]

        train_idx = [idx_map[s] for s in fold["train_subjects"]]
        val_idx = [idx_map[s] for s in fold["val_subjects"]]

        classifier = NeuroVFMClassifier(hidden_dim=768, n_classes=3, dropout=0.3).to(device)
        optimizer = optim.AdamW(classifier.parameters(), lr=args.lr)
        criterion = nn.CrossEntropyLoss()

        best_val_acc = 0.0
        for epoch in range(1, args.n_epochs + 1):
            classifier.train()
            train_correct, train_total = 0, 0
            for i in train_idx:
                tokens = all_tokens[i].to(device)
                label = torch.tensor([all_labels[i]], device=device)

                optimizer.zero_grad()
                logits = classifier(tokens)  # [1, 3]
                loss = criterion(logits, label)
                loss.backward()
                optimizer.step()

                train_correct += (logits.argmax(dim=-1) == label).sum().item()
                train_total += 1

            # validate
            classifier.eval()
            val_correct, val_total = 0, 0
            with torch.no_grad():
                for i in val_idx:
                    tokens = all_tokens[i].to(device)
                    label = int(all_labels[i])
                    logits = classifier(tokens)
                    val_correct += (logits.argmax(dim=-1).item() == label)
                    val_total += 1

            train_acc = train_correct / train_total
            val_acc = val_correct / val_total

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(classifier.state_dict(), out_dir / f"fold{k+1}_best.pt")

            print(
                f"  Fold {k+1} | Epoch {epoch:2d}/{args.n_epochs} | "
                f"train acc {train_acc:.4f} | val acc {val_acc:.4f}"
            )

        all_results.append(best_val_acc)
        print(f"  -> Fold {k+1} best val acc: {best_val_acc:.4f}")

    results = {
        "per_fold_val_acc": all_results,
        "mean": float(torch.tensor(all_results).mean()),
        "std": float(torch.tensor(all_results).std()),
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(out_dir / "results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fold", "val_acc"])
        for k, acc in enumerate(all_results):
            w.writerow([k + 1, round(acc, 4)])

    print(f"\nMean +/- std: {results['mean']:.4f} +/- {results['std']:.4f}")
    print(f"Saved to {out_dir / 'results.json'}")
    print(f"Cache file: {cache_path} (delete to re-extract)")


if __name__ == "__main__":
    main()
