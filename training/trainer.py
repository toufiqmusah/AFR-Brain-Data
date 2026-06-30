import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Callable
from torch.utils.data import DataLoader
from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        head: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Callable] = None,
        loss_fn: Callable = nn.CrossEntropyLoss(label_smoothing=0.1),
        device: str = "cuda",
        max_epochs: int = 50,
        patience: int = 15,
        monitor: str = "val_loss",
        mode: str = "min",
        save_dir: str = "outputs/checkpoints",
        project: str = "default",
        fold: int = 0,
    ):
        self.model = model
        self.head = head
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.device = device
        self.max_epochs = max_epochs
        self.patience = patience
        self.monitor = monitor
        self.mode = mode
        self.fold = fold

        self.save_dir = Path(save_dir) / project
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.best_score = -float("inf") if mode == "max" else float("inf")
        self.best_epoch = 0
        self.epochs_without_improvement = 0
        self.history = []

    def train_epoch(self) -> Dict:
        self.model.eval()
        self.head.train()
        total_loss = 0.0
        all_preds, all_labels = [], []
        for batch in tqdm(self.train_loader, desc="  Train", leave=False):
            x = batch["volume"].to(self.device)
            labels = batch["label"].to(self.device).long()
            self.optimizer.zero_grad()
            with torch.no_grad():
                features = self.model(x)
            if isinstance(features, (tuple, list)):
                features = features[0]
            logits = self.head(features)
            loss = self.loss_fn(logits, labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * x.size(0)
            all_preds.extend(logits.argmax(dim=-1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        avg_loss = total_loss / len(self.train_loader.dataset)
        return {"loss": avg_loss}

    @torch.no_grad()
    def val_epoch(self) -> Dict:
        self.model.eval()
        self.head.eval()
        total_loss = 0.0
        all_preds, all_labels, all_probs = [], [], []
        for batch in tqdm(self.val_loader, desc="  Val", leave=False):
            x = batch["volume"].to(self.device)
            labels = batch["label"].to(self.device).long()
            features = self.model(x)
            if isinstance(features, (tuple, list)):
                features = features[0]
            logits = self.head(features)
            loss = self.loss_fn(logits, labels)
            total_loss += loss.item() * x.size(0)
            all_preds.extend(logits.argmax(dim=-1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
        avg_loss = total_loss / len(self.val_loader.dataset)
        all_probs = np.concatenate(all_probs, axis=0)
        return {"loss": avg_loss, "preds": all_preds, "labels": all_labels, "probs": all_probs}

    def fit(self) -> nn.Module:
        epoch_iter = tqdm(range(self.max_epochs), desc=f"Fold {self.fold}")
        for epoch in epoch_iter:
            train_metrics = self.train_epoch()
            val_metrics = self.val_epoch()
            score = val_metrics["loss"] if self.monitor == "val_loss" else -val_metrics["loss"]

            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler._LRScheduler):
                    self.scheduler.step()

            improvement = (
                score < self.best_score if self.mode == "min" else score > self.best_score
            )
            if improvement:
                self.best_score = score
                self.best_epoch = epoch
                self.epochs_without_improvement = 0
                self._save_checkpoint(epoch, val_metrics)
            else:
                self.epochs_without_improvement += 1

            self.history.append({"epoch": epoch, "train_loss": train_metrics["loss"], "val_loss": val_metrics["loss"]})
            epoch_iter.set_postfix(train_loss=train_metrics["loss"], val_loss=val_metrics["loss"])

            if self.epochs_without_improvement >= self.patience:
                break

        return self.load_best()

    def _save_checkpoint(self, epoch: int, val_metrics: Dict):
        path = self.save_dir / f"fold_{self.fold}_best.pt"
        torch.save({
            "head_state_dict": self.head.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": epoch,
            "val_metrics": val_metrics,
            "fold": self.fold,
        }, path)

    def load_best(self) -> nn.Module:
        path = self.save_dir / f"fold_{self.fold}_best.pt"
        if path.exists():
            ckpt = torch.load(path, map_location=self.device, weights_only=True)
            self.head.load_state_dict(ckpt["head_state_dict"])
        return self.head
