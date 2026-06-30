import json
import numpy as np
from sklearn.model_selection import StratifiedKFold
from pathlib import Path
from typing import Dict, List, Optional, Tuple


DEFAULT_OUTPUT_DIR = Path("outputs/splits")


def build_stratification_labels(index: List[dict]) -> Tuple[List[int], List[int], List[str]]:
    subjects = []
    labels = []
    sites = []
    seen = set()
    for entry in index:
        s = entry["subject"]
        if s not in seen:
            seen.add(s)
            subjects.append(s)
            labels.append(entry["label"])
            sites.append(entry["site"])
    return subjects, labels, sites


def generate_splits(
    index: List[dict],
    n_folds: int = 5,
    seed: int = 42,
    output_dir: Optional[str] = None,
) -> List[Dict]:
    subjects, labels, sites = build_stratification_labels(index)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = []
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(subjects, labels)):
        train_subs = [subjects[i] for i in train_idx]
        test_subs = [subjects[i] for i in test_idx]
        fold = {
            "fold": fold_idx,
            "train_subjects": train_subs,
            "test_subjects": test_subs,
            "train_labels": [labels[i] for i in train_idx],
            "test_labels": [labels[i] for i in test_idx],
            "train_sites": [sites[i] for i in train_idx],
            "test_sites": [sites[i] for i in test_idx],
        }
        folds.append(fold)

    if output_dir:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for fold in folds:
            with open(out_dir / f"fold_{fold['fold']}.json", "w") as f:
                json.dump(fold, f, indent=2)

    return folds


def load_fold(fold_path: str) -> dict:
    with open(fold_path) as f:
        return json.load(f)


def get_fold_split_ids(folds: List[dict], fold_idx: int, key: str = "train") -> List[int]:
    return folds[fold_idx][f"{key}_subjects"]


def get_split_index(index: List[dict], subject_ids: List[int]) -> List[dict]:
    sid_set = set(subject_ids)
    return [entry for entry in index if entry["subject"] in sid_set]
