import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

RAW_DATA_ROOT = Path("/teamspace/studios/this_studio/Dataset")
DEFAULT_LABEL_MAP = {"Control": 0, "Dementia": 1, "Parkinson": 2}
ORIENTATION_PRIORITY = {"axial": 0, "coronal": 1, "sagittal": 2}

SESSION_RE = re.compile(r"^sub-(?P<subject>\d+)(?:\.ses-run-(?P<session>\d+))?$")
DESC_RE = re.compile(
    r"sub-(?P<subject>\d+?)"
    r"(?:_acq-(?P<orientation>axial|coronal|sagittal))?"
    r"(?:_(?P<contrast>ce-gadolinium))?"
    r"(?:_dir-(?P<dir>\w+))?"
    r"(?:_run-(?P<run>\d+))?"
    r"_(?P<modality>T1w|T2w|FLAIR|DWI|dwi)"
    r"\.nii\.gz"
)


def _parse_session_dir(dirname: str) -> Optional[Tuple[str, Optional[int]]]:
    m = SESSION_RE.match(dirname)
    if m is None:
        return None
    subject = m.group("subject")
    session = int(m.group("session")) if m.group("session") else None
    return subject, session


MODALITY_NORM = {"t1w": "T1w", "t2w": "T2w", "flair": "FLAIR", "dwi": "DWI"}


def _parse_desc(desc: str) -> Optional[dict]:
    m = DESC_RE.match(desc)
    if m is None:
        return None
    raw_mod = m.group("modality")
    modality = MODALITY_NORM.get(raw_mod.lower(), raw_mod.upper())
    return {
        "subject": m.group("subject"),
        "session": int(m.group("run")) if m.group("run") else None,
        "orientation": m.group("orientation") or "unknown",
        "contrast": bool(m.group("contrast")),
        "modality": modality,
    }


def _extract_site_from_meta(meta: dict) -> str:
    inst = (meta.get("InstitutionName") or "").lower()
    if "intercontinental" in inst:
        return "intercontinental"
    if "life bridge" in inst or "lifebridge" in inst:
        return "lifebridge"
    if "rsuth" in inst or "braithwaite" in inst or "port harcourt" in inst:
        return "rusth"
    manufacturer = (meta.get("Manufacturer") or "").lower()
    if manufacturer == "ge":
        return "rusth"
    return "unknown"


def _extract_field_strength_from_meta(meta: dict) -> float:
    raw = meta.get("MagneticFieldStrength")
    if raw is not None:
        return float(raw)
    freq = meta.get("ImagingFrequency")
    if freq is not None:
        f = float(freq)
        if 12.0 < f < 13.0:
            return 0.3
        if 63.0 < f < 64.0:
            return 1.5
    device = (meta.get("DeviceSerialNumber") or "").lower()
    if "0.3t" in device:
        return 0.3
    return -1.0


def _walk_subject_dir(subj_path: Path, subject_id: str, session: Optional[int] = None):
    records = []
    for task_dir in sorted(subj_path.iterdir()):
        if not task_dir.is_dir():
            continue
        info_path = task_dir / "_info.json"
        if not info_path.exists():
            continue
        with open(info_path) as f:
            try:
                info = json.load(f)
            except json.JSONDecodeError:
                continue
        meta = info.get("meta", {})
        desc = info.get("desc", "")
        parsed = _parse_desc(desc)
        if parsed is None:
            continue
        nifti_files = sorted(task_dir.glob("*.nii.gz"))
        if not nifti_files:
            continue
        nii_path = nifti_files[0]
        parsed["subject"] = subject_id
        parsed["session"] = session
        records.append(
            {
                "path": str(nii_path),
                "subject": int(subject_id),
                "session": session or 1,
                "orientation": parsed["orientation"],
                "contrast": parsed["contrast"],
                "modality": parsed["modality"],
                "site": _extract_site_from_meta(meta),
                "field_strength": _extract_field_strength_from_meta(meta),
                "meta": meta,
                "tags": info.get("tags", []),
            }
        )
    return records


def scan_raw_dataset(root: Optional[Path] = None) -> List[dict]:
    if root is None:
        root = RAW_DATA_ROOT
    root = Path(root)
    all_records = []
    seen_paths = set()
    for entry in sorted(root.iterdir()):
        if not entry.is_dir() or not entry.name.startswith("sub-"):
            continue
        parsed = _parse_session_dir(entry.name)
        if parsed is None:
            continue
        subject_id, session = parsed
        records = _walk_subject_dir(entry, subject_id, session)
        for r in records:
            if r["path"] not in seen_paths:
                seen_paths.add(r["path"])
                all_records.append(r)
    return all_records


def load_participant_tsv(tsv_path, label_map: Optional[Dict[str, int]] = None) -> Dict[int, int]:
    if label_map is None:
        label_map = DEFAULT_LABEL_MAP
    import csv

    mapping = {}
    with open(tsv_path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        reader.fieldnames = [h.strip() for h in reader.fieldnames]
        for row in reader:
            row = {k.strip(): v.strip() for k, v in row.items()}
            subj = int(row["Subject"])
            group = row["ClincalGroup"]
            if group in label_map:
                mapping[subj] = label_map[group]
    return mapping


def select_best_record(
    candidates: List[dict],
    orientation_priority: Tuple[str, ...] = ("axial", "coronal", "sagittal"),
):
    if len(candidates) == 1:
        return candidates[0]
    best = None
    best_rank = -1
    for r in candidates:
        rank = orientation_priority.index(r["orientation"]) if r["orientation"] in orientation_priority else 99
        if best is None or rank < best_rank:
            best = r
            best_rank = rank
    return best


class NigerianBrainDataset(Dataset):

    def __init__(
        self,
        root_dir: str = "/teamspace/studios/this_studio/Dataset",
        participant_tsv: Optional[str] = None,
        modalities: Tuple[str, ...] = ("T1w",),
        orientation_priority: Tuple[str, ...] = ("axial", "coronal", "sagittal"),
        exclude_gadolinium: bool = True,
        use_quality_selection: bool = False,
        quality_cache: Optional[str] = None,
        target_size: Tuple[int, int, int] = (96, 112, 96),
        split_ids: Optional[List[int]] = None,
        label_map: Optional[Dict[str, int]] = None,
        missing_modality_strategy: str = "drop",
        dwi_mode: bool = False,
        binary_target: Optional[Tuple[str, str]] = None,
        transform: Optional[callable] = None,
    ):
        self.root_dir = Path(root_dir)
        self.modalities = modalities
        self.orientation_priority = orientation_priority
        self.exclude_gadolinium = exclude_gadolinium
        self.use_quality_selection = use_quality_selection
        self.quality_cache = quality_cache
        self.target_size = target_size
        self.split_ids = split_ids
        self.binary_target = binary_target
        self.transform = transform
        self.label_map = label_map or DEFAULT_LABEL_MAP
        self.missing_modality_strategy = missing_modality_strategy
        if dwi_mode:
            self.modalities = ("DWI",)

        all_records = scan_raw_dataset(self.root_dir)

        if participant_tsv is not None:
            self.labels = load_participant_tsv(participant_tsv, self.label_map)
        else:
            psv_path = self.root_dir / "participant-info.tsv"
            if psv_path.exists():
                self.labels = load_participant_tsv(str(psv_path), self.label_map)
            else:
                self.labels = {}

        self.index = self._build_index(all_records)

    def _modality_str(self, m: str) -> str:
        return m.lower().replace("_", "")

    def _build_index(self, all_records):
        subj_map = {}
        for r in all_records:
            s = r["subject"]
            if self.split_ids is not None and s not in self.split_ids:
                continue
            if s not in subj_map:
                subj_map[s] = {}
            m = r["modality"]
            if m not in subj_map[s]:
                subj_map[s][m] = []
            subj_map[s][m].append(r)

        index = []
        for subj in sorted(subj_map.keys()):
            label = self.labels.get(subj)
            if label is None:
                continue
            if self.binary_target is not None:
                pos, neg = self.binary_target
                pos_val = self.label_map.get(pos)
                neg_val = self.label_map.get(neg)
                if label == pos_val:
                    label = 1
                elif label == neg_val:
                    label = 0
                else:
                    continue

            available = subj_map[subj]
            selected = {}
            loaded = []
            ce_gad = False
            site = "unknown"
            field_strength = -1.0

            for mod in self.modalities:
                candidates = available.get(mod, [])
                if self.exclude_gadolinium and mod == "T1w":
                    candidates = [c for c in candidates if not c["contrast"]]
                if not candidates:
                    if self._modality_str(self.modalities[0]) != self._modality_str(mod):
                        continue
                    if self._modality_str(mod) == self._modality_str("DWI") and not candidates:
                        continue
                    if self.split_ids is not None:
                        continue
                if not candidates:
                    if self.missing_modality_strategy == "drop":
                        break
                    continue
                best = select_best_record(candidates, self.orientation_priority)
                selected[mod] = best
                loaded.append(mod)
                if best["contrast"]:
                    ce_gad = True
                if best["site"] != "unknown":
                    site = best["site"]
                if best["field_strength"] > 0:
                    field_strength = best["field_strength"]

            if len(loaded) == 0:
                continue
            if not self._check_modalities_loaded(loaded):
                continue

            entry = {
                "subject": subj,
                "label": label,
                "selected": selected,
                "modalities_loaded": loaded,
                "ce_gadolinium": ce_gad or any(v["contrast"] for v in selected.values()),
                "site": site,
                "field_strength": field_strength,
            }
            index.append(entry)

        return index

    def _check_modalities_loaded(self, loaded):
        return True

    def _load_volume(self, path: str) -> np.ndarray:
        nii = nib.load(path)
        vol = nii.get_fdata(dtype=np.float32)
        vol = np.nan_to_num(vol, nan=0.0, posinf=0.0, neginf=0.0)
        return vol

    def _resample_volume(self, vol: np.ndarray) -> np.ndarray:
        import scipy.ndimage

        if vol.ndim == 2:
            vol = vol[np.newaxis, ...]
        factors = (
            self.target_size[0] / vol.shape[0],
            self.target_size[1] / vol.shape[1],
            self.target_size[2] / vol.shape[2],
        )
        return scipy.ndimage.zoom(vol, factors, order=1)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        entry = self.index[idx]
        volumes = []
        for mod in self.modalities:
            rec = entry["selected"].get(mod)
            if rec is None:
                vol = np.zeros((1, *self.target_size), dtype=np.float32)
                volumes.append(vol)
                continue
            vol = self._load_volume(rec["path"])
            vol = self._resample_volume(vol)
            vol = (vol - vol.mean()) / (vol.std() + 1e-8)
            vol = vol[np.newaxis, ...]
            volumes.append(vol)

        if len(volumes) == 1:
            volume = volumes[0]
        else:
            volume = np.concatenate(volumes, axis=0)

        volume = torch.from_numpy(volume).float()

        if self.transform:
            volume = self.transform(volume)

        return {
            "volume": volume,
            "label": entry["label"],
            "subject_id": entry["subject"],
            "modalities_loaded": entry["modalities_loaded"],
            "ce_gadolinium": entry["ce_gadolinium"],
            "site": entry["site"],
            "field_strength": entry["field_strength"],
        }
