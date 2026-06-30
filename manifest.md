# Data Manifest — Nigerian Brain Dataset

## Structure
```
train_data/
├── Control/       # 35 subjects
│   ├── T1w/       # 142 scans (30/35 subjects)
│   ├── T2w/       #  68 scans (27/35 subjects)
│   ├── FLAIR/     #  35 scans (28/35 subjects)
│   └── DWI/       #  14 scans (14/35 subjects)
├── Dementia/      # 31 subjects
│   ├── T1w/       # 166 scans (31/31 subjects)
│   ├── T2w/       #  79 scans (31/31 subjects)
│   ├── FLAIR/     #  47 scans (30/31 subjects)
│   └── DWI/       #   2 scans ( 2/31 subjects)
└── Parkinson/     # 22 subjects
    ├── T1w/       # 134 scans (22/22 subjects)
    ├── T2w/       #  58 scans (22/22 subjects)
    ├── FLAIR/     #  32 scans (22/22 subjects)
    └── DWI/       #   9 scans ( 9/22 subjects)
```

Total: **787 NIfTI volumes, 3.2 GB** (across 88 subjects: 35 Control, 31 Dementia, 22 Parkinson)

---

## Structure
```
Dataset/
├── sub-01/                 # Base subject directory
│   ├── acq-axial_T1w/      # Task directories named from _info.json desc
│   │   ├── _info.json
│   │   └── sub-01_acq-axial_T1w.nii.gz
│   ├── acq-axial_ce-gadolinium_T1w/
│   ├── acq-coronal_T1w/
│   └── ...
├── sub-03.ses-run-1/       # Session subdirectories (~40 subjects)
│   ├── ..._run-1_T1w/      # desc field uses _run-N (not .ses-run-N)
│   │   └── sub-03_acq-axial_run-1_T1w.nii.gz
│   └── ...
└── participant-info.tsv    # Subject → clinical group mapping
```

## Per-Subject Naming
`sub-{ID}[.ses-run-{N}]_acq-{orientation}[_ce-gadolinium][_dir-{label}][_run-{N}]_{Modality}.nii.gz`
- **ID**: 01–88 (zero-padded)
- **ses-run-N**: multiple sessions (~40 subjects have these subdirectories)
- **acq**: axial / coronal / sagittal (missing in 3 edge-case files → "unknown")
- **ce-gadolinium**: contrast-enhanced (gadolinium) variant
- **dir-PA**: phase-encoding direction (some scans)
- **run-N**: session-specific run number (in desc field, mirrors .ses-run-N)

### Regex
The parser (`data/dataset.py:DESC_RE`) handles all variant patterns including edge cases with missing `_acq-`, `_dir-PA`, and `_run-N` suffixes.

---

## Modality Coverage by Group

| Modality | Control (n=35) | Dementia (n=31) | Parkinson (n=22) |
|----------|:--------------:|:----------------:|:-----------------:|
| **T1w**  | 30/35 (86%)    | 31/31 (100%)     | 22/22 (100%)      |
| **T2w**  | 27/35 (77%)    | 31/31 (100%)     | 22/22 (100%)      |
| **FLAIR**| 28/35 (80%)    | 30/31 (97%)      | 22/22 (100%)      |
| **DWI**  | 14/35 (40%)    |  2/31 (6%)       |  9/22 (41%)       |

## Subjects Missing Specific Modalities

**Control subjects missing T1w** (5): sub-03, sub-05, sub-10, sub-12, sub-16  
**Control subjects missing T2w** (8): sub-02, sub-03, sub-05, sub-06, sub-08, sub-10, sub-14, sub-15  
**Control subjects missing FLAIR** (7): sub-06, sub-08, sub-11, sub-12, sub-15, sub-16, sub-17  
**Dementia subjects missing FLAIR** (1): sub-58

All Dementia and Parkinson subjects have both T1w + T2w.

---

## Multi-Modal Intersection

Subjects that have **all three anatomical modalities** (T1w ∩ T2w ∩ FLAIR):

| Group     | Count | Percentage |
|-----------|:-----:|:----------:|
| Control   | 25/35 |    71%     |
| Dementia  | 30/31 |    97%     |
| Parkinson | 22/22 |   100%     |

**77/88 subjects** (87%) have complete T1w+T2w+FLAIR coverage — suitable for multi-modal experiments.

---

## Pipeline Curation

The `NigerianBrainDataset` applies these selection rules:
1. **Label required**: subjects without `participant-info.tsv` entry are dropped (0 of 88)
2. **Gadolinium exclusion**: `ce-gadolinium` T1w scans excluded by default (opt-in via `exclude_gadolinium=False`)
3. **Orientation priority**: axial > coronal > sagittal (fixed, not IQA-driven)
4. **Multi-session**: best orientation ranked across all sessions; IQA tiebreak not yet used
5. **Missing modality**: configurable — `"drop"` (skip subject) or `"zero"` (zero-pad channel)

### Quality Scoring (`data/quality.py`)

Per-slice BRISQUE + CLIP-IQA aggregated to volume-level composite. Run offline via:
```bash
python scripts/precompute_quality.py
```

**Caveat**: Both BRISQUE and CLIP-IQA were trained on natural images, not clinical MRI. Their validity for this domain is unverified. Manual inspection of a stratified sample is recommended before trusting automated selection.

### Final Probing Dataset (multi-modal, T1w+T2w+FLAIR, no DWI)

| Group     | Raw Subjects | After Filtration |
|-----------|:------------:|:----------------:|
| Control   | 35           | 25               |
| Dementia  | 31           | 30               |
| Parkinson | 22           | 22               |
| **Total** | **88**       | **77**           |

11 subjects lost (all Control) due to missing one or more of T1w/T2w/FLAIR.

### Notes

- **Multi-orientation**: Many subjects have all 3 orientations (axial, coronal, sagittal) for T1w and T2w, often with and without contrast — up to 6 T1w scans per subject (3 orients × 2 contrast).
- **DWI is sparse**: Only 25 DWI scans across 14+2+9 = 25 subjects. Use with caution; likely insufficient for training but usable for evaluation.
- **Acquisition quality varies**: 3 sites (0.3T Hitachi, 1.5T GE, 1.5T Toshiba), clinical protocols. Some scans have visible artifacts — this is known and discussed in Wogu et al. (2025).
- **Contrast-enhanced scans**: ~64 subjects have ce-gadolinium T1w; these are flagged in filenames and could be confound if not accounted for.
