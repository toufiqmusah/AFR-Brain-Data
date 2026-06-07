import argparse
import csv
import json
import os
import shutil

NIFTI_NAMES = {'t1': 'T1w', 't2': 'T2w', 'flair': 'FLAIR', 'dwi': 'DWI'}


def load_participants(tsv_path):
    mapping = {}
    with open(tsv_path, newline='') as f:
        reader = csv.DictReader(f, delimiter='\t')
        reader.fieldnames = [h.strip() for h in reader.fieldnames]
        for row in reader:
            row = {k.strip(): v.strip() for k, v in row.items()}
            mapping[row['Subject'].zfill(2)] = row['ClincalGroup']
    return mapping


def iter_subject_dirs(raw_dir):
    for entry in sorted(os.listdir(raw_dir)):
        full = os.path.join(raw_dir, entry)
        if not os.path.isdir(full) or not entry.startswith('sub-'):
            continue
        yield entry, full


def find_nifti_info(base_dir):
    results = []
    for root, dirs, files in os.walk(base_dir):
        nifti_files = [f for f in files if f.endswith('.nii.gz')]
        if not nifti_files:
            continue
        info_path = os.path.join(root, '_info.json')
        meta = {}
        if os.path.exists(info_path):
            with open(info_path) as fh:
                meta = json.load(fh).get('meta', {})
        for nf in nifti_files:
            results.append({'path': os.path.join(root, nf), 'filename': nf, 'meta': meta})
    return results


def make_output_name(subject_dir, info):
    meta = info['meta']
    nf = info['filename']
    stem = os.path.splitext(nf)[0]
    stem = os.path.splitext(stem)[0]
    raw_mod = stem.lower()
    modality = NIFTI_NAMES.get(raw_mod, raw_mod.upper())

    acq = meta.get('acq', '')
    ce = meta.get('ce', '')

    parts = [subject_dir]
    if acq:
        parts.append(f'acq-{acq}')
    if ce and ce not in ('', 'none'):
        parts.append('ce-gadolinium')

    return f'{"_".join(parts)}_{modality}.nii.gz', modality


def main():
    parser = argparse.ArgumentParser(description="Preprocess and organize NIfTI brain imaging data.")
    parser.add_argument(
        '--raw-dir', 
        type=str, 
        default='Dataset', 
        help="Path to the raw dataset directory (default: 'Dataset')"
    )
    parser.add_argument(
        '--participant-tsv', 
        type=str, 
        default='/content/AFR-Brain-Data/participant-info.tsv', 
        help="Path to the participant info TSV file (default: '/content/AFR-Brain-Data/participant-info.tsv')"
    )
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default='/content/Preprocessed-Dataset', 
        help="Path to the output directory (default: '/content/Preprocessed-Dataset')"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    participant_map = load_participants(args.participant_tsv)
    print(f"Loaded {len(participant_map)} participant mappings")

    stats = {}
    copied = 0

    for subject_dir, full_path in iter_subject_dirs(args.raw_dir):
        subj_id = subject_dir.split('-')[1].split('.')[0].zfill(2)
        diagnosis = participant_map.get(subj_id)
        if not diagnosis:
            print(f"  SKIP {subject_dir}: no diagnosis")
            continue

        nifti_infos = find_nifti_info(full_path)
        if not nifti_infos:
            print(f"  SKIP {subject_dir}: no NIfTI files")
            continue

        for info in nifti_infos:
            out_name, modality = make_output_name(subject_dir, info)
            out_dir = os.path.join(args.output_dir, diagnosis, modality)
            os.makedirs(out_dir, exist_ok=True)

            dst = os.path.join(out_dir, out_name)
            if os.path.exists(dst):
                continue

            shutil.copy2(info['path'], dst)
            copied += 1
            key = (diagnosis, modality)
            stats[key] = stats.get(key, 0) + 1

    print(f"\nCopied {copied} files")
    print(f"\n{'Diagnosis':<12} {'T1w':>6} {'T2w':>6} {'FLAIR':>6} {'DWI':>6}")
    print('-' * 36)
    for d in sorted(set(k[0] for k in stats)):
        row = [d]
        for m in ['T1w', 'T2w', 'FLAIR', 'DWI']:
            row.append(str(stats.get((d, m), 0)))
        print(f"{row[0]:<12} {row[1]:>6} {row[2]:>6} {row[3]:>6} {row[4]:>6}")
    print(f"\nTotal: {copied} files")


if __name__ == '__main__':
    main()