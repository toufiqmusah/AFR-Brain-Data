#!/usr/bin/env python3
"""
Skull-stripping script using HD-BET v2 (GPU recommended).

HD-BET v2 uses nnU-Net under the hood. The simplest way is via the
command-line tool:  hd-bet -i INPUT -o OUTPUT -device cuda

This script provides a Python wrapper for batch processing.

Usage in Colab:
    !pip install hd-bet
    !python scripts/skullstrip_colab.py \
        --input-dir /content/Dataset \
        --output-dir /content/Dataset_SkullStripped \
        --device cuda
"""

import argparse
import torch
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Skull-strip using HD-BET")
    parser.add_argument("--input-dir", required=True, help="Directory with NIfTI files (.nii.gz)")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu", "mps"])
    parser.add_argument("--disable-tta", action="store_true", help="Disable test-time augmentation (faster)")
    parser.add_argument("--save-mask", action="store_true", help="Keep the brain mask files")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        from HD_BET.checkpoint_download import maybe_download_parameters
        from HD_BET.hd_bet_prediction import get_hdbet_predictor, hdbet_predict
    except ImportError:
        print("Install HD-BET: pip install hd-bet")
        print("Note: HD-BET requires a GPU. Run this in Colab.")
        return

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    nifti_files = sorted(input_dir.rglob("*.nii.gz"))
    if not nifti_files:
        nifti_files = sorted(input_dir.rglob("*.nii"))
    if not nifti_files:
        print(f"No NIfTI files found in {input_dir}")
        return

    print(f"Found {len(nifti_files)} NIfTI files")

    maybe_download_parameters()
    predictor = get_hdbet_predictor(
        use_tta=not args.disable_tta,
        device=torch.device(args.device),
        verbose=args.verbose,
    )

    for i, fpath in enumerate(nifti_files):
        rel = fpath.relative_to(input_dir)
        out_path = output_dir / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[{i+1}/{len(nifti_files)}] {rel}")
        hdbet_predict(
            str(fpath),
            str(out_path),
            predictor,
            keep_brain_mask=args.save_mask,
            compute_brain_extracted_image=True,
        )

    print(f"\nDone. Output: {output_dir}")


def skullstrip_single(input_path: str, output_path: str, device: str = "cuda", disable_tta: bool = False):
    """Strip a single volume. Import-friendly."""
    from HD_BET.checkpoint_download import maybe_download_parameters
    from HD_BET.hd_bet_prediction import get_hdbet_predictor, hdbet_predict

    maybe_download_parameters()
    predictor = get_hdbet_predictor(
        use_tta=not disable_tta,
        device=torch.device(device),
        verbose=False,
    )
    hdbet_predict(input_path, output_path, predictor, keep_brain_mask=False, compute_brain_extracted_image=True)


if __name__ == "__main__":
    main()
