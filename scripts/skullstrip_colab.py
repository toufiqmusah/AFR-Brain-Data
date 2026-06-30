#!/usr/bin/env python3
"""
Skull-stripping script for Google Colab (GPU required).

Uses HD-BET to strip skulls from the raw or prepared dataset.
Run this in Colab, not locally (no GPU here).

Usage in Colab:
    !pip install hd-bet
    !python scripts/skullstrip_colab.py \
        --input-dir /content/Dataset \
        --output-dir /content/Dataset_SkullStripped \
        --device cuda
"""

import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Skull-strip using HD-BET")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices="fast accurate", default="accurate")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--n-processes", type=int, default=2)
    parser.add_argument("--nifti-only", action="store_true", help="Only process NIfTI files")
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        from HD_BET import setup as hd_bet_setup
        from HD_BET.run import hd_bet
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
    print(f"Running HD-BET (mode={args.mode}, device={args.device})...")

    hd_bet(
        [str(f) for f in nifti_files],
        [str(output_dir / f.relative_to(input_dir).parent / f.name) for f in nifti_files],
        mode=args.mode,
        device=args.device,
        n_processes=args.n_processes,
    )

    print(f"Skull-stripping complete. Output: {output_dir}")


def skullstrip_single(input_path, output_path, device="cuda"):
    """Strip a single volume. Import-friendly."""
    from HD_BET import setup as hd_bet_setup
    from HD_BET.run import hd_bet

    hd_bet(
        [input_path],
        [output_path],
        mode="accurate",
        device=device,
        n_processes=1,
    )


if __name__ == "__main__":
    main()
