#!/usr/bin/env python3
"""
Offline BRISQUE + CLIP-IQA quality score computation.

Pre-computes quality scores for all scans and caches them.
Requires piq: pip install piq

Usage:
    python scripts/precompute_quality.py \
        --output outputs/quality_cache/scores.json \
        --n-slices 10
"""

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/quality_cache/scores.json")
    parser.add_argument("--n-slices", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.5)
    return parser.parse_args()


def main():
    args = parse_args()
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from data.dataset import scan_raw_dataset
    from data.quality import precompute_quality_scores

    records = scan_raw_dataset()
    print(f"Scanned {len(records)} records")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cache = precompute_quality_scores(
        records,
        output_path=str(output_path),
        n_slices=args.n_slices,
        alpha=args.alpha,
    )
    print(f"Computed quality scores for {len(cache)} volumes")
    print(f"Cache saved to {output_path}")


if __name__ == "__main__":
    main()
