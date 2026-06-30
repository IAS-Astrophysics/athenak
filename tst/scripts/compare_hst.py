#!/usr/bin/env python3
"""
Compare two AthenaK .hst (history) files for bitwise regression testing.

Usage:
    python compare_hst.py <file_old> <file_new> [--threshold 1e-14]

Exit codes:
    0 = files match within threshold
    1 = files differ beyond threshold
"""

import argparse
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "vis", "python"))
import athena_read  # noqa: E402


def compare_hst_files(file_old, file_new, threshold=1.0e-14):
    data_old = athena_read.hst(file_old)
    data_new = athena_read.hst(file_new)

    keys_old = set(data_old.keys())
    keys_new = set(data_new.keys())
    if keys_old != keys_new:
        print("FAIL: column mismatch")
        print(f"  old only: {keys_old - keys_new}")
        print(f"  new only: {keys_new - keys_old}")
        return False

    all_pass = True
    max_diff_overall = 0.0
    max_reldiff_overall = 0.0

    for key in sorted(data_old.keys()):
        old_vals = data_old[key]
        new_vals = data_new[key]

        if len(old_vals) != len(new_vals):
            print(f"FAIL: '{key}' row count differs: {len(old_vals)} vs {len(new_vals)}")
            all_pass = False
            continue

        abs_diff = np.abs(old_vals - new_vals)
        max_abs = np.max(abs_diff)
        max_diff_overall = max(max_diff_overall, max_abs)

        scale = np.maximum(np.abs(old_vals), np.abs(new_vals))
        nonzero = scale > 0
        if np.any(nonzero):
            rel_diff = np.max(abs_diff[nonzero] / scale[nonzero])
        else:
            rel_diff = 0.0
        max_reldiff_overall = max(max_reldiff_overall, rel_diff)

        if max_abs > threshold:
            print(f"FAIL: '{key}' max |diff| = {max_abs:.3e} > {threshold:.1e}")
            all_pass = False
        else:
            print(f"  OK: '{key}' max |diff| = {max_abs:.3e}")

    print(f"\nOverall max |diff|    = {max_diff_overall:.3e}")
    print(f"Overall max |rel diff| = {max_reldiff_overall:.3e}")
    print(f"Threshold              = {threshold:.1e}")

    if all_pass:
        print("\nPASS: all columns match within threshold")
    else:
        print("\nFAIL: some columns exceed threshold")

    return all_pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare two .hst files")
    parser.add_argument("file_old", help="Reference (old) .hst file")
    parser.add_argument("file_new", help="New .hst file to compare")
    parser.add_argument("--threshold", type=float, default=1.0e-14,
                        help="Max allowed absolute difference (default: 1e-14)")
    args = parser.parse_args()

    ok = compare_hst_files(args.file_old, args.file_new, args.threshold)
    sys.exit(0 if ok else 1)
