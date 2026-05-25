#!/usr/bin/env python3
"""Compare AthenaK ADM tab outputs from CTT relaxation and TwoPunctures runs."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path


def load_tab(path: Path) -> list[list[float]]:
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            rows.append([float(x) for x in stripped.split()])
    if not rows:
        raise ValueError(f"{path} did not contain numeric rows")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError(f"{path} has inconsistent row widths")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare common-cell ADM tab dumps from CTT relaxation and TwoPunctures.")
    parser.add_argument("--relax", required=True, type=Path,
                        help="ADM tab file from the hyperbolic CTT relaxation run.")
    parser.add_argument("--twopuncture", required=True, type=Path,
                        help="ADM tab file from the z4c_two_puncture reference run.")
    parser.add_argument("--coord-cols", default=3, type=int,
                        help="Number of leading coordinate columns to ignore.")
    parser.add_argument("--atol-coords", default=1.0e-12, type=float,
                        help="Coordinate match tolerance.")
    args = parser.parse_args()

    relax = load_tab(args.relax)
    ref = load_tab(args.twopuncture)
    if len(relax) != len(ref) or len(relax[0]) != len(ref[0]):
        print(
            f"shape mismatch: relax=({len(relax)}, {len(relax[0])}), "
            f"twopuncture=({len(ref)}, {len(ref[0])})",
            file=sys.stderr)
        return 2
    if args.coord_cols > 0:
        delta_x = 0.0
        for arow, brow in zip(relax, ref):
            for col in range(args.coord_cols):
                delta_x = max(delta_x, abs(arow[col] - brow[col]))
        if delta_x > args.atol_coords:
            print(f"coordinate mismatch max={delta_x:.6e}", file=sys.stderr)
            return 2

    ncols = len(relax[0]) - args.coord_cols
    sum2_by_col = [0.0]*ncols
    max_by_col = [0.0]*ncols
    total_sum2 = 0.0
    total_count = 0
    total_max = 0.0
    for arow, brow in zip(relax, ref):
        for out_col, col in enumerate(range(args.coord_cols, len(arow))):
            diff = arow[col] - brow[col]
            adiff = abs(diff)
            sum2_by_col[out_col] += diff*diff
            max_by_col[out_col] = max(max_by_col[out_col], adiff)
            total_sum2 += diff*diff
            total_max = max(total_max, adiff)
            total_count += 1
    l2_by_col = [math.sqrt(value/len(relax)) for value in sum2_by_col]
    total_l2 = math.sqrt(total_sum2/max(total_count, 1))

    print(f"total_l2 {total_l2:.16e}")
    print(f"total_max {total_max:.16e}")
    for n, (l2, mx) in enumerate(zip(l2_by_col, max_by_col), start=args.coord_cols):
        print(f"col_{n}_l2 {l2:.16e} col_{n}_max {mx:.16e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
