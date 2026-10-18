#!/usr/bin/env python3
"""Check saved production topology against the ghost/CT boundary support guard.

Reads actual MeshBlock geometry rather than inferring support from a coarse
preflight or refinement-region input. Binary levels are relative to root level.
This is a support audit, not a numerical accuracy or reflux validation.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / 'tests'))
from test_binary_zoom import load
from analyze_grmhd_characteristic import params


def report(snapshot):
    data, _ = load(snapshot)
    cfg = params(data['header'])
    geometry = data['mb_geometry']
    levels = data['mb_logical'][:, 3]
    outer = np.zeros(len(levels), dtype=bool)
    faces = {}
    for axis in range(3):
        for side, label in enumerate(('i', 'o')):
            if cfg[f'mesh/{label}x{axis+1}_bc'] != 'user':
                continue
            bound = float(cfg[f'mesh/x{axis+1}{"min" if side == 0 else "max"}'])
            mask = abs(geometry[:, 2*axis+side] - bound) < 1e-9
            outer |= mask
            faces[f'{label}x{axis+1}'] = {
                str(int(level)): int(np.sum(mask & (levels == level)))
                for level in np.unique(levels[mask])
            }
    sizes = [data[f'nx{axis}_mb'] for axis in (1, 2, 3)]
    refined = outer & (levels != 0)
    return dict(snapshot=str(snapshot), time=data['time'], meshblock_shape=sizes,
                total_blocks=len(levels), physical_boundary_blocks=int(outer.sum()),
                refined_physical_boundary_blocks=int(refined.sum()),
                physical_boundary_levels={str(int(level)): int(np.sum(outer & (levels == level)))
                                          for level in np.unique(levels[outer])},
                user_faces=faces, supported=bool(min(sizes) >= 3 and not refined.any()),
                scope='Saved topology support only; no numerical accuracy claim')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshots', type=Path, nargs='+', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--require-supported', action='store_true')
    args = parser.parse_args()
    rows = [report(path) for path in args.snapshots]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2) + '\n')
    if args.require_supported and not all(row['supported'] for row in rows):
        raise SystemExit('Unsupported ghost/CT physical-boundary topology; see report')
