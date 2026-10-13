#!/usr/bin/env python3
"""Validate a chronological restart chain and link its unique saved fluid states.

The output directory must be new. Source snapshots are never modified. The
manifest records source paths and hashes; duplicate times must have identical
fields and mesh geometry. This does not reconstruct frozen zoom storage.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'vis/python'))
from bin_convert import read_binary

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--runs', type=Path, nargs='+', required=True)
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
a.out.mkdir(parents=True, exist_ok=False)
(a.out / 'bin').mkdir()
manifest, seen, logs, resources = [], {}, [], []
for run in a.runs:
    reports = [json.loads(f.read_text()) for f in sorted(run.glob('resources-*.json'))]
    memory = [json.loads(f.read_text()) for f in sorted(run.glob('memory-*.json'))]
    assert len(reports) == len(memory) == 12, run
    assert all(r['returncode'] == 0 and r['stopped'] is None for r in reports), run
    assert all(m['combined_peak_bytes'] <= m['budget_bytes'] for m in memory), run
    rss = sum(r['peak_rss_bytes'] for r in reports)
    allocations = sum(m['combined_peak_bytes'] for m in memory)
    assert rss + allocations < 30e9, run  # Conservative: includes some double counting.
    resources.append(dict(run=run.name, rank_count=12, rss_peaks_sum=rss,
                          allocation_peaks_sum=allocations, guards_clean=True))
    logs.append((run / 'rank-0.log').read_text())
    for path in sorted((run / 'bin').glob('*.mhd_w_bcc.*.bin')):
        d = read_binary(str(path))
        fields = {k: np.asarray(v) for k, v in d['mb_data'].items()}
        assert all(np.isfinite(v).all() for v in fields.values()), path
        assert fields['dens'].min() > 0 and fields['press'].min() > 0, path
        t = float(d['time'])
        if t in seen:
            prior = read_binary(str(seen[t]))
            assert np.array_equal(prior['mb_geometry'], d['mb_geometry']), path
            assert all(np.array_equal(prior['mb_data'][k], v) for k, v in fields.items()), path
            continue
        assert not manifest or t > manifest[-1]['time'], path
        seen[t] = path
        index = len(manifest)
        name = f'matched_bondi.mhd_w_bcc.{index:05d}.bin'
        (a.out / 'bin' / name).symlink_to(path.resolve())
        manifest.append(dict(file=name, source=f'{run.name}/bin/{path.name}', time=t,
                             cycle=d['cycle'], blocks=d['n_mbs'],
                             sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                             density_min=float(fields['dens'].min()),
                             pressure_min=float(fields['press'].min()), finite=True))
(a.out / 'rank-0.log').write_text('\n'.join(logs))
(a.out / 'chain-validation.json').write_text(json.dumps(
    dict(resources=resources, snapshots=manifest), indent=2) + '\n')
print(f'Validated {len(manifest)} unique states through t={manifest[-1]["time"]}', flush=True)
