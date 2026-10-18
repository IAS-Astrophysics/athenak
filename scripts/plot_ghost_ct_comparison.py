#!/usr/bin/env python3
"""Compare audited ghost/CT and historical face-only controls, with clear scope."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--results', type=Path, required=True)
parser.add_argument('--face-results', type=Path, required=True)
args = parser.parse_args()
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)
records = []
labels = [('Fixed', args.results, 'fixed'),
          ('Face-only characteristic (v2)', args.face_results, 'order3'),
          ('Ghost/CT characteristic (v3)', args.results, 'order3')]
for label, root, treatment in labels:
    planar = next(row for row in json.loads((root/'full_mode_acoustic.json').read_text())['series']
                  if row['n'] == 64 and row['treatment'] == treatment)
    rotating = next(row for row in json.loads((root/'rotating_time_series.json').read_text())['series']
                    if row['frame'] == 'rotating' and row['n'] == 24
                    and row['treatment'] == ('fixed' if treatment == 'fixed' else 'characteristic'))
    runtime = json.loads((root/'runtime_comparison.json').read_text())['comparisons']
    pr = next(row for row in runtime if row['family'] == 'planar acoustic' and row['n'] == 64
              and row['treatment'] == ('fixed' if treatment == 'fixed' else 'limited quadratic'))
    rr = None if treatment == 'fixed' else next(row for row in runtime
              if row['family'] == 'acoustic3d' and row['frame'] == 'rotating' and row['n'] == 24)
    records.append(dict(treatment=label,
        planar_final_percent_seed=100*planar['incoming_pressure_error_over_seed'][-1],
        planar_peak_percent_seed=100*max(planar['incoming_pressure_error_over_seed']),
        rotating_final_percent_seed=100*rotating['incoming_proxy_max_over_seed'][-1],
        rotating_peak_percent_seed=100*max(rotating['incoming_proxy_max_over_seed']),
        planar_runtime_ratio=pr['ratio_to_fixed'], rotating_runtime_ratio=rr['ratio_to_fixed'] if rr else 1.0))
    axes[0].plot(planar['times'], 100*np.array(planar['incoming_pressure_error_over_seed']), label=label)
    axes[1].plot(rotating['times'], 100*np.array(rotating['incoming_proxy_max_over_seed']), label=label)
axes[0].set(title='Planar acoustic, nz64', xlabel='t / M', ylabel='Incoming pressure difference (% of seed)')
axes[1].set(title='Rotating compact acoustic, n24', xlabel='t / M', ylabel='Incoming z proxy difference (% of seed)')
x = np.arange(2)
for index, row in enumerate(records):
    axes[2].bar(x+(index-1)*.24, [row['planar_runtime_ratio'], row['rotating_runtime_ratio']], .24,
                label=row['treatment'])
axes[2].set(title='Matched-cadence runtime ratios', ylabel='Runtime / fresh fixed control',
            xticks=x, xticklabels=['Planar nz64', 'Rotating n24'])
axes[2].axhline(1, color='black', lw=.8)
for ax in axes:
    ax.grid(alpha=.2)
    ax.legend(fontsize=7)
fig.savefig(args.results/'versus_face_only.png', dpi=160)
plt.close(fig)
(args.results/'three_method_comparison.json').write_text(json.dumps(dict(
    scope='Residuals against matched larger-domain controls, not nonlinear reflection coefficients. Runtime ratios use each method allocation\'s fresh fixed control; absolute allocation times are not directly compared.',
    comparisons=records), indent=2)+'\n')
