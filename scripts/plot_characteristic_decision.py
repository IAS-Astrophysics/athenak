#!/usr/bin/env python3
"""Compact accuracy comparison from audited full-mode and rotating histories."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);a=p.parse_args();root=a.results
modes=json.loads((root/'full_mode_acoustic.json').read_text());rot=json.loads((root/'rotating_time_series.json').read_text());fig,axs=plt.subplots(1,2,figsize=(11.6,4.2),constrained_layout=True)
labels={'fixed':'Fixed reservoir','order1':'Characteristic, order 1','order2':'Characteristic, order 2','order3':'Characteristic, quadratic'}
for row in modes['series']:
 if row['n']==64:axs[0].plot(row['times'],100*np.asarray(row['incoming_pressure_error_over_seed']),label=labels[row['treatment']])
axs[0].set(title='Planar acoustic: full seven-wave decomposition (nz64)',xlabel='t / M',ylabel='Incoming pressure difference (% of seed)')
for row in rot['series']:
 if row['frame']=='rotating' and row['n']==24:axs[1].plot(row['times'],100*np.asarray(row['incoming_proxy_max_over_seed']),label='Fixed reservoir' if row['treatment']=='fixed' else 'Characteristic, quadratic')
axs[1].set(title='Rotating compact acoustic packet (n24)',xlabel='t / M',ylabel='Incoming z acoustic proxy difference (% of seed)')
for ax in axs:ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.savefig(root/'acoustic_decision.png',dpi=170);plt.close(fig)
