#!/usr/bin/env python3
"""Same-allocation GPU wall-time comparisons, including boundary trace ablations."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--previous',type=Path,help='Accepted for compatibility; historical output cadence differs, so timings are not mixed');p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
rows={r['name']:r for r in json.loads(a.results.read_text())};summary=[]
fig,axs=plt.subplots(1,2,figsize=(12,4.2),constrained_layout=True);x=np.arange(2);width=.17
for i,(label,key) in enumerate([('fixed','fixed'),('optimized order1','order1'),('limited order2','order2'),('limited quadratic','order3')]):
 values=[]
 for n in (32,64):
  r=rows[f'planar-acoustic-{key}-{n}'];fixed=rows[f'planar-acoustic-fixed-{n}'];v=r['max_rank_seconds'];ratio=v/fixed['max_rank_seconds'];values.append(ratio)
  summary.append(dict(family='planar acoustic',n=n,treatment=label,seconds=v,ratio_to_fixed=ratio,cycles=r['states'][-1]['cycle']))
 axs[0].bar(x+(i-1.5)*width,values,width,label=label)
# Historical runs saved fewer snapshots; use only matched-cadence current ratios.
axs[0].set(xticks=x,xticklabels=['nz32','nz64'],ylabel='Runtime / fixed',title='Planar acoustics, same grid and cycles');axs[0].legend(fontsize=8);axs[0].set_ylim(0,2.65)
labels=[];values=[]
for family,frame,ns in [('acoustic3d','rotating',(16,24)),('acoustic3d','fastrotation',(16,32)),('entropy','fastrotation',(16,32))]:
 for n in ns:
  fixed=rows[f'{family}-{frame}-fixed-{n}'];char=rows[f'{family}-{frame}-characteristic-{n}'];ratio=char['max_rank_seconds']/fixed['max_rank_seconds'];values.append(ratio);labels.append(f'{"acoustic" if family=="acoustic3d" else "entropy"}\n{"production Ω" if frame=="rotating" else "high shift"}\nn{n}')
  assert fixed['states'][-1]['cycle']==char['states'][-1]['cycle']
  summary.append(dict(family=family,frame=frame,n=n,treatment='limited quadratic',seconds=char['max_rank_seconds'],fixed_seconds=fixed['max_rank_seconds'],ratio_to_fixed=ratio,cycles=char['states'][-1]['cycle']))
axs[1].bar(np.arange(len(values)),values,color='tab:orange');axs[1].set(xticks=np.arange(len(values)),xticklabels=labels,ylabel='Runtime / fixed',title='Rotating 3D controls, same grid and cycles');axs[1].tick_params(axis='x',labelsize=8)
for ax in axs:ax.axhline(1,color='k',lw=.8);ax.grid(axis='y',alpha=.2)
fig.savefig(a.out/'runtime_comparison.png',dpi=170);plt.close(fig)
(a.out/'runtime_comparison.json').write_text(json.dumps(dict(scope='Maximum per-rank elapsed wall time within one sequential one-node capacity allocation, including setup/output; not a production performance benchmark.',comparisons=summary),indent=2)+'\n')
print(json.dumps(summary,indent=2))
