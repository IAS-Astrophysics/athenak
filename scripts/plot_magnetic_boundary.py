#!/usr/bin/env python3
"""Plot six-face active-cell boundary diagnostics produced by the MHD movie tool."""
import argparse,json,os
from pathlib import Path
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser(description=__doc__);p.add_argument('input',type=Path);a=p.parse_args()
d=json.loads(a.input.read_text());r=d['rows'];ts=[x['time'] for x in r]
means=np.array([[[v['mean'] for v in q] for q in x['profiles']] for x in r])
maxabs=np.array([[[v['maxabs'] for v in q] for q in x['profiles']] for x in r])
assert np.isfinite(means).all() and np.isfinite(maxabs).all()
# A strict sanity check for cancellation of rotation in the initial active cells.
assert ts[0]==0 and maxabs[0,:,0,3].max()<2e-6
fig,axs=plt.subplots(2,2,figsize=(11,7),dpi=140)
for i,name in enumerate(d['face_order']):
 axs[0,0].plot(ts,means[:,i,0,6],label=name)
 axs[0,1].plot(ts,np.maximum(maxabs[:,i,0,4],maxabs[:,i,0,5]),label=name)
 axs[1,0].plot(d['cell_distances_from_face_M'],means[-1,i,:,6],label=name)
 axs[1,1].plot(d['cell_distances_from_face_M'],means[-1,i,:,3],label=name)
for ax in axs.flat:ax.grid(alpha=.25)
axs[0,0].set(title='Adjacent mean Btilde_z/Binf − 1',xlabel='Global time / M')
axs[0,1].set(title='Adjacent max |Btilde_x,y|/Binf',xlabel='Global time / M')
axs[1,0].set(title='Final mean Btilde_z/Binf − 1',xlabel='Distance inward from face / M')
axs[1,1].set(title='Final mean inertial coordinate speed / c',xlabel='Distance inward from face / M')
axs[0,0].legend(ncol=2,fontsize=8);fig.suptitle('Outer active cells: magnetic response and inertial-frame motion')
fig.tight_layout(rect=(0,0,1,.95));fig.savefig(a.input.parent/'boundary_magnetic.png');plt.close(fig)
summary={'time_range_M':[ts[0],ts[-1]],'initial_max_inertial_speed_c':float(maxabs[0,:,0,3].max()),
'final_adjacent_face_means':{name:dict(zip(d['fields'],means[-1,i,0].tolist())) for i,name in enumerate(d['face_order'])},
'final_adjacent_max_abs_over_all_cells_faces':dict(zip(d['fields'],maxabs[-1,:,0].max(axis=0).tolist())),
'final_adjacent_max_abs_face_mean':dict(zip(d['fields'],np.abs(means[-1,:,0]).max(axis=0).tolist())),
'initial_adjacent_max_abs_face_mean':dict(zip(d['fields'],np.abs(means[0,:,0]).max(axis=0).tolist())),
'limits':d['limits']}
(a.input.parent/'boundary_summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
