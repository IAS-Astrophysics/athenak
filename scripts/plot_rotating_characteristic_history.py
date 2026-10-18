#!/usr/bin/env python3
"""Time histories for matched rotating compact acoustic controls.

The z-directed incoming weak-acoustic proxy is not an exact oblique-wave
reflection coefficient. It is compared with the same-resolution larger domain
at identical saved times and cycle counts, including pressure and velocity.
"""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze_rotating_characteristic import cube
p=argparse.ArgumentParser();p.add_argument('--runs',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
series=[];fig,axs=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
for frame,ns,ax in [('rotating',(16,24),axs[0]),('fastrotation',(16,32),axs[1])]:
 for n in ns:
  ref=a.runs/f'acoustic3d-{frame}-reference-{n}';refs=sorted((ref/'bin').glob('*.mhd_w_bcc.*.bin'))
  for choice in ('fixed','characteristic'):
   case=a.runs/f'acoustic3d-{frame}-{choice}-{n}';files=sorted((case/'bin').glob('*.mhd_w_bcc.*.bin'));assert len(files)==len(refs)
   times=[];errors=[];rms=[]
   for path,rpath in zip(files,refs):
    d,c,f,v,proxy=cube(case,snapshot=path);dr,cr,fr,vr,pr=cube(ref,snapshot=rpath)
    assert d['time']==dr['time'] and d['cycle']==dr['cycle']
    error=(proxy-pr)/float(c['problem/reservoir_test_pulse']);times.append(d['time']);errors.append(float(abs(error).max()));rms.append(float(np.sqrt(np.mean(error*error))))
   series.append(dict(frame=frame,n=n,treatment=choice,times=times,incoming_proxy_max_over_seed=errors,incoming_proxy_rms_over_seed=rms));ax.plot(times,errors,label=f'{choice}, n{n}',ls='--' if n==16 else '-')
 ax.set(title='Production rotation' if frame=='rotating' else 'High coordinate shift',xlabel='t / M',ylabel='Max incoming z acoustic difference / seed');ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.savefig(a.out/'rotating_time_series.png',dpi=170);plt.close(fig)
(a.out/'rotating_time_series.json').write_text(json.dumps(dict(scope='Weak z-directed acoustic proxy difference from matched-grid, matched-time larger domain; not an exact nonlinear/oblique reflection coefficient.',series=series),indent=2)+'\n')
