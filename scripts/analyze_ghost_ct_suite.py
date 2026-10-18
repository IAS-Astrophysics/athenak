#!/usr/bin/env python3
"""Audit the ghost/native-CT suite, including transverse magnetic packets.

No submissions. Old face-only results remain immutable comparison data.
"""
import argparse,json,subprocess,sys,re
from pathlib import Path
import numpy as np
from analyze_grmhd_characteristic import audit
p=argparse.ArgumentParser();p.add_argument('--runs',type=Path,required=True);p.add_argument('--previous',type=Path,required=True);p.add_argument('--face-results',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
subprocess.run([sys.executable,str(Path(__file__).with_name('analyze_improved_boundary_suite.py')),'--runs',str(a.runs),'--previous',str(a.previous),'--out',str(a.out)],check=True)
rows=json.loads((a.out/'all_audits.json').read_text())
alf=[]
for choice in ('reference','fixed','order3'):
 row=audit(a.runs/f'ct-alfven-{choice}-32',target=180,boundary='characteristic' if choice=='order3' else 'fixed',wave_mode='alfven');rows.append(row);alf.append(row)
# Check the exact tested pipeline from saved headers, not from directory names.
from analyze_grmhd_characteristic import params
from test_binary_zoom import load,final_output
for row in rows:
 d,_=load(final_output(a.runs/row['name'],'mhd_w_bcc'));cfg=params(d['header'])
 char=cfg['problem/reservoir_boundary']=='characteristic'
 assert (cfg.get('problem/reservoir_ghost_ct')=='true')==char,row['name']
 row['ghost_ct']=char
 repairs=[]
 for rank in range(12):
  text=(a.runs/row['name']/f'rank-{rank}.log').read_text()
  repairs.extend(line for line in text.splitlines() if re.search(r'MHD.*(?:nonfinite.*repair|finite-EMF fallback)',line))
 row['nonfinite_repair_messages']=repairs
 assert not repairs,(row['name'],'Nonfinite repairs/fallbacks require diagnosis',repairs[:5])
 if char:
  row['projection_columns_in_last_logged_counters']=row.pop('faces_in_last_logged_counters')
  row['limited_ghost_cells_in_last_logged_counters']=row.pop('limited_faces_in_last_logged_counters')
assert len(rows)==33
(a.out/'all_audits.json').write_text(json.dumps(rows,indent=2)+'\n')
def profile(row):
 q=np.asarray(row['profile']);z=np.unique(q[:,0]);return z,np.array([q[q[:,0]==zz,1].mean() for zz in z])
rz,rv=profile(alf[0]);comparison=[]
for row in alf[1:]:
 z,v=profile(row);j=np.searchsorted(rz,z);assert np.allclose(rz[j],z,atol=1e-10);err=(v-rv[j])/.01
 comparison.append(dict(name=row['name'],max_incoming_error_over_seed=float(abs(err).max()),rms_incoming_error_over_seed=float(np.sqrt(np.mean(err*err))),max_rank_seconds=row['max_rank_seconds']))
(a.out/'alfven_comparison.json').write_text(json.dumps(comparison,indent=2)+'\n')
new=json.loads((a.out/'rotating/comparisons.json').read_text());old={r['name']:r for r in json.loads((a.face_results/'rotating/comparisons.json').read_text())};paired=[]
for row in new:
 baseline=old[row['name']]
 paired.append(dict(name=row['name'],ghost_ct_incoming_proxy_max=row['incoming_proxy_max'],face_only_incoming_proxy_max=baseline['incoming_proxy_max'],ghost_ct_incoming_proxy_rms=row['incoming_proxy_rms'],face_only_incoming_proxy_rms=baseline['incoming_proxy_rms']))
(a.out/'versus_face_only.json').write_text(json.dumps(paired,indent=2)+'\n')
print('All33 ghost/native-CT cases audited; inspect diagnostics before adoption.')
