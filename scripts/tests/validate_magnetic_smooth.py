#!/usr/bin/env python3
"""Validate resources/fields plus inherited smooth physics and event accounting."""
import argparse,json,re,subprocess,sys
from pathlib import Path
from magnetic_smooth_state import checkpoint_state
from test_binary_zoom import load
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
subprocess.run([sys.executable,str(Path(__file__).with_name('validate_magnetic_pilot.py')),'--run',str(a.run),'--out',str(a.out)],check=True)
s=checkpoint_state(sorted((a.run/'rst').glob('*.rst'))[-1]);par=s.pop('parameters')
assert par['coord/smooth_excision'] in ('true','1') and par['cyclic_zoom/restart_reset_finest_dwell'] in ('false','0')
for key in ('smooth_excision_inflow','smooth_excision_b_damping'):assert par['coord/'+key] in ('false','0')
assert float(par['coord/smooth_excision_puncture_width_fraction'])==1 and float(par['coord/smooth_excision_puncture_weight_exponent'])==1
for path in (a.run/'bin').glob('*.mhd_w_bcc.*.bin'):
 d,_=load(path);block='';params={}
 for line in d['header']:
  line=line.split('#',1)[0].strip()
  if line.startswith('<') and line.endswith('>'):block=line[1:-1]
  elif '=' in line:
   k,v=line.split('=',1);params[block+'/'+k.strip()]=v.strip()
 assert params['coord/smooth_excision'] in ('true','1'),path
log=(a.run/'rank-0.log').read_text();resets=re.findall(r'reset finest dwell once at time=(\S+) id=(\d+) old_next=(\S+) next_time=(\S+)',log)
if resets:
 assert len(resets)==1
 t,event,old,new=resets[0];assert abs(float(t)-9903.615850407483)<1e-8 and int(event)==14 and abs(float(new)-float(t)-250)<1e-8
v=json.loads(a.out.read_text());v['smooth_checkpoint']=s;v['residence_reset_records']=resets
if s['time']>=29500-1e-6:
 assert s['zone']==0 and s['id']==42 and s['time']-(s['next_time']-250)>=200
 v['two_complete_smooth_cycles']=True
else:v['two_complete_smooth_cycles']=False
v['interpretation']='smooth-excision continuation of evolved hard-excision data; finite fields do not establish relaxation'
a.out.write_text(json.dumps(v,indent=2)+'\n');print('Smooth checkpoint:',s)
