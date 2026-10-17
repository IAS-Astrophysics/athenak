#!/usr/bin/env python3
"""Integration regression: reset finest residence once, retain it on next restart."""
import argparse,json,subprocess
from pathlib import Path
from test_binary_zoom import ROOT
from magnetic_smooth_state import checkpoint_state
p=argparse.ArgumentParser();p.add_argument('--exe',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out=a.out.resolve();a.out.mkdir(exist_ok=True,parents=True)
text=(ROOT/'inputs/dyngr/dynbbh_bondi_zoom.athinput').read_text().replace('num_levels = 9','num_levels = 3').replace('nlevels = 9','nlevels = 3').replace('level = 8','level = 2').replace('r_0 = 2','r_0 = 32')
text=text.replace('<cyclic_zoom>','<cyclic_zoom>\nrestart_reset_finest_dwell = false').replace('trun_fac_0 = 0.02','trun_fac_0 = 10')
base=a.out/'base.athinput';base.write_text(text)
def run(name,args,ok=True):
 d=a.out/name;d.mkdir(exist_ok=True)
 cmd=[str(a.exe.resolve()),'-d',str(d)]+list(map(str,args))
 with (d/'run.log').open('w') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=60)
 assert (r.returncode==0)==ok,(name,(d/'run.log').read_text()[-3000:])
 return d
first=run('first',['-i',base,'time/nlim=1']);rst=sorted((first/'rst').glob('*.rst'))[-1];old=checkpoint_state(rst);assert old['zone']==0
second=run('reset',['-r',rst,'cyclic_zoom/restart_reset_finest_dwell=true','time/nlim=2']);rst2=sorted((second/'rst').glob('*.rst'))[-1];new=checkpoint_state(rst2)
assert abs(new['next_time']-(old['time']+10))<1e-10,(old,new)
assert new['parameters']['cyclic_zoom/restart_reset_finest_dwell']in ('false','0')
third=run('resume',['-r',rst2,'time/nlim=3']);new2=checkpoint_state(sorted((third/'rst').glob('*.rst'))[-1]);assert new2['next_time']==new['next_time'];assert 'reset finest dwell once' not in (third/'run.log').read_text()
failed=run('reject-fresh',['-i',base,'cyclic_zoom/restart_reset_finest_dwell=true'],False);assert 'requires a finest-zone restart' in (failed/'run.log').read_text()
(a.out/'validation.json').write_text(json.dumps({'original':old,'reset':new,'resumed':new2,'fresh_start_rejected':True},indent=2)+'\n');print('One-shot dwell reset and native checkpoint persistence passed.')
