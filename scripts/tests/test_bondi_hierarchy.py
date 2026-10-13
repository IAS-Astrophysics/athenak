#!/usr/bin/env python3
"""Bounded complete merge/split sweeps, boundary-cache and restart equivalence."""
import argparse,json,re,subprocess
from pathlib import Path
import numpy as np
from make_bondi_hierarchy import make
from test_binary_zoom import inspect,load,final_output,compare_restart

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--exe',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
p.add_argument('--check-only',action='store_true')
a=p.parse_args();a.out=a.out.resolve();a.out.mkdir(parents=True,exist_ok=True)
text,_=make(False,1e-6)
for k,v in {'num_levels':5,'nlevels':5,'max_nmb_per_rank':640,'max_nzmb_per_dvce':640,
            'max_nzmb_per_host':640,'f_ref':4,'sep':4,'frame_omega':.125,'bondi_cavity_radius':4,
            'require_resolved_horizon':'false','level':4,'tlim':100,'nlim':8,'ndiag':1}.items():
 text=re.sub(r'^'+k+r'\s*=.*$',f'{k} = {v}',text,flags=re.M)
# This intentionally underresolved hierarchy tests topology and cache invalidation only.
text=re.sub(r'<refined_region1>.*?(?=<amr_criterion1>)',''.join(
 f'<refined_region{i}>\nlevel = 4\nx1min = {x-.25}\nx1max = {x+.25}\nx2min = -.25\nx2max = .25\nx3min = -.25\nx3max = .25\n'
 for i,x in enumerate((2,-2),1)),text,flags=re.S)
for d in (1,2,3):
 text=text.replace(f'nx{d} = 64',f'nx{d} = 16').replace(f'nx{d} = 8',f'nx{d} = 4')
 text=text.replace(f'x{d}min = -256',f'x{d}min = -32').replace(f'x{d}max = 256',f'x{d}max = 32')
text=text.replace('file_type = bin','file_type = bin\ndcycle = 1')
text=text.replace('file_type = rst','file_type = rst\ndcycle = 4')
input_path=a.out/'small.athinput';input_path.write_text(text)
def run(name,flags=(),restart=None):
 out=a.out/name
 if a.check_only: return out
 out.mkdir(exist_ok=True)
 cmd=[str(a.exe.resolve()),'-i',str(input_path),'-d',str(out)]+list(flags)
 if restart:cmd+=['-r',str(restart)]
 with (out/'run.log').open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=240)
 return out
cached=run('cached');uncached=run('uncached',['problem/bondi_cache_boundary=false'])
report={'cached':inspect(cached)}
for var in ('mhd_w_bcc','adm','mhd_divb'):
 _,f=load(final_output(cached,var));_,g=load(final_output(uncached,var))
 assert all(np.array_equal(f[k],g[k]) for k in f),var
levels=[s['max_level'] for s in report['cached']['states']]
assert min(levels)==0 and levels[-1]==4,levels
first,_=load(sorted((cached/'bin').glob('*.mhd_w_bcc.*.bin'))[0]);last,_=load(final_output(cached,'mhd_w_bcc'))
assert np.array_equal(first['mb_logical'],last['mb_logical'])
checkpoint=sorted((cached/'rst').glob('*.rst'))[1]
resumed=run('restart',restart=checkpoint)
report['restart']=compare_restart(cached,resumed)
assert all(v['max_abs_error']==0 for v in report['restart'].values())
no_cache_restart=run('restart_uncached',['problem/bondi_cache_boundary=false'],checkpoint)
for var in ('mhd_w_bcc','adm','mhd_divb'):
 _,f=load(final_output(resumed,var));_,g=load(final_output(no_cache_restart,var))
 assert all(np.array_equal(f[k],g[k]) for k in f),var
report['cache_bitwise_equal']=True
log=(cached/'run.log').read_text()
runtimes=[float(t) for t in re.findall(r'CyclicZoom: time = \S+ runtime = (\S+)',log)]
expected=[0,0,80e-6,10*(8**3)**.5*1e-6,80e-6,0,0,50e-6]
assert np.allclose(runtimes,expected,rtol=2e-6,atol=1e-12),(runtimes,expected)
report['stage_runtimes']=runtimes
(a.out/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
print('Merge/split topology, Eq.17 timing, cache equivalence, and restart passed.')
