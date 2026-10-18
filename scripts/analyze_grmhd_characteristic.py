#!/usr/bin/env python3
"""Audit one completed full-GRMHD boundary suite; retain reference-subtracted diagnostics."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import argparse,json,re,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent/'tests'))
from test_binary_zoom import load

def params(header):
 out={};block=''
 for line in header:
  line=line.split('#',1)[0].strip()
  if line.startswith('<') and line.endswith('>'):block=line[1:-1]
  elif '=' in line:
   key,value=line.split('=',1);out[block+'/'+key.strip()]=value.strip()
 return out

def audit(case,target=None,plane=True,boundary=None,wave_mode=None):
 rss=tracked=budget=0;hosts=set();limited=faces=0;seconds=[]
 for rank in range(12):
  r=json.loads((case/f'resources-{rank}.json').read_text());m=json.loads((case/f'memory-{rank}.json').read_text())
  assert r['returncode']==0 and r['stopped'] is None,(case,rank,r)
  assert r['peak_rss_bytes']<=r['memory_budget_bytes'] and m['combined_peak_bytes']<=m['budget_bytes']
  assert r['command'][r['command'].index('-t')+1]=='00:55:00'
  log=(case/f'rank-{rank}.log').read_text();identity=re.search(rf'rank={rank} host=(\S+) affinity={rank//2}\.{rank%2}\s',log);assert identity
  hosts.add(identity[1]);rss+=r['peak_rss_bytes'];tracked+=m['combined_peak_bytes'];budget+=r['memory_budget_bytes']+m['budget_bytes']
  seconds.append(r['seconds'])
  counts=re.findall(r'faces=(\d+) limited=(\d+)',log)
  if counts:faces+=int(counts[-1][0]);limited+=int(counts[-1][1])
 assert len(hosts)==1 and budget<=30e9 and rss+tracked<30e9
 assert 'Terminating on time limit' in (case/'rank-0.log').read_text()
 states=[];final=None
 for path in sorted((case/'bin').glob('*.mhd_w_bcc.*.bin')):
  d,f=load(path);cfg=params(d['header'])
  assert all(np.isfinite(v).all() for v in f.values()) and f['dens'].min()>0 and f['press'].min()>0
  sl=tuple(slice((d[f'nx{axis}_out_mb']-d[f'nx{axis}_mb'])//2,(d[f'nx{axis}_out_mb']-d[f'nx{axis}_mb'])//2+d[f'nx{axis}_mb']) for axis in (3,2,1))
  B=np.sqrt(sum(f[f'bcc{i}'][(slice(None),)+sl].astype(float)**2 for i in (1,2,3)))
  dd,df=load(path.with_name(path.name.replace('mhd_w_bcc','mhd_divb')))
  assert dd['time']==d['time'] and np.array_equal(dd['mb_geometry'],d['mb_geometry'])
  div=df['divb'];assert div.shape==B.shape and np.isfinite(div).all()
  dx=np.min(np.array([(d['mb_geometry'][:,2*i+1]-d['mb_geometry'][:,2*i])/d[f'nx{i+1}_mb'] for i in range(3)]),axis=0)
  gamma=float(cfg['mhd/gamma']);cs2=2/float(cfg['problem/r_bondi']);pinf=float(cfg['problem/rho_inf'])*cs2/(gamma*(1-cs2/(gamma-1)))
  ambient=np.sqrt(2*pinf/float(cfg['problem/beta_inf']))
  scaled=float(np.max(abs(div)*dx[:,None,None,None]/np.maximum(B,.001*ambient)))
  assert scaled<1e-9,(case,scaled)
  states.append(dict(time=d['time'],cycle=d['cycle'],min_density=float(f['dens'].min()),min_pressure=float(f['press'].min()),max_scaled_divB=scaled))
  final=d,f,cfg,sl
 assert len(states)>=2 and final is not None
 d,f,cfg,sl=final
 if target is None:target=.05 if case.name.startswith('curved-amr') else 180
 assert d['time']==target,(case,d['time'])
 expected=boundary or ('characteristic' if ('characteristic' in case.name or 'firstorder' in case.name) else 'fixed')
 assert cfg['problem/reservoir_boundary']==expected
 if expected=='characteristic':assert faces>0
 result=dict(name=case.name,states=states,rss_plus_tracked_bytes=rss+tracked,max_rank_seconds=max(seconds),limited_faces_in_last_logged_counters=limited,faces_in_last_logged_counters=faces)
 if plane and not case.name.startswith('curved-amr'):
  assert cfg['problem/reservoir_test_flat_metric']=='true'
  gamma=float(cfg['mhd/gamma']);rho=float(cfg['problem/rho_inf']);cs2=2/float(cfg['problem/r_bondi']);pinf=rho*cs2/(gamma*(1-cs2/(gamma-1)));Binf=np.sqrt(2*pinf/float(cfg['problem/beta_inf']));inertia=np.sqrt(rho+gamma*pinf/(gamma-1)+Binf**2)
  velocity=np.array([f['vel'+c][(slice(None),)+sl] for c in 'xyz'],dtype=float);velocity/=np.sqrt(1+np.sum(velocity**2,axis=0))
  dens=f['dens'][(slice(None),)+sl];Bx=f['bcc1'][(slice(None),)+sl]
  incoming=(dens/rho-1-velocity[2]/np.sqrt(cs2))/2 if (wave_mode=='acoustic' or case.name.startswith('acoustic')) else (velocity[0]+Bx/inertia)/2
  profiles=[];density_range={}
  for mb,box in enumerate(d['mb_geometry']):
   zc=box[4]+(np.arange(d['nx3_mb'])+.5)*(box[5]-box[4])/d['nx3_mb']
   for iz,z in enumerate(zc):
    profiles.append([float(z),float(incoming[mb,iz].mean())]);density_range.setdefault(round(float(z),10),[]).append([float(dens[mb,iz].min()),float(dens[mb,iz].max())])
  spread=max(max(v[1] for v in values)-min(v[0] for v in values) for values in density_range.values())
  assert spread<2e-6,(case,'broken transverse periodicity',spread)
  result.update(profile=sorted(profiles),time=d['time'],seed_amplitude=float(cfg['problem/reservoir_test_pulse']),transverse_density_spread=spread)
 return result

def compare_curved(runs):
 """Same saved topology and interior cells: a short integration-isolation check."""
 pair=[]
 for boundary in ('fixed','characteristic'):
  path=sorted((runs/f'curved-amr-{boundary}'/'bin').glob('*.mhd_w_bcc.*.bin'))[-1]
  pair.append(load(path))
 (a,af),(b,bf)=pair
 assert a['time']==b['time']==.05
 for key in ('mb_geometry','mb_logical'):
  assert np.array_equal(a[key],b[key]),key
 assert all(a[f'nx{i}_out_mb']==a[f'nx{i}_mb'] for i in (1,2,3))
 xyz=[]
 for i in range(3):
  n=a[f'nx{i+1}_mb'];box=a['mb_geometry']
  xyz.append(box[:,2*i,None]+(np.arange(n)+.5)*(box[:,2*i+1]-box[:,2*i])[:,None]/n)
 radius2=xyz[0][:,None,None,:]**2+xyz[1][:,None,:,None]**2+xyz[2][:,:,None,None]**2
 mask=radius2<100**2
 fields={}
 for key in af:
  av,bv=af[key][mask].astype(float),bf[key][mask].astype(float)
  error=float(np.max(abs(av-bv)));scale=max(float(np.max(abs(av))),float(np.max(abs(bv))),1e-30)
  fields[key]=dict(max_abs_difference=error,relative_to_field_max=error/scale,bitwise_equal=bool(np.array_equal(av,bv)))
 return dict(time=.05,radius=100,cell_count=int(mask.sum()),identical_topology=True,fields=fields,
             scope='Short integration-isolation check; not a curved-space reflection or long-time test')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--runs',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 cases=sorted([d for d in a.runs.iterdir() if d.is_dir() and (d.name.startswith(('acoustic-','alfven-','curved-amr-')))])
 assert len(cases)==14,('Missing tests',len(cases))
 rows=[]
 for case in cases:
  row=audit(case);rows.append(row);print(json.dumps({k:v for k,v in row.items() if k!='profile'}),flush=True)
 (a.out/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
 (a.out/'curved_inner_comparison.json').write_text(json.dumps(compare_curved(a.runs),indent=2)+'\n')
