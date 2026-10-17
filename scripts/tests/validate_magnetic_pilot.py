#!/usr/bin/env python3
"""Check a completed 12-tile magnetic pilot without interpreting it as equilibrium."""
import argparse
import json
from pathlib import Path
import re
import numpy as np
from test_binary_zoom import load, coords

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--run', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
p.add_argument('--roundtrip', action='store_true')
a = p.parse_args()
rss = allocations = budget = 0
ranks = []
hosts = set()
termination = None
for rank in range(12):
    resource = json.loads((a.run/f'resources-{rank}.json').read_text())
    memory = json.loads((a.run/f'memory-{rank}.json').read_text())
    assert resource['returncode'] == 0 and resource['stopped'] is None, (rank, resource)
    assert resource['peak_rss_bytes'] <= resource['memory_budget_bytes']
    assert memory['combined_peak_bytes'] <= memory['budget_bytes']
    text = (a.run/f'rank-{rank}.log').read_text()
    affinity = f'{rank//2}.{rank%2}'
    identity = re.search(rf'rank={rank} host=(\S+) affinity={re.escape(affinity)}\s', text)
    assert identity
    hosts.add(identity.group(1))
    if rank == 0:
        ended = re.search(r'Terminating on (cycle limit|time limit|wall clock limit)',text)
        assert ended, 'Missing normal solver termination'
        termination = ended.group(1)
        if not a.roundtrip: assert termination != 'cycle limit'
    if not a.roundtrip:
        command = resource['command']
        assert command[command.index('-t')+1] == '00:55:00'
        for flag in command:
            if flag.startswith('amr_criterion1/residence_scale='):
                assert float(flag.split('=',1)[1]) == 1.0
    assert not re.search(r'NaN|finite-check failure|nonfinite.*repair', text)
    rss += resource['peak_rss_bytes']
    allocations += memory['combined_peak_bytes']
    budget += resource['memory_budget_bytes']+memory['budget_bytes']
    ranks.append({'rank': rank, 'affinity': affinity, 'rss_bytes': resource['peak_rss_bytes'],
                  'tracked_bytes': memory['combined_peak_bytes'], 'seconds': resource['seconds'],
                  'metric_evaluations': memory.get('metric_evaluations')})
assert len(hosts) == 1, hosts
assert budget <= 30_000_000_000 and rss+allocations < 30_000_000_000
paths = sorted((a.run/'bin').glob('*.mhd_w_bcc.*.bin'))
assert len(paths) >= 2, 'Need both initial and evolved fluid data'
selected = paths  # Validate every saved fluid state, including intermediate stages.
states = []
initial = None
restored = False
different_topology_seen = False
maxdiv = max_scaled_div = max_exterior_div = 0.
field_floor = None
excision_spheres = None
paired_div = set()
for path in selected:
    d, f = load(path)
    assert all(np.isfinite(v).all() for v in f.values()), path
    assert f['dens'].min() > 0 and f['press'].min() > 0, path
    bnorm = np.sqrt(sum(f[f'bcc{i}'].astype(np.float64)**2 for i in (1,2,3)))
    bmax = float(bnorm.max())
    dx=(d['mb_geometry'][:,1]-d['mb_geometry'][:,0])/d['nx1_mb']
    if field_floor is None:
        params={};block=''
        for line in d['header']:
            line=line.split('#',1)[0].strip()
            if line.startswith('<') and line.endswith('>'):block=line[1:-1]
            elif '=' in line:
                key,value=line.split('=',1);params[block+'/'+key.strip()]=value.strip()
        if params['problem/single_black_hole']=='true':
            centers=[0.]
        else:
            assert float(params['problem/q'])==1.
            sep=float(params['problem/sep']);centers=[sep/2,-sep/2]
        excision_spheres=[{'center':center,'radius':float(params[f'coord/excise_{i+1}_rad'])}
                          for i,center in enumerate(centers)]
        ambient=float(np.median(bnorm[dx==dx.max()]))
        assert ambient>0
        field_floor=1e-3*ambient
    divpath=path.with_name(path.name.replace('.mhd_w_bcc.','.mhd_divb.'))
    dd,df=load(divpath);paired_div.add(divpath)
    for key in ('mb_logical','mb_geometry'):
        assert np.array_equal(d[key],dd[key]), 'Unmatched fluid/divB topology'
    assert (d['cycle'],d['time'])==(dd['cycle'],dd['time'])
    div=df['divb'];assert np.isfinite(div).all() and div.shape==bnorm.shape
    raw=float(abs(div).max())
    scaled=float((abs(div)*dx[:,None,None,None]/np.maximum(bnorm,field_floor)).max())
    x,y,z=np.broadcast_arrays(*coords(d))
    exterior=np.ones_like(div,dtype=bool)
    for sphere in excision_spheres:
        exterior &= (x-sphere['center'])**2+y*y+z*z>=sphere['radius']**2
    external=float(abs(div[exterior]).max())
    maxdiv=max(maxdiv,raw);max_scaled_div=max(max_scaled_div,scaled)
    max_exterior_div=max(max_exterior_div,external)
    assert bmax > 0, 'This must be genuinely magnetized'
    if initial is None:
        initial = d
    elif not np.array_equal(initial['mb_logical'], d['mb_logical']):
        different_topology_seen = True
    elif different_topology_seen:
        restored = True
    states.append({'file': path.name, 'time': d['time'], 'cycle': d['cycle'],
                   'blocks': d['n_mbs'], 'max_level': int(d['mb_logical'][:,3].max()),
                   'min_density': float(f['dens'].min()), 'min_pressure': float(f['press'].min()),
                   'max_Btilde_coordinate_norm': bmax,
                   'max_Btilde_exterior':float(bnorm[exterior].max()),
                   'max_abs_divb':raw,'max_abs_divb_exterior':external,
                   'max_scaled_divb':scaled})
assert states[-1]['time'] > states[0]['time'] and states[-1]['cycle'] > 0
if a.roundtrip:
    assert restored, 'No return to the initial resolved topology was saved'
    assert len({s['max_level'] for s in states}) >= 8, 'Not all eight zoom stages sampled'
assert paired_div == set((a.run/'bin').glob('*.mhd_divb.*.bin'))
# CT is a relative face-flux constraint. The raw absolute error grows with
# the very large fields inside puncture excision; retain both diagnostics.
assert max_scaled_div < 1e-9, max_scaled_div
result = {'ranks': ranks, 'hosts': sorted(hosts), 'termination': termination, 'aggregate_peak_rss_bytes': rss,
          'sum_tracked_allocation_peaks_bytes': allocations,
          'conservative_sum_bytes': rss+allocations, 'sum_guard_budgets_bytes': budget,
          'max_abs_divb': maxdiv, 'max_abs_divb_exterior':max_exterior_div,
          'max_scaled_divb':max_scaled_div,'scaled_divb_definition':'dx*abs(divBtilde)/max(norm(Btilde),0.001*ambient)',
          'field_normalization_floor':field_floor, 'diagnostic_excision_spheres':excision_spheres, 'states': states, 'restored_topology': restored,
          'interpretation': 'bounded magnetized evolution; no relaxation claim'}
a.out.parent.mkdir(parents=True, exist_ok=True)
a.out.write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('ranks','states')}, indent=2))
print('Final fluid state:', states[-1])
