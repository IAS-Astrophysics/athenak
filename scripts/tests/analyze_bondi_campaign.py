#!/usr/bin/env python3
"""Check saved Aurora Bondi fields and quantify frame/configuration differences."""
import argparse
import json
from pathlib import Path

import numpy as np

from test_binary_zoom import coords, load


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--tlim',type=float,default=4.0)
    args=parser.parse_args()
    root=args.root.resolve()
    result={}
    states={}
    for name in ('upstream','single_inertial','single_rotating','binary_rotating'):
        case=root/name
        variable='hydro_w' if name == 'upstream' else 'mhd_w_bcc'
        paths=sorted((case/'bin').glob('*.'+variable+'.*.bin'))
        assert len(paths) >= 2, (name,paths)
        first, initial=load(paths[0])
        last, fields=load(paths[-1])
        minima={}
        for path in paths:
            _, f=load(path)
            assert all(np.isfinite(a).all() for a in f.values()), path
            assert np.min(f['dens']) > 0, path
            p=f.get('press',f.get('eint'))
            assert np.min(p) > 0, path
            minima[path.name]={'density':float(f['dens'].min()),'thermal':float(p.min())}
        record={'time':last['time'],'cycle':last['cycle'],
                'initial_blocks':first['n_mbs'],'final_blocks':last['n_mbs'],
                'topology_restored':bool(np.array_equal(first['mb_logical'],last['mb_logical'])),
                'minima':minima}
        assert last['time'] == args.tlim, ('Incomplete physical-time target', name, last['time'])
        assert record['topology_restored'], record
        for f in ('resources.json','allocations.json'):
            record[f]=json.loads((case/f).read_text())
        record['max_divb']=0.0
        for path in (case/'bin').glob('*.mhd_divb.*.bin'):
            _, f=load(path)
            record['max_divb']=max(record['max_divb'],
                                   max(float(abs(a).max()) for a in f.values()))
        assert record['max_divb'] < 1e-9
        assert record['resources.json']['returncode'] == 0
        assert record['allocations.json']['combined_peak_bytes'] < 24e9
        if name != 'upstream':
            assert record['allocations.json']['metric_evaluations'] == 7, record
            x,y,z=coords(last)
            r=np.sqrt(x*x+y*y+z*z)
            vol=((last['mb_geometry'][:,1]-last['mb_geometry'][:,0])/
                 last['nx1_out_mb'])**3
            weights=np.broadcast_to(vol[:,None,None,None],r.shape)
            for lo,hi in ((3,32),(64,192)):
                mask=(r>lo)&(r<hi)
                for key in ('dens','press'):
                    record[f'{key}_L1_r{lo}_{hi}']=float(
                        np.sum(weights[mask]*abs(fields[key][mask]-initial[key][mask]))/
                        np.sum(weights[mask]*abs(initial[key][mask])))
            states[name]=(last,fields,r,weights)
        result[name]=record
    for other in ('single_inertial','binary_rotating'):
        a,av,r,weights=states['single_rotating']
        b,bv,_,_=states[other]
        assert a['time'] == b['time'] and np.array_equal(a['mb_logical'],b['mb_logical'])
        comparison={}
        for lo,hi in ((3,32),(64,192)):
            mask=(r>lo)&(r<hi)
            comparison[f'r{lo}_{hi}']={
                key:float(np.sum(weights[mask]*abs(av[key][mask]-bv[key][mask]))/
                          np.sum(weights[mask]*abs(bv[key][mask])))
                for key in ('dens','press')}
            if other == 'single_inertial':
                assert max(comparison[f'r{lo}_{hi}'].values()) < 0.05, comparison
        result['single_rotating_vs_'+other]=comparison
    (root/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:{p:v for p,v in d.items() if p != 'minima'}
                      for k,d in result.items()},indent=2))


if __name__ == '__main__':
    main()
