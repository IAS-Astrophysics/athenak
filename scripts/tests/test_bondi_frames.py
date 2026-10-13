#!/usr/bin/env python3
"""Compare inertial and rotating Michel evolution, including a superluminal shift.

Run under run_bounded.py; requires a dynbbh executable and NumPy.
"""
import argparse
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from test_binary_zoom import ROOT, coords, inspect, load, final_output


def error(a, b, mask):
    return float(np.mean(abs(a[mask]-b[mask]))/np.mean(abs(b[mask])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exe', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    exe, out = args.exe.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    base = ROOT/'inputs/dyngr/bondi_matched_zoom.athinput'
    results = {}
    for nx in (16, 32):
        states = {}
        for frame, omega in [('inertial', 0.0), ('rotating', 0.2)]:
            case = out/f'{frame}{nx}'
            case.mkdir(exist_ok=True)
            flags = ['mesh_refinement/refinement=none', 'problem/single_black_hole=true',
                     f'problem/corotating={str(omega != 0).lower()}',
                     f'problem/frame_omega={omega}', 'time/tlim=1', 'time/nlim=2000',
                     'time/ndiag=100', 'output1/dt=0.5', 'output2/dt=1', 'output3/dt=1']
            for dim in (1, 2, 3):
                flags += [f'mesh/nx{dim}={nx}', f'mesh/x{dim}min=-8',
                          f'mesh/x{dim}max=8']
            started = time.monotonic()
            with (case/'run.log').open('w') as log:
                proc = subprocess.run([str(exe), '-i', str(base), '-d', str(case)]+flags,
                                      stdout=log, stderr=subprocess.STDOUT, timeout=180)
            assert proc.returncode == 0, (case, (case/'run.log').read_text()[-2500:])
            data, last = load(final_output(case, 'mhd_w_bcc'))
            _, first = load(sorted((case/'bin').glob('*.mhd_w_bcc.*.bin'))[0])
            _, adm = load(final_output(case, 'adm'))
            x, y, z = coords(data)
            radius = np.sqrt(x*x+y*y+z*z)
            mask = (radius > 3) & (radius < 7)
            record = inspect(case)
            record['seconds'] = time.monotonic()-started
            record['density_L1'] = error(last['dens'], first['dens'], mask)
            record['pressure_L1'] = error(last['press'], first['press'], mask)
            # Recover inertial coordinate velocities, explicitly retaining beta.
            u = np.stack([last[f'vel{a}'] for a in 'xyz'])
            g = np.array([[adm['adm_gxx'], adm['adm_gxy'], adm['adm_gxz']],
                          [adm['adm_gxy'], adm['adm_gyy'], adm['adm_gyz']],
                          [adm['adm_gxz'], adm['adm_gyz'], adm['adm_gzz']]])
            w = np.sqrt(1+np.einsum('ab...,a...,b...->...', g, u, u))
            beta = np.stack([adm[f'adm_beta{a}'] for a in 'xyz'])
            vc = adm['adm_alpha']*u/w-beta
            vc[0] -= omega*y
            vc[1] += omega*x
            radial = (vc[0]*x+vc[1]*y+vc[2]*z)/radius
            tangent = np.sqrt(np.maximum(np.sum(vc*vc, axis=0)-radial*radial, 0))
            record['tangential_to_radial_L1'] = float(np.mean(tangent[mask])/
                                                     np.mean(abs(radial[mask])))
            assert data['time'] == 1.0
            assert record['density_L1'] < 0.1 and record['pressure_L1'] < 0.1, record
            results[f'{frame}{nx}'] = record
            states[frame] = last
            print(f'{frame}{nx}: {record["seconds"]:.2f}s, '
                  f'density L1={record["density_L1"]:.3e}', flush=True)
        results[f'comparison{nx}'] = {
            key: error(states['rotating'][key], states['inertial'][key], mask)
            for key in ('dens', 'press')}
        assert max(results[f'comparison{nx}'].values()) < 0.05
    for frame in ('inertial', 'rotating'):
        for key in ('density_L1', 'pressure_L1'):
            assert results[f'{frame}32'][key] < results[f'{frame}16'][key], results
    for key in ('dens', 'press'):
        assert results['comparison32'][key] < results['comparison16'][key], results
    # Independently evaluate the original upstream/Minghao pgen in this executable.
    case=out/'upstream32'
    case.mkdir(exist_ok=True)
    flags=['mesh_refinement/refinement=none', 'problem/r_crit=8',
           'problem/user_hist=false', 'time/tlim=1', 'time/nlim=1000']
    for dim in (1, 2, 3):
        flags += [f'mesh/nx{dim}=32', f'meshblock/nx{dim}=8',
                  f'mesh/x{dim}min=-8', f'mesh/x{dim}max=8']
    with (case/'run.log').open('w') as log:
        proc=subprocess.run([str(exe),'-i',str(ROOT/'inputs/zoom/zoom_bondi.athinput'),
                             '-d',str(case)]+flags,
                            stdout=log,stderr=subprocess.STDOUT,timeout=60)
    assert proc.returncode == 0
    _, old=load(sorted((case/'bin').glob('*.hydro_w.*.bin'))[0])
    _, new=load(sorted((out/'inertial32/bin').glob('*.mhd_w_bcc.*.bin'))[0])
    reference_errors={}
    for key in ('dens','press','velx','vely','velz'):
        reference=old['eint']*(2/3) if key == 'press' else old[key]
        reference_errors[key]=float(np.max(abs(new[key][mask]-reference[mask])))
        assert np.allclose(new[key][mask],reference[mask],rtol=2e-6,atol=1e-12), key
    results['upstream_initial_max_errors']=reference_errors
    # A restart must restore the analytic constants used by the upstream BC.
    # Split on a cycle, so both runs have the same timestep history.
    restart_states = {}
    checkpoint = None
    for name, limit in [('seed', 4), ('continuous', 1000), ('resumed', 1000)]:
        restart_case = out/('upstream_restart_'+name)
        restart_case.mkdir(exist_ok=True)
        command = [str(exe), '-d', str(restart_case)]
        if name == 'resumed':
            command += ['-r', str(checkpoint), 'time/tlim=2', 'time/nlim=1000']
        else:
            command += ['-i', str(ROOT/'inputs/zoom/zoom_bondi.athinput')]
            command += flags+['time/tlim=2', 'time/nlim='+str(limit)]
        with (restart_case/'run.log').open('w') as log:
            proc = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                  timeout=60)
        assert proc.returncode == 0, (restart_case/'run.log').read_text()[-2000:]
        data, fields = load(final_output(restart_case, 'hydro_w'))
        assert all(np.isfinite(a).all() for a in fields.values())
        assert fields['dens'].min() > 0 and fields['eint'].min() > 0
        if name == 'seed':
            checkpoint = sorted((restart_case/'rst').glob('*.rst'))[-1]
        else:
            assert data['time'] == 2.0
            restart_states[name] = (data, fields)
    reference_data, reference = restart_states['continuous']
    resumed_data, resumed = restart_states['resumed']
    assert reference_data['cycle'] == resumed_data['cycle']
    restart_errors = {}
    for key in reference:
        scale = max(float(abs(reference[key]).max()), 1e-30)
        restart_errors[key] = float(abs(reference[key]-resumed[key]).max())/scale
        assert restart_errors[key] < 1e-5, (key, restart_errors[key])
    results['upstream_restart_relative_max_errors'] = restart_errors
    # Exercise a complete upstream hydro zoom sweep with a fixed block budget.
    # Previously its geometric return sweep refined unrelated siblings.
    zoom_case = out/'upstream_zoom_roundtrip'
    zoom_case.mkdir(exist_ok=True)
    zoom_input = out/'upstream_zoom.athinput'
    zoom_input.write_text((ROOT/'inputs/zoom/zoom_bondi.athinput').read_text().replace(
        '<amr_criterion1>', '<amr_criterion1>\nf_exc = 0'))
    zoom_flags = ['problem/r_crit=8', 'problem/user_hist=false',
                  'mesh_refinement/num_levels=6', 'refined_region1/level=5',
                  'mesh_refinement/max_nmb_per_rank=600',
                  'amr_criterion1/nlevels=4', 'amr_criterion1/trun_fac=0',
                  'amr_criterion1/trun_fac_0=0', 'amr_criterion1/trun_fac_1=0',
                  'amr_criterion1/trun_fac_2=0', 'amr_criterion1/trun_pow=0',
                  'cyclic_zoom/verbose=false', 'time/tlim=100', 'time/nlim=6',
                  'output4/dcycle=1']
    for dim in (1, 2, 3):
        zoom_flags += [f'mesh/nx{dim}=16', f'meshblock/nx{dim}=8',
                       f'mesh/x{dim}min=-16', f'mesh/x{dim}max=16']
    with (zoom_case/'run.log').open('w') as log:
        proc = subprocess.run([str(exe), '-i',
                               str(zoom_input),
                               '-d', str(zoom_case)]+zoom_flags,
                              stdout=log, stderr=subprocess.STDOUT, timeout=90)
    assert proc.returncode == 0, (zoom_case/'run.log').read_text()[-2500:]
    zoom_paths = sorted((zoom_case/'bin').glob('*.hydro_w.*.bin'))
    initial_data, _ = load(zoom_paths[0])
    final_data, _ = load(zoom_paths[-1])
    assert np.array_equal(initial_data['mb_logical'], final_data['mb_logical'])
    for path in zoom_paths:
        _, fields = load(path)
        assert all(np.isfinite(a).all() for a in fields.values()), path
        assert fields['dens'].min() > 0 and fields['eint'].min() > 0, path
    results['upstream_zoom_roundtrip'] = {
        'initial_blocks': initial_data['n_mbs'], 'final_blocks': final_data['n_mbs'],
        'cycle': final_data['cycle'], 'time': final_data['time'],
        'topology_restored': True}
    (out/'results.json').write_text(json.dumps(results, indent=2)+'\n')
    print('Rotating/inertial Bondi checks passed.', flush=True)


if __name__ == '__main__':
    main()
