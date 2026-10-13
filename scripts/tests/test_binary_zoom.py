#!/usr/bin/env python3
"""Bounded local integration checks for the prescribed circular binary/zoom prototype.

Run under run_bounded.py. Requires a PROBLEM=dynbbh executable and numpy.
"""
import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'vis/python'))
from bin_convert import read_binary


def load(path):
    data = read_binary(str(path))
    return data, {k: np.asarray(v) for k, v in data['mb_data'].items()}


def final_output(case, variable):
    return sorted((case/'bin').glob(f'*.{variable}.*.bin'))[-1]


def coords(data):
    geom = data['mb_geometry']
    nx, ny, nz = (data[f'nx{i}_out_mb'] for i in (1, 2, 3))
    x = geom[:, 0, None]+(np.arange(nx)[None, :]+0.5)*(geom[:, 1]-geom[:, 0])[:, None]/nx
    y = geom[:, 2, None]+(np.arange(ny)[None, :]+0.5)*(geom[:, 3]-geom[:, 2])[:, None]/ny
    z = geom[:, 4, None]+(np.arange(nz)[None, :]+0.5)*(geom[:, 5]-geom[:, 4])[:, None]/nz
    return x[:, None, None, :], y[:, None, :, None], z[:, :, None, None]


def inspect(case):
    metrics = []
    for path in sorted((case/'bin').glob('*.mhd_w_bcc.*.bin')):
        data, fields = load(path)
        assert all(np.isfinite(a).all() for a in fields.values()), path
        density = fields['dens']
        pressure = fields.get('press', fields.get('eint'))
        assert density.min() > 0 and pressure.min() > 0, path
        metrics.append(dict(cycle=data['cycle'], time=data['time'], blocks=data['n_mbs'],
                            max_level=int(data['mb_logical'][:, 3].max()),
                            min_density=float(density.min()), min_pressure=float(pressure.min())))
    divmax = 0.0
    for path in (case/'bin').glob('*.mhd_divb.*.bin'):
        _, fields = load(path)
        for a in fields.values():
            assert np.isfinite(a).all(), path
            divmax = max(divmax, float(abs(a).max()))
    assert divmax < 1e-9, (case, divmax)
    return dict(states=metrics, max_abs_divb=divmax)


def make_deep_input(base):
    text = base.read_text().replace('num_levels = 9', 'num_levels = 18')
    text = text.replace('nlevels = 9', 'nlevels = 18').replace('level = 8', 'level = 17')
    text = text.replace('max_nmb_per_rank = 2048', 'max_nmb_per_rank = 4096')
    text = text.replace('max_nzmb_per_host = 2048', 'max_nzmb_per_host = 4096')
    text = text.replace(' = -256', ' = -1048576').replace(' = 256', ' = 1048576')
    text = text.replace('31.75', '4095.75').replace('32.25', '4096.25')
    text = text.replace('x1c = 32', 'x1c = 4096').replace('x1c2 = -32', 'x1c2 = -4096')
    text = text.replace('r_0 = 2', 'r_0 = 16')
    return text.replace('sep = 64', 'sep = 8192').replace('r_bondi = 2048', 'r_bondi = 1048576')


def compare_restart(reference, resumed):
    errors = {}
    for variable in ['mhd_w_bcc', 'adm', 'mhd_divb']:
        a, av = load(final_output(reference, variable))
        b, bv = load(final_output(resumed, variable))
        assert a['cycle'] == b['cycle'] and a['time'] == b['time'], variable
        assert np.array_equal(a['mb_logical'], b['mb_logical']), variable
        for key in av:
            error = float(abs(av[key]-bv[key]).max())
            scale = max(float(abs(av[key]).max()), float(abs(bv[key]).max()), 1e-30)
            errors[key] = dict(max_abs_error=error, relative_to_field_max=error/scale)
            if variable == 'adm':
                assert np.array_equal(av[key], bv[key]), key
            elif variable == 'mhd_divb':
                assert error < 1e-12, (key, error)
            else:
                # Bound the accumulated fluid difference after ghost reconstruction
                # and primitive recovery. Record the residual, rather than asserting
                # bitwise fluid reproducibility from float32 visualization output.
                tolerance = 1e-5*scale+1e-13
                assert error <= tolerance, (key, error, tolerance)
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exe', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--skip-deep', action='store_true')
    parser.add_argument('--quick-only', action='store_true')
    parser.add_argument('--deep-only', action='store_true')
    args = parser.parse_args()
    exe, out = args.exe.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    base = ROOT/'inputs/dyngr/dynbbh_bondi_zoom.athinput'
    results = {}

    def run(name, flags=(), input_file=base, restart=None, expect_failure=None):
        case = out/name
        if case.exists():
            shutil.rmtree(case)
        case.mkdir()
        cmd = [str(exe), '-i', str(input_file), '-d', str(case)]
        if restart is not None:
            cmd += ['-r', str(restart)]
        cmd += list(flags)
        started = time.monotonic()
        with (case/'run.log').open('w') as log:
            proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=360)
        elapsed = time.monotonic()-started
        text = (case/'run.log').read_text()
        if expect_failure:
            assert proc.returncode != 0 and expect_failure in text, (name, text[-3000:])
        else:
            assert proc.returncode == 0, (name, proc.returncode, text[-3000:])
            assert not re.search(r'NaN|finite-check failure|nonfinite.*repair', text), text[-3000:]
        results[name] = dict(seconds=elapsed, returncode=proc.returncode)
        print(f'{name}: {elapsed:.2f}s, exit={proc.returncode}', flush=True)
        return case

    def check_deep():
        deep_input = out/'deep.athinput'
        deep_input.write_text(make_deep_input(base))
        deep = run('deep18', ['time/nlim=34', 'output1/dcycle=1', 'output3/dcycle=1'],
                   input_file=deep_input)
        results['deep18'].update(inspect(deep))
        levels = [s['max_level'] for s in results['deep18']['states']]
        assert max(levels) == 17 and min(levels) == 0 and levels[-1] == 17, levels
        first, _ = load(sorted((deep/'bin').glob('*.mhd_w_bcc.*.bin'))[0])
        last, _ = load(final_output(deep, 'mhd_w_bcc'))
        assert np.array_equal(first['mb_logical'], last['mb_logical'])

    if args.deep_only:
        check_deep()
        (out/'results.json').write_text(json.dumps(results, indent=2)+'\n')
        print('Deep binary zoom checks passed.', flush=True)
        return

    # Exact coordinate transformation: compare two independently evaluated ADM backgrounds.
    probe = ['mesh_refinement/refinement=none', 'mesh/nx1=4', 'mesh/nx2=4', 'mesh/nx3=4',
             'mesh/x1min=700', 'mesh/x1max=708', 'mesh/x2min=600', 'mesh/x2max=608',
             'mesh/x3min=5', 'mesh/x3max=13', 'time/nlim=0']
    inertial = run('metric_inertial', probe+['problem/corotating=false', 'adm/dynamic=true'])
    rotating = run('metric_rotating', probe)
    later = run('high_shift', [v for v in probe if v != 'time/nlim=0']+['time/nlim=4'])
    finite_difference = run('metric_fd', probe+['problem/metric_derivative=fd'])
    geom, rot = load(final_output(rotating, 'adm'))
    _, lab = load(final_output(inertial, 'adm'))
    _, late = load(final_output(later, 'adm'))
    _, fd = load(final_output(finite_difference, 'adm'))
    x, y, _ = coords(geom)
    omega = 64**-1.5
    errors = {}
    for key in rot:
        if key == 'adm_betax':
            expected = lab[key]-omega*y
        elif key == 'adm_betay':
            expected = lab[key]+omega*x
        else:
            expected = lab[key]
        errors[key] = float(abs(rot[key]-expected).max())
        assert np.allclose(rot[key], expected, rtol=3e-6, atol=3e-10), (key, errors[key])
        assert np.array_equal(rot[key], late[key]), key
        assert np.allclose(rot[key], fd[key], rtol=3e-5, atol=2e-8), key
    assert np.max(abs(rot['adm_betay'])) > 1.0  # grid lies beyond the light cylinder
    step_text = (rotating/'run.log').read_text()
    dt = float(re.search(r'cycle=0 .*?dt=(\S+)', step_text).group(1))
    assert dt < 0.2*2/(1+np.max(abs(rot['adm_betay'])))*1.01, dt
    results['metric_transform'] = dict(max_error=max(errors.values()), coordinate_dt=dt)
    # Evolve beyond the light cylinder: coordinate shifts >c are allowed, CFL must shrink.
    results['high_shift'].update(inspect(later))

    run('reject_tilted', probe+['problem/a1=0.5', 'problem/th_a1=30'],
        expect_failure='corotating requires')
    run('reject_dynamic_zoom', ['adm/dynamic=true'], expect_failure='stationary prescribed ADM')
    run('reject_levels', ['amr_criterion1/nlevels=1'], expect_failure='Invalid cyclic zoom levels')
    run('reject_boundaries', probe+['mesh/ix2_bc=periodic', 'mesh/ox2_bc=periodic'],
        expect_failure='six user reservoir boundaries')

    # Fast magnetized restart regression, including a pending coarsening event.
    small_text = base.read_text().replace('num_levels = 9', 'num_levels = 3')
    small_text = small_text.replace('nlevels = 9', 'nlevels = 3').replace('level = 8', 'level = 2')
    small_text = small_text.replace('max_nmb_per_rank = 2048', 'max_nmb_per_rank = 512')
    small_text = small_text.replace('max_nzmb_per_host = 2048', 'max_nzmb_per_host = 512')
    small_text = small_text.replace('max_nzmb_per_dvce = 512', 'max_nzmb_per_dvce = 256')
    small_text = small_text.replace('r_0 = 2', 'r_0 = 32').replace('dcycle = 4', 'dcycle = 1')
    small_input = out/'small.athinput'
    small_input.write_text(small_text)
    small = run('small', ['time/nlim=5'], input_file=small_input)
    results['small'].update(inspect(small))
    checkpoint = sorted((small/'rst').glob('*.rst'))[1]
    small_restart = run('small_restart', ['time/nlim=5'], input_file=small_input,
                        restart=checkpoint)
    for variable in ['mhd_w_bcc', 'adm', 'mhd_divb']:
        a, av = load(final_output(small, variable))
        b, bv = load(final_output(small_restart, variable))
        assert a['time'] == b['time'] and np.array_equal(a['mb_logical'], b['mb_logical'])
        for key in av:
            assert np.allclose(av[key], bv[key], rtol=3e-11, atol=3e-13), (variable, key)
    if args.quick_only:
        (out/'results.json').write_text(json.dumps(results, indent=2)+'\n')
        print('Quick binary zoom checks passed.', flush=True)
        return

    # Nine total physical levels with 4^3 blocks, all the way out and back in.
    full = run('roundtrip', ['time/nlim=16', 'output1/dcycle=1', 'output3/dcycle=1'])
    checks = inspect(full)
    results['roundtrip'].update(checks)
    levels = [v['max_level'] for v in checks['states']]
    assert min(levels) == 0 and max(levels) == 8 and levels[-1] == 8, levels
    first, _ = load(sorted((full/'bin').glob('*.mhd_w_bcc.*.bin'))[0])
    last, _ = load(final_output(full, 'mhd_w_bcc'))
    assert np.array_equal(first['mb_logical'], last['mb_logical'])
    # A scheduled checkpoint is before the AMR event for that cycle.
    checkpoint = sorted((full/'rst').glob('*.rst'))[1]  # cycle 4
    resumed = run('restart_pre_amr', ['time/nlim=16', 'output1/dcycle=1', 'output3/dcycle=1'],
                  restart=checkpoint)
    results['restart_pre_amr'].update(inspect(resumed))
    results['restart_pre_amr']['comparison'] = compare_restart(full, resumed)
    # A final checkpoint is after AMR. Zero-interval events must not repeat on restart.
    checkpoint = sorted((full/'rst').glob('*.rst'))[-1]
    continued = run('continued', ['time/nlim=21'])
    resumed = run('restart_post_amr', ['time/nlim=21'], restart=checkpoint)
    results['restart_post_amr'].update(inspect(resumed))
    results['restart_post_amr']['comparison'] = compare_restart(continued, resumed)

    # A checkpoint in the second outward sweep must exclude stale snapshot counts.
    checkpoint = sorted((continued/'rst').glob('*.rst'))[-2]  # scheduled cycle 20
    second = run('restart_second_sweep', ['time/nlim=21'], restart=checkpoint)
    results['restart_second_sweep'].update(inspect(second))
    results['restart_second_sweep']['comparison'] = compare_restart(continued, second)

    if not args.skip_deep:
        check_deep()
    (out/'results.json').write_text(json.dumps(results, indent=2)+'\n')
    print('All binary zoom checks passed.', flush=True)


if __name__ == '__main__':
    main()
