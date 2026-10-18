#!/usr/bin/env python3
"""Shared, finite-support rotating-boundary test definitions; no submissions."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from prepare_characteristic_control import prepare
OMEGA=32**-1.5

def prepared_input(source):
    text=prepare(source,characteristic=True)
    # Required explicit keys for bounded/forced-FOFC controls.
    text=text.replace('<mhd>','<mhd>\ndmp_M = 1.2')
    return text

def flags(family,frame,choice,n,block_z=8):
    assert family in ('acoustic3d','entropy')
    assert frame in ('inertial','rotating','fastrotation')
    assert choice in ('reference','fixed','characteristic','firstorder')
    reference=choice=='reference';localized=family=='entropy' or frame=='fastrotation';half=((16 if localized else 24) if reference else 8)
    omega={'inertial':0.,'rotating':OMEGA,'fastrotation':2**.5}[frame]
    entropy=family=='entropy';target=.4 if localized else 180.
    center=(6.,6.,0.) if localized else (0.,0.,4.)
    width=1. if localized else 2.5
    # Flat rotating ADM has a constant light-cone timestep. Reduce only the
    # short-box CFL so its timestep matches the larger-domain control exactly.
    dx=16/n;reference_half=16 if localized else 24
    cfl=.25 if reference else .25*(1+abs(omega)*(8-dx/2))/(1+abs(omega)*(reference_half-dx/2))
    args=[f'meshblock/nx3={block_z}',f'time/cfl_number={cfl}','mesh_refinement/refinement=none','coord/excise=false','coord/require_resolved_horizon=false',
          'problem/single_black_hole=true','problem/reservoir_test_flat_metric=true',
          'problem/reservoir_pressure_perturbation=0',f'problem/corotating={str(omega!=0).lower()}',
          f'problem/frame_omega={omega}',f'problem/reservoir_test_pulse_mode={family}',
          'problem/reservoir_test_pulse=0.01',f'problem/reservoir_test_pulse_width={width}',
          f'problem/reservoir_test_pulse_transverse={1. if localized else 3.}',
          f'problem/reservoir_boundary={"fixed" if choice in ("fixed","reference") else "characteristic"}',
          f'problem/reservoir_trace_order={1 if choice=="firstorder" else 3}',
          f'time/tlim={target}','time/nlim=50000',f'output1/dt={.05 if localized else 20}',f'output2/dt={target}',
          f'output3/dt={.05 if localized else 20}','output4/dt=0','output1/ghost_zones=false','output2/ghost_zones=false']
    for d in range(1,4):
        h=24 if reference and not localized and d==3 else half
        cells=(2*n if localized else 3*n) if reference else n
        args += [f'mesh/nx{d}={cells}',f'mesh/x{d}min={-h}',f'mesh/x{d}max={h}',
                 f'mesh/ix{d}_bc=user',f'mesh/ox{d}_bc=user',
                 f'problem/reservoir_test_pulse_{"xyz"[d-1]}={center[d-1]}']
    return args

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('family');p.add_argument('frame');p.add_argument('choice');p.add_argument('n',type=int);p.add_argument('--block-z',type=int,default=8);a=p.parse_args()
    print('\n'.join(flags(a.family,a.frame,a.choice,a.n,a.block_z)))
