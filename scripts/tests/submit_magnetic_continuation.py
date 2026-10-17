#!/usr/bin/env python3
"""Submit one idempotent physical-pilot continuation after a validated clean stop.

Run on Aurora from the isolated pilot root. Does not monitor, cancel, or touch
other jobs. Restart layout is this branch's double-precision 9-Real RegionSize
and 19-int RegionIndcs, verified against src/outputs/restart.cpp.
"""
import argparse
import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import subprocess


def restart_state(path):
    with path.open('rb') as f:
        data = f.read(65536)
    marker = b'<par_end>\n'
    pos = data.find(marker)
    assert pos >= 0, 'Missing restart parameter header'
    base = pos+len(marker)
    blocks, root_level = struct.unpack_from('<ii',data,base)
    mesh = struct.unpack_from('<19i',data,base+8+9*8)
    mb = struct.unpack_from('<19i',data,base+8+9*8+19*4)
    t, dt, cycle = struct.unpack_from('<ddi',data,base+8+9*8+2*19*4)
    assert blocks > 0 and root_level >= 0 and cycle >= 0
    assert math.isfinite(t) and t >= 0 and math.isfinite(dt) and dt > 0
    assert all(mesh[i] > 0 and mb[i] > 0 and mesh[i]%mb[i] == 0 for i in (1,2,3))
    block = ''; params = {}
    for line in data[:pos].decode().splitlines():
        line = line.split('#',1)[0].strip()
        if line.startswith('<') and line.endswith('>'): block = line[1:-1]
        elif '=' in line:
            key,value = line.split('=',1);params[(block,key.strip())] = value.strip()
    assert params['amr_criterion1','bondi_mhd_schedule'] == 'true'
    assert float(params['amr_criterion1','residence_scale']) == 1.0
    assert params['problem','initial_condition'] == 'bondi_reservoir'
    assert float(params['problem','beta_inf']) > 0
    assert params['adm','dynamic'] == 'false'
    return {'time':t,'dt':dt,'cycle':cycle,'blocks':blocks}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path.cwd())
    p.add_argument('--predecessor',type=int,required=True)
    p.add_argument('--target',type=float,default=20000.)
    a = p.parse_args();root = a.root.resolve()
    assert 0 < a.target <= 20000, 'This pilot does not authorize an unbounded target'
    receipt = root/f'magnetic-continuation-{a.predecessor}.receipt'
    if receipt.exists():
        print(receipt.read_text());return  # Never duplicate an accepted/unknown attempt.
    def run(cmd):
        return subprocess.run(cmd,cwd=root,capture_output=True,text=True,check=True,timeout=60)
    q = run(['qstat','-xf',str(a.predecessor)]).stdout
    assert re.search(r'job_state = F\s',q) and re.search(r'Exit_status = 0\s',q), 'Predecessor is not a completed successful job'
    case = root/f'evolution-magnetic-{a.predecessor}'
    v = json.loads((case/'validation.json').read_text())
    assert len(v['ranks']) == 12 and len(v['hosts']) == 1
    assert v['max_scaled_divb'] < 1e-9 and v['conservative_sum_bytes'] < 30e9
    checkpoints = sorted((case/'rst').glob('*.rst'))
    assert checkpoints and checkpoints[-1].stat().st_size > 0
    checkpoint = checkpoints[-1];state = restart_state(checkpoint)
    last = v['states'][-1]
    assert state['cycle'] == last['cycle'] and state['blocks'] == last['blocks']
    assert abs(state['time']-last['time']) < max(1e-4,5e-6*state['time'])
    if state['time'] >= a.target-1e-6:
        print('Target reached; no continuation:',state);return
    assert v['termination'] == 'wall clock limit'
    occupied = run(['qselect','-u','hzhu','-q','debug-scaling']).stdout.strip()
    if occupied:raise SystemExit('Queue occupied; no submission or receipt: '+occupied)
    for f in ('build-magnetic/source.sha256','build-magnetic/executable.sha256'):
        print(run(['sha256sum','-c',f]).stdout,end='')
    sha = hashlib.sha256()
    with checkpoint.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):sha.update(chunk)
    command = ['qsub','-v',f'CZ_RESTART={checkpoint},CZ_TLIM={a.target:g}',
               'source-magnetic/scripts/tests/aurora_bondi_magnetic.pbs']
    record = {'predecessor':a.predecessor,'restart':str(checkpoint),
              'restart_state':state,'restart_sha256':sha.hexdigest(),
              'target':a.target,'attempt_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'command':command}
    # Receipt exists before submission, including if observing qsub is interrupted.
    with receipt.open('x') as f:f.write(json.dumps(record,indent=2)+'\n')
    result = subprocess.run(command,cwd=root,capture_output=True,text=True,timeout=60)
    record.update(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr)
    receipt.write_text(json.dumps(record,indent=2)+'\n');print(receipt.read_text())
    if result.returncode:raise SystemExit(result.returncode)


if __name__ == '__main__':main()
