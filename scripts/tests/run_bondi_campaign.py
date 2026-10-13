#!/usr/bin/env python3
"""Run four bounded cases serially inside one Aurora debug allocation."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args()
    root=args.root.resolve()
    source=root/'source'
    exe=root/'build-sycl2025/src/athena'
    results={}
    for line in (root/'campaign/campaign.txt').read_text().splitlines():
        name, input_name, *flags=line.split()
        case=root/'results'/name
        case.mkdir(parents=True,exist_ok=True)
        env=os.environ.copy()
        env['KOKKOS_MEMORY_REPORT']=str(case/'allocations.json')
        command=['mpiexec','-n','1','-ppn','1','--cpu-bind','list:1',
                 '/usr/bin/python3.11',str(source/'scripts/tests/run_bounded.py'),
                 '--seconds','420','--memory-gb','20','--gpu-allocations',
                 '--report',str(case/'resources.json'),
                 '--',str(exe),'-i',str(root/'campaign'/input_name),'-d',str(case),
                 '-t','00:06:00','time/ndiag=100']+flags
        started=time.monotonic()
        with (case/'run.log').open('w') as log:
            proc=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,
                                timeout=450)
        log=(case/'run.log').read_text()
        endings=re.findall(r'^time=(\S+) cycle=(\d+)',log,re.M)
        results[name]={'returncode':proc.returncode,'seconds':time.monotonic()-started,
                       'command':command,'completed':endings[-1] if endings else None}
        (root/'results/campaign-results.json').write_text(json.dumps(results,indent=2)+'\n')
        print(name,results[name]['returncode'],results[name]['completed'],flush=True)
        if proc.returncode != 0:
            print(log[-4000:],flush=True)
            raise SystemExit(proc.returncode)


if __name__ == '__main__':
    main()
