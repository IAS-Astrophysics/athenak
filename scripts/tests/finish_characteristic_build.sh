#!/bin/bash
# Complete a known staged fix after the initial isolated build has stopped.
# No queue submissions and no retries after build failure.
set -euo pipefail
cd "$1"
for attempt in $(seq 1 180); do
 if [[ -s build-characteristic/resources.json ]]; then
  status=$(/usr/bin/python3.11 -c 'import json;print(json.load(open("build-characteristic/resources.json"))["returncode"])')
  [[ "$status" == 0 ]] || { echo "Initial build failed: $status"; exit 1; }
  [[ -s build-characteristic/executable.sha256 ]] && break
 fi
 [[ $attempt != 180 ]] || exit 124
 sleep 10
done
[[ ! -e build-characteristic/final-fix-started.txt ]] || exit 2
printf 'known periodic-ghost conversion fix\n' > build-characteristic/final-fix-started.txt
cp build-characteristic/resources.json build-characteristic/initial-resources.json
cp build-characteristic/executable.sha256 build-characteristic/initial-executable.sha256
cp dyn_grmhd-periodic-fix.cpp source-characteristic/src/dyn_grmhd/dyn_grmhd.cpp
cp characteristic-reservoir-final.hpp source-characteristic/src/dyn_grmhd/characteristic_reservoir.hpp
cp prepare_characteristic_control.py source-characteristic/scripts/tests/prepare_characteristic_control.py
cp test_grmhd_characteristic.py source-characteristic/scripts/tests/test_grmhd_characteristic.py
sha256sum source-characteristic/src/{coordinates/coordinates.hpp,dyn_grmhd/dyn_grmhd.cpp,dyn_grmhd/dyn_grmhd_fluxes.cpp,dyn_grmhd/characteristic_reservoir.hpp,pgen/dynbbh.cpp} > build-characteristic/source.sha256
bash source-characteristic/scripts/tests/aurora_sane_build.sh "$PWD/source-characteristic" "$PWD/build-characteristic"
sha256sum -c build-characteristic/source.sha256
/usr/bin/python3.11 - <<'PY'
import json,hashlib
from pathlib import Path
p=Path('build-characteristic');r=json.loads((p/'resources.json').read_text());assert r['returncode']==0 and r['stopped'] is None
(p/'final-ready.json').write_text(json.dumps(dict(status='built',periodic_ghost_fix=True,executable_sha256=hashlib.sha256((p/'src/athena').read_bytes()).hexdigest()),indent=2)+'\n')
PY
