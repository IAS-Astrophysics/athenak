#!/bin/bash
# One serialized, guarded build in the existing isolated characteristic checkout.
set -euo pipefail
cd "$1"
trial="$PWD/boundary-characteristic-rotating-20261018-v3"
[[ ! -e "$trial/build.started" && ! -e "$trial/submission.json" ]] || exit 2
qstat -xf 8942061 > "$trial/predecessor-qstat.txt"
grep -q 'job_state = F' "$trial/predecessor-qstat.txt"
grep -q 'Exit_status = 0' "$trial/predecessor-qstat.txt"
printf 'ghost reconstruction/native CT\n' > "$trial/build.started"
archive="$PWD/characteristic-v2-8942061"
mkdir "$archive"
cp build-characteristic/src/athena "$archive/athena"
tar -czf "$archive/source.tar.gz" source-characteristic/src/coordinates/coordinates.hpp \
 source-characteristic/src/dyn_grmhd/{characteristic_reservoir.hpp,dyn_grmhd.cpp,dyn_grmhd_fluxes.cpp,dyn_grmhd_fofc.cpp} source-characteristic/src/pgen/dynbbh.cpp
sha256sum "$archive/athena" > "$archive/executable.sha256"
grep -q '^e90830db0dcd0e3442f5c07dd4b0e0a8d6e25ac4bd6059467e742244d8fb2c81 ' "$archive/executable.sha256"
cp -R "$trial/source-overlay/." source-characteristic/
sha256sum source-characteristic/src/{coordinates/coordinates.hpp,dyn_grmhd/characteristic_reservoir.hpp,dyn_grmhd/characteristic_ghost_ct.hpp,dyn_grmhd/dyn_grmhd.cpp,dyn_grmhd/dyn_grmhd_fluxes.cpp,dyn_grmhd/dyn_grmhd_fofc.cpp,pgen/dynbbh.cpp} > "$trial/source.sha256"
bash source-characteristic/scripts/tests/aurora_sane_build.sh "$PWD/source-characteristic" "$PWD/build-characteristic"
sha256sum -c "$trial/source.sha256" > "$trial/build-source-verification.txt"
python3.11 - <<'PY'
from pathlib import Path
import hashlib,json
trial=Path('boundary-characteristic-rotating-20261018-v3');build=Path('build-characteristic')
r=json.loads((build/'resources.json').read_text());assert r['returncode']==0 and r['stopped'] is None
(trial/'build-resources.json').write_text(json.dumps(r,indent=2)+'\n')
(trial/'build-ready.json').write_text(json.dumps(dict(status='built',ghost_ct=True,executable_sha256=hashlib.sha256((build/'src/athena').read_bytes()).hexdigest()),indent=2)+'\n')
PY
