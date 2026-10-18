#!/bin/bash
# Final dependency fix, applied only after the preceding isolated build is idle.
set -euo pipefail
cd "$1"
for attempt in $(seq 1 180); do
 [[ ! -s build-characteristic/final-ready.json ]] || break
 [[ $attempt != 180 ]] || exit 124
 sleep 10
done
[[ ! -e build-characteristic/fofc-fix-started.txt ]] || exit 2
printf 'boundary flux participates in predictor and fallback\n' > build-characteristic/fofc-fix-started.txt
mv build-characteristic/final-ready.json build-characteristic/prefofc-ready.json
cp build-characteristic/resources.json build-characteristic/prefofc-resources.json
cp dyn_grmhd_fluxes-fofc.cpp source-characteristic/src/dyn_grmhd/dyn_grmhd_fluxes.cpp
cp characteristic-reservoir-fofc.hpp source-characteristic/src/dyn_grmhd/characteristic_reservoir.hpp
sha256sum source-characteristic/src/{coordinates/coordinates.hpp,dyn_grmhd/dyn_grmhd.cpp,dyn_grmhd/dyn_grmhd_fluxes.cpp,dyn_grmhd/characteristic_reservoir.hpp,pgen/dynbbh.cpp} > build-characteristic/source.sha256
bash source-characteristic/scripts/tests/aurora_sane_build.sh "$PWD/source-characteristic" "$PWD/build-characteristic"
sha256sum -c build-characteristic/source.sha256
/usr/bin/python3.11 - <<'PY'
import json,hashlib
from pathlib import Path
p=Path('build-characteristic');r=json.loads((p/'resources.json').read_text());assert r['returncode']==0 and r['stopped'] is None
(p/'final-ready.json').write_text(json.dumps(dict(status='built',periodic_ghost_fix=True,linearization='physical_primitive_mean',fofc_boundary_predictor=True,executable_sha256=hashlib.sha256((p/'src/athena').read_bytes()).hexdigest()),indent=2)+'\n')
PY
