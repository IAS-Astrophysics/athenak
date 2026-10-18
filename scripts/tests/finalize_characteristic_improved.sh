#!/bin/bash
# Serialize the final call-site comment/source manifest after the initial build.
set -euo pipefail
cd "$1"
for attempt in $(seq 1 180);do
 [[ ! -e build-characteristic/improved-ready.json ]] || break
 [[ $attempt != 180 ]] || exit 124
 sleep 10
done
[[ ! -e characteristic-improved-final.started ]] || exit 2
printf 'final source manifest\n' > characteristic-improved-final.started
mv build-characteristic/improved-ready.json build-characteristic/improved-initial-ready.json
cp build-characteristic/resources.json build-characteristic/improved-initial-resources.json
cp characteristic-improved-final-fluxes.cpp source-characteristic/src/dyn_grmhd/dyn_grmhd_fluxes.cpp
cp characteristic-improved-final.hpp source-characteristic/src/dyn_grmhd/characteristic_reservoir.hpp
cp characteristic-improved-final-pgen.cpp source-characteristic/src/pgen/dynbbh.cpp
sha256sum source-characteristic/src/{coordinates/coordinates.hpp,dyn_grmhd/characteristic_reservoir.hpp,dyn_grmhd/dyn_grmhd.cpp,dyn_grmhd/dyn_grmhd_fluxes.cpp,dyn_grmhd/dyn_grmhd_fofc.cpp,pgen/dynbbh.cpp} > build-characteristic/improved-source.sha256
bash source-characteristic/scripts/tests/aurora_sane_build.sh "$PWD/source-characteristic" "$PWD/build-characteristic"
sha256sum -c build-characteristic/improved-source.sha256
python3.11 - <<'PY'
from pathlib import Path
import hashlib,json
p=Path('build-characteristic');r=json.loads((p/'resources.json').read_text());assert r['returncode']==0 and r['stopped'] is None
(p/'improved-ready.json').write_text(json.dumps(dict(status='built',trace_order=3,selective_fofc=True,native_fast_speed_bounds=True,final_source=True,executable_sha256=hashlib.sha256((p/'src/athena').read_bytes()).hexdigest()),indent=2)+'\n')
PY
