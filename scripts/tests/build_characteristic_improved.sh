#!/bin/bash
# One isolated incremental build; no job submission and no production paths.
set -euo pipefail
cd "$1"
[[ ! -e characteristic-improved-build.started ]] || exit 2
printf '%s\n' 'limited trace; selective FOFC; native fast-speed bounds' > characteristic-improved-build.started
sha256sum source-characteristic/src/{coordinates/coordinates.hpp,dyn_grmhd/characteristic_reservoir.hpp,dyn_grmhd/dyn_grmhd_fluxes.cpp,dyn_grmhd/dyn_grmhd_fofc.cpp,pgen/dynbbh.cpp} > build-characteristic/improved-source.sha256
bash source-characteristic/scripts/tests/aurora_sane_build.sh "$PWD/source-characteristic" "$PWD/build-characteristic"
sha256sum -c build-characteristic/improved-source.sha256
python3.11 - <<'PY'
from pathlib import Path
import hashlib,json
p=Path('build-characteristic');r=json.loads((p/'resources.json').read_text());assert r['returncode']==0 and r['stopped'] is None
(p/'improved-ready.json').write_text(json.dumps(dict(status='built',trace_order=2,selective_fofc=True,native_fast_speed_bounds=True,executable_sha256=hashlib.sha256((p/'src/athena').read_bytes()).hexdigest()),indent=2)+'\n')
PY
