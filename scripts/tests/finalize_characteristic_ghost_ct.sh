#!/bin/bash
# Serialize final support guards and surface-only divergence kernels after broad build.
set -euo pipefail
cd "$1"
trial="$PWD/boundary-characteristic-rotating-20261018-v3"
for attempt in $(seq 1 241);do
 [[ ! -e "$trial/build-ready.json" ]] || break
 [[ $attempt != 241 ]] || exit 124
 sleep 10
done
[[ ! -e "$trial/finalize.started" && ! -e "$trial/submission.json" ]] || exit 2
printf 'guarded 3D/root physical faces; surface ghost divergence\n' > "$trial/finalize.started"
mv "$trial/build-ready.json" "$trial/build-initial-ready.json"
cp "$trial/build-resources.json" "$trial/build-initial-resources.json"
cp "$trial/source.sha256" "$trial/source-initial.sha256"
cp "$trial/characteristic_ghost_ct-final.hpp" source-characteristic/src/dyn_grmhd/characteristic_ghost_ct.hpp
sha256sum source-characteristic/src/{coordinates/coordinates.hpp,dyn_grmhd/characteristic_reservoir.hpp,dyn_grmhd/characteristic_ghost_ct.hpp,dyn_grmhd/dyn_grmhd.cpp,dyn_grmhd/dyn_grmhd_fluxes.cpp,dyn_grmhd/dyn_grmhd_fofc.cpp,pgen/dynbbh.cpp} > "$trial/source.sha256"
bash source-characteristic/scripts/tests/aurora_sane_build.sh "$PWD/source-characteristic" "$PWD/build-characteristic"
sha256sum -c "$trial/source.sha256" > "$trial/build-source-verification.txt"
python3.11 - <<'PY'
from pathlib import Path
import hashlib,json
trial=Path('boundary-characteristic-rotating-20261018-v3');build=Path('build-characteristic')
r=json.loads((build/'resources.json').read_text());assert r['returncode']==0 and r['stopped'] is None
(trial/'build-resources.json').write_text(json.dumps(r,indent=2)+'\n')
(trial/'build-ready.json').write_text(json.dumps(dict(status='built',ghost_ct=True,final_source=True,unrefined_physical_faces=True,surface_divergence_kernels=True,executable_sha256=hashlib.sha256((build/'src/athena').read_bytes()).hexdigest()),indent=2)+'\n')
PY
