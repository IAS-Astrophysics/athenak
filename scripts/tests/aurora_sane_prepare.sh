#!/bin/bash
# Copy dependency/launcher provenance into a NEW isolated source directory.
set -euo pipefail
source_root=$1
isolated=$2
sane=${3:-/lus/flare/projects/MHDTidal/hzhu/SANE_continued/20261001/source_mpi_tag_v2}
[[ ! -e "$isolated" ]] || { echo 'Destination must be new' >&2; exit 2; }
[[ $(git -C "$sane/kokkos" rev-parse HEAD) == 6739bc623081648af9e752b616d9671527922cbf ]]
rsync -a --exclude=.git --exclude=/kokkos/ --exclude='/build*/' "$source_root/" "$isolated/"
rsync -a --exclude=.git "$sane/kokkos/" "$isolated/kokkos/"
git -C "$sane/kokkos" diff > "$isolated/sane-kokkos-compat.patch"
cp "$sane/experiments/telegrapher_aurora_20260929/sycl_compile.py" "$isolated/scripts/tests/sane_sycl_compile.py"
chmod +x "$isolated/scripts/tests/sane_sycl_compile.py"
