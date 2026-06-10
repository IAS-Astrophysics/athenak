#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/runs}"
POST_ROOT="${POST_ROOT:-${PROJECT_ROOT}/post}"
SUBMIT_DIR="${SUBMIT_DIR:-${PROJECT_ROOT}/submit}"
BUILD_DIR="${BUILD_DIR:-${PROJECT_ROOT}/build/aurora-intel-gpu-z4c_tov_ks}"
ATHENA_EXE="${ATHENA_EXE:-${BUILD_DIR}/src/athena}"
PBS_SCRIPT="${PBS_SCRIPT:-${SCRIPT_DIR}/submit_aurora_case.pbs}"

CASE_NAME="${CASE_NAME:-minkowski_static_selfgrav_amr_64c}"
INPUT_DECK="${INPUT_DECK:-${REPO_DIR}/inputs/tde/aurora/z4c_tov_ks_n3_minkowski_static_selfgrav_amr_64c_aurora.athinput}"
JOB_NAME="${JOB_NAME:-z4c_static_selfgrav64}"
QUEUE="${QUEUE:-debug-scaling}"
SELECT_RESOURCE="${SELECT_RESOURCE:-64}"
PBS_WALLTIME="${PBS_WALLTIME:-01:00:00}"
ATHENA_WALLTIME="${ATHENA_WALLTIME:-00:50:00}"
RANKS_PER_NODE="${RANKS_PER_NODE:-12}"

mkdir -p "${SUBMIT_DIR}" "${RUN_ROOT}" "${POST_ROOT}"

vars="CASE_NAME=${CASE_NAME},INPUT_DECK=${INPUT_DECK},REPO_DIR=${REPO_DIR},PROJECT_ROOT=${PROJECT_ROOT},RUN_ROOT=${RUN_ROOT},POST_ROOT=${POST_ROOT},BUILD_DIR=${BUILD_DIR},ATHENA_EXE=${ATHENA_EXE},RANKS_PER_NODE=${RANKS_PER_NODE},ATHENA_WALLTIME=${ATHENA_WALLTIME},POSTPROCESS=0"

cmd=(qsub
  -A MHDTidal
  -N "${JOB_NAME}"
  -q "${QUEUE}"
  -l "select=${SELECT_RESOURCE}"
  -l "walltime=${PBS_WALLTIME}"
  -l "filesystems=home:flare"
  -l "place=scatter"
  -v "${vars}"
  "${PBS_SCRIPT}")

cd "${SUBMIT_DIR}"

printf 'Prepared qsub command:'
printf ' %q' "${cmd[@]}"
printf '\n'

if [[ "${1:-}" == "--submit" ]]; then
  "${cmd[@]}"
else
  echo "Dry run only. Re-run with --submit to submit."
fi
