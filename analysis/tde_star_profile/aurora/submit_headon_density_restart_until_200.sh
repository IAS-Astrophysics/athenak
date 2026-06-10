#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/runs}"
SUBMIT_DIR="${SUBMIT_DIR:-${PROJECT_ROOT}/submit}"
BUILD_DIR="${BUILD_DIR:-${PROJECT_ROOT}/build/aurora-intel-gpu-z4c_tov_ks}"
ATHENA_EXE="${ATHENA_EXE:-${BUILD_DIR}/src/athena}"
PBS_SCRIPT="${PBS_SCRIPT:-${SCRIPT_DIR}/restart_until_tlim.pbs}"

CASE_NAME="${CASE_NAME:-schwarzschild_headon_density_amr_64c}"
RUN_DIR="${RUN_DIR:-${RUN_ROOT}/${CASE_NAME}}"
TARGET_TLIM="${TARGET_TLIM:-200.0}"
RANKS_PER_NODE="${RANKS_PER_NODE:-12}"
ATHENA_WALLTIME="${ATHENA_WALLTIME:-00:50:00}"
QUEUE="${QUEUE:-debug-scaling}"
SELECT_RESOURCE="${SELECT_RESOURCE:-64}"
PBS_WALLTIME="${PBS_WALLTIME:-01:00:00}"
JOB_NAME="${JOB_NAME:-z4c_headon_r200}"
RESUBMIT="${RESUBMIT:-1}"

mkdir -p "${SUBMIT_DIR}"

vars="CASE_NAME=${CASE_NAME},RUN_DIR=${RUN_DIR},REPO_DIR=${REPO_DIR},PROJECT_ROOT=${PROJECT_ROOT},RUN_ROOT=${RUN_ROOT},SUBMIT_DIR=${SUBMIT_DIR},BUILD_DIR=${BUILD_DIR},ATHENA_EXE=${ATHENA_EXE},PBS_SCRIPT=${PBS_SCRIPT},TARGET_TLIM=${TARGET_TLIM},RANKS_PER_NODE=${RANKS_PER_NODE},ATHENA_WALLTIME=${ATHENA_WALLTIME},QUEUE=${QUEUE},SELECT_RESOURCE=${SELECT_RESOURCE},PBS_WALLTIME=${PBS_WALLTIME},JOB_NAME=${JOB_NAME},RESUBMIT=${RESUBMIT}"

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
