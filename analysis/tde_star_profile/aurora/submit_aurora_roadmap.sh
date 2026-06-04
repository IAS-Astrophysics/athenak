#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/runs}"
POST_ROOT="${POST_ROOT:-${PROJECT_ROOT}/post}"
BUILD_DIR="${BUILD_DIR:-${PROJECT_ROOT}/build/aurora-intel-gpu-z4c_tov_ks}"
ATHENA_EXE="${ATHENA_EXE:-${BUILD_DIR}/src/athena}"
PBS_SCRIPT="${PBS_SCRIPT:-${SCRIPT_DIR}/submit_aurora_case.pbs}"
SUBMIT_DIR="${SUBMIT_DIR:-${PROJECT_ROOT}/submit}"
QUEUE="${QUEUE:-debug}"
SELECT_RESOURCE="${SELECT_RESOURCE:-1}"
RANKS_PER_NODE="${RANKS_PER_NODE:-12}"
CASE_FILTER="${CASE_FILTER:-}"
DRY_RUN=0

if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
fi

mkdir -p "${SUBMIT_DIR}" "${RUN_ROOT}" "${POST_ROOT}" "${PROJECT_ROOT}/logs"
cd "${SUBMIT_DIR}"

declare -a CASES=(
  "minkowski_static_uniform_dense|z4c_tov_ks_minkowski_static_sym_uniform_dense_aurora.athinput|00:20:00"
  "minkowski_static_smr_dense|z4c_tov_ks_minkowski_static_sym_smr_dense_aurora.athinput|00:20:00"
  "schwarzschild_infall_smr_dense|z4c_tov_ks_schwarzschild_infall_sym_smr_dense_aurora.athinput|00:20:00"
  "schwarzschild_zero_feedback_smr_dense|z4c_tov_ks_schwarzschild_infall_zero_feedback_smr_dense_aurora.athinput|00:20:00"
  "schwarzschild_fixed_mhd_tmunu_smr_dense|z4c_tov_ks_schwarzschild_fixed_mhd_tmunu_smr_dense_aurora.athinput|00:20:00"
  "schwarzschild_fixed_mhd_refresh_tmunu_smr_dense|z4c_tov_ks_schwarzschild_fixed_mhd_refresh_tmunu_smr_dense_aurora.athinput|00:20:00"
)

previous_job=""
for spec in "${CASES[@]}"; do
  IFS="|" read -r case_name input_name walltime <<< "${spec}"
  if [[ -n "${CASE_FILTER}" && "${case_name}" != "${CASE_FILTER}" ]]; then
    continue
  fi
  input_deck="${REPO_DIR}/inputs/tde/aurora/${input_name}"
  job_name="z4c_${case_name}"
  vars="CASE_NAME=${case_name},INPUT_DECK=${input_deck},REPO_DIR=${REPO_DIR},PROJECT_ROOT=${PROJECT_ROOT},RUN_ROOT=${RUN_ROOT},POST_ROOT=${POST_ROOT},BUILD_DIR=${BUILD_DIR},ATHENA_EXE=${ATHENA_EXE},RANKS_PER_NODE=${RANKS_PER_NODE},ATHENA_WALLTIME=00:19:00"

  cmd=(qsub
    -A MHDTidal
    -N "${job_name}"
    -q "${QUEUE}"
    -l "select=${SELECT_RESOURCE}"
    -l "walltime=${walltime}"
    -l "filesystems=home:flare"
    -l "place=scatter"
    -v "${vars}")

  if [[ -n "${previous_job}" ]]; then
    cmd+=(-W "depend=afterok:${previous_job}")
  fi
  cmd+=("${PBS_SCRIPT}")

  printf 'Submitting:'
  printf ' %q' "${cmd[@]}"
  printf '\n'

  if [[ "${DRY_RUN}" == "1" ]]; then
    previous_job="DRYRUN_${case_name}"
  else
    job_id="$("${cmd[@]}")"
    echo "Submitted ${case_name}: ${job_id}"
    previous_job="${job_id%%.*}"
  fi
done
