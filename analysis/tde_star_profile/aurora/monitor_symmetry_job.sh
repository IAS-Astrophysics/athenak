#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: monitor_symmetry_job.sh JOB_ID CASE_NAME [--once]

Poll a PBS job for the TOV symmetry discriminator suite without producing
large logs. By default it checks every 10 minutes until the job leaves qstat.

Environment:
  INTERVAL_SECONDS  Poll interval, default 600.
  MAX_CHECKS        Stop after this many checks, default 0 for no limit.
  PROJECT_ROOT      Default /lus/flare/projects/MHDTidal/hzhu/tde_n3_validation.
  RUN_ROOT          Default $PROJECT_ROOT/runs.
  POST_ROOT         Default $PROJECT_ROOT/post.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if (( $# < 2 )); then
  usage >&2
  exit 2
fi

JOB_ID="$1"
CASE_NAME="$2"
ONCE=0
if [[ "${3:-}" == "--once" ]]; then
  ONCE=1
fi

INTERVAL_SECONDS="${INTERVAL_SECONDS:-600}"
MAX_CHECKS="${MAX_CHECKS:-0}"
PROJECT_ROOT="${PROJECT_ROOT:-/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/runs}"
POST_ROOT="${POST_ROOT:-${PROJECT_ROOT}/post}"
RUN_DIR="${RUN_ROOT}/${CASE_NAME}"
POST_DIR="${POST_ROOT}/${CASE_NAME}"

timestamp() {
  date -Is
}

job_state() {
  local out state
  if ! out="$(qstat "${JOB_ID}" 2>&1)"; then
    echo "DONE_OR_UNKNOWN"
    return
  fi
  state="$(awk -v id="${JOB_ID%%.*}" '$1 ~ id {print $(NF-1); exit}' <<< "${out}")"
  echo "${state:-UNKNOWN}"
}

print_outputs() {
  echo "run_dir=${RUN_DIR}"
  echo "post_dir=${POST_DIR}"
  if [[ -f "${POST_DIR}/symmetry_metrics.csv" ]]; then
    echo "metrics_csv=${POST_DIR}/symmetry_metrics.csv"
    echo "metrics_json=${POST_DIR}/symmetry_metrics.json"
    echo "norm_plot=${POST_DIR}/symmetry_norms_peak_relative.png"
  else
    echo "metrics_csv=not yet present"
  fi
}

checks=0
last_state=""

while true; do
  state="$(job_state)"
  if [[ "${state}" != "${last_state}" ]]; then
    echo "[$(timestamp)] job=${JOB_ID} case=${CASE_NAME} state=${state}"
    last_state="${state}"
  else
    echo "[$(timestamp)] state=${state}"
  fi

  if [[ "${state}" == "DONE_OR_UNKNOWN" || "${ONCE}" == "1" ]]; then
    break
  fi

  checks=$((checks + 1))
  if (( MAX_CHECKS > 0 && checks >= MAX_CHECKS )); then
    echo "[$(timestamp)] reached MAX_CHECKS=${MAX_CHECKS}"
    break
  fi

  sleep "${INTERVAL_SECONDS}"
done

print_outputs
