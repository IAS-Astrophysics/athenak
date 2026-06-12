#!/bin/bash
# Prepare, but do not submit by default, the 64-node debug-scaling restart
# chain for the 5e6:1 parabolic TDE setup.

set -u

REPO_DIR="${REPO_DIR:-/home/hzhu/athenak_tde}"
SCRIPT_DIR="${REPO_DIR}/analysis/tde_star_profile/aurora"
MONITOR="${MONITOR:-${SCRIPT_DIR}/infall_chain_monitor_case.sh}"
CASE_NAME="${CASE_NAME:-z4c_tov_ks_n3_schwarzschild_bgadapt_fullgauge_rk3_tde5e6_parabolic}"
INPUT_DECK="${INPUT_DECK:-${REPO_DIR}/inputs/tde/aurora/z4c_tov_ks_n3_schwarzschild_bgadapt_fullgauge_rk3_tde5e6_parabolic_aurora.athinput}"
JOB_NAME="${JOB_NAME:-tde5e6p64}"
LOG_PATH="${LOG_PATH:-${REPO_DIR}/tde5e6_parabolic_chain_monitor.log}"

export CASE_NAME
export INPUT_DECK
export JOB_NAME
export QUEUE="${QUEUE:-debug-scaling}"
export PROJECT="${PROJECT:-MHDTidal}"
export SELECT_NODES="${SELECT_NODES:-64}"
export PBS_WALLTIME="${PBS_WALLTIME:-01:00:00}"
export ATHENA_WALLTIME="${ATHENA_WALLTIME:-00:50:00}"
export ITER_NEED_S="${ITER_NEED_S:-3000}"
export T_STOP="${T_STOP:-284.7291594418413}"
export R_STOP="${R_STOP:-0.0}"
export RHO_STOP="${RHO_STOP:-0.0}"
export MAX_JOBS="${MAX_JOBS:-30}"
export KEEP_RST="${KEEP_RST:-6}"
export RANKS_PER_NODE="${RANKS_PER_NODE:-12}"
export ATHENA_EXTRA_ARGS="${ATHENA_EXTRA_ARGS:-output3/dt=10.0:z4c/damp_kappa1=0.1:z4c/damp_kappa2=0.0:z4c/rhs_term_debug=true:z4c/rhs_term_debug_stride=400:problem/excision_freeze_radius=1.0:problem/excision_ramp_radius=1.4:mesh_refinement/max_nmb_per_rank=320}"

cat <<EOF
Prepared 64-node debug-scaling chain:
  CASE_NAME=${CASE_NAME}
  INPUT_DECK=${INPUT_DECK}
  JOB_NAME=${JOB_NAME}
  QUEUE=${QUEUE}
  PROJECT=${PROJECT}
  SELECT_NODES=${SELECT_NODES}
  PBS_WALLTIME=${PBS_WALLTIME}
  ATHENA_WALLTIME=${ATHENA_WALLTIME}
  T_STOP=${T_STOP}
  LOG_PATH=${LOG_PATH}

Monitor command:
  ${MONITOR} > ${LOG_PATH} 2>&1 &
EOF

if [[ "${1:-}" == "--submit" ]]; then
  "${MONITOR}" > "${LOG_PATH}" 2>&1 &
  echo "Started monitor PID=$! log=${LOG_PATH}"
else
  echo "Dry run only. Re-run with --submit after inspection."
fi
