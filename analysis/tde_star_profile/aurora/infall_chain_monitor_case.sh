#!/bin/bash
# Login-node driver for one star-infall restart chain.
# Parameters are supplied through environment variables so multiple chains can
# run concurrently with distinct case/job names.

set -u

REPO_DIR="${REPO_DIR:-/home/hzhu/athenak_tde}"
PBS_SCRIPT="${PBS_SCRIPT:-${REPO_DIR}/analysis/tde_star_profile/aurora/submit_aurora_chain.pbs}"
CASE_NAME="${CASE_NAME:?CASE_NAME must be set}"
INPUT_DECK="${INPUT_DECK:?INPUT_DECK must be set}"
RUN_DIR="${RUN_DIR:-/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/runs/${CASE_NAME}}"
JOB_NAME="${JOB_NAME:-${CASE_NAME:0:15}}"
POLL_S="${POLL_S:-300}"
R_STOP="${R_STOP:-1.8}"
RHO_STOP="${RHO_STOP:-1.0e-8}"
T_STOP="${T_STOP:-74.5}"
MAX_JOBS="${MAX_JOBS:-20}"
ATHENA_WALLTIME="${ATHENA_WALLTIME:-00:44:00}"
PBS_WALLTIME="${PBS_WALLTIME:-06:00:00}"
ITER_NEED_S="${ITER_NEED_S:-3000}"
KEEP_RST="${KEEP_RST:-6}"
RANKS_PER_NODE="${RANKS_PER_NODE:-12}"
QUEUE="${QUEUE:-capacity}"
PROJECT="${PROJECT:-MHDTidal}"
SELECT_NODES="${SELECT_NODES:-2}"

njobs=0
current_job=""

job_state() {
  qstat -x "$1" 2>/dev/null | tail -n 1 | awk '{print $5}' | grep -E "^[A-Z]$" || true
}

submit_next() {
  local existing
  existing=$(qstat -u hzhu 2>/dev/null | awk -v jn="${JOB_NAME}" \
             '$4 ~ jn && ($10 == "Q" || $10 == "R" || $10 == "H") {print $1}' | head -n 1)
  if [[ -n "${existing}" ]]; then
    current_job="${existing%%.*}"
    echo "CHAIN_INFO adopting existing job ${existing}"
    return 0
  fi

  local extra="output3/dt=2.0:z4c/damp_kappa1=0.5:z4c/damp_kappa2=0.0"
  extra+=":z4c/rhs_term_debug=true:z4c/rhs_term_debug_stride=400"
  extra+=":problem/excision_freeze_radius=1.0:problem/excision_ramp_radius=1.4"
  extra+=":mesh_refinement/max_nmb_per_rank=320"
  if [[ -n "${ATHENA_EXTRA_ARGS:-}" ]]; then
    extra+=":${ATHENA_EXTRA_ARGS}"
  fi

  local qsub_v="CASE_NAME=${CASE_NAME},INPUT_DECK=${INPUT_DECK}"
  qsub_v+=",ATHENA_EXTRA_ARGS=${extra},ATHENA_WALLTIME=${ATHENA_WALLTIME}"
  qsub_v+=",ITER_NEED_S=${ITER_NEED_S},KEEP_RST=${KEEP_RST},T_STOP=${T_STOP}"
  qsub_v+=",RANKS_PER_NODE=${RANKS_PER_NODE}"

  local out
  out=$(qsub -N "${JOB_NAME}" -v "${qsub_v}" \
        -q "${QUEUE}" -A "${PROJECT}" -l select="${SELECT_NODES}" -l walltime="${PBS_WALLTIME}" \
        "${PBS_SCRIPT}" 2>&1)
  if [[ "${out}" == *aurora* ]]; then
    current_job="${out%%.*}"
    njobs=$((njobs + 1))
    echo "CHAIN_SUBMIT job=${out} n=${njobs}"
    return 0
  fi
  echo "CHAIN_INFO capacity submission failed (${out}); will retry next poll"
  current_job=""
  return 1
}

latest_star_track() {
  local f line
  for f in $(ls -t "${REPO_DIR}/${JOB_NAME}".o* 2>/dev/null | head -n 3); do
    line=$(grep -h "STAR_TRACK" "${f}" 2>/dev/null | tail -n 1)
    if [[ -n "${line}" ]]; then
      echo "${line}"
      return
    fi
  done
}

check_stop() {
  local zhst="${RUN_DIR}/${CASE_NAME}_aurora.z4c.user.hst"
  [[ -f "${zhst}" ]] || zhst=$(ls "${RUN_DIR}"/*.z4c.user.hst 2>/dev/null | head -n 1)
  if [[ -n "${zhst:-}" && -f "${zhst}" ]]; then
    local lastz
    lastz=$(tail -n 1 "${zhst}")
    if echo "${lastz}" | grep -qiE "nan|inf"; then
      echo "CHAIN_STOP NONFINITE ${lastz}"
      return 0
    fi
    local t
    t=$(echo "${lastz}" | awk '{print $1+0}')
    if awk -v t="${t}" -v ts="${T_STOP}" 'BEGIN{exit !(t>=ts)}'; then
      echo "CHAIN_STOP TLIM t=${t}"
      return 0
    fi
  fi

  local track
  track=$(latest_star_track)
  if [[ -n "${track}" ]]; then
    local rbh rho
    rbh=$(echo "${track}" | sed -n 's/.*r_bh=\([^ ]*\).*/\1/p')
    rho=$(echo "${track}" | sed -n 's/.*rho_max=\([^ ]*\).*/\1/p')
    if [[ -n "${rbh}" ]] && awk -v r="${rbh}" -v rs="${R_STOP}" 'BEGIN{exit !(r<rs && r>0)}'; then
      echo "CHAIN_STOP MERGED ${track}"
      return 0
    fi
    if [[ -n "${rho}" ]] && awk -v d="${rho}" -v ds="${RHO_STOP}" 'BEGIN{exit !(d<ds && d>0)}'; then
      echo "CHAIN_STOP ACCRETED ${track}"
      return 0
    fi
  fi
  return 1
}

print_status() {
  local zhst lastz="" track
  zhst=$(ls "${RUN_DIR}"/*.z4c.user.hst 2>/dev/null | head -n 1)
  [[ -n "${zhst}" && -f "${zhst}" ]] && lastz=$(tail -n 1 "${zhst}")
  track=$(latest_star_track)
  local t="" C="" H="" M="" Th="" Ci="" Hi=""
  if [[ -n "${lastz}" && "${lastz}" != \#* ]]; then
    t=$(echo "${lastz}" | awk '{print $1+0}')
    C=$(echo "${lastz}" | awk '{print $3+0}')
    H=$(echo "${lastz}" | awk '{print $4+0}')
    M=$(echo "${lastz}" | awk '{print $5+0}')
    Th=$(echo "${lastz}" | awk '{print $10+0}')
    Ci=$(echo "${lastz}" | awk '{print $12+0}')
    Hi=$(echo "${lastz}" | awk '{print $13+0}')
  fi
  local rbh="" rho=""
  if [[ -n "${track}" ]]; then
    rbh=$(echo "${track}" | sed -n 's/.*r_bh=\([^ ]*\).*/\1/p')
    rho=$(echo "${track}" | sed -n 's/.*rho_max=\([^ ]*\).*/\1/p')
  fi
  local st=""
  [[ -n "${current_job}" ]] && st=$(job_state "${current_job}")
  echo "CHAIN_STATUS $(date +%H:%M) job=${current_job:-none}(${st:--}) njobs=${njobs} t=${t:-?} C=${C:-?} H=${H:-?} M=${M:-?} Theta=${Th:-?} Cint=${Ci:-?} Hint=${Hi:-?} r_bh=${rbh:-?} rho_max=${rho:-?}"
}

echo "CHAIN_BEGIN case=${CASE_NAME} run_dir=${RUN_DIR} job_name=${JOB_NAME}"
while true; do
  state=""
  [[ -n "${current_job}" ]] && state=$(job_state "${current_job}")
  if [[ "${state}" != "Q" && "${state}" != "R" && "${state}" != "H" ]]; then
    if check_stop; then
      print_status
      echo "CHAIN_END"
      exit 0
    fi
    if (( njobs >= MAX_JOBS )); then
      echo "CHAIN_STOP MAXJOBS njobs=${njobs}"
      exit 1
    fi
    submit_next
  fi
  print_status
  sleep "${POLL_S}"
done
