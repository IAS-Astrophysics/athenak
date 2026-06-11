#!/bin/bash
# Login-node driver for the star-infall restart chain.
# Submits chained 1-hour debug jobs until one of:
#   - star center (rho-max) crosses the horizon (r_bh < R_STOP)  -> CHAIN_STOP MERGED
#   - star fully accreted (rho_max < RHO_STOP)                   -> CHAIN_STOP ACCRETED
#   - simulation reaches T_STOP                                  -> CHAIN_STOP TLIM
#   - nonfinite constraints detected                             -> CHAIN_STOP NONFINITE
#   - MAX_JOBS chain jobs exhausted                              -> CHAIN_STOP MAXJOBS
# Prints CHAIN_STATUS lines (time, constraint norms, star position) every poll.

set -u

REPO_DIR=/home/hzhu/athenak_tde
PBS_SCRIPT=${REPO_DIR}/analysis/tde_star_profile/aurora/submit_aurora_chain.pbs
CASE_NAME=z4c_tov_ks_n3_schwarzschild_bgadapt_fullgauge_rk3_infall20
INPUT_DECK=${REPO_DIR}/inputs/tde/aurora/z4c_tov_ks_n3_schwarzschild_bgadapt_fullgauge_rk3_infall20_aurora.athinput
RUN_DIR=/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/runs/${CASE_NAME}
JOB_NAME=infall20
POLL_S=300
R_STOP=1.8
RHO_STOP=1.0e-8
T_STOP=74.5
MAX_JOBS=20

njobs=0
current_job=""

job_state() {
  # Echo state letter (Q/R/H/...) of $1, or empty if gone.
  qstat -x "$1" 2>/dev/null | tail -n 1 | awk '{print $5}' | grep -E "^[A-Z]$" || true
}

submit_next() {
  # Guard against duplicate submissions (e.g. transient qstat failures).
  local existing
  existing=$(qstat -u hzhu 2>/dev/null | awk -v jn="${JOB_NAME}" \
             '$4 ~ jn && ($10 == "Q" || $10 == "R" || $10 == "H") {print $1}' | head -n 1)
  if [[ -n "${existing}" ]]; then
    current_job="${existing%%.*}"
    echo "CHAIN_INFO adopting existing job ${existing}"
    return 0
  fi
  # rst dt=2.0 override: periodic 29 GB dumps every 0.5 M cost ~10 min of
  # wall each; the Finalize dump still captures the end state of every job.
  # 1-hour 2-node windows backfill quickly; each runs a single 44-min Athena
  # segment (validated margin for the final rst dump), and the per-job
  # mpiexec relaunch avoids Intel GPU memory fragmentation buildup.
  # damp_kappa1=0.1 enabled from t~32 (link 10): constraint norms grew
  # exponentially (e-fold ~2.2 M by t=29) with the debugging holdover
  # kappa1=0, consistent with undamped violations injected at AMR
  # boundaries as the star moves; gauge residuals stayed flat.
  # Colon-separated; the PBS script converts ':' to spaces (qsub -v cannot
  # reliably carry space-containing values).
  local extra="output3/dt=2.0:z4c/damp_kappa1=0.1:z4c/damp_kappa2=0.0"
  local qsub_v="CASE_NAME=${CASE_NAME},INPUT_DECK=${INPUT_DECK}"
  qsub_v+=",ATHENA_EXTRA_ARGS=${extra},ATHENA_WALLTIME=00:44:00,ITER_NEED_S=3000"
  local out
  out=$(qsub -N ${JOB_NAME} -v "${qsub_v}" \
        -q capacity -l walltime=01:00:00 "${PBS_SCRIPT}" 2>&1)
  if [[ "${out}" != *aurora* ]]; then
    echo "CHAIN_INFO capacity queue rejected (${out}); trying debug"
    out=$(qsub -N ${JOB_NAME} -v "${qsub_v}" \
          -q debug -l walltime=01:00:00 "${PBS_SCRIPT}" 2>&1)
  fi
  if [[ "${out}" != *aurora* ]]; then
    echo "CHAIN_INFO debug queue rejected (${out}); trying debug-scaling"
    out=$(qsub -N ${JOB_NAME} -v "${qsub_v}" \
          -q debug-scaling -l walltime=01:00:00 "${PBS_SCRIPT}" 2>&1)
  fi
  if [[ "${out}" == *aurora* ]]; then
    current_job="${out%%.*}"
    njobs=$((njobs + 1))
    echo "CHAIN_SUBMIT job=${out} n=${njobs}"
    return 0
  fi
  echo "CHAIN_INFO submission failed (${out}); will retry next poll"
  current_job=""
  return 1
}

latest_star_track() {
  # Last STAR_TRACK line of the newest stdout file that has one.
  local f line
  for f in $(ls -t ${REPO_DIR}/${JOB_NAME}.o* 2>/dev/null | head -n 3); do
    line=$(grep -h "STAR_TRACK" "${f}" 2>/dev/null | tail -n 1)
    if [[ -n "${line}" ]]; then
      echo "${line}"
      return
    fi
  done
}

check_stop() {
  # Returns 0 and prints CHAIN_STOP reason when chain should end.
  local zhst="${RUN_DIR}/${CASE_NAME}_aurora.z4c.user.hst"
  [[ -f "${zhst}" ]] || zhst=$(ls ${RUN_DIR}/*.z4c.user.hst 2>/dev/null | head -n 1)
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
  zhst=$(ls ${RUN_DIR}/*.z4c.user.hst 2>/dev/null | head -n 1)
  [[ -n "${zhst}" && -f "${zhst}" ]] && lastz=$(tail -n 1 "${zhst}")
  track=$(latest_star_track)
  local t="" C="" H="" M="" Th=""
  if [[ -n "${lastz}" && "${lastz}" != \#* ]]; then
    t=$(echo "${lastz}" | awk '{print $1+0}')
    C=$(echo "${lastz}" | awk '{print $3+0}')
    H=$(echo "${lastz}" | awk '{print $4+0}')
    M=$(echo "${lastz}" | awk '{print $5+0}')
    Th=$(echo "${lastz}" | awk '{print $7+0}')
  fi
  local rbh="" rho=""
  if [[ -n "${track}" ]]; then
    rbh=$(echo "${track}" | sed -n 's/.*r_bh=\([^ ]*\).*/\1/p')
    rho=$(echo "${track}" | sed -n 's/.*rho_max=\([^ ]*\).*/\1/p')
  fi
  local st=""
  [[ -n "${current_job}" ]] && st=$(job_state "${current_job}")
  echo "CHAIN_STATUS $(date +%H:%M) job=${current_job:-none}(${st:--}) njobs=${njobs} t=${t:-?} C=${C:-?} H=${H:-?} M=${M:-?} Theta=${Th:-?} r_bh=${rbh:-?} rho_max=${rho:-?}"
}

echo "CHAIN_BEGIN case=${CASE_NAME} run_dir=${RUN_DIR}"
while true; do
  state=""
  [[ -n "${current_job}" ]] && state=$(job_state "${current_job}")
  if [[ "${state}" != "Q" && "${state}" != "R" && "${state}" != "H" ]]; then
    # No active job: evaluate stop conditions, else (re)submit.
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
  sleep ${POLL_S}
done
