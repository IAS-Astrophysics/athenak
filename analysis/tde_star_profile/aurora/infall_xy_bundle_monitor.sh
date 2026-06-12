#!/bin/bash
# Login-node monitor for the paired 16+16 node infall xy runs.
# Submits one 32-node debug-scaling bundle at a time and resubmits only if both
# cases ended cleanly with finite, bounded constraint histories and restarts.

set -u

REPO_DIR="${REPO_DIR:-/home/hzhu/athenak_tde}"
PBS_SCRIPT="${PBS_SCRIPT:-${REPO_DIR}/analysis/tde_star_profile/aurora/submit_infall_xy_bundle_debug_scaling.pbs}"
RUN_ROOT="${RUN_ROOT:-/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/runs}"
CASE_A="${CASE_A:-z4c_tov_ks_n3_schwarzschild_bgadapt_fullgauge_rk3_infall20_xy}"
CASE_B="${CASE_B:-z4c_tov_ks_n3_schwarzschild_bgadapt_fullgauge_rk3_infall20_y10_xy}"
JOB_NAME="${JOB_NAME:-infallxy32}"
POLL_S="${POLL_S:-600}"
MAX_JOBS="${MAX_JOBS:-20}"
T_STOP="${T_STOP:-74.5}"
MAX_CONSTRAINT="${MAX_CONSTRAINT:-100.0}"

current_job=""
njobs=0

job_state() {
  qstat -x "$1" 2>/dev/null | tail -n 1 | awk '{print $5}' | grep -E "^[A-Z]$" || true
}

latest_hst() {
  local case_name="$1"
  ls "${RUN_ROOT}/${case_name}"/*.z4c.user.hst 2>/dev/null | head -n 1
}

latest_rst() {
  local case_name="$1"
  ls -v "${RUN_ROOT}/${case_name}/rst"/*.rst 2>/dev/null | tail -n 1 || true
}

case_status() {
  local case_name="$1"
  local hst rst last
  hst=$(latest_hst "${case_name}")
  rst=$(latest_rst "${case_name}")
  if [[ -z "${hst}" || ! -f "${hst}" ]]; then
    echo "MISSING_HST"
    return 1
  fi
  if [[ -z "${rst}" || ! -f "${rst}" ]]; then
    echo "MISSING_RST"
    return 1
  fi
  if grep -qiE "nan|inf" "${hst}"; then
    echo "NONFINITE"
    return 1
  fi
  last=$(tail -n 1 "${hst}")
  awk -v maxc="${MAX_CONSTRAINT}" -v tstop="${T_STOP}" '
    BEGIN { ok = 1 }
    {
      t = $1 + 0;
      vals[1] = $3 + 0; vals[2] = $4 + 0; vals[3] = $5 + 0;
      vals[4] = $10 + 0; vals[5] = $12 + 0; vals[6] = $13 + 0;
      for (i = 1; i <= 6; ++i) {
        v = vals[i]; if (v < 0) v = -v;
        if (v > maxc) ok = 0;
      }
      if (ok) {
        printf("CLEAN t=%g rst=%s", t, "'"${rst}"'");
        exit 0;
      }
      printf("LARGE_CONSTRAINT t=%g line=%s", t, $0);
      exit 1;
    }' <<< "${last}"
}

submit_next() {
  local existing out
  existing=$(qstat -u hzhu 2>/dev/null | awk -v jn="${JOB_NAME}" \
             '$4 ~ jn && ($10 == "Q" || $10 == "R" || $10 == "H") {print $1}' | head -n 1)
  if [[ -n "${existing}" ]]; then
    current_job="${existing%%.*}"
    echo "BUNDLE_MONITOR adopt job=${existing}"
    return 0
  fi
  out=$(qsub -N "${JOB_NAME}" -q debug-scaling -A MHDTidal \
        -l select=32 -l walltime=01:00:00 "${PBS_SCRIPT}" 2>&1)
  if [[ "${out}" == *aurora* ]]; then
    current_job="${out%%.*}"
    njobs=$((njobs + 1))
    echo "BUNDLE_MONITOR submit job=${out} n=${njobs}"
    return 0
  fi
  echo "BUNDLE_MONITOR submit_failed ${out}"
  current_job=""
  return 1
}

print_status() {
  local st="" sa sb
  [[ -n "${current_job}" ]] && st=$(job_state "${current_job}")
  sa=$(case_status "${CASE_A}" 2>/dev/null || true)
  sb=$(case_status "${CASE_B}" 2>/dev/null || true)
  echo "BUNDLE_MONITOR status $(date +%H:%M) job=${current_job:-none}(${st:--}) n=${njobs} A=${sa:-?} B=${sb:-?}"
}

echo "BUNDLE_MONITOR begin case_a=${CASE_A} case_b=${CASE_B}"
while true; do
  state=""
  [[ -n "${current_job}" ]] && state=$(job_state "${current_job}")
  if [[ "${state}" != "Q" && "${state}" != "R" && "${state}" != "H" ]]; then
    if (( njobs > 0 )); then
      status_a=$(case_status "${CASE_A}" || true)
      status_b=$(case_status "${CASE_B}" || true)
      echo "BUNDLE_MONITOR check A=${status_a} B=${status_b}"
      if [[ "${status_a}" != CLEAN* || "${status_b}" != CLEAN* ]]; then
        echo "BUNDLE_MONITOR stop unsafe_state"
        print_status
        exit 1
      fi
      ta=$(echo "${status_a}" | sed -n 's/.*t=\([^ ]*\).*/\1/p')
      tb=$(echo "${status_b}" | sed -n 's/.*t=\([^ ]*\).*/\1/p')
      if awk -v a="${ta:-0}" -v b="${tb:-0}" -v ts="${T_STOP}" 'BEGIN{exit !(a>=ts && b>=ts)}'; then
        echo "BUNDLE_MONITOR stop tlim A=${ta} B=${tb}"
        print_status
        exit 0
      fi
    fi
    if (( njobs >= MAX_JOBS )); then
      echo "BUNDLE_MONITOR stop max_jobs n=${njobs}"
      exit 1
    fi
    submit_next
  fi
  print_status
  sleep "${POLL_S}"
done
