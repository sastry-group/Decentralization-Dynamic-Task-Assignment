#!/usr/bin/env bash
# =============================================================
# run_sweep.sh — Parameter sweep for benchmark_routing.py
#
# Sweeps over:
#   1. Algorithm (--baseline)
#   2. Depot order (--depot-order)
#   3. Comms graph structure (--comms_mode)
#
# Results are written to results/<timestamp>/ with one subfolder
# per configuration.  Logs and a summary CSV are produced.
# =============================================================

set -euo pipefail

# ── Conda / Python ────────────────────────────────────────────
PYTHON="/home/gaby/anaconda3/envs/dynMATA/bin/python"
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"   # adjust if script lives in scripts/
SCRIPT="${SCRIPT_DIR}/scripts/benchmark_routing.py"

# ── Fixed parameters ──────────────────────────────────────────
TRIALS=100
TIMESTEPS=200
N_DRONES=15
N_DEPOTS=5
NEW_REQ_PROB=0.5
TIME_WINDOW=45
INIT_METHOD="empty"

# ── Sweep axes (edit these lists) ─────────────────────────────
# ALGOS=("edd" "hungarian" "ibr")
ALGOS=("ibr")
# DEPOT_ORDERS=("asc" "desc" "random")
DEPOT_ORDERS=("random")
COMMS_MODES=("full" "rm_12" "rm_12_31" "rm_12_31_43"  "none")

# ── Output directory ──────────────────────────────────────────
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_DIR="${SCRIPT_DIR}/results/sweep_${TIMESTAMP}"
mkdir -p "${RESULTS_DIR}"

# CSV header
SUMMARY="${RESULTS_DIR}/summary.csv"
echo "algo,depot_order,comms_mode,status,log_file" > "${SUMMARY}"

# ── Total run count ───────────────────────────────────────────
TOTAL=$(( ${#ALGOS[@]} * ${#DEPOT_ORDERS[@]} * ${#COMMS_MODES[@]} ))
RUN=0

echo "=========================================="
echo " Parameter sweep: ${TOTAL} configurations"
echo " Results → ${RESULTS_DIR}"
echo "=========================================="

for ALGO in "${ALGOS[@]}"; do
  for DEPOT in "${DEPOT_ORDERS[@]}"; do
    for COMMS in "${COMMS_MODES[@]}"; do

      RUN=$((RUN + 1))
      TAG="${ALGO}__depot-${DEPOT}__comms-${COMMS}"
      RUN_DIR="${RESULTS_DIR}/${TAG}"
      mkdir -p "${RUN_DIR}"
      LOG="${RUN_DIR}/run.log"

      echo ""
      echo "[${RUN}/${TOTAL}] algo=${ALGO}  depot=${DEPOT}  comms=${COMMS}"
      echo "  log → ${LOG}"

      # ── Launch ──────────────────────────────────────────────
      set +e
      PYTHONPATH="${SCRIPT_DIR}" "${PYTHON}" "${SCRIPT}" \
        --trials       "${TRIALS}" \
        --timesteps    "${TIMESTEPS}" \
        --allow-overlap \
        --comms_mode   "${COMMS}" \
        --n_drones     "${N_DRONES}" \
        --n_depots     "${N_DEPOTS}" \
        --new_request_prob "${NEW_REQ_PROB}" \
        --time_window  "${TIME_WINDOW}" \
        --baseline     "${ALGO}" \
        --num-init-requests "100" \
        --dynamic_tasks \
        --init_method  "${INIT_METHOD}" \
        --depot-order  "${DEPOT}" \
        > "${LOG}" 2>&1
      EXIT_CODE=$?
      set -e

      if [ ${EXIT_CODE} -eq 0 ]; then
        STATUS="ok"
      else
        STATUS="FAIL(${EXIT_CODE})"
        echo "  ⚠  exited with code ${EXIT_CODE}"
      fi

      echo "${ALGO},${DEPOT},${COMMS},${STATUS},${LOG}" >> "${SUMMARY}"

    done
  done
done

echo ""
echo "=========================================="
echo " Sweep complete.  ${TOTAL} runs."
echo " Summary CSV → ${SUMMARY}"
echo "=========================================="