set -euo pipefail

# ── Conda / Python ────────────────────────────────────────────
PYTHON="/home/gaby/anaconda3/envs/dynMATA/bin/python"
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPT="${SCRIPT_DIR}/scripts/benchmark_routing.py"

# ── Fixed parameters ──────────────────────────────────────────
TRIALS=100
TIMESTEPS=200
N_DRONES=50
N_DEPOTS=5
INIT_METHOD="empty"
NUM_INIT_REQUESTS=75
PKG_DENSITY="nominal"

# ── Sweep axes (edit these lists) ─────────────────────────────
# ALGOS=("edd" "hungarian" "ibr" "scoba")
ALGOS=("ibr")
# ALGOS=("scoba")
DEPOT_ORDERS=("random")
COMMS_MODES=("full" "rm_12_31_43" "none" "rm_12_31" "rm_12")
# COMMS_MODES=("brm_12_45" "ring" "star" "brm_12" "none" "full")
# REQ_PROBS=(0.5 0.75 1.0)
REQ_PROBS=(0.5)
# TIME_WINDOWS=(15 30 45)
TIME_WINDOWS=(30)

# ── Output directory ──────────────────────────────────────────
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_DIR="${SCRIPT_DIR}/results/sweep_${TIMESTAMP}"
mkdir -p "${RESULTS_DIR}"

SUMMARY="${RESULTS_DIR}/summary.csv"
echo "algo,depot_order,comms_mode,req_prob,time_window,overlap,status,log_file" > "${SUMMARY}"

TOTAL=$(( ${#ALGOS[@]} * ${#DEPOT_ORDERS[@]} * ${#COMMS_MODES[@]} \
        * ${#REQ_PROBS[@]} * ${#TIME_WINDOWS[@]} ))
RUN=0

echo "=========================================="
echo " Parameter sweep: ${TOTAL} configurations"
echo " Results → ${RESULTS_DIR}"
echo "=========================================="

for ALGO in "${ALGOS[@]}"; do
  for DEPOT in "${DEPOT_ORDERS[@]}"; do
    for COMMS in "${COMMS_MODES[@]}"; do
      for PROB in "${REQ_PROBS[@]}"; do
        for WIN in "${TIME_WINDOWS[@]}"; do

          RUN=$((RUN + 1))
          TAG="${ALGO}__depot-${DEPOT}__comms-${COMMS}__prob-${PROB}__win-${WIN}"
          RUN_DIR="${RESULTS_DIR}/${TAG}"
          mkdir -p "${RUN_DIR}"
          LOG="${RUN_DIR}/run.log"

          # scoba does not take overlapping claims
          if [ "${ALGO}" = "scoba-full" ]; then
            OVERLAP_FLAG=()
            OVERLAP_TAG="no"
          else
            OVERLAP_FLAG=(--allow-overlap)
            OVERLAP_TAG="yes"
          fi

          echo ""
          echo "[${RUN}/${TOTAL}] algo=${ALGO} depot=${DEPOT} comms=${COMMS} prob=${PROB} win=${WIN} overlap=${OVERLAP_TAG}"
          echo "  log → ${LOG}"

          set +e
          PYTHONPATH="${SCRIPT_DIR}" "${PYTHON}" "${SCRIPT}" \
            --trials       "${TRIALS}" \
            --timesteps    "${TIMESTEPS}" \
            "${OVERLAP_FLAG[@]}" \
            --comms_mode   "${COMMS}" \
            --n_drones     "${N_DRONES}" \
            --n_depots     "${N_DEPOTS}" \
            --new_request_prob "${PROB}" \
            --time_window  "${WIN}" \
            --baseline     "${ALGO}" \
            --num-init-requests "${NUM_INIT_REQUESTS}" \
            --dynamic_tasks \
            --init_method  "${INIT_METHOD}" \
            --depot-order  "${DEPOT}" \
            --package-density "${PKG_DENSITY}" \
            > "${LOG}" 2>&1
          EXIT_CODE=$?
          set -e

          if [ ${EXIT_CODE} -eq 0 ]; then
            STATUS="ok"
          else
            STATUS="FAIL(${EXIT_CODE})"
            echo "  exited with code ${EXIT_CODE}"
          fi

          echo "${ALGO},${DEPOT},${COMMS},${PROB},${WIN},${OVERLAP_TAG},${STATUS},${LOG}" >> "${SUMMARY}"

        done
      done
    done
  done
done

echo ""
echo "=========================================="
echo " Sweep complete.  ${TOTAL} runs."
echo " Summary CSV → ${SUMMARY}"
echo "=========================================="