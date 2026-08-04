#!/bin/bash
#SBATCH --job-name=cuml-kmeans-run
#SBATCH --time=03:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=144
#SBATCH --mem=0
#SBATCH --exclusive

set -euo pipefail

if [[ -z "${SCRIPT_DIR:-}" ]]; then
  echo "Error: SCRIPT_DIR is required." >&2
  exit 1
fi
if [[ -z "${WORKER_IMAGE:-}" ]]; then
  echo "Error: WORKER_IMAGE is required." >&2
  exit 1
fi
if [[ -z "${N_SAMPLES:-}" ]]; then
  echo "Error: N_SAMPLES is required." >&2
  exit 1
fi

export VT_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." >/dev/null 2>&1 && pwd -P)"
source "${SCRIPT_DIR}/defaults.env"
export IMAGE_DIR
export DATASET_ROOT
export RESULTS_BASE
export DASK_SCHEDULER_PORT
export DASK_DASHBOARD_PORT
export DASK_PROTOCOL

export LOGS="${SCRIPT_DIR}/logs"
export RESULT_DIR="${SCRIPT_DIR}/result_dir"
export NUM_NODES="${SLURM_JOB_NUM_NODES}"
export COORD
COORD=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
export COORD_IP
coord_addr_line="$(scontrol getaddrs "${COORD}" 2>/dev/null | awk 'NR==1{print $2}')"
if [[ -n "${coord_addr_line:-}" ]]; then
  COORD_IP="${coord_addr_line%%:*}"
else
  COORD_IP="$(getent hosts "${COORD}" | awk 'NR==1{print $1}')"
fi
export TOTAL_WORKERS=$(( NUM_NODES * NUM_GPUS_PER_NODE ))
export DASK_SCHEDULER_ADDRESS="${DASK_PROTOCOL}://${COORD_IP:-$COORD}:${DASK_SCHEDULER_PORT}"

mkdir -p "${LOGS}" "${RESULT_DIR}"

echo "========================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Nodes: ${NUM_NODES}"
echo "Coordinator node: ${COORD}"
echo "Coordinator IP: ${COORD_IP:-unknown}"
echo "Worker image: ${WORKER_IMAGE}"
echo "Total workers: ${TOTAL_WORKERS}"
echo "Dataset rows: ${N_SAMPLES}"
echo "Dask scheduler: ${DASK_SCHEDULER_ADDRESS}"
echo "========================================"

bash "${SCRIPT_DIR}/run-cuml-kmeans.sh"
