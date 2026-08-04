#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

cd "$(dirname "$0")"
source ./defaults.env

rm -rf result_dir logs 2>/dev/null || true
rm -f ./*.out ./*.err 2>/dev/null || true
mkdir -p result_dir logs

NODES_COUNT=""
NUM_GPUS_PER_NODE="4"
ITERATIONS="5"
WORKER_IMAGE=""
OUTPUT_PATH=""
SAMPLES_PER_GPU=""
N_SAMPLES=""
N_FEATURES="64"
N_CLUSTERS="100"
MAX_ITER="100"
SEED="42"
CHUNK_ROWS="0"
WORKER_ENV_FILE="$PWD/worker.env"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -n|--nodes) NODES_COUNT="$2"; shift 2 ;;
    -g|--num-gpus-per-node) NUM_GPUS_PER_NODE="$2"; shift 2 ;;
    -i|--iterations) ITERATIONS="$2"; shift 2 ;;
    -w|--worker-image) WORKER_IMAGE="$2"; shift 2 ;;
    -o|--output-path) OUTPUT_PATH="$2"; shift 2 ;;
    --samples-per-gpu) SAMPLES_PER_GPU="$2"; shift 2 ;;
    --n-samples) N_SAMPLES="$2"; shift 2 ;;
    --n-features) N_FEATURES="$2"; shift 2 ;;
    --n-clusters) N_CLUSTERS="$2"; shift 2 ;;
    --max-iter) MAX_ITER="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --chunk-rows) CHUNK_ROWS="$2"; shift 2 ;;
    --worker-env-file) WORKER_ENV_FILE="$2"; shift 2 ;;
    --) shift; break ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done

[[ -n "${NODES_COUNT}" ]] || { echo "Error: --nodes is required"; exit 1; }
[[ -n "${WORKER_IMAGE}" ]] || { echo "Error: --worker-image is required"; exit 1; }
[[ -n "${N_SAMPLES}" || -n "${SAMPLES_PER_GPU}" ]] || {
  echo "Error: provide either --n-samples or --samples-per-gpu"
  exit 1
}

if [[ -z "${N_SAMPLES}" ]]; then
  TOTAL_GPUS=$(( NODES_COUNT * NUM_GPUS_PER_NODE ))
  N_SAMPLES=$(( SAMPLES_PER_GPU * TOTAL_GPUS ))
fi

OUT_FMT="logs/cuml-kmeans-run_n${NODES_COUNT}_s${N_SAMPLES}_i${ITERATIONS}_%j.out"
ERR_FMT="logs/cuml-kmeans-run_n${NODES_COUNT}_s${N_SAMPLES}_i${ITERATIONS}_%j.err"
JOB_NAME="cuml-kmeans-run_n${NODES_COUNT}_s${N_SAMPLES}"

NODELIST="${NODELIST:-}"
NODELIST_ARG=()
if [[ -n "${NODELIST}" ]]; then
  NODELIST_ARG=(--nodelist="${NODELIST}")
fi

EXPORT_VARS="ALL"
EXPORT_VARS+=",SCRIPT_DIR=${PWD}"
EXPORT_VARS+=",NUM_GPUS_PER_NODE=${NUM_GPUS_PER_NODE}"
EXPORT_VARS+=",WORKER_IMAGE=${WORKER_IMAGE}"
EXPORT_VARS+=",ITERATIONS=${ITERATIONS}"
EXPORT_VARS+=",N_SAMPLES=${N_SAMPLES}"
EXPORT_VARS+=",N_FEATURES=${N_FEATURES}"
EXPORT_VARS+=",N_CLUSTERS=${N_CLUSTERS}"
EXPORT_VARS+=",MAX_ITER=${MAX_ITER}"
EXPORT_VARS+=",SEED=${SEED}"
EXPORT_VARS+=",CHUNK_ROWS=${CHUNK_ROWS}"
EXPORT_VARS+=",WORKER_ENV_FILE=${WORKER_ENV_FILE}"
if [[ -n "${SAMPLES_PER_GPU}" ]]; then
  EXPORT_VARS+=",SAMPLES_PER_GPU=${SAMPLES_PER_GPU}"
fi

JOB_ID=$(sbatch \
  --job-name="${JOB_NAME}" \
  --nodes="${NODES_COUNT}" \
  "${NODELIST_ARG[@]}" \
  --gres="gpu:${NUM_GPUS_PER_NODE}" \
  --export="${EXPORT_VARS}" \
  --output="${OUT_FMT}" \
  --error="${ERR_FMT}" \
  "${EXTRA_ARGS[@]}" \
  run-cuml-kmeans.slurm | awk '{print $NF}')

OUT_FILE="${OUT_FMT//%j/${JOB_ID}}"
ERR_FILE="${ERR_FMT//%j/${JOB_ID}}"

echo "Job submitted with ID: ${JOB_ID}"
echo "Monitor with:"
echo "  squeue -j ${JOB_ID}"
echo "  tail -f ${OUT_FILE}"
echo "  tail -f ${ERR_FILE}"
echo "  tail -f logs/scheduler.log"
echo "  tail -f logs/worker_*.log"
echo ""
echo "Waiting for job completion..."

while squeue -j "${JOB_ID}" 2>/dev/null | grep -q "${JOB_ID}"; do
  sleep 5
done

echo "Job completed."

if [[ -n "${OUTPUT_PATH}" ]]; then
  echo "Copying results to ${OUTPUT_PATH}..."
  mkdir -p "${OUTPUT_PATH}"
  cp -r result_dir/. "${OUTPUT_PATH}/"
fi
