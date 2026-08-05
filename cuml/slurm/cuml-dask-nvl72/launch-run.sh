#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

CALLER_DIR=$(pwd -P)
SCRIPT_DIR=$(cd -- "$(dirname -- "$0")" && pwd -P)
cd "${SCRIPT_DIR}"
# shellcheck source=defaults.env
# shellcheck disable=SC1091
source ./defaults.env

NODES_COUNT=""
NUM_GPUS_PER_NODE="4"
WORKER_IMAGE=""
OUTPUT_PATH=""
OVERWRITE_OUTPUT=0
SAMPLES_PER_GPU=""
N_SAMPLES=""
N_FEATURES=""
N_CLUSTERS="100"
MAX_ITER="20"
TOL="0.0001"
SEED="12345"
CLUSTER_STD="0.01"
WARMUP_RUNS="1"
REPEATS="3"
DEVICE_BUFFER_SAMPLES="4000000"
INPUT_FBIN=""
WORKER_ENV_FILE="$PWD/worker.env"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -n|--nodes) NODES_COUNT="$2"; shift 2 ;;
    -g|--num-gpus-per-node) NUM_GPUS_PER_NODE="$2"; shift 2 ;;
    -w|--worker-image) WORKER_IMAGE="$2"; shift 2 ;;
    -o|--output-path) OUTPUT_PATH="$2"; shift 2 ;;
    --overwrite-output) OVERWRITE_OUTPUT=1; shift ;;
    --samples-per-gpu) SAMPLES_PER_GPU="$2"; shift 2 ;;
    --n-samples) N_SAMPLES="$2"; shift 2 ;;
    --n-features) N_FEATURES="$2"; shift 2 ;;
    --n-clusters) N_CLUSTERS="$2"; shift 2 ;;
    --max-iter) MAX_ITER="$2"; shift 2 ;;
    --tol) TOL="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --cluster-std) CLUSTER_STD="$2"; shift 2 ;;
    --warmup-runs) WARMUP_RUNS="$2"; shift 2 ;;
    -i|--iterations|--repeats) REPEATS="$2"; shift 2 ;;
    --device-buffer-samples) DEVICE_BUFFER_SAMPLES="$2"; shift 2 ;;
    --input-fbin) INPUT_FBIN="$2"; shift 2 ;;
    --worker-env-file) WORKER_ENV_FILE="$2"; shift 2 ;;
    --) shift; EXTRA_ARGS+=("$@"); break ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done

[[ -n "${NODES_COUNT}" ]] || { echo "Error: --nodes is required"; exit 1; }
[[ -n "${WORKER_IMAGE}" ]] || { echo "Error: --worker-image is required"; exit 1; }
[[ -n "${N_SAMPLES}" || -n "${SAMPLES_PER_GPU}" ]] || {
  echo "Error: provide either --n-samples or --samples-per-gpu"
  exit 1
}
(( DEVICE_BUFFER_SAMPLES > 0 )) || {
  echo "Error: --device-buffer-samples must be positive for out-of-core KMeans"
  exit 1
}
(( REPEATS > 0 )) || { echo "Error: --repeats must be positive"; exit 1; }
(( WARMUP_RUNS >= 0 )) || { echo "Error: --warmup-runs must be nonnegative"; exit 1; }
(( SEED >= 0 )) || { echo "Error: --seed must be nonnegative"; exit 1; }

if [[ -z "${N_SAMPLES}" ]]; then
  TOTAL_GPUS=$(( NODES_COUNT * NUM_GPUS_PER_NODE ))
  N_SAMPLES=$(( SAMPLES_PER_GPU * TOTAL_GPUS ))
elif [[ -n "${SAMPLES_PER_GPU}" ]]; then
  EXPECTED_SAMPLES=$(( SAMPLES_PER_GPU * NODES_COUNT * NUM_GPUS_PER_NODE ))
  [[ "${N_SAMPLES}" -eq "${EXPECTED_SAMPLES}" ]] || {
    echo "Error: --n-samples must equal samples-per-gpu * nodes * GPUs per node"
    exit 1
  }
fi

INPUT_FBIN_CONTAINER=""
if [[ -n "${INPUT_FBIN}" ]]; then
  if [[ "${INPUT_FBIN}" != /* ]]; then
    INPUT_FBIN="${CALLER_DIR}/${INPUT_FBIN}"
  fi
  [[ -d "${DATASET_ROOT}" ]] || {
    echo "Error: DATASET_ROOT is not a directory: ${DATASET_ROOT}"
    exit 1
  }
  DATASET_ROOT_REAL=$(realpath -e "${DATASET_ROOT}")
  INPUT_FBIN_REAL=$(realpath -e "${INPUT_FBIN}") || {
    echo "Error: input fbin not found: ${INPUT_FBIN}"
    exit 1
  }
  [[ -f "${INPUT_FBIN_REAL}" ]] || {
    echo "Error: input fbin is not a file: ${INPUT_FBIN_REAL}"
    exit 1
  }
  case "${INPUT_FBIN_REAL}" in
    "${DATASET_ROOT_REAL}"/*)
      INPUT_FBIN_RELATIVE=${INPUT_FBIN_REAL#"${DATASET_ROOT_REAL}"/}
      INPUT_FBIN_CONTAINER="/datasets/${INPUT_FBIN_RELATIVE}"
      ;;
    *)
      echo "Error: --input-fbin must be below DATASET_ROOT (${DATASET_ROOT_REAL})"
      exit 1
      ;;
  esac
fi

if [[ -n "${OUTPUT_PATH}" ]]; then
  if [[ "${OUTPUT_PATH}" != /* ]]; then
    OUTPUT_PATH="${CALLER_DIR}/${OUTPUT_PATH}"
  fi
  OUTPUT_PATH=$(realpath -m -- "${OUTPUT_PATH}")
  for protected_path in "${HOME}" "${CALLER_DIR}" "${SCRIPT_DIR}"; do
    if [[ "${OUTPUT_PATH}" == "/" || "${protected_path}" == "${OUTPUT_PATH}" || "${protected_path}" == "${OUTPUT_PATH}/"* ]]; then
      echo "Error: refusing to use protected output path: ${OUTPUT_PATH}"
      exit 1
    fi
  done
  if [[ -e "${OUTPUT_PATH}" ]]; then
    if (( OVERWRITE_OUTPUT == 0 )); then
      echo "Error: output path already exists; pass --overwrite-output: ${OUTPUT_PATH}"
      exit 1
    fi
    rm -rf -- "${OUTPUT_PATH}"
  fi
fi

rm -rf result_dir logs 2>/dev/null || true
rm -f ./*.out ./*.err 2>/dev/null || true
mkdir -p result_dir logs

OUT_FMT="logs/cuml-kmeans-run_n${NODES_COUNT}_s${N_SAMPLES}_r${REPEATS}_%j.out"
ERR_FMT="logs/cuml-kmeans-run_n${NODES_COUNT}_s${N_SAMPLES}_r${REPEATS}_%j.err"
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
EXPORT_VARS+=",WARMUP_RUNS=${WARMUP_RUNS}"
EXPORT_VARS+=",REPEATS=${REPEATS}"
EXPORT_VARS+=",N_SAMPLES=${N_SAMPLES}"
EXPORT_VARS+=",N_CLUSTERS=${N_CLUSTERS}"
EXPORT_VARS+=",MAX_ITER=${MAX_ITER}"
EXPORT_VARS+=",TOL=${TOL}"
EXPORT_VARS+=",SEED=${SEED}"
EXPORT_VARS+=",CLUSTER_STD=${CLUSTER_STD}"
EXPORT_VARS+=",DEVICE_BUFFER_SAMPLES=${DEVICE_BUFFER_SAMPLES}"
EXPORT_VARS+=",WORKER_ENV_FILE=${WORKER_ENV_FILE}"
EXPORT_VARS+=",N_FEATURES=${N_FEATURES}"
EXPORT_VARS+=",SAMPLES_PER_GPU=${SAMPLES_PER_GPU}"
EXPORT_VARS+=",INPUT_FBIN=${INPUT_FBIN_CONTAINER}"

JOB_ID=$(sbatch \
  --parsable \
  --job-name="${JOB_NAME}" \
  --nodes="${NODES_COUNT}" \
  "${NODELIST_ARG[@]}" \
  --gres="gpu:${NUM_GPUS_PER_NODE}" \
  --export="${EXPORT_VARS}" \
  --output="${OUT_FMT}" \
  --error="${ERR_FMT}" \
  "${EXTRA_ARGS[@]}" \
  run-cuml-kmeans.slurm)
JOB_ID=${JOB_ID%%;*}

OUT_FILE=${OUT_FMT//%j/${JOB_ID}}
ERR_FILE=${ERR_FMT//%j/${JOB_ID}}

echo "Job submitted with ID: ${JOB_ID}"
echo "Monitor with:"
echo "  squeue -j ${JOB_ID}"
echo "  tail -f ${OUT_FILE}"
echo "  tail -f ${ERR_FILE}"
echo "  tail -f logs/scheduler.log"
echo "  tail -f logs/worker_*.log"
echo
echo "Waiting for job completion..."

cancel_on_interrupt() {
  echo "Cancelling job ${JOB_ID}..." >&2
  scancel "${JOB_ID}" >/dev/null 2>&1 || true
  exit 130
}
trap cancel_on_interrupt INT TERM

JOB_FAILED=0
while true; do
  JOB_LINE=$(squeue -h -j "${JOB_ID}" -o "%T|%r" 2>/dev/null || true)
  [[ -n "${JOB_LINE}" ]] || break
  JOB_REASON=${JOB_LINE#*|}
  if [[ "${JOB_REASON,,}" == *held* ]]; then
    echo "Error: job ${JOB_ID} is held: ${JOB_REASON}" >&2
    scontrol show job -dd "${JOB_ID}" >&2 || true
    scancel "${JOB_ID}" >/dev/null 2>&1 || true
    JOB_FAILED=1
    break
  fi
  sleep 5
done
trap - INT TERM

ACCOUNTING_RECORD=""
if (( JOB_FAILED == 0 )); then
  for _ in $(seq 1 10); do
    ACCOUNTING_RECORD=$(sacct -X -n -P -j "${JOB_ID}" --format=JobIDRaw,State,ExitCode 2>/dev/null \
      | awk -F '|' -v job_id="${JOB_ID}" '$1 == job_id { print; exit }' || true)
    [[ -n "${ACCOUNTING_RECORD}" ]] && break
    sleep 1
  done

  if [[ -z "${ACCOUNTING_RECORD}" ]]; then
    echo "Error: no accounting record found for job ${JOB_ID}" >&2
    JOB_FAILED=1
  else
    IFS='|' read -r _ FINAL_STATE FINAL_EXIT_CODE <<< "${ACCOUNTING_RECORD}"
    if [[ "${FINAL_STATE}" != "COMPLETED" || "${FINAL_EXIT_CODE}" != "0:0" ]]; then
      echo "Error: job ${JOB_ID} finished with state=${FINAL_STATE} exit_code=${FINAL_EXIT_CODE}" >&2
      JOB_FAILED=1
    fi
  fi
fi

if [[ -n "${OUTPUT_PATH}" ]]; then
  echo "Copying results to ${OUTPUT_PATH}..."
  mkdir -p "${OUTPUT_PATH}"
  cp -r result_dir/. "${OUTPUT_PATH}/"
  cp -r logs/. "${OUTPUT_PATH}/" 2>/dev/null || true
fi

if (( JOB_FAILED != 0 )); then
  exit 1
fi

echo "Job ${JOB_ID} completed successfully."
