#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/defaults.env"

NODE_COUNTS=(1 2 4)
NUM_GPUS_PER_NODE=4
ITERATIONS=5
WORKER_IMAGE=""
SAMPLES_PER_GPU=""
N_FEATURES=64
N_CLUSTERS=100
MAX_ITER=100
SEED=42
CHUNK_ROWS=0
INTER_RUN_SLEEP=60

while [[ $# -gt 0 ]]; do
  case "$1" in
    -n|--nodes) read -ra NODE_COUNTS <<< "$2"; shift 2 ;;
    -g|--num-gpus-per-node) NUM_GPUS_PER_NODE="$2"; shift 2 ;;
    -i|--iterations) ITERATIONS="$2"; shift 2 ;;
    -w|--worker-image) WORKER_IMAGE="$2"; shift 2 ;;
    --samples-per-gpu) SAMPLES_PER_GPU="$2"; shift 2 ;;
    --n-features) N_FEATURES="$2"; shift 2 ;;
    --n-clusters) N_CLUSTERS="$2"; shift 2 ;;
    --max-iter) MAX_ITER="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --chunk-rows) CHUNK_ROWS="$2"; shift 2 ;;
    --inter-run-sleep) INTER_RUN_SLEEP="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

[[ -n "${WORKER_IMAGE}" ]] || { echo "Error: --worker-image is required"; exit 1; }
[[ -n "${SAMPLES_PER_GPU}" ]] || { echo "Error: --samples-per-gpu is required"; exit 1; }

total=${#NODE_COUNTS[@]}
run=0
for nodes in "${NODE_COUNTS[@]}"; do
  run=$((run + 1))
  total_samples=$((SAMPLES_PER_GPU * NUM_GPUS_PER_NODE * nodes))
  output_dir="${RESULTS_BASE}/kmeans_weakscale_n${nodes}_s${total_samples}"

  echo "========================================"
  echo "Run ${run}/${total}: nodes=${nodes}, samples=${total_samples}"
  echo "========================================"

  rm -rf "${output_dir}"
  "${SCRIPT_DIR}/launch-run.sh" \
    --nodes "${nodes}" \
    --num-gpus-per-node "${NUM_GPUS_PER_NODE}" \
    --iterations "${ITERATIONS}" \
    --worker-image "${WORKER_IMAGE}" \
    --samples-per-gpu "${SAMPLES_PER_GPU}" \
    --n-features "${N_FEATURES}" \
    --n-clusters "${N_CLUSTERS}" \
    --max-iter "${MAX_ITER}" \
    --seed "${SEED}" \
    --chunk-rows "${CHUNK_ROWS}" \
    --output-path "${output_dir}"

  if (( run < total )); then
    echo "Cooling down for ${INTER_RUN_SLEEP}s before next run..."
    sleep "${INTER_RUN_SLEEP}"
  fi
done

echo "Weak-scaling sweep complete."
