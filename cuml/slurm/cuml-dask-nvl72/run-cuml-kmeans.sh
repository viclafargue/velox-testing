#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

source "${SCRIPT_DIR}/echo_helpers.sh"
source "${SCRIPT_DIR}/functions.sh"

trap 'inject_benchmark_metadata; collect_results; cleanup_cluster' EXIT

echo "Setting up benchmark environment..."
setup

echo "Starting Dask scheduler on ${COORD}..."
run_scheduler
wait_for_scheduler

echo "Starting ${TOTAL_WORKERS} Dask workers..."
worker_id=0
for node in $(scontrol show hostnames "${SLURM_JOB_NODELIST}"); do
  for gpu_id in $(seq 0 $((NUM_GPUS_PER_NODE - 1))); do
    echo "  Starting worker ${worker_id} on ${node} (GPU ${gpu_id})"
    run_worker "${gpu_id}" "${node}" "${worker_id}"
    worker_id=$((worker_id + 1))
  done
done
wait_for_workers_to_register "${TOTAL_WORKERS}"

echo "Running cuML Dask KMeans benchmark..."
run_kmeans_benchmark

echo "Benchmark run finished. Artifacts in ${RESULT_DIR}"
