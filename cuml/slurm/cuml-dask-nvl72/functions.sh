#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# shellcheck disable=SC1083,SC2153

function validate_environment_preconditions {
  local missing=()
  for var in "$@"; do
    if [[ -z "${!var+x}" || -z "${!var}" ]]; then
      missing+=("$var")
    fi
  done
  if ((${#missing[@]})); then
    echo_error "required env var ${missing[*]} not set"
  fi
}

function setup {
  validate_environment_preconditions \
    VT_ROOT IMAGE_DIR WORKER_IMAGE LOGS RESULT_DIR \
    DASK_SCHEDULER_ADDRESS NUM_GPUS_PER_NODE NUM_NODES TOTAL_WORKERS \
    N_SAMPLES N_CLUSTERS WARMUP_RUNS REPEATS DEVICE_BUFFER_SAMPLES

  [[ -d "${VT_ROOT}" ]] || echo_error "VT_ROOT must be a valid directory"

  local worker_image_path="${IMAGE_DIR}/${WORKER_IMAGE}.sqsh"
  [[ -f "${worker_image_path}" ]] || echo_error "worker image does not exist at ${worker_image_path}"
  [[ -f "${WORKER_ENV_FILE}" ]] || echo_error "worker env file does not exist at ${WORKER_ENV_FILE}"
  if [[ -n "${INPUT_FBIN:-}" ]]; then
    [[ -d "${DATASET_ROOT}" ]] || echo_error "DATASET_ROOT must be a valid directory for fbin input"
  fi

  mkdir -p "${LOGS}" "${RESULT_DIR}"
}

function run_scheduler {
  local worker_image="${IMAGE_DIR}/${WORKER_IMAGE}.sqsh"

  srun -N1 -w "${COORD}" --ntasks=1 --overlap \
    --container-image="${worker_image}" \
    --container-remap-root \
    --export=ALL \
    --container-mounts="${VT_ROOT}:/workspace,${WORKER_ENV_FILE}:/var/worker_env_file" \
    -- bash -lc "
set -euo pipefail
set -a
source /var/worker_env_file
set +a
export HOME=/tmp
export XDG_CACHE_HOME=/tmp/.cache
export CUPY_CACHE_DIR=/tmp/cupy-kernel-cache
mkdir -p "\${XDG_CACHE_HOME}" "\${CUPY_CACHE_DIR}"
exec dask scheduler \
  --host ${COORD_IP:-0.0.0.0} \
  --port ${DASK_SCHEDULER_PORT} \
  --dashboard-address :${DASK_DASHBOARD_PORT}
" >> "${LOGS}/scheduler.log" 2>&1 &
}

function run_worker {
  [[ $# -eq 3 ]] || echo_error "$0 expected arguments 'gpu_id', 'node_id', and 'worker_id'"
  local gpu_id="$1"
  local node="$2"
  local worker_id="$3"

  local worker_image="${IMAGE_DIR}/${WORKER_IMAGE}.sqsh"
  local libnvidia_ml_host=""
  local dataset_mount=""

  if [[ -n "${INPUT_FBIN:-}" ]]; then
    dataset_mount=",${DATASET_ROOT}:/datasets:ro"
  fi

  for candidate in \
    /usr/lib/aarch64-linux-gnu/libnvidia-ml.so.1 \
    /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1
  do
    if [[ -e "${candidate}" ]]; then
      libnvidia_ml_host="${candidate}"
      break
    fi
  done

  local driver_mounts=""
  if [[ -n "${libnvidia_ml_host}" ]]; then
    driver_mounts+=",${libnvidia_ml_host}:/usr/local/lib/libnvidia-ml.so.1"
  fi

  srun -N1 -w "${node}" --ntasks=1 --overlap \
    --container-image="${worker_image}" \
    --container-remap-root \
    --export=ALL,NVIDIA_VISIBLE_DEVICES=all,NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    --container-mounts="${VT_ROOT}:/workspace,${WORKER_ENV_FILE}:/var/worker_env_file${driver_mounts}${dataset_mount}" \
    -- bash -lc "
set -euo pipefail
set -a
source /var/worker_env_file
set +a
export LD_LIBRARY_PATH=/usr/local/lib:\${LD_LIBRARY_PATH:-}
export CUDA_VISIBLE_DEVICES=${gpu_id}
export HOME=/tmp
export XDG_CACHE_HOME=/tmp/.cache
export CUPY_CACHE_DIR=/tmp/cupy-kernel-cache
mkdir -p "\${XDG_CACHE_HOME}" "\${CUPY_CACHE_DIR}"
exec dask-cuda-worker ${DASK_SCHEDULER_ADDRESS} \
  --rmm-pool-size \${RMM_POOL_SIZE} \
  --device-memory-limit \${DASK_CUDA_DEVICE_MEMORY_LIMIT} \
  --memory-limit \${DASK_CUDA_MEMORY_LIMIT}
" >> "${LOGS}/worker_${worker_id}.log" 2>&1 &
}

function wait_for_scheduler {
  local retries=60
  local worker_image="${IMAGE_DIR}/${WORKER_IMAGE}.sqsh"
  if srun -N1 -w "${COORD}" --ntasks=1 --overlap \
    --container-image="${worker_image}" \
    --container-remap-root \
    --export=ALL,VT_WAIT_RETRIES="${retries}" \
    --container-mounts="${VT_ROOT}:/workspace,${WORKER_ENV_FILE}:/var/worker_env_file" \
    -- bash -lc "python - <<'PY'
import os
import sys
import time
from distributed import Client

addr = os.environ['DASK_SCHEDULER_ADDRESS']
retries = int(os.environ.get('VT_WAIT_RETRIES', '60'))

for _ in range(retries):
    try:
        c = Client(addr, timeout='3s')
        c.close()
        sys.exit(0)
    except Exception:
        time.sleep(2)
sys.exit(1)
PY"
  then
    echo_success "Dask scheduler is reachable at ${DASK_SCHEDULER_ADDRESS}"
    return 0
  fi
  echo_error "Dask scheduler did not become reachable in time"
}

function wait_for_workers_to_register {
  [[ $# -eq 1 ]] || echo_error "$0 expected one argument for expected workers"
  local expected_workers="$1"
  local retries=120
  local worker_image="${IMAGE_DIR}/${WORKER_IMAGE}.sqsh"

  if srun -N1 -w "${COORD}" --ntasks=1 --overlap \
    --container-image="${worker_image}" \
    --container-remap-root \
    --export=ALL,VT_EXPECTED_WORKERS="${expected_workers}",VT_WAIT_RETRIES="${retries}" \
    --container-mounts="${VT_ROOT}:/workspace,${WORKER_ENV_FILE}:/var/worker_env_file" \
    -- bash -lc "python - <<'PY'
import os
import sys
from distributed import Client

addr = os.environ['DASK_SCHEDULER_ADDRESS']
expected = int(os.environ['VT_EXPECTED_WORKERS'])
timeout_s = int(os.environ.get('VT_WAIT_RETRIES', '120')) * 2

try:
    c = Client(addr, timeout='10s')
except Exception:
    sys.exit(1)

try:
    c.wait_for_workers(expected, timeout=timeout_s)
    c.close()
    sys.exit(0)
except Exception:
    try:
        count = len(c.scheduler_info().get('workers', {}))
    except Exception:
        count = -1
    print('wait_for_workers timeout/failure: expected={} observed={}'.format(expected, count), file=sys.stderr)
    c.close()
    sys.exit(1)
PY"
  then
    echo_success "All ${expected_workers} Dask workers registered"
    return 0
  fi
  echo_error "Timed out waiting for ${expected_workers} workers to register"
}

function run_kmeans_benchmark {
  local worker_image="${IMAGE_DIR}/${WORKER_IMAGE}.sqsh"
  local script="/workspace/cuml/testing/performance_benchmarks/run_kmeans_benchmark.py"
  local result_dir="/workspace/cuml/slurm/cuml-dask-nvl72/result_dir"
  local dataset_mount=""

  if [[ -n "${INPUT_FBIN:-}" ]]; then
    dataset_mount=",${DATASET_ROOT}:/datasets:ro"
  fi

  local samples_per_gpu_arg=""
  if [[ -n "${SAMPLES_PER_GPU:-}" ]]; then
    samples_per_gpu_arg="--samples-per-gpu ${SAMPLES_PER_GPU}"
  fi

  local n_features_arg=""
  if [[ -n "${N_FEATURES:-}" ]]; then
    n_features_arg="--n-features ${N_FEATURES}"
  fi

  local input_fbin_arg=""
  if [[ -n "${INPUT_FBIN:-}" ]]; then
    input_fbin_arg="--input-fbin ${INPUT_FBIN}"
  fi

  srun -N1 -w "${COORD}" --ntasks=1 --overlap \
    --container-image="${worker_image}" \
    --container-remap-root \
    --export=ALL \
    --container-mounts="${VT_ROOT}:/workspace,${WORKER_ENV_FILE}:/var/worker_env_file${dataset_mount}" \
    -- bash -lc "
set -euo pipefail
set -a
source /var/worker_env_file
set +a
export HOME=/tmp
export XDG_CACHE_HOME=/tmp/.cache
export CUPY_CACHE_DIR=/tmp/cupy-kernel-cache
mkdir -p "\${XDG_CACHE_HOME}" "\${CUPY_CACHE_DIR}"
python ${script} \
  --scheduler-address ${DASK_SCHEDULER_ADDRESS} \
  --output-dir ${result_dir} \
  --warmup-runs ${WARMUP_RUNS} \
  --repeats ${REPEATS} \
  --n-samples ${N_SAMPLES} \
  --n-clusters ${N_CLUSTERS} \
  --max-iter ${MAX_ITER} \
  --tol ${TOL} \
  --seed ${SEED} \
  --cluster-std ${CLUSTER_STD} \
  --device-buffer-samples ${DEVICE_BUFFER_SAMPLES} \
  --expected-workers ${TOTAL_WORKERS} \
  --node-count ${NUM_NODES} \
  --gpus-per-node ${NUM_GPUS_PER_NODE} \
  ${n_features_arg} \
  ${input_fbin_arg} \
  ${samples_per_gpu_arg}
" >> "${LOGS}/driver.log" 2>&1
}

function collect_results {
  cp "${LOGS}"/*.log "${RESULT_DIR}/" 2>/dev/null || true
  cp "${LOGS}"/*.out "${LOGS}"/*.err "${RESULT_DIR}/" 2>/dev/null || true
}

function inject_benchmark_metadata {
  local result_file="${RESULT_DIR}/benchmark_result.json"
  if [[ ! -f "${result_file}" ]]; then
    echo_warning "benchmark_result.json not found, skipping metadata injection"
    return
  fi

  local worker_image_path="${IMAGE_DIR}/${WORKER_IMAGE}.sqsh"
  local image_digest
  image_digest=$(sha256sum "${worker_image_path}" | awk '{print $1}') || true
  image_digest="${image_digest:-unknown}"

  IMAGE_DIGEST="${image_digest}" python - <<'PY'
import json
import os
from datetime import datetime, timezone
from pathlib import Path

result_path = Path(os.environ["RESULT_DIR"]) / "benchmark_result.json"
data = json.loads(result_path.read_text())
context = data.setdefault("context", {})
context.update(
    {
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "engine": "cuml-dask-gpu",
        "kind": "single-node" if int(os.environ["NUM_NODES"]) == 1 else "multi-node",
        "worker_count": int(os.environ["TOTAL_WORKERS"]),
        "node_count": int(os.environ["NUM_NODES"]),
        "gpu_count": int(os.environ["TOTAL_WORKERS"]),
        "image_digest": os.environ["IMAGE_DIGEST"],
        "worker_image": os.environ["WORKER_IMAGE"],
    }
)
result_path.write_text(json.dumps(data, indent=2))
PY
}

function cleanup_cluster {
  pkill -f "dask scheduler --host" >/dev/null 2>&1 || true
  pkill -f "dask-cuda-worker ${DASK_SCHEDULER_ADDRESS}" >/dev/null 2>&1 || true
}
