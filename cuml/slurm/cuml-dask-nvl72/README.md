# cuML Dask KMeans Weak-Scaling (NVL72-style)

This directory provides SLURM orchestration for running single-node and multi-node
cuML Dask KMeans benchmarks with weak-scaling semantics.

## What this module does

- Launches a Dask scheduler on the head node.
- Launches one `dask-cuda-worker` per GPU across allocated nodes.
- Runs a benchmark driver (`cuml/testing/performance_benchmarks/run_kmeans_benchmark.py`)
  against that cluster.
- Writes local artifacts under `result_dir/`:
  - `benchmark_result.json`
  - `benchmark_result.txt`
  - `weak_scaling.csv`
  - copied logs

## Prerequisites

- SLURM cluster with Pyxis/Enroot style container execution.
- A `.sqsh` image that contains at least:
  - Python 3
  - `dask`, `distributed`, `dask-cuda`
  - `cuml`, `cupy`, `dask-array`
- Image available at `${IMAGE_DIR}/<worker-image>.sqsh` (default `IMAGE_DIR=/scratch/${USER}/images/cuml`).

## Quick start

From this directory:

```bash
# 1) Single-node smoke
./launch-run.sh \
  --nodes 1 \
  --num-gpus-per-node 4 \
  --worker-image <your-image-name> \
  --samples-per-gpu 1000000 \
  --n-features 32 \
  --n-clusters 50 \
  --iterations 3

# 2) Multi-node weak scaling
./run-sweep.sh \
  --worker-image <your-image-name> \
  --nodes "1 2 4" \
  --num-gpus-per-node 4 \
  --samples-per-gpu 5000000 \
  --n-features 64 \
  --n-clusters 100 \
  --iterations 5
```

Weak scaling uses:

`total_samples = samples_per_gpu * gpus_per_node * node_count`

## Main scripts

- `launch-run.sh`: submits one benchmark run.
- `run-cuml-kmeans.slurm`: SLURM envelope and exported environment checks.
- `run-cuml-kmeans.sh`: runtime flow (scheduler -> workers -> benchmark).
- `run-sweep.sh`: weak-scaling loop over node counts.

## Validation checklist

- Verify scheduler/worker registration in `logs/scheduler.log` and `logs/worker_*.log`.
- Confirm `benchmark_result.json` exists and includes:
  - `context.node_count`, `context.worker_count`, `context.n_samples`
- Confirm `weak_scaling.csv` has one row per iteration.
- For smoke tests, start with small `samples_per_gpu` and increase gradually.
