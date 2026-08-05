# cuML Dask Out-of-Core KMeans on NVL72

This directory runs generated-data and `.fbin` cuML Dask KMeans benchmarks on
a SLURM cluster with Pyxis/Enroot. It starts one Dask scheduler on the first
allocated node and one `dask-cuda-worker` per GPU.

The benchmark keeps input partitions in host memory. A positive
`device_buffer_samples` makes cuML stream bounded row batches to each GPU.

## Prerequisites

- A SLURM cluster with Pyxis/Enroot container support.
- A shared repository path visible on every allocated node.
- A `.sqsh` image containing Python, Dask/Distributed, Dask-CUDA, cuML, CuPy,
  and NumPy.
- For `.fbin` input, a shared `DATASET_ROOT` visible at the same host path on
  every node.

Set the storage locations before launching:

```bash
export IMAGE_DIR="$HOME/velox-testing-data/images/cuml"
export DATASET_ROOT="$HOME/velox-testing-data/datasets"
export RESULTS_BASE="$HOME/velox-testing-data/results/cuml-kmeans"

mkdir -p "$IMAGE_DIR" "$DATASET_ROOT" "$RESULTS_BASE"
```

## Input data

Without `--input-fbin`, the driver generates deterministic, balanced float32
blobs directly on the workers. Generation is independent of partitioning, so
strong-scaling runs use the same logical rows at every node count.

With `--input-fbin`, the file must be below `DATASET_ROOT`. Both layouts below
are supported:

- 8-byte little-endian header: `uint32 rows`, `uint32 features`.
- 16-byte little-endian header: `uint64 rows`, `uint64 features`.

The payload must be a dense, row-major float32 matrix whose size exactly
matches the header. Each worker opens only its assigned contiguous row range
with a read-only NumPy memmap. The driver validates that every worker can read
the mounted file before fitting.

## Single benchmark point

Generated-data smoke test:

```bash
./launch-run.sh \
  --nodes 1 \
  --num-gpus-per-node 4 \
  --worker-image cuml-bench \
  --n-samples 4000000 \
  --n-features 32 \
  --n-clusters 50 \
  --device-buffer-samples 262144 \
  --warmup-runs 1 \
  --repeats 3 \
  --output-path "$RESULTS_BASE/smoke-generated"
```

`.fbin` smoke test; omit `--n-features` to use the header value:

```bash
./launch-run.sh \
  --nodes 1 \
  --num-gpus-per-node 4 \
  --worker-image cuml-bench \
  --n-samples 4000000 \
  --n-clusters 50 \
  --input-fbin "$DATASET_ROOT/example.fbin" \
  --output-path "$RESULTS_BASE/smoke-fbin"
```

Output paths must not already exist. Use `--overwrite-output` only when an
existing single-point result should be replaced.

## Scaling experiments

Weak scaling keeps samples per GPU constant:

```bash
./run-sweep.sh weak \
  --nodes "1 2 4" \
  --num-gpus-per-node 4 \
  --worker-image cuml-bench \
  --samples-per-gpu 5000000 \
  --input-fbin "$DATASET_ROOT/example.fbin" \
  --n-clusters 100 \
  --output-dir "$RESULTS_BASE/fbin-weak"
```

Strong scaling keeps total samples constant and uses the same `.fbin` prefix
at every node count:

```bash
./run-sweep.sh strong \
  --nodes "1 2 4" \
  --num-gpus-per-node 4 \
  --worker-image cuml-bench \
  --n-samples 80000000 \
  --input-fbin "$DATASET_ROOT/example.fbin" \
  --n-clusters 100 \
  --output-dir "$RESULTS_BASE/fbin-strong"
```

Omit `--input-fbin` for generated-data sweeps. Pass `--overwrite` to replace an
existing experiment directory.

The benchmark defaults mirror the standalone SNMG methodology where useful:

- 128 generated features and cluster standard deviation `0.01`.
- `max_iter=20`, `tol=1e-4`, `n_init=1`, and `init="k-means||"`.
- One excluded warm-up and three measured fits.
- A 4,000,000-row device staging buffer.

All values have corresponding command-line overrides.

## Results

Each point directory contains:

- `benchmark_result.json`: compatible raw and aggregate timing keys plus
  warm-up, iteration, inertia, data-setup, throughput, and dataset metadata.
- `benchmark_result.txt`: human-readable summary.
- `fit_results.csv`: one row per warm-up or measured fit.
- Scheduler, driver, worker, and SLURM logs.

The experiment directory also contains `scaling_summary.json` and
`scaling_summary.csv`. Scaling efficiency is relative to the smallest GPU
count in the requested sweep.

The launcher verifies the final `sacct` state. Failed or held jobs return
nonzero and preserve any available logs instead of being reported as complete.
