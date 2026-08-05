#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run one cuML Dask out-of-core KMeans benchmark point."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import struct
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_FEATURES = 128
DEFAULT_CLUSTERS = 100
DEFAULT_DEVICE_BUFFER_SAMPLES = 4_000_000
DEFAULT_REPEATS = 3
DEFAULT_WARMUP_RUNS = 1
DEFAULT_MAX_ITER = 20
DEFAULT_TOL = 1e-4
DEFAULT_SEED = 12345
DEFAULT_CLUSTER_STD = 0.01
DTYPE_BYTES = 4
GIB = 1024**3
LEGACY_FBIN_HEADER_BYTES = 8
EXTENDED_FBIN_HEADER_BYTES = 16
GENERATION_BLOCK_ROWS = 65_536


@dataclass(frozen=True)
class FbinHeader:
    rows: int
    features: int
    header_bytes: int


@dataclass(frozen=True)
class PartitionSpec:
    index: int
    start_row: int
    rows: int
    worker: str


@dataclass
class FitRecord:
    phase: str
    iteration: int
    runtime_ms: float | None
    status: str
    inertia: float | None
    n_iter: int | None
    error: str | None


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be > 0")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be >= 0")
    return parsed


def read_fbin_header(path: Path) -> FbinHeader:
    """Read a float32 fbin header and validate it against the file size."""
    file_size = path.stat().st_size
    with path.open("rb") as stream:
        head = stream.read(EXTENDED_FBIN_HEADER_BYTES)

    if len(head) < LEGACY_FBIN_HEADER_BYTES:
        raise ValueError(f"file too small to contain an fbin header: {path}")

    rows32, features32 = struct.unpack("<II", head[:LEGACY_FBIN_HEADER_BYTES])
    expected32 = LEGACY_FBIN_HEADER_BYTES + rows32 * features32 * DTYPE_BYTES
    if rows32 > 0 and features32 > 0 and file_size == expected32:
        return FbinHeader(int(rows32), int(features32), LEGACY_FBIN_HEADER_BYTES)

    if len(head) == EXTENDED_FBIN_HEADER_BYTES:
        rows64, features64 = struct.unpack("<QQ", head)
        expected64 = EXTENDED_FBIN_HEADER_BYTES + rows64 * features64 * DTYPE_BYTES
        if rows64 > 0 and features64 > 0 and file_size == expected64:
            return FbinHeader(int(rows64), int(features64), EXTENDED_FBIN_HEADER_BYTES)

    raise ValueError(f"file size does not match a float32 fbin header layout: {path}")


def partition_specs(n_samples: int, workers: list[str]) -> list[PartitionSpec]:
    """Split a row prefix evenly into one non-empty partition per worker."""
    if not workers:
        raise ValueError("at least one Dask worker is required")
    if n_samples < len(workers):
        raise ValueError(
            f"n_samples={n_samples} must be >= worker_count={len(workers)} to create one partition per worker"
        )

    base_rows, remainder = divmod(n_samples, len(workers))
    specs: list[PartitionSpec] = []
    start_row = 0
    for index, worker in enumerate(workers):
        rows = base_rows + (1 if index < remainder else 0)
        specs.append(PartitionSpec(index, start_row, rows, worker))
        start_row += rows
    return specs


def load_fbin_partition(
    path: str,
    header_bytes: int,
    start_row: int,
    rows: int,
    features: int,
) -> Any:
    """Open a worker-local, read-only memmap for one contiguous fbin slice."""
    import numpy as np

    offset = header_bytes + start_row * features * DTYPE_BYTES
    return np.memmap(
        path,
        dtype=np.float32,
        mode="r",
        offset=offset,
        shape=(rows, features),
        order="C",
    )


def _hash_uniform(rows: Any, features: Any, seed: int) -> Any:
    """Return deterministic uniform values indexed by global row and feature."""
    import numpy as np

    values = (
        rows[:, None] * np.uint64(0x9E3779B97F4A7C15)
        + features[None, :] * np.uint64(0xBF58476D1CE4E5B9)
        + np.uint64(seed % (1 << 64))
    )
    values ^= values >> np.uint64(30)
    values *= np.uint64(0xBF58476D1CE4E5B9)
    values ^= values >> np.uint64(27)
    values *= np.uint64(0x94D049BB133111EB)
    values ^= values >> np.uint64(31)
    return ((values >> np.uint64(40)).astype(np.float32) + np.float32(0.5)) / np.float32(1 << 24)


def _hash_standard_normal(rows: Any, features: Any, seed: int) -> Any:
    """Generate partition-independent standard normal values with Box-Muller."""
    import numpy as np

    first = _hash_uniform(rows, features, seed)
    second_seed = (seed + 0xD1B54A32D192ED03) % (1 << 64)
    second = _hash_uniform(rows, features, second_seed)
    np.log(first, out=first)
    first *= np.float32(-2.0)
    np.sqrt(first, out=first)
    second *= np.float32(2.0 * np.pi)
    np.cos(second, out=second)
    first *= second
    return first


def generate_blob_partition(
    start_row: int,
    rows: int,
    centers: Any,
    cluster_std: float,
    seed: int,
) -> Any:
    """Generate a deterministic host-resident blob partition."""
    import numpy as np

    centers = np.asarray(centers, dtype=np.float32, order="C")
    n_clusters, n_features = centers.shape
    feature_ids = np.arange(n_features, dtype=np.uint64)
    result = np.empty((rows, n_features), dtype=np.float32, order="C")

    for block_start in range(0, rows, GENERATION_BLOCK_ROWS):
        block_end = min(rows, block_start + GENERATION_BLOCK_ROWS)
        global_rows = np.arange(
            start_row + block_start,
            start_row + block_end,
            dtype=np.uint64,
        )
        labels = (global_rows % np.uint64(n_clusters)).astype(np.int64)
        block = centers[labels].copy(order="C")
        block += _hash_standard_normal(global_rows, feature_ids, seed) * np.float32(cluster_std)
        result[block_start:block_end] = block

    return result


def _path_preflight(path: str) -> str | None:
    candidate = Path(path)
    if not candidate.is_file():
        return f"not a readable file: {path}"
    try:
        with candidate.open("rb") as stream:
            stream.read(1)
    except OSError as exc:
        return f"cannot read {path}: {exc}"
    return None


def _make_centers(n_clusters: int, n_features: int, seed: int) -> Any:
    import numpy as np

    rng = np.random.default_rng(seed)
    return np.ascontiguousarray(
        rng.uniform(-10.0, 10.0, size=(n_clusters, n_features)),
        dtype=np.float32,
    )


def _worker_pinned_array(*, tasks: list[Any], specs: list[PartitionSpec], n_features: int) -> Any:
    """Build an array whose concrete block tasks have hard worker restrictions."""
    import numpy as np
    from dask.array import Array
    from dask.base import tokenize
    from dask.highlevelgraph import HighLevelGraph, MaterializedLayer

    if len(tasks) != len(specs):
        raise ValueError("one input task is required per partition")

    name = f"worker-pinned-kmeans-input-{tokenize(tasks, specs, n_features)}"
    keys = [(name, spec.index, 0) for spec in specs]
    worker_by_key = {key: spec.worker for key, spec in zip(keys, specs)}

    def worker_for_key(key: Any) -> str:
        return worker_by_key[key]

    layer = MaterializedLayer(
        dict(zip(keys, tasks)),
        annotations={
            "workers": worker_for_key,
            "allow_other_workers": False,
        },
    )
    graph = HighLevelGraph({name: layer}, {name: set()})
    return Array(
        graph,
        name,
        chunks=(tuple(spec.rows for spec in specs), (n_features,)),
        dtype=np.float32,
        meta=np.empty((0, n_features), dtype=np.float32),
    )


def _persist_worker_pinned_array(*, client: Any, array: Any, specs: list[PartitionSpec]) -> Any:
    """Materialize final array blocks and verify their exact worker locations."""
    from distributed import wait

    persisted = client.persist(array)
    futures = client.futures_of(persisted)
    wait(futures)
    future_by_key = {future.key: future for future in futures}
    locations = client.who_has(futures)

    for spec in specs:
        key = (persisted.name, spec.index, 0)
        future = future_by_key.get(key)
        if future is None:
            raise RuntimeError(f"missing persisted future for partition {spec.index}: {key}")
        if future.status == "error":
            raise future.exception()
        if future.status != "finished":
            raise RuntimeError(f"partition {spec.index} has unexpected status {future.status}")
        actual_workers = set(locations.get(key, ()))
        if actual_workers != {spec.worker}:
            raise RuntimeError(f"partition {spec.index} expected on {spec.worker}, found on {sorted(actual_workers)}")
        print(
            f"input partition {spec.index}: rows=[{spec.start_row}, {spec.start_row + spec.rows}) worker={spec.worker}",
            flush=True,
        )

    return persisted


def build_generated_input_array(
    *,
    client: Any,
    workers: list[str],
    n_samples: int,
    n_features: int,
    n_clusters: int,
    seed: int,
    cluster_std: float,
) -> tuple[Any, list[PartitionSpec]]:
    specs = partition_specs(n_samples, workers)
    centers = _make_centers(n_clusters, n_features, seed)
    tasks = [
        (
            generate_blob_partition,
            spec.start_row,
            spec.rows,
            centers,
            cluster_std,
            seed,
        )
        for spec in specs
    ]
    array = _worker_pinned_array(
        tasks=tasks,
        specs=specs,
        n_features=n_features,
    )
    persisted = _persist_worker_pinned_array(
        client=client,
        array=array,
        specs=specs,
    )
    return persisted, specs


def build_fbin_input_array(
    *,
    client: Any,
    workers: list[str],
    n_samples: int,
    n_features: int,
    input_fbin: Path,
    fbin_header: FbinHeader,
) -> tuple[Any, list[PartitionSpec]]:
    preflight = client.run(_path_preflight, str(input_fbin))
    failures = {worker: error for worker, error in preflight.items() if error}
    if failures:
        details = "; ".join(f"{worker}: {error}" for worker, error in sorted(failures.items()))
        raise RuntimeError(f"input fbin is not readable on every worker: {details}")

    specs = partition_specs(n_samples, workers)
    tasks = [
        (
            load_fbin_partition,
            str(input_fbin),
            fbin_header.header_bytes,
            spec.start_row,
            spec.rows,
            n_features,
        )
        for spec in specs
    ]
    array = _worker_pinned_array(
        tasks=tasks,
        specs=specs,
        n_features=n_features,
    )
    persisted = _persist_worker_pinned_array(
        client=client,
        array=array,
        specs=specs,
    )
    return persisted, specs


def aggregate_times(times_ms: list[float]) -> dict[str, float]:
    if not times_ms:
        return {}
    return {
        "avg": statistics.mean(times_ms),
        "min": min(times_ms),
        "max": max(times_ms),
        "median": statistics.median(times_ms),
        "geometric_mean": statistics.geometric_mean(times_ms),
    }


def throughput_metrics(*, n_samples: int, n_features: int, median_ms: float, median_n_iter: float) -> dict[str, float]:
    median_s = median_ms / 1000.0
    input_gib = n_samples * n_features * DTYPE_BYTES / GIB
    return {
        "rows_per_s": n_samples / median_s,
        "row_iters_per_s": n_samples * median_n_iter / median_s,
        "gib_iters_per_s": input_gib * median_n_iter / median_s,
    }


def write_result_files(
    *,
    output_dir: Path,
    context: dict[str, Any],
    records: list[FitRecord],
    data_setup_ms: float | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    measured = [record for record in records if record.phase == "fit"]
    warmups = [record for record in records if record.phase == "warmup"]
    failures = {
        f"{record.phase}_{record.iteration}": record.error
        for record in records
        if record.status != "success" and record.error
    }
    raw_times = [record.runtime_ms for record in measured]
    successful = [record for record in measured if record.runtime_ms is not None]
    successful_times = [float(record.runtime_ms) for record in successful]
    successful_iters = [int(record.n_iter) for record in successful if record.n_iter is not None]
    agg = aggregate_times(successful_times)
    if successful_times:
        agg["lukewarm"] = successful_times[0]
    throughput = {}
    if agg and successful_iters:
        throughput = throughput_metrics(
            n_samples=int(context["n_samples"]),
            n_features=int(context["n_features"]),
            median_ms=agg["median"],
            median_n_iter=statistics.median(successful_iters),
        )

    benchmark_json = {
        "context": context,
        "kmeans": {
            "raw_times_ms": {"fit": raw_times},
            "failed_queries": failures,
            "agg_times_ms": {key: {"fit": value} for key, value in agg.items()},
            "warmup_times_ms": [record.runtime_ms for record in warmups],
            "raw_inertias": [record.inertia for record in measured],
            "raw_n_iters": [record.n_iter for record in measured],
            "data_setup_ms": data_setup_ms,
            "throughput": throughput,
        },
    }
    (output_dir / "benchmark_result.json").write_text(
        json.dumps(benchmark_json, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "cuML Dask Out-of-Core KMeans Benchmark",
        "=======================================",
        f"nodes: {context['node_count']}",
        f"workers: {context['worker_count']}",
        f"samples: {context['n_samples']}",
        f"features: {context['n_features']}",
        f"clusters: {context['n_clusters']}",
        f"data source: {context['data_source']}",
        f"input GiB: {context['input_gib']:.3f}",
        f"device buffer samples: {context['device_buffer_samples']}",
        f"data setup ms: {data_setup_ms}",
        "",
    ]
    for record in records:
        lines.append(
            f"phase={record.phase} iteration={record.iteration} status={record.status} "
            f"time_ms={record.runtime_ms} n_iter={record.n_iter} inertia={record.inertia}"
        )
    if agg:
        lines.extend(["", "Measured fit aggregates (ms)"])
        lines.extend(f"- {key}: {value}" for key, value in agg.items())
    if throughput:
        lines.extend(["", "Throughput"])
        lines.extend(f"- {key}: {value}" for key, value in throughput.items())
    if failures:
        lines.extend(["", "Failures"])
        lines.extend(f"- {key}: {value}" for key, value in failures.items())
    (output_dir / "benchmark_result.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    fieldnames = [
        "phase",
        "iteration",
        "status",
        "runtime_ms",
        "inertia",
        "n_iter",
        "error",
        "node_count",
        "worker_count",
        "gpu_count",
        "n_samples",
        "samples_per_gpu",
        "n_features",
        "n_clusters",
        "device_buffer_samples",
        "data_source",
    ]
    with (output_dir / "fit_results.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = asdict(record)
            row.update({key: context[key] for key in fieldnames if key in context})
            writer.writerow(row)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run cuML Dask out-of-core KMeans.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-samples", type=_positive_int, required=True)
    parser.add_argument("--n-features", type=_positive_int)
    parser.add_argument("--n-clusters", type=_positive_int, default=DEFAULT_CLUSTERS)
    parser.add_argument("--input-fbin", type=Path)
    parser.add_argument("--device-buffer-samples", type=_positive_int, default=DEFAULT_DEVICE_BUFFER_SAMPLES)
    parser.add_argument("--warmup-runs", type=_nonnegative_int, default=DEFAULT_WARMUP_RUNS)
    parser.add_argument("--repeats", type=_positive_int, default=DEFAULT_REPEATS)
    parser.add_argument("--max-iter", type=_positive_int, default=DEFAULT_MAX_ITER)
    parser.add_argument("--tol", type=float, default=DEFAULT_TOL)
    parser.add_argument("--seed", type=_nonnegative_int, default=DEFAULT_SEED)
    parser.add_argument("--cluster-std", type=float, default=DEFAULT_CLUSTER_STD)
    parser.add_argument("--samples-per-gpu", type=_positive_int)
    parser.add_argument("--node-count", type=_positive_int, default=1)
    parser.add_argument("--gpus-per-node", type=_positive_int, default=1)
    parser.add_argument("--expected-workers", type=_positive_int)
    parser.add_argument("--scheduler-address", default="")
    parser.add_argument("--local-cluster", action="store_true")
    return parser


def resolve_dataset(args: argparse.Namespace) -> FbinHeader | None:
    if args.cluster_std <= 0:
        raise ValueError("--cluster-std must be positive")
    if args.tol <= 0:
        raise ValueError("--tol must be positive")
    if args.n_samples < args.n_clusters:
        raise ValueError("--n-samples must be >= --n-clusters")

    if args.input_fbin is None:
        if args.n_features is None:
            args.n_features = DEFAULT_FEATURES
        return None

    if args.input_fbin.suffix != ".fbin":
        raise ValueError("--input-fbin must point to a .fbin file")
    if not args.input_fbin.is_file():
        raise ValueError(f"input fbin not found: {args.input_fbin}")
    header = read_fbin_header(args.input_fbin)
    if args.n_features is None:
        args.n_features = header.features
    elif args.n_features != header.features:
        raise ValueError(
            f"--n-features={args.n_features} does not match fbin feature count {header.features}: {args.input_fbin}"
        )
    if args.n_samples > header.rows:
        raise ValueError(f"requested {args.n_samples} rows, but {args.input_fbin} contains {header.rows}")
    return header


def _make_model(args: argparse.Namespace, client: Any) -> Any:
    try:
        from cuml.dask.cluster import KMeans as DaskKMeans
    except Exception as exc:  # pragma: no cover - runtime dependency guard
        raise RuntimeError("cuml.dask.cluster.KMeans is required in the benchmark image") from exc

    return DaskKMeans(
        client=client,
        n_clusters=args.n_clusters,
        max_iter=args.max_iter,
        tol=args.tol,
        n_init=1,
        random_state=args.seed,
        init="k-means||",
        device_buffer_samples=args.device_buffer_samples,
    )


def _fit_once(args: argparse.Namespace, client: Any, array: Any, phase: str, iteration: int) -> FitRecord:
    model = _make_model(args, client)
    start = time.perf_counter()
    try:
        model.fit(array)
        runtime_ms = (time.perf_counter() - start) * 1000.0
        return FitRecord(
            phase=phase,
            iteration=iteration,
            runtime_ms=runtime_ms,
            status="success",
            inertia=float(model.inertia_),
            n_iter=int(model.n_iter_),
            error=None,
        )
    except Exception:
        error = traceback.format_exc()
        print(error, file=sys.stderr, flush=True)
        return FitRecord(
            phase=phase,
            iteration=iteration,
            runtime_ms=None,
            status="error",
            inertia=None,
            n_iter=None,
            error=error,
        )


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    fbin_header = resolve_dataset(args)
    if args.local_cluster and args.scheduler_address:
        raise ValueError("--local-cluster and --scheduler-address are mutually exclusive")

    from dask.distributed import Client
    from dask_cuda import LocalCUDACluster

    local_cluster = None
    if args.local_cluster:
        local_cluster = LocalCUDACluster()
        client = Client(local_cluster)
    else:
        if not args.scheduler_address:
            raise ValueError("Either --scheduler-address or --local-cluster is required")
        client = Client(args.scheduler_address)

    records: list[FitRecord] = []
    data_setup_ms: float | None = None
    exit_code = 0
    try:
        if args.expected_workers:
            client.wait_for_workers(args.expected_workers, timeout=300)
        workers = sorted(client.scheduler_info().get("workers", {}))
        if args.expected_workers and len(workers) != args.expected_workers:
            raise RuntimeError(f"expected {args.expected_workers} workers, found {len(workers)}")

        setup_start = time.perf_counter()
        if args.input_fbin is None:
            array, specs = build_generated_input_array(
                client=client,
                workers=workers,
                n_samples=args.n_samples,
                n_features=args.n_features,
                n_clusters=args.n_clusters,
                seed=args.seed,
                cluster_std=args.cluster_std,
            )
        else:
            assert fbin_header is not None
            array, specs = build_fbin_input_array(
                client=client,
                workers=workers,
                n_samples=args.n_samples,
                n_features=args.n_features,
                input_fbin=args.input_fbin,
                fbin_header=fbin_header,
            )
        data_setup_ms = (time.perf_counter() - setup_start) * 1000.0

        worker_count = len(workers)
        node_count = (
            len(
                {
                    details.get("host")
                    for details in client.scheduler_info().get("workers", {}).values()
                    if details.get("host")
                }
            )
            or args.node_count
        )
        samples_per_gpu = args.samples_per_gpu or args.n_samples // max(1, worker_count)
        input_gib = args.n_samples * args.n_features * DTYPE_BYTES / GIB
        context: dict[str, Any] = {
            "benchmark": ["kmeans"],
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "engine": "cuml-dask-gpu",
            "fit_mode": "out-of-core",
            "kind": "single-node" if node_count == 1 else "multi-node",
            "iterations_count": args.repeats,
            "warmup_runs": args.warmup_runs,
            "node_count": node_count,
            "worker_count": worker_count,
            "gpu_count": worker_count,
            "gpus_per_node": args.gpus_per_node,
            "n_samples": args.n_samples,
            "rows_used": args.n_samples,
            "samples_per_gpu": samples_per_gpu,
            "n_features": args.n_features,
            "n_clusters": args.n_clusters,
            "dtype": "float32",
            "input_gib": input_gib,
            "partition_count": len(specs),
            "data_source": "fbin" if args.input_fbin else "generated",
            "dataset_name": args.input_fbin.name if args.input_fbin else "generated-blobs",
            "input_fbin": str(args.input_fbin) if args.input_fbin else None,
            "fbin_rows": fbin_header.rows if fbin_header else None,
            "fbin_features": fbin_header.features if fbin_header else None,
            "fbin_header_bytes": fbin_header.header_bytes if fbin_header else None,
            "device_buffer_samples": args.device_buffer_samples,
            "max_iter": args.max_iter,
            "tol": args.tol,
            "n_init": 1,
            "init": "k-means||",
            "seed": args.seed,
            "cluster_std": args.cluster_std if args.input_fbin is None else None,
        }

        for iteration in range(1, args.warmup_runs + 1):
            record = _fit_once(args, client, array, "warmup", iteration)
            records.append(record)
            if record.status != "success":
                exit_code = 1
                break

        if exit_code == 0:
            for iteration in range(1, args.repeats + 1):
                record = _fit_once(args, client, array, "fit", iteration)
                records.append(record)
                if record.status != "success":
                    exit_code = 1
                    break

        write_result_files(
            output_dir=args.output_dir,
            context=context,
            records=records,
            data_setup_ms=data_setup_ms,
        )
    finally:
        client.close()
        if local_cluster is not None:
            local_cluster.close()

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
