#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import dask.array as da
from dask.distributed import Client
from dask_cuda import LocalCUDACluster

try:
    from cuml.dask.cluster import KMeans as DaskKMeans
except Exception as exc:  # pragma: no cover - runtime dependency guard
    raise RuntimeError("cuml.dask.cluster.KMeans is required in the benchmark image") from exc


@dataclass
class IterationRecord:
    iteration: int
    runtime_ms: float | None
    status: str
    inertia: float | None
    error: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run cuML Dask KMeans benchmark.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--n-samples", type=int, required=True)
    parser.add_argument("--n-features", type=int, default=64)
    parser.add_argument("--n-clusters", type=int, default=100)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk-rows", type=int, default=0)
    parser.add_argument("--samples-per-gpu", type=int, default=0)
    parser.add_argument("--node-count", type=int, default=1)
    parser.add_argument("--gpus-per-node", type=int, default=1)
    parser.add_argument("--expected-workers", type=int, default=0)
    parser.add_argument("--scheduler-address", default="")
    parser.add_argument("--local-cluster", action="store_true")
    return parser.parse_args()


def geometric_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return math.exp(sum(math.log(v) for v in values) / len(values))


def aggregate_times(times_ms: list[float]) -> dict[str, float]:
    if not times_ms:
        return {}
    return {
        "avg": statistics.mean(times_ms),
        "min": min(times_ms),
        "max": max(times_ms),
        "median": statistics.median(times_ms),
        "geometric_mean": geometric_mean(times_ms),
        "lukewarm": times_ms[0],
    }


def _hostname_set(scheduler_info: dict) -> set[str]:
    hosts = set()
    for worker in scheduler_info.get("workers", {}).values():
        host = worker.get("host")
        if host:
            hosts.add(host)
    return hosts


def create_input_array(args: argparse.Namespace) -> da.Array:
    chunk_rows = args.chunk_rows
    if chunk_rows <= 0:
        # Keep chunks balanced across workers without producing tiny partitions.
        workers = max(1, args.expected_workers or args.node_count * args.gpus_per_node)
        chunk_rows = max(100_000, args.n_samples // workers)
    rs = da.random.RandomState(args.seed)
    return rs.random(
        (args.n_samples, args.n_features),
        chunks=(chunk_rows, args.n_features),
    ).astype("float32")


def write_result_files(
    output_dir: Path,
    context: dict,
    records: list[IterationRecord],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_times = [record.runtime_ms for record in records]
    failures = {
        f"fit_{record.iteration}": record.error
        for record in records
        if record.status != "success" and record.error
    }
    success_times = [value for value in raw_times if value is not None]
    agg = aggregate_times(success_times)
    benchmark_json = {
        "context": context,
        "kmeans": {
            "raw_times_ms": {"fit": raw_times},
            "failed_queries": failures,
            "agg_times_ms": {
                key: {"fit": value} for key, value in agg.items()
            },
        },
    }

    (output_dir / "benchmark_result.json").write_text(json.dumps(benchmark_json, indent=2))

    with (output_dir / "benchmark_result.txt").open("w", encoding="utf-8") as f:
        f.write("cuML Dask KMeans Benchmark\n")
        f.write("==========================\n")
        f.write(f"workers: {context['worker_count']}\n")
        f.write(f"nodes: {context['node_count']}\n")
        f.write(f"samples: {context['n_samples']}\n")
        f.write(f"features: {context['n_features']}\n")
        f.write(f"clusters: {context['n_clusters']}\n")
        f.write("\n")
        for record in records:
            f.write(
                f"iter={record.iteration} status={record.status} "
                f"time_ms={record.runtime_ms} inertia={record.inertia}\n"
            )
        if agg:
            f.write("\nAggregates (ms)\n")
            for key, value in agg.items():
                f.write(f"- {key}: {value}\n")

    with (output_dir / "weak_scaling.csv").open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "timestamp",
                "iteration",
                "status",
                "runtime_ms",
                "inertia",
                "node_count",
                "worker_count",
                "gpus_per_node",
                "n_samples",
                "samples_per_gpu",
                "n_features",
                "n_clusters",
            ],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "timestamp": context["timestamp"],
                    "iteration": record.iteration,
                    "status": record.status,
                    "runtime_ms": record.runtime_ms,
                    "inertia": record.inertia,
                    "node_count": context["node_count"],
                    "worker_count": context["worker_count"],
                    "gpus_per_node": context["gpus_per_node"],
                    "n_samples": context["n_samples"],
                    "samples_per_gpu": context["samples_per_gpu"],
                    "n_features": context["n_features"],
                    "n_clusters": context["n_clusters"],
                }
            )


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    local_cluster = None

    if args.local_cluster and args.scheduler_address:
        raise ValueError("--local-cluster and --scheduler-address are mutually exclusive")

    if args.local_cluster:
        local_cluster = LocalCUDACluster()
        client = Client(local_cluster)
    else:
        if not args.scheduler_address:
            raise ValueError("Either --scheduler-address or --local-cluster is required")
        client = Client(args.scheduler_address)

    try:
        if args.expected_workers > 0:
            client.wait_for_workers(args.expected_workers, timeout=300)

        scheduler_info = client.scheduler_info()
        worker_count = len(scheduler_info.get("workers", {}))
        node_count = len(_hostname_set(scheduler_info)) or args.node_count
        samples_per_gpu = args.samples_per_gpu if args.samples_per_gpu > 0 else args.n_samples // max(1, worker_count)

        X = create_input_array(args).persist()
        client.rebalance()
        _ = X.shape  # Keep for clarity and to avoid lint warning.

        records: list[IterationRecord] = []
        for iteration in range(1, args.iterations + 1):
            model = DaskKMeans(
                n_clusters=args.n_clusters,
                max_iter=args.max_iter,
                random_state=args.seed + iteration,
                init="k-means||",
            )
            t0 = time.perf_counter()
            try:
                model.fit(X)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                inertia = float(model.inertia_)
                records.append(
                    IterationRecord(
                        iteration=iteration,
                        runtime_ms=elapsed_ms,
                        status="success",
                        inertia=inertia,
                        error=None,
                    )
                )
            except Exception as exc:
                records.append(
                    IterationRecord(
                        iteration=iteration,
                        runtime_ms=None,
                        status="error",
                        inertia=None,
                        error=str(exc),
                    )
                )

        context = {
            "benchmark": ["kmeans"],
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "engine": "cuml-dask-gpu",
            "kind": "single-node" if worker_count <= args.gpus_per_node else "multi-node",
            "iterations_count": args.iterations,
            "node_count": node_count,
            "worker_count": worker_count,
            "gpu_count": worker_count,
            "gpus_per_node": args.gpus_per_node,
            "n_samples": args.n_samples,
            "samples_per_gpu": samples_per_gpu,
            "n_features": args.n_features,
            "n_clusters": args.n_clusters,
            "seed": args.seed,
        }

        write_result_files(output_dir, context, records)
    finally:
        client.close()
        if local_cluster is not None:
            local_cluster.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
