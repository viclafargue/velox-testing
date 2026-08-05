#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run and summarize a weak- or strong-scaling KMeans experiment."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

DEFAULT_DEVICE_BUFFER_SAMPLES = 4_000_000
DEFAULT_CLUSTERS = 100
DEFAULT_WARMUP_RUNS = 1
DEFAULT_REPEATS = 3
DEFAULT_MAX_ITER = 20
DEFAULT_TOL = 1e-4
DEFAULT_SEED = 12345
DEFAULT_CLUSTER_STD = 0.01


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


def parse_node_counts(value: str) -> list[int]:
    try:
        nodes = [int(item) for item in value.replace(",", " ").split()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--nodes must contain positive integers") from exc
    if not nodes or any(node <= 0 for node in nodes):
        raise argparse.ArgumentTypeError("--nodes must contain positive integers")
    if len(set(nodes)) != len(nodes):
        raise argparse.ArgumentTypeError("--nodes must not contain duplicates")
    return sorted(nodes)


def samples_for_point(
    *,
    mode: str,
    nodes: int,
    gpus_per_node: int,
    samples_per_gpu: int | None,
    n_samples: int | None,
) -> int:
    if mode == "weak":
        assert samples_per_gpu is not None
        return nodes * gpus_per_node * samples_per_gpu
    assert n_samples is not None
    return n_samples


def add_scaling_metrics(runs: list[dict[str, Any]]) -> None:
    if not runs:
        return
    baseline = min(runs, key=lambda run: int(run["gpu_count"]))
    baseline_gpus = int(baseline["gpu_count"])
    baseline_ms = float(baseline["median_ms"])
    baseline_rows_per_s = float(baseline["rows_per_s"])

    for run in runs:
        resource_multiplier = int(run["gpu_count"]) / baseline_gpus
        throughput_speedup = float(run["rows_per_s"]) / baseline_rows_per_s
        run["resource_multiplier"] = resource_multiplier
        run["time_ratio_vs_baseline"] = baseline_ms / float(run["median_ms"])
        run["throughput_speedup_vs_baseline"] = throughput_speedup
        run["scaling_efficiency"] = throughput_speedup / resource_multiplier


def _read_point_result(point_dir: Path) -> dict[str, Any]:
    result_path = point_dir / "benchmark_result.json"
    if not result_path.is_file():
        raise RuntimeError(f"benchmark did not produce {result_path}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    context = result["context"]
    kmeans = result["kmeans"]
    failures = kmeans.get("failed_queries", {})
    if failures:
        raise RuntimeError(f"benchmark reported failed fits: {failures}")
    return {
        "node_count": int(context["node_count"]),
        "gpu_count": int(context["gpu_count"]),
        "worker_count": int(context["worker_count"]),
        "n_samples": int(context["n_samples"]),
        "samples_per_gpu": int(context["samples_per_gpu"]),
        "n_features": int(context["n_features"]),
        "n_clusters": int(context["n_clusters"]),
        "input_gib": float(context["input_gib"]),
        "data_source": context["data_source"],
        "dataset_name": context["dataset_name"],
        "device_buffer_samples": int(context["device_buffer_samples"]),
        "data_setup_ms": kmeans.get("data_setup_ms"),
        "median_ms": float(kmeans["agg_times_ms"]["median"]["fit"]),
        "mean_ms": float(kmeans["agg_times_ms"]["avg"]["fit"]),
        "min_ms": float(kmeans["agg_times_ms"]["min"]["fit"]),
        "max_ms": float(kmeans["agg_times_ms"]["max"]["fit"]),
        "median_n_iter": float(statistics.median(kmeans["raw_n_iters"])),
        **{key: float(value) for key, value in kmeans["throughput"].items()},
        "result_dir": str(point_dir),
    }


def _write_summary(output_dir: Path, mode: str, args: argparse.Namespace, runs: list[dict[str, Any]]) -> None:
    summary = {
        "scaling": mode,
        "nodes": args.nodes,
        "gpus_per_node": args.num_gpus_per_node,
        "samples_per_gpu": args.samples_per_gpu,
        "n_samples": args.n_samples,
        "input_fbin": None if args.input_fbin is None else str(args.input_fbin),
        "n_features": runs[0]["n_features"] if runs else args.n_features,
        "n_clusters": args.n_clusters,
        "device_buffer_samples": args.device_buffer_samples,
        "warmup_runs": args.warmup_runs,
        "repeats": args.repeats,
        "runs": runs,
    }
    (output_dir / "scaling_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    fieldnames = [
        "node_count",
        "gpu_count",
        "worker_count",
        "n_samples",
        "samples_per_gpu",
        "n_features",
        "n_clusters",
        "input_gib",
        "data_source",
        "dataset_name",
        "device_buffer_samples",
        "data_setup_ms",
        "median_ms",
        "mean_ms",
        "min_ms",
        "max_ms",
        "median_n_iter",
        "rows_per_s",
        "row_iters_per_s",
        "gib_iters_per_s",
        "resource_multiplier",
        "time_ratio_vs_baseline",
        "throughput_speedup_vs_baseline",
        "scaling_efficiency",
        "result_dir",
    ]
    with (output_dir / "scaling_summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(runs)


def _print_run(run: dict[str, Any], mode: str) -> None:
    print(
        f"{mode:>6} | nodes={run['node_count']} | gpus={run['gpu_count']} | "
        f"rows={run['n_samples']:,} | median={run['median_ms'] / 1000.0:.3f}s | "
        f"iters={run['median_n_iter']:.0f} | rows/s={run['rows_per_s']:,.0f} | "
        f"speedup={run['throughput_speedup_vs_baseline']:.2f}x | "
        f"efficiency={100.0 * run['scaling_efficiency']:.1f}%"
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Dask out-of-core KMeans scaling experiments.")
    parser.add_argument("scaling", choices=("weak", "strong"))
    parser.add_argument("--nodes", type=parse_node_counts, default=[1, 2, 4])
    parser.add_argument("--num-gpus-per-node", type=_positive_int, default=4)
    parser.add_argument("--worker-image", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples-per-gpu", type=_positive_int)
    parser.add_argument("--n-samples", type=_positive_int)
    parser.add_argument("--input-fbin", type=Path)
    parser.add_argument("--n-features", type=_positive_int)
    parser.add_argument("--n-clusters", type=_positive_int, default=DEFAULT_CLUSTERS)
    parser.add_argument("--device-buffer-samples", type=_positive_int, default=DEFAULT_DEVICE_BUFFER_SAMPLES)
    parser.add_argument("--warmup-runs", type=_nonnegative_int, default=DEFAULT_WARMUP_RUNS)
    parser.add_argument("--repeats", type=_positive_int, default=DEFAULT_REPEATS)
    parser.add_argument("--max-iter", type=_positive_int, default=DEFAULT_MAX_ITER)
    parser.add_argument("--tol", type=float, default=DEFAULT_TOL)
    parser.add_argument("--seed", type=_nonnegative_int, default=DEFAULT_SEED)
    parser.add_argument("--cluster-std", type=float, default=DEFAULT_CLUSTER_STD)
    parser.add_argument("--inter-run-sleep", type=_nonnegative_int, default=60)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.scaling == "weak":
        if args.samples_per_gpu is None:
            raise ValueError("weak scaling requires --samples-per-gpu")
        if args.n_samples is not None:
            raise ValueError("--n-samples is only valid for strong scaling")
    else:
        if args.n_samples is None:
            raise ValueError("strong scaling requires --n-samples")
        if args.samples_per_gpu is not None:
            raise ValueError("--samples-per-gpu is only valid for weak scaling")
    if args.tol <= 0:
        raise ValueError("--tol must be positive")
    if args.cluster_std <= 0:
        raise ValueError("--cluster-std must be positive")


def prepare_output_dir(output_dir: Path, overwrite: bool) -> Path:
    resolved = output_dir.resolve()
    cwd = Path.cwd().resolve()
    script = Path(__file__).resolve()
    if resolved in {Path("/"), Path.home().resolve(), cwd} or resolved in cwd.parents or resolved in script.parents:
        raise ValueError(f"refusing to use protected output directory: {resolved}")
    if resolved.exists():
        if not overwrite:
            raise ValueError(f"output directory already exists; pass --overwrite: {resolved}")
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True)
    return resolved


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    validate_args(args)
    script_dir = Path(__file__).resolve().parent
    launcher = script_dir / "launch-run.sh"

    if args.input_fbin is not None:
        args.input_fbin = args.input_fbin.resolve(strict=True)
        if not args.input_fbin.is_file() or args.input_fbin.suffix != ".fbin":
            raise ValueError(f"--input-fbin must point to a .fbin file: {args.input_fbin}")
    args.output_dir = prepare_output_dir(args.output_dir, args.overwrite)

    runs: list[dict[str, Any]] = []
    for index, nodes in enumerate(args.nodes, start=1):
        gpu_count = nodes * args.num_gpus_per_node
        n_samples = samples_for_point(
            mode=args.scaling,
            nodes=nodes,
            gpus_per_node=args.num_gpus_per_node,
            samples_per_gpu=args.samples_per_gpu,
            n_samples=args.n_samples,
        )
        point_dir = args.output_dir / f"{args.scaling}_n{nodes}_g{gpu_count}_s{n_samples}"
        command = [
            str(launcher),
            "--nodes",
            str(nodes),
            "--num-gpus-per-node",
            str(args.num_gpus_per_node),
            "--worker-image",
            args.worker_image,
            "--output-path",
            str(point_dir),
            "--n-samples",
            str(n_samples),
            "--n-clusters",
            str(args.n_clusters),
            "--device-buffer-samples",
            str(args.device_buffer_samples),
            "--warmup-runs",
            str(args.warmup_runs),
            "--repeats",
            str(args.repeats),
            "--max-iter",
            str(args.max_iter),
            "--tol",
            str(args.tol),
            "--seed",
            str(args.seed),
            "--cluster-std",
            str(args.cluster_std),
        ]
        if args.scaling == "weak":
            command.extend(["--samples-per-gpu", str(args.samples_per_gpu)])
        if args.input_fbin is not None:
            command.extend(["--input-fbin", str(args.input_fbin)])
        if args.n_features is not None:
            command.extend(["--n-features", str(args.n_features)])

        print(
            f"Run {index}/{len(args.nodes)}: mode={args.scaling} nodes={nodes} gpus={gpu_count} samples={n_samples}",
            flush=True,
        )
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as exc:
            raise SystemExit(f"benchmark point failed with exit code {exc.returncode}: nodes={nodes}") from exc
        runs.append(_read_point_result(point_dir))

        if index < len(args.nodes) and args.inter_run_sleep:
            time.sleep(args.inter_run_sleep)

    add_scaling_metrics(runs)
    _write_summary(args.output_dir, args.scaling, args, runs)
    for run in runs:
        _print_run(run, args.scaling)
    print(f"Results: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
