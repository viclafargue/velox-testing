# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "slurm" / "cuml-dask-nvl72" / "run_scaling_experiment.py"
SPEC = importlib.util.spec_from_file_location("cuml_kmeans_scaling", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
scaling = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(scaling)


class ScalingExperimentTest(unittest.TestCase):
    def test_parser_accepts_weak_scaling_interface(self) -> None:
        args = scaling.build_arg_parser().parse_args(
            [
                "weak",
                "--nodes",
                "1 2",
                "--worker-image",
                "cuml-bench",
                "--output-dir",
                "results",
                "--samples-per-gpu",
                "100",
            ]
        )
        scaling.validate_args(args)
        self.assertEqual(args.nodes, [1, 2])

    def test_parse_node_counts(self) -> None:
        self.assertEqual(scaling.parse_node_counts("4 1 2"), [1, 2, 4])
        self.assertEqual(scaling.parse_node_counts("1,2,4"), [1, 2, 4])
        with self.assertRaisesRegex(Exception, "duplicates"):
            scaling.parse_node_counts("1 2 2")

    def test_weak_samples_scale_with_gpu_count(self) -> None:
        self.assertEqual(
            scaling.samples_for_point(
                mode="weak",
                nodes=2,
                gpus_per_node=4,
                samples_per_gpu=100,
                n_samples=None,
            ),
            800,
        )

    def test_strong_samples_stay_constant(self) -> None:
        self.assertEqual(
            scaling.samples_for_point(
                mode="strong",
                nodes=4,
                gpus_per_node=4,
                samples_per_gpu=None,
                n_samples=1_000,
            ),
            1_000,
        )

    def test_scaling_metrics_use_smallest_gpu_count_as_baseline(self) -> None:
        runs = [
            {"gpu_count": 8, "median_ms": 60.0, "rows_per_s": 180.0},
            {"gpu_count": 4, "median_ms": 100.0, "rows_per_s": 100.0},
        ]
        scaling.add_scaling_metrics(runs)
        baseline = runs[1]
        larger = runs[0]
        self.assertEqual(baseline["resource_multiplier"], 1.0)
        self.assertEqual(baseline["scaling_efficiency"], 1.0)
        self.assertEqual(larger["resource_multiplier"], 2.0)
        self.assertAlmostEqual(larger["throughput_speedup_vs_baseline"], 1.8)
        self.assertAlmostEqual(larger["scaling_efficiency"], 0.9)

    def test_output_directory_requires_explicit_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "results"
            output_dir.mkdir()
            marker = output_dir / "marker"
            marker.write_text("old")
            with self.assertRaisesRegex(ValueError, "already exists"):
                scaling.prepare_output_dir(output_dir, overwrite=False)
            resolved = scaling.prepare_output_dir(output_dir, overwrite=True)
            self.assertEqual(resolved, output_dir.resolve())
            self.assertTrue(output_dir.is_dir())
            self.assertFalse(marker.exists())

    def test_output_directory_rejects_working_directory(self) -> None:
        with self.assertRaisesRegex(ValueError, "protected output directory"):
            scaling.prepare_output_dir(Path.cwd(), overwrite=True)

    def test_writes_json_and_csv_summaries(self) -> None:
        args = scaling.build_arg_parser().parse_args(
            [
                "strong",
                "--nodes",
                "1",
                "--worker-image",
                "cuml-bench",
                "--output-dir",
                "unused",
                "--n-samples",
                "100",
            ]
        )
        run = {
            "node_count": 1,
            "gpu_count": 4,
            "worker_count": 4,
            "n_samples": 100,
            "samples_per_gpu": 25,
            "n_features": 8,
            "n_clusters": 2,
            "input_gib": 0.1,
            "data_source": "generated",
            "dataset_name": "generated-blobs",
            "device_buffer_samples": 10,
            "data_setup_ms": 1.0,
            "median_ms": 10.0,
            "mean_ms": 10.0,
            "min_ms": 9.0,
            "max_ms": 11.0,
            "median_n_iter": 2.0,
            "rows_per_s": 10_000.0,
            "row_iters_per_s": 20_000.0,
            "gib_iters_per_s": 20.0,
            "resource_multiplier": 1.0,
            "time_ratio_vs_baseline": 1.0,
            "throughput_speedup_vs_baseline": 1.0,
            "scaling_efficiency": 1.0,
            "result_dir": "point",
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            scaling._write_summary(output_dir, "strong", args, [run])
            summary = json.loads((output_dir / "scaling_summary.json").read_text())
            self.assertEqual(summary["runs"][0]["n_samples"], 100)
            self.assertTrue((output_dir / "scaling_summary.csv").is_file())


if __name__ == "__main__":
    unittest.main()
