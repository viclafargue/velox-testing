# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_PATH = Path(__file__).with_name("run_kmeans_benchmark.py")
SPEC = importlib.util.spec_from_file_location("cuml_kmeans_benchmark", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
benchmark = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = benchmark
SPEC.loader.exec_module(benchmark)


def _write_fbin(path: Path, rows: int, features: int, extended: bool = False) -> None:
    header = struct.pack("<QQ" if extended else "<II", rows, features)
    path.write_bytes(header + bytes(rows * features * benchmark.DTYPE_BYTES))


class FbinHeaderTest(unittest.TestCase):
    def test_reads_legacy_header(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "legacy.fbin"
            _write_fbin(path, rows=7, features=3)
            self.assertEqual(
                benchmark.read_fbin_header(path),
                benchmark.FbinHeader(rows=7, features=3, header_bytes=8),
            )

    def test_reads_extended_header(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "extended.fbin"
            _write_fbin(path, rows=7, features=3, extended=True)
            self.assertEqual(
                benchmark.read_fbin_header(path),
                benchmark.FbinHeader(rows=7, features=3, header_bytes=16),
            )

    def test_rejects_truncated_header(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "truncated.fbin"
            path.write_bytes(b"short")
            with self.assertRaisesRegex(ValueError, "too small"):
                benchmark.read_fbin_header(path)

    def test_rejects_size_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "mismatch.fbin"
            path.write_bytes(struct.pack("<II", 7, 3) + b"bad")
            with self.assertRaisesRegex(ValueError, "file size does not match"):
                benchmark.read_fbin_header(path)

    def test_resolve_dataset_derives_and_validates_features(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "data.fbin"
            _write_fbin(path, rows=20, features=4)
            parser = benchmark.build_arg_parser()
            args = parser.parse_args(
                [
                    "--output-dir",
                    tmpdir,
                    "--n-samples",
                    "10",
                    "--n-clusters",
                    "2",
                    "--input-fbin",
                    str(path),
                ]
            )
            benchmark.resolve_dataset(args)
            self.assertEqual(args.n_features, 4)

            args = parser.parse_args(
                [
                    "--output-dir",
                    tmpdir,
                    "--n-samples",
                    "10",
                    "--n-clusters",
                    "2",
                    "--n-features",
                    "5",
                    "--input-fbin",
                    str(path),
                ]
            )
            with self.assertRaisesRegex(ValueError, "does not match"):
                benchmark.resolve_dataset(args)

    def test_resolve_dataset_rejects_too_many_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "data.fbin"
            _write_fbin(path, rows=20, features=4)
            args = benchmark.build_arg_parser().parse_args(
                [
                    "--output-dir",
                    tmpdir,
                    "--n-samples",
                    "21",
                    "--n-clusters",
                    "2",
                    "--input-fbin",
                    str(path),
                ]
            )
            with self.assertRaisesRegex(ValueError, "contains 20"):
                benchmark.resolve_dataset(args)


class PartitionTest(unittest.TestCase):
    def test_partition_specs_cover_prefix_without_overlap(self) -> None:
        specs = benchmark.partition_specs(10, ["w0", "w1", "w2"])
        self.assertEqual([spec.rows for spec in specs], [4, 3, 3])
        self.assertEqual([spec.start_row for spec in specs], [0, 4, 7])
        self.assertEqual(sum(spec.rows for spec in specs), 10)

    def test_partition_specs_require_one_row_per_worker(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be >= worker_count"):
            benchmark.partition_specs(1, ["w0", "w1"])


try:
    import numpy as np
except ImportError:  # pragma: no cover - depends on the local test environment
    np = None

try:
    import dask.array as da
except ImportError:  # pragma: no cover - depends on the local test environment
    da = None


@unittest.skipIf(np is None, "NumPy is not installed")
class NumpyDatasetTest(unittest.TestCase):
    def test_memmap_partitions_reconstruct_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "data.fbin"
            values = np.arange(30, dtype=np.float32).reshape(10, 3)
            path.write_bytes(struct.pack("<II", 10, 3) + values.tobytes())
            first = benchmark.load_fbin_partition(str(path), 8, 0, 4, 3)
            second = benchmark.load_fbin_partition(str(path), 8, 4, 6, 3)
            np.testing.assert_array_equal(np.concatenate([first, second]), values)
            self.assertEqual(first.shape, (4, 3))
            self.assertEqual(second.shape, (6, 3))
            self.assertEqual(first.dtype, np.dtype("float32"))
            self.assertEqual(first.mode, "r")

    def test_generated_data_is_independent_of_partitioning(self) -> None:
        centers = benchmark._make_centers(3, 4, seed=12)
        whole = benchmark.generate_blob_partition(0, 17, centers, 0.01, seed=12)
        split = np.concatenate(
            [
                benchmark.generate_blob_partition(0, 5, centers, 0.01, seed=12),
                benchmark.generate_blob_partition(5, 12, centers, 0.01, seed=12),
            ]
        )
        np.testing.assert_array_equal(split, whole)
        self.assertEqual(whole.dtype, np.float32)
        self.assertTrue(whole.flags.c_contiguous)

    def test_generated_data_uses_requested_standard_deviation(self) -> None:
        centers = np.zeros((1, 2), dtype=np.float32)
        values = benchmark.generate_blob_partition(0, 20_000, centers, 0.01, seed=12)
        self.assertAlmostEqual(float(values.mean()), 0.0, delta=0.0002)
        self.assertAlmostEqual(float(values.std()), 0.01, delta=0.0002)


@unittest.skipIf(da is None, "Dask is not installed")
class DaskInputGraphTest(unittest.TestCase):
    def test_worker_pinned_blocks_are_concrete_tasks(self) -> None:
        tasks = [
            (benchmark._path_preflight, "first"),
            (benchmark._path_preflight, "second"),
        ]
        specs = benchmark.partition_specs(10, ["worker-0", "worker-1"])
        array = benchmark._worker_pinned_array(
            tasks=tasks,
            specs=specs,
            n_features=3,
        )

        self.assertEqual(array.chunks, ((5, 5), (3,)))
        layer = array.dask.layers[array.name]
        for key, task in layer.items():
            self.assertIs(task[0], benchmark._path_preflight)
            expected_worker = specs[key[1]].worker
            self.assertEqual(layer.annotations["workers"](key), expected_worker)
        self.assertFalse(layer.annotations["allow_other_workers"])


class ResultTest(unittest.TestCase):
    def test_writes_compatible_result_and_fit_csv(self) -> None:
        context = {
            "node_count": 1,
            "worker_count": 4,
            "gpu_count": 4,
            "n_samples": 100,
            "samples_per_gpu": 25,
            "n_features": 8,
            "n_clusters": 2,
            "input_gib": 100 * 8 * 4 / benchmark.GIB,
            "device_buffer_samples": 10,
            "data_source": "generated",
        }
        records = [
            benchmark.FitRecord("warmup", 1, 20.0, "success", 5.0, 3, None),
            benchmark.FitRecord("fit", 1, 10.0, "success", 4.0, 2, None),
            benchmark.FitRecord("fit", 2, 12.0, "success", 4.0, 2, None),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            benchmark.write_result_files(
                output_dir=output_dir,
                context=context,
                records=records,
                data_setup_ms=7.0,
            )
            result = json.loads((output_dir / "benchmark_result.json").read_text())
            self.assertEqual(result["kmeans"]["raw_times_ms"]["fit"], [10.0, 12.0])
            self.assertEqual(result["kmeans"]["warmup_times_ms"], [20.0])
            self.assertEqual(result["kmeans"]["agg_times_ms"]["median"]["fit"], 11.0)
            self.assertEqual(result["kmeans"]["agg_times_ms"]["lukewarm"]["fit"], 10.0)
            self.assertTrue((output_dir / "fit_results.csv").is_file())


if __name__ == "__main__":
    unittest.main()
