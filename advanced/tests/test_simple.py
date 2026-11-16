"""Keep the short teaching path aligned with its real source window."""
import csv
import os
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


class SimpleExampleTests(unittest.TestCase):
    def test_teaching_window_matches_source(self):
        sample = np.loadtxt(ROOT / "data/clock.csv", delimiter=",", skiprows=1)
        rows = [line.split() for line in (ROOT / "data/sample/CLK1B_2025-06-01_to_2025-06-03_C.txt").read_text().splitlines()]
        source = np.array([[float(row[2]), float(row[5])] for row in rows
                           if sample[0, 0] <= int(row[2]) <= sample[-1, 0]])
        self.assertEqual(sample.shape, (2520, 2))
        np.testing.assert_array_equal(sample, source)
        self.assertTrue(np.all(np.diff(sample[:, 0]) == 10))

    def test_short_entrypoint_and_csv_metrics(self):
        # Emulate an English Windows console; Chinese progress must remain printable.
        env = dict(os.environ, PYTHONIOENCODING="cp1252")
        run = subprocess.run([sys.executable, str(ROOT / "main.py"), "--model", "linear"],
                             check=True, capture_output=True, env=env)
        self.assertIn("历史", run.stdout.decode("utf-8"))
        result = np.loadtxt(ROOT / "outputs/linear-cpu/prediction.csv", delimiter=",", skiprows=1)
        sample = np.loadtxt(ROOT / "data/clock.csv", delimiter=",", skiprows=1)
        self.assertEqual(result.shape, (360, 4))
        self.assertEqual(result[0, 0], sample[2159, 0] + 10)
        np.testing.assert_array_equal(result[:, :2], sample[2160:])
        error = (result[:, 2] - result[:, 1]) * 1e9
        np.testing.assert_allclose(result[:, 3], error, atol=1e-5)
        with (ROOT / "advanced/reference/metrics_by_window.csv").open(encoding="utf-8-sig", newline="") as stream:
            reference = next(row for row in csv.DictReader(stream)
                             if row["satellite"] == "C" and row["origin_receiver_time"] == "2025-06-02 00:00:00"
                             and row["method"] == "anchored_linear_10m" and row["horizon_minutes"] == "60")
        self.assertAlmostEqual(float(np.sqrt(np.mean(result[:, 3]**2))), float(reference["rmse_ns"]), delta=1e-6)


if __name__ == "__main__":
    unittest.main()
