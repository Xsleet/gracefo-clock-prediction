"""Keep the short teaching path aligned with its real source window."""
import csv
from contextlib import redirect_stdout
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import main as simple


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
        self.assertEqual(result.shape, (360, 8))
        self.assertEqual(result[0, 0], sample[2159, 0] + 10)
        np.testing.assert_array_equal(result[:, :2], sample[2160:])
        for prediction_column in (2, 4, 6):
            error = (result[:, prediction_column] - result[:, 1]) * 1e9
            np.testing.assert_allclose(result[:, prediction_column + 1], error, atol=1e-5)
        with (ROOT / "advanced/reference/metrics_by_window.csv").open(encoding="utf-8-sig", newline="") as stream:
            reference = next(row for row in csv.DictReader(stream)
                             if row["satellite"] == "C" and row["origin_receiver_time"] == "2025-06-02 00:00:00"
                             and row["method"] == "anchored_linear_10m" and row["horizon_minutes"] == "60")
        self.assertAlmostEqual(float(np.sqrt(np.mean(result[:, 3]**2))), float(reference["rmse_ns"]), delta=1e-6)

    def test_quadratic_forecast_matches_analytical_future(self):
        past = np.arange(1 - 2160, 1) / 360
        future = np.arange(1, 361) / 360
        polynomial, periodic, _ = simple.polynomial_forecasts(3 + 2 * past + 0.4 * past**2, 360)
        expected = 3 + 2 * future + 0.4 * future**2
        np.testing.assert_allclose(polynomial, expected, atol=1e-9)
        np.testing.assert_allclose(periodic, expected, atol=1e-9)

    def test_recovers_known_period_and_forecasts_unseen_cycle(self):
        def signal(hours):
            phase = 2 * np.pi * hours / (97 / 60)
            return 3 + 2 * hours + 0.4 * hours**2 + 2 * np.sin(phase) - 0.7 * np.cos(phase)
        past = np.arange(1 - 2160, 1) / 360
        future = np.arange(1, 361) / 360
        _, periodic, period = simple.polynomial_forecasts(signal(past), 360)
        self.assertEqual(period, 97)
        np.testing.assert_allclose(periodic, signal(future), atol=1e-9)

    def test_future_observations_do_not_change_any_baseline_prediction(self):
        sample = np.loadtxt(ROOT / "data/clock.csv", delimiter=",", skiprows=1)
        changed = sample.copy()
        changed[2160:, 1] += 30e-9
        results = []
        with tempfile.TemporaryDirectory() as directory:
            for data in (sample, changed):
                with patch.object(simple, "ROOT", Path(directory)), \
                        patch.object(simple.np, "loadtxt", return_value=data), \
                        patch.object(sys, "argv", ["main.py", "--model", "linear"]), \
                        redirect_stdout(io.StringIO()):
                    simple.main()
                results.append(np.loadtxt(Path(directory) / "outputs/linear-cpu/prediction.csv",
                                          delimiter=",", skiprows=1))
        np.testing.assert_array_equal(results[0][:, [2, 4, 6]], results[1][:, [2, 4, 6]])
        np.testing.assert_allclose(results[1][:, [3, 5, 7]] - results[0][:, [3, 5, 7]],
                                   -30, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
