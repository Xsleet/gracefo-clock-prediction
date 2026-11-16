"""Tests protect scientific contracts: units, forecast origin, gaps and data provenance."""
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiment import CONTEXT, HORIZON, ROOT, STEP, evaluate, load_clock, make_windows, numerical_check, prepare_history
from forecast import forecast_history
from prepare_data import merge_native


class ForecastTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = {}
        cls.windows = []
        cls.excluded = []
        for sat in "CD":
            path = ROOT / "data/sample" / f"CLK1B_2025-06-01_to_2025-06-03_{sat}.txt"
            data, _ = load_clock(path, sat)
            cls.data[sat] = data
            windows, excluded = make_windows(data, sat)
            cls.windows.extend(windows)
            cls.excluded.extend(excluded)

    def test_units_and_first_forecast_step(self):
        numerical_check()  # analytical linear clock: first target is +10 s, not 0

    def test_quadratic_extrapolation(self):
        x = np.arange(-CONTEXT + 1, 1) * STEP / 3600
        y = .01 + (5 * x * x + 2 * x) * 1e-9
        predicted = prepare_history(y)["predictions"]["quadratic_6h"]
        future = np.arange(1, HORIZON + 1) * STEP / 3600
        np.testing.assert_allclose(predicted, 5 * future**2 + 2 * future, atol=1e-6)

    def test_future_truth_does_not_change_prediction(self):
        original = self.data["C"]
        changed = {key: value.copy() for key, value in original.items()}
        origin = self.windows[0]["origin"]
        changed["y"][changed["t"] > origin] += 1e-5
        windows, _ = make_windows(changed, "C")
        for method, values in self.windows[0]["predictions"].items():
            np.testing.assert_array_equal(values, windows[0]["predictions"][method])
        self.assertFalse(np.array_equal(windows[0]["truth_seconds"], self.windows[0]["truth_seconds"]))

    def test_pilot_population_and_gap_exclusion(self):
        self.assertEqual(sum(w["satellite"] == "C" for w in self.windows), 8)
        self.assertEqual(sum(w["satellite"] == "D" for w in self.windows), 7)
        self.assertEqual(len(self.excluded), 1)
        self.assertEqual(self.excluded[0]["origin_receiver_time"], "2025-06-03 00:00:00")

    def test_baselines_match_archived_experiment(self):
        _, summary = evaluate(self.windows)
        with (ROOT / "reference/metrics_summary.csv").open(encoding="utf-8-sig", newline="") as stream:
            reference = {(r["satellite"], r["method"], int(r["horizon_minutes"])): float(r["rmse_ns"])
                         for r in csv.DictReader(stream)}
        for row in summary:
            key = (row["satellite"], row["method"], row["horizon_minutes"])
            self.assertAlmostEqual(row["rmse_ns"], reference[key], delta=1e-6)

    def test_history_only_prediction(self):
        # The function receives no target observations and can forecast from exactly 2160 rows.
        history = {key: value[:CONTEXT].copy() for key, value in self.data["C"].items()}
        origin, result = forecast_history(history)
        self.assertEqual(origin, int(history["t"][-1]))
        self.assertNotIn("truth_seconds", result)
        self.assertTrue(all(len(p) == HORIZON for p in result["predictions"].values()))

    def test_forecast_rejects_gap_flags_and_clock_switch(self):
        for field, value in [("t", 1), ("flags", 128), ("clock_ids", 100)]:
            with self.subTest(field=field):
                history = {key: v[:CONTEXT].copy() for key, v in self.data["C"].items()}
                history[field][10] += value
                with self.assertRaises(ValueError):
                    forecast_history(history)

    def test_sample_hashes(self):
        for manifest_path in (ROOT / "data/sample").glob("*.manifest.json"):
            manifest = json.loads(manifest_path.read_text())
            path = manifest_path.parent / manifest["output_file"]
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), manifest["output_sha256"])

    def test_merge_overlap_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "CLK1B_2025-06-01_C_04.txt"
            second = Path(tmp) / "CLK1B_2025-06-02_C_04.txt"
            first.write_text("# End of YAML header\n100 C -1 .01 0 0 0 00000000\n110 C -1 .02 0 0 0 00000000\n")
            second.write_text("# End of YAML header\n110 C -1 .03 0 0 0 00000000\n120 C -1 .04 0 0 0 10000000\n")
            text, manifest = merge_native([second, first], "C")
            self.assertEqual(len(text.splitlines()), 2)
            self.assertEqual(text.splitlines()[1].split()[5], ".03")
            self.assertEqual(manifest["counts"]["nonzero_quality_records_removed"], 1)

    def test_reader_rejects_duplicate_epochs(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "duplicate.txt"
            row = "2000-01-01 12:01:40 100 C -1 .01 0 0 0 00000000\n"
            path.write_text(row + row)
            with self.assertRaises(ValueError):
                load_clock(path, "C")


if __name__ == "__main__":
    unittest.main()
