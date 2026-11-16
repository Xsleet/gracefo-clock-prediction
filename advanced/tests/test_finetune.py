"""Without downloading Torch, verify the chronological fine-tuning data contract."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from finetune import EPOCH, HISTORY, HORIZON, make_windows


class FinetuneDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = ROOT / "data/finetune.csv"
        cls.data = np.loadtxt(cls.source, delimiter=",", skiprows=1)
        cls.train = make_windows(cls.data, "2025-05-27", "2025-05-31", 3600)
        cls.val = make_windows(cls.data, "2025-05-31", "2025-06-01", 7200)

    def test_training_data_provenance(self):
        manifest = json.loads(self.source.with_suffix(".manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(hashlib.sha256(self.source.read_bytes()).hexdigest(), manifest["output_sha256"])
        self.assertEqual(self.data.shape, (43164, 2))
        self.assertTrue(np.all(np.diff(self.data[:, 0]) > 0))
        self.assertEqual(len(manifest["source_files"]), 5)

    def test_train_validation_and_demonstration_epochs_are_disjoint(self):
        self.assertEqual(len(self.train[0]), 63)
        self.assertEqual(len(self.val[0]), 9)
        train_end = self.train[2].max() + HORIZON * 10
        val_start = self.val[2].min() - (HISTORY - 1) * 10
        val_end = self.val[2].max() + HORIZON * 10
        demonstration = np.loadtxt(ROOT / "data/clock.csv", delimiter=",", skiprows=1)
        self.assertLess(train_end, val_start)
        self.assertLess(val_end, demonstration[0, 0])
        for _, _, origins in (self.train, self.val):
            first = origins - (HISTORY - 1) * 10
            last = origins + HORIZON * 10
            np.testing.assert_array_equal((first + 43200) // 86400, (last + 43200) // 86400)

    def test_labels_are_centered_on_last_history_value(self):
        x, y, origins = self.train
        origin = origins[0]
        i = np.searchsorted(self.data[:, 0], origin)
        expected_y = (self.data[i + 1:i + 1 + HORIZON, 1] - self.data[i, 1]) * 1e9
        np.testing.assert_array_equal(y[0], expected_y)
        self.assertEqual(x[0, -1], 0)
        self.assertEqual(self.data[i + 1, 0], origin + 10)

    def test_gaps_and_out_of_split_observations_are_excluded(self):
        missing = self.data[self.data[:, 0] != self.train[2][0]]
        _, _, origins = make_windows(missing, "2025-05-27", "2025-05-31", 3600)
        self.assertNotIn(self.train[2][0], origins)
        changed = self.data.copy()
        cutoff = (datetime(2025, 5, 31) - EPOCH).total_seconds()
        changed[changed[:, 0] >= cutoff, 1] += 1e-6
        recomputed = make_windows(changed, "2025-05-27", "2025-05-31", 3600)
        for original, new in zip(self.train, recomputed):
            np.testing.assert_array_equal(original, new)


if __name__ == "__main__":
    unittest.main()
