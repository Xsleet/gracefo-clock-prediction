"""从导出的逐点 CSV 独立复算汇总误差，不调用 experiment.evaluate。"""
import argparse
from collections import defaultdict
import csv
import math
from pathlib import Path


def verify(directory):
    groups = defaultdict(list)
    with (directory / "predictions.csv").open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            error = float(row["error_ns"])
            reconstructed = (float(row["forecast_seconds"]) - float(row["actual_seconds"])) * 1e9
            if not math.isfinite(error) or not math.isclose(error, reconstructed, abs_tol=1e-5, rel_tol=0):
                raise ValueError("Serialized absolute clocks disagree with error_ns.")
            for minutes in [5, 15, 30, 60]:
                if int(row["lead_seconds"]) <= minutes * 60:
                    groups[(row["satellite"], row["method"], minutes)].append(error)
    checked = set()
    with (directory / "metrics_summary.csv").open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            key = (row["satellite"], row["method"], int(row["horizon_minutes"]))
            if key in checked:
                raise ValueError("Duplicate summary key.")
            checked.add(key)
            errors = groups[key]
            if len(errors) != int(row["points"]) or not errors:
                raise ValueError("Prediction count mismatch.")
            expected = {
                "rmse_ns": math.sqrt(math.fsum(e * e for e in errors) / len(errors)),
                "mae_ns": math.fsum(abs(e) for e in errors) / len(errors),
                "max_abs_error_ns": max(abs(e) for e in errors),
            }
            for name, value in expected.items():
                if not math.isclose(value, float(row[name]), rel_tol=1e-10, abs_tol=1e-10):
                    raise ValueError(f"Mismatch: {key}, {name}")
    if checked != set(groups) or not checked:
        raise ValueError("Missing summary rows.")
    print(f"Verified {len(checked)} metric rows from exported point predictions.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    verify(parser.parse_args().directory)
