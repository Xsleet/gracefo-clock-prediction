"""只需历史数据的预报入口；输出晚于最后一个输入历元的 360 个预测。"""
import argparse
from pathlib import Path

import numpy as np

from experiment import CONTEXT, HORIZON, ROOT, STEP, label, load_clock, prepare_history, write_rows


def forecast_history(data):
    """Validate the last six hours and fit baselines without any holdout truth."""
    if len(data["t"]) < CONTEXT:
        raise ValueError(f"Need at least {CONTEXT} rows of history.")
    t = data["t"][-CONTEXT:]
    if not np.all(np.diff(t) == STEP):
        raise ValueError("History contains a gap; no interpolation is performed.")
    if np.any(data["flags"][-CONTEXT:] != 0):
        raise ValueError("History contains a nonzero quality flag under the conservative pilot policy.")
    if len(np.unique(data["clock_ids"][-CONTEXT:])) != 1:
        raise ValueError("Clock ID changes within the history.")
    result = prepare_history(data["y"][-CONTEXT:])
    result["quantiles"] = {}
    return int(t[-1]), result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("history", type=Path)
    parser.add_argument("--satellite", choices=["C", "D"], required=True)
    parser.add_argument("--model", choices=["baseline", "timesfm"], default="baseline")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "models/timesfm-3.0-pytorch")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/future.csv")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; choose a new --output path.")
    data, _ = load_clock(args.history, args.satellite)
    origin, prepared = forecast_history(data)
    if args.model == "timesfm":
        from timesfm_backend import add_timesfm_predictions
        add_timesfm_predictions([prepared], args.checkpoint, args.device)
    rows = []
    for method, values in prepared["predictions"].items():
        for i in range(HORIZON):
            rows.append({
                "satellite": args.satellite, "origin_receiver_time": label(origin),
                "target_receiver_time": label(origin + (i + 1) * STEP),
                "lead_seconds": (i + 1) * STEP, "method": method,
                "forecast_seconds": format(prepared["reference_seconds"] + values[i] * 1e-9, ".17g"),
            })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_rows(args.output, rows)
    print(f"Saved future predictions to {args.output}; no future observations were read.")


if __name__ == "__main__":
    main()
