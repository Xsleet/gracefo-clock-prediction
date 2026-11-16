"""GRACE-FO 教学实验：历史观测 -> 预报 -> 与留后真值比较。

默认仅运行 CPU 多项式基线；--model timesfm 增加预训练模型。
所有拟合只使用历史，不插值缺口，不微调，不读取未来作为预测输入。
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timedelta
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
ROOT = Path(__file__).resolve().parent
EPOCH = datetime(2000, 1, 1, 12)
STEP = 10
CONTEXT = 2160  # 6 hours at 10 seconds
HORIZON = 360  # 60 minutes at 10 seconds
HORIZONS = [5, 15, 30, 60]
METHODS = {
    "linear_1h": "Linear (1 h fit)",
    "linear_6h": "Linear (6 h fit)",
    "quadratic_6h": "Quadratic (6 h fit)",
    "anchored_linear_10m": "Anchored linear (10 min fit)",
    "anchored_quadratic_10m": "Anchored quadratic (10 min fit)",
    "timesfm_direct": "TimesFM: centered clock",
    "timesfm_residual": "TimesFM: linear residual",
}
COLORS = dict(zip(METHODS, ["#7a8288", "#42938a", "#a078bb", "#3063ae", "#706233", "#d98d25", "#c7424e"]))


def label(t):
    return (EPOCH + timedelta(seconds=int(t))).isoformat(sep=" ")


def load_clock(path, satellite):
    # Merged exports add date and time before the eight native CLK1B columns.
    rows = [line.split() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not rows or not all(len(r) == 10 and r[3] == satellite for r in rows):
        raise ValueError("Expected nonempty 10-column merged CLK1B data for this satellite; use prepare_data.py for native files.")
    t = np.array([int(r[2]) for r in rows], dtype=np.int64)
    y = np.array([float(r[5]) for r in rows], dtype=np.float64)
    flags = np.array([int(r[9], 2) for r in rows], dtype=np.uint8)
    clock_ids = np.array([int(r[4]) for r in rows])
    if not np.all(np.diff(t) > 0):
        raise ValueError("Duplicate or unsorted epochs require explicit resolution.")
    if not np.isfinite(y).all():
        raise ValueError("Non-finite clock bias.")
    if not all(f"{r[0]} {r[1]}" == label(epoch) for r, epoch in zip(rows, t)):
        raise ValueError("Calendar labels disagree with receiver-time seconds.")
    gap_indices = np.flatnonzero(np.diff(t) != STEP)
    jump_indices = np.flatnonzero(np.abs(np.diff(y)) > 1e-5)
    audit = {
        "file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "records": len(t), "start_receiver_time": label(t[0]), "end_receiver_time": label(t[-1]),
        "step_counts_seconds": {str(k): int(v) for k, v in Counter(np.diff(t)).items()},
        "quality_flags": {str(k): int(v) for k, v in Counter(flags).items()},
        "clock_ids": {str(k): int(v) for k, v in Counter(clock_ids).items()},
        "gaps": [{"before": label(t[i]), "after": label(t[i+1]), "interval_seconds": int(t[i+1]-t[i])} for i in gap_indices],
        "changes_above_10_microseconds": [{"before": label(t[i]), "after": label(t[i+1]), "change_seconds": float(y[i+1]-y[i])} for i in jump_indices],
        "note": "Large changes are reported, not removed based on forecast error. Gaps are never interpolated.",
    }
    return {"t": t, "y": y, "flags": flags, "clock_ids": clock_ids}, audit


def prepare_history(history):
    """先用 float64 减去最后观测值，再换算 ns；返回未来相对钟差。"""
    history = np.asarray(history, dtype=np.float64)
    if history.shape != (CONTEXT,) or not np.isfinite(history).all():
        raise ValueError(f"Expected {CONTEXT} finite historical clock values.")
    reference = float(history[-1])
    centered = (history - reference) * 1e9  # subtraction in float64 before casting
    x = np.arange(-CONTEXT + 1, 1, dtype=np.float64) * STEP / 3600
    future_x = np.arange(1, HORIZON + 1, dtype=np.float64) * STEP / 3600
    line = np.polynomial.polynomial.polyfit(x, centered, 1)
    line_recent = np.polynomial.polynomial.polyfit(x[-360:], centered[-360:], 1)
    quadratic = np.polynomial.polynomial.polyfit(x, centered, 2)
    local_line = np.polynomial.polynomial.polyfit(x[-60:], centered[-60:], 1)
    local_quadratic = np.polynomial.polynomial.polyfit(x[-60:], centered[-60:], 2)
    historical_trend = np.polynomial.polynomial.polyval(x, line)
    future_trend = np.polynomial.polynomial.polyval(future_x, line)
    return {
        "reference_seconds": reference,
        "centered_history_ns": centered,
        "residual_history_ns": centered - historical_trend,
        "historical_trend_ns": historical_trend,
        "future_trend_ns": future_trend,
        "predictions": {
            "linear_1h": np.polynomial.polynomial.polyval(future_x, line_recent),
            "linear_6h": future_trend,
            "quadratic_6h": np.polynomial.polynomial.polyval(future_x, quadratic),
            "anchored_linear_10m": np.polynomial.polynomial.polyval(future_x, local_line)-local_line[0],
            "anchored_quadratic_10m": np.polynomial.polynomial.polyval(future_x, local_quadratic)-local_quadratic[0],
        },
    }


def numerical_check():
    # Detect unit, horizon indexing and float32 precision errors on a known clock.
    x = np.arange(-CONTEXT + 1, 1, dtype=np.float64) * STEP
    history = -0.008 - 1.5e-8*x
    prepared = prepare_history(history)
    expected = -15 * np.arange(1, HORIZON + 1) * STEP
    for value in prepared["predictions"].values():
        np.testing.assert_allclose(value, expected, atol=1e-6)
    assert np.max(np.abs(prepared["residual_history_ns"])) < 1e-6


def make_windows(data, satellite):
    accepted, excluded = [], []
    for day in [2, 3]:
        for hour in [0, 6, 12, 18]:
            origin = int((datetime(2025, 6, day, hour) - EPOCH).total_seconds())
            desired = origin + np.arange(-CONTEXT + 1, HORIZON + 1) * STEP
            start = int(np.searchsorted(data["t"], desired[0]))
            stop = start + len(desired)
            record = {"satellite": satellite, "origin_receiver_time": label(origin)}
            if not np.array_equal(data["t"][start:stop], desired):
                excluded.append({**record, "reason": "Missing epochs inside history or holdout; no interpolation"})
                continue
            if np.any(data["flags"][start:stop] != 0):
                excluded.append({**record, "reason": "Nonzero quality flag under the conservative pilot policy"})
                continue
            if len(np.unique(data["clock_ids"][start:stop])) != 1:
                excluded.append({**record, "reason": "Clock identifier changes inside window"})
                continue
            history = data["y"][start:start+CONTEXT].copy()
            truth = data["y"][start+CONTEXT:stop].copy()
            assert np.all(desired[:CONTEXT] <= origin) and np.all(desired[CONTEXT:] > origin)
            prepared = prepare_history(history)
            prepared.update(record)
            prepared.update({"origin": origin, "truth_seconds": truth,
                             "truth_relative_ns": (truth-prepared["reference_seconds"])*1e9,
                             "history_start": label(desired[0]), "horizon_end": label(desired[-1]),
                             "quantiles": {}})
            accepted.append(prepared)
    return accepted, excluded


def write_rows(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate(windows, methods=None):
    methods = list(windows[0]["predictions"]) if methods is None else methods
    records = []
    for w in windows:
        for method, prediction in w["predictions"].items():
            for minutes in HORIZONS:
                n = minutes*60//STEP
                error = prediction[:n]-w["truth_relative_ns"][:n]
                qs = w["quantiles"].get(method)
                coverage = float(np.mean((w["truth_relative_ns"][:n] >= qs[:n, 0]) &
                                         (w["truth_relative_ns"][:n] <= qs[:n, -1]))) if qs is not None else None
                records.append({"satellite": w["satellite"], "origin_receiver_time": w["origin_receiver_time"],
                                "method": method, "horizon_minutes": minutes, "points": n,
                                "rmse_ns": float(np.sqrt(np.mean(error**2))), "mae_ns": float(np.mean(np.abs(error))),
                                "max_abs_error_ns": float(np.max(np.abs(error))), "endpoint_error_ns": float(error[-1]),
                                "q10_q90_coverage": coverage})
    aggregated = []
    for satellite in ["C", "D"]:
        for minutes in HORIZONS:
            for method in methods:
                group = [r for r in records if r["satellite"] == satellite and r["method"] == method and r["horizon_minutes"] == minutes]
                assert group
                aggregated.append({"satellite": satellite, "method": method, "horizon_minutes": minutes,
                                   "windows": len(group), "points": sum(r["points"] for r in group),
                                   "rmse_ns": float(np.sqrt(np.mean([r["rmse_ns"]**2 for r in group]))),
                                   "mae_ns": float(np.mean([r["mae_ns"] for r in group])),
                                   "max_abs_error_ns": max(r["max_abs_error_ns"] for r in group),
                                   "median_window_rmse_ns": float(np.median([r["rmse_ns"] for r in group])),
                                   "q10_q90_coverage": float(np.mean([r["q10_q90_coverage"] for r in group])) if method.startswith("timesfm") else None})
    return records, aggregated


def plots(windows, aggregated, out):
    methods = list(windows[0]["predictions"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True)
    for ax, satellite in zip(axes, ["C", "D"]):
        for method in methods:
            name = METHODS[method]
            rows = [r for r in aggregated if r["satellite"] == satellite and r["method"] == method]
            ax.plot(HORIZONS, [r["rmse_ns"] for r in rows], "o-", label=name, color=COLORS[method], lw=1.8)
        count = sum(w["satellite"] == satellite for w in windows)
        ax.set_title(f"GRACE-FO {satellite} | {count} chronological holdouts")
        ax.set_xticks(HORIZONS)
        ax.set_xlabel("Forecast horizon (minutes)")
        ax.set_yscale("log")
        ax.grid(which="both", alpha=.18)
    axes[0].set_ylabel("Pooled RMSE over forecast interval (ns, log scale)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9)
    fig.suptitle("CLK1B clock prediction | 10 s samples | 6 h context | no fine-tuning")
    fig.tight_layout(rect=(0, .18, 1, .94))
    fig.savefig(out / "rmse_comparison.png", dpi=170)
    plt.close(fig)

    # Fixed choice: first accepted C origin, selected before examining errors.
    w = next(w for w in windows if w["satellite"] == "C")
    lead = np.arange(1, HORIZON+1)*STEP/60
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    historical_x = np.arange(-359, 1)*STEP/60
    axes[0].plot(historical_x, w["residual_history_ns"][-360:], color="#707b87", label="Observed history (last 1 h)")
    axes[0].plot(lead, w["truth_relative_ns"]-w["future_trend_ns"], color="black", lw=2, label="Held-out actual")
    for method in methods:
        name = METHODS[method]
        if method.startswith("anchored_"):
            continue  # Additional short-window baselines appear in the aggregate plot.
        prediction = w["predictions"][method]
        axes[0].plot(lead, prediction-w["future_trend_ns"], color=COLORS[method], label=name, lw=1.5)
        axes[1].plot(lead, prediction-w["truth_relative_ns"], color=COLORS[method], label=name, lw=1.5)
    qs = w["quantiles"].get("timesfm_residual")
    if qs is not None:
        axes[0].fill_between(lead, qs[:, 0]-w["future_trend_ns"], qs[:, -1]-w["future_trend_ns"],
                             color=COLORS["timesfm_residual"], alpha=.10, label="Residual-model q10-q90")
    axes[0].axvline(0, color="black", ls="--", lw=1)
    axes[0].set_ylabel("Clock minus past-fitted linear trend (ns)")
    axes[0].set_xlabel("Minutes from forecast origin")
    axes[0].legend(loc="best", fontsize=8, ncol=2)
    axes[1].axhline(0, color="black", lw=.8)
    axes[1].set_ylabel("Prediction - actual (ns)")
    axes[1].set_xlabel("Forecast lead (minutes)")
    for ax in axes: ax.grid(alpha=.18)
    fig.suptitle(f"GRACE-FO C | first scheduled holdout: {w['origin_receiver_time']} receiver time")
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(out / "example_C_forecast.png", dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/sample")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/baseline")
    parser.add_argument("--model", choices=["baseline", "timesfm"], default="baseline")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "models/timesfm-3.0-pytorch")
    args = parser.parse_args()
    out = args.output_dir
    if out.exists() and any(out.iterdir()):
        parser.error("Output directory is nonempty. Choose a new --output-dir to preserve previous experiments.")
    out.mkdir(parents=True, exist_ok=True)
    numerical_check()
    windows, exclusions, audits = [], [], {}
    for satellite in ["C", "D"]:
        path = args.data_dir / f"CLK1B_2025-06-01_to_2025-06-03_{satellite}.txt"
        data, audits[satellite] = load_clock(path, satellite)
        accepted, excluded = make_windows(data, satellite)
        windows.extend(accepted)
        exclusions.extend(excluded)
        print(f"{satellite}: {len(data['t'])} records, {len(accepted)} windows, {len(excluded)} excluded", flush=True)
    (out / "data_audit.json").write_text(json.dumps(audits, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "excluded_windows.json").write_text(json.dumps(exclusions, indent=2), encoding="utf-8")
    if not all(any(w["satellite"] == sat for w in windows) for sat in ["C", "D"]):
        raise ValueError("Both satellites need an accepted window in the fixed June 2025 schedule.")
    model_metadata = {}
    start = time.perf_counter()
    if args.model == "timesfm":
        from timesfm_backend import add_timesfm_predictions
        model_metadata = add_timesfm_predictions(windows, args.checkpoint, args.device)
    inference_seconds = time.perf_counter()-start
    per_window, aggregate = evaluate(windows)
    write_rows(out / "metrics_by_window.csv", per_window)
    write_rows(out / "metrics_summary.csv", aggregate)
    predictions = []
    for w in windows:
        for method, values in w["predictions"].items():
            qs = w["quantiles"].get(method)
            for i in range(HORIZON):
                predictions.append({"satellite": w["satellite"], "origin_receiver_time": w["origin_receiver_time"],
                                    "target_receiver_time": label(w["origin"]+(i+1)*STEP), "lead_seconds": (i+1)*STEP,
                                    "method": method, "actual_seconds": format(w["truth_seconds"][i], ".17g"),
                                    "forecast_seconds": format(w["reference_seconds"]+values[i]*1e-9, ".17g"),
                                    "error_ns": float(values[i]-w["truth_relative_ns"][i]),
                                    "q10_seconds": format(w["reference_seconds"]+qs[i, 0]*1e-9, ".17g") if qs is not None else "",
                                    "q90_seconds": format(w["reference_seconds"]+qs[i, -1]*1e-9, ".17g") if qs is not None else ""})
    write_rows(out / "predictions.csv", predictions)
    plots(windows, aggregate, out)
    summary = {
        "experiment": "Fixed chronological CLK1B holdout, 2025-06-01 through 2025-06-03 exports",
        "satellites": ["C", "D"], "sampling_seconds": STEP, "context_points": CONTEXT,
        "horizon_points": HORIZON, "horizons_minutes": HORIZONS,
        "schedule": "Last observed epoch at 00, 06, 12, 18 on June 2 and June 3; 16 scheduled windows",
        "accepted_windows": len(windows), "excluded_windows": exclusions,
        "timestamp_scale": "Receiver-time epoch seconds from 2000-01-01 12:00:00; not labeled UTC",
        "target": "CLK1B eps_time [seconds], GPS Time = rcv_time + eps_time",
        "evaluation": "Metrics at shorter horizons use prefixes of the same 60-minute forecast; no future observations are fed back.",
        "preprocessing": "float64 last-value centering to ns; optional 6-hour OLS detrending using history only; then float32 model inputs; no fill or smoothing",
        "model_options": {"per_core_batch_size": 4, "use_symmetric_averaging": False, "make_positive": False, "sort_quantiles": True},
        "model_note": "Direct input still uses the checkpoint's default internal linear-detrending machinery. No fine-tuning or held-out hyperparameter selection.",
        "baseline_note": "After the initial three long-window baselines, two fixed 10-minute anchored baselines were added as an exploratory fairness check. All five baselines are reported; no window length search was performed.",
        "package_versions": {p: importlib.metadata.version(p) for p in ["numpy", "matplotlib"]},
        "mode": args.model, "timesfm": model_metadata,
        "optional_model_seconds_including_loading": inference_seconds,
        "metrics": aggregate,
        "limitations": ["Three-day pilot, not full-year validation", "Gapped windows excluded and disclosed; no claim of reset prediction",
                        "Postprocessed CLK1B reference products; not a live operational test", "Upstream pretraining exposure to GRACE-FO data is unknown",
                        "Quantile calibration is measured on these holdouts only; baseline parameter uncertainty is not included in residual-model bands"],
    }
    (out / "results.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    for satellite in ["C", "D"]:
        print(f"\n{satellite} pooled RMSE [ns]", flush=True)
        for method in windows[0]["predictions"]:
            values = [r["rmse_ns"] for r in aggregate if r["satellite"] == satellite and r["method"] == method]
            print(method, dict(zip(HORIZONS, [round(v, 6) for v in values])), flush=True)
    print(f"\nSaved: {out}", flush=True)


if __name__ == "__main__":
    main()
