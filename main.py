"""读 main.py 即可：读钟差 -> 用历史预测未来 -> 画去趋势后的预报图。"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# 10 秒一个点：2160 点约 6 小时，360 点为未来 60 分钟。
HISTORY_POINTS = 2160
PREDICT_POINTS = 360
ROOT = Path(__file__).resolve().parent


def polynomial_forecasts(history_ns, horizon):
    """两个对照都只看历史：二次多项式，以及二次多项式 + 一组正弦/余弦。"""
    t = np.arange(1 - len(history_ns), 1) / 360  # 小时，起报点为 0
    future = np.arange(1, horizon + 1) / 360
    polynomial = np.polyval(np.polyfit(t, history_ns, 2), future)

    def features(hours, period_minutes):
        phase = 2 * np.pi * hours / (period_minutes / 60)
        return np.column_stack([np.ones_like(hours), hours, hours**2,
                                np.sin(phase), np.cos(phase)])

    # 预先固定搜索范围和步长；按历史拟合误差选周期，不查看未来真值。
    best_mse = np.inf
    for period in range(60, 151):  # 60–150 分钟，每次增加 1 分钟
        design = features(t, period)
        coefficients = np.linalg.lstsq(design, history_ns, rcond=None)[0]
        mse = np.mean((design @ coefficients - history_ns)**2)
        if mse < best_mse:
            best_mse, best_period = mse, period
            periodic = features(future, period) @ coefficients
    return polynomial, periodic, best_period


def main():
    # 1. 选择 CPU 或 GPU；两者运行相同的模型。
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--model", choices=["timesfm", "linear"], default="timesfm")
    parser.add_argument("--checkpoint", default="google/timesfm-3.0-pytorch")
    parser.add_argument("--finetuned", nargs="?", type=Path,
                        const=ROOT / "outputs/finetune/head.pt", help="加入本机微调的预测头对比")
    args = parser.parse_args()
    if args.model == "linear" and args.device == "gpu":
        parser.error("线性示例使用 CPU；GPU 模式请用默认的 TimesFM。")
    if args.finetuned and (args.model != "timesfm" or not args.finetuned.is_file()):
        parser.error("微调对比需选择 TimesFM，并先运行 python finetune.py 生成预测头权重。")

    # 2. 读两列数据：接收机时间（秒）、钟差（秒）。先划分，后处理。
    data = np.loadtxt(ROOT / "data/clock.csv", delimiter=",", skiprows=1)
    if HISTORY_POINTS < 60 or PREDICT_POINTS < 1:
        parser.error("至少需要 60 个历史点和 1 个预测点。")
    sample = data[:HISTORY_POINTS + PREDICT_POINTS]
    if len(sample) != HISTORY_POINTS + PREDICT_POINTS or not np.isfinite(sample).all():
        parser.error("数据长度不足，或包含无效数值。")
    if not np.all(np.diff(sample[:, 0]) == 10):
        parser.error("数据必须每 10 秒一条，不允许跨缺口。")
    history = sample[:HISTORY_POINTS, 1]
    truth = sample[HISTORY_POINTS:, 1]  # 真值留到最后评价，不传给模型。

    # 3. 减去最后一个历史钟差，再从秒转为纳秒（ns），避免精度损失。
    last = history[-1]
    history_ns = (history - last) * 1e9
    print(f"历史 {len(history)} 点 -> 预报 {PREDICT_POINTS} 点，设备：{args.device}")

    # 4. 各模型预测时只读取历史；多项式对照也使用完整的 6 小时。
    polynomial, periodic, period = polynomial_forecasts(history_ns, PREDICT_POINTS)
    # 保留 --model linear 快速示例：最近 10 分钟的斜率，从最后观测值延伸。
    slope, _ = np.polyfit(np.arange(-59, 1), history_ns[-60:], 1)
    linear = slope * np.arange(1, PREDICT_POINTS + 1)
    prediction = linear
    if args.model == "timesfm":
        import torch
        from timesfm3 import ModelConfig, TimesFM3Evaluator
        if args.device == "gpu" and not torch.cuda.is_available():
            parser.error("没有可用的 NVIDIA GPU。请用 --device cpu，或安装 CUDA 版 PyTorch。")
        print("加载 TimesFM（首次运行会下载约 1.32 GB 权重）……", flush=True)
        model = TimesFM3Evaluator(ModelConfig(
            checkpoint_path=args.checkpoint, revision="43046b85ec22d584a13f8098c2ed39c889e129c2",
            device="cuda" if args.device == "gpu" else "cpu", per_core_batch_size=1))
        def predict_timesfm():
            with torch.inference_mode():
                result = list(model.predict_batch(
                    [history_ns.astype(np.float32)], horizon=PREDICT_POINTS,
                    return_quantiles=False, make_positive=False, use_symmetric_averaging=False))
            return result[0].forecast.astype(np.float64)
        prediction = predict_timesfm()  # 先预测并保存原始模型结果，再加载微调权重。
    forecasts = {args.model: prediction, "polynomial": polynomial, "periodic": periodic}
    if args.finetuned:
        saved = torch.load(args.finetuned, map_location="cpu", weights_only=True)
        meta = saved["metadata"]
        if (meta["method"] != "output_head_finetuning" or
                meta["revision"] != "43046b85ec22d584a13f8098c2ed39c889e129c2" or
                meta["history_points"] != HISTORY_POINTS or meta["horizon_points"] != PREDICT_POINTS or
                meta["step_seconds"] != 10 or meta["last_training_or_validation_epoch"] >= sample[0, 0]):
            parser.error("微调权重配置不匹配，或其训练/验证数据与展示窗口重叠。")
        model.model.output_head.load_state_dict(saved["head"], strict=True)
        forecasts = {"timesfm": prediction, "finetuned": predict_timesfm(),
                     "polynomial": polynomial, "periodic": periodic}

    # 5. 预测完成后才与真实值比较。RMSE 越小，预报越准。
    truth_ns = (truth - last) * 1e9
    labels = {"timesfm": "TimesFM (original)", "finetuned": "TimesFM (head-tuned)", "linear": "Linear (10 min)",
              "polynomial": "Polynomial (degree 2)", "periodic": "Polynomial + periodic"}
    columns = [sample[HISTORY_POINTS:, 0], truth]
    header = ["receiver_seconds", "true_seconds"]
    rmses = {}
    print(f"历史拟合选出的周期：{period} 分钟")
    for name, values in forecasts.items():
        if values.shape != (PREDICT_POINTS,) or not np.isfinite(values).all():
            raise ValueError(f"{name} 没有返回完整、有效的预报。")
        error = values - truth_ns
        rmses[name] = np.sqrt(np.mean(error**2))
        print(f"{labels[name]} RMSE = {rmses[name]:.4f} ns")
        columns.extend([last + values * 1e-9, error])
        header.extend([f"{name}_seconds", f"{name}_error_ns"])

    # 6. 只输出一个 CSV 和一张图。再次运行会更新同一目录。
    run_name = "comparison" if args.finetuned else args.model
    output = ROOT / "outputs" / f"{run_name}-{args.device}"
    output.mkdir(parents=True, exist_ok=True)
    np.savetxt(output / "prediction.csv", np.column_stack(columns),
        delimiter=",", header=",".join(header), comments="",
        fmt=["%.0f", "%.17g"] + ["%.17g", "%.10g"] * len(forecasts))
    minutes = np.arange(1, PREDICT_POINTS + 1) * 10 / 60
    history_minutes = np.arange(1 - HISTORY_POINTS, 1) * 10 / 60
    # 只为绘图：用过去 6 小时拟合直线，历史、真值、预测都减去同一条线。
    # 趋势不使用未来数据；不改变上面的模型输入、预测结果和 CSV。
    trend = np.polyfit(history_minutes, history_ns, 1)
    future_trend = np.polyval(trend, minutes)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(history_minutes, history_ns - np.polyval(trend, history_minutes),
            color="#596773", label="Observed history")
    colors = {"timesfm": "#2463a6", "linear": "#2463a6", "finetuned": "#a43486",
              "polynomial": "#d76a13", "periodic": "#168568"}
    for name, values in forecasts.items():
        ax.plot(minutes, values - future_trend, color=colors[name], linewidth=2,
                label=f"{labels[name]} (RMSE {rmses[name]:.3f} ns)")
    ax.plot(minutes, truth_ns - future_trend, ":", color="black", label="Future observations")
    ax.axvspan(0, minutes[-1], color="#f1b75c", alpha=0.18)
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.text(history_minutes[0] / 2, 0.78, f"History: {HISTORY_POINTS * 10 / 3600:g} h",
            transform=ax.get_xaxis_transform(), ha="center")
    ax.text(minutes[-1] / 2, 0.08, f"Forecast\n{minutes[-1]:g} min",
            transform=ax.get_xaxis_transform(), ha="center", fontsize=9)
    ax.set(xlim=(-HISTORY_POINTS * 10 / 60, minutes[-1]), ylabel="Detrended clock offset (ns)",
           xlabel="Minutes relative to forecast start (0)",
           title=f"GRACE-FO C | linear trend fitted on history only | {args.device.upper()}")
    ax.set_xticks(np.r_[np.arange(-HISTORY_POINTS * 10 / 60, 0, 60), 0, minutes[-1]])
    ax.legend(loc="upper left", ncol=3, fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "result.png", dpi=150)
    plt.close(fig)
    print(f"结果：{output / 'result.png'}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    main()
