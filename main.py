"""读 main.py 即可：读钟差 -> 用历史预测未来 -> 画误差图。"""
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


def main():
    # 1. 选择 CPU 或 GPU；两者运行相同的模型。
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--model", choices=["timesfm", "linear"], default="timesfm")
    parser.add_argument("--checkpoint", default="google/timesfm-3.0-pytorch")
    args = parser.parse_args()
    if args.model == "linear" and args.device == "gpu":
        parser.error("线性示例使用 CPU；GPU 模式请用默认的 TimesFM。")

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

    # 4. 简单对照：用最近 10 分钟拟合斜率，从最后观测值向后延伸。
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
        with torch.inference_mode():
            result = list(model.predict_batch(
                [history_ns.astype(np.float32)], horizon=PREDICT_POINTS,
                return_quantiles=False, make_positive=False, use_symmetric_averaging=False))
        prediction = result[0].forecast.astype(np.float64)
    if prediction.shape != (PREDICT_POINTS,) or not np.isfinite(prediction).all():
        raise ValueError("模型没有返回完整、有效的预报。")

    # 5. 预测完成后才与真实值比较。RMSE 越小，预报越准。
    truth_ns = (truth - last) * 1e9
    error = prediction - truth_ns
    linear_error = linear - truth_ns
    rmse = np.sqrt(np.mean(error**2))
    print(f"{args.model} RMSE = {rmse:.4f} ns；线性对照 = {np.sqrt(np.mean(linear_error**2)):.4f} ns")

    # 6. 只输出一个 CSV 和一张图。再次运行会更新同一目录。
    output = ROOT / "outputs" / f"{args.model}-{args.device}"
    output.mkdir(parents=True, exist_ok=True)
    np.savetxt(output / "prediction.csv", np.column_stack([
        sample[HISTORY_POINTS:, 0], truth, last + prediction * 1e-9, error]),
        delimiter=",", header="receiver_seconds,true_seconds,predicted_seconds,error_ns",
        comments="", fmt=["%.0f", "%.17g", "%.17g", "%.10g"])
    minutes = np.arange(1, PREDICT_POINTS + 1) * 10 / 60
    history_minutes = np.arange(1 - HISTORY_POINTS, 1) * 10 / 60
    fig, (overview, errors) = plt.subplots(2, 1, figsize=(10, 7))
    # 上图展示完整历史与未来；0 分钟是最后一个已知观测，预测从 +10 秒开始。
    overview.plot(history_minutes, history * 1e3, color="#596773", label="Observed history")
    overview.plot(minutes, (last + prediction * 1e-9) * 1e3, color="#2463a6", linewidth=2.5,
                  label="TimesFM forecast" if args.model == "timesfm" else "Linear forecast")
    overview.plot(minutes, truth * 1e3, ":", color="black", label="Future observations (comparison only)")
    overview.axvspan(0, minutes[-1], color="#f1b75c", alpha=0.18)
    overview.axvline(0, color="black", linestyle="--", linewidth=1)
    overview.text(history_minutes[0] / 2, 0.92, f"History: {HISTORY_POINTS * 10 / 3600:g} h",
                  transform=overview.get_xaxis_transform(), ha="center")
    overview.text(minutes[-1] / 2, 0.87, f"Forecast\n{minutes[-1]:g} min",
                  transform=overview.get_xaxis_transform(), ha="center", fontsize=9)
    overview.set(xlim=(-HISTORY_POINTS * 10 / 60, minutes[-1]), ylabel="Clock offset (ms)",
                 xlabel="Minutes relative to forecast start (0)", title=f"GRACE-FO C | {args.device.upper()}")
    overview.set_xticks(np.r_[np.arange(-HISTORY_POINTS * 10 / 60, 0, 60), 0, minutes[-1]])
    overview.legend(loc="lower left", fontsize=8)
    # 下图放大未来误差：上图两条曲线几乎重合时，也能判断预报好坏。
    errors.plot(minutes, linear_error, "--", color="gray", label="Linear baseline")
    if args.model == "timesfm":
        errors.plot(minutes, error, color="#2463a6", label=f"TimesFM | RMSE {rmse:.3f} ns")
    errors.axhline(0, color="black", linewidth=0.8)
    errors.set(xlim=(0, minutes[-1]), xlabel="Minutes ahead", ylabel="Prediction - observation (ns)",
               title="Forecast error | closer to zero is better")
    errors.legend()
    for ax in (overview, errors):
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "result.png", dpi=150)
    plt.close(fig)
    print(f"结果：{output / 'result.png'}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    main()
