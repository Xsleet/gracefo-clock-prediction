"""低轨卫星精密钟差预报Demo：依次运行 GRACE-FO 的 8 个固定窗口，比较三模型。"""
import argparse
import csv
from datetime import timedelta
from pathlib import Path
import subprocess
import sys

import numpy as np

from main import EPOCH, plt

ROOT = Path(__file__).resolve().parent
METHODS = ["timesfm", "polynomial", "periodic"]
NAMES = ["TimesFM (original)", "Polynomial (degree 2)", "Polynomial + periodic"]


def summarize(scores):
    """每窗口都有 360 个未来点；总体 RMSE = sqrt(mean(各窗口 RMSE²))。"""
    return [{"model": name, "windows": len(scores), "points": len(scores) * 360,
             "pooled_rmse_ns": float(np.sqrt(np.mean(scores[:, i]**2))),
             "median_window_rmse_ns": float(np.median(scores[:, i])),
             "wins": int(np.sum(np.argmin(scores, axis=1) == i))}
            for i, name in enumerate(METHODS)]


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--checkpoint", default="google/timesfm-3.0-pytorch")
    args = parser.parse_args()
    rows, scores, ticks = [], [], []
    for number in range(1, 9):
        # 直接复用学生已经读过的 main.py；每个窗口各保存一张图和一份 CSV。
        subprocess.run([sys.executable, str(ROOT / "main.py"), "--device", args.device,
                        "--checkpoint", args.checkpoint, "--window", str(number)], check=True)
        path = ROOT / "outputs" / f"timesfm-{args.device}" / f"window-{number:02d}" / "prediction.csv"
        data = np.genfromtxt(path, delimiter=",", names=True)
        if len(data) != 360:
            raise ValueError("每个窗口必须有 360 个预测点")
        # 从导出的绝对钟差独立复算，而不是抄终端打印的数值。
        errors = [(data[f"{name}_seconds"] - data["true_seconds"]) * 1e9 for name in METHODS]
        for name, error in zip(METHODS, errors):
            np.testing.assert_allclose(error, data[f"{name}_error_ns"], atol=1e-5)
        rmse = [float(np.sqrt(np.mean(error**2))) for error in errors]
        origin = EPOCH + timedelta(seconds=int(data["receiver_seconds"][0]) - 10)
        rows.append({"window": number, "origin_receiver_time": origin.isoformat(sep=" "),
                     **{f"{name}_rmse_ns": value for name, value in zip(METHODS, rmse)},
                     "best_model": METHODS[int(np.argmin(rmse))]})
        scores.append(rmse)
        ticks.append(origin.strftime("%m-%d\n%H:%M"))

    scores = np.array(scores)
    summary = summarize(scores)
    output = ROOT / "outputs" / f"windows-{args.device}"
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "by_window.csv", rows)
    write_csv(output / "summary.csv", summary)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for i, (name, color) in enumerate(zip(NAMES, ["#2463a6", "#d76a13", "#168568"])):
        ax.plot(range(1, 9), scores[:, i], "o-", color=color, label=name)
    ax.set(xticks=range(1, 9), xticklabels=ticks, ylabel="60-minute forecast RMSE (ns)",
           xlabel="Forecast origin (receiver time)",
           title=f"GRACE-FO C | 8 fixed windows | 6 h history -> 60 min forecast | {args.device.upper()}")
    ax.legend(fontsize=9)
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(output / "comparison.png", dpi=150)
    plt.close(fig)
    for row in summary:
        print(f"{row['model']}: 总体 RMSE {row['pooled_rmse_ns']:.4f} ns；最优 {row['wins']}/8 个窗口")
    print(f"汇总结果：{output}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    main()
