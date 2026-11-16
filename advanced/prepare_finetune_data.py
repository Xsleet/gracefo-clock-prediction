"""从 5 月 27–31 日官方 C 星 CLK1B 文件生成教学微调数据及来源清单。"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np

from prepare_data import EPOCH, merge_native

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="存放官方每日子目录的 2025 数据目录")
    args = parser.parse_args()
    paths = [args.source / f"gracefo_1B_2025-05-{day}_RL04.ascii.noLRI" /
             f"CLK1B_2025-05-{day}_C_04.txt" for day in range(27, 32)]
    text, manifest = merge_native(paths, "C")
    start = (datetime(2025, 5, 27) - EPOCH).total_seconds()
    end = (datetime(2025, 6, 1) - EPOCH).total_seconds()
    rows = [line.split() for line in text.splitlines() if start <= int(line.split()[2]) < end]
    if {row[4] for row in rows} != {"-1"}:
        raise ValueError("Unexpected clock identifier")
    data = np.array([[float(row[2]), float(row[5])] for row in rows])
    output = ROOT / "data/finetune.csv"
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        np.savetxt(stream, data, delimiter=",", header="receiver_seconds,clock_seconds",
                   comments="", fmt=["%.0f", "%.17g"])
    manifest.update(output_file=output.name,
                    output_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                    rows=len(data), selected_clock_id=-1,
                    train_dates="2025-05-27 through 2025-05-30",
                    validation_dates="2025-05-31",
                    window_rule="6 h history + 1 h target, within one receiver calendar day; no gaps",
                    selection_rule="Fixed dates before the existing June 2 demonstration; not selected by error")
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved {len(data)} records")


if __name__ == "__main__":
    main()
