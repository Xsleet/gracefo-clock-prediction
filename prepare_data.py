"""合并官方 CLK1B 日文件。仅保留 qualflg==0，重叠历元选最后记录。"""
import argparse
from collections import Counter
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
import re

EPOCH = datetime(2000, 1, 1, 12)


def merge_native(paths, satellite):
    # 必须按文件名中的日期排序，避免绝对路径决定日界选择。
    paths = sorted(paths, key=lambda p: p.name)
    if not paths or len({p.name for p in paths}) != len(paths):
        raise ValueError("Need source files with distinct daily filenames.")
    selected, stats, sources = {}, Counter(), []
    for path in paths:
        if not re.fullmatch(rf"CLK1B_\d{{4}}-\d{{2}}-\d{{2}}_{satellite}_\d{{2}}\.txt", path.name):
            raise ValueError(f"Unexpected daily filename: {path.name}")
        content = path.read_text(encoding="utf-8-sig")
        if "# End of YAML header" not in content:
            raise ValueError(f"Missing official YAML header: {path.name}")
        sources.append({"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        for line in content.split("# End of YAML header", 1)[1].splitlines():
            if not line.strip():
                continue
            row = line.split()
            if len(row) != 8 or row[1] != satellite:
                raise ValueError(f"Malformed CLK1B row in {path.name}")
            t = int(row[0])
            flag = int(row[7], 2)
            if not 0 <= flag <= 255:
                raise ValueError("Quality flag must be 8-bit.")
            stats["raw_records"] += 1
            # 教学试验采用保守规则：只使用全部质量位均为 0 的记录。
            # 某些非零位仅描述日界平滑，不等于普遍无效；本规则需随结果披露。
            if flag != 0:
                stats["nonzero_quality_records_removed"] += 1
                continue
            if t in selected:
                stats["duplicate_records"] += 1
                if float(row[3]) != float(selected[t][3]):
                    stats["different_bias_duplicate_records"] += 1
            selected[t] = row
    lines = []
    for t, row in sorted(selected.items()):
        calendar = (EPOCH + timedelta(seconds=t)).isoformat(sep=" ")
        lines.append(calendar + " " + " ".join(row))
    stats["merged_records"] = len(lines)
    if not lines:
        raise ValueError("No CLK1B records found.")
    return "\n".join(lines) + "\n", {
        "satellite": satellite,
        "time_system": "receiver time, seconds since 2000-01-01 12:00:00; not UTC",
        "target": "eps_time [s]; GPS Time = rcv_time + eps_time",
        "source_product_doi": "10.5067/GFJPL-L1B04",
        "source_files": sources,
        "merge_rule": "Keep qualflg == 0 records only; daily filenames ascending, last retained row at each epoch. No interpolation or clock alignment. Nonzero quality flags do not universally mean invalid data; this is a conservative pilot selection policy.",
        "counts": dict(stats),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", type=Path, nargs="+", help="Explicit native daily CLK1B files, same satellite")
    parser.add_argument("--satellite", choices=["C", "D"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.output.with_suffix(".manifest.json")
    if args.output.exists() or manifest_path.exists():
        parser.error("Output or manifest exists; choose a new output path.")
    text, manifest = merge_native(args.files, args.satellite)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8", newline="\n")
    manifest["output_file"] = args.output.name
    manifest["output_sha256"] = hashlib.sha256(args.output.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved {manifest['counts']['merged_records']} records to {args.output}")


if __name__ == "__main__":
    main()
