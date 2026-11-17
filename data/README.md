# GRACE-FO 钟差数据与使用方法

可直接下载 [clock.csv](https://raw.githubusercontent.com/Xsleet/gracefo-clock-prediction/main/data/clock.csv)。它来自 GRACE-FO C 星 2025 年 6 月 1–3 日的 NASA/JPL RL04 CLK1B 日产品，含 25,981 条记录。下载或克隆整个仓库后，`main.py` 默认读取这个 CSV。

| 列名 | 含义 |
|---|---|
| `receiver_seconds` | 自 2000-01-01 12:00:00 起的接收机时间秒数。输入起报时刻时也使用接收机时标 |
| `clock_seconds` | CLK1B 的 eps_time，单位为秒；GPS Time = receiver time + eps_time |

数据时间范围为 2025-05-31 23:55:00 至 2025-06-04 00:05:00，包含原日产品的边缘重叠段。此 C 星子集每 10 秒一条且连续；质量标记全部为 0，clock_id 全部为 -1。

`main.py` 按 `--window 1` 至 `--window 8` 选择窗口，每次提取 2520 行，前 2160 行是历史，后 360 行用于评价预测。固定起报时间为 6 月 2、3 日各自的 00、06、12、18 时。这与原完整实验的 C 星起报表一致，不按效果选窗。

安装环境见 [仓库首页](../README.md#2-选-cpu-或-gpu-运行)。在仓库根目录运行：

```bash
python main.py --device cpu --window 2
python compare_windows.py --device cpu
```

有 NVIDIA GPU 时可把 `cpu` 改成 `gpu`。窗口 2 在 6 月 2 日 06:00 起报，预测 06:00:10 到 07:00。单窗口的 `prediction.csv` 和 `result.png` 位于 `outputs/timesfm-cpu/window-02/`（GPU 对应 `timesfm-gpu`）。8 个窗口的 `by_window.csv` 和 `summary.csv` 位于 `outputs/windows-cpu/`（GPU 对应 `windows-gpu`）。预测钟差以秒保存，误差与 RMSE 用 ns，图上统一减去由历史拟合的直线。

## 换任意起报时间

例如用附带数据回测 6 月 3 日 09:00 起报。脚本需要起报前的 2,160 个点（含起报点）和随后 360 个点，未来观测只用于评分。时间须落在整 10 秒历元，窗口内缺测会报错。

```bash
python main.py --device cpu --origin 2025-06-03T09:00:00 --output outputs/C-jun03-0900
```

## 换日期或卫星

1. 从 [NASA Earthdata 的 GRACE-FO RL04 数据集合](https://search.earthdata.nasa.gov/search/granules?p=C2036882118-POCLOUD)选择需要的日期，下载并解压日产品。把同一颗卫星的 `CLK1B_YYYY-MM-DD_C_04.txt`（D 星文件名用 `D`）放到 `data/raw/`。下载日期应覆盖起报前 6 小时和起报后 60 分钟，跨日时也要下载相邻日。
2. 用 `advanced/prepare_data.py` 生成两列 CSV。脚本保留质量标记为 0 的记录，日界重复历元保留后读到的有效记录，并保存相邻的 `.manifest.json`，记录源文件和 SHA-256。
3. 用 `--data` 读取新 CSV，`--origin` 设定接收机起报时间。用不同的 `--output` 为各次实验保存独立结果。

下面以 C 星两天原始文件为例。换日期时替换文件名和起报时间，换卫星时同时替换文件名中的卫星字母和 `--satellite`。

```bash
python advanced/prepare_data.py data/raw/CLK1B_2025-06-02_C_04.txt data/raw/CLK1B_2025-06-03_C_04.txt --satellite C --output outputs/C-clk1b.csv
python main.py --device cpu --data outputs/C-clk1b.csv --origin 2025-06-03T09:00:00 --output outputs/C-custom-test
```

比较多个模型时，先定好一组起报时刻，再在相同窗口内评分，并记录缺测及排除理由。仓库还附有 [C/D 星完整字段子集](sample/)和来源清单。[日界处理、质量规则与钟跳说明](../advanced/docs/data.md) · [数据引用与许可](../DATA_SOURCES.md)
