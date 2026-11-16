# GRACE-FO 钟差预报 · 简明版

**只读 [main.py](main.py) 一个文件，就能看完实验。**

用 C 星**过去 6 小时的钟差，预测未来 60 分钟**。数据已附带，无需自己下载。

默认使用 Google 官方预训练 **TimesFM 3.0**（`google/timesfm-3.0-pytorch`），没有在钟差数据上微调，也没有加载 LoRA；本例是零样本预测。模型自身的内部去趋势保留官方默认设置。

```text
读数据 → 取历史 → TimesFM 预测 → 与真实值比较 → 看去趋势后的预报图
```

## 1. 准备

安装 Python **3.11 或 3.12** 和 Git，打开终端：

```bash
git clone https://github.com/Xsleet/gracefo-clock-prediction.git
cd gracefo-clock-prediction
python -m venv .venv
```

激活环境：Windows 命令提示符用 `.venv\Scripts\activate.bat`；Linux/macOS 用 `source .venv/bin/activate`。**下面命令均在激活后的终端执行。** PowerShell 用户可直接把 `python` 换成 `.\.venv\Scripts\python.exe`。

## 2. 选 CPU 或 GPU 运行

**普通电脑：CPU**

```bash
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-timesfm.txt
python main.py --device cpu
```

**有 NVIDIA 显卡：GPU**（需要支持 CUDA 12.8 的驱动）

```bash
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-timesfm.txt
python main.py --device gpu
```

CPU、GPU 使用**相同数据、相同模型**，只是运行设备不同；数值可能有细小差异。首次运行自动下载约 **1.32 GB** 模型，需要联网，无需训练。macOS 的 CPU 路径把第一条安装命令换为 `python -m pip install torch==2.8.0`。

## 3. 看结果

打开 `outputs/timesfm-cpu/result.png`（GPU 对应 `timesfm-gpu`）：

**一张去趋势后的图：过去 6 小时观测 → 起报虚线 → 未来 60 分钟预测。** 灰线是历史，蓝线是预测，黑色点线是未来真实值，仅用于比较；浅黄色区域为预测窗口。

趋势是**仅用历史数据拟合的直线**。绘图时，历史、未来真值和预测统一减去这条直线及其外推值，单位为 ns。这样能看清细小变化；这一步只影响显示，不改变模型输入、预测值或 RMSE，也不是模型微调。

![去趋势后的历史观测与未来预报](data/example.png)

同目录的 `prediction.csv` 有四列：时间、真实钟差、预测钟差、误差。终端会显示 RMSE，单位 ns，越小越好。再次运行会更新结果文件。

读代码时只跟着六个中文注释走：选设备、读数据、换单位、预测、算误差、画图。`truth` 是未来真实值，**只用于最后评分，不能输入模型**。

想先试一个不用大模型的版本：

```bash
python -m pip install -r requirements.txt
python main.py --model linear
```

它只做线性预报，运行在 CPU。跑通后，再运行上面的 CPU/GPU TimesFM 命令。

本例是一个固定窗口的入门实验，不代表全年精度。数据使用说明见 [data/README.md](data/README.md)；模型权重限教学研究等许可用途，见 [来源与许可](DATA_SOURCES.md)。完整多窗口实验放在 [advanced/](advanced/README.md)，入门时可以先不看。
