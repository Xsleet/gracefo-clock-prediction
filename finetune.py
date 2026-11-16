"""教学微调：冻结 TimesFM 主体，只更新最后的预测头，按验证集误差保存权重。"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parent
REVISION = "43046b85ec22d584a13f8098c2ed39c889e129c2"
HISTORY, HORIZON = 2160, 360
EPOCH = datetime(2000, 1, 1, 12)


def make_windows(data, first_date, end_date, stride):
    """先按日期隔离，再切窗口；输入和标签都必须在该日期范围内。"""
    start = (datetime.fromisoformat(first_date) - EPOCH).total_seconds()
    stop = (datetime.fromisoformat(end_date) - EPOCH).total_seconds()
    data = data[(data[:, 0] >= start) & (data[:, 0] < stop)]
    x, y, origins = [], [], []
    for origin in range(int(start) + 6 * 3600, int(stop), stride):
        expected = origin + np.arange(1 - HISTORY, HORIZON + 1) * 10
        # 不跨日，避免把日产品间的基准差异当成训练信号。
        if (expected[0] + 43200) // 86400 != (expected[-1] + 43200) // 86400:
            continue
        i = np.searchsorted(data[:, 0], expected[0])
        window = data[i:i + HISTORY + HORIZON]
        if not np.array_equal(window[:, 0], expected) or not np.isfinite(window).all():
            continue
        centered = (window[:, 1] - window[HISTORY - 1, 1]) * 1e9
        x.append(centered[:HISTORY])
        y.append(centered[HISTORY:])
        origins.append(origin)
    if not x:
        raise ValueError("没有连续完整的训练/验证窗口")
    return np.array(x, dtype=np.float32), np.array(y), np.array(origins)


def frozen_fingerprint(model):
    """独立检查：训练前后，预测头以外的权重必须完全相同。"""
    digest = hashlib.sha256()
    for name, value in model.named_parameters():
        if not name.startswith("output_head."):
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main():
    # 1. 设备、固定训练预算；不以展示窗口效果调整参数。
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--checkpoint", default="google/timesfm-3.0-pytorch")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/finetune")
    args = parser.parse_args()
    import torch
    from timesfm3 import ModelConfig, TimesFM3Evaluator
    if args.steps < 1:
        parser.error("--steps 必须大于 0")
    if args.device == "gpu" and not torch.cuda.is_available():
        parser.error("没有可用的 NVIDIA GPU，请使用 --device cpu")
    torch.manual_seed(2026)
    rng = np.random.default_rng(2026)
    device = "cuda" if args.device == "gpu" else "cpu"
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    # 2. 只读 5 月数据；训练不读取 main.py 的 6 月展示窗口。
    source = ROOT / "data/finetune.csv"
    data = np.loadtxt(source, delimiter=",", skiprows=1)
    train_x, train_y, train_origins = make_windows(data, "2025-05-27", "2025-05-31", 3600)
    val_x, val_y, val_origins = make_windows(data, "2025-05-31", "2025-06-01", 7200)
    print(f"训练 {len(train_x)} 个窗口；验证 {len(val_x)} 个窗口；设备 {args.device}", flush=True)

    # 3. 只让 output_head 的权重和偏置参与优化，其余约 3.3 亿参数保持冻结。
    evaluator = TimesFM3Evaluator(ModelConfig(checkpoint_path=args.checkpoint,
        revision=REVISION, device=device, per_core_batch_size=1))
    model = evaluator.model
    model.requires_grad_(False)
    model.output_head.requires_grad_(True)
    before = frozen_fingerprint(model)
    original_head = {k: v.detach().cpu().clone() for k, v in model.output_head.state_dict().items()}
    optimizer = torch.optim.Adam(model.output_head.parameters(), lr=1e-5)

    def predict(history):
        tensor = torch.as_tensor(history, dtype=torch.float32, device=device)
        # 官方 decode 带 no_grad；仅训练时去掉该包装，保持原有计算过程。
        quantiles = type(model).decode.__wrapped__(model, target=tensor[:, None], horizon=HORIZON)
        return quantiles[:, 0].sort(dim=-1).values[:, :, 4]  # 与原始推理相同的中位数

    def validation_rmse():
        model.eval()
        with torch.no_grad():
            predictions = np.concatenate([predict(row[None]).cpu().numpy() for row in val_x])
        return float(np.sqrt(np.mean((predictions - val_y)**2)))

    # 4. 用训练标签反向传播；只按独立验证集挑选最佳训练步。
    start_time = time.perf_counter()
    baseline = validation_rmse()
    # 确认训练用的可导计算，与 main.py 的官方预测接口完全对齐。
    official = list(evaluator.predict_batch([val_x[0]], horizon=HORIZON,
                    return_quantiles=False, make_positive=False, use_symmetric_averaging=False))[0].forecast
    with torch.no_grad():
        np.testing.assert_array_equal(predict(val_x[:1]).cpu().numpy()[0], official)
    records = [{"step": 0, "train_mse_ns2": None, "validation_rmse_ns": baseline}]
    best, best_step = np.inf, None
    metadata = {"method": "output_head_finetuning", "revision": REVISION,
        "history_points": HISTORY, "horizon_points": HORIZON, "step_seconds": 10,
        "seed": 2026, "learning_rate": 1e-5, "batch_size": 4, "steps": args.steps,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "data_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "train_origins": train_origins.tolist(), "validation_origins": val_origins.tolist(),
        "last_training_or_validation_epoch": int(val_origins.max() + HORIZON * 10),
        "selection": "Minimum validation RMSE among trained checkpoints; demonstration never read",
        "baseline_validation_rmse_ns": baseline}
    print(f"原始模型验证 RMSE = {baseline:.4f} ns", flush=True)
    for step in range(1, args.steps + 1):
        model.train()
        indices = rng.choice(len(train_x), size=4, replace=False)
        target = torch.as_tensor(train_y[indices], dtype=torch.float64, device=device)
        optimizer.zero_grad(set_to_none=True)
        loss = ((predict(train_x[indices]).double() - target)**2).mean()
        if not torch.isfinite(loss):
            raise ValueError("训练损失出现无效数值")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.output_head.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if step % 20 == 0 or step == args.steps:
            score = validation_rmse()
            records.append({"step": step, "train_mse_ns2": float(loss.detach()), "validation_rmse_ns": score})
            if score < best:
                best, best_step = score, step
                state = {k: v.detach().cpu().clone() for k, v in model.output_head.state_dict().items()}
                torch.save({"head": state, "metadata": {**metadata, "best_step": step,
                            "best_validation_rmse_ns": score}}, output / "head.pt")
            print(f"第 {step}/{args.steps} 步：验证 RMSE {score:.4f} ns；最佳步骤 {best_step}", flush=True)

    # 5. 核对基座未变、预测头确实更新，并重载保存的结果验证。
    assert frozen_fingerprint(model) == before, "冻结权重发生变化"
    saved = torch.load(output / "head.pt", map_location=device, weights_only=True)
    assert any(not torch.equal(saved["head"][k].cpu(), v) for k, v in original_head.items())
    model.output_head.load_state_dict(saved["head"])
    assert abs(validation_rmse() - best) < 1e-6, "权重重载结果不一致"
    report = {**saved["metadata"], "frozen_weights_unchanged": True,
        "head_changed": True, "reload_verified": True, "official_api_parity_verified": True, "history": records,
        "device": args.device, "torch_version": str(torch.__version__),
        "elapsed_seconds": time.perf_counter() - start_time}
    (output / "training.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f'已保存 {output / "head.pt"}；运行 main.py --device {args.device} --finetuned "{output / "head.pt"}"', flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    main()
