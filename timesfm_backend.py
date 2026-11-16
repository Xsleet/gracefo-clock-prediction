"""可选 TimesFM 后端；核心基线不导入 PyTorch，也不联网。"""
import hashlib
import importlib.metadata
import time
from pathlib import Path

import numpy as np

MODEL_REVISION = "43046b85ec22d584a13f8098c2ed39c889e129c2"
MODEL_SHA256 = "a7592b0a8432baee54483254e5647856911ce69e09d09a9bb65904b2d98f17da"


def add_timesfm_predictions(windows, checkpoint, device="cpu"):
    """只向模型传历史数组，不传 truth；同时生成直接/残差两组预测。"""
    import torch
    from timesfm3 import ModelConfig, TimesFM3Evaluator

    checkpoint = Path(checkpoint)
    weight = checkpoint / "model.safetensors"
    if not weight.is_file():
        raise FileNotFoundError("Run download_model.py first, or supply --checkpoint.")
    with weight.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != MODEL_SHA256:
        raise ValueError("Checkpoint differs from the pinned teaching experiment.")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable; use --device cpu or install a matching PyTorch build.")
    # 本教学版本只复现实验的固定 60 min 输出（10 s 采样）。
    horizon = 360
    start = time.perf_counter()
    model = TimesFM3Evaluator(ModelConfig(
        checkpoint_path=str(checkpoint), per_core_batch_size=4,
        device=device, local_files_only=True))
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    load_seconds = time.perf_counter() - start
    start = time.perf_counter()
    jobs = [(i, kind) for i in range(len(windows))
            for kind in ["timesfm_direct", "timesfm_residual"]]
    for first in range(0, len(jobs), 4):
        batch = jobs[first:first + 4]
        contexts = [windows[i]["centered_history_ns" if kind == "timesfm_direct"
                               else "residual_history_ns"].astype(np.float32)
                    for i, kind in batch]
        with torch.inference_mode():
            outputs = list(model.predict_batch(
                contexts, horizon=horizon, return_quantiles=True,
                use_symmetric_averaging=False, make_positive=False, sort_quantiles=True))
        if len(outputs) != len(batch):
            raise ValueError("Unexpected number of model outputs.")
        for (i, kind), output in zip(batch, outputs):
            point = output.forecast.astype(np.float64)
            quantiles = output.quantiles.astype(np.float64)
            if point.shape != (horizon,) or quantiles.shape != (horizon, 9):
                raise ValueError("Unexpected forecast shape.")
            if not np.isfinite(point).all() or not np.isfinite(quantiles).all():
                raise ValueError("Non-finite model output.")
            np.testing.assert_allclose(point, quantiles[:, 4], atol=1e-6)
            if np.any(np.diff(quantiles, axis=-1) < -1e-6):
                raise ValueError("Unordered quantiles.")
            if kind == "timesfm_residual":
                point += windows[i]["future_trend_ns"]
                quantiles += windows[i]["future_trend_ns"][:, None]
            windows[i]["predictions"][kind] = point
            windows[i]["quantiles"][kind] = quantiles
        print(f"TimesFM forecasts: {min(first + 4, len(jobs))}/{len(jobs)}", flush=True)
    if device == "cuda":
        torch.cuda.synchronize()
    return {
        "model_revision": MODEL_REVISION, "weight_sha256": digest,
        "device": device, "load_seconds": load_seconds,
        "inference_seconds": time.perf_counter() - start,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else None,
        "packages": {name: importlib.metadata.version(name) for name in ["torch", "timesfm"]},
    }
