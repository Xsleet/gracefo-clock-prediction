"""从官方 Hugging Face 下载固定版本 TimesFM 3.0；约 1.32 GB。"""
import argparse
import hashlib
from pathlib import Path

from huggingface_hub import snapshot_download

from timesfm_backend import MODEL_REVISION, MODEL_SHA256


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "models/timesfm-3.0-pytorch")
    args = parser.parse_args()
    snapshot_download(
        repo_id="google/timesfm-3.0-pytorch", revision=MODEL_REVISION,
        local_dir=args.output, token=False, max_workers=2,
        allow_patterns=["config.json", "model.safetensors", "README.md", "LICENSE"],
    )
    with (args.output / "model.safetensors").open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != MODEL_SHA256:
            raise RuntimeError("Downloaded weights failed SHA-256 verification.")
    print(f"Verified weights saved to {args.output}")
