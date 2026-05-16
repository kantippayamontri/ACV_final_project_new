#!/usr/bin/env python3
# Run: HF_TOKEN=hf_xxxx uv run python scripts/download_llama32_1b.py
"""Download Llama-3.2-1B from HuggingFace.

Requires a HuggingFace token with access to gated models (https://huggingface.co/meta-llama/Llama-3.2-1B).
Set the HF_TOKEN environment variable or run `huggingface-cli login` first.

Usage:
    HF_TOKEN=hf_xxxx uv run python scripts/download_llama32_1b.py
    uv run python scripts/download_llama32_1b.py --output-dir /path/to/save
"""
import argparse
import os
import sys
from pathlib import Path

MODEL_ID = "meta-llama/Llama-3.2-1B"
DEFAULT_OUTPUT = "models/Llama-3.2-1B"


def parse_args():
    parser = argparse.ArgumentParser(description="Download Llama-3.2-1B from HuggingFace")
    parser.add_argument("--token", type=str, default=os.environ.get("HF_TOKEN", None),
                        help="HuggingFace token (or set HF_TOKEN env var)")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT,
                        help=f"Output directory (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--resume", action="store_true", default=True,
                        help="Resume partial downloads (default)")
    parser.add_argument("--force", action="store_true",
                        help="Force re-download even if files exist")
    return parser.parse_args()


def download_model(model_id: str, output_dir: Path, token: str | None, resume: bool):
    from huggingface_hub import snapshot_download

    output_dir = output_dir.resolve()

    if not output_dir.parent.exists():
        output_dir.parent.mkdir(parents=True, exist_ok=True)

    if output_dir.exists() and any(output_dir.iterdir()):
        if resume:
            print(f"Resuming download into {output_dir} (existing files will be skipped)...")
        else:
            print(f"Error: {output_dir} already exists and is not empty. Use --force to overwrite.")
            sys.exit(1)

    print(f"Downloading {model_id}...")
    snapshot_download(
        repo_id=model_id,
        local_dir=str(output_dir),
        token=token,
        resume_download=resume,
    )
    print(f"Download complete: {output_dir}")


def main():
    args = parse_args()

    project_root = Path(__file__).resolve().parent.parent
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir

    token = args.token
    if token is None:
        print("No HF_TOKEN found in environment. Attempting cached login...")
        from huggingface_hub import whoami
        try:
            whoami()
        except Exception:
            print("Error: You must provide a HuggingFace token (--token or HF_TOKEN env var)")
            print("Get one at: https://huggingface.co/settings/tokens")
            sys.exit(1)

    download_model(MODEL_ID, output_dir, token, args.resume)


if __name__ == "__main__":
    main()
