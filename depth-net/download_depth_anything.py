#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download Depth Anything V2 to a local directory for offline training.

Run once before training:
    python download_depth_anything.py

After this succeeds, train.py loads the model from utils.DPT_LOCAL_MODEL_DIR
with local_files_only=True, so training will not contact HuggingFace.
"""

import argparse
import os
from pathlib import Path

import utils


def parse_args():
    parser = argparse.ArgumentParser(description="Download Depth Anything model locally")
    parser.add_argument("--model-name", default=utils.DPT_MODEL_NAME, help="HuggingFace model name")
    parser.add_argument(
        "--output-dir",
        default=utils.DPT_LOCAL_MODEL_DIR,
        help="Local directory where the model will be saved",
    )
    parser.add_argument(
        "--endpoint",
        default=getattr(utils, "DPT_DOWNLOAD_ENDPOINT", "https://hf-mirror.com"),
        help="HuggingFace endpoint or mirror, e.g. https://hf-mirror.com",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.endpoint:
        os.environ["HF_ENDPOINT"] = args.endpoint

    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading processor and model: {args.model_name}")
    print(f"Using HuggingFace endpoint: {os.environ.get('HF_ENDPOINT', 'https://huggingface.co')}")
    print(f"Saving to: {output_dir}")

    processor = AutoImageProcessor.from_pretrained(args.model_name)
    model = AutoModelForDepthEstimation.from_pretrained(args.model_name)

    processor.save_pretrained(output_dir)
    model.save_pretrained(output_dir)

    print("Done. You can now run training with local Depth Anything files.")


if __name__ == "__main__":
    main()

