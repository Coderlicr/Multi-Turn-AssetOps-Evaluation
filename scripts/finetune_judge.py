#!/usr/bin/env python3
"""
Submit a judge fine-tuning job to OpenAI and wait until it finishes.

This wraps the official ``openai`` Python SDK so you don't have to write the
upload/poll boilerplate yourself.

Run from repo root:

    python scripts/finetune_judge.py \
        --train ./judge_train.jsonl \
        --val ./judge_train.val.jsonl \
        --base-model gpt-4o-mini-2024-07-18 \
        --suffix dialog-judge-v1

When the job succeeds, the script prints the fine-tuned model id like
``ft:gpt-4o-mini-2024-07-18:my-org:dialog-judge-v1:abc123`` which you can
plug directly into ``python main.py llm-judge --judge openai --model <id>``.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


def _human_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.1f}{unit}"
        size /= 1024.0
    return f"{num_bytes}B"


def main() -> int:
    parser = argparse.ArgumentParser(description="Fine-tune an OpenAI judge model from JSONL files.")
    parser.add_argument("--train", required=True, type=str, help="Training JSONL path")
    parser.add_argument("--val", default=None, type=str, help="Validation JSONL path (optional but recommended)")
    parser.add_argument(
        "--base-model",
        default="gpt-4o-mini-2024-07-18",
        dest="base_model",
        help="Base model that supports fine-tuning (default: gpt-4o-mini-2024-07-18)",
    )
    parser.add_argument(
        "--suffix",
        default="dialog-judge-v1",
        help="Suffix tag added to the fine-tuned model id (3-18 chars)",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=30,
        dest="poll_interval",
        help="Seconds between status polls (default: 30)",
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        dest="no_wait",
        help="Submit the job and exit without polling (you can check status later)",
    )
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("error: OPENAI_API_KEY is not set in your environment.", file=sys.stderr)
        return 2

    try:
        from openai import OpenAI, OpenAIError
    except ImportError:
        print("error: install the openai SDK first:  pip install 'openai>=1.40'", file=sys.stderr)
        return 2

    client = OpenAI()

    train_path = Path(args.train)
    if not train_path.is_file():
        print(f"error: training file not found: {train_path}", file=sys.stderr)
        return 2

    print(f"[1/3] Uploading training file: {train_path} ({_human_size(train_path.stat().st_size)})")
    try:
        with train_path.open("rb") as f:
            train_file = client.files.create(file=f, purpose="fine-tune")
    except OpenAIError as exc:
        print(f"error: upload failed: {exc}", file=sys.stderr)
        return 2
    print(f"      -> training file id: {train_file.id}")

    val_file_id: str | None = None
    if args.val:
        val_path = Path(args.val)
        if not val_path.is_file():
            print(f"error: validation file not found: {val_path}", file=sys.stderr)
            return 2
        print(f"[2/3] Uploading validation file: {val_path} ({_human_size(val_path.stat().st_size)})")
        try:
            with val_path.open("rb") as f:
                val_file = client.files.create(file=f, purpose="fine-tune")
        except OpenAIError as exc:
            print(f"error: validation upload failed: {exc}", file=sys.stderr)
            return 2
        val_file_id = val_file.id
        print(f"      -> validation file id: {val_file_id}")
    else:
        print("[2/3] No validation file provided, skipping.")

    create_kwargs: dict[str, str] = {
        "training_file": train_file.id,
        "model": args.base_model,
        "suffix": args.suffix,
    }
    if val_file_id:
        create_kwargs["validation_file"] = val_file_id

    print(f"[3/3] Creating fine-tuning job (base={args.base_model}, suffix={args.suffix}) ...")
    try:
        job = client.fine_tuning.jobs.create(**create_kwargs)
    except OpenAIError as exc:
        print(f"error: job creation failed: {exc}", file=sys.stderr)
        return 2
    print(f"      -> job id: {job.id}")
    print(f"      -> initial status: {job.status}")

    if args.no_wait:
        print()
        print("--no-wait set; exiting now.")
        print(f"  Check status:  python -c \"from openai import OpenAI; print(OpenAI().fine_tuning.jobs.retrieve('{job.id}'))\"")
        return 0

    print()
    print(f"Polling every {args.poll_interval}s. Typical jobs take 10-60 minutes.")
    last_status = job.status
    print(f"[{time.strftime('%H:%M:%S')}] status: {last_status}")
    while True:
        try:
            job = client.fine_tuning.jobs.retrieve(job.id)
        except OpenAIError as exc:
            print(f"warning: poll failed, will retry: {exc}", file=sys.stderr)
            time.sleep(args.poll_interval)
            continue
        if job.status != last_status:
            print(f"[{time.strftime('%H:%M:%S')}] status: {job.status}")
            last_status = job.status
        if job.status in ("succeeded", "failed", "cancelled"):
            break
        time.sleep(args.poll_interval)

    if job.status == "succeeded":
        print()
        print("Fine-tuning succeeded.")
        print(f"  fine_tuned_model: {job.fine_tuned_model}")
        print()
        print("Use it as a judge from this repo:")
        print()
        print(
            f"  python main.py llm-judge --input ./data --judge openai --model {job.fine_tuned_model}"
        )
        return 0

    print(f"error: job ended with status={job.status}", file=sys.stderr)
    if getattr(job, "error", None):
        print(f"  details: {job.error}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
