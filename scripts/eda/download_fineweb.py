#!/usr/bin/env python3
import argparse
import json
import os

from datasets import load_dataset


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="datasets/raw")
    p.add_argument("--shards", type=int, default=20)
    p.add_argument("--skip", type=int, default=0)
    p.add_argument("--target-bytes", type=int, default=None)
    p.add_argument("--n-docs", type=int, default=None)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    files = [open(os.path.join(args.out_dir, f"shard_{i:02d}.jsonl"), "a") for i in range(args.shards)]

    ds = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True)
    if args.skip:
        ds = ds.skip(args.skip)

    total_bytes = 0
    for i, row in enumerate(ds):
        if args.n_docs is not None and i >= args.n_docs:
            break
        if args.target_bytes is not None and total_bytes >= args.target_bytes:
            break
        line = json.dumps({"text": row["text"]}) + "\n"
        files[i % args.shards].write(line)
        total_bytes += len(line.encode("utf-8"))

    for f in files:
        f.close()


if __name__ == "__main__":
    main()
