#!/usr/bin/env python3

import argparse
import json
import os
import urllib.request

QMSUM = "https://raw.githubusercontent.com/Yale-LILY/QMSum/main/data/ALL/jsonl/{split}.jsonl"
SPLITS = {"train": "train", "val": "validation", "test": "test"}


def qmsum(out_dir: str) -> None:
    for split in SPLITS:
        path = os.path.join(out_dir, f"qmsum_{split}.jsonl")
        urllib.request.urlretrieve(QMSUM.format(split=split), path)
        print(path)


def cnndm(out_dir: str, limit: int) -> None:
    from datasets import load_dataset

    for split, hf_split in SPLITS.items():
        path = os.path.join(out_dir, f"cnndm_{split}.jsonl")
        rows = load_dataset("abisee/cnn_dailymail", "3.0.0", split=hf_split)
        with open(path, "w") as out:
            for i, row in enumerate(rows):
                if limit and i >= limit:
                    break
                out.write(json.dumps({"id": row["id"], "article": row["article"], "highlights": row["highlights"]}) + "\n")
        print(path)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["qmsum", "cnndm"], required=True)
    p.add_argument("--out-dir", default="datasets/raw")
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    if a.dataset == "qmsum":
        qmsum(a.out_dir)
    else:
        cnndm(a.out_dir, a.limit)


if __name__ == "__main__":
    main()
