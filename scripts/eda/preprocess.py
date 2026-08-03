#!/usr/bin/env python3
import argparse
import glob
import json
import multiprocessing as mp
import os
import struct

from tokenizers import Tokenizer

TOKENIZER_PATH = None


def _init(tokenizer_path: str) -> None:
    global TOKENIZER_PATH
    TOKENIZER_PATH = tokenizer_path


def _encode_shard(args: tuple[str, str]) -> str:
    shard_path, out_path = args
    tok = Tokenizer.from_file(TOKENIZER_PATH)
    with open(shard_path) as f, open(out_path, "wb") as out:
        for line in f:
            ids = tok.encode(json.loads(line)["text"]).ids
            out.write(struct.pack(f"<{len(ids)}I", *ids))
    return out_path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", default="datasets/raw")
    p.add_argument("--tokenizer", default="datasets/tokenizer/tokenizer.json")
    p.add_argument("--out", default="datasets/tokens/train.bin")
    p.add_argument("--jobs", type=int, default=min(20, os.cpu_count() or 1))
    args = p.parse_args()

    tmp_dir = os.path.join(os.path.dirname(args.out), "_shards")
    os.makedirs(tmp_dir, exist_ok=True)
    shards = sorted(glob.glob(os.path.join(args.raw_dir, "*.jsonl")))
    jobs = [(s, os.path.join(tmp_dir, os.path.basename(s) + ".bin")) for s in shards]

    with mp.Pool(min(args.jobs, len(jobs)), initializer=_init, initargs=(args.tokenizer,)) as pool:
        out_paths = pool.map(_encode_shard, jobs)

    with open(args.out, "wb") as out:
        for path in out_paths:
            with open(path, "rb") as f:
                out.write(f.read())
            os.remove(path)
    os.rmdir(tmp_dir)


if __name__ == "__main__":
    main()
