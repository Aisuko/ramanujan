#!/usr/bin/env python3
import argparse
import glob
import json
import os

from tokenizers import ByteLevelBPETokenizer


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", default="datasets/raw")
    p.add_argument("--out-dir", default="datasets/tokenizer")
    p.add_argument("--vocab-size", type=int, default=8191)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    txt_path = os.path.join(args.out_dir, "_train_corpus.txt")
    with open(txt_path, "w") as out:
        for shard in sorted(glob.glob(os.path.join(args.raw_dir, "*.jsonl"))):
            with open(shard) as f:
                for line in f:
                    out.write(json.loads(line)["text"].replace("\n", " ") + "\n")

    tok = ByteLevelBPETokenizer()
    tok.train([txt_path], vocab_size=args.vocab_size, min_frequency=2)
    tok.save(os.path.join(args.out_dir, "tokenizer.json"))
    os.remove(txt_path)


if __name__ == "__main__":
    main()
