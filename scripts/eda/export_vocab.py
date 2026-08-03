#!/usr/bin/env python3
import argparse
import json
import os


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer-dir", default="datasets/tokenizer")
    args = p.parse_args()

    with open(os.path.join(args.tokenizer_dir, "tokenizer.json")) as f:
        model = json.load(f)["model"]

    with open(os.path.join(args.tokenizer_dir, "vocab.json"), "w") as f:
        json.dump(model["vocab"], f)

    merges = [m if isinstance(m, str) else " ".join(m) for m in model["merges"]]
    with open(os.path.join(args.tokenizer_dir, "merges.txt"), "w") as f:
        f.write("\n".join(merges) + "\n")


if __name__ == "__main__":
    main()
