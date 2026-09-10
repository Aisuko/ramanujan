.PHONY: test prepare download data

raw_dir ?= datasets/raw
out_dir ?= datasets/examples
request ?= examples/request.json
limit ?=

test:
	cargo test

prepare:
	cargo run --release --bin ramanujan -- prepare < $(request)

download:
	python3 scripts/download.py --dataset qmsum --out-dir $(raw_dir)
	python3 scripts/download.py --dataset cnndm --out-dir $(raw_dir)

data:
	mkdir -p $(out_dir)
	for split in train val test; do \
		for ds in qmsum cnndm; do \
			cargo run --release --bin ramanujan -- convert --dataset $$ds \
				--src $(raw_dir)/$${ds}_$$split.jsonl --out $(out_dir)/$${ds}_$$split.jsonl \
				$(if $(limit),--limit $(limit),) || exit 1; \
		done; \
	done
