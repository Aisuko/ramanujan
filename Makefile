.PHONY: test training chat site

pre-processed-dataset ?=
smoke ?= no
batch_size ?= 32
seq_len ?= 256
epoch ?= 1
lr ?= 1e-3
weight_decay ?= 0.01
mask_prob ?= 0.30
seed ?= 42
warmup ?= 2000
max_steps ?= 98000
clip ?= 1.0
ckpt_every ?= 2000
ckpt_path ?= checkpoints/react_6m.fzm
device ?= gpu
ckpt ?= checkpoints/react_6m.fzm
tokenizer ?= datasets/tokenizer
topk ?= 5
gen ?= 16
port ?= 8000

test:
	cargo test

training:
	cargo run --release --bin train -- \
		--dataset "$(pre-processed-dataset)" \
		--smoke $(smoke) \
		--batch-size $(batch_size) \
		--seq-len $(seq_len) \
		--epochs $(epoch) \
		--lr $(lr) \
		--weight-decay $(weight_decay) \
		--mask-prob $(mask_prob) \
		--seed $(seed) \
		--warmup $(warmup) \
		--max-steps $(max_steps) \
		--clip $(clip) \
		--ckpt-every $(ckpt_every) \
		--ckpt-path $(ckpt_path) \
		--device $(device)

chat:
	cargo run --release --bin chat -- \
		--ckpt "$(ckpt)" \
		--tokenizer $(tokenizer) \
		--topk $(topk) \
		--gen $(gen) \
		--device $(device)

site:
	python3 -m http.server $(port) --directory site
