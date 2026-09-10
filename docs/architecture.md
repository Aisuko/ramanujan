# Ramanujan Architecture

Intent-aware neural answer compression. Ramanujan takes the outputs of one or more AI tools and the QA pairs a user picked in an external harness. It returns one paragraph of 1–3 sentences that is grounded in evidence from those outputs.

## Boundaries

| Component | Owns |
|---|---|
| Harness | Tool orchestration, user interaction, QA collection |
| Ramanujan | Input processing, intent conditioning, evidence selection, summary generation, training, evaluation |
| Forge | Tensors, kernels, generic encoder–decoder execution, FZM loading, CPU/WGPU/WebGPU |

Forge contains no evidence-selection or summarisation logic. Ramanujan never runs its own tensor engine.

## QESC Model

Query-Guided Evidence Selection and Compression:

```text
tool outputs ─► segmentation ─► source encoder ─┐
selected QA  ─► intent encoder ─────────────────┴─► fusion ─► evidence selector ─► redundancy ─► top-k ─► decoder ─► summary
```

1. **Source encoder**: encodes the tool outputs and produces a representation for each sentence.
2. **Intent encoder**: encodes the selected QA pairs and keeps them apart from unselected ones.
3. **Evidence selector**: `relevance = MLP(sentence, intent)`.
4. **Redundancy**: penalises candidates that are close to evidence already chosen.
5. **Decoder**: autoregressive, with cross-attention over the selected evidence and the intent.

| Setting | Value |
|---|---|
| Encoder / decoder layers | 6 / 4 |
| Hidden / heads / FFN | 384 / 6 / 1536 |
| Vocabulary | 32K byte-level BPE |
| Max input / output | 2048 / 96 tokens |
| Candidates / selected | ≤64 / 3–8 sentences |
| Output | 1–3 sentences |

Weights are trained from scratch in Rust on Forge. The config lives in `src/model/config.rs`.

## Contracts

`src/core`:

- `ToolOutput { source_id, content }`
- `Sentence { source_id, index, start, end, text }`: `index` is the candidate index across all sources, and `start..end` are byte offsets into the source `content`.
- `QaPair { question_id?, question, answer, selected }`
- `Intent { selected, unselected }`
- `Evidence { source_id, text_span, sentence_index?, start?, end?, score? }`
- `Constraints { min_sentences = 1, max_sentences = 3, style = "concise", preserve_factuality = true }`

Segmentation is rule-based: it splits on terminal punctuation and newlines and handles abbreviations, initials and decimals. Candidates beyond 64 are truncated, and the truncation is reported.

## Harness

Request:

```json
{
  "request_id": "req-001",
  "task": "focused_summarisation",
  "tool_outputs": [{ "source_id": "search-001", "content": "..." }],
  "selected_qa": [{ "question_id": "q-001", "question": "What deployment do you prefer?", "answer": "Self-hosted" }],
  "constraints": { "min_sentences": 1, "max_sentences": 3, "style": "concise", "preserve_factuality": true },
  "previous_request_id": null
}
```

Response:

```json
{
  "request_id": "req-001",
  "summary": "Gemma is a suitable candidate for self-hosted deployment.",
  "sentence_count": 1,
  "evidence": [{ "source_id": "search-001", "text_span": "..." }],
  "metrics": { "selected_evidence": 1, "input_tokens": 1200, "output_tokens": 12 }
}
```

Refinement re-sends the same tool outputs with new QA selections and `previous_request_id`. Source encodings are cached under a key made of source content hash, model version, tokenizer version and preprocessing config. Intent and evidence are always recomputed.

## Dataset

`RamanujanExample`, one JSON object per line:

```json
{
  "id": "qmsum-train-0-2",
  "tool_outputs": [{ "source_id": "turns-40-45", "content": "..." }],
  "qa_pairs": [{ "question_id": "q-2", "question": "What do you want to know?", "answer": "<query>", "selected": true }],
  "evidence_labels": [3, 4, 9],
  "target_summary": "...",
  "provenance": { "dataset": "qmsum", "group": "train-0" }
}
```

| Source | Conversion |
|---|---|
| QMSum | Relevant turns plus context, split into sources of ≤6 turns. Each specific query becomes a QA pair, with one selected per example. Evidence comes from `relevant_text_span`. Queries from the same meeting share a `group`, which feeds refinement training. |
| CNN/DailyMail | Article split into sources of 8 sentences. One generic QA pair. Highlights (≤3) become the target. Evidence comes from a greedy token-overlap oracle. |
| QuerySum, Ramanujan-QA-Selection | Planned. |

## Training

| Stage | Goal |
|---|---|
| A | Adapt the encoder to tool-output text and QA conditioning |
| B | Evidence selector: relevance and redundancy |
| C | Conditional decoder that keeps to 1–3 sentences |
| D | Joint: `generation + λ1·evidence + λ2·faithfulness` |
| E | Progressive refinement: same sources, different selections |

## Evaluation

ROUGE-L, BERTScore, evidence precision/recall, factual consistency, redundancy rate, intent alignment, compression ratio, latency, peak memory.

**Intent Sensitivity Score**: keep the tool outputs fixed, change only the selected QA pairs, and measure whether relevant facts change, irrelevant ones drop, still-relevant facts stay, and every statement stays grounded.

## FZM v2

A single file: `FZM2` magic, a JSON manifest (`format, version, architecture, task, model, components, quantization, config, provenance`), named assets (tokenizer files), then named tensors stored as q4, f32 or bf16. The loader dispatches on the magic, and v1 `FZM1` still loads. Safetensors is the interchange format; FZM is the deployment format.

## Milestones

| Ramanujan | | Forge | |
|---|---|---|---|
| M0 | Spec, contracts, schemas | F0 | FZM v2 manifest |
| M1 | Input and intent pipeline | F1 | Encoder–decoder execution |
| M2 | Evidence selection | F2 | Cross-attention |
| M3 | Conditional generation | F3 | Segment / positional embeddings |
| M4 | Training pipeline | F4 | Classification and top-k |
| M5 | Forge and FZM integration | F5 | Encoder output caching |
| M6 | Harness integration | F6 | Packed q4 kernels |
| M7 | Release and benchmarks | F7–F8 | WGPU parity, WebGPU inference |

Done so far: M0, the M1 data structures, F0, F1, F2, and the training-data converter.
