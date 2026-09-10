# Ramanujan

Ramanujan is a portable neural network for intent-aware neural answer compression, running on [Forge](https://github.com/Aisuko/forge).

Demo: **[aisuko.github.io/ramanujan](https://aisuko.github.io/ramanujan/)**

Input: AI tool outputs plus the QA pairs a user selected. Output: one paragraph of 1–3 grounded sentences, with evidence references.

```sh
make test
make prepare request=examples/request.json
make download
make data limit=1000
```

Architecture, contracts and roadmap: [docs/architecture.md](docs/architecture.md).
