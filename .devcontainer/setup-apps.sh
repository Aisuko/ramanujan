#!/usr/bin/env bash
# Application-level setup for the Ramanujan devcontainer. Runs once, on create, last —
# after setup-system.sh. Everything here installs outside apt and is skipped if the
# tool is already on PATH.
set -euo pipefail

# --- 4. Claude Code ------------------------------------------------------------
command -v claude >/dev/null 2>&1 || curl -fsSL https://claude.ai/install.sh | bash -s stable

# Merge, rather than overwrite, so an existing ~/.claude/settings.json survives.
mkdir -p ~/.claude
python3 - <<'EOF'
import json, pathlib
p = pathlib.Path.home() / ".claude" / "settings.json"
s = json.loads(p.read_text()) if p.exists() else {}
s["theme"] = "auto"
p.write_text(json.dumps(s, indent=2) + "\n")
EOF

# wasm-pack builds crates/ramanujan-wasm and drives the headless-Chrome test runner.
# wasm-bindgen-cli is deliberately not installed here: its version must match the
# wasm-bindgen crate in Cargo.lock, so install it from the workspace once that exists
# (`cargo install --locked wasm-bindgen-cli --version <lockfile version>`).
command -v wasm-pack >/dev/null 2>&1 || cargo install --locked wasm-pack
