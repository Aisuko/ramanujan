#!/usr/bin/env bash
# System-level setup for the Ramanujan devcontainer. Runs once, on create, before
# setup-apps.sh. Three steps, in this order:
#   1. Update the OS.
#   2. Vulkan: Mesa's lavapipe (CPU Vulkan, so wgpu/Dawn come up with no GPU attached)
#      plus the NVIDIA ICD wiring, which needs the loader from step 1 in place.
#   3. tmux: install, configure, and start the `harnesses` session used for the
#      Claude Code / CodeX terminals.
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive

# --- 1. OS ---------------------------------------------------------------------
sudo apt-get update
sudo apt-get upgrade -y

# --- 2. Vulkan -----------------------------------------------------------------
# lavapipe is slow — a correctness target for .fzm checkpoints and WGSL kernels,
# not a place to measure references/PLAN.md §2 latencies.
sudo apt-get install -y --no-install-recommends \
	libvulkan1 \
	mesa-vulkan-drivers \
	vulkan-tools \
	libegl1 \
	libgl1 \
	chromium \
	chromium-driver \
	pkg-config

# No-op on a host without an NVIDIA GPU; also re-run on every container start.
bash scripts/devcontainer/setup_nvidia_vulkan.sh || true

echo
echo "== Vulkan adapters visible to wgpu/Dawn =="
vulkaninfo --summary 2>/dev/null | sed -n '/Devices:/,/^$/p' || echo "vulkaninfo failed — WebGPU will not come up"

# --- 3. tmux -------------------------------------------------------------------
sudo apt-get install -y --no-install-recommends tmux
sudo rm -rf /var/lib/apt/lists/*

TMUX_CONF=~/.tmux.conf
if ! grep -q '^# >>> devcontainer tmux (claude) >>>' "$TMUX_CONF" 2>/dev/null; then
	cat >>"$TMUX_CONF" <<'EOF'
# >>> devcontainer tmux (claude) >>>
set -g mouse on
set -g history-limit 200000
set -g default-terminal "tmux-256color"
set -ag terminal-overrides ",xterm-256color:RGB,*256col*:RGB"
set -s escape-time 0
set -g focus-events on
set -g set-clipboard on
set -g allow-passthrough on
set -g base-index 1
setw -g pane-base-index 1
setw -g mode-keys vi
setw -g aggressive-resize on
set -g status-interval 5
# <<< devcontainer tmux (claude) <<<
EOF
fi

tmux has-session -t harnesses 2>/dev/null || tmux new-session -d -s harnesses -c "$PWD"
tmux source-file "$TMUX_CONF"
