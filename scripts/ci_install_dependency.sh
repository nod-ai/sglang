#!/bin/bash
set -euxo pipefail

# Install the dependency in CI.
set -euxo pipefail

# Use repo from environment variable, passed from GitHub Actions
FLASHINFER_REPO="${FLASHINFER_REPO:-https://flashinfer.ai/whl/cu124/torch2.4/flashinfer}"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
bash "${SCRIPT_DIR}/killall_sglang.sh"

# Update pip
pip install --upgrade pip
pip install -e "python[all]" --find-links https://flashinfer.ai/whl/cu124/torch2.4/flashinfer/

# Force reinstall flashinfer and torch_memory_saver
pip install flashinfer==0.1.6 --find-links ${FLASHINFER_REPO} --force-reinstall --no-deps
pip install torch_memory_saver --force-reinstall

# Install the main package
pip install -e "python[dev]"

# Install additional dependencies
pip install mooncake-transfer-engine==0.3.2.post1 nvidia-cuda-nvrtc-cu12

# For compling xgrammar kernels
pip install cuda-python nvidia-cuda-nvrtc-cu12

# reinstall sgl-kernel
pip install sgl-kernel --force-reinstall --no-deps
