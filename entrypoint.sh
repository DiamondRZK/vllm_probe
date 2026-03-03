#!/bin/bash
# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  vllm_probe entrypoint — Pre-flight checks before benchmarking         ║
# ╚══════════════════════════════════════════════════════════════════════════╝
set -euo pipefail

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  vllm_probe v1.0.0                                      ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# ── Pre-flight: CUDA ────────────────────────────────────────────────────
echo "┌─ Pre-flight Checks"
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)
    GPU_NAMES=$(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' ', ' | sed 's/, $//')
    echo "│  ✓ CUDA available: ${GPU_COUNT} GPU(s) — ${GPU_NAMES}"

    # Check NVLink status via topology matrix (definitive method)
    # nvidia-smi topo -m shows "NV#" for NVLink, "PHB"/"PIX" for PCIe
    TOPO_OUTPUT=$(nvidia-smi topo -m 2>&1)
    if echo "$TOPO_OUTPUT" | grep -P "GPU\d.*\tNV\d" &> /dev/null; then
        echo "│  ✓ NVLink: DETECTED — high-speed GPU interconnect available"
    elif echo "$TOPO_OUTPUT" | grep -qE "PHB|PIX|SYS|NODE" &> /dev/null; then
        echo "│  ⓘ NVLink: NOT AVAILABLE — GPUs communicate via PCIe only"
    else
        echo "│  ⓘ NVLink: UNKNOWN — topology:"
        echo "$TOPO_OUTPUT" | head -5 | sed 's/^/│    /'
    fi
else
    echo "│  ✗ nvidia-smi not found — GPU benchmarks will fail"
    echo "│    Ensure --gpus all is passed to docker run"
fi

# ── Pre-flight: tc (Traffic Control) ────────────────────────────────────
if tc qdisc show dev lo &> /dev/null; then
    echo "│  ✓ Traffic Control (tc) available — latency injection ready"
else
    echo "│  ⚠ Traffic Control requires NET_ADMIN capability"
    echo "│    Use: docker run --cap-add=NET_ADMIN ..."
fi

# ── Pre-flight: Python packages ─────────────────────────────────────────
VLLM_VERSION=$(python3 -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo "NOT INSTALLED")
TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null || echo "NOT INSTALLED")
NCCL_VERSION=$(python3 -c "import torch; print(torch.cuda.nccl.version())" 2>/dev/null || echo "N/A")

echo "│  Python packages:"
echo "│    vLLM:  ${VLLM_VERSION}"
echo "│    Torch: ${TORCH_VERSION}"
echo "│    NCCL:  ${NCCL_VERSION}"
echo "└──────────────────────────────────────────────────────────"
echo ""

# ── Execute vllm_probe with all passed arguments ───────────────────────
exec python3 /app/vllm_probe.py "$@"
