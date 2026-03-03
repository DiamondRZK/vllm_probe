# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  vllm_probe — Docker Image                                                 ║
# ║                                                                             ║
# ║  WHY DOCKER AND NOT BARE METAL?                                             ║
# ║  ──────────────────────────────                                             ║
# ║  1. NETWORK NAMESPACE ISOLATION                                             ║
# ║     When we inject latency via `tc` (Traffic Control), it affects ONLY the  ║
# ║     container's virtual network stack. The host machine is never touched.   ║
# ║     On bare metal, a forgotten `tc` cleanup could cripple the server.       ║
# ║                                                                             ║
# ║  2. CUDA VERSION PINNING                                                    ║
# ║     vLLM performance varies significantly between CUDA versions (12.1 vs    ║
# ║     12.4 can differ by 15% on FlashAttention kernels). Docker pins the      ║
# ║     exact CUDA runtime, ensuring benchmark reproducibility.                 ║
# ║                                                                             ║
# ║  3. NCCL CONSISTENCY                                                        ║
# ║     NCCL (NVIDIA Collective Communication Library) handles GPU-to-GPU       ║
# ║     synchronization. Different NCCL versions use different AllReduce         ║
# ║     algorithms (ring vs tree). Pinning NCCL ensures consistent behavior.    ║
# ║                                                                             ║
# ║  4. PORTABLE ACROSS TOPOLOGIES                                              ║
# ║     Same image runs on:                                                     ║
# ║       • Single server 2× RTX 4000 (TP=2, local)                           ║
# ║       • DGX-like NVSwitch cluster (TP=8)                                   ║
# ║       • Cross-DC distributed setup (Santiago ↔ Eindhoven)                  ║
# ║       • CI/CD pipeline for regression testing                               ║
# ║                                                                             ║
# ║  IMAGE SIZE NOTE:                                                           ║
# ║  The base image (nvidia/cuda) is ~4GB. vLLM adds ~2GB. Total: ~6-8GB.     ║
# ║  This is unavoidable — CUDA runtime + cuBLAS + cuDNN are large.            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# ── Stage 1: Base with CUDA runtime ─────────────────────────────────────────
# We use the CUDA 12.4 devel image because vLLM compiles custom CUDA kernels
# (FlashAttention, PagedAttention) that need nvcc at build time.
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS base

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# ── System Dependencies ─────────────────────────────────────────────────────
# iproute2: Contains `tc` (Traffic Control) for network latency simulation
# pciutils: Contains `lspci` for PCIe topology inspection
# numactl:  For NUMA-aware process binding (important for multi-socket servers)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3-pip \
    iproute2 \
    pciutils \
    numactl \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Ensure python3 points to 3.10
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# ── Python Dependencies ─────────────────────────────────────────────────────
WORKDIR /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt

# ── Application Code ────────────────────────────────────────────────────────
COPY vllm_probe.py .
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh vllm_probe.py

# ── Runtime Configuration ───────────────────────────────────────────────────
# NCCL tuning for reproducible benchmarks
# NCCL_DEBUG=INFO: Log communication patterns (useful for debugging TP issues)
# NCCL_IB_DISABLE=1: Disable InfiniBand (we're simulating, not using real IB)
ENV NCCL_DEBUG=WARN \
    NCCL_IB_DISABLE=1 \
    PYTHONUNBUFFERED=1 \
    VLLM_LOGGING_LEVEL=WARNING

# Results directory (mount a volume here for persistence)
RUN mkdir -p /results
VOLUME ["/results"]

# ── Health Check ─────────────────────────────────────────────────────────────
# Verify CUDA is accessible from Python
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'" || exit 1

ENTRYPOINT ["./entrypoint.sh"]
CMD ["--help"]
