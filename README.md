# vllm_probe

**LLM Inference Benchmarking Suite with Physics-Based Network Simulation**

A standardized tool for measuring vLLM inference performance under simulated distributed conditions. Injects real network latency at the Linux kernel level to empirically prove whether Tensor Parallelism is viable across different infrastructure topologies — without needing multiple physical locations.

## Key Results

| Metric | TP=2 Local | TP=2 Cross-Continent (100ms) | Change |
|--------|-----------|------------------------------|--------|
| TPOT | 11.76 ms | 115.82 ms | **9.8× slower** |
| Throughput | 76.56 tok/s | 7.77 tok/s | **9.9× lower** |
| Total (2K tokens) | 26.75 s | 263.44 s (4m 23s) | **9.8× longer** |

**Conclusion:** Tensor Parallelism across Santiago ↔ Eindhoven (~200ms RTT) is architecturally infeasible for real-time inference. Each of the 2,048 decode steps pays the full network round-trip cost.

## Hardware Tested

- 2× NVIDIA RTX 4000 Ada Generation (20 GB VRAM each)
- PCIe Host Bridge (PHB) — No NVLink
- vLLM 0.15.1 | PyTorch 2.9.1+cu128 | NCCL 2.27.5

## How It Works

### Three-Layer Simulation

```
Layer 1: GPU Interconnect (NCCL env vars)
  NVLink ↔ PCIe fallback via NCCL_P2P_DISABLE

Layer 2: Network Latency (Linux tc netem)
  Kernel-level packet delay on loopback interface

Layer 3: Workload Scenarios (analyst/creative/chatbot)
  Isolate prefill vs decode vs scheduler bottlenecks
```

### Critical Discovery: NCCL Transport Bypass

Initial tests showed **zero effect** from `tc netem` latency injection. NCCL uses direct PCIe P2P transfers that bypass the Linux network stack entirely. The fix:

```bash
NCCL_P2P_DISABLE=1    # Disable direct GPU memory access
NCCL_SHM_DISABLE=1    # Disable shared memory transport  
NCCL_NET=Socket        # Force TCP socket communication
```

This forces NCCL through TCP sockets where `tc netem` can intercept packets. These variables are set automatically when latency injection is active.

## Quick Start

```bash
git clone https://github.com/DiamondRZK/vllm_probe.git
cd vllm_probe
docker compose build

# Test 1: Analyst baseline (TP=1, ~3 min)
docker compose run --rm probe-local --scenario analyst --tp 1 --rounds 3 -o /results

# Test 2: Creative decode stress (TP=2, ~3 min)
docker compose run --rm probe-local --scenario creative --tp 2 --rounds 3 -o /results

# Test 3: Cross-continent simulation (TP=2, ~25 min)
docker compose run --rm probe-local \
    --scenario creative --tp 2 --network cross-continent --rounds 3 -o /results

# Test 4: Production load (TP=2, 32 users, ~2 min)
docker compose run --rm probe-local --scenario chatbot --tp 2 --rounds 3 -o /results
```

## Workload Scenarios

| Scenario | Input | Output | Concurrency | Bottleneck | Network Sensitivity |
|----------|-------|--------|-------------|------------|-------------------|
| **analyst** | 8,192 | 128 | 1 | GPU TFLOPS (Compute) | Low |
| **creative** | 64 | 2,048 | 1 | Memory BW + Interconnect | **Critical** |
| **chatbot** | 512 | 512 | 32 | Scheduler + VRAM | Moderate |

## Network Profiles

| Profile | One-way Delay | Real-world Equivalent |
|---------|--------------|----------------------|
| `local` | 0 ms | Same rack / NVLink domain |
| `datacenter` | 1 ms ± 0.1 ms | Same building, different racks |
| `regional` | 15 ms ± 2 ms | Santiago ↔ Buenos Aires (~1,500 km) |
| `cross-continent` | 100 ms ± 10 ms | Santiago ↔ Eindhoven (~12,000 km) |
| `worst-case` | 250 ms ± 30 ms | Chile ↔ Southeast Asia via Pacific |

## CLI Reference

```bash
# List all scenarios with justifications
docker compose run --rm probe-local --list-scenarios

# List all network profiles with overhead estimates
docker compose run --rm probe-local --list-profiles

# Discover GPUs and check NVLink
docker compose run --rm probe-local --discover-gpus

# Dry run (show config without running inference)
docker compose run --rm probe-local --scenario analyst --dry-run

# Custom model
docker compose run --rm probe-local --scenario creative --tp 2 \
    --model Qwen/Qwen2.5-7B --rounds 5 -o /results
```

## Output Format

Each run produces a JSON report in `results/`:

```json
{
  "results": {
    "timing": {
      "ttft": { "mean_s": 0.267, "p50_s": 0.268, "p99_s": 0.268 },
      "tpot": { "mean_s": 0.01895, "p50_s": 0.01896, "p99_s": 0.01898 },
      "throughput": { "tokens_per_second": 47.87 },
      "total_wall_clock_s": 2.674
    },
    "analysis": {
      "network_overhead_pct": 0.0,
      "decode_bottleneck": "COMPUTE/MEMORY-BOUND."
    }
  }
}
```

## Project Structure

```
vllm_probe/
├── vllm_probe.py          # Core benchmark engine (~1400 lines)
├── entrypoint.sh           # Pre-flight checks (CUDA, NVLink, tc)
├── Dockerfile              # CUDA 12.4 + iproute2 + vLLM
├── docker-compose.yml      # Single-node + cluster deployment modes
├── requirements.txt        # Python dependencies (vllm>=0.6.0)
├── results/                # JSON benchmark reports
└── report/
    └── vllm_probe_report.tex  # LaTeX report (compile on Overleaf)
```

## Architecture

```
┌──────────────────────────────────────────────┐
│  Docker Container (NET_ADMIN)                 │
│                                               │
│  ┌─────────┐  PCIe/NVLink  ┌─────────┐      │
│  │  GPU 0  │◄─────────────►│  GPU 1  │      │
│  └─────────┘               └─────────┘      │
│       ↑                         ↑            │
│       └──── NCCL AllReduce ─────┘            │
│              (P2P or Socket)                  │
│                    ↕                          │
│         tc netem on loopback                  │
│         (latency injection)                   │
└──────────────────────────────────────────────┘
```

## Requirements

- NVIDIA GPU(s) with CUDA 12.1+
- Docker with NVIDIA Container Toolkit
- `--cap-add=NET_ADMIN` for traffic control (handled by docker-compose.yml)

## License

MIT
