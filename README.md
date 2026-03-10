# vllm_probe

**LLM Inference Benchmarking Suite with Physics-Based Network Simulation**

A standardized tool for measuring vLLM inference performance under simulated distributed conditions. Injects real network latency at the Linux kernel level to empirically prove whether Tensor Parallelism is viable across different infrastructure topologies — without needing multiple physical locations.

---

## Key Results

| Metric | TP=2 Local | TP=2 Cross-Continent (100ms) | Change |
|--------|-----------|------------------------------|--------|
| TPOT | 11.76 ms | 115.82 ms | **9.8× slower** |
| Throughput | 76.56 tok/s | 7.77 tok/s | **9.9× lower** |
| Total (2K tokens) | 26.75 s | 263.44 s (4m 23s) | **9.8× longer** |

**Conclusion:** Tensor Parallelism across Santiago ↔ Eindhoven (~200ms RTT) is architecturally infeasible for real-time inference. Each of the 2,048 decode steps pays the full network round-trip cost.

---

## Table of Contents

- [How It Works](#how-it-works)
- [Quick Start](#quick-start)
- [Examples — From Simple to Advanced](#examples--from-simple-to-advanced)
- [Workload Scenarios](#workload-scenarios)
- [Network Profiles](#network-profiles)
- [Input Modes](#input-modes)
- [GPU Configuration](#gpu-configuration)
- [Understanding the Metrics](#understanding-the-metrics)
- [Warmup Rounds vs. Benchmark Rounds](#warmup-rounds-vs-benchmark-rounds)
- [Output Format](#output-format)
- [How vllm_probe Differs From Existing Tools](#how-vllm_probe-differs-from-existing-tools)
- [Complete Flag Reference](#complete-flag-reference)
- [Requirements](#requirements)
- [Architecture](#architecture)

---

## How It Works

### Three-Layer Simulation

```
Layer 1: GPU Interconnect (NCCL env vars)
  NVLink ↔ PCIe fallback via NCCL_P2P_DISABLE

Layer 2: Network Latency (Linux tc netem)
  Kernel-level packet delay on loopback interface

Layer 3: Workload Scenarios (analyst/creative/chatbot/custom)
  Isolate prefill vs decode vs scheduler bottlenecks
```

### What Does This Tool Actually Do Internally?

vllm_probe does **not** call vLLM's `benchmark_serving.py` or `benchmark_throughput.py`. It builds its own inference pipeline using vLLM's Python API (`vllm.LLM` class) directly — no HTTP server, no API calls.

The flow is:

1. **Initialize the vLLM engine** with your TP configuration, selected GPUs, and memory budget:
   ```python
   llm = LLM(
       model=config.model,
       tensor_parallel_size=config.tensor_parallel,
       gpu_memory_utilization=config.gpu_mem,
       max_model_len=target_model_len,
       enforce_eager=True,
       enable_prefix_caching=False,  # No KV-cache reuse between rounds
   )
   ```

2. **Build the prompt** — one of three modes:
   - **Default (repeated):** A base sentence is tokenized and repeated to fill exactly X positions. Deterministic but could benefit from cache-friendly patterns.
   - **`--random-input`:** Truly random token IDs. Every position is unique — no prefix caching shortcuts. Use `--seed` for reproducibility.
   - **`--prompt-file report.txt`:** A real document is tokenized and used as-is. The most realistic mode — known content, deterministic, no artificial patterns.

3. **Generate with `ignore_eos=True`** — forces the model to produce **exactly** `max_tokens` output tokens, regardless of whether the model would naturally emit an EOS (end-of-sequence) token. Without this, the model could stop early (e.g., generating 47 tokens instead of 2,048), making TPOT and throughput measurements inconsistent and unreliable.

4. **Measure with `time.perf_counter()`** — the highest-resolution monotonic clock in Python:
   - TTFT = time from request submission to first token arrival
   - TPOT = `(total_time - ttft) / (output_tokens - 1)`
   - Throughput = `total_output_tokens / total_time`
   - Every individual round is recorded separately, then aggregated.

### Why `enable_prefix_caching=False`?

vLLM has a feature called Automatic Prefix Caching (APC): when a new request arrives with the same token prefix as a previous request, it skips recomputing the KV-cache for the matching portion. Since warmup rounds and benchmark rounds use the same prompt, APC could make benchmark TTFT artificially faster — the prefill work was already done during warmup.

With `enable_prefix_caching=False`, every round computes the full prefill from scratch. The warmup rounds warm the **hardware** (CUDA kernels, GPU clocks, NCCL channels), not the **data cache**.

### Critical Discovery: NCCL Transport Bypass

Initial tests showed **zero effect** from `tc netem` latency injection. NCCL uses direct PCIe P2P transfers that bypass the Linux network stack entirely. The fix:

```bash
NCCL_P2P_DISABLE=1    # Disable direct GPU memory access
NCCL_SHM_DISABLE=1    # Disable shared memory transport
NCCL_NET=Socket        # Force TCP socket communication
```

This forces NCCL through TCP sockets where `tc netem` can intercept packets. These variables are set automatically when latency injection is active.

---

## Quick Start

```bash
git clone https://github.com/DiamondRZK/vllm_probe.git
cd vllm_probe
docker compose build

# Discover your GPUs first
docker compose run --rm probe-local --discover-gpus

# Run a basic benchmark
docker compose run --rm probe-local --scenario creative --tp 2 --rounds 3 -o /results
```

---

## Examples — From Simple to Advanced

### 1. Basic: Single GPU Baseline

The simplest possible run. One GPU, no network simulation, default model.

```bash
docker compose run --rm probe-local \
    --scenario analyst \
    --tp 1 \
    --rounds 3 \
    -o /results
```

**What this answers:** "How fast is one GPU?" — your baseline TTFT and compute speed.

### 2. Two GPUs: Does Tensor Parallelism Help?

Split the model across both GPUs on the same machine.

```bash
docker compose run --rm probe-local \
    --scenario creative \
    --tp 2 \
    --rounds 3 \
    -o /results
```

**What this answers:** "Is TP=2 faster than TP=1?" — compare TPOT with Test 1.

### 3. Cross-Continent: The Proof of Infeasibility

Same as Test 2, but with 100ms one-way latency injected (simulating 12,000 km submarine cable).

```bash
docker compose run --rm probe-local \
    --scenario creative \
    --tp 2 \
    --network cross-continent \
    --rounds 3 \
    -o /results
```

**What this answers:** "Can I split my model across Santiago and Eindhoven?" — the answer will be a definitive no.

### 4. Production Load: How Many Users?

32 concurrent users, moderate context. Stress-tests vLLM's scheduler.

```bash
docker compose run --rm probe-local \
    --scenario chatbot \
    --tp 2 \
    --rounds 3 \
    -o /results
```

**What this answers:** "How many concurrent users can my 2×GPU setup serve?"

### 5. Custom Scenario: Your Own Parameters

You are not limited to the three built-in scenarios. Define your own workload:

```bash
docker compose run --rm probe-local \
    --scenario custom \
    --input-tokens 4096 \
    --output-tokens 1024 \
    --concurrent 8 \
    --tp 2 \
    --rounds 3 \
    -o /results
```

**When to use this:** When the built-in scenarios don't match your production workload. For example, your API receives ~4K token documents and generates ~1K token summaries with 8 users.

### 6. Custom Network: Your Measured RTT

If you know your actual inter-node latency (e.g., from `ping` or `iperf3`):

```bash
docker compose run --rm probe-local \
    --scenario creative \
    --tp 2 \
    --network custom \
    --delay-ms 75 \
    --jitter-ms 5 \
    --rounds 3 \
    -o /results
```

**When to use this:** When you've measured your real network RTT and want to simulate that exact value instead of using a pre-built profile.

### 7. Random Input: Realistic Prefill Measurement

By default, the tool repeats a base sentence to fill the input length. This is deterministic but could benefit from prefix cache-friendly patterns. For more realistic prefill measurement:

```bash
docker compose run --rm probe-local \
    --scenario analyst \
    --tp 1 \
    --random-input \
    --seed 42 \
    --rounds 3 \
    -o /results
```

**When to use this:** When you care about absolute TTFT accuracy and want to ensure no caching shortcuts. The `--seed` makes it reproducible.

### 8. Real Document: Use Your Own Prompt

Use an actual text file as the prompt instead of synthetic tokens:

```bash
docker compose run --rm probe-local \
    --scenario custom \
    --prompt-file /results/my_document.txt \
    --output-tokens 512 \
    --tp 2 \
    --rounds 3 \
    -o /results
```

The tool tokenizes the file and uses those token IDs as the prompt. If you also specify `--input-tokens`, it truncates to that length. If you don't, it uses the full tokenized length of the file.

**When to use this:** When you want to benchmark with real data — a legal contract, a financial report, a code file — instead of synthetic tokens. This is the most scientifically honest input mode.

### 9. Specific GPUs: Choose Which Hardware

On a multi-GPU server, select exactly which GPUs to use:

```bash
# Use only GPU 2 for a single-GPU test
docker compose run --rm probe-local \
    --scenario analyst \
    --tp 1 \
    --gpu-ids 2 \
    --rounds 3 \
    -o /results

# Use GPUs 2 and 3 for TP=2 (skip GPUs 0 and 1)
docker compose run --rm probe-local \
    --scenario creative \
    --tp 2 \
    --gpu-ids 2,3 \
    --rounds 3 \
    -o /results
```

**When to use this:** When other processes are using GPUs 0–1, or when you want to test specific GPU pairs (e.g., comparing GPUs connected via NVLink vs. GPUs on different PCIe switches).

### 10. GPU Memory Tuning

Control how much VRAM vLLM allocates:

```bash
# 90% — maximize KV-cache for chatbot (32 concurrent users need more cache)
docker compose run --rm probe-local \
    --scenario chatbot \
    --tp 2 \
    --gpu-mem 0.90 \
    --rounds 3 \
    -o /results

# 50% — safe when other processes share the GPU
docker compose run --rm probe-local \
    --scenario analyst \
    --tp 1 \
    --gpu-mem 0.50 \
    --rounds 3 \
    -o /results
```

**When to change it:** Default is 0.70 (safe for most cases). Lower it if you get OOM errors or share the GPU. Raise it for the chatbot scenario where 32 users need more KV-cache space.

### 11. Dry Run: Preview Before Committing

See exactly what the tool will do without loading the model or running inference:

```bash
docker compose run --rm probe-local \
    --scenario creative \
    --tp 2 \
    --network cross-continent \
    --dry-run
```

**When to use this:** Before a long run (~25 min for cross-continent). Validates your configuration and shows the theoretical network overhead calculation.

### 12. Everything Combined: Full Custom Run

All features at once:

```bash
docker compose run --rm probe-local \
    --scenario custom \
    --input-tokens 2048 \
    --output-tokens 512 \
    --concurrent 4 \
    --network custom \
    --delay-ms 50 \
    --jitter-ms 3 \
    --tp 2 \
    --gpu-ids 2,3 \
    --gpu-mem 0.85 \
    --random-input \
    --seed 123 \
    --warmup 3 \
    --rounds 5 \
    -o /results
```

---

## Workload Scenarios

### Pre-built Scenarios

| Scenario | Input | Output | Concurrency | Bottleneck Tested | Network Sensitivity |
|----------|-------|--------|-------------|-------------------|-------------------|
| **analyst** | 8,192 | 128 | 1 | GPU TFLOPS (prefill) | Low |
| **creative** | 64 | 2,048 | 1 | Mem BW + Interconnect (decode) | **Critical** |
| **chatbot** | 512 | 512 | 32 | Scheduler + VRAM | Moderate |

### Why These Specific Numbers?

Each scenario is engineered as a **controlled experiment** to isolate a specific bottleneck. The values are not arbitrary:

- **analyst (8,192 in / 128 out):** At 8,192 input tokens, the prefill phase (processing all tokens in parallel through 36 transformer layers) dominates total time. With only 128 output tokens, the decode phase is negligible. If you see a TPOT difference between TP=1 and TP=2 here, it's about compute scaling, not interconnect.

- **creative (64 in / 2,048 out):** The opposite extreme — tiny prefill (64 tokens, essentially instant), but 2,048 sequential decode steps. Each step requires an AllReduce synchronization across GPUs. This mathematically maximizes the proportion of time spent on GPU-to-GPU communication. If you used 500 output tokens instead, the network overhead would be 4× smaller and might not clearly dominate the measurement.

- **chatbot (512/512 × 32 users):** Represents a real API serving workload. 32 concurrent requests stress vLLM's PagedAttention scheduler and KV-cache memory management. The throughput number here tells you how many users your hardware can actually serve.

### Custom Scenarios

When the built-in scenarios don't match your workload, use `--scenario custom`:

```bash
docker compose run --rm probe-local \
    --scenario custom \
    --input-tokens 4096 \
    --output-tokens 1024 \
    --concurrent 8 \
    --tp 2 -o /results
```

The `--input-tokens`, `--output-tokens`, and `--concurrent` flags are **only used** when `--scenario custom` is selected. They are ignored for built-in scenarios.

---

## Network Profiles

### Pre-built Profiles

| Profile | One-way Delay | Jitter | Real-world Equivalent |
|---------|--------------|--------|-----------------------|
| `local` | 0 ms | — | Same rack / NVLink domain |
| `datacenter` | 1 ms | ± 0.1 ms | Same building, different racks |
| `regional` | 15 ms | ± 2 ms | Santiago ↔ Buenos Aires (~1,500 km) |
| `cross-continent` | 100 ms | ± 10 ms | Santiago ↔ Eindhoven (~12,000 km) |
| `worst-case` | 250 ms | ± 30 ms | Chile ↔ Southeast Asia via Pacific |

**Why these values?** Each profile maps to a real-world distance using the industry rule: **~10 ms RTT per 1,000 km of fiber** ([TeleGeography, 2025](https://blog.telegeography.com/its-time-to-learn-about-latency)). Light in optical fiber travels at ~200,000 km/s (the vacuum speed of light c ≈ 300,000 km/s divided by the fiber's refractive index n ≈ 1.47). The EllaLink submarine cable between Portugal and Brazil achieves <60 ms RTT over ~6,200 km, consistent with this formula.

### Custom Network Profile

```bash
docker compose run --rm probe-local \
    --scenario creative --tp 2 \
    --network custom \
    --delay-ms 75 \
    --jitter-ms 5 \
    -o /results
```

The `--delay-ms` and `--jitter-ms` flags are **only used** when `--network custom` is selected.

---

## Input Modes

vllm_probe supports three input modes, each with different trade-offs:

### 1. Repeated Sentence (Default)

A base sentence is tokenized and repeated/truncated to fill exactly X input positions.

```bash
# Uses repeated tokens — deterministic, same input every run
docker compose run --rm probe-local --scenario analyst --tp 1 -o /results
```

**Pros:** Fully deterministic. Identical input across runs for perfect A/B comparisons.
**Cons:** The repeating pattern `[A, B, C, A, B, C, ...]` could benefit from cache-friendly memory access patterns in the attention mechanism.

### 2. Random Tokens (`--random-input`)

Truly random token IDs where every position is unique. No repeating patterns.

```bash
docker compose run --rm probe-local \
    --scenario analyst --tp 1 \
    --random-input --seed 42 \
    -o /results
```

**Pros:** No caching shortcuts. More realistic prefill measurement. Use `--seed` for reproducibility.
**Cons:** Slightly different each run without `--seed`. The random IDs might hit unusual tokens (mitigated by skipping special token ranges).

### 3. Real Document (`--prompt-file`)

Tokenize a real .txt file and use those tokens as input.

```bash
docker compose run --rm probe-local \
    --scenario custom \
    --prompt-file /results/contract.txt \
    --output-tokens 512 \
    --tp 2 -o /results
```

**Pros:** The most scientifically honest mode. Known content, deterministic, realistic token distribution. If you specify `--input-tokens` it truncates to that length; otherwise it uses the full file.
**Cons:** Input length depends on file content. Must provide the file inside the Docker volume.

### Which Mode Should I Use?

| Use case | Recommended mode | Why |
|----------|-----------------|-----|
| Comparing TP=1 vs TP=2 | Default (repeated) | Same input guarantees only TP changed |
| Absolute TTFT measurement | `--random-input --seed 42` | No cache-friendly patterns |
| Benchmarking your real workload | `--prompt-file` | Actual data your users send |
| Reproducing published results | Default (repeated) | Deterministic across machines |

---

## GPU Configuration

### GPU Memory (`--gpu-mem`)

Controls what fraction of GPU VRAM vLLM allocates. Default: `0.70`.

| Value | Use case |
|-------|----------|
| `0.50` | Other processes sharing the GPU |
| `0.70` | Default — safe for most scenarios |
| `0.85` | Chatbot with 32 users (needs more KV-cache) |
| `0.95` | Dedicated GPU, maximum performance |

If you get OOM errors, **lower this value first**.

### GPU Selection (`--gpu-ids`)

Select specific physical GPUs via `CUDA_VISIBLE_DEVICES`. Independent of `--tp`:

```bash
# Test GPU 2 alone
--tp 1 --gpu-ids 2

# Test GPUs 2 and 3 together
--tp 2 --gpu-ids 2,3

# Test GPUs 4,5,6,7 on a DGX server
--tp 4 --gpu-ids 4,5,6,7
```

The number of selected GPUs must be ≥ `--tp`. This is useful when other workloads occupy certain GPUs, or when you want to compare different GPU pairs (e.g., GPUs connected via NVLink vs. GPUs across PCIe switches).

### NVLink Control (`--nvlink`)

Controls whether NCCL uses NVLink for GPU-to-GPU communication:

- `--nvlink enable` (default): Use whatever the fastest available interconnect is.
- `--nvlink disable`: Force NCCL to use PCIe by setting `NCCL_P2P_DISABLE=1`.

**Note:** If your hardware has no NVLink (e.g., RTX 4000 Ada), this flag has no effect — NCCL uses PCIe regardless. The tool detects this automatically via `nvidia-smi topo -m` and logs the actual interconnect being used.

---

## Understanding the Metrics

### TTFT — Time To First Token

Measures the **prefill phase**: all input tokens processed in parallel through every transformer layer. This is compute-bound (TFLOPS-dominated).

TTFT is mostly **insensitive to network latency** because the synchronization cost is amortized across thousands of tokens processed in parallel. A 100ms AllReduce delay spread across 8,192 parallel tokens barely registers.

**Real-world meaning:** The "thinking" pause before text starts appearing. Under 500ms feels instant, 1–2s feels acceptable, over 3s feels broken.

### TPOT — Time Per Output Token

Measures the **decode phase**: tokens generated one at a time, sequentially. Each token requires reading model weights from VRAM (memory bandwidth-bound), computing attention and feed-forward layers, and — with Tensor Parallelism — performing an AllReduce synchronization across GPUs **before it can start the next token**.

The critical difference: in prefill, the network cost is paid **once**. In decode, it's paid **per token**. If you generate 2,048 tokens and each AllReduce round-trip takes 200ms, that's 2,048 × 200ms = **409.6 seconds** of pure network waiting.

**Real-world meaning:** TPOT determines streaming speed. At 20ms/token → 50 tok/s (fast). At 100ms → 10 tok/s (readable but slow). At 200ms from network latency → architecturally broken.

### Why `ignore_eos=True` Is Critical

The `max_tokens` parameter in vLLM's `SamplingParams` is a **maximum**, not a target. If the model emits an EOS token before reaching the specified count, it stops early. For example, if the prompt leads to a short answer, the model might generate 3 tokens instead of 2,048. This makes TPOT measurements meaningless — you'd be dividing by 2 instead of 2,047.

Setting `ignore_eos=True` forces the model to generate **exactly** the specified number of tokens every time, making measurements precise and reproducible.

---

## Warmup Rounds vs. Benchmark Rounds

### What Are Warmup Rounds?

Warmup rounds (default: 2) run full inference but **discard the results**. They exist to pay one-time hardware initialization costs:

1. **CUDA kernel JIT** — the first time a specific kernel shape runs, CUDA compiles it. Subsequent calls use the cached kernel.
2. **GPU clock ramp-up** — GPUs start at base clock (~1.5 GHz) and boost to turbo (~2.3 GHz) under sustained load. The first round runs partially at base clock.
3. **NCCL initialization** — the first AllReduce call sets up communication channels, negotiates protocols, allocates buffers.
4. **Memory allocator warmup** — PyTorch's CUDA allocator is lazy. First allocation triggers pool creation.

These are **hardware-level** warmups, not data warmups.

### Don't Warmup Rounds Create Prefix Cache?

This was a real concern: since warmup and benchmark rounds use the same prompt, vLLM's Automatic Prefix Caching (APC) could skip prefill recomputation in benchmark rounds, making TTFT artificially fast.

The fix: `enable_prefix_caching=False` in the LLM constructor. Every round — warmup and benchmark — computes the full prefill from scratch. **Warmup warms the hardware, not the data.**

### What Are Benchmark Rounds?

Benchmark rounds (default: 3) are the actual measurements recorded into the JSON report. Each round's individual metrics are stored separately, then aggregated into mean, p50, and p99 statistics.

### Visual Summary

```
Warmup Round 1:  prompt → full prefill → SLOW (JIT + clock ramp)     → DISCARDED
Warmup Round 2:  prompt → full prefill → faster (JIT cached)         → DISCARDED
────────────────────────────────────────────────────────────────────────────────
Benchmark Round 1:  prompt → full prefill → steady-state speed       → RECORDED
Benchmark Round 2:  prompt → full prefill → steady-state speed       → RECORDED
Benchmark Round 3:  prompt → full prefill → steady-state speed       → RECORDED
                                                                        ↓
                                                              Aggregate: mean, p50, p99
```

---

## Output Format

Each run produces a JSON report in `results/`. The report includes **per-round granularity** — every individual round's measurements — plus the final aggregates:

```json
{
  "config": {
    "model": "Qwen/Qwen2.5-3B",
    "scenario": "creative",
    "tensor_parallel_size": 2,
    "gpu_memory_utilization": 0.70,
    "random_input": false,
    "gpu_ids": null
  },
  "results": {
    "warmup_rounds": [
      {
        "round": 1,
        "ttft_s": 3.412,
        "tpot_s": 0.01921,
        "total_time_s": 42.01,
        "note": "DISCARDED — CUDA JIT + clock ramp + NCCL init"
      },
      {
        "round": 2,
        "ttft_s": 2.689,
        "tpot_s": 0.01182,
        "total_time_s": 26.89,
        "note": "DISCARDED — hardware stabilization"
      }
    ],
    "rounds": [
      {
        "round": 1,
        "ttft_s": 2.674,
        "tpot_s": 0.01176,
        "throughput_tok_s": 76.06,
        "total_time_s": 26.79,
        "output_tokens_generated": 2048
      },
      {
        "round": 2,
        "ttft_s": 2.677,
        "tpot_s": 0.01177,
        "throughput_tok_s": 76.95,
        "total_time_s": 26.72,
        "output_tokens_generated": 2048
      },
      {
        "round": 3,
        "ttft_s": 2.673,
        "tpot_s": 0.01175,
        "throughput_tok_s": 76.68,
        "total_time_s": 26.74,
        "output_tokens_generated": 2048
      }
    ],
    "aggregate": {
      "ttft":  { "mean_s": 2.675, "p50_s": 2.674, "p99_s": 2.677 },
      "tpot":  { "mean_s": 0.01176, "p50_s": 0.01176, "p99_s": 0.01177 },
      "throughput": { "mean_tok_s": 76.56, "p50_tok_s": 76.68, "p99_tok_s": 76.95 },
      "total_wall_clock": { "mean_s": 26.75, "p50_s": 26.74, "p99_s": 26.79 }
    },
    "analysis": {
      "network_overhead_pct": 0.0,
      "decode_bottleneck": "COMPUTE/MEMORY-BOUND",
      "round_variance_pct": 0.15
    }
  }
}
```

**What to look for:**
- `warmup_rounds[0].ttft_s` is higher than benchmark rounds — this proves CUDA JIT warmup is real.
- `output_tokens_generated: 2048` in every round — confirms `ignore_eos=True` is working.
- `round_variance_pct < 1%` — confirms reproducible measurements.

---

## How vllm_probe Differs From Existing Tools

| Question you need answered | nvidia-smi | vLLM bench | vllm_probe |
|---------------------------|:----------:|:----------:|:----------:|
| How much VRAM do I have? | ✓ | | |
| How fast is inference right now? | | ✓ | ✓ |
| What if I lose NVLink? | | | ✓ |
| What if my GPUs are 12,000 km apart? | | | ✓ |
| Should I buy NVLink hardware? | | | ✓ |
| Is cross-continent TP viable? | | | ✓ |
| Is network or compute the bottleneck? | | Partially | ✓ |
| How many concurrent users? | | ✓ | ✓ |

vLLM's benchmarks measure **engine speed**. vllm_probe measures **speed under infrastructure constraints**.

### Detailed Comparison

| Feature | `benchmark_serving.py` | `benchmark_throughput.py` | `vllm_probe` |
|---------|:-----:|:-----:|:-----:|
| Measures TTFT / TPOT | ✓ / ✓ | ✗ / ✗ | ✓ / ✓ |
| Aggregate throughput | ✓ | ✓ | ✓ |
| Network latency simulation | ✗ | ✗ | ✓ |
| NCCL transport control | ✗ | ✗ | ✓ |
| NVLink auto-detection | ✗ | ✗ | ✓ |
| Bottleneck classification | ✗ | ✗ | ✓ |
| Pre-built scenarios | ✗ | ✗ | ✓ |
| Custom scenarios | ✓ (manual) | ✓ (manual) | ✓ |
| Per-round granularity | ✗ | ✗ | ✓ |
| Docker containerized | ✗ | ✗ | ✓ |
| Requires running server | ✓ | ✗ | ✗ |
| Primary use case | API load testing | Raw engine speed | **Infrastructure decisions** |

---

## Complete Flag Reference

### Required

| Flag | Description |
|------|-------------|
| `--scenario` / `-s` | `analyst`, `creative`, `chatbot`, `custom`, or `all` |

### Model & Engine

| Flag | Default | Description |
|------|---------|-------------|
| `--model` / `-m` | `Qwen/Qwen2.5-3B` | HuggingFace model ID or local path |
| `--tp` | `1` | Tensor Parallelism degree (number of GPUs) |
| `--gpu-ids` | all GPUs | Comma-separated GPU IDs (e.g., `2,3`). Sets `CUDA_VISIBLE_DEVICES` |
| `--gpu-mem` | `0.70` | GPU memory utilization 0.0–1.0 |
| `--nvlink` | `enable` | `enable` or `disable` NVLink |

### Input Configuration

| Flag | Default | Description |
|------|---------|-------------|
| `--prompt-file` | — | Path to .txt file to use as prompt |
| `--random-input` | off | Use random token IDs instead of repeated sentence |
| `--seed` | `42` | Random seed (only with `--random-input`) |

### Custom Scenario (only with `--scenario custom`)

| Flag | Default | Description |
|------|---------|-------------|
| `--input-tokens` | — | Number of input tokens **(required)** |
| `--output-tokens` | — | Number of output tokens **(required)** |
| `--concurrent` | `1` | Number of concurrent requests |

### Network Configuration

| Flag | Default | Description |
|------|---------|-------------|
| `--network` | `local` | `local`, `datacenter`, `regional`, `cross-continent`, `worst-case`, or `custom` |
| `--delay-ms` | — | One-way delay in ms (only with `--network custom`, **required**) |
| `--jitter-ms` | `0` | Jitter in ms (only with `--network custom`) |

### Benchmark Control

| Flag | Default | Description |
|------|---------|-------------|
| `--warmup` | `2` | Warmup rounds (hardware init, results discarded) |
| `--rounds` | `3` | Benchmark rounds (results recorded) |
| `-o` | `./results` | Output directory for JSON reports |

### Info Commands (no inference required)

| Flag | Description |
|------|-------------|
| `--discover-gpus` | Show GPU topology, NVLink status, VRAM |
| `--list-scenarios` | Print all pre-built scenarios with parameters |
| `--list-profiles` | Print all network profiles with latency values |
| `--dry-run` | Print full configuration without running inference |
| `--version` | Print tool version |

---

## Requirements

- NVIDIA GPU(s) with CUDA 12.1+
- NVIDIA Driver ≥ 535
- Docker Engine ≥ 24.0 with Compose v2
- NVIDIA Container Toolkit
- ~20 GB disk space (Docker image + model weights)
- `--cap-add=NET_ADMIN` for traffic control (handled by docker-compose.yml)

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

## Project Structure

```
vllm_probe/
├── vllm_probe.py          # Core benchmark engine
├── entrypoint.sh           # Pre-flight checks (CUDA, NVLink, tc)
├── Dockerfile              # CUDA 12.4 + iproute2 + vLLM
├── docker-compose.yml      # Single-node + cluster deployment modes
├── requirements.txt        # Python dependencies (vllm>=0.6.0)
├── results/                # JSON benchmark reports
└── report/
    └── vllm_probe_report.tex  # LaTeX report (compile on Overleaf)
```

## License

MIT

## Author

Norman Cortes — norman.cortes@usm.cl
Universidad Técnica Federico Santa María

Repository: [github.com/DiamondRZK/vllm_probe](https://github.com/DiamondRZK/vllm_probe)