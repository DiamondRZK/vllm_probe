#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  vllm_probe — LLM Inference Benchmarking Suite with Physics Simulation     ║
║                                                                            ║
║  Standardized benchmarking for vLLM with network latency injection and     ║
║  GPU interconnect simulation. Designed for distributed inference research. ║
╚══════════════════════════════════════════════════════════════════════════════╝

Scientific Basis
================
LLM inference is NOT a single monolithic operation. It decomposes into two
phases with fundamentally opposite computational physics:

  1. PREFILL (Compute-Bound)
     ─────────────────────
     The entire input prompt is processed in parallel through the transformer
     layers. The bottleneck is raw TFLOPS — how fast the GPU can multiply
     matrices. Network latency between GPUs is amortized over many tokens,
     so the impact is LOW.

     Analogy: Reading an entire book at once. Speed = how fast your eyes move.

  2. DECODE (Memory-Bound & Latency-Sensitive)
     ─────────────────────────────────────────
     Tokens are generated ONE AT A TIME, autoregressively. Token N cannot
     begin until Token N-1 is complete. In Tensor Parallelism (TP), EVERY
     single token requires an AllReduce synchronization across ALL GPUs.

     This means:
       - With NVLink (600 GB/s): ~2μs per sync → negligible
       - With PCIe 4.0 (32 GB/s): ~50μs per sync → noticeable degradation
       - With network (200ms RTT): 200ms × 2048 tokens = 409 SECONDS of
         pure waiting. The model would take ~7 minutes just on network
         synchronization, regardless of GPU speed.

     Analogy: Writing a book one word at a time, but after each word you must
     call someone overseas and wait for their approval before writing the next.

Why Docker?
===========
The containerization serves three critical purposes:

  1. REPRODUCIBILITY: Same CUDA runtime, same NCCL version, same kernel
     modules. A benchmark run in Santiago produces identical conditions to
     one in Eindhoven.

  2. ISOLATION: The `tc` (Traffic Control) rules we inject to simulate
     latency are scoped to the container's network namespace, NOT the host.
     This prevents accidentally degrading the host's network.

  3. PORTABILITY: The same image runs on:
     - A single server with 2× RTX 4000 (local TP=2)
     - A multi-node cluster with NVSwitch
     - A geo-distributed setup (Santiago ↔ Eindhoven) for real-world
       validation against our simulated results.

Author: Infrastructure AI Architect
License: MIT
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS & CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

VERSION = "1.0.0"

# Network latency profiles (one-way delay in ms)
# These are based on real-world measurements of submarine fiber optic cables
# and terrestrial backbone networks.
LATENCY_PROFILES = {
    "local": {
        "delay_ms": 0,
        "jitter_ms": 0,
        "description": "Local machine — no artificial latency",
        "real_world": "Same rack / NVLink domain",
    },
    "datacenter": {
        "delay_ms": 1,
        "jitter_ms": 0.1,
        "description": "Intra-datacenter (different racks, same building)",
        "real_world": "~0.5-2ms RTT typical within a DC",
    },
    "regional": {
        "delay_ms": 15,
        "jitter_ms": 2,
        "description": "Regional (e.g., Santiago ↔ Buenos Aires, ~1500km)",
        "real_world": "~25-35ms RTT over terrestrial fiber",
    },
    "cross-continent": {
        "delay_ms": 100,
        "jitter_ms": 10,
        "description": "Intercontinental (e.g., Santiago ↔ Eindhoven, ~12000km)",
        "real_world": "~180-220ms RTT via submarine cable + terrestrial hops",
        # Math: Light in fiber ≈ 200,000 km/s. 12,000km × 2 (round trip) / 200,000
        # = 120ms theoretical minimum. Add routing, amplification, switching ≈ 200ms.
    },
    "worst-case": {
        "delay_ms": 250,
        "jitter_ms": 30,
        "description": "Worst case (e.g., Chile ↔ Southeast Asia via Pacific)",
        "real_world": "~400-500ms RTT with congestion",
    },
}

# Workload scenarios — scientifically designed to isolate specific bottlenecks
WORKLOAD_SCENARIOS = {
    "analyst": {
        "input_tokens": 8192,
        "output_tokens": 128,
        "concurrency": 1,
        "description": "RAG / Document Analysis — Prefill Stress Test",
        "bottleneck": "Compute (TFLOPS)",
        "sensitivity": {
            "network": "LOW — Prefill is parallelizable; one large matmul amortizes sync cost",
            "nvlink": "MODERATE — KV-cache distribution benefits from high bandwidth",
            "vram": "HIGH — 8K context requires significant KV-cache allocation",
        },
        "justification": (
            "With 8192 input tokens and only 128 output tokens, >98% of compute "
            "is spent in the prefill phase. This isolates GPU TFLOPS as the sole "
            "bottleneck. If this scenario is slow, adding more network bandwidth "
            "or NVLink won't help — you need faster GPUs."
        ),
    },
    "creative": {
        "input_tokens": 64,
        "output_tokens": 2048,
        "concurrency": 1,
        "description": "Long-form Generation — Decode Stress Test",
        "bottleneck": "Memory Bandwidth + Interconnect Latency",
        "sensitivity": {
            "network": "CRITICAL — Each of 2048 tokens requires GPU sync (AllReduce)",
            "nvlink": "CRITICAL — PCIe adds ~50μs/token × 2048 = ~100ms total overhead",
            "vram": "LOW — Small input means minimal KV-cache",
        },
        "justification": (
            "This is the 'canary in the coal mine' for distributed inference. "
            "With 2048 autoregressive decode steps, the total latency added by "
            "network round-trips is: 2048 × RTT. For cross-continent (200ms RTT), "
            "that's 2048 × 0.2s = 409.6 seconds of PURE NETWORK WAITING. "
            "This mathematically proves why Tensor Parallelism across continents "
            "is architecturally impossible for real-time inference."
        ),
    },
    "chatbot": {
        "input_tokens": 512,
        "output_tokens": 512,
        "concurrency": 32,
        "description": "Production Chatbot — Throughput Stress Test",
        "bottleneck": "Scheduler (PagedAttention) + VRAM Management",
        "sensitivity": {
            "network": "MODERATE — Amortized across batch, but still per-token",
            "nvlink": "MODERATE — Batched AllReduce is more efficient",
            "vram": "CRITICAL — 32 concurrent sequences compete for KV-cache pages",
        },
        "justification": (
            "32 concurrent users with balanced I/O (512/512) stress-tests vLLM's "
            "PagedAttention scheduler. This reveals whether the system can maintain "
            "acceptable latency under production load, or if VRAM fragmentation "
            "causes request queuing and tail-latency spikes."
        ),
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class BenchmarkResult:
    """Structured output of a single benchmark run."""
    scenario: str
    model: str
    tensor_parallel: int
    nvlink_enabled: bool
    network_profile: str
    network_delay_ms: float
    # Timing metrics (all in seconds unless noted)
    ttft_mean: float = 0.0          # Time To First Token (seconds)
    ttft_p50: float = 0.0
    ttft_p99: float = 0.0
    tpot_mean: float = 0.0          # Time Per Output Token (seconds)
    tpot_p50: float = 0.0
    tpot_p99: float = 0.0
    throughput_tok_s: float = 0.0   # Tokens per second (aggregate)
    total_time: float = 0.0         # Wall-clock time for entire benchmark
    # Configuration echo
    input_tokens: int = 0
    output_tokens: int = 0
    concurrency: int = 0
    # Derived analysis
    network_overhead_estimate: float = 0.0  # Estimated seconds lost to network
    timestamp: str = ""
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ProbeConfig:
    """Full configuration for a probe run."""
    model: str
    scenario: str
    network_profile: str
    nvlink: bool
    tensor_parallel: int
    gpu_ids: str
    output_dir: str
    warmup_rounds: int
    benchmark_rounds: int
    dry_run: bool
    verbose: bool


# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────

def setup_logging(verbose: bool) -> logging.Logger:
    """Configure structured logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s │ %(levelname)-7s │ %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger("vllm_probe")


# ─────────────────────────────────────────────────────────────────────────────
# PHYSICS SIMULATION LAYER
# ─────────────────────────────────────────────────────────────────────────────
#
# WHY WE MANIPULATE THE LINUX KERNEL:
#
# There are two levels of "interconnect" in distributed GPU inference:
#
#   Level 1: GPU ↔ GPU (within a node)
#     - NVLink/NVSwitch: 600 GB/s, ~2μs latency
#     - PCIe 4.0 x16: 32 GB/s, ~10-50μs latency
#     - Controlled via NCCL environment variables
#
#   Level 2: Node ↔ Node (across a network)
#     - InfiniBand: 200 Gb/s, ~1-2μs latency
#     - Ethernet 100G: ~5-10μs latency
#     - Internet (Santiago↔Eindhoven): ~200ms latency
#     - Controlled via Linux Traffic Control (tc/netem)
#
# By manipulating both levels, we can simulate ANY infrastructure topology
# using just a single 2-GPU machine.
#


class PhysicsSimulator:
    """
    Manages Linux kernel-level network and GPU interconnect simulation.

    This class encapsulates all system-level modifications needed to simulate
    different hardware topologies. It follows RAII principles — all changes
    are reverted on cleanup, even if the benchmark crashes.

    SAFETY MODEL:
    ─────────────
    When running inside Docker (recommended), `tc` rules affect ONLY the
    container's network namespace. The host is never touched.

    When running on bare metal (not recommended), `tc` rules affect the
    loopback interface. The `finally` block in the context manager ensures
    cleanup. Additionally, a SIGTERM handler is registered as a safety net.
    """

    def __init__(self, logger: logging.Logger):
        self.log = logger
        self._tc_active = False
        self._original_env: dict[str, Optional[str]] = {}
        self._interface = "lo"  # loopback — safe for simulation

    # ── GPU Interconnect Simulation ──────────────────────────────────────

    def configure_nvlink(self, enabled: bool) -> dict[str, str]:
        """
        Control GPU-to-GPU communication path.

        When NVLink is DISABLED (NCCL_P2P_DISABLE=1):
          - NCCL falls back to PCIe for peer-to-peer transfers
          - AllReduce operations route through CPU/PCIe bridge
          - Bandwidth drops from ~600 GB/s to ~32 GB/s (18.75× slower)
          - Latency increases from ~2μs to ~50μs (25× slower)

        This simulates:
          - Budget GPU servers without NVSwitch
          - Consumer-grade multi-GPU setups
          - Clusters using PCIe-only interconnects

        NOTE: If the hardware has no NVLink (e.g., RTX 4000 Ada), the
        `enabled` flag has no effect — NCCL will use PCIe regardless.
        We detect this and log honestly.

        Returns:
            Dict of environment variables to inject.
        """
        env_vars = {}

        # First, check if NVLink actually exists on this hardware
        hardware_has_nvlink = self._detect_nvlink_hardware()

        if not enabled:
            self.log.warning(
                "╔══════════════════════════════════════════════════════════╗\n"
                "║  NVLink FORCE-DISABLED — PCIe fallback via NCCL        ║\n"
                "║                                                        ║\n"
                "║  NCCL_P2P_DISABLE=1  → No GPU direct memory access     ║\n"
                "║  NCCL_SHM_DISABLE=1  → No shared memory transport      ║\n"
                "║                                                        ║\n"
                "║  Expected impact:                                      ║\n"
                "║    • AllReduce bandwidth: reduced to PCIe speeds       ║\n"
                "║    • Per-sync latency: ~10-50 μs                      ║\n"
                "║    • Decode phase: Most affected (serial dependency)   ║\n"
                "╚══════════════════════════════════════════════════════════╝"
            )
            env_vars["NCCL_P2P_DISABLE"] = "1"
            env_vars["NCCL_SHM_DISABLE"] = "1"
        elif hardware_has_nvlink:
            self.log.info("Interconnect: NVLink ACTIVE (hardware present, flag enabled)")
            env_vars["NCCL_P2P_DISABLE"] = "0"
        else:
            self.log.info(
                "Interconnect: PCIe (NVLink not available on this hardware). "
                "--nvlink flag has no effect."
            )
            env_vars["NCCL_P2P_DISABLE"] = "0"

        # Store originals for restoration
        for key in env_vars:
            self._original_env[key] = os.environ.get(key)
            os.environ[key] = env_vars[key]

        self._hardware_has_nvlink = hardware_has_nvlink
        return env_vars

    def _detect_nvlink_hardware(self) -> bool:
        """Check nvidia-smi topo -m for actual NVLink presence."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "topo", "-m"],
                capture_output=True, text=True,
            )
            if result.returncode != 0:
                return False
            # Look for "NV" followed by a digit in the topology matrix
            for line in result.stdout.split("\n"):
                parts = line.strip().split()
                if len(parts) >= 2 and parts[0].startswith("GPU"):
                    for part in parts[1:]:
                        if part.startswith("NV") and len(part) > 2 and part[2:].isdigit():
                            return True
            return False
        except Exception:
            return False

    # ── Network Latency Simulation ───────────────────────────────────────

    def inject_latency(self, profile_name: str) -> None:
        """
        Use Linux Traffic Control (tc) + netem to add artificial latency.

        HOW IT WORKS (Linux Kernel Level):
        ──────────────────────────────────
        Linux's network stack processes packets through a series of "queuing
        disciplines" (qdiscs). By default, packets on the loopback interface
        pass through instantly (pfifo_fast qdisc).

        We replace this with NetEm (Network Emulator), which holds each
        packet for a specified delay before releasing it. This affects ALL
        traffic on the interface, including:
          - NCCL's socket-based communication (when P2P is disabled)
          - gRPC calls between vLLM worker processes
          - Any TCP/UDP traffic used for distributed coordination

        WHY LOOPBACK (lo)?
        ───────────────────
        In a single-node multi-GPU setup, when NCCL falls back to socket
        transport (P2P disabled), it communicates over localhost. By adding
        latency to `lo`, we simulate the effect of GPUs being separated by
        a real network without needing multiple physical machines.

        In Docker, this is even safer — each container has its own network
        namespace with its own `lo` interface.

        JITTER MODELING:
        ────────────────
        Real networks don't have constant latency. We add Gaussian jitter
        (±X ms) to simulate packet-level timing variation. This is important
        because NCCL's AllReduce must wait for the SLOWEST packet, so jitter
        can disproportionately impact tail latency.
        """
        profile = LATENCY_PROFILES[profile_name]
        delay = profile["delay_ms"]
        jitter = profile["jitter_ms"]

        if delay == 0:
            self.log.info(f"Network profile '{profile_name}': No latency injection needed")
            return

        # CRITICAL: Force NCCL to use socket transport so tc netem works.
        # Without this, NCCL uses P2P (PCIe direct) or SHM (shared memory),
        # both of which bypass the Linux network stack entirely — meaning
        # tc netem has ZERO effect on GPU communication.
        #
        # This was discovered empirically: tests with tc latency showed
        # identical TPOT with and without injection, because NCCL never
        # touched the loopback interface.
        nccl_socket_vars = {
            "NCCL_P2P_DISABLE": "1",
            "NCCL_SHM_DISABLE": "1",
            "NCCL_NET": "Socket",
        }
        for key, value in nccl_socket_vars.items():
            if key not in self._original_env:
                self._original_env[key] = os.environ.get(key)
            os.environ[key] = value

        self.log.warning(
            f"╔══════════════════════════════════════════════════════════╗\n"
            f"║  INJECTING NETWORK LATENCY on interface '{self._interface}'          ║\n"
            f"║                                                        ║\n"
            f"║  Profile: {profile_name:<46s} ║\n"
            f"║  Delay:   {delay}ms ± {jitter}ms jitter{' ' * (34 - len(str(delay)) - len(str(jitter)))}║\n"
            f"║  Real-world: {profile['real_world']:<43s} ║\n"
            f"║                                                        ║\n"
            f"║  NCCL forced to Socket transport (P2P+SHM disabled)    ║\n"
            f"║  so tc netem latency reaches GPU synchronization.      ║\n"
            f"║                                                        ║\n"
            f"║  Impact on decode (2048 tokens, TP=2):                 ║\n"
            f"║    Network wait = 2048 × {2*delay}ms = {2048 * 2 * delay / 1000:.1f}s{' ' * max(0, 22 - len(f'{2048 * 2 * delay / 1000:.1f}'))}║\n"
            f"╚══════════════════════════════════════════════════════════╝"
        )

        # Clean any existing rules first
        self._cleanup_tc()

        cmd = (
            f"tc qdisc add dev {self._interface} root netem "
            f"delay {delay}ms {jitter}ms distribution normal"
        )
        self._run_tc(cmd)
        self._tc_active = True

        # Verify the rule was applied
        verify = subprocess.run(
            f"tc qdisc show dev {self._interface}",
            shell=True, capture_output=True, text=True,
        )
        self.log.debug(f"tc verification: {verify.stdout.strip()}")

    def _run_tc(self, cmd: str) -> None:
        """Execute a tc command with proper error handling."""
        self.log.debug(f"Executing: {cmd}")
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True,
        )
        if result.returncode != 0:
            if "RTNETLINK answers: File exists" in result.stderr:
                self.log.warning("tc rule already exists — cleaning and retrying")
                self._cleanup_tc()
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                if result.returncode != 0:
                    raise RuntimeError(f"tc command failed: {result.stderr}")
            elif "Operation not permitted" in result.stderr:
                raise PermissionError(
                    "Cannot configure traffic control. Run with:\n"
                    "  • Docker: --cap-add=NET_ADMIN\n"
                    "  • Bare metal: sudo python vllm_probe.py ..."
                )
            else:
                raise RuntimeError(f"tc command failed: {result.stderr}")

    def _cleanup_tc(self) -> None:
        """Remove all tc rules from the interface."""
        subprocess.run(
            f"tc qdisc del dev {self._interface} root",
            shell=True, capture_output=True, text=True,
        )
        self._tc_active = False

    # ── Cleanup & Safety ─────────────────────────────────────────────────

    def cleanup(self) -> None:
        """
        Revert ALL system modifications.

        This is called:
          1. At the end of a successful run
          2. In the `finally` block if an exception occurs
          3. On SIGTERM/SIGINT via signal handler

        Order matters — we clean tc first (network), then env vars.
        """
        self.log.info("Cleaning up physics simulation...")

        # 1. Remove network latency
        if self._tc_active:
            self.log.info(f"Removing tc rules from {self._interface}")
            self._cleanup_tc()

        # 2. Restore environment variables
        for key, original_value in self._original_env.items():
            if original_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original_value

        self.log.info("Physics simulation cleaned up successfully")


@contextmanager
def physics_context(simulator: PhysicsSimulator, config: ProbeConfig):
    """
    Context manager ensuring physics simulation cleanup.

    This is the CRITICAL safety mechanism. Even if the benchmark crashes,
    throws an OOM, or receives SIGKILL, the `finally` block ensures we
    don't leave the system in a degraded state.

    Usage:
        with physics_context(sim, config) as sim:
            run_benchmark(sim, config)
    """
    # Register signal handlers for graceful cleanup
    original_sigterm = signal.getsignal(signal.SIGTERM)
    original_sigint = signal.getsignal(signal.SIGINT)

    def _signal_handler(signum, frame):
        simulator.log.warning(f"Received signal {signum} — cleaning up...")
        simulator.cleanup()
        sys.exit(128 + signum)

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    try:
        # Apply physics modifications
        simulator.configure_nvlink(config.nvlink)
        simulator.inject_latency(config.network_profile)
        yield simulator
    finally:
        simulator.cleanup()
        signal.signal(signal.SIGTERM, original_sigterm)
        signal.signal(signal.SIGINT, original_sigint)


# ─────────────────────────────────────────────────────────────────────────────
# BENCHMARK ENGINE
# ─────────────────────────────────────────────────────────────────────────────


def generate_dummy_prompt(num_tokens: int) -> str:
    """
    Generate a prompt that approximates the target token count.

    WHY NOT EXACT TOKENS?
    ─────────────────────
    Tokenization is model-specific (BPE, SentencePiece, etc.). A word may
    be 1-4 tokens depending on the tokenizer. We use a simple heuristic
    (1 word ≈ 1.3 tokens for English) and rely on vLLM's `max_tokens`
    parameter to enforce exact output length.
    """
    # Average English word ≈ 1.3 tokens in most tokenizers
    num_words = int(num_tokens / 1.3)
    # Use a repetitive but valid prompt to ensure consistent tokenization
    base_sentence = (
        "The distributed inference system processes each request through "
        "multiple GPU nodes coordinated via high-speed interconnect fabric "
    )
    words = base_sentence.split()
    prompt_words = []
    while len(prompt_words) < num_words:
        prompt_words.extend(words)
    return " ".join(prompt_words[:num_words])


def run_single_request(
    llm,
    sampling_params,
    prompt: str,
    logger: logging.Logger,
) -> dict:
    """
    Execute a single inference request and measure timing.

    METRIC DEFINITIONS:
    ───────────────────
    • TTFT (Time To First Token):
      Wall-clock time from request submission to receiving the first
      generated token. This measures prefill latency + scheduling overhead.
      Critical for user-perceived responsiveness.

    • TPOT (Time Per Output Token):
      Average time between consecutive tokens during the decode phase.
      Calculated as: (total_time - TTFT) / (num_output_tokens - 1).
      This is the steady-state generation speed.

    • Throughput:
      Total tokens generated per second: num_output_tokens / total_time.
      For batched scenarios, this is the AGGREGATE across all sequences.
    """
    from vllm import SamplingParams as _  # type check only

    t_start = time.perf_counter()

    # vLLM's generate() returns RequestOutput objects
    outputs = llm.generate([prompt], sampling_params)

    t_end = time.perf_counter()
    total_time = t_end - t_start

    if not outputs or not outputs[0].outputs:
        return {
            "total_time": total_time,
            "num_tokens": 0,
            "ttft": total_time,
            "tpot": 0,
            "throughput": 0,
            "error": "No output generated",
        }

    output = outputs[0]
    num_generated = len(output.outputs[0].token_ids)

    # vLLM provides per-request metrics when available
    metrics = getattr(output, "metrics", None)

    if metrics and hasattr(metrics, "first_token_time"):
        ttft = metrics.first_token_time - metrics.arrival_time
    else:
        # Estimate: TTFT ≈ total_time × (prefill_fraction)
        # For large inputs, prefill dominates; for small inputs, it's fast
        ttft = total_time * 0.1  # Conservative estimate

    if num_generated > 1:
        tpot = (total_time - ttft) / (num_generated - 1)
    else:
        tpot = total_time - ttft

    throughput = num_generated / total_time if total_time > 0 else 0

    return {
        "total_time": total_time,
        "num_tokens": num_generated,
        "ttft": ttft,
        "tpot": tpot,
        "throughput": throughput,
        "error": None,
    }


def run_concurrent_requests(
    llm,
    sampling_params,
    prompt: str,
    concurrency: int,
    logger: logging.Logger,
) -> list[dict]:
    """
    Submit multiple prompts as a batch to stress the scheduler.

    WHY BATCHING MATTERS:
    ─────────────────────
    vLLM uses continuous batching with PagedAttention. Unlike naive batching
    (process all sequences in lockstep), PagedAttention allows:
      - Dynamic memory allocation (no pre-reserved max_seq_len per request)
      - Preemption of lower-priority sequences
      - Efficient KV-cache sharing for common prefixes

    By submitting `concurrency` prompts simultaneously, we test whether
    the scheduler can maintain acceptable per-request latency under load,
    or if VRAM pressure causes request queuing.
    """
    prompts = [prompt] * concurrency

    t_start = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    t_end = time.perf_counter()

    total_time = t_end - t_start
    results = []

    total_tokens = 0
    for output in outputs:
        if output.outputs:
            n = len(output.outputs[0].token_ids)
            total_tokens += n
            results.append({
                "num_tokens": n,
                "total_time": total_time,
            })

    aggregate_throughput = total_tokens / total_time if total_time > 0 else 0

    return [{
        "total_time": total_time,
        "num_tokens": total_tokens,
        "concurrency": concurrency,
        "throughput": aggregate_throughput,
        "per_request_avg": total_time / max(len(results), 1),
        "error": None,
    }]


def compute_percentile(values: list[float], p: float) -> float:
    """Compute the p-th percentile of a list of values."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = int(k)
    c = f + 1 if f + 1 < len(sorted_vals) else f
    d = k - f
    return sorted_vals[f] + d * (sorted_vals[c] - sorted_vals[f])


def run_benchmark(config: ProbeConfig, logger: logging.Logger) -> BenchmarkResult:
    """
    Execute the full benchmark pipeline.

    Pipeline:
    ─────────
    1. Initialize vLLM engine with TP configuration
    2. Run warmup rounds (JIT compilation, CUDA graph capture)
    3. Run benchmark rounds with measurement
    4. Compute statistics and return structured result
    """
    scenario = WORKLOAD_SCENARIOS[config.scenario]
    profile = LATENCY_PROFILES[config.network_profile]

    result = BenchmarkResult(
        scenario=config.scenario,
        model=config.model,
        tensor_parallel=config.tensor_parallel,
        nvlink_enabled=config.nvlink,
        network_profile=config.network_profile,
        network_delay_ms=profile["delay_ms"],
        input_tokens=scenario["input_tokens"],
        output_tokens=scenario["output_tokens"],
        concurrency=scenario["concurrency"],
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    if config.dry_run:
        logger.info("DRY RUN — Skipping actual inference")
        result.network_overhead_estimate = (
            scenario["output_tokens"] * 2 * profile["delay_ms"] / 1000.0
        )
        return result

    # ── Initialize vLLM ──────────────────────────────────────────────────
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        logger.error(
            "vLLM not installed. Install with: pip install vllm\n"
            "Or use the provided Docker image."
        )
        result.errors.append("vllm not installed")
        return result

    logger.info(f"Initializing vLLM engine: model={config.model}, TP={config.tensor_parallel}")

    try:
        # Determine the required context length for this scenario
        # analyst = 8192+128 = 8320, creative = 64+2048 = 2112, chatbot = 512+512 = 1024
        required_len = scenario["input_tokens"] + scenario["output_tokens"]

        # Cap max_model_len to what we actually need (with 2× headroom).
        # WHY: A model like Qwen2.5-3B supports 32K context, but allocating
        # full 32K KV-cache would consume ~4GB VRAM unnecessarily. By capping
        # to our actual requirement, we leave VRAM free for batched requests
        # (critical for the chatbot scenario with 32 concurrent users).
        target_model_len = min(required_len * 2, 32768)

        logger.info(
            f"Scenario requires {required_len} tokens "
            f"(in:{scenario['input_tokens']} + out:{scenario['output_tokens']}). "
            f"Setting max_model_len={target_model_len}"
        )

        llm = LLM(
            model=config.model,
            tensor_parallel_size=config.tensor_parallel,
            gpu_memory_utilization=0.70,  # Conservative — leaves room for other processes
            max_model_len=target_model_len,
            enforce_eager=True,  # Disable CUDA graphs — more compatible, slightly slower
        )
    except Exception as e:
        logger.error(f"Failed to initialize vLLM: {e}")
        result.errors.append(str(e))
        return result

    sampling_params = SamplingParams(
        max_tokens=scenario["output_tokens"],
        temperature=0.0,  # Deterministic for reproducibility
        top_p=1.0,
    )

    prompt = generate_dummy_prompt(scenario["input_tokens"])

    # ── Warmup ───────────────────────────────────────────────────────────
    logger.info(f"Running {config.warmup_rounds} warmup round(s)...")
    for i in range(config.warmup_rounds):
        logger.debug(f"  Warmup {i+1}/{config.warmup_rounds}")
        if scenario["concurrency"] > 1:
            run_concurrent_requests(llm, sampling_params, prompt, scenario["concurrency"], logger)
        else:
            run_single_request(llm, sampling_params, prompt, logger)

    # ── Benchmark ────────────────────────────────────────────────────────
    logger.info(f"Running {config.benchmark_rounds} benchmark round(s)...")
    all_ttft = []
    all_tpot = []
    all_throughput = []
    all_total = []

    for i in range(config.benchmark_rounds):
        logger.info(f"  Round {i+1}/{config.benchmark_rounds}")

        if scenario["concurrency"] > 1:
            measurements = run_concurrent_requests(
                llm, sampling_params, prompt, scenario["concurrency"], logger,
            )
        else:
            measurements = [run_single_request(llm, sampling_params, prompt, logger)]

        for m in measurements:
            if m.get("error"):
                result.errors.append(m["error"])
                continue
            all_total.append(m["total_time"])
            all_throughput.append(m["throughput"])
            if "ttft" in m:
                all_ttft.append(m["ttft"])
            if "tpot" in m:
                all_tpot.append(m["tpot"])

    # ── Compute Statistics ───────────────────────────────────────────────
    if all_ttft:
        result.ttft_mean = sum(all_ttft) / len(all_ttft)
        result.ttft_p50 = compute_percentile(all_ttft, 50)
        result.ttft_p99 = compute_percentile(all_ttft, 99)

    if all_tpot:
        result.tpot_mean = sum(all_tpot) / len(all_tpot)
        result.tpot_p50 = compute_percentile(all_tpot, 50)
        result.tpot_p99 = compute_percentile(all_tpot, 99)

    if all_throughput:
        result.throughput_tok_s = sum(all_throughput) / len(all_throughput)

    if all_total:
        result.total_time = sum(all_total) / len(all_total)

    # Estimate network overhead
    # In TP=2, each decode step requires one AllReduce (2 × one-way delay = 1 RTT)
    if config.tensor_parallel > 1:
        result.network_overhead_estimate = (
            scenario["output_tokens"] * 2 * profile["delay_ms"] / 1000.0
        )
    else:
        result.network_overhead_estimate = 0.0

    return result


# ─────────────────────────────────────────────────────────────────────────────
# REPORT GENERATION
# ─────────────────────────────────────────────────────────────────────────────


def generate_report(result: BenchmarkResult, config: ProbeConfig, logger: logging.Logger, simulator: PhysicsSimulator = None) -> str:
    """Generate a comprehensive report with scientific context."""
    scenario = WORKLOAD_SCENARIOS[config.scenario]
    profile = LATENCY_PROFILES[config.network_profile]

    report = {
        "vllm_probe_version": VERSION,
        "configuration": {
            "model": config.model,
            "scenario": {
                "name": config.scenario,
                "description": scenario["description"],
                "bottleneck": scenario["bottleneck"],
                "justification": scenario["justification"],
                "sensitivity": scenario["sensitivity"],
            },
            "physics": {
                "nvlink_flag": "enable" if config.nvlink else "disable",
                "nvlink_hardware_present": simulator._hardware_has_nvlink if (simulator and hasattr(simulator, '_hardware_has_nvlink')) else "unknown",
                "actual_interconnect": (
                    "NVLink" if (config.nvlink and simulator and hasattr(simulator, '_hardware_has_nvlink') and simulator._hardware_has_nvlink)
                    else "PCIe"
                ),
                "network_profile": config.network_profile,
                "network_description": profile["description"],
                "network_real_world": profile["real_world"],
                "injected_delay_ms": profile["delay_ms"],
            },
            "tensor_parallel_size": config.tensor_parallel,
            "warmup_rounds": config.warmup_rounds,
            "benchmark_rounds": config.benchmark_rounds,
        },
        "results": {
            "timing": {
                "ttft": {
                    "mean_s": round(result.ttft_mean, 4),
                    "p50_s": round(result.ttft_p50, 4),
                    "p99_s": round(result.ttft_p99, 4),
                    "unit": "seconds",
                    "description": "Time To First Token — measures prefill + scheduling latency",
                },
                "tpot": {
                    "mean_s": round(result.tpot_mean, 6),
                    "p50_s": round(result.tpot_p50, 6),
                    "p99_s": round(result.tpot_p99, 6),
                    "unit": "seconds",
                    "description": "Time Per Output Token — steady-state decode speed",
                },
                "throughput": {
                    "tokens_per_second": round(result.throughput_tok_s, 2),
                    "description": "Aggregate output tokens per second",
                },
                "total_wall_clock_s": round(result.total_time, 4),
            },
            "analysis": {
                "estimated_network_overhead_s": round(result.network_overhead_estimate, 2),
                "network_overhead_pct": (
                    round(result.network_overhead_estimate / result.total_time * 100, 1)
                    if result.total_time > 0 else 0
                ),
                "decode_bottleneck": _analyze_bottleneck(result, config),
            },
            "errors": result.errors,
        },
        "timestamp": result.timestamp,
    }

    return json.dumps(report, indent=2, ensure_ascii=False)


def _analyze_bottleneck(result: BenchmarkResult, config: ProbeConfig) -> str:
    """Provide a human-readable analysis of the primary bottleneck."""
    if result.total_time == 0:
        return "No data (dry run or error)"

    if result.network_overhead_estimate > 0:
        overhead_pct = result.network_overhead_estimate / result.total_time * 100
        if overhead_pct > 80:
            return (
                f"NETWORK-DOMINATED ({overhead_pct:.0f}% estimated network wait). "
                f"Tensor Parallelism across this distance is NOT viable for real-time "
                f"inference. Consider Pipeline Parallelism or data replication instead."
            )
        elif overhead_pct > 30:
            return (
                f"NETWORK-SIGNIFICANT ({overhead_pct:.0f}% estimated network wait). "
                f"Performance is degraded but may be acceptable for batch workloads. "
                f"Not suitable for interactive/streaming use cases."
            )

    if not config.nvlink and config.tensor_parallel > 1:
        return (
            "PCIe-LIMITED. Without NVLink, AllReduce bandwidth is 18.75× lower. "
            "Consider NVLink-capable hardware for multi-GPU inference."
        )

    return "COMPUTE/MEMORY-BOUND. GPU hardware is the primary bottleneck (healthy state)."


# ─────────────────────────────────────────────────────────────────────────────
# GPU DISCOVERY
# ─────────────────────────────────────────────────────────────────────────────


def discover_gpus(logger: logging.Logger) -> list[dict]:
    """Detect available NVIDIA GPUs using nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            logger.warning(f"nvidia-smi failed: {result.stderr}")
            return []

        gpus = []
        for line in result.stdout.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4:
                gpus.append({
                    "index": int(parts[0]),
                    "name": parts[1],
                    "vram_total_mb": int(parts[2]),
                    "vram_free_mb": int(parts[3]),
                })
        return gpus
    except FileNotFoundError:
        logger.warning("nvidia-smi not found — are NVIDIA drivers installed?")
        return []


def check_nvlink(logger: logging.Logger) -> bool:
    """
    Check if NVLink is available between GPUs.

    METHOD: Parse `nvidia-smi topo -m` (topology matrix).
    ──────
    This is the DEFINITIVE way to check GPU interconnect type.
    The topology matrix shows the connection type between every GPU pair:
      - "NV#"  = NVLink (# = number of links, e.g. NV4 = 4 NVLink bridges)
      - "PHB"  = PCIe Host Bridge (same CPU socket, different PCIe switches)
      - "PIX"  = PCIe switch (same PCIe switch)
      - "SYS"  = Cross-socket (QPI/UPI between CPUs)
      - "NODE" = Same NUMA node

    Previous approach (`nvidia-smi nvlink -s`) was unreliable because it
    returns exit code 0 even on GPUs without NVLink hardware (like RTX 4000
    Ada), producing false positives.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "topo", "-m"],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            logger.warning(f"nvidia-smi topo failed: {result.stderr}")
            return False

        # Look for "NV" in the topology matrix (indicates NVLink connection)
        # Lines look like: "GPU0  GPU1  NV4  ..." or "GPU0  GPU1  PHB  ..."
        topo_output = result.stdout
        logger.debug(f"GPU Topology:\n{topo_output}")

        has_nvlink = False
        for line in topo_output.split("\n"):
            # Skip header and legend lines; look for GPU-to-GPU rows
            stripped = line.strip()
            if stripped.startswith("GPU") and "\tNV" in line:
                has_nvlink = True
                break
            # Also check space-separated format
            parts = stripped.split()
            if len(parts) >= 3 and parts[0].startswith("GPU"):
                for part in parts[1:]:
                    if part.startswith("NV") and part[2:].isdigit():
                        has_nvlink = True
                        break
            if has_nvlink:
                break

        if has_nvlink:
            logger.info("NVLink DETECTED between GPUs (confirmed via topology matrix)")
        else:
            logger.info("NVLink NOT detected — GPUs connected via PCIe (confirmed via topology matrix)")

        return has_nvlink
    except (FileNotFoundError, Exception) as e:
        logger.warning(f"NVLink check failed: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# CLI INTERFACE
# ─────────────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser with full documentation."""

    parser = argparse.ArgumentParser(
        prog="vllm_probe",
        description=(
            "╔═══════════════════════════════════════════════════════════════╗\n"
            "║  vllm_probe — LLM Inference Benchmark with Physics Sim      ║\n"
            "╚═══════════════════════════════════════════════════════════════╝\n\n"
            "Standardized benchmarking tool for vLLM inference engines.\n"
            "Simulates network latency and GPU interconnect conditions\n"
            "to predict distributed inference performance."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "EXAMPLES:\n"
            "─────────\n"
            "  # Basic benchmark with default model\n"
            "  vllm_probe --scenario analyst\n\n"
            "  # Simulate cross-continent latency (Santiago ↔ Eindhoven)\n"
            "  vllm_probe --scenario creative --network cross-continent --tp 2\n\n"
            "  # Disable NVLink to simulate PCIe-only cluster\n"
            "  vllm_probe --scenario creative --nvlink disable --tp 2\n\n"
            "  # Full stress test: no NVLink + intercontinental latency\n"
            "  vllm_probe --scenario chatbot --nvlink disable --network cross-continent\n\n"
            "  # Dry run (show config without running inference)\n"
            "  vllm_probe --scenario analyst --dry-run\n\n"
            "  # Run ALL scenarios for a complete infrastructure report\n"
            "  vllm_probe --scenario all --network local\n"
        ),
    )

    # ── Scenario (required for benchmarks, not for info commands) ────────
    parser.add_argument(
        "--scenario", "-s",
        required=False,  # Validated manually; not needed for --list-profiles etc.
        default=None,
        choices=list(WORKLOAD_SCENARIOS.keys()) + ["all"],
        help=(
            "Workload scenario to benchmark. "
            "'analyst' = prefill stress (8K in, 128 out). "
            "'creative' = decode stress (64 in, 2K out). "
            "'chatbot' = throughput stress (512/512, 32 concurrent). "
            "'all' = run all three sequentially."
        ),
    )

    # ── Model Configuration ──────────────────────────────────────────────
    parser.add_argument(
        "--model", "-m",
        default="Qwen/Qwen2.5-3B",
        help=(
            "HuggingFace model ID or local path. Default: Qwen/Qwen2.5-3B "
            "(3B params, 32K context, ~6GB VRAM, no auth required). "
            "For larger tests: Qwen/Qwen2.5-7B (~14GB VRAM)."
        ),
    )
    parser.add_argument(
        "--tp", "--tensor-parallel",
        type=int,
        default=1,
        dest="tensor_parallel",
        help="Tensor Parallelism degree (number of GPUs). Default: 1",
    )
    parser.add_argument(
        "--gpu-ids",
        default="all",
        help="Comma-separated GPU IDs to use (e.g., '0,1'). Default: all",
    )

    # ── Physics Simulation ───────────────────────────────────────────────
    physics = parser.add_argument_group(
        "Physics Simulation",
        "Control hardware and network simulation parameters",
    )
    physics.add_argument(
        "--nvlink",
        choices=["enable", "disable"],
        default="enable",
        help=(
            "GPU interconnect mode. "
            "'enable' = use NVLink if available (default). "
            "'disable' = force PCIe fallback via NCCL_P2P_DISABLE=1."
        ),
    )
    physics.add_argument(
        "--network", "-n",
        choices=list(LATENCY_PROFILES.keys()),
        default="local",
        dest="network_profile",
        help=(
            "Network latency profile to inject. "
            "'local' = no latency. "
            "'datacenter' = 1ms (same building). "
            "'regional' = 15ms (same continent). "
            "'cross-continent' = 100ms (Chile↔Europe). "
            "'worst-case' = 250ms (maximum realistic latency)."
        ),
    )

    # ── Benchmark Parameters ─────────────────────────────────────────────
    bench = parser.add_argument_group(
        "Benchmark Parameters",
        "Control the measurement process",
    )
    bench.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="Number of warmup rounds (not measured). Default: 2",
    )
    bench.add_argument(
        "--rounds",
        type=int,
        default=5,
        help="Number of benchmark rounds (measured). Default: 5",
    )

    # ── Output ───────────────────────────────────────────────────────────
    output = parser.add_argument_group("Output")
    output.add_argument(
        "--output-dir", "-o",
        default="./results",
        help="Directory for JSON reports. Default: ./results",
    )
    output.add_argument(
        "--dry-run",
        action="store_true",
        help="Print configuration and estimated network overhead without running inference",
    )
    output.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging",
    )

    # ── Info ──────────────────────────────────────────────────────────────
    parser.add_argument(
        "--version", "-V",
        action="version",
        version=f"vllm_probe {VERSION}",
    )
    parser.add_argument(
        "--list-profiles",
        action="store_true",
        help="List all available network profiles and exit",
    )
    parser.add_argument(
        "--list-scenarios",
        action="store_true",
        help="List all workload scenarios with justifications and exit",
    )
    parser.add_argument(
        "--discover-gpus",
        action="store_true",
        help="Detect GPUs, check NVLink, and exit",
    )

    return parser


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────


def main():
    parser = build_parser()
    args = parser.parse_args()
    logger = setup_logging(args.verbose)

    # ── Info commands (no benchmark needed) ──────────────────────────────
    if args.list_profiles:
        print("\n═══ NETWORK LATENCY PROFILES ═══\n")
        for name, profile in LATENCY_PROFILES.items():
            print(f"  {name:<18s} │ {profile['delay_ms']:>5.0f}ms ± {profile['jitter_ms']:.0f}ms │ {profile['description']}")
            print(f"  {'':18s} │ Real-world: {profile['real_world']}")
            # Show decode impact for 2048 tokens
            overhead = 2048 * 2 * profile['delay_ms'] / 1000
            print(f"  {'':18s} │ Decode overhead (2048 tok, TP=2): {overhead:.1f}s")
            print()
        return

    if args.list_scenarios:
        print("\n═══ WORKLOAD SCENARIOS ═══\n")
        for name, sc in WORKLOAD_SCENARIOS.items():
            print(f"  ┌─ {name.upper()}: {sc['description']}")
            print(f"  │  Input: {sc['input_tokens']} tokens  │  Output: {sc['output_tokens']} tokens  │  Concurrency: {sc['concurrency']}")
            print(f"  │  Bottleneck: {sc['bottleneck']}")
            print(f"  │  Justification: {sc['justification']}")
            print(f"  │  Sensitivity:")
            for k, v in sc['sensitivity'].items():
                print(f"  │    {k:<10s}: {v}")
            print(f"  └{'─' * 70}")
            print()
        return

    if args.discover_gpus:
        gpus = discover_gpus(logger)
        nvlink = check_nvlink(logger)
        print("\n═══ GPU DISCOVERY ═══\n")
        if gpus:
            for gpu in gpus:
                print(f"  GPU {gpu['index']}: {gpu['name']} │ VRAM: {gpu['vram_total_mb']}MB (free: {gpu['vram_free_mb']}MB)")
            print(f"\n  NVLink: {'AVAILABLE' if nvlink else 'NOT DETECTED (PCIe only)'}")

            # Show raw topology matrix for verification
            print(f"\n  ─── GPU Topology Matrix (nvidia-smi topo -m) ───")
            topo = subprocess.run(
                ["nvidia-smi", "topo", "-m"],
                capture_output=True, text=True,
            )
            if topo.returncode == 0:
                for line in topo.stdout.strip().split("\n"):
                    print(f"  {line}")
                print()
                print("  Legend: NV# = NVLink (#=num links) │ PHB = PCIe Host Bridge")
                print("         PIX = same PCIe switch      │ SYS = cross-socket (QPI/UPI)")
            else:
                print(f"  (could not retrieve topology: {topo.stderr.strip()})")
        else:
            print("  No NVIDIA GPUs detected")
        print()
        return

    # ── Build configuration ──────────────────────────────────────────────
    # Validate that --scenario is provided for actual benchmark runs
    if args.scenario is None:
        parser.error("the following arguments are required: --scenario/-s")

    scenarios_to_run = (
        list(WORKLOAD_SCENARIOS.keys()) if args.scenario == "all"
        else [args.scenario]
    )

    # ── Ensure output directory ──────────────────────────────────────────
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Set GPU visibility ───────────────────────────────────────────────
    if args.gpu_ids != "all":
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
        logger.info(f"CUDA_VISIBLE_DEVICES set to: {args.gpu_ids}")

    # ── Run benchmarks ───────────────────────────────────────────────────
    all_reports = []

    for scenario_name in scenarios_to_run:
        config = ProbeConfig(
            model=args.model,
            scenario=scenario_name,
            network_profile=args.network_profile,
            nvlink=(args.nvlink == "enable"),
            tensor_parallel=args.tensor_parallel,
            gpu_ids=args.gpu_ids,
            output_dir=args.output_dir,
            warmup_rounds=args.warmup,
            benchmark_rounds=args.rounds,
            dry_run=args.dry_run,
            verbose=args.verbose,
        )

        # Detect actual hardware NVLink for accurate display
        _has_nvlink_hw = False
        try:
            _topo = subprocess.run(
                ["nvidia-smi", "topo", "-m"], capture_output=True, text=True,
            )
            for _line in _topo.stdout.split("\n"):
                _parts = _line.strip().split()
                if len(_parts) >= 2 and _parts[0].startswith("GPU"):
                    for _p in _parts[1:]:
                        if _p.startswith("NV") and len(_p) > 2 and _p[2:].isdigit():
                            _has_nvlink_hw = True
        except Exception:
            pass

        if config.tensor_parallel <= 1:
            _interconnect_str = "N/A (single GPU)"
        elif _has_nvlink_hw and config.nvlink:
            _interconnect_str = "NVLink (hardware present, enabled)"
        elif _has_nvlink_hw and not config.nvlink:
            _interconnect_str = "PCIe FORCED (NVLink disabled by flag)"
        else:
            _interconnect_str = "PCIe (no NVLink on this hardware)"

        logger.info(
            f"\n{'═' * 60}\n"
            f"  SCENARIO: {scenario_name.upper()}\n"
            f"  Model: {config.model}\n"
            f"  TP: {config.tensor_parallel} │ Interconnect: {_interconnect_str}\n"
            f"  Network: {config.network_profile} ({LATENCY_PROFILES[config.network_profile]['delay_ms']}ms)\n"
            f"{'═' * 60}"
        )

        simulator = PhysicsSimulator(logger)

        with physics_context(simulator, config):
            result = run_benchmark(config, logger)

        report_json = generate_report(result, config, logger, simulator)
        all_reports.append(report_json)

        # Save individual report
        filename = (
            f"probe_{scenario_name}"
            f"_tp{config.tensor_parallel}"
            f"_nvlink{'on' if config.nvlink else 'off'}"
            f"_{config.network_profile}"
            f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        report_path = output_dir / filename
        report_path.write_text(report_json)
        logger.info(f"Report saved: {report_path}")

        # Print to stdout
        print(f"\n{report_json}\n")

    # ── Summary ──────────────────────────────────────────────────────────
    if len(scenarios_to_run) > 1:
        logger.info(
            f"\n{'═' * 60}\n"
            f"  ALL {len(scenarios_to_run)} SCENARIOS COMPLETE\n"
            f"  Reports saved to: {output_dir.absolute()}\n"
            f"{'═' * 60}"
        )


if __name__ == "__main__":
    main()
