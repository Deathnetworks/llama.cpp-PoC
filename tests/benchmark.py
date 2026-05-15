#!/usr/bin/env python3
"""
Comprehensive benchmark for decoupled attention/FFN inference.
Tests local-gpu vs local-ssd across multiple models with Q8 KV cache.

Metrics collected:
- VRAM usage (from memory breakdown)
- System RAM usage
- Time to first token
- Average time per token (warm)
- Output text (for coherency comparison)
- Tokens per second (prompt and generation)
"""

import subprocess
import os
import sys
import time
import json
import re

LLAMA_CLI = "/home/deathnetworks/Decoupled Attn - PoC/llama.cpp-PoC/build_opencl/bin/llama-cli"
MODELS_DIR = "/home/deathnetworks/Decoupled Attn - PoC/Models"
RESULTS_DIR = "/home/deathnetworks/Decoupled Attn - PoC/results"

CTX = 4096
N_TOKENS = 64
TEMP = 0.0
PROMPT = "The quantum mechanical properties of electrons in a crystal lattice can be described by Bloch's theorem, which states that"

os.makedirs(RESULTS_DIR, exist_ok=True)

def run_benchmark(model_path, split_mode, ctx=CTX, n_tokens=N_TOKENS, temp=TEMP, prompt=PROMPT):
    """Run llama-cli and collect metrics."""
    cmd = [
        LLAMA_CLI,
        "-m", model_path,
        "-p", prompt,
        "-n", str(n_tokens),
        "--temp", str(temp),
        "--single-turn",
        "-c", str(ctx),
        "-ctk", "q8_0",
        "-ctv", "q8_0",
        "--split-mode", split_mode,
        "-ngl", "99",
    ]

    print(f"    Running: {' '.join(cmd[-6:])}")

    start = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
        )
        elapsed = time.time() - start
    except subprocess.TimeoutExpired:
        print(f"    TIMEOUT after 600s")
        return None

    output = result.stdout + result.stderr

    # Parse metrics
    metrics = {
        "split_mode": split_mode,
        "wall_time_s": elapsed,
        "return_code": result.returncode,
    }

    # Memory breakdown
    for line in output.split("\n"):
        if "GPUOpenCL" in line and "memory breakdown" not in line:
            # Parse: GPUOpenCL (...) | total = free + (self = compute + model + context) + unaccounted
            m = re.search(r'\|\s*(\d+)\s*=\s*(\d+)\s*\(\s*(\d+)\s*=\s*(\d+)\s*\+\s*(\d+)\s*\+\s*(\d+)\s*\)', line)
            if m:
                metrics["gpu_total"] = int(m.group(1))
                metrics["gpu_free"] = int(m.group(2))
                metrics["gpu_self"] = int(m.group(3))
                metrics["gpu_compute"] = int(m.group(4))
                metrics["gpu_model"] = int(m.group(5))
                metrics["gpu_context"] = int(m.group(6))
        if "Host" in line and "memory breakdown" not in line and "GPUOpenCL" not in line:
            m = re.search(r'\|\s*(\d+)\s*=\s*(\d+)\s*\+\s*(\d+)\s*\+\s*(\d+)', line)
            if m:
                metrics["host_total"] = int(m.group(1))
                metrics["host_compute"] = int(m.group(2))
                metrics["host_model"] = int(m.group(3))
                metrics["host_context"] = int(m.group(4))

    # Performance metrics
    for line in output.split("\n"):
        if "Prompt:" in line and "t/s" in line:
            # Parse: [ Prompt: 238.1 t/s | Generation: 100.5 t/s ]
            m = re.search(r'Prompt:\s*([\d.]+)\s*t/s.*Generation:\s*([\d.]+)\s*t/s', line)
            if m:
                metrics["prompt_tps"] = float(m.group(1))
                metrics["gen_tps"] = float(m.group(2))

    # Extract generated text (lines after the prompt)
    gen_lines = []
    in_output = False
    for line in output.split("\n"):
        if line.startswith("> "):
            in_output = True
            continue
        if in_output and line.strip():
            gen_lines.append(line.strip())
    metrics["generated_text"] = " ".join(gen_lines[:20])
    metrics["generated_tokens"] = len(gen_lines)

    # Save raw output
    model_name = os.path.basename(model_path).replace(".gguf", "")
    raw_file = os.path.join(RESULTS_DIR, f"{model_name}_{split_mode}_raw.txt")
    with open(raw_file, "w") as f:
        f.write(output)

    return metrics


def main():
    models = [
        "Qwen3.5-2B-Q4_K_M.gguf",
        "Qwen3.5-4B-Q4_K_M.gguf",
        "Qwen3.5-9B-Q4_K_M.gguf",
        "Qwen3.5-27B-Q4_K_M.gguf",
    ]

    all_results = {}

    for model in models:
        model_path = os.path.join(MODELS_DIR, model)
        if not os.path.exists(model_path):
            print(f"SKIP: {model} not found")
            continue

        model_name = model.replace(".gguf", "")
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")

        model_results = {}

        for split in ["local-gpu", "local-ssd"]:
            print(f"\n  Testing: {split}")
            metrics = run_benchmark(model_path, split)
            if metrics:
                model_results[split] = metrics
                print(f"    GPU model: {metrics.get('gpu_model', '?')} MiB")
                print(f"    GPU self: {metrics.get('gpu_self', '?')} MiB")
                print(f"    Host total: {metrics.get('host_total', '?')} MiB")
                print(f"    Prompt: {metrics.get('prompt_tps', '?')} t/s")
                print(f"    Generation: {metrics.get('gen_tps', '?')} t/s")
                print(f"    Wall time: {metrics.get('wall_time_s', '?'):.1f}s")
            else:
                print(f"    FAILED or TIMEOUT")

        all_results[model_name] = model_results

    # Save all results
    results_file = os.path.join(RESULTS_DIR, "benchmark_results.json")
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary table
    print(f"\n\n{'='*80}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*80}")
    print(f"{'Model':<20} {'Mode':<12} {'GPU Model':>10} {'GPU Self':>10} {'Host':>8} {'Prompt/s':>10} {'Gen/s':>10} {'Wall(s)':>8}")
    print("-" * 80)

    for model_name, model_results in all_results.items():
        for split, metrics in model_results.items():
            print(f"{model_name:<20} {split:<12} "
                  f"{metrics.get('gpu_model', '-'):>10} "
                  f"{metrics.get('gpu_self', '-'):>10} "
                  f"{metrics.get('host_total', '-'):>8} "
                  f"{metrics.get('prompt_tps', '-'):>10.1f} "
                  f"{metrics.get('gen_tps', '-'):>10.1f} "
                  f"{metrics.get('wall_time_s', -1):>8.1f}")

    # Print VRAM savings
    print(f"\n{'='*80}")
    print("VRAM SAVINGS (local-gpu vs local-ssd)")
    print(f"{'='*80}")
    print(f"{'Model':<20} {'GPU Self (gpu)':>15} {'GPU Self (ssd)':>15} {'Savings':>10} {'Savings %':>10}")
    print("-" * 80)

    for model_name, model_results in all_results.items():
        if "local-gpu" in model_results and "local-ssd" in model_results:
            gpu_self = model_results["local-gpu"].get("gpu_self", 0)
            ssd_self = model_results["local-ssd"].get("gpu_self", 0)
            if gpu_self and ssd_self:
                savings = gpu_self - ssd_self
                pct = (savings / gpu_self) * 100
                print(f"{model_name:<20} {gpu_self:>15} {ssd_self:>15} {savings:>10} {pct:>9.1f}%")

    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
