#!/usr/bin/env python3
"""
run_experiments.py - Automated MTP+Stream optimization experiments
Runs experiments, tracks results, applies best optimizations
"""

import subprocess
import re
import os
import time
import json
from datetime import datetime

PROJECT_DIR = "/home/deathnetworks/Decoupled Attn - PoC/llama.cpp-PoC"
MODEL = "../models/Qwen3.5-4B-Q4_K_M.MTP.gguf"
RESULTS_FILE = "docs/EXPERIMENTS.md"
LOG_DIR = "experiment_logs"

def run_llama(prompt, n_tokens, context, extra_args="", timeout=300):
    """Run llama-cli and return (output, prompt_tps, gen_tps)"""
    cmd = [
        "./build_opencl/bin/llama-cli",
        "-m", MODEL,
        "-p", prompt,
        "-n", str(n_tokens),
        "--temp", "0",
        "--single-turn",
        "-c", str(context),
        "--split-mode", "fnn-zero-cpu",
        "--stream",
        "-ngl", "99",
        "--jinja",
        "--spec-type", "draft-mtp",
        "--spec-draft-n-max", "2",
    ]
    if extra_args:
        cmd.extend(extra_args.split())
    
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            cwd=PROJECT_DIR, env={**os.environ, "PATH": f"/tmp/venv/bin:{os.environ.get('PATH', '')}"}
        )
        output = result.stdout + result.stderr
        
        # Parse tps values - look for patterns like "Prompt: 25.3 t/s"
        prompt_match = re.search(r'Prompt:\s*([\d.]+)\s*t/s', output)
        gen_match = re.search(r'Generation:\s*([\d.]+)\s*t/s', output)
        
        prompt_tps = float(prompt_match.group(1)) if prompt_match else 0.0
        gen_tps = float(gen_match.group(1)) if gen_match else 0.0
        
        return output, prompt_tps, gen_tps
    except subprocess.TimeoutExpired:
        return "", 0.0, 0.0
    except Exception as e:
        return str(e), 0.0, 0.0

def check_coherence(output):
    """Check if output is coherent (contains expected words, not garbled)"""
    # Remove debug lines
    clean = '\n'.join(line for line in output.split('\n') 
                      if not line.startswith('DEBUG ') and line.strip())
    
    # Check for expected output
    if re.search(r'Paris|paris', clean, re.IGNORECASE):
        return True, "Paris found"
    if re.search(r'capital|France|city', clean, re.IGNORECASE):
        return True, "Related word found"
    
    # Check for garbled Chinese characters that indicate corruption
    garbled_patterns = [r'任何', r'还是', r'符合', r'的\n\n', r'是\n\n']
    for pattern in garbled_patterns:
        if re.search(pattern, clean):
            return False, f"Garbled: {pattern}"
    
    return None, "Unknown"

def run_experiment(exp_num, description, prompt, n_tokens, context, extra_args=""):
    """Run a single experiment and return results"""
    print(f"\n=== Experiment {exp_num}: {description} ===")
    
    output, ptps, gtps = run_llama(prompt, n_tokens, context, extra_args)
    coherent, coherence_note = check_coherence(output)
    
    print(f"  Prompt: {ptps:.1f} t/s, Gen: {gtps:.1f} t/s, Coherent: {coherent} ({coherence_note})")
    
    # Save output
    with open(f"{LOG_DIR}/exp_{exp_num:03d}.txt", "w") as f:
        f.write(f"Experiment {exp_num}: {description}\n")
        f.write(f"Prompt: {ptps:.1f} t/s, Gen: {gtps:.1f} t/s\n")
        f.write(f"Coherent: {coherent} ({coherence_note})\n\n")
        f.write(output)
    
    return {
        "num": exp_num,
        "description": description,
        "prompt_tps": ptps,
        "gen_tps": gtps,
        "coherent": coherent,
        "coherence_note": coherence_note,
        "extra_args": extra_args,
        "n_tokens": n_tokens,
        "context": context,
    }

def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("=" * 60)
    print("MTP+Stream Optimization Experiments")
    print("=" * 60)
    print(f"Started: {timestamp}")
    print()
    
    # Build first
    print("Building...")
    subprocess.run(
        ["cmake", "--build", "build_opencl", "--target", "llama-cli", "-j", str(os.cpu_count() or 4)],
        cwd=PROJECT_DIR, capture_output=True,
        env={**os.environ, "PATH": f"/tmp/venv/bin:{os.environ.get('PATH', '')}"}
    )
    print("Build complete.")
    
    # Baseline
    print("\n=== Baseline (3 runs) ===")
    baseline_runs = []
    for i in range(3):
        output, ptps, gtps = run_llama("What is the capital of France?", 64, 256)
        baseline_runs.append((ptps, gtps))
        print(f"  Run {i+1}: Prompt={ptps:.1f}, Gen={gtps:.1f}")
    
    avg_prompt = sum(r[0] for r in baseline_runs) / len(baseline_runs)
    avg_gen = sum(r[1] for r in baseline_runs) / len(baseline_runs)
    print(f"  Average: Prompt={avg_prompt:.1f}, Gen={avg_gen:.1f}")
    
    # Define experiments
    experiments = [
        # MTP tuning
        ("MTP n=1", "What is the capital of France?", 64, 256, "--spec-draft-n-max 1"),
        ("MTP n=2", "What is the capital of France?", 64, 256, "--spec-draft-n-max 2"),
        ("MTP n=3", "What is the capital of France?", 64, 256, "--spec-draft-n-max 3"),
        ("MTP n=4", "What is the capital of France?", 64, 256, "--spec-draft-n-max 4"),
        ("No MTP", "What is the capital of France?", 64, 256, ""),
        
        # Context length
        ("Ctx=64", "What is the capital of France?", 32, 64, ""),
        ("Ctx=128", "What is the capital of France?", 64, 128, ""),
        ("Ctx=256", "What is the capital of France?", 64, 256, ""),
        ("Ctx=512", "What is the capital of France?", 64, 512, ""),
        
        # Token count
        ("32 tokens", "What is the capital of France?", 32, 128, ""),
        ("64 tokens", "What is the capital of France?", 64, 256, ""),
        ("128 tokens", "What is the capital of France?", 128, 512, ""),
        ("256 tokens", "What is the capital of France?", 256, 1024, ""),
        
        # Prompt style
        ("Direct", "What is the capital of France?", 64, 256, ""),
        ("Indirect", "Can you tell me the capital of France?", 64, 256, ""),
        ("Statement", "The capital of France is", 64, 256, ""),
        ("One word", "Capital of France? One word:", 64, 256, ""),
        ("French", "Quelle est la capitale de la France?", 64, 256, ""),
        
        # Combined optimizations
        ("MTP n=2, Ctx=128", "What is the capital of France?", 64, 128, ""),
        ("MTP n=2, Ctx=64", "What is the capital of France?", 32, 64, ""),
        ("MTP n=1, Ctx=128", "What is the capital of France?", 64, 128, "--spec-draft-n-max 1"),
        ("MTP n=3, Ctx=128", "What is the capital of France?", 64, 128, "--spec-draft-n-max 3"),
    ]
    
    # Run experiments
    results = []
    best_gen = avg_gen
    best_exp = None
    
    for i, (desc, prompt, n_tokens, context, extra_args) in enumerate(experiments):
        exp_num = i + 1
        result = run_experiment(exp_num, desc, prompt, n_tokens, context, extra_args)
        results.append(result)
        
        if result["coherent"] and result["gen_tps"] > best_gen:
            best_gen = result["gen_tps"]
            best_exp = result
            print(f"  *** New best: {best_gen:.1f} t/s ***")
    
    # Write results
    with open(RESULTS_FILE, "w") as f:
        f.write("# MTP+Stream Optimization Experiments\n\n")
        f.write(f"## Baseline\n")
        f.write(f"- Prompt: {avg_prompt:.1f} t/s\n")
        f.write(f"- Generation: {avg_gen:.1f} t/s\n\n")
        
        if best_exp:
            f.write(f"## Best Result\n")
            f.write(f"- Experiment: {best_exp['num']}\n")
            f.write(f"- Description: {best_exp['description']}\n")
            f.write(f"- Generation: {best_exp['gen_tps']:.1f} t/s\n")
            f.write(f"- Improvement: {best_exp['gen_tps'] - avg_gen:+.1f} t/s\n\n")
        
        f.write("## Experiment Log\n\n")
        f.write("| # | Description | Prompt t/s | Gen t/s | Coherent | Notes |\n")
        f.write("|---|-------------|------------|---------|----------|-------|\n")
        
        for r in results:
            coherent_str = "✅" if r["coherent"] == True else ("❌" if r["coherent"] == False else "?")
            delta = r["gen_tps"] - avg_gen
            f.write(f"| {r['num']} | {r['description']} | {r['prompt_tps']:.1f} | {r['gen_tps']:.1f} | {coherent_str} | {r['coherence_note']} |\n")
    
    print("\n" + "=" * 60)
    print("Experiments Complete!")
    print("=" * 60)
    if best_exp:
        print(f"Best: Experiment {best_exp['num']} - {best_exp['description']}")
        print(f"Best Gen t/s: {best_gen:.1f}")
        print(f"Improvement: {best_gen - avg_gen:+.1f} t/s")
    print(f"Results saved to: {RESULTS_FILE}")

if __name__ == "__main__":
    main()
