import subprocess
import os
import sys
import re
import time

MODEL_PATH = r"d:\User Files\Desktop\Decoupled Attn - PoC\models\Qwen3.5-9B-Q4_K_M.gguf"
PPL_FILE = r"d:\User Files\Desktop\Decoupled Attn - PoC\llama.cpp-PoC\scratch\ppl_test.txt"
BIN_DIR = r"d:\User Files\Desktop\Decoupled Attn - PoC\llama.cpp-PoC\build\bin"

MODES = [
    "gpu-only",
    "fnn-ram-cpu",
    "fnn-ram-cpu-other",
    "fnn-ram-cpu-all",
    "fnn-zero-cpu",
    "fnn-zero-cpu-other",
    "fnn-zero-cpu-all"
]

def run_command(cmd, timeout=600):
    env = os.environ.copy()
    # Construct the full command string
    # We use ^ to escape characters in the cmd string for the nested shell if needed,
    # but here we just pass it as a single string to cmd /c.
    full_cmd = f'call "C:\\Program Files (x86)\\Intel\\oneAPI\\setvars.bat" intel64 --force > nul && {cmd}'
    
    start_time = time.time()
    try:
        # We use shell=True to allow 'call' and '&&'
        result = subprocess.run(full_cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        duration = time.time() - start_time
        return result.stdout, result.stderr, result.returncode, duration
    except subprocess.TimeoutExpired as e:
        duration = time.time() - start_time
        return e.stdout.decode() if e.stdout else "", e.stderr.decode() if e.stderr else "Timeout", -1, duration

results = []

print(f"Starting tests with model: {MODEL_PATH}")

for mode in MODES:
    print(f"\nTesting mode: {mode}")
    
    # 1. Coherency Test
    print(f"  Running coherency test...")
    # Use --simple-io to avoid interactive mode and [Start thinking] blocks
    cli_cmd = f'"{os.path.join(BIN_DIR, "llama-cli.exe")}" -m "{MODEL_PATH}" -p "The capital of France is" -n 10 --split-mode {mode} -c 128 --simple-io'
    stdout, stderr, code, duration = run_command(cli_cmd)
    
    if code != 0:
        print(f"    Error: llama-cli exited with code {code}")
        print(f"    Stderr: {stderr[:200]}...")
    
    # In simple-io mode, stdout should contain the generated text
    coherent = "Paris" in stdout or "Paris" in stderr
    gen_text = stdout.strip() if coherent else "FAIL"
    
    # 2. Perplexity Test
    print(f"  Running perplexity test...")
    ppl_cmd = f'"{os.path.join(BIN_DIR, "llama-perplexity.exe")}" -m "{MODEL_PATH}" -f "{PPL_FILE}" --split-mode {mode} -c 512'
    stdout_ppl, stderr_ppl, code_ppl, duration_ppl = run_command(ppl_cmd)
    
    if code_ppl != 0:
        print(f"    Error: llama-perplexity exited with code {code_ppl}")
        # print(f"    Stderr: {stderr_ppl[:200]}...")

    # Extract PPL score
    # Look for "Final estimate: PPL = 12.3456" in stdout or stderr
    combined_ppl = stdout_ppl + stderr_ppl
    ppl_match = re.search(r"Final estimate: PPL\s+=\s+([\d\.]+)", combined_ppl)
    ppl_score = ppl_match.group(1) if ppl_match else "N/A"
    
    results.append({
        "mode": mode,
        "coherent": "PASS" if coherent else "FAIL",
        "gen_text": gen_text.replace("\n", " ")[:50],
        "ppl": ppl_score,
        "time_ppl": f"{duration_ppl:.2f}s"
    })

# Output Markdown Table
print("\n\n### Split Mode Test Results\n")
print("| Mode | Coherency | Sample Output | PPL Score | PPL Duration |")
print("|------|-----------|---------------|-----------|--------------|")
for r in results:
    print(f"| {r['mode']} | {r['coherent']} | {r['gen_text']} | {r['ppl']} | {r['time_ppl']} |")
