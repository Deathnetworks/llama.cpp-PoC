#!/usr/bin/env bash
# tests/run_all.sh
# Run all tests in correct order. Exit 1 if any fail.
# Usage: bash tests/run_all.sh [--skip-quality]

set -euo pipefail
PASS=0; FAIL=0

run() {
    local label="$1" cmd="$2"
    printf "▶ %-55s" "$label"
    if eval "$cmd" > /tmp/test_out.txt 2>&1; then
        ((PASS++)); echo " ✓ PASS"
    else
        ((FAIL++)); echo " ✗ FAIL"
        tail -5 /tmp/test_out.txt | sed 's/^/    /'
    fi
}

# ── Pre-flight: ensure model present ─────────────────────────────────────
run "Model: ensure Qwen3 extracted" \
    "python3 tests/helpers/ensure_model.py"

# ── Unit tests ────────────────────────────────────────────────────────────
run "Unit: routing predicate"      "python3 tests/unit/test_routing.py"
run "Unit: beta regression"        "python3 tests/unit/test_beta.py"
run "Unit: mmap flags in source"   "python3 tests/unit/test_mmap_flags.py"
run "Unit: f16 roundtrip"          "python3 tests/unit/test_f16_roundtrip.py"

# ── Phase 1 correctness: local-ssd == local-gpu ───────────────────────────
run "Phase1: local-ssd == local-gpu" \
    "diff <(llama.cpp-PoC/build/bin/llama-cli \
        -m models/Qwen3-0.6B-Q4_K_M.gguf \
        --split-mode local-gpu \
        -p 'The capital of France is' -n 5 \
        --temp 0 --log-disable) \
    tests/fixtures/baseline_local_gpu.txt"

# ── Integration ───────────────────────────────────────────────────────────
run "Integration: E2E correctness" \
    "python3 tests/integration/compare_outputs.py \
        --model models/Qwen3-0.6B-Q4_K_M.gguf \
        --prompt 'The capital of France is' \
        --n-tokens 10 --tolerance 0.05"

# ── Regression: default path unchanged ────────────────────────────────────
run "Regression: local-gpu unchanged" \
    "diff <(llama.cpp-PoC/build/bin/llama-cli \
        -m models/Qwen3-0.6B-Q4_K_M.gguf --split-mode local-gpu \
        -p 'The capital of France is' -n 5 \
        --temp 0 --log-disable) \
    tests/fixtures/baseline_local_gpu.txt"

# ── Quality tests ─────────────────────────────────────────────────────────
if [[ "${1:-}" != "--skip-quality" ]]; then
    run "Quality: PPL <= 1% delta" "python3 tests/quality/run_ppl.py"
    run "Quality: Needle"          "python3 tests/quality/run_needle.py"
    run "Quality: Coherency >=7/8" "python3 tests/quality/run_coherency.py"
    run "Quality: Benchmark"       "python3 tests/quality/run_benchmark.py"
fi

# ── Fork cleanliness ──────────────────────────────────────────────────────
run "Fork: only allowed files modified" \
    "! git -C llama.cpp-PoC diff --name-only HEAD \
     | grep -vE \
       'src/llama-ffn-local\.(h|cpp)|src/llama-model-loader\.(cpp|h)|\
src/llama-context\.(h|cpp)|src/llama\.cpp|\
examples/llama-cli/llama-cli\.cpp|examples/server/server\.cpp|\
tools/llama-slice/llama-slice\.cpp|CMakeLists\.txt|\
src/llama-graph\.cpp|src/models/qwen3\.cpp' \
     | grep ."

echo ""
echo "══════════════════════════════════════════"
echo " Results: ${PASS} passed, ${FAIL} failed"
echo "══════════════════════════════════════════"
[[ $FAIL -eq 0 ]]
