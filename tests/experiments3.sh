#!/bin/bash
# experiments3.sh — Partial FFN offload experiments
# Tests moving only specific FFN tensor types to CPU

LLAMA_CLI="/home/deathnetworks/Decoupled Attn - PoC/llama.cpp-PoC/build_opencl/bin/llama-cli"
MODELS_DIR="/home/deathnetworks/Decoupled Attn - PoC/Models"
RESULTS_DIR="/home/deathnetworks/Decoupled Attn - PoC/results/experiments"
mkdir -p "$RESULTS_DIR"

CTX=256
N_TOKENS=32
TEMP=0.0
PROMPT="The quantum mechanical properties of electrons in a crystal lattice"

echo "============================================"
echo "Partial FFN Offload Experiments"
echo "============================================"

MODEL="Qwen3.5-2B-Q4_K_M"
MODEL_PATH="$MODELS_DIR/${MODEL}.gguf"

# We need to test different is_ffn_tensor() configurations
# This requires modifying the code and rebuilding
# For now, let's test what we can with existing flags

# Test with different context lengths to see KV cache impact
for CTX in 128 256 512 1024 2048; do
    for MODE in "local-gpu" "local-ssd"; do
        RESULT_FILE="$RESULTS_DIR/exp3_ctx${CTX}_${MODE}.txt"
        echo "=== ctx=$CTX mode=$MODE ==="

        timeout 300 "$LLAMA_CLI" \
            -m "$MODEL_PATH" \
            -p "$PROMPT" \
            -n "$N_TOKENS" \
            --temp "$TEMP" \
            --single-turn \
            -c "$CTX" \
            --no-mmap \
            --reasoning off \
            --split-mode "$MODE" \
            -ngl 99 \
            > "$RESULT_FILE" 2>&1

        GPU_SELF=$(grep "GPUOpenCL" "$RESULT_FILE" | head -1 | grep -oP '\(\s*\K\d+(?=\s*=)' || echo "?")
        HOST_TOTAL=$(grep "^common_memory_breakdown_print.*Host" "$RESULT_FILE" | head -1 | grep -oP '\|\s*\K\d+(?=\s*=)' || echo "?")
        PROMPT_TPS=$(grep "Prompt:" "$RESULT_FILE" | head -1 | grep -oP 'Prompt:\s*\K[\d.]+' || echo "?")
        GEN_TPS=$(grep "Generation:" "$RESULT_FILE" | head -1 | grep -oP 'Generation:\s*\K[\d.]+' || echo "?")

        printf "  ctx=%-6s mode=%-10s GPU=%-8s Host=%-8s Prompt=%-8s Gen=%-8s\n" \
            "$CTX" "$MODE" "${GPU_SELF}MiB" "${HOST_TOTAL}MiB" "$PROMPT_TPS" "$GEN_TPS"
    done
done

echo ""
echo "============================================"
echo "Context length experiments complete"
echo "============================================"
