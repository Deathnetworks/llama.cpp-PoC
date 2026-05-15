#!/bin/bash
# benchmark_baseline.sh — Establish baselines for all models
# Uses ctx=256, n=32, temp=0, --no-mmap, --reasoning off

LLAMA_CLI="/home/deathnetworks/Decoupled Attn - PoC/llama.cpp-PoC/build_opencl/bin/llama-cli"
MODELS_DIR="/home/deathnetworks/Decoupled Attn - PoC/Models"
RESULTS_DIR="/home/deathnetworks/Decoupled Attn - PoC/results"
mkdir -p "$RESULTS_DIR"

CTX=256
N_TOKENS=32
TEMP=0.0
PROMPT="The quantum mechanical properties of electrons in a crystal lattice"

echo "============================================"
echo "Baseline Benchmark"
echo "Ctx=$CTX, N=$N_TOKENS, Temp=$TEMP"
echo "============================================"

for MODEL in Qwen3.5-2B-Q4_K_M Qwen3.5-4B-Q4_K_M Qwen3.5-9B-Q4_K_M; do
    MODEL_PATH="$MODELS_DIR/${MODEL}.gguf"
    if [ ! -f "$MODEL_PATH" ]; then
        echo "SKIP: $MODEL not found"
        continue
    fi

    for MODE in "local-gpu" "local-ssd"; do
        RESULT_FILE="$RESULTS_DIR/baseline_${MODEL}_${MODE}.txt"
        echo ""
        echo "=== $MODEL $MODE ==="

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

        # Extract metrics
        GPU_LINE=$(grep "GPUOpenCL" "$RESULT_FILE" | head -1)
        HOST_LINE=$(grep "^common_memory_breakdown_print.*Host" "$RESULT_FILE" | head -1)
        PERF_LINE=$(grep "Prompt:" "$RESULT_FILE" | head -1)

        echo "  GPU: $GPU_LINE"
        echo "  Host: $HOST_LINE"
        echo "  Perf: $PERF_LINE"
    done
done

echo ""
echo "============================================"
echo "Baseline complete"
echo "============================================"
