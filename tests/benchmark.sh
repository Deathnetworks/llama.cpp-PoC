#!/bin/bash
# benchmark.sh - Comprehensive benchmark for decoupled attention/FFN
# Tests local-gpu vs local-ssd across multiple models with Q8 KV cache

set -e

LLAMA_CLI="/home/deathnetworks/Decoupled Attn - PoC/llama.cpp-PoC/build_opencl/bin/llama-cli"
MODELS_DIR="/home/deathnetworks/Decoupled Attn - PoC/Models"
RESULTS_DIR="/home/deathnetworks/Decoupled Attn - PoC/results"
mkdir -p "$RESULTS_DIR"

CTX=4096
PROMPT="The quantum mechanical properties of electrons in a crystal lattice can be described by Bloch's theorem, which states that"
N_TOKENS=64
TEMP=0.0

# Models to test (smallest to largest)
MODELS=(
    "Qwen3.5-2B-Q4_K_M.gguf"
    "Qwen3.5-4B-Q4_K_M.gguf"
    "Qwen3.5-9B-Q4_K_M.gguf"
    "Qwen3.5-27B-Q4_K_M.gguf"
)

SPLIT_MODES=("local-gpu" "local-ssd")

echo "============================================"
echo "Decoupled Attention/FFN Benchmark"
echo "Context: $CTX, Tokens: $N_TOKENS, Temp: $TEMP"
echo "KV Cache: Q8_0"
echo "============================================"

for MODEL in "${MODELS[@]}"; do
    MODEL_PATH="$MODELS_DIR/$MODEL"
    if [ ! -f "$MODEL_PATH" ]; then
        echo "SKIP: $MODEL not found"
        continue
    fi

    MODEL_NAME="${MODEL%.gguf}"
    echo ""
    echo "--------------------------------------------"
    echo "Model: $MODEL_NAME"
    echo "--------------------------------------------"

    for SPLIT in "${SPLIT_MODES[@]}"; do
        RESULT_FILE="$RESULTS_DIR/${MODEL_NAME}_${SPLIT}.txt"
        echo ""
        echo ">>> Testing: $SPLIT"

        # Run llama-cli and capture output
        # Use --single-turn to avoid interactive mode
        # Use -ctk q8_0 -ctv q8_0 for Q8 KV cache
        # Use -c $CTX for context size
        # Use -n $N_TOKENS for generation length
        # Use --temp $TEMP for deterministic output
        # Use --log-disable to reduce noise (but we need timing info)
        # Use --simple-io for cleaner output

        START_TIME=$(date +%s%N)

        timeout 600 "$LLAMA_CLI" \
            -m "$MODEL_PATH" \
            -p "$PROMPT" \
            -n "$N_TOKENS" \
            --temp "$TEMP" \
            --single-turn \
            -c "$CTX" \
            -ctk q8_0 \
            -ctv q8_0 \
            --split-mode "$SPLIT" \
            -ngl 99 \
            --simple-io \
            2>&1 | tee "$RESULT_FILE"

        END_TIME=$(date +%s%N)
        ELAPSED=$(( (END_TIME - START_TIME) / 1000000 ))

        echo "Total wall time: ${ELAPSED}ms" | tee -a "$RESULT_FILE"

        # Extract key metrics from the output
        echo "" | tee -a "$RESULT_FILE"
        echo "--- Extracted Metrics ---" | tee -a "$RESULT_FILE"

        # Memory breakdown
        grep "memory breakdown" "$RESULT_FILE" | tee -a "$RESULT_FILE" || true
        grep "GPUOpenCL" "$RESULT_FILE" | tee -a "$RESULT_FILE" || true
        grep "Host" "$RESULT_FILE" | tee -a "$RESULT_FILE" || true

        # Performance metrics
        grep "Prompt:" "$RESULT_FILE" | tee -a "$RESULT_FILE" || true

        # Extract generated text for coherency check
        echo "" | tee -a "$RESULT_FILE"
        echo "--- Generated Text ---" | tee -a "$RESULT_FILE"
        # The generated text comes after the prompt
        sed -n '/^> /,$ p' "$RESULT_FILE" | head -20 | tee -a "$RESULT_FILE" || true

        echo "" | tee -a "$RESULT_FILE"
        echo "=== End $SPLIT ===" | tee -a "$RESULT_FILE"
    done
done

echo ""
echo "============================================"
echo "Benchmark complete. Results in $RESULTS_DIR"
echo "============================================"
