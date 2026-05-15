#!/bin/bash
# experiments2.sh — Fine-grained split experiments
# Tests per-layer-type routing to find optimal CPU/GPU placement

LLAMA_CLI="/home/deathnetworks/Decoupled Attn - PoC/llama.cpp-PoC/build_opencl/bin/llama-cli"
MODELS_DIR="/home/deathnetworks/Decoupled Attn - PoC/Models"
RESULTS_DIR="/home/deathnetworks/Decoupled Attn - PoC/results/experiments"
mkdir -p "$RESULTS_DIR"

CTX=256
N_TOKENS=32
TEMP=0.0
PROMPT="The quantum mechanical properties of electrons in a crystal lattice"

echo "============================================"
echo "Fine-Grained Split Experiments"
echo "============================================"

# Use 2B model for speed (only one that completes quickly on HDD)
MODEL="Qwen3.5-2B-Q4_K_M"
MODEL_PATH="$MODELS_DIR/${MODEL}.gguf"

# Experiment configurations for 2B model
# Testing different combinations of tensor placement
EXPERIMENTS=(
    # Baseline
    "2B_GPU-ONLY|local-gpu||"

    # Current FNN-RAM-CPU (FFN on CPU, rest on GPU)
    "2B_FFN-CPU|local-ssd||"

    # FFN + SSM on CPU, Attention on GPU
    "2B_FFN+SSM-CPU|local-ssd|LLAMA_SPLIT_OTHER=cpu"

    # Test: What if we keep FFN on GPU but move SSM to CPU?
    # This requires a custom split mode — we'll simulate by overriding specific tensors
    # Not directly supported, so we skip for now

    # Test: Embedding on CPU (saves ~398 MB but may slow down)
    # Not directly supported in current implementation

    # Test: Output head on CPU (saves ~13 MB)
    # Not directly supported in current implementation
)

printf "%-30s %-12s %-12s %-12s %-10s %-10s\n" \
    "Experiment" "GPU VRAM" "Host RAM" "VRAM Saved" "Prompt/s" "Gen/s"
printf "%-30s %-12s %-12s %-12s %-10s %-10s\n" \
    "------------------------------" "------------" "------------" "------------" "----------" "----------"

BASELINE_GPU=1479

for EXP in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r NAME SPLIT_MODE EXTRA_ENV <<< "$EXP"
    RESULT_FILE="$RESULTS_DIR/exp2_${NAME}.txt"

    timeout 300 env $EXTRA_ENV "$LLAMA_CLI" \
        -m "$MODEL_PATH" \
        -p "$PROMPT" \
        -n "$N_TOKENS" \
        --temp "$TEMP" \
        --single-turn \
        -c "$CTX" \
        --no-mmap \
        --reasoning off \
        --split-mode "$SPLIT_MODE" \
        -ngl 99 \
        > "$RESULT_FILE" 2>&1

    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 124 ]; then
        printf "%-30s %-12s %-12s %-12s %-10s %-10s\n" \
            "$NAME" "TIMEOUT" "-" "-" "-" "-"
        continue
    fi

    GPU_SELF=$(grep "GPUOpenCL" "$RESULT_FILE" | head -1 | grep -oP '\(\s*\K\d+(?=\s*=)' || echo "?")
    HOST_TOTAL=$(grep "^common_memory_breakdown_print.*Host" "$RESULT_FILE" | head -1 | grep -oP '\|\s*\K\d+(?=\s*=)' || echo "?")
    PROMPT_TPS=$(grep "Prompt:" "$RESULT_FILE" | head -1 | grep -oP 'Prompt:\s*\K[\d.]+' || echo "?")
    GEN_TPS=$(grep "Generation:" "$RESULT_FILE" | head -1 | grep -oP 'Generation:\s*\K[\d.]+' || echo "?")

    if [ "$GPU_SELF" != "?" ]; then
        VRAM_SAVED=$((BASELINE_GPU - GPU_SELF))
        VRAM_PCT=$(echo "scale=1; $VRAM_SAVED * 100 / $BASELINE_GPU" | bc 2>/dev/null || echo "?")
        SAVED_STR="${VRAM_SAVED} MiB (${VRAM_PCT}%)"
    else
        SAVED_STR="?"
    fi

    printf "%-30s %-12s %-12s %-12s %-10s %-10s\n" \
        "$NAME" "${GPU_SELF} MiB" "${HOST_TOTAL} MiB" "$SAVED_STR" "$PROMPT_TPS" "$GEN_TPS"
done

echo ""
echo "============================================"
echo "Fine-grained experiments complete"
echo "============================================"
