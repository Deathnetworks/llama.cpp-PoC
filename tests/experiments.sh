#!/bin/bash
# experiments.sh — Systematic split experiments
# Tests different tensor routing configurations to find optimal CPU/GPU split

LLAMA_CLI="/home/deathnetworks/Decoupled Attn - PoC/llama.cpp-PoC/build_opencl/bin/llama-cli"
MODELS_DIR="/home/deathnetworks/Decoupled Attn - PoC/Models"
RESULTS_DIR="/home/deathnetworks/Decoupled Attn - PoC/results/experiments"
mkdir -p "$RESULTS_DIR"

CTX=256
N_TOKENS=32
TEMP=0.0
PROMPT="The quantum mechanical properties of electrons in a crystal lattice"

# Experiment configurations
# Format: "name|model|split_mode|extra_env"
EXPERIMENTS=(
    # 2B model experiments
    "2B_GPU-ONLY|Qwen3.5-2B-Q4_K_M|local-gpu|"
    "2B_FFN-CPU|Qwen3.5-2B-Q4_K_M|local-ssd|"
    "2B_FFN+OTHER-CPU|Qwen3.5-2B-Q4_K_M|local-ssd|LLAMA_SPLIT_OTHER=cpu"

    # 4B model experiments
    "4B_GPU-ONLY|Qwen3.5-4B-Q4_K_M|local-gpu|"
    "4B_FFN-CPU|Qwen3.5-4B-Q4_K_M|local-ssd|"
    "4B_FFN+OTHER-CPU|Qwen3.5-4B-Q4_K_M|local-ssd|LLAMA_SPLIT_OTHER=cpu"

    # 9B model experiments
    "9B_GPU-ONLY|Qwen3.5-9B-Q4_K_M|local-gpu|"
    "9B_FFN-CPU|Qwen3.5-9B-Q4_K_M|local-ssd|"
    "9B_FFN+OTHER-CPU|Qwen3.5-9B-Q4_K_M|local-ssd|LLAMA_SPLIT_OTHER=cpu"

    # MoE model experiments
    "MoE_GPU-ONLY|gemma-4-26B-A4B-it-UD-Q4_K_M|local-gpu|"
    "MoE_FFN-CPU|gemma-4-26B-A4B-it-UD-Q4_K_M|local-ssd|"
    "MoE_FFN+OTHER-CPU|gemma-4-26B-A4B-it-UD-Q4_K_M|local-ssd|LLAMA_SPLIT_OTHER=cpu"
)

echo "============================================"
echo "Split Optimization Experiments"
echo "Ctx=$CTX, N=$N_TOKENS, Temp=$TEMP"
echo "============================================"

# Table header
printf "%-25s %-12s %-10s %-12s %-12s %-10s %-10s\n" \
    "Experiment" "Model" "Mode" "GPU VRAM" "Host RAM" "Prompt/s" "Gen/s"
printf "%-25s %-12s %-10s %-12s %-12s %-10s %-10s\n" \
    "-------------------------" "------------" "----------" "------------" "------------" "----------" "----------"

for EXP in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r NAME MODEL SPLIT_MODE EXTRA_ENV <<< "$EXP"
    MODEL_PATH="$MODELS_DIR/${MODEL}.gguf"

    if [ ! -f "$MODEL_PATH" ]; then
        printf "%-25s %-12s %-12s\n" "$NAME" "$MODEL" "NOT FOUND"
        continue
    fi

    RESULT_FILE="$RESULTS_DIR/${NAME}.txt"

    # Run with timeout
    timeout 600 env $EXTRA_ENV "$LLAMA_CLI" \
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
        printf "%-25s %-12s %-10s %-12s %-12s %-10s %-10s\n" \
            "$NAME" "$MODEL" "$SPLIT_MODE" "TIMEOUT" "-" "-" "-"
        continue
    fi

    # Extract metrics
    GPU_SELF=$(grep "GPUOpenCL" "$RESULT_FILE" | head -1 | grep -oP '\(\s*\K\d+(?=\s*=)' || echo "?")
    HOST_TOTAL=$(grep "^common_memory_breakdown_print.*Host" "$RESULT_FILE" | head -1 | grep -oP '\|\s*\K\d+(?=\s*=)' || echo "?")
        PROMPT_TPS=$(grep "Prompt:" "$RESULT_FILE" | head -1 | grep -oP 'Prompt:\s*\K[\d.]+' || echo "?")
    GEN_TPS=$(grep "Generation:" "$RESULT_FILE" | head -1 | grep -oP 'Generation:\s*\K[\d.]+' || echo "?")

    printf "%-25s %-12s %-10s %-12s %-12s %-10s %-10s\n" \
        "$NAME" "$MODEL" "$SPLIT_MODE" "${GPU_SELF} MiB" "${HOST_TOTAL} MiB" "$PROMPT_TPS" "$GEN_TPS"
done

echo ""
echo "============================================"
echo "Experiments complete. Results in $RESULTS_DIR"
echo "============================================"
