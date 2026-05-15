#!/bin/bash
# experiments_full.sh — Comprehensive split optimization experiments
# Tests all tensor routing configurations across all models

LLAMA_CLI="/home/deathnetworks/Decoupled Attn - PoC/llama.cpp-PoC/build_opencl/bin/llama-cli"
MODELS_DIR="/home/deathnetworks/Decoupled Attn - PoC/Models"
RESULTS_DIR="/home/deathnetworks/Decoupled Attn - PoC/results/experiments"
mkdir -p "$RESULTS_DIR"

CTX=256
N_TOKENS=32
TEMP=0.0
PROMPT="The quantum mechanical properties of electrons in a crystal lattice"

echo "============================================"
echo "Comprehensive Split Optimization Experiments"
echo "Ctx=$CTX, N=$N_TOKENS, Temp=$TEMP"
echo "============================================"

# All experiments: name|model|split_mode|env_vars
EXPERIMENTS=(
    # 2B model — all configurations
    "2B_GPU-ONLY|Qwen3.5-2B-Q4_K_M|local-gpu|"
    "2B_FFN-CPU|Qwen3.5-2B-Q4_K_M|local-ssd|"
    "2B_FFN+OTHER-CPU|Qwen3.5-2B-Q4_K_M|local-ssd|LLAMA_SPLIT_OTHER=cpu"
    "2B_FFN+ALL-CPU|Qwen3.5-2B-Q4_K_M|local-ssd|LLAMA_SPLIT_OTHER=all_cpu"

    # 4B model
    "4B_GPU-ONLY|Qwen3.5-4B-Q4_K_M|local-gpu|"
    "4B_FFN-CPU|Qwen3.5-4B-Q4_K_M|local-ssd|"
    "4B_FFN+OTHER-CPU|Qwen3.5-4B-Q4_K_M|local-ssd|LLAMA_SPLIT_OTHER=cpu"
    "4B_FFN+ALL-CPU|Qwen3.5-4B-Q4_K_M|local-ssd|LLAMA_SPLIT_OTHER=all_cpu"

    # 9B model
    "9B_GPU-ONLY|Qwen3.5-9B-Q4_K_M|local-gpu|"
    "9B_FFN-CPU|Qwen3.5-9B-Q4_K_M|local-ssd|"
    "9B_FFN+OTHER-CPU|Qwen3.5-9B-Q4_K_M|local-ssd|LLAMA_SPLIT_OTHER=cpu"
    "9B_FFN+ALL-CPU|Qwen3.5-9B-Q4_K_M|local-ssd|LLAMA_SPLIT_OTHER=all_cpu"

    # MoE model
    "MoE_GPU-ONLY|gemma-4-26B-A4B-it-UD-Q4_K_M|local-gpu|"
    "MoE_FFN-CPU|gemma-4-26B-A4B-it-UD-Q4_K_M|local-ssd|"
    "MoE_FFN+OTHER-CPU|gemma-4-26B-A4B-it-UD-Q4_K_M|local-ssd|LLAMA_SPLIT_OTHER=cpu"
    "MoE_FFN+ALL-CPU|gemma-4-26B-A4B-it-UD-Q4_K_M|local-ssd|LLAMA_SPLIT_OTHER=all_cpu"
)

# Table header
printf "%-30s %-10s %-10s %-12s %-12s %-10s %-10s\n" \
    "Experiment" "Model" "Config" "GPU VRAM" "Host RAM" "Prompt/s" "Gen/s"
printf "%-30s %-10s %-10s %-12s %-12s %-10s %-10s\n" \
    "------------------------------" "----------" "----------" "------------" "------------" "----------" "----------"

for EXP in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r NAME MODEL SPLIT_MODE EXTRA_ENV <<< "$EXP"
    MODEL_PATH="$MODELS_DIR/${MODEL}.gguf"

    if [ ! -f "$MODEL_PATH" ]; then
        printf "%-30s %-10s %-10s %-12s %-12s %-10s %-10s\n" \
            "$NAME" "$MODEL" "$SPLIT_MODE" "NOT FOUND" "-" "-" "-"
        continue
    fi

    RESULT_FILE="$RESULTS_DIR/full_${NAME}.txt"

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
        printf "%-30s %-10s %-10s %-12s %-12s %-10s %-10s\n" \
            "$NAME" "$MODEL" "$SPLIT_MODE" "TIMEOUT" "-" "-" "-"
        continue
    fi

    GPU_SELF=$(grep "GPUOpenCL" "$RESULT_FILE" | head -1 | grep -oP '\(\s*\K\d+(?=\s*=)' || echo "?")
    HOST_TOTAL=$(grep "^common_memory_breakdown_print.*Host" "$RESULT_FILE" | head -1 | grep -oP '\|\s*\K\d+(?=\s*=)' || echo "?")
    PROMPT_TPS=$(grep "Prompt:" "$RESULT_FILE" | head -1 | grep -oP 'Prompt:\s*\K[\d.]+' || echo "?")
    GEN_TPS=$(grep "Generation:" "$RESULT_FILE" | head -1 | grep -oP 'Generation:\s*\K[\d.]+' || echo "?")

    # Short model name
    SHORT_MODEL=$(echo "$MODEL" | sed 's/Qwen3.5-//' | sed 's/-Q4_K_M//' | sed 's/gemma-4-//' | sed 's/-it-UD-Q4_K_M//')

    printf "%-30s %-10s %-10s %-12s %-12s %-10s %-10s\n" \
        "$NAME" "$SHORT_MODEL" "$SPLIT_MODE" "${GPU_SELF}MiB" "${HOST_TOTAL}MiB" "$PROMPT_TPS" "$GEN_TPS"
done

echo ""
echo "============================================"
echo "All experiments complete"
echo "============================================"
