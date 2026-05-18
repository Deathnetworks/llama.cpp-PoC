#!/bin/bash
# baseline_measure.sh - Establish baseline t/s with MTP+stream
# Run 5 times and take the median

MODEL="../models/Qwen3.5-4B-Q4_K_M.MTP.gguf"
PROMPT="The capital of France is"
N_TOKENS=128
CONTEXT=256
N_RUNS=5

PROMPT_TPS=""
GEN_TPS=""

for i in $(seq 1 $N_RUNS); do
    echo "=== Run $i ==="
    OUTPUT=$(timeout 300 ./build_opencl/bin/llama-cli \
        -m "$MODEL" \
        -p "$PROMPT" -n $N_TOKENS --temp 0 --single-turn \
        -c $CONTEXT --split-mode fnn-zero-cpu --stream -ngl 99 --jinja \
        --spec-type draft-mtp --spec-draft-n-max 2 2>&1)
    
    PTPS=$(echo "$OUTPUT" | grep "Prompt:" | grep -oP '[\d.]+' | head -1)
    GTPS=$(echo "$OUTPUT" | grep "Generation:" | grep -oP '[\d.]+' | head -1)
    
    echo "  Prompt: $PTPS t/s | Generation: $GTPS t/s"
    PROMPT_TPS="$PROMPT_TPS $PTPS"
    GEN_TPS="$GEN_TPS $GTPS"
done

echo ""
echo "=== Baseline Results ==="
echo "Prompt t/s: $PROMPT_TPS"
echo "Generation t/s: $GEN_TPS"

# Calculate median
echo ""
echo "Prompt median: $(echo $PROMPT_TPS | tr ' ' '\n' | sort -n | head -1)"
echo "Generation median: $(echo $GEN_TPS | tr ' ' '\n' | sort -n | head -1)"
