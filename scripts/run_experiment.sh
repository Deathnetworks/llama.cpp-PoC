#!/bin/bash
# run_experiment.sh - Run a single experiment and record results
# Usage: run_experiment.sh <experiment_number> <description> [extra_args]

EXP_NUM=$1
DESC=$2
shift 2
EXTRA_ARGS="$@"

MODEL="../models/Qwen3.5-4B-Q4_K_M.MTP.gguf"
PROMPT="The capital of France is a city located in the north-central part of the country."
N_TOKENS=64
CONTEXT=256

LOG_FILE="docs/EXPERIMENTS.md"

echo "=== Experiment $EXP_NUM: $DESC ==="
echo "Extra args: $EXTRA_ARGS"
echo ""

# Run 3 times and take the best
BEST_GEN=0
BEST_PROMPT=0
BEST_OUTPUT=""

for i in 1 2 3; do
    OUTPUT=$(timeout 300 ./build_opencl/bin/llama-cli \
        -m "$MODEL" \
        -p "$PROMPT" -n $N_TOKENS --temp 0 --single-turn \
        -c $CONTEXT --split-mode fnn-zero-cpu --stream -ngl 99 --jinja \
        --spec-type draft-mtp --spec-draft-n-max 2 \
        $EXTRA_ARGS 2>&1)
    
    PTPS=$(echo "$OUTPUT" | grep "Prompt:" | grep -oP '[\d.]+' | head -1)
    GTPS=$(echo "$OUTPUT" | grep "Generation:" | grep -oP '[\d.]+' | head -1)
    
    echo "  Run $i: Prompt=$PTPS t/s, Gen=$GTPS t/s"
    
    # Check if this is the best
    if (( $(echo "$GTPS > $BEST_GEN" | bc -l) )); then
        BEST_GEN=$GTPS
        BEST_PROMPT=$PTPS
        BEST_OUTPUT=$OUTPUT
    fi
done

echo ""
echo "=== Experiment $EXP_NUM Result ==="
echo "Best: Prompt=$BEST_PROMPT t/s, Generation=$BEST_GEN t/s"

# Check coherence
COHERENT="?"
if echo "$BEST_OUTPUT" | grep -qi "Paris\|france\|city\|capital"; then
    COHERENT="✅"
else
    COHERENT="❌"
fi
echo "Coherent: $COHERENT"

# Append to experiment log
echo "| $EXP_NUM | $DESC | $BEST_PROMPT | $BEST_GEN | $(echo "$BEST_GEN - 22.5" | bc) | $COHERENT | |" >> $LOG_FILE

echo ""
echo "Result appended to $LOG_FILE"
