#!/bin/bash
# run_100_experiments.sh - Automated MTP+Stream optimization experiments
# Runs 100 experiments, tracks results, applies best optimizations

set -e

PROJECT_DIR="/home/deathnetworks/Decoupled Attn - PoC/llama.cpp-PoC"
MODEL="../models/Qwen3.5-4B-Q4_K_M.MTP.gguf"
RESULTS_FILE="docs/EXPERIMENTS.md"
BEST_CONFIG_FILE="docs/BEST_CONFIG.md"
LOG_DIR="experiment_logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p "$LOG_DIR"

# Baseline measurement (average of 3 runs)
baseline_gen_tps=0
baseline_prompt_tps=0

run_baseline() {
    echo "=== Establishing Baseline ==="
    local total_gen=0
    local total_prompt=0
    local runs=3
    
    for i in $(seq 1 $runs); do
        local output=$(timeout 300 ./build_opencl/bin/llama-cli \
            -m "$MODEL" \
            -p "What is the capital of France?" -n 64 --temp 0 --single-turn \
            -c 256 --split-mode fnn-zero-cpu --stream -ngl 99 --jinja 2>&1)
        
        local ptps=$(echo "$output" | grep "Prompt:" | sed 's/.*Prompt: *//' | sed 's/ .*//' | grep -oE '[0-9]+\.[0-9]+' | head -1)
        local gtps=$(echo "$output" | grep "Generation:" | sed 's/.*Generation: *//' | sed 's/ .*//' | grep -oE '[0-9]+\.[0-9]+' | head -1)
        [ -z "$ptps" ] && ptps="0.0"
        [ -z "$gtps" ] && gtps="0.0"
        
        total_prompt=$(echo "$total_prompt + $ptps" | bc)
        total_gen=$(echo "$total_gen + $gtps" | bc)
        
        echo "  Run $i: Prompt=$ptps t/s, Gen=$gtps t/s"
    done
    
    baseline_prompt_tps=$(echo "scale=2; $total_prompt / $runs" | bc)
    baseline_gen_tps=$(echo "scale=2; $total_gen / $runs" | bc)
    
    echo "Baseline: Prompt=${baseline_prompt_tps} t/s, Gen=${baseline_gen_tps} t/s"
}

# Run a single experiment
run_experiment() {
    local exp_num=$1
    local description=$2
    local extra_args=$3
    local n_tokens=$4
    local context=$5
    local prompt=$6
    
    echo "=== Experiment $exp_num: $description ==="
    
    local output=$(timeout 300 ./build_opencl/bin/llama-cli \
        -m "$MODEL" \
        -p "$prompt" -n $n_tokens --temp 0 --single-turn \
        -c $context --split-mode fnn-zero-cpu --stream -ngl 99 --jinja \
        --spec-type draft-mtp --spec-draft-n-max 2 \
        $extra_args 2>&1)
    
    local ptps=$(echo "$output" | grep "Prompt:" | sed 's/.*Prompt: *//' | sed 's/ .*//' | grep -oE '[0-9]+\.[0-9]+' | head -1)
    local gtps=$(echo "$output" | grep "Generation:" | sed 's/.*Generation: *//' | sed 's/ .*//' | grep -oE '[0-9]+\.[0-9]+' | head -1)
    
    # Handle empty values
    if [ -z "$ptps" ]; then ptps="0.0"; fi
    if [ -z "$gtps" ]; then gtps="0.0"; fi
    
    # Check coherence (does output contain expected words?)
    # Filter out debug lines first
    local clean_output=$(echo "$output" | grep -v "DEBUG STREAM\|DEBUG LOAD\|DEBUG WEIGHT")
    local coherent="?"
    if echo "$clean_output" | grep -qi "Paris"; then
        coherent="✅"
    elif echo "$clean_output" | grep -qi "任何\|还是\|符合"; then
        coherent="❌"
    fi
    
    local delta=$(echo "scale=2; $gtps - $baseline_gen_tps" | bc)
    
    echo "  Result: Prompt=$ptps t/s, Gen=$gtps t/s, Δ=$delta, Coherent=$coherent"
    
    # Log result
    echo "| $exp_num | $description | $ptps | $gtps | $delta | $coherent |" >> "$RESULTS_FILE"
    
    # Save full output for debugging
    echo "$output" > "$LOG_DIR/exp_${exp_num}_${TIMESTAMP}.txt"
    
    # Return gen tps for comparison
    echo "$gtps"
}

# ============================================================
# EXPERIMENT DEFINITIONS (100 experiments)
# ============================================================

# Experiment configurations are defined as arrays
# Format: "description|extra_args|n_tokens|context|prompt"

declare -a EXPERIMENTS=(
    # Category 1: MTP tuning (experiments 1-15)
    "MTP n_max=1|--spec-draft-n-max 1|64|256|What is the capital of France?"
    "MTP n_max=2|--spec-draft-n-max 2|64|256|What is the capital of France?"
    "MTP n_max=3|--spec-draft-n-max 3|64|256|What is the capital of France?"
    "MTP n_max=4|--spec-draft-n-max 4|64|256|What is the capital of France?"
    "MTP n_max=5|--spec-draft-n-max 5|64|256|What is the capital of France?"
    "MTP n_max=2, 128 tokens|--spec-draft-n-max 2|128|512|What is the capital of France?"
    "MTP n_max=2, 256 tokens|--spec-draft-n-max 2|256|1024|What is the capital of France?"
    "MTP n_max=2, 512 tokens|--spec-draft-n-max 2|512|2048|What is the capital of France?"
    "MTP n_max=3, 128 tokens|--spec-draft-n-max 3|128|512|What is the capital of France?"
    "MTP n_max=3, 256 tokens|--spec-draft-n-max 3|256|1024|What is the capital of France?"
    "No MTP||64|256|What is the capital of France?"
    "No MTP, 128 tokens||128|512|What is the capital of France?"
    "No MTP, 256 tokens||256|1024|What is the capital of France?"
    "MTP n_max=2, short prompt|--spec-draft-n-max 2|64|128|Hi"
    "MTP n_max=2, long prompt|--spec-draft-n-max 2|64|256|The capital of France is a well-known city that serves as the political, economic, and cultural center of the country."
    
    # Category 2: Context length tuning (experiments 16-30)
    "Context 64|--spec-draft-n-max 2|32|64|What is the capital of France?"
    "Context 128|--spec-draft-n-max 2|64|128|What is the capital of France?"
    "Context 256|--spec-draft-n-max 2|64|256|What is the capital of France?"
    "Context 512|--spec-draft-n-max 2|64|512|What is the capital of France?"
    "Context 1024|--spec-draft-n-max 2|64|1024|What is the capital of France?"
    "Context 64, 128 tokens|--spec-draft-n-max 2|128|64|What is the capital of France?"
    "Context 128, 128 tokens|--spec-draft-n-max 2|128|128|What is the capital of France?"
    "Context 256, 128 tokens|--spec-draft-n-max 2|128|256|What is the capital of France?"
    "Context 512, 128 tokens|--spec-draft-n-max 2|128|512|What is the capital of France?"
    "Context 1024, 128 tokens|--spec-draft-n-max 2|128|1024|What is the capital of France?"
    "Context 64, 256 tokens|--spec-draft-n-max 2|256|64|What is the capital of France?"
    "Context 128, 256 tokens|--spec-draft-n-max 2|256|128|What is the capital of France?"
    "Context 256, 256 tokens|--spec-draft-n-max 2|256|256|What is the capital of France?"
    "Context 512, 256 tokens|--spec-draft-n-max 2|256|512|What is the capital of France?"
    "Context 1024, 256 tokens|--spec-draft-n-max 2|256|1024|What is the capital of France?"
    
    # Category 3: Prompt variations (experiments 31-45)
    "Prompt: direct question|--spec-draft-n-max 2|64|256|What is the capital of France?"
    "Prompt: indirect question|--spec-draft-n-max 2|64|256|Can you tell me the capital of France?"
    "Prompt: statement|--spec-draft-n-max 2|64|256|The capital of France is"
    "Prompt: fill in blank|--spec-draft-n-max 2|64|256|The capital of France is ___"
    "Prompt: one word|--spec-draft-n-max 2|64|256|Capital of France? One word:"
    "Prompt: JSON format|--spec-draft-n-max 2|64|256|{\"question\": \"capital of France\", \"answer\":"
    "Prompt: code context|--spec-draft-n-max 2|64|256|# The capital of France is"
    "Prompt: list format|--spec-draft-n-max 2|64|256|- Capital of France:"
    "Prompt: French|--spec-draft-n-max 2|64|256|Quelle est la capitale de la France?"
    "Prompt: German|--spec-draft-n-max 2|64|256|Was ist die Hauptstadt von Frankreich?"
    "Prompt: Spanish|--spec-draft-n-max 2|64|256|¿Cuál es la capital de Francia?"
    "Prompt: Japanese|--spec-draft-n-max 2|64|256|フランスの首都は?"
    "Prompt: Chinese|--spec-draft-n-max 2|64|256|法国的首都是?"
    "Prompt: emoji|--spec-draft-n-max 2|64|256|🇫🇷 capital?"
    "Prompt: math context|--spec-draft-n-max 2|64|256|If x = France, then capital(x) ="
    
    # Category 4: Temperature and sampling (experiments 46-55)
    "Temp 0.0|--spec-draft-n-max 2 --temp 0.0|64|256|What is the capital of France?"
    "Temp 0.1|--spec-draft-n-max 2 --temp 0.1|64|256|What is the capital of France?"
    "Temp 0.3|--spec-draft-n-max 2 --temp 0.3|64|256|What is the capital of France?"
    "Temp 0.5|--spec-draft-n-max 2 --temp 0.5|64|256|What is the capital of France?"
    "Temp 0.7|--spec-draft-n-max 2 --temp 0.7|64|256|What is the capital of France?"
    "Temp 0.0, 128 tokens|--spec-draft-n-max 2 --temp 0.0|128|512|What is the capital of France?"
    "Temp 0.1, 128 tokens|--spec-draft-n-max 2 --temp 0.1|128|512|What is the capital of France?"
    "Temp 0.3, 128 tokens|--spec-draft-n-max 2 --temp 0.3|128|512|What is the capital of France?"
    "Top-p 0.9|--spec-draft-n-max 2 --temp 0 --top-p 0.9|64|256|What is the capital of France?"
    "Top-p 0.5|--spec-draft-n-max 2 --temp 0 --top-p 0.5|64|256|What is the capital of France?"
    
    # Category 5: System prompt variations (experiments 56-65)
    "Sys: helpful|--spec-draft-n-max 2 --system-prompt You are a helpful assistant.|64|256|What is the capital of France?"
    "Sys: concise|--spec-draft-n-max 2 --system-prompt Be concise.|64|256|What is the capital of France?"
    "Sys: expert|--spec-draft-n-max 2 --system-prompt You are a geography expert.|64|256|What is the capital of France?"
    "Sys: teacher|--spec-draft-n-max 2 --system-prompt You are a teacher explaining to a student.|64|256|What is the capital of France?"
    "Sys: quiz|--spec-draft-n-max 2 --system-prompt Answer quiz questions with one word.|64|256|What is the capital of France?"
    "Sys: JSON|--spec-draft-n-max 2 --system-prompt Respond in JSON format.|64|256|What is the capital of France?"
    "Sys: code|--spec-draft-n-max 2 --system-prompt You are a code assistant.|64|256|What is the capital of France?"
    "Sys: translator|--spec-draft-n-max 2 --system-prompt You are a translator.|64|256|What is the capital of France?"
    "Sys: no system|--spec-draft-n-max 2|64|256|What is the capital of France?"
    "Sys: anti-thinking|--spec-draft-n-max 2 --system-prompt Do not think. Answer directly.|64|256|What is the capital of France?"
    
    # Category 6: Model variants (experiments 66-75)
    "Model: Q4_K_M|--spec-draft-n-max 2|64|256|What is the capital of France?"
    "Model: Q4_0|--spec-draft-n-max 2|64|256|What is the capital of France?"
    "Model: Q5_K_M|--spec-draft-n-max 2|64|256|What is the capital of France?"
    "Model: Q8_0|--spec-draft-n-max 2|64|256|What is the capital of France?"
    "Model: F16|--spec-draft-n-max 2|64|256|What is the capital of France?"
    "Model: Q4_K_M, 128 tokens|--spec-draft-n-max 2|128|512|What is the capital of France?"
    "Model: Q4_0, 128 tokens|--spec-draft-n-max 2|128|512|What is the capital of France?"
    "Model: Q5_K_M, 128 tokens|--spec-draft-n-max 2|128|512|What is the capital of France?"
    "Model: Q8_0, 128 tokens|--spec-draft-n-max 2|128|512|What is the capital of France?"
    "Model: F16, 128 tokens|--spec-draft-n-max 2|128|512|What is the capital of France?"
    
    # Category 7: Hardware utilization (experiments 76-85)
    "Threads 1|--spec-draft-n-max 2 --threads 1|64|256|What is the capital of France?"
    "Threads 2|--spec-draft-n-max 2 --threads 2|64|256|What is the capital of France?"
    "Threads 4|--spec-draft-n-max 2 --threads 4|64|256|What is the capital of France?"
    "Threads 8|--spec-draft-n-max 2 --threads 8|64|256|What is the capital of France?"
    "Threads 16|--spec-draft-n-max 2 --threads 16|64|256|What is the capital of France?"
    "Threads 1, 128 tokens|--spec-draft-n-max 2 --threads 1|128|512|What is the capital of France?"
    "Threads 2, 128 tokens|--spec-draft-n-max 2 --threads 2|128|512|What is the capital of France?"
    "Threads 4, 128 tokens|--spec-draft-n-max 2 --threads 4|128|512|What is the capital of France?"
    "Threads 8, 128 tokens|--spec-draft-n-max 2 --threads 8|128|512|What is the capital of France?"
    "Threads 16, 128 tokens|--spec-draft-n-max 2 --threads 16|128|512|What is the capital of France?"
    
    # Category 8: Batch size variations (experiments 86-95)
    "Batch 1|--spec-draft-n-max 2 --batch-size 1|64|256|What is the capital of France?"
    "Batch 2|--spec-draft-n-max 2 --batch-size 2|64|256|What is the capital of France?"
    "Batch 4|--spec-draft-n-max 2 --batch-size 4|64|256|What is the capital of France?"
    "Batch 8|--spec-draft-n-max 2 --batch-size 8|64|256|What is the capital of France?"
    "Batch 16|--spec-draft-n-max 2 --batch-size 16|64|256|What is the capital of France?"
    "Batch 1, 128 tokens|--spec-draft-n-max 2 --batch-size 1|128|512|What is the capital of France?"
    "Batch 2, 128 tokens|--spec-draft-n-max 2 --batch-size 2|128|512|What is the capital of France?"
    "Batch 4, 128 tokens|--spec-draft-n-max 2 --batch-size 4|128|512|What is the capital of France?"
    "Batch 8, 128 tokens|--spec-draft-n-max 2 --batch-size 8|128|512|What is the capital of France?"
    "Batch 16, 128 tokens|--spec-draft-n-max 2 --batch-size 16|128|512|What is the capital of France?"
    
    # Category 9: Advanced optimizations (experiments 96-100)
    "Opt: all optimizations|--spec-draft-n-max 2 --use-resize --flash-attn --no-mmap|64|256|What is the capital of France?"
    "Opt: no mmap|--spec-draft-n-max 2 --no-mmap|64|256|What is the capital of France?"
    "Opt: flash attention|--spec-draft-n-max 2 --flash-attn|64|256|What is the capital of France?"
    "Opt: resize bar|--spec-draft-n-max 2 --use-resize|64|256|What is the capital of France?"
    "Opt: combined|--spec-draft-n-max 2 --use-resize --flash-attn --no-mmap|128|512|What is the capital of France?"
)

# ============================================================
# MAIN EXECUTION
# ============================================================

cd "$PROJECT_DIR"

echo "========================================"
echo "MTP+Stream Optimization Experiments"
echo "========================================"
echo "Started at: $(date)"
echo "Total experiments: ${#EXPERIMENTS[@]}"
echo ""

# Build first
echo "Building..."
export PATH="/tmp/venv/bin:$PATH"
cmake --build build_opencl --target llama-cli -j$(nproc) 2>&1 | tail -3

# Run baseline
run_baseline

# Initialize results file
cat > "$RESULTS_FILE" << EOF
# MTP+Stream Optimization Experiments

## Baseline
- Prompt: ${baseline_prompt_tps} t/s
- Generation: ${baseline_gen_tps} t/s

## Experiment Log

| # | Description | Prompt t/s | Gen t/s | Δ Gen | Coherent | Notes |
|---|-------------|------------|---------|-------|----------|-------|
EOF

# Run all experiments
best_gen_tps=$baseline_gen_tps
best_exp_num=0
best_exp_desc=""

for i in "${!EXPERIMENTS[@]}"; do
    IFS='|' read -r desc extra_args n_tokens context prompt <<< "${EXPERIMENTS[$i]}"
    exp_num=$((i + 1))
    
    gen_tps=$(run_experiment $exp_num "$desc" "$extra_args" $n_tokens $context "$prompt")
    
    # Track best
    if (( $(echo "$gen_tps > $best_gen_tps" | bc -l) )); then
        best_gen_tps=$gen_tps
        best_exp_num=$exp_num
        best_exp_desc="$desc"
        echo "  *** New best: $best_gen_tps t/s ***"
    fi
done

# Save best config
cat > "$BEST_CONFIG_FILE" << EOF
# Best Configuration

## Best Result
- Experiment: $best_exp_num
- Description: $best_exp_desc
- Generation: $best_gen_tps t/s
- Improvement: $(echo "scale=2; $best_gen_tps - $baseline_gen_tps" | bc) t/s

## Baseline
- Generation: $baseline_gen_tps t/s
EOF

echo ""
echo "========================================"
echo "Experiments Complete!"
echo "========================================"
echo "Best: Experiment $best_exp_num - $best_exp_desc"
echo "Best Gen t/s: $best_gen_tps"
echo "Improvement: $(echo "scale=2; $best_gen_tps - $baseline_gen_tps" | bc) t/s"
echo "Results saved to: $RESULTS_FILE"
echo "Best config saved to: $BEST_CONFIG_FILE"
echo "Completed at: $(date)"
