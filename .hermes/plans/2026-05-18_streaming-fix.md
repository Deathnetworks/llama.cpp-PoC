# Plan: Fix --stream Pipeline for Correct English Output + MTP Performance

## Goal
Fix the `--stream` pipeline so that:
1. Model responds in English when asked in English (not Chinese)
2. MTP is faster than no-MTP (currently 8.0 t/s vs 10.2 t/s — wrong direction)
3. `--stream` actually streams weights from SSD, not silently falling back to mmap
4. No `--reasoning off` hack needed

## Root Causes Identified

### 1. Weight Map Never Populated (CRITICAL)
The `ffn_weight_map` is never populated in `llama.cpp`. The population code existed in commit `b2a4755f7` but was lost during git checkups. Without the weight map:
- `model->ffn_weight_map` is NULL
- Context init skips streaming setup: `if (params.zero_ram && model->ffn_weight_map && ...)` fails
- Streaming engine never initializes
- All FFN weights use mmap fallback → Chinese output (corruption from mmap page cache)

**Fix**: Add weight map population code back into `llama_model_load()` in `llama.cpp`, right after `load_tensors()` completes.

### 2. Weight Map Population Code (from commit b2a4755f7)
```cpp
// Populate FFN weight offset map for streaming mode
if (params.zero_ram || params.ffn_split_mode >= 4) {
    auto * weight_map = new std::unordered_map<std::string, ffn_weight_offset>();
    for (const auto & it : ml.weights_map) {
        const std::string & name = it.first;
        if (is_ffn_tensor(name.c_str())) {
            // Skip MTP predictor layer FFN tensors — they stay on GPU
            if (params.ffn_split_mode >= 4 && ml.has_mtp && 
                is_ffn_in_mtp_layer(name.c_str(), model->hparams.n_layer)) {
                continue;
            }
            ffn_weight_offset wo;
            wo.file_idx = it.second.idx;
            wo.file_off = it.second.offs;
            wo.size = ggml_nbytes(it.second.tensor);
            weight_map->emplace(name, wo);
        }
    }
    model->ffn_weight_map = weight_map;
    model->ffn_n_layers = model->hparams.n_layer;
    model->ffn_model_path = fname;
}
```

### 3. MTP Performance Issue
MTP should be faster because speculative decoding accepts multiple tokens per forward pass. Currently slower because:
- Streaming engine not actually working (falling back to mmap)
- MTP predictor FFN tensors may be incorrectly routed to CPU instead of GPU
- The `is_ffn_in_mtp_layer()` check may be wrong — MTP predictor is in the LAST layer (`blk.{n_layer-1}`), not a separate "nextn" layer

**Fix**: Ensure MTP predictor tensors stay on GPU. The `has_mtp` pre-scan in the model loader must correctly identify MTP models.

### 4. Remaining fprintf(stderr) Usage
Several `fprintf(stderr, ...)` calls remain that should be `LLAMA_LOG_DEBUG`. These clutter output and can cause the "garbled" appearance.

## Step-by-Step Implementation Plan

### Step 1: Add Weight Map Population to llama.cpp
- File: `src/llama.cpp`
- Location: After `load_tensors()` completes, before `malloc_trim`
- Add the weight map population code from commit b2a4755f7
- Use `LLAMA_LOG_INFO` for the debug message

### Step 2: Fix MTP Tensor Routing
- File: `src/llama-model-loader.cpp`
- Verify `is_ffn_in_mtp_layer()` correctly identifies last layer FFN tensors
- Ensure `has_mtp` is set correctly during model loading
- MTP predictor tensors must stay on GPU (route_to_cpu = false)

### Step 3: Fix Remaining fprintf(stderr) in context.cpp
- Replace all `fprintf(stderr, "DEBUG STREAM: ...")` with `LLAMA_LOG_DEBUG`
- Replace `fprintf(stderr, "streaming: ...")` with `LLAMA_LOG_DEBUG`

### Step 4: Build and Test
- Build: `cmake --build build_opencl --target llama-cli -j$(nproc)`
- Test 1: `--stream --jinja --reasoning off` → should produce English output
- Test 2: `--stream --jinja --reasoning off --spec-type draft-mtp --spec-draft-n-max 2` → MTP should be faster than no-MTP
- Test 3: `--stream --jinja` (reasoning on) → should produce English thinking + answer
- Verify: Check that weight map is populated (should see "populated FFN weight map with N entries" message)

### Step 5: Validate RAM Usage
- Measure RSS with `--stream` vs without
- Should show ~88MB for streaming buffers vs ~3.7GB for mmap page cache

## Files to Modify
1. `src/llama.cpp` — Add weight map population (line ~450, after load_tensors)
2. `src/llama-context.cpp` — Fix fprintf → LLAMA_LOG_DEBUG
3. `src/llama-model-loader.cpp` — Verify MTP routing (may not need changes)

## Risks
- The `ml.weights_map` may not contain the expected tensor names — need to verify format
- `ffn_split_mode` enum may not match `params.ffn_split_mode` type — need to check
- MTP detection (`ml.has_mtp`) may not be set correctly — need to trace

## Verification
1. Weight map populated: Look for "populated FFN weight map with N entries" in output
2. English output: Model responds in English to English questions
3. MTP faster: Generation t/s with MTP > generation t/s without MTP
4. RAM reduction: RSS with --stream significantly lower than without
