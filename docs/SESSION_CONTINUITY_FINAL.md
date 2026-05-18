# Session Continuity — May 17 2026 (Final)

## Branch: `decoupled-split`
## Last Commit: `a9c4087df` — "feat: build succeeds with streaming infrastructure"

## Goal
Run 1T+ parameter models on consumer hardware (16GB RAM + 32GB VRAM) by streaming
FFN weights from SSD per-layer, keeping only attention weights in VRAM.

## What Was Accomplished This Session

### 1. Analyzed Why madvise Doesn't Work
- `MADV_DONTNEED` after loading doesn't help — pages fault back during inference
- `MADV_PAGEOUT` only works for dirty pages (weights are read-only)
- The OS page cache is designed to keep accessed pages in RAM
- For deterministic linear access (LLM layers), ALL weights end up in RAM
- User confirmed: 13.8GB RAM stays occupied during generation

### 2. Designed Async Double-Buffer Streaming Architecture
- Read FFN weights from SSD into temp buffer before each layer
- Only ~1 layer of weights in RAM at any time
- Double-buffer with async I/O (io_uring on Linux, Overlapped on Windows)
- Overlap disk I/O with GPU attention compute
- On-the-fly Q4→F32 dequantization during matmul (future optimization)

### 3. Implemented Infrastructure
- `--stream` CLI flag for zero-RAM streaming mode
- `ffn_async_buffer` struct with double-buffering support
- `ffn_async_init()`, `ffn_async_load_layer()`, `ffn_async_free()` functions
- Per-layer file offset tracking in `ffn_layer_ptrs_t`
- Streaming callback in `graph_compute()` using `cb_eval`
- Updated CMakeLists.txt to include new source files
- Build succeeds, basic inference works

### 4. Created Documentation
- `docs/IMPLEMENTATION_PLAN.md` — Full architecture and phased plan
- `docs/SESSION_CONTINUITY.md` — Previous continuity document
- Updated `docs/USER_GUIDE.md` with `--stream` examples

## Current Status

### What Works
- Build succeeds for OpenCL backend
- Basic inference works with `--split-mode fnn-zero-cpu`
- `--stream` flag is recognized and sets `zero_ram = true`
- Per-layer file offsets are tracked in the loader

### What's Missing (Next Steps)
The streaming callback is NOT yet functional because:

1. **ffn_mmap_t not stored in context**: The `ffn_mmap_t` created during model load
   is not accessible from `graph_compute()`. Need to store it in the context or model.

2. **ffn_async_buffer not initialized**: The async buffer is never created/initialized.
   Need to call `ffn_async_init()` during context creation.

3. **Streaming callback doesn't actually load weights**: The callback checks for
   `ffn_mmap` and `ffn_async` but they're nullptr.

4. **Tensor data pointers not updated**: Even if weights are loaded, the tensor
   data pointers in the graph still point to the mmap region, not the loaded buffer.

### The Core Problem
The GGML graph scheduler executes pre-built nodes. The weight tensors have their
data pointers set during graph construction (pointing to mmap region). To stream
weights, we need to either:
- (a) Update tensor data pointers during execution (requires scheduler hooks)
- (b) Build the graph with "load weight" nodes that read from SSD
- (c) Use a custom GGML op that loads weights as part of computation

Option (c) is the cleanest approach but requires writing a custom GGML op.

## Recommended Next Steps

### Immediate (Next Session)
1. Store `ffn_mmap_t` pointer in `llama_model` or `llama_context`
2. Initialize `ffn_async_buffer` in `llama_init_from_model()` when `zero_ram` is set
3. Make the streaming callback functional by passing the ffn_mmap and async_buf
4. Test with actual weight loading and measure RAM usage

### Short-Term
5. Implement proper weight loading in the streaming callback
6. Update tensor data pointers to point to loaded buffer
7. Test RAM usage with `--stream` flag
8. Measure performance impact

### Medium-Term
9. Implement io_uring async reads for overlap with GPU compute
10. Add on-the-fly dequantization
11. Windows Overlapped I/O port

## Key Code Locations

### Streaming Engine
- `src/llama-ffn-async.cpp` — Async buffer functions (init, load, free)
- `src/llama-ffn-local.h:195` — Async buffer API declarations
- `src/llama-context.cpp:2361` — Streaming callback in graph_compute()

### Model Loading
- `src/llama-ffn-local.cpp:274` — `ffn_mmap_from_full_gguf()` — loads weights, stores offsets
- `src/llama-context.h:370` — Context streaming state (ffn_mmap, ffn_async, ffn_n_layers)

### Graph Execution
- `src/llama-context.cpp:2342` — `graph_compute()` — where streaming callback is set
- `ggml/src/ggml-backend.cpp:1682` — Scheduler calls cb_eval for each node

### CLI Flags
- `common/arg.cpp:2432` — `--use-resize` flag
- `common/arg.cpp:2439` — `--zero-ram` flag  
- `common/arg.cpp:2446` — `--stream` flag

## Test Commands
```bash
# Basic FFN-ZERO-CPU (working)
./build_opencl/bin/llama-cli -m model.gguf \
    --split-mode fnn-zero-cpu -p "test" -n 8 --temp 0 \
    -c 64 --single-turn --jinja

# With --stream (infrastructure in place, not yet functional)
./build_opencl/bin/llama-cli -m model.gguf \
    --split-mode fnn-zero-cpu --stream -p "test" -n 8 --temp 0 \
    -c 64 --single-turn --jinja

# Monitor RAM usage
watch -n 1 'free -h && ps aux | grep llama'
```

## RAM Usage Expectations
| Model | GPU-ONLY | mmap+madvise | --stream (target) |
|-------|----------|--------------|-------------------|
| 4B | 3.5 GB | 0.8 GB | 0.2 GB |
| 35B | 22 GB | 2.0 GB | 0.5 GB |
| 1T | N/A | N/A | ~2 GB |

## Files Modified This Session
- `common/arg.cpp` — Fixed lambda signatures
- `src/CMakeLists.txt` — Added new source files
- `src/llama-context.cpp` — Added streaming callback infrastructure
- `src/llama-context.h` — Added streaming state fields
- `src/llama-ffn-local.cpp` — Updated loader for async offsets, simplified eviction
- `src/llama-ffn-local.h` — Added async_load_layer declaration
- `docs/IMPLEMENTATION_PLAN.md` — New file
- `docs/USER_GUIDE.md` — Updated with --stream examples

## Build Command
```bash
cd /home/deathnetworks/Decoupled\ Attn\ -\ PoC/llama.cpp-PoC
export PATH="/tmp/venv/bin:$PATH"
cmake --build build_opencl --target llama-cli -j$(nproc)
```
