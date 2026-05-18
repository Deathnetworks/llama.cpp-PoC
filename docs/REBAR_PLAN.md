# ReBAR Integration Plan

## Goal
Optimize residual stream transfers between CPU (FFN) and GPU (Attention) using Resizable BAR (ReBAR).

## Architecture
```
Current (without ReBAR):
  GPU→CPU: ggml_backend_tensor_get() → GPU DMA to host buffer → CPU reads
  CPU→GPU: ggml_backend_tensor_set() → CPU memcpy to GPU buffer → GPU reads

With ReBAR:
  GPU→CPU: GPU DMA to pinned host buffer → CPU reads (same as before, but host buffer is USM)
  CPU→GPU: CPU memcpy directly to GPU VRAM (ReBAR-mapped device pointer) — FAST
```

## Implementation Steps

### Step 1: ReBAR Detection
- Add `ggml_sycl_check_rebar_status()` function
- Check if `max_mem_alloc_size > 256MB` (standard BAR limit)
- Store result in a global/backend flag
- Add `--use-resize` CLI flag (already exists, currently used for callback path)

### Step 2: Buffer Allocation Strategy
- **GPU Receive Buffer (Attention input)**: Allocate with `sycl::malloc_device` (VRAM, ReBAR-mapped)
- **CPU Receive Buffer (FFN input)**: Allocate with `sycl::malloc_host` (pinned host RAM)
- The residual stream tensors need to be allocated with the right memory type

### Step 3: Transfer Functions
- **CPU→GPU (ReBAR path)**: Use `memcpy()` directly to device pointer instead of `ggml_backend_tensor_set()`
- **GPU→CPU (DMA path)**: Use SYCL queue `memcpy()` to push from device to host (same as before)

### Step 4: Integration Points
- Modify `ggml_backend_sycl_cpy_tensor()` or the buffer copy function
- When ReBAR is enabled AND tensor is a residual stream (small, < 1MB):
  - CPU→GPU: memcpy to device pointer
  - GPU→CPU: SYCL queue memcpy to host pointer

### Step 5: Synchronization
- CPU→GPU: Need memory fence after memcpy to ensure GPU sees the data
- GPU→CPU: Need to wait for DMA completion before CPU reads

## Key Files to Modify
1. `ggml/src/ggml-sycl/ggml-sycl.cpp` — ReBAR detection, buffer allocation, transfer functions
2. `ggml/src/ggml-sycl/ggml-sycl.h` — ReBAR status flag
3. `src/llama-context.cpp` — Pass ReBAR flag to graph compute
4. `common/arg.cpp` — `--use-resize` flag wiring

## Tests
1. **Detection test**: Verify ReBAR status is correctly detected
2. **Correctness test**: Compare output with and without ReBAR (should be identical)
3. **Performance test**: Measure token/s improvement with ReBAR
4. **Memory test**: Verify no additional RAM usage with ReBAR
