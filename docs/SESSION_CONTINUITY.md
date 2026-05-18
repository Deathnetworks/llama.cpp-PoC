# Decoupled FFN — Session Continuity Document

## Current Branch: `decoupled-split`
## Last Commit: `31ffebde2` — "feat: add --stream flag and per-layer mmap tracking for zero-RAM mode"

---

## What Works Right Now

### Modes
| Mode | Flag | VRAM | RAM | Speed |
|------|------|------|-----|-------|
| GPU-ONLY | `--split-mode gpu-only` | All | All | 100% |
| FNN-RAM-CPU | `--split-mode fnn-ram-cpu` | ~40% | ~100% | ~40% |
| FNN-ZERO-CPU | `--split-mode fnn-zero-cpu` | ~40% | ~15% | ~10% |
| + MTP | `--spec-type draft-mtp` | +500MB | — | +30% |
| + ReBAR | `--use-resize` | — | — | +?% |
| + Stream | `--stream` | ~40% | ~5% | TBD |

### Key Files
- `src/llama-ffn-local.h` — Layer struct, routing predicates, async buffer API
- `src/llama-ffn-local.cpp` — Loader, CPU FFN compute, callback, per-layer eviction
- `src/llama-ffn-async.cpp` — Async double-buffer engine (synchronous prototype)
- `src/llama-model-loader.cpp` — Tensor routing (CPU vs GPU buffer)
- `src/llama-graph.cpp` — Graph construction
- `src/llama-context.cpp/h` — Context params, mode initialization
- `src/llama.cpp` — Model load, global flags (g_use_resize, g_zero_ram)
- `common/arg.cpp` — CLI argument parsing
- `common/common.h` — Parameter structs (use_resize, zero_ram)
- `include/llama.h` — Public API
- `docs/USER_GUIDE.md` — User documentation
- `docs/IMPLEMENTATION_PLAN.md` — Architecture and plan

### Build Commands
```bash
# OpenCL (Intel Arc)
cmake -B build_opencl -DCMAKE_BUILD_TYPE=Release \
    -DGGML_OPENCL=ON -DGGML_OPENCL_EMBED_KERNELS=ON \
    -DGGML_OPENCL_USE_ADRENO_KERNELS=OFF -DGGML_BLAS=ON
cmake --build build_opencl --target llama-cli -j$(nproc)

# CPU-only (for testing)
cmake -B build_cpu -DCMAKE_BUILD_TYPE=Release -DGGML_BLAS=ON
cmake --build build_cpu --target llama-cli -j$(nproc)
```

### Test Commands
```bash
# Basic FFN-ZERO-CPU
./build_opencl/bin/llama-cli -m model.gguf \
    --split-mode fnn-zero-cpu -p "test" -n 4 --temp 0 \
    -c 64 --single-turn --jinja

# With MTP
./build_opencl/bin/llama-cli -m model.MTP.gguf \
    --split-mode fnn-zero-cpu --spec-type draft-mtp \
    --spec-draft-n-max 6 -p "test" -n 4 --temp 0 \
    -c 64 --single-turn --jinja

# With --stream (zero-RAM streaming)
./build_opencl/bin/llama-cli -m model.gguf \
    --split-mode fnn-zero-cpu --stream -p "test" -n 4 --temp 0 \
    -c 64 --single-turn --jinja
```

---

## The Problem We're Solving

**Goal**: Run a 1T parameter model on a system with 16GB RAM + 32GB VRAM.

**Current state**: FFN-ZERO-CPU uses mmap, but during inference the OS page cache
holds all accessed weights. With 33+ layers, by the time layer 33 runs,
layers 1-32 are in the cache. Result: all model weights end up in RAM.

**User's observation**: During generation, ~13.8GB RAM stays occupied even after
manual cache clearing between prompts.

---

## Why madvise Doesn't Work

1. `MADV_DONTNEED` after loading → pages get faulted back during inference
2. `MADV_DONTNEED` after each layer → causes hard page faults on next access
3. `MADV_PAGEOUT` → only for dirty pages; model weights are read-only
4. The kernel makes bad eviction decisions for deterministic linear access

**Conclusion**: We need to bypass the page cache entirely.

---

## The Solution: Async Double-Buffered Streaming

### Architecture
```
Time Step N:
  [GPU computes Attention(layer N)]
  [CPU reads FFN weights for layer N+1 into Buffer B via async DMA]
  [Swap: Buffer A is now active for layer N+1]

Time Step N+1:
  [GPU computes Attention(layer N+1)]
  [CPU reads FFN weights for layer N+2 into Buffer A via async DMA]
  [CPU computes FFN(layer N+1) using Buffer B]
  [Swap: Buffer B is now active for layer N+2]
```

### Key Components

1. **File offset tracking**: Store GGUF file offsets for each layer's FFN weights
2. **Double buffer**: Two aligned buffers (2× max_layer_size)
3. **Async I/O**: `io_uring` on Linux, Overlapped I/O on Windows
4. **On-the-fly dequantization**: Dequantize Q4→F32 during matmul (no separate buffer)

### RAM Budget
| Model | GPU-ONLY | mmap+madvise | --stream (sync) | --stream (async) |
|-------|----------|--------------|-----------------|------------------|
| 4B | 3.5 GB | 0.8 GB | 0.2 GB | 0.2 GB |
| 35B | 22 GB | 2.0 GB | 0.5 GB | 0.5 GB |
| 1T | N/A | N/A | ~2 GB | ~2 GB |

---

## Implementation Status

### Phase 1: Synchronous Streaming (IN PROGRESS)
- [x] Store file offsets for each layer's FFN weights
- [x] Allocate double buffer for weight streaming
- [x] Implement synchronous read() per layer
- [x] Add --stream CLI flag
- [x] Per-layer mmap range tracking
- [ ] **INTEGRATE WITH GRAPH SCHEDULER** ← current blocker
- [ ] Test and measure RAM usage

### Phase 2: Async I/O with io_uring (PENDING)
- [ ] Implement io_uring-based async reads
- [ ] Double-buffer ping-pong: load N+1 while computing N
- [ ] Overlap disk I/O with GPU attention compute

### Phase 3: Windows Native (PENDING)
- [ ] Implement Overlapped I/O with IOCP
- [ ] Same double-buffer architecture

### Phase 4: On-the-fly Dequantization (PENDING)
- [ ] Custom CPU kernels that dequantize Q4→F32 during matmul
- [ ] Eliminate separate F32 buffer

---

## Current Blocker: Graph Scheduler Integration

The FFN weights are accessed by the GGML graph scheduler during cross-backend
execution. The scheduler automatically inserts GPU→CPU copies for the residual
and CPU→GPU copies for the result.

**The problem**: We need to load FFN weights from SSD BEFORE the scheduler
accesses them, but the scheduler doesn't have a hook for "before this layer's
FFN weights are needed."

### Possible Solutions

1. **Use `cb_eval` callback**: The scheduler calls `cb_eval` for every tensor.
   We can detect when an FFN weight tensor is about to be used and load it.
   Risk: callback is called for ALL tensors, need to filter efficiently.

2. **Custom GGML op**: Create a custom `GGML_OP_FFN_STREAM` op that loads
   weights from SSD, then performs the FFN computation. This gives us full
   control over when weights are loaded.

3. **Pre-load before graph compute**: Before calling
   `ggml_backend_sched_graph_compute_async`, load all FFN weights for the
   current layer. This is the simplest approach but requires knowing which
   layer is about to execute.

4. **Modify the model files**: Each model file (qwen3.cpp, llama.cpp, etc.)
   calls `build_ffn()`. We can modify these to insert a "load weights" node
   before the FFN computation.

**Recommended approach**: Option 2 (custom GGML op). This gives us:
- Full control over weight loading timing
- Ability to overlap I/O with GPU compute
- Clean separation from the graph scheduler
- Works with any model architecture

---

## Next Session Priority

1. **Implement custom GGML op for streaming FFN**
   - Create `GGML_OP_FFN_STREAM` op
   - Op loads weights from SSD into buffer
   - Op performs FFN computation using loaded weights
   - Register op in GGML backend

2. **Integrate with model graph builders**
   - Modify `build_ffn()` in model files to use streaming op
   - Pass file offsets and buffer pointers through op params

3. **Test synchronous streaming**
   - Measure RAM usage with `--stream` flag
   - Verify correctness of output
   - Benchmark performance impact

4. **Implement io_uring async reads**
   - Replace synchronous read() with io_uring
   - Double-buffer ping-pong
   - Overlap with GPU attention compute

---

## Key Code Locations

### Where FFN weights are loaded
- `src/llama-ffn-local.cpp:242` — `ffn_mmap_from_full_gguf()` — loads weights from GGUF
- `src/llama-model-loader.cpp` — routes tensors to CPU/GPU buffers

### Where FFN graph is built
- `src/llama-graph.cpp:1156` — `build_ffn()` — builds FFN compute graph
- `src/models/qwen3.cpp` — calls `build_ffn()` for Qwen3 models
- `src/models/llama.cpp` — calls `build_ffn()` for Llama models

### Where graph is executed
- `src/llama-context.cpp:2361` — `ggml_backend_sched_graph_compute_async()`

### Where cross-backend copies happen
- GGML backend scheduler (in ggml/src/)
- Automatically inserts copies when tensors are on different backends

### Global flags
- `src/llama.cpp:25` — `g_use_resize`
- `src/llama.cpp:29` — `g_zero_ram`

---

## Test Results (Previous Sessions)

### Qwen3.5-4B-MTP (2.8 GB)
| Mode | Prompt t/s | Gen t/s | RAM |
|------|-----------|---------|-----|
| GPU-ONLY | 62.7 | 37.8 | ~3.5 GB |
| FNN-RAM-CPU | 28.7 | 12.3 | ~4.0 GB |
| FNN-ZERO-CPU | 28.3 | 12.1 | ~0.8 GB |
| FNN-ZERO-CPU+MTP | 35.3 | 11.6 | ~1.3 GB |

### Qwen3.6-35B-MTP (22.6 GB)
| Mode | Prompt t/s | Gen t/s | RAM |
|------|-----------|---------|-----|
| GPU-ONLY | 29.4 | 11.4 | ~22 GB |
| FNN-ZERO-CPU+MTP | 40.4 | 8.6 | ~2.0 GB |

### Memory Eviction Test
- After load: Cached drops from 23GB → 3GB (Linux)
- After `drop_caches`: 3GB → 253MB
- During inference: 13.8GB stays occupied (Windows/WSL2)
- **This is the problem we need to solve**

---

## Environment Variables
```bash
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
export LLAMA_ARG_USE_RESIZE=1
export LLAMA_ARG_STREAM=1
```

---

## Windows Binaries
Located in `/home/deathnetworks/Decoupled Attn - PoC/windows/`
- `llama-cli.exe` — CLI tool
- `llama-server.exe` — Server tool

Copy from build: `cp build_windows_cpu/bin/llama-cli.exe windows/`
