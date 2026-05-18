# Decoupled Attention/FFN Inference — User Documentation

## Overview

This fork of llama.cpp implements **FFN-to-CPU offload**, which reduces GPU VRAM usage by keeping FFN (Feed-Forward Network) weights in system RAM while running attention layers on GPU. This enables running larger models on GPUs with limited VRAM.

### Two Modes

| Mode | Split Mode Flag | Description |
|------|----------------|-------------|
| **GPU-ONLY** | `--split-mode gpu-only` (default) | All weights on GPU. Maximum performance. |
| **FNN-RAM-CPU** | `--split-mode fnn-ram-cpu` | FFN weights on CPU RAM, attention on GPU. Reduces VRAM. |
| **FNN-RAM-CPU-OTHER** | `--split-mode fnn-ram-cpu-other` | FFN + SSM/other on CPU RAM, embedding on GPU. |
| **FNN-RAM-CPU-ALL** | `--split-mode fnn-ram-cpu-all` | All non-attention weights on CPU RAM. Max VRAM savings. |
| **FNN-ZERO-CPU** | `--split-mode fnn-zero-cpu` | FFN weights mmap'd from SSD. Minimal RAM usage. |
| **FNN-ZERO-CPU-OTHER** | `--split-mode fnn-zero-cpu-other` | FFN + SSM/other mmap'd from SSD. |
| **FNN-ZERO-CPU-ALL** | `--split-mode fnn-zero-cpu-all` | All non-attention weights mmap'd from SSD. |

### How It Works

```
┌─────────────────────────────────────────────────────────────┐
│                    INFERENCE LOOP                            │
│                                                              │
│  embed(tokens) → residual                    [GPU VRAM]     │
│  for L in 0..N_LAYERS:                                       │
│    residual += attn(residual, L)             [GPU VRAM]     │
│    ┌─────────────────────────────────────────┐               │
│    │  RESIDUAL SHUTTLE (per layer)           │               │
│    │  1. clEnqueueReadBuffer(GPU→CPU)        │               │
│    │  2. CPU: rms_norm + gate/up + down      │               │
│    │  3. clEnqueueWriteBuffer(CPU→GPU)        │               │
│    └─────────────────────────────────────────┘               │
│  logits = lm_head(residual)                  [GPU VRAM]     │
│  sample → next token                                        │
└─────────────────────────────────────────────────────────────┘

VRAM Budget (26B model @ Q4):
  Attention weights + KV cache: ~1.5 GB  ← GPU VRAM
  FFN weights: ~14 GB                     ← CPU RAM (not VRAM!)
  Total GPU VRAM: ~2.2 GB (was 16 GB)
```

## Quick Start

### Build

```bash
# Clone
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp

# Create build directory
mkdir build && cd build

# Configure (OpenCL for Intel Arc)
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_OPENCL=ON \
    -DGGML_OPENCL_EMBED_KERNELS=ON \
    -DGGML_OPENCL_USE_ADRENO_KERNELS=OFF \
    -DGGML_BLAS=ON

# Build
cmake --build . --target llama-cli llama-server -j$(nproc)
```

### Run

```bash
# GPU-ONLY mode (default, all on GPU)
./llama-cli -m model.gguf -p "Hello" -n 128

# FNN-RAM-CPU mode (FFN on CPU, attention on GPU)
./llama-cli -m model.gguf -p "Hello" -n 128 --split-mode fnn-ram-cpu

# With Q8 KV cache (saves more VRAM)
./llama-cli -m model.gguf -p "Hello" -n 128 --split-mode fnn-ram-cpu -ctk q8_0 -ctv q8_0

# Server mode
./llama-server -m model.gguf --split-mode fnn-ram-cpu --port 8080
```

## Build Configurations

### OpenCL (Intel Arc, AMD, NVIDIA)

```bash
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_OPENCL=ON \
    -DGGML_OPENCL_EMBED_KERNELS=ON \
    -DGGML_OPENCL_USE_ADRENO_KERNELS=OFF \
    -DGGML_BLAS=ON
```

**Requirements**: OpenCL drivers installed
- Intel Arc: `intel-opencl-icd` package
- AMD: `rocm-opencl-runtime`
- NVIDIA: Comes with CUDA toolkit

### CUDA (NVIDIA)

```bash
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=ON \
    -DGGML_BLAS=ON
```

**Requirements**: CUDA toolkit 11.8+

### Vulkan (Cross-platform GPU)

```bash
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_VULKAN=ON \
    -DGGML_BLAS=ON
```

**Requirements**: Vulkan SDK

### SYCL (Intel XPU)

```bash
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_SYCL=ON \
    -DGGML_BLAS=ON
```

**Requirements**: Intel oneAPI BaseKit

### CPU-only (no GPU)

```bash
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_BLAS=ON \
    -DGGML_OPENMP=ON
```

## Command-Line Reference

### Split Mode Flags

| Flag | Description |
|------|-------------|
| `--split-mode gpu-only` | All weights on GPU (default) |
| `--split-mode fnn-ram-cpu` | FFN weights on CPU, attention on GPU |
| `--split-mode fnn-zero-cpu` | FFN weights mmap'd from SSD (zero-copy) |
| `--split-mode none` | Single GPU, no splitting |
| `--split-mode layer` | Split layers across multiple GPUs |
| `--split-mode row` | Split rows across multiple GPUs |
| `--use-resize` | Use Resizable BAR for zero-copy transfers (Intel GPU) |

### FNN-RAM-CPU Specific Flags

| Flag | Description |
|------|-------------|
| `--split-mode fnn-ram-cpu` | Enable FFN-to-CPU offload |
| `--split-mode fnn-zero-cpu` | Enable FFN-to-SSD zero-copy offload |
| `LLAMA_SPLIT_OTHER=cpu` | Also offload SSM/other tensors to CPU |
| `--no-mmap` | Load weights directly into RAM (recommended for FNN-RAM-CPU) |
| `-ctk q8_0 -ctv q8_0` | Q8 KV cache (saves ~50% KV memory) |
| `-c N` | Context length (smaller = less KV cache) |
| `--reasoning off` | Disable thinking mode (faster) |

### Memory Reduction Flags

| Flag | VRAM Savings | Performance Impact |
|------|-------------|-------------------|
| `--split-mode fnn-ram-cpu` | 40-60% (FFN weights) | 60-90% slower |
| `--split-mode fnn-zero-cpu` | 50-70% (FFN weights) | 80-95% slower |
| `-ctk q8_0 -ctv q8_0` | ~50% KV cache | Minimal |
| `-c 4096` | ~75% KV vs 128K | None |
| `--no-mmap` | None | Faster loading |
| `--n-gpu-layers N` | Partial offload | Varies |

### Recommended Configurations

**For 2B-4B models (FFN on CPU is fast enough):**
```bash
llama-cli -m model.gguf \
    --split-mode fnn-ram-cpu \
    --no-mmap \
    -c 4096 \
    -n 256
```

**For 9B-27B models (keep FFN on GPU, reduce KV):**
```bash
llama-cli -m model.gguf \
    --split-mode gpu-only \
    -ctk q8_0 -ctv q8_0 \
    -c 4096 \
    -n 256
```

**For MoE models (FFN on CPU is FASTER):**
```bash
llama-cli -m model.gguf \
    --split-mode fnn-ram-cpu \
    --no-mmap \
    -c 4096 \
    -n 256
```

**For maximum VRAM reduction (all small tensors on CPU):**
```bash
LLAMA_SPLIT_OTHER=cpu llama-cli -m model.gguf \
    --split-mode fnn-ram-cpu \
    --no-mmap \
    -c 2048 \
    -ctk q8_0 -ctv q8_0 \
    -n 256
```

## Server Usage

```bash
# Start server with FNN-RAM-CPU
./llama-server \
    -m model.gguf \
    --split-mode fnn-ram-cpu \
    --no-mmap \
    -c 4096 \
    --host 0.0.0.0 \
    --port 8080

# Query
curl http://localhost:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":128}'
```

## Performance Tips

1. **Use SSD storage**: FNN-RAM-CPU requires fast storage for model weights. HDD causes severe slowdown for models >4GB.

2. **Use BLAS**: Install OpenBLAS or MKL for faster CPU FFN compute:
   ```bash
   sudo apt install libopenblas-dev  # Ubuntu
   ```

3. **Reduce context length**: KV cache dominates memory for long contexts. Use `-c 4096` instead of `-c 131072`.

4. **Use Q8 KV cache**: `-ctk q8_0 -ctv q8_0` saves ~50% KV memory with minimal quality loss.

5. **Disable thinking**: `--reasoning off` skips thinking tokens for faster generation.

6. **Use `--no-mmap`**: Loads weights directly into RAM, avoiding page faults during inference.

7. **For MoE models**: FNN-RAM-CPU is often FASTER than GPU-ONLY because it avoids VRAM spill.

## Troubleshooting

### "SET_ROWS" or "unsupported operation" error
The OpenCL backend doesn't support all operations. Use `--no-kv-offload` or switch to CPU backend for KV cache.

### Out of memory
- Reduce context length (`-c 4096`)
- Use Q8 KV cache (`-ctk q8_0 -ctv q8_0`)
- Use FNN-RAM-CPU mode (`--split-mode fnn-ram-cpu`)
- Reduce GPU layers (`--n-gpu-layers 20`)

### Slow inference on FNN-RAM-CPU
- Ensure model is on SSD, not HDD
- Install OpenBLAS: `sudo apt install libopenblas-dev`
- Use `--no-mmap` to load weights into RAM
- Reduce context length
- For models >9B, consider keeping FFN on GPU

### Output differs between GPU-ONLY and FNN-RAM-CPU
Expected behavior. FP16 (GPU) vs FP32 (CPU) precision differences accumulate through layers. Outputs are semantically equivalent but not byte-identical.

## GPU-SSD Mode (Future)

GPU-SSD mode will use zero-copy mmap from SSD to minimize system RAM usage. Instead of loading all FFN weights into RAM, weights are mmap'd from the SSD file and accessed via page faults.

**Design**:
- FFN weights are mmap'd from the GGUF file on SSD
- Only actively-used pages are loaded into RAM
- OS page cache manages which weights stay in RAM
- No explicit `cpy_tensor` needed — OS handles it

**Requirements**:
- SSD storage (mandatory — HDD too slow for random access)
- Sufficient RAM for page cache (at least 2× model size recommended)
- Linux kernel with `madvise` support

**Status**: Not yet implemented. Currently FNN-RAM-CPU loads all weights into RAM.

## Architecture Details

### Tensor Routing

In FNN-RAM-CPU mode, tensors are routed to different buffers:

| Tensor Type | Buffer | Reason |
|-------------|--------|--------|
| FFN weights (gate, up, down, norm) | CPU RAM | VRAM reduction |
| Attention weights (Q, K, V, O, norm) | GPU VRAM | Performance |
| SSM/other weights | GPU VRAM (default) or CPU (`LLAMA_SPLIT_OTHER=cpu`) | Configurable |
| Embedding | GPU VRAM | Small, frequently accessed |
| Output head | GPU VRAM | Small, used every token |
| KV cache | GPU VRAM | Must be fast for attention |

### Cross-Backend Transfers

The GGML graph scheduler automatically inserts copy operations when a tensor needs to move between backends:

1. After attention (GPU), the residual is copied to CPU
2. FFN computation runs on CPU
3. FFN output is copied back to GPU
4. Next layer's attention runs on GPU

Transfer size per layer: `n_embd × 2 bytes` (f16 residual)
- 2B: ~4 KB/layer × 24 layers = ~96 KB total
- 27B: ~10 KB/layer × 64 layers = ~640 KB total

Transfer is negligible compared to compute time.

### Why Not Custom Ops?

Initial approach used `ggml_map_custom1` to run FFN on CPU via a callback. This failed because:
1. OpenCL backend doesn't implement `GGML_OP_MAP_CUSTOM1`
2. Graph scheduler can't split graphs at custom op boundaries
3. Cross-backend tensor copies aren't supported for custom ops

The current approach uses standard GGML ops (norm, matmul, silu, mul, add) with weights on the CPU buffer. The graph scheduler handles all cross-backend copies automatically.

## Known Limitations

1. **Performance penalty**: 60-90% slower for dense models (CPU FFN compute bottleneck)
2. **HDD incompatibility**: Models >4GB on HDD are extremely slow with FNN-RAM-CPU
3. **No async transfers**: OpenCL backend doesn't support async cross-backend copies
4. **Output parity**: Outputs may differ slightly between GPU-ONLY and FNN-RAM-CPU due to FP16 vs FP32 precision
5. **No multi-GPU**: FNN-RAM-CPU doesn't support multiple GPUs yet

## Contributing

This is a private fork. For upstream llama.cpp contributions, see [CONTRIBUTING.md](CONTRIBUTING.md).

Key files modified:
- `src/llama-model-loader.cpp` — Tensor routing to CPU buffer, memory eviction
- `src/llama-graph.cpp` — FFN graph construction
- `src/llama-ffn-local.h` — Routing predicates, split modes, MTP detection
- `src/llama-ffn-local.cpp` — CPU FFN compute, callback with USM path
- `ggml/src/ggml-opencl/ggml-opencl.cpp` — `cpy_tensor` implementation
- `src/llama.cpp` — Mode initialization, global ReBAR flag
- `src/llama-context.cpp` — Context split mode initialization
- `src/llama-context.h` — Context split mode storage
- `common/arg.cpp` — CLI argument parsing
- `common/common.cpp` — Parameter conversion
- `common/common.h` — Parameter struct definitions
- `include/llama.h` — Public API parameter structs

---

# FFN-ZERO-CPU Mode (Phase 3)

## Overview

FFN-ZERO-CPU mode extends FNN-RAM-CPU by using **mmap'd weights from SSD** instead of loading all FFN weights into RAM. This minimizes system RAM usage to near-zero for the weights, while keeping attention on GPU.

### Three Zero-CPU Modes

| Mode | Flag | Description |
|------|------|-------------|
| **FNN-RAM-CPU** | `--split-mode fnn-ram-cpu` | FFN weights loaded into CPU RAM |
| **FNN-ZERO-CPU** | `--split-mode fnn-zero-cpu` | FFN weights mmap'd from SSD (minimal RAM) |
| **FNN-ZERO-CPU-OTHER** | `--split-mode fnn-zero-cpu-other` | FFN + SSM/other mmap'd from SSD |
| **FNN-ZERO-CPU-ALL** | `--split-mode fnn-zero-cpu-all` | All non-attention weights mmap'd from SSD |

### How It Works

```
┌─────────────────────────────────────────────────────────────┐
│                  FFN-ZERO-CPU FLOW                           │
│                                                              │
│  ┌──────────┐     ┌──────────┐     ┌──────────┐             │
│  │   SSD    │────▶│  mmap()  │────▶│ CPU FFN  │             │
│  │  (GGUF)  │     │ (kernel  │     │ (compute)│             │
│  │          │     │  page    │     │          │             │
│  │          │     │  cache)  │     │          │             │
│  └──────────┘     └──────────┘     └──────────┘             │
│       │                │                │                    │
│       │         MADV_DONTNEED          │                    │
│       │         (evict after           │                    │
│       │          loading)              │                    │
│       │                │                │                    │
│       ▼                ▼                ▼                    │
│  Weights stay    Pages evicted    FFN runs on               │
│  on SSD, not     from page cache  CPU with                  │
│  consuming RAM   after loading    direct mmap               │
│                                                              │
│  RAM usage: ~500 MB (attention + KV cache only)             │
│  VRAM usage: ~1.5 GB (attention weights + KV cache)         │
└─────────────────────────────────────────────────────────────┘
```

### Memory Eviction

After model loading, the following eviction steps are performed:

1. **`posix_fadvise(POSIX_FADV_DONTNEED)`** — Drops file data from OS page cache
2. **`posix_madvise(POSIX_MADV_DONTNEED)`** — Marks mmap pages as reclaimable
3. **`malloc_trim(0)`** — Returns free heap memory to the OS (Linux)
4. **`EmptyWorkingSet`** — Pushes heap pages to standby list (Windows)
5. **`DiscardVirtualMemory`** — Discards mmap pages entirely (Windows)

This ensures that after loading, the system RAM usage is minimized. The OS can still page-fault weights from SSD when needed during inference.

### MTP (Multi-Token Predictor) Support

FFN-ZERO-CPU mode automatically detects MTP models (tensors with `nextn.` prefix) and keeps ALL tensors in the MTP predictor layer on GPU. This is necessary because:

1. MTP runs during speculative decoding and needs low-latency access
2. If MTP FFN weights are on CPU/SSD, prediction latency reduces speculative decoding effectiveness
3. The MTP layer is small (~500MB) compared to the full model

### Command-Line Reference

| Flag | Description |
|------|-------------|
| `--split-mode fnn-zero-cpu` | FFN weights mmap'd from SSD |
| `--split-mode fnn-zero-cpu-other` | FFN + SSM/other mmap'd from SSD |
| `--split-mode fnn-zero-cpu-all` | All non-attention weights mmap'd from SSD |
| `--use-resize` | Use Resizable BAR for zero-copy CPU-GPU transfers (Intel GPU only) |
| `--no-mmap` | Load weights directly into RAM (disables mmap) |

### Recommended Configurations

**For minimal RAM usage (4B model):**
```bash
llama-cli -m model.gguf \
    --split-mode fnn-zero-cpu \
    -c 4096 \
    -n 256 \
    --jinja
```

**For minimal RAM usage with MTP:**
```bash
llama-cli -m model.MTP.gguf \
    --split-mode fnn-zero-cpu \
    --spec-type draft-mtp \
    --spec-draft-n-max 6 \
    -c 4096 \
    -n 256 \
    --jinja
```

**For Intel Arc with Resizable BAR:**
```bash
llama-cli -m model.gguf \
    --split-mode fnn-zero-cpu \
    --use-resize \
    -c 4096 \
    -n 256 \
    --jinja
```

**For streaming FFN weights from SSD (minimal RAM):**
```bash
llama-cli -m model.gguf \
    --split-mode fnn-zero-cpu \
    --stream \
    -c 4096 \
    -n 256 \
    --jinja
```

**For maximum VRAM reduction (all small tensors on CPU):**
```bash
llama-cli -m model.gguf \
    --split-mode fnn-zero-cpu-all \
    -c 2048 \
    -ctk q8_0 -ctv q8_0 \
    -n 256 \
    --jinja
```

### Benchmark Results

| Model | Mode | Prompt t/s | Gen t/s | RAM Usage | VRAM Usage |
|-------|------|-----------|---------|-----------|------------|
| Qwen3.5-2B | GPU-ONLY | 108.9 | 52.4 | ~2.0 GB | ~2.0 GB |
| Qwen3.5-2B | FNN-RAM-CPU | 54.1 | 16.2 | ~2.5 GB | ~0.7 GB |
| Qwen3.5-2B | FNN-ZERO-CPU | 10.1 | 2.6 | ~0.5 GB | ~0.7 GB |
| Qwen3.5-4B-MTP | GPU-ONLY | 62.7 | 37.8 | ~3.5 GB | ~3.5 GB |
| Qwen3.5-4B-MTP | FNN-RAM-CPU | 28.7 | 12.3 | ~4.0 GB | ~1.3 GB |
| Qwen3.5-4B-MTP | FNN-ZERO-CPU | 28.3 | 12.1 | ~0.8 GB | ~1.3 GB |
| Qwen3.5-4B-MTP | FNN-ZERO-CPU+MTP | 35.3 | 11.6 | ~1.3 GB | ~1.8 GB |
| Qwen3.6-35B-MTP | GPU-ONLY | 29.4 | 11.4 | ~22 GB | ~22 GB |
| Qwen3.6-35B-MTP | FNN-ZERO-CPU+MTP | 40.4 | 8.6 | ~2.0 GB | ~4.0 GB |

**Notes:**
- FNN-ZERO-CPU uses mmap from SSD, so RAM usage is minimal
- MTP predictor layer stays on GPU, adding ~500MB VRAM
- Qwen3.6-35B with FNN-ZERO-CPU+MTP is faster than GPU-ONLY because it avoids VRAM spill
- RAM usage with eviction: ~500MB-2GB depending on model size

### Resizable BAR (ReBAR) Mode

`--use-resize` enables zero-copy CPU-GPU tensor transfers using Intel's Resizable BAR technology. This eliminates the need for explicit `clEnqueueReadBuffer`/`clEnqueueWriteBuffer` calls during inference.

**How it works:**
1. With ReBAR, the entire GPU VRAM is mapped into the CPU's physical address space
2. The CPU can write directly to GPU VRAM using standard memory operations
3. This eliminates driver submission overhead for each transfer

**Requirements:**
- Intel Arc GPU with ReBAR enabled in BIOS
- Linux: `cl_intel_unified_shared_memory` extension
- Windows: Intel Level Zero driver with ReBAR support

**Expected improvement:** 2-5x reduction in transfer overhead for CPU-GPU interleaved inference.

**Environment variables:**
```bash
# Enable immediate command lists (reduces submission overhead)
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
```

### Troubleshooting

#### High RAM usage after loading
Run `echo 3 | sudo tee /proc/sys/vm/drop_caches` to drop the page cache. The eviction code should do this automatically, but manual clearing may be needed.

#### Slow inference with FNN-ZERO-CPU
- Ensure model is on SSD, not HDD
- Use `--no-mmap` to load weights into RAM if you have sufficient RAM
- Install OpenBLAS for faster CPU compute: `sudo apt install libopenblas-dev`

#### MTP produces gibberish
- Ensure MTP predictor layer tensors stay on GPU (automatic with FNN-ZERO-CPU)
- Use `--spec-type draft-mtp` with `--spec-draft-n-max 6`

#### Chat template crash
- Use `--jinja` flag for models with embedded chat templates
- Use `--reasoning off` to disable thinking mode
- Some models require `--single-turn` for non-interactive use
