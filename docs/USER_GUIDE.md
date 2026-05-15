# Decoupled Attention/FFN Inference — User Documentation

## Overview

This fork of llama.cpp implements **FFN-to-CPU offload**, which reduces GPU VRAM usage by keeping FFN (Feed-Forward Network) weights in system RAM while running attention layers on GPU. This enables running larger models on GPUs with limited VRAM.

### Two Modes

| Mode | Split Mode Flag | Description |
|------|----------------|-------------|
| **GPU-ONLY** | `--split-mode local-gpu` (default) | All weights on GPU. Maximum performance. |
| **FNN-RAM-CPU** | `--split-mode local-ssd` | FFN weights on CPU, attention on GPU. Reduces VRAM. |
| **GPU-SSD** | (future) | Zero-copy mmap from SSD. Minimal RAM usage. |

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
./llama-cli -m model.gguf -p "Hello" -n 128 --split-mode local-ssd

# With Q8 KV cache (saves more VRAM)
./llama-cli -m model.gguf -p "Hello" -n 128 --split-mode local-ssd -ctk q8_0 -ctv q8_0

# Server mode
./llama-server -m model.gguf --split-mode local-ssd --port 8080
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
| `--split-mode local-gpu` | All weights on GPU (default) |
| `--split-mode local-ssd` | FFN weights on CPU, attention on GPU |
| `--split-mode none` | Single GPU, no splitting |
| `--split-mode layer` | Split layers across multiple GPUs |
| `--split-mode row` | Split rows across multiple GPUs |

### FNN-RAM-CPU Specific Flags

| Flag | Description |
|------|-------------|
| `--split-mode local-ssd` | Enable FFN-to-CPU offload |
| `LLAMA_SPLIT_OTHER=cpu` | Also offload SSM/other tensors to CPU |
| `--no-mmap` | Load weights directly into RAM (recommended for FNN-RAM-CPU) |
| `-ctk q8_0 -ctv q8_0` | Q8 KV cache (saves ~50% KV memory) |
| `-c N` | Context length (smaller = less KV cache) |
| `--reasoning off` | Disable thinking mode (faster) |

### Memory Reduction Flags

| Flag | VRAM Savings | Performance Impact |
|------|-------------|-------------------|
| `--split-mode local-ssd` | 40-60% (FFN weights) | 60-90% slower |
| `-ctk q8_0 -ctv q8_0` | ~50% KV cache | Minimal |
| `-c 4096` | ~75% KV vs 128K | None |
| `--no-mmap` | None | Faster loading |
| `--n-gpu-layers N` | Partial offload | Varies |

### Recommended Configurations

**For 2B-4B models (FFN on CPU is fast enough):**
```bash
llama-cli -m model.gguf \
    --split-mode local-ssd \
    --no-mmap \
    -c 4096 \
    -n 256
```

**For 9B-27B models (keep FFN on GPU, reduce KV):**
```bash
llama-cli -m model.gguf \
    --split-mode local-gpu \
    -ctk q8_0 -ctv q8_0 \
    -c 4096 \
    -n 256
```

**For MoE models (FFN on CPU is FASTER):**
```bash
llama-cli -m model.gguf \
    --split-mode local-ssd \
    --no-mmap \
    -c 4096 \
    -n 256
```

**For maximum VRAM reduction (all small tensors on CPU):**
```bash
LLAMA_SPLIT_OTHER=cpu llama-cli -m model.gguf \
    --split-mode local-ssd \
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
    --split-mode local-ssd \
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
- Use FNN-RAM-CPU mode (`--split-mode local-ssd`)
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
- `src/llama-model-loader.cpp` — Tensor routing to CPU buffer
- `src/llama-graph.cpp` — FFN graph construction
- `src/llama-ffn-local.h` — Routing predicates and split modes
- `ggml/src/ggml-opencl/ggml-opencl.cpp` — `cpy_tensor` implementation
- `src/llama.cpp` — Mode initialization
