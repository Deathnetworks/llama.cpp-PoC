# Decoupled Attention/FFN Inference — Implementation Plan

## Goal
Run massive models (1T+ parameters) on consumer hardware with minimal RAM (16GB) and VRAM (32GB).
Only attention weights + KV cache stay in VRAM. FFN weights stream from SSD per-layer.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     INFERENCE PIPELINE                           │
│                                                                  │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐                   │
│  │  GPU VRAM │    │  CPU RAM │    │   SSD    │                   │
│  │          │    │          │    │          │                   │
│  │ Attention│◄──►│ Residual │    │ FFN      │                   │
│  │ Weights  │    │ Stream   │    │ Weights  │                   │
│  │ KV Cache │    │ (4-16KB) │    │ (Q4)     │                   │
│  └──────────┘    └──────────┘    └──────────┘                   │
│       │              │               │                          │
│       │         ┌────┴────┐          │                          │
│       │         │ Double  │          │                          │
│       │         │ Buffer  │◄─────────┘                          │
│       │         │ (2× max │  async read                         │
│       │         │  layer) │                                     │
│       │         └─────────┘                                     │
│       │                                                         │
│  Per-token flow:                                                │
│  1. GPU computes Attention(layer N)                             │
│  2. While GPU waits, CPU reads FFN weights for layer N+1        │
│  3. GPU→CPU: copy residual (4KB)                                │
│  4. CPU: dequantize Q4→F32, compute FFN                         │
│  5. CPU→GPU: copy result (4KB)                                  │
│  6. Evict layer N weights from RAM                              │
│  7. Next layer                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Implementation Phases

### Phase 1: Synchronous Streaming (CURRENT)
- [x] Store file offsets for each layer's FFN weights
- [x] Allocate double buffer for weight streaming
- [x] Implement synchronous read() per layer
- [x] Add --stream CLI flag
- [ ] Integrate with graph scheduler (load before each FFN layer)
- [ ] Test and measure RAM usage

### Phase 2: Async I/O with io_uring (Linux)
- [ ] Implement io_uring-based async reads
- [ ] Double-buffer ping-pong: load N+1 while computing N
- [ ] Overlap disk I/O with GPU attention compute
- [ ] Measure latency improvement

### Phase 3: Windows Native (Overlapped I/O)
- [ ] Implement Overlapped I/O with IOCP
- [ ] Same double-buffer architecture
- [ ] Test on Windows native SYCL backend

### Phase 4: On-the-fly Dequantization
- [ ] Write custom CPU kernels that dequantize Q4→F32 during matmul
- [ ] Eliminate separate F32 buffer entirely
- [ ] RAM usage: only Q4 weights in buffer (~50% of F32)
- [ ] Better cache utilization

### Phase 5: 1T Model Support
- [ ] Handle models with 200+ layers
- [ ] Optimize for very large hidden dimensions
- [ ] Test with actual 1T parameter models

## Key Technical Decisions

### Why not mmap + madvise?
- mmap keeps pages in OS page cache during use
- MADV_DONTNEED after loading doesn't help — pages get faulted back during inference
- With 33+ layers, by the time layer 33 runs, layers 1-32 are in cache
- Result: all model weights end up in RAM anyway

### Why streaming is better:
- Only 1 layer's weights in RAM at any time
- Predictable, bounded RAM usage
- Works for arbitrarily large models
- Trade-off: disk I/O per layer (hidden by async overlap)

### RAM Budget (4B model, Q4):
- Attention weights + KV cache: ~1.3 GB (VRAM)
- 1 layer FFN weights (Q4): ~50 MB (RAM)
- Residual buffer: ~16 KB (RAM)
- Total RAM: ~100 MB (vs 3.5 GB for GPU-ONLY)

### RAM Budget (1T model, Q4, estimated):
- Attention weights + KV cache: ~20 GB (VRAM)
- 1 layer FFN weights (Q4): ~2 GB (RAM)
- Residual buffer: ~64 KB (RAM)
- Total RAM: ~2.5 GB (vs 250+ GB for GPU-ONLY)

## File Structure

```
src/
  llama-ffn-local.h      — Layer struct, routing predicates
  llama-ffn-local.cpp    — Loader, CPU FFN compute, callback
  llama-ffn-async.cpp    — Async double-buffer engine
  llama-model-loader.cpp — Tensor routing (CPU vs GPU buffer)
  llama-graph.cpp        — Graph construction
  llama-context.cpp/h    — Context params, mode initialization
  llama.cpp              — Model load, global flags
common/
  arg.cpp                — CLI argument parsing
  common.h               — Parameter structs
  common.cpp             — Parameter conversion
include/
  llama.h                — Public API
docs/
  USER_GUIDE.md          — User documentation
```

## Current Blockers

1. **Graph scheduler integration**: Need to hook into FFN layer execution to load weights before each layer
2. **GGUF alignment**: Tensor data offsets may not be 4KB aligned (required for O_DIRECT)
3. **Dequantization**: Current approach dequantizes entire layer to F32 buffer (needs on-the-fly for optimal RAM)

## Performance Targets

| Metric | GPU-ONLY | mmap+madvise | --stream (sync) | --stream (async) |
|--------|----------|--------------|-----------------|------------------|
| RAM (4B) | 3.5 GB | 0.8 GB | 0.2 GB | 0.2 GB |
| RAM (35B) | 22 GB | 2.0 GB | 0.5 GB | 0.5 GB |
| Speed (4B) | 100% | 40% | 30% | 80% |
| Speed (35B) | 100% | 70% | 50% | 90% |

## Next Immediate Steps

1. Complete synchronous streaming integration with graph scheduler
2. Test RAM usage with --stream flag
3. Measure performance impact
4. Implement io_uring async reads
5. Add on-the-fly dequantization
