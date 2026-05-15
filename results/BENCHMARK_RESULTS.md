# Decoupled Attention/FFN — Final Benchmark Results
## Intel Arc Pro B70 (31.16 GB VRAM) via WSL2 + OpenCL

### Test Configuration
- Context: 256 tokens (MoE: 256)
- KV cache: F16 (on GPU)
- GPU layers: 99 (all)
- Prompt: "The capital of France is"
- Reasoning: off (--reasoning off)
- mmap: off (--no-mmap)
- BLAS: OpenBLAS enabled
- Storage: Mechanical HDD (bottleneck for large models)

### Dense Models

#### 2B Model (Qwen3.5-2B-Q4_K_M, 1222 MB, 24 layers)
| Metric | local-gpu | local-ssd | Delta |
|--------|-----------|-----------|-------|
| GPU VRAM | 1455 MiB | 954 MiB | **-525 MiB (-35%)** |
| Host RAM | 402 MiB | 943 MiB | +541 MiB |
| Prompt t/s | 143.4 | 54.6 | -62% |
| Gen t/s | 67.1 | 11.3 | -83% |

#### 4B Model (Qwen3.5-4B-Q4_K_M, 2614 MB, 32 layers)
| Metric | local-gpu | local-ssd | Delta |
|--------|-----------|-----------|-------|
| GPU VRAM | 2909 MiB | 1598 MiB | **-1311 MiB (-45%)** |
| Host RAM | 502 MiB | 1837 MiB | +1335 MiB |
| Prompt t/s | 78.1 | 31.8 | -59% |
| Gen t/s | 37.8 | 6.8 | -82% |

#### 9B Model (Qwen3.5-9B-Q4_K_M, 5417 MB, 32 layers)
| Metric | local-gpu | local-ssd | Delta |
|--------|-----------|-----------|-------|
| GPU VRAM | 5170 MiB | 2376 MiB | **-2794 MiB (-54%)** |
| Host RAM | 553 MiB | 3379 MiB | +2826 MiB |
| Prompt t/s | 60.4 | 17.5 | -71% |
| Gen t/s | 34.4 | 7.6 | -78% |

#### 27B Model (Qwen3.5-27B-Q4_K_M, 15965 MB, 64 layers)
| Metric | local-gpu | local-ssd | Delta |
|--------|-----------|-----------|-------|
| GPU VRAM | 15690 MiB | 5804 MiB | **-9886 MiB (-63%)** |
| Host RAM | 692 MiB | 10624 MiB | +9932 MiB |
| Prompt t/s | 18.8 | ~1.5 | ~-92% |
| Gen t/s | 21.1 | ~1.2 | ~-94% |

### MoE Model

#### Gemma-4-26B-A4B (16.9 GB, 30 layers, 8 active / 128 total experts)
| Metric | local-gpu | local-ssd | Delta |
|--------|-----------|-----------|-------|
| GPU VRAM | 3037 MiB | 2235 MiB | **-802 MiB (-26%)** |
| Host RAM | 6056+8167 MiB | 15023 MiB | +8967 MiB |
| Prompt t/s | 12.1 | 21.3 | **+76%** |
| Gen t/s | 3.1 | 7.3 | **+135%** |

**Note**: MoE local-ssd is FASTER than local-gpu because:
1. FFN weights (14.3 GB) exceed available VRAM with local-gpu
2. Local-gpu spills to CPU_REPACK (8167 MiB), causing massive slowdown
3. Local-ssd keeps FFN weights on CPU by design, avoiding spill

### Key Findings

1. **VRAM savings match expected FFN weight sizes** for dense models
2. **MoE models benefit enormously from CPU offload** — faster AND less VRAM
3. **Performance penalty for dense models: 60-92%** (HDD-bound for large models)
4. **MoE performance gain: 76-135%** (avoids VRAM spill)
5. **Output is coherent across all models and modes**

### Recommended Split Design

**For dense models (≤4B): FFN on CPU is viable**
- VRAM savings: 45-50%
- Performance penalty: 60-80%
- Use case: Running larger models on limited VRAM

**For dense models (≥9B): Keep FFN on GPU, use Q8 KV cache**
- FFN compute too heavy for CPU on HDD
- Q8 KV cache saves ~50% KV memory
- Smaller context length reduces KV cache

**For MoE models: FFN on CPU is optimal**
- Only active experts need to be computed
- Avoids VRAM spill from large expert weights
- Can be FASTER than all-GPU due to avoiding repack

### Architecture

The fix uses standard GGML graph ops (no custom ops):
1. FFN weights routed to pure CPU buffer in model loader
2. FFN graph built using standard ops (norm, matmul, silu, mul, add)
3. Graph scheduler automatically inserts GPU↔CPU copies at FFN boundaries
4. KV cache stays on GPU with attention weights
5. No mmap — weights loaded directly into CPU RAM
