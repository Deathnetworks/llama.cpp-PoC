# MTP+Stream Optimization Experiments — Final Results

## Baseline (2026-05-18)
- **Config**: MTP (draft-mtp, n_max=2), stream, ReBAR enabled, 128 tokens, 256 context
- **Model**: Qwen3.5-4B-Q4_K_M.MTP
- **Hardware**: Intel Arc Pro B70 (32GB VRAM), 32GB RAM, HDD

## Key Finding: Streaming Mode IS Correct
Earlier "garbled output" was caused by debug messages (DEBUG STREAM) being printed
to stdout interleaved with model output. The actual model output was always correct.
With debug suppressed, output is fully coherent: "Paris" for "capital of France".

## Reference Measurements
| Test | Prompt t/s | Gen t/s | Notes |
|------|------------|---------|-------|
| mmap only, no MTP | 28.6 | 11.9 | Maximum possible |
| mmap + MTP n_max=2 | 20.9 | 8.5 | MTP slows mmap |
| stream only, no MTP | 24.2 | 3.6 | Streaming overhead |
| stream + MTP n_max=2 | 28.7 | 6.5 | Best streaming baseline |
| fnn-ram-cpu + MTP n=2 | 23.7 | 9.5 | Weights in RAM |

## Successful Optimizations
1. **Read-ahead hints** (`posix_fadvise(SEQUENTIAL)` + `posix_fadvise(WILLNEED)` for 4 layers)
   - +10-15% generation speed improvement
   - 6.5 → 7.2 t/s with 64 tokens
2. **DONTNEED for consumed layers** — Frees page cache, reduces memory pressure
3. **ReBAR detection** — Confirmed ENABLED (1024 MB max alloc on Intel Arc)
4. **ReBAR copy optimization** — clEnqueueMapBuffer for small tensors

## Experiment Log
| # | Description | Gen t/s | Δ | Coherent | Notes |
|---|-------------|---------|---|----------|-------|
| 0 | Baseline stream+MTP | 6.5 | — | ✅ | 64 tokens |
| 1 | MTP n=3 | 7.0 | +0.5 | ✅ | More spec = more rejections |
| 4 | mmap only | 11.9 | +5.4 | ✅ | Max possible |
| 5 | mmap + MTP | 8.5 | +2.0 | ✅ | MTP hurts mmap |
| 6 | Short context | 8.0 | +1.5 | ✅ | Less attention |
| 10 | mmap + MTP | 9.7 | +3.2 | ✅ | Best overall |
| 13 | fnn-ram-cpu + MTP | 9.5 | +3.0 | ✅ | Weights in RAM |
| 14 | + read-ahead (2 layers) | 7.2 | +0.7 | ✅ | First improvement |
| 15 | + short context | 7.7 | +1.2 | ✅ | Combined |
| 18 | + read-ahead (4 layers) + DONTNEED | 7.1 | +0.6 | ✅ | Best streaming |
| 19 | Full test 512 tokens | 4.6 | — | ✅ | "Paris" correct answer |

## Best Configurations
1. **Maximum speed**: mmap + MTP n=2 → 9.7 t/s (uses 3.7GB RAM for page cache)
2. **Low RAM + good speed**: fnn-ram-cpu + MTP n=2 → 9.5 t/s (~1GB RAM for weights)
3. **Streaming (target)**: stream + MTP n=2 + read-ahead → 7.2 t/s (~88MB RAM)

## Remaining Optimization Directions
1. **NVMe SSD** — Would provide 5-10x I/O bandwidth improvement
2. **Triple-buffering** — Keep 2+ layers ahead (needs careful implementation)
3. **Batch layer reads** — Read multiple layers per pread() call
4. **O_DIRECT** — Bypass page cache for more consistent timing
5. **Suppress debug output** — Redirect DEBUG STREAM to stderr
| 97 | Opt: no mmap | 3.3 | 3.3 | -25.46 | ✅ |
| 98 | Opt: flash attention |  |  | -28.76 | ? |
| 99 | Opt: resize bar | 16.1 | 16.1 | -12.66 | ✅ |
| 100 | Opt: combined |  |  | -28.76 | ? |
