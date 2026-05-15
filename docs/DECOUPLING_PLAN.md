# Decoupled Attention/FFN Inference — WSL2 + OpenCL
## Intel Arc Pro B70 via WSL2 GPU Passthrough

### Environment
- GPU: Intel Arc Pro B70 (31.16 GB VRAM, 256 CUs @ 2800 MHz)
- WSL2: Ubuntu with /dev/dxg (Windows GPU passthrough bridge)
- OpenCL: Working via intel-opencl-icd (libigdrcl.so) — GPU initialized and verified
- Vulkan: Broken — missing dzn_icd.json, /dev/dri absent. OpenCL is the viable path.
- D3D12 libs: /usr/lib/wsl/lib/libd3d12.so and libd3d12core.so present

### Build
```bash
source /tmp/venv/bin/activate
cmake -B build_opencl \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_OPENCL=ON \
    -DGGML_OPENCL_EMBED_KERNELS=ON \
    -DGGML_OPENCL_USE_ADRENO_KERNELS=OFF \
    -DOpenCL_LIBRARY=/usr/lib/x86_64-linux-gnu/libOpenCL.so.1 \
    -DOpenCL_INCLUDE_DIR=/usr/include
cmake --build build_opencl --target llama-cli -j$(nproc)
```

### Key Findings

#### 1. OpenCL Backend Works
- GPU detected: Intel(R) Graphics [0xe223] (OpenCL 3.0 NEO)
- Driver: 26.18.38308.1
- FP16 support: true
- Max mem alloc size: 1024 MB (per buffer)
- Global mem: 31906 MB
- Max workgroup size: 1024

#### 2. GGML_OPENCL_USE_ADRENO_KERNELS Must Be OFF
The default ON causes "drop unsupported device" on Intel Arc.
Must rebuild with -DGGML_OPENCL_USE_ADRENO_KERNELS=OFF.

#### 3. Model Memory Usage
Qwen3-0.6B-Q4_K_M.gguf (462 MB on disk):
- GPU model buffer: ~4480 MiB (reported by memory breakdown)
- This is the GGML buffer allocation including all weights + alignment padding
- The OpenCL backend uses ggml_nbytes(tensor) for buffer sizes (compressed)
- The 4480 MiB includes multiple 1GB chunks (max_alloc_size = 1024 MB)
- Actual quantized weight data: ~257 MB (per layer size analysis)

#### 4. Split Mode Behavior
- local-gpu: All weights on GPU, all compute on GPU. Fast (238 t/s prompt).
- local-ssd: All weights STILL on GPU, but FFN computed via ffn_local_callback
  (GPU→CPU transfer, CPU FFN, CPU→GPU transfer). Slow (2.3 t/s prompt).
- The split mode does NOT control weight placement — it controls compute path.
- Weight placement is controlled by -ngl (number of GPU layers).

#### 5. The Real Decoupling Challenge
The current FFN_LOCAL path in llama-graph.cpp intercepts FFN computation
to use CPU, but the weights are still allocated on the GPU buffer.
For true VRAM savings, we need to:
  a) Keep attention weights + KV cache on GPU
  b) Keep FFN weights in CPU RAM (mmap'd from GGUF)
  c) Transfer residual stream GPU↔CPU between layers
  d) Compute FFN on CPU with mmap'd weights

This requires modifying the weight loading to NOT allocate FFN weights
on the GPU buffer when ffn_mode == FFN_LOCAL.

### Performance Comparison
| Mode | Prompt t/s | Gen t/s | GPU VRAM |
|------|-----------|---------|----------|
| local-gpu, -ngl 99 | 238.1 | 100.5 | ~5151 MiB |
| local-ssd | 2.3 | 1.9 | ~4998 MiB |
| local-gpu, -ngl 0 | 38.2 | 15.1 | ~1024 MiB |

### Next Steps for True Decoupling
1. Modify weight loading to skip FFN weight allocation on GPU when FFN_LOCAL
2. Implement CPU-side FFN compute using mmap'd GGUF weights
3. Add GPU↔CPU residual stream transfer (clEnqueueReadBuffer/WriteBuffer)
4. Verify output matches baseline
