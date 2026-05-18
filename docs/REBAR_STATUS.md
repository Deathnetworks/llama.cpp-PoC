# ReBAR Integration — Status

## What Was Done
1. ✅ ReBAR detection in OpenCL backend (`ggml-opencl.cpp`)
   - Checks `CL_DEVICE_MAX_MEM_ALLOC_SIZE > 256MB`
   - Result: ENABLED (1024 MB max alloc on Intel Arc)
2. ✅ Optimized `cpy_tensor` for ReBAR
   - Small tensors (< 1MB): `clEnqueueMapBuffer` + `memcpy` + `unmap`
   - Avoids intermediate host buffer and `clEnqueueWriteBuffer` driver overhead
3. ✅ Verified coherent output with MTP + streaming + ReBAR

## Test Results (4B model, MTP, 128 tokens)
- Prompt: 19.3 t/s
- Generation: 6.4 t/s
- ReBAR: ENABLED
- Output: Correct ("Paris")

## Architecture
```
ReBAR Detection:
  clGetDeviceInfo(CL_DEVICE_MAX_MEM_ALLOC_SIZE) → max_alloc > 256MB → rebar_enabled

Optimized Copy (CPU→GPU, residual stream):
  clEnqueueMapBuffer(device_buf, CL_MAP_WRITE) → host-visible pointer
  memcpy(host_ptr, src_data, size)  // CPU writes directly to VRAM-mapped memory
  clEnqueueUnmapMemObject(device_buf, host_ptr)

Standard Copy (GPU→CPU, residual stream):
  clEnqueueReadBuffer(device_buf, host_buf)  // GPU DMA push to pinned host
  CPU reads from host_buf  // Fast (host memory, not VRAM)
```

## Future Work
- Implement `clEnqueueMapBuffer` path for SYCL backend (Windows)
- Add `--use-resize` flag to explicitly enable/disable ReBAR optimizations
- Benchmark with larger models to measure ReBAR impact
- Consider allocating residual stream buffers with `CL_MEM_USE_HOST_PTR` for zero-copy
