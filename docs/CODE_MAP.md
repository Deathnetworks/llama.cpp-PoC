# CODE_MAP — Compact Reference (Updated)

## Architecture (Per Spec)
```
FFN weights: ONE contiguous block read per layer (gate+up+down)
Buffer A/B: 2× page-aligned, sized to largest FFN layer
I/O: O_DIRECT+io_uring (Linux) / FILE_FLAG_NO_BUFFERING+IOCP (Windows)
Tensor pointers: swapped in ggml graph before each FFN layer compute
NEVER: mmap/madvise for FFN weights
```

## Key Files
```
llama-model.h:585-590    — Model fields (ffn_weight_map, ffn_n_layers)
llama-model.cpp:964-970  — Destructor (delete ffn_weight_map)
llama.cpp:445-462        — Populate ffn_weight_map from ml.weights_map
llama-context.h:248-260  — Context streaming fields (public, before private:)
llama-context.cpp:3605-3630 — Init streaming (copy from model, init async_buf)
llama-context.cpp:2368-2470 — graph_compute(): stream_cb setup + callback
llama-ffn-local.h:11-20   — ffn_weight_offset struct (shared)
llama-ffn-local.h:19-50   — ffn_layer_ptrs_t (file_off_*, size_*, async_valid)
llama-ffn-async.cpp:ALL   — Async engine (init, load_layer, free)
src/CMakeLists.txt:31-34  — Build (llama-ffn-local.cpp, llama-ffn-async.cpp)
common/arg.cpp:2432-2455  — CLI flags
```

## Data Flow
```
MODEL LOAD:
  llama.cpp:llama_model_load()
    → loader.load_all_data() → FFN tensors → CPU buffer (mmap'd)
    → NEW: ffn_weight_map["blk.N.ffn_gate.weight"] = {idx, off, size}
    → model->ffn_weight_map = weight_map
    → model->ffn_n_layers = model->hparams.n_layer

CONTEXT INIT:
  llama_init_from_model()
    → ctx->ffn_weight_map = model->ffn_weight_map
    → ctx->ffn_n_layers = model->ffn_n_layers
    → ctx->ffn_model_path = model->path
    → ctx->ffn_async = new ffn_async_buffer()
    → ffn_async_init(ctx->ffn_async, model->path, nullptr, n_layers)

GRAPH EXECUTE:
  graph_compute()
    → stream_ud = {ffn_weight_map, ffn_async, n_layers, -1, -1}
    → if (zero_ram && ffn_weight_map && ffn_async->initialized):
        ggml_backend_sched_set_eval_callback(sched, stream_cb, &stream_ud)
    → ggml_backend_sched_graph_compute_async(sched, gf)

STREAMING CALLBACK:
  stream_cb(t, ask, ud):
    if (!ask) return true
    if (t->op != MUL_MAT) return false
    w = t->src[0]
    if (w.name not in ffn_weight_map) return false
    layer = parse_layer(w.name)
    if (layer != loaded_layer):
      ffn_async_load_layer(async_buf, layer, nullptr)  ← loads ALL FFN tensors for layer
      loaded_layer = layer
    offset = compute_offset(w.name, ffn_weight_map)  ← cumulative size offset
    w->data = buf + offset
    return true

ffn_async_load_layer(ab, layer, ffn):
  → Reads ALL FFN tensors for layer from mmap (fallback)
  → TODO: Replace with O_DIRECT pread() of contiguous block
```

## Missing Pieces (Priority Order)
1. Build currently fails — need to fix compilation errors
2. ffn_async_load_layer uses memcpy from mmap — must use O_DIRECT pread()
3. Need AsyncLayerStreamer abstract class (per spec)
4. Need io_uring implementation (Linux)
5. Need IOCP implementation (Windows)
6. GGUF 4KB alignment check
7. Test RAM usage with --stream
