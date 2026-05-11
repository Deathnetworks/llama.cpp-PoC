with open("llama.cpp-PoC/src/llama-graph.cpp", "r") as f:
    content = f.read()

# I will implement do_ffn_cpu_path inside `llama-graph.cpp` or maybe right after building the ffn graph?
# Wait, Phase 3 spec says:
# The only change to `llama_decode_internal` is replacing the unconditional FFN graph call with a dispatch:
# ```cpp
# // for each layer il:
# cur = llm_build_attn(ctx0, batch, il, kv_cache);  // unchanged
# ggml_backend_graph_compute(gpu_backend, attn_graph);
#
# if (lctx.ffn_mode == FFN_LOCAL) {
#     // spec §5.3 — transfer, BLAS, transfer back
#     do_ffn_cpu_path(lctx, il, cur, n_tokens, n_embd);
# } else {
#     cur = llm_build_ffn(ctx0, cur, ...);           // unchanged default path
# }
# ```
# But `llama_decode_internal` doesn't exist anymore! It's now `llama_model::build_graph` and architecture-specific `build_arch_graph`.
# The layer loop is in `build_arch_graph`! For instance, in `src/models/llama.cpp`.
