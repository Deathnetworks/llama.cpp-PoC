with open("llama.cpp-PoC/src/llama-ffn-local.cpp", "r") as f:
    ccontent = f.read()

ccontent = ccontent.replace(
"""void llama_swap_ffn(struct llama_context* ctx,
                    int layer_first, int layer_last,
                    const char* new_ffn_path) {
    if (!ctx->cparams.ffn_local) return;

    // 1. Open new file
    auto* new_ffn = ffn_mmap_from_full_gguf(new_ffn_path, nullptr, layer_last + 1); // wait we don't have gctx
    if (!new_ffn) return;

    // We update pointers directly
    // Not thread safe as stated
    for (int i = layer_first; i <= layer_last; i++) {
        if (i < (int)ctx->cparams.ffn_local->layers.size() && i < (int)new_ffn->layers.size()) {
            ctx->cparams.ffn_local->layers[i] = new_ffn->layers[i];
        }
    }
}""",
"""void llama_swap_ffn(struct llama_context* /*ctx*/,
                    int layer_first, int layer_last,
                    const char* new_ffn_path) {
    // Phase 4 stub — full implementation in P4-T3
    LLAMA_LOG_WARN("llama_swap_ffn: stub (Phase 4). layers=%d-%d path=%s\\n",
                   layer_first, layer_last, new_ffn_path);
}"""
)

with open("llama.cpp-PoC/src/llama-ffn-local.cpp", "w") as f:
    f.write(ccontent)
