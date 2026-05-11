with open("llama.cpp-PoC/src/llama-graph.cpp", "r") as f:
    ccontent = f.read()

# Since `llm_graph_context` doesn't have `llama_model`, it has `mctx` or `cparams`?
# Wait, I added `ffn_local` to `llama_context` in `ctx->ffn_local = model.ffn_local`.
# We don't have access to `llama_context` here?
# Let's check `llm_graph_params` in `llama-graph.h`.
