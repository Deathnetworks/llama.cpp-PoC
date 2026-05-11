with open("llama.cpp-PoC/src/llama-graph.cpp", "r") as f:
    ccontent = f.read()

ccontent = ccontent.replace("ud->ffn_local = model.ffn_local;", "ud->ffn_local = mctx->ffn_local;")
# But wait, `llm_graph_context` might not have `mctx->ffn_local`.
# Let's check `llm_graph_context` members.
