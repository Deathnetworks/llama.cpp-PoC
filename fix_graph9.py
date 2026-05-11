with open("llama.cpp-PoC/src/llama-graph.cpp", "r") as f:
    ccontent = f.read()

ccontent = ccontent.replace("ud->ffn_local = mctx->ffn_local;", "ud->ffn_local = model.ffn_local;")
ccontent = ccontent.replace("    ud->ffn_local = model.ffn_local;", "    ud->ffn_local = model.ffn_local;")

with open("llama.cpp-PoC/src/llama-graph.cpp", "w") as f:
    f.write(ccontent)
