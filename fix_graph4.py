with open("llama.cpp-PoC/src/llama-graph.cpp", "r") as f:
    ccontent = f.read()

if "build_ffn_local" not in ccontent:
    ccontent = ccontent.replace(
        "ggml_tensor * llm_graph_context::build_ffn(",
        """
static void llm_ffn_local_op(struct ggml_tensor * dst , const struct ggml_tensor * a, int ith, int nth, void * userdata) {
    // We execute our local ffn logic here!
    // But wait, we need the ffn_local pointer, the layer index.
    // We can store a pointer to a struct in userdata.
}

ggml_tensor * llm_graph_context::build_ffn_local(
         ggml_tensor * cur,
                 int   il) const {
    // To do this properly we need to put a custom operator that will be run by the CPU backend.
    return cur;
}

ggml_tensor * llm_graph_context::build_ffn("""
    )
    with open("llama.cpp-PoC/src/llama-graph.cpp", "w") as f:
        f.write(ccontent)
