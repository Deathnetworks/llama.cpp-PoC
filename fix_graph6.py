with open("llama.cpp-PoC/src/llama-graph.cpp", "r") as f:
    ccontent = f.read()

custom_op = """
struct ffn_local_userdata {
    const ffn_mmap_t* ffn_local;
    int il;
};

static void llm_ffn_local_op(struct ggml_tensor * dst , const struct ggml_tensor * a, int ith, int nth, void * userdata) {
    if (ith != 0) return; // Only run on first thread to avoid race conditions. Actually we could parallelize, but BLAS does that.
    auto* ud = (ffn_local_userdata*)userdata;
    int n_tokens = dst->ne[1];
    int n_embd = dst->ne[0];

    // Prefetch next
    ffn_mmap_prefetch(ud->ffn_local, ud->il);

    // CPU BLAS FFN (in-place on cpu_hidden_f32)
    // Wait! `dst->data` is fp32 if we convert it.
    // In our custom op, `dst` is whatever `cur` was. Usually F32.
    // Let's check `cur->type`.
    if (dst->type == GGML_TYPE_F32) {
        llm_compute_ffn_cpu(ud->ffn_local, ud->il, (float*)dst->data, n_tokens, n_embd);
    } else {
        // We only support F32 for now
    }
}

ggml_tensor * llm_graph_context::build_ffn_local(
         ggml_tensor * cur,
                 int   il) const {

    // We need to pass the userdata.
    // We can allocate it in the ggml context.
    ffn_local_userdata* ud = (ffn_local_userdata*) ggml_new_tensor_1d(ctx0, GGML_TYPE_I8, sizeof(ffn_local_userdata))->data;
    ud->ffn_local = model.ffn_local;
    ud->il = il;

    // Ensure cur is F32 for our custom BLAS op
    if (cur->type != GGML_TYPE_F32) {
        cur = ggml_cast(ctx0, cur, GGML_TYPE_F32);
    }

    cur = ggml_map_custom1_inplace(ctx0, cur, llm_ffn_local_op, 1, ud);

    return cur;
}
"""

ccontent = ccontent.replace(
"""static void llm_ffn_local_op(struct ggml_tensor * dst , const struct ggml_tensor * a, int ith, int nth, void * userdata) {
    // We execute our local ffn logic here!
    // But wait, we need the ffn_local pointer, the layer index.
    // We can store a pointer to a struct in userdata.
}

ggml_tensor * llm_graph_context::build_ffn_local(
         ggml_tensor * cur,
                 int   il) const {
    // To do this properly we need to put a custom operator that will be run by the CPU backend.
    return cur;
}""", custom_op)

# Include the headers
ccontent = ccontent.replace(
    '#include "llama-graph.h"',
    '#include "llama-graph.h"\n#include "llama-ffn-local.h"\n#include "llama-model.h"'
)

with open("llama.cpp-PoC/src/llama-graph.cpp", "w") as f:
    f.write(ccontent)
