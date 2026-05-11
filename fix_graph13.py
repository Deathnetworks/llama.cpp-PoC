with open("llama.cpp-PoC/src/llama-graph.cpp", "r") as f:
    ccontent = f.read()

# Since we don't have access to ffn_local directly in llm_graph_context or llm_graph_params, we can add it to `llama_cparams`!
# Let's add it to `llama_cparams.h`.

with open("llama.cpp-PoC/src/llama-cparams.h", "r") as f:
    hcontent = f.read()

if "ffn_local" not in hcontent:
    hcontent = hcontent.replace(
        "    int32_t n_threads_batch;",
        "    int32_t n_threads_batch;\n\n    struct ffn_mmap_t* ffn_local = nullptr;\n    int ffn_mode = 0;"
    )
    # Include llama-ffn-local.h
    hcontent = hcontent.replace(
        '#include "llama.h"',
        '#include "llama.h"\nstruct ffn_mmap_t;'
    )
    with open("llama.cpp-PoC/src/llama-cparams.h", "w") as f:
        f.write(hcontent)

# And now populate it in `llama_cparams::llama_cparams` in `llama-cparams.cpp`.
with open("llama.cpp-PoC/src/llama-cparams.cpp", "r") as f:
    cppcontent = f.read()

if "ffn_local" not in cppcontent:
    cppcontent = cppcontent.replace(
        "    n_threads_batch(params.n_threads_batch)",
        "    n_threads_batch(params.n_threads_batch),\n    ffn_mode(params.ffn_mode)"
    )
    with open("llama.cpp-PoC/src/llama-cparams.cpp", "w") as f:
        f.write(cppcontent)

# And in `llama_init_from_model` in `llama-context.cpp`:
with open("llama.cpp-PoC/src/llama-context.cpp", "r") as f:
    ctxcontent = f.read()

if "cparams.ffn_local =" not in ctxcontent:
    ctxcontent = ctxcontent.replace(
        "ctx->cparams = llama_cparams(params);",
        "ctx->cparams = llama_cparams(params);\n    ctx->cparams.ffn_local = model->ffn_local;"
    )
    with open("llama.cpp-PoC/src/llama-context.cpp", "w") as f:
        f.write(ctxcontent)

# And then in `llama-graph.cpp`, we can use `cparams.ffn_local`.
ccontent = ccontent.replace("    ud->ffn_local = model.ffn_local;", "    ud->ffn_local = cparams.ffn_local;")

with open("llama.cpp-PoC/src/llama-graph.cpp", "w") as f:
    f.write(ccontent)
