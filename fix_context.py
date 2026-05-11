import re

with open("llama.cpp-PoC/src/llama-context.h", "r") as f:
    hcontent = f.read()

# Add ffn_local
if "ffn_mmap_t* ffn_local;" not in hcontent:
    hcontent = hcontent.replace(
        "    struct ggml_threadpool * threadpool_batch = nullptr;",
        "    struct ggml_threadpool * threadpool_batch = nullptr;\n\n    struct ffn_mmap_t* ffn_local = nullptr;\n    ggml_fp16_t* cpu_hidden_f16 = nullptr;\n    float* cpu_hidden_f32 = nullptr;\n    ffn_mode_t ffn_mode = FFN_GPU;"
    )
    # Include llama-ffn-local.h at the top
    hcontent = hcontent.replace(
        '#include "llama.h"',
        '#include "llama.h"\n#include "llama-ffn-local.h"'
    )
    with open("llama.cpp-PoC/src/llama-context.h", "w") as f:
        f.write(hcontent)

with open("llama.cpp-PoC/src/llama-model.h", "r") as f:
    hmcontent = f.read()

if "ffn_mmap_t* ffn_local;" not in hmcontent:
    hmcontent = hmcontent.replace(
        "    int64_t t_start_us = 0;",
        "    int64_t t_start_us = 0;\n    struct ffn_mmap_t* ffn_local = nullptr;"
    )
    hmcontent = hmcontent.replace(
        '#include "llama.h"',
        '#include "llama.h"\n#include "llama-ffn-local.h"'
    )
    with open("llama.cpp-PoC/src/llama-model.h", "w") as f:
        f.write(hmcontent)

with open("llama.cpp-PoC/include/llama.h", "r") as f:
    ccontent = f.read()

if "int ffn_mode" not in ccontent:
    ccontent = ccontent.replace(
        "    struct llama_context_params {",
        "    struct llama_context_params {\n        int ffn_mode;"
    )
    # also add it to model params
    ccontent = ccontent.replace(
        "    struct llama_model_params {",
        "    struct llama_model_params {\n        int ffn_mode;\n        const char* ffn_file;"
    )
    with open("llama.cpp-PoC/include/llama.h", "w") as f:
        f.write(ccontent)

# update llama-context.cpp
with open("llama.cpp-PoC/src/llama-context.cpp", "r") as f:
    cppcontent = f.read()

if "cparams.ffn_mode == FFN_LOCAL" not in cppcontent:
    cppcontent = cppcontent.replace(
        "    if (params.seed == LLAMA_DEFAULT_SEED) {",
        """
    if (cparams.ffn_mode == FFN_LOCAL) {
        size_t hbytes_f16 = (size_t)hparams.n_embd * cparams.n_batch * sizeof(ggml_fp16_t);
        size_t hbytes_f32 = (size_t)hparams.n_embd * cparams.n_batch * sizeof(float);
        ctx->cpu_hidden_f16 = static_cast<ggml_fp16_t*>(malloc(hbytes_f16));
        ctx->cpu_hidden_f32 = static_cast<float*>(malloc(hbytes_f32));
        GGML_ASSERT(ctx->cpu_hidden_f16 && ctx->cpu_hidden_f32);
        ctx->ffn_local = model->ffn_local;
        ctx->ffn_mode = (ffn_mode_t)cparams.ffn_mode;
    }
    if (params.seed == LLAMA_DEFAULT_SEED) {"""
    )
    with open("llama.cpp-PoC/src/llama-context.cpp", "w") as f:
        f.write(cppcontent)

# update llama-model.cpp default params
with open("llama.cpp-PoC/src/llama-model.cpp", "r") as f:
    mcppcontent = f.read()

if "/* .ffn_mode */ FFN_GPU" not in mcppcontent:
    mcppcontent = mcppcontent.replace(
        "        /*.devices                      =*/ nullptr,",
        "        /*.ffn_mode                     =*/ 0,\n        /*.ffn_file                     =*/ nullptr,\n        /*.devices                      =*/ nullptr,"
    )
    # the loader needs to be updated to map ffn_local
    mcppcontent = mcppcontent.replace(
        "llama_model_base::llama_model_base(const struct llama_model_params & params) : llama_model(params), model(this), tn(model->arch)",
        "llama_model_base::llama_model_base(const struct llama_model_params & params) : llama_model(params), model(this), tn(model->arch)"
    )
    # Actually wait. Let's just put it in load_tensors.
    insert_ffn = """
    if (params.ffn_mode == FFN_LOCAL) {
        if (params.ffn_file) {
            model->ffn_local = ffn_mmap_from_full_gguf(params.ffn_file, ml.metadata, hparams.n_layer);
        } else {
            model->ffn_local = ffn_mmap_from_full_gguf(ml.file ? ml.file->path().c_str() : "", ml.metadata, hparams.n_layer);
        }
    }
"""
    mcppcontent = mcppcontent.replace("ml.init_mappings(true, use_mlock ? &pimpl->mlock_mmaps : nullptr);", "ml.init_mappings(true, use_mlock ? &pimpl->mlock_mmaps : nullptr);\n" + insert_ffn)

    with open("llama.cpp-PoC/src/llama-model.cpp", "w") as f:
        f.write(mcppcontent)
