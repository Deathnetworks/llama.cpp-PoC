with open("llama.cpp-PoC/common/arg.cpp", "r") as f:
    content = f.read()

# Add local-ssd and local-ram to --split-mode choices
old_sm = """        {"-sm", "--split-mode"}, "{none,layer,row,tensor}",
        "how to split the model across multiple GPUs, one of:\\n"
        "- none: use one GPU only\\n"
        "- layer (default): split layers and KV across GPUs (pipelined)\\n"
        "- row: split weight across GPUs by rows (parallelized)\\n"
        "- tensor: split weights and KV across GPUs (parallelized, EXPERIMENTAL)","""

new_sm = """        {"-sm", "--split-mode"}, "{none,layer,row,tensor,local-ssd,local-ram}",
        "how to split the model across multiple GPUs, one of:\\n"
        "- none: use one GPU only\\n"
        "- layer (default): split layers and KV across GPUs (pipelined)\\n"
        "- row: split weight across GPUs by rows (parallelized)\\n"
        "- tensor: split weights and KV across GPUs (parallelized, EXPERIMENTAL)\\n"
        "- local-ssd: attention GPU, FFN mmap'd from NVMe\\n"
        "- local-ram: attention GPU, FFN locked into RAM","""

content = content.replace(old_sm, new_sm)

# Add --ffn-file
# We need to add it somewhere, e.g. after --split-mode
old_sm_cb = """        [](common_params & params, const std::string & value) {
            if (value == "none") {
                params.split_mode = LLAMA_SPLIT_MODE_NONE;
            } else if (value == "layer") {
                params.split_mode = LLAMA_SPLIT_MODE_LAYER;
            } else if (value == "row") {
                params.split_mode = LLAMA_SPLIT_MODE_ROW;
            } else if (value == "tensor") {
                params.split_mode = LLAMA_SPLIT_MODE_TENSOR;
            } else {
                throw std::invalid_argument("invalid value");
            }"""

new_sm_cb = """        [](common_params & params, const std::string & value) {
            if (value == "none") {
                params.split_mode = LLAMA_SPLIT_MODE_NONE;
            } else if (value == "layer") {
                params.split_mode = LLAMA_SPLIT_MODE_LAYER;
            } else if (value == "row") {
                params.split_mode = LLAMA_SPLIT_MODE_ROW;
            } else if (value == "tensor") {
                params.split_mode = LLAMA_SPLIT_MODE_TENSOR;
            } else if (value == "local-ssd" || value == "local-ram") {
                params.split_mode = LLAMA_SPLIT_MODE_LAYER; // we fallback to layer for GPU stuff
                params.ffn_mode = 1; // FFN_LOCAL
            } else {
                throw std::invalid_argument("invalid value");
            }"""
content = content.replace(old_sm_cb, new_sm_cb)

# add --ffn-file flag
ffn_flag = """    add_opt(common_arg(
        {"--ffn-file"}, "FNAME",
        "path to pre-split FFN file",
        [](common_params & params, const std::string & value) {
            params.ffn_file = value;
        }
    ));
"""

content = content.replace('    ).set_env("LLAMA_ARG_MAIN_GPU"));', '    ).set_env("LLAMA_ARG_MAIN_GPU"));\n' + ffn_flag)

with open("llama.cpp-PoC/common/arg.cpp", "w") as f:
    f.write(content)

with open("llama.cpp-PoC/common/common.h", "r") as f:
    hcontent = f.read()

if "ffn_file" not in hcontent:
    hcontent = hcontent.replace(
        "    int32_t main_gpu                = 0;",
        "    int32_t main_gpu                = 0;\n    int ffn_mode = 0;\n    std::string ffn_file;"
    )
    with open("llama.cpp-PoC/common/common.h", "w") as f:
        f.write(hcontent)

with open("llama.cpp-PoC/common/common.cpp", "r") as f:
    ccontent = f.read()

if "mparams.ffn_mode = params.ffn_mode;" not in ccontent:
    ccontent = ccontent.replace(
        "    mparams.main_gpu            = params.main_gpu;",
        "    mparams.main_gpu            = params.main_gpu;\n    mparams.ffn_mode = params.ffn_mode;\n    mparams.ffn_file = params.ffn_file.empty() ? nullptr : params.ffn_file.c_str();"
    )
    with open("llama.cpp-PoC/common/common.cpp", "w") as f:
        f.write(ccontent)
