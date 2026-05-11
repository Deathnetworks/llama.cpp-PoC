with open("llama.cpp-PoC/src/llama-cparams.h", "r") as f:
    hcontent = f.read()

# Add ffn_local
if "ffn_mmap_t* ffn_local" not in hcontent:
    hcontent = hcontent.replace(
        "    int32_t  n_threads_batch; // number of threads to use for batch processing",
        "    int32_t  n_threads_batch; // number of threads to use for batch processing\n\n    struct ffn_mmap_t* ffn_local = nullptr;\n    int ffn_mode = 0;"
    )
    with open("llama.cpp-PoC/src/llama-cparams.h", "w") as f:
        f.write(hcontent)

with open("llama.cpp-PoC/src/llama-cparams.cpp", "r") as f:
    cppcontent = f.read()

if "ffn_mode(params.ffn_mode)" not in cppcontent:
    cppcontent = cppcontent.replace(
        "    n_threads_batch(params.n_threads_batch),",
        "    n_threads_batch(params.n_threads_batch),\n    ffn_mode(params.ffn_mode),"
    )
    with open("llama.cpp-PoC/src/llama-cparams.cpp", "w") as f:
        f.write(cppcontent)
