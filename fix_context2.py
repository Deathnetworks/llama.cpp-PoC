with open("llama.cpp-PoC/src/llama-context.cpp", "r") as f:
    ccontent = f.read()

# fix llama_context_default_params
if "/* .ffn_mode */ 0," not in ccontent:
    ccontent = ccontent.replace(
        "        /*.seed                        =*/ LLAMA_DEFAULT_SEED,",
        "        /*.ffn_mode                    =*/ 0,\n        /*.seed                        =*/ LLAMA_DEFAULT_SEED,"
    )
    with open("llama.cpp-PoC/src/llama-context.cpp", "w") as f:
        f.write(ccontent)
