with open("llama.cpp-PoC/src/llama-context.cpp", "r") as f:
    ccontent = f.read()

ccontent = ccontent.replace(
"""llama_context_params llama_context_default_params() {
    llama_context_params result = {
        /*.n_ctx                       =*/ 512,""",
"""llama_context_params llama_context_default_params() {
    llama_context_params result = {
        /*.ffn_mode                    =*/ 0,
        /*.n_ctx                       =*/ 512,""")

with open("llama.cpp-PoC/src/llama-context.cpp", "w") as f:
    f.write(ccontent)
