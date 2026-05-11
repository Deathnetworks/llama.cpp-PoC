with open("llama.cpp-PoC/src/llama-graph.h", "r") as f:
    hcontent = f.read()

# Add a build_ffn_local signature
if "build_ffn_local" not in hcontent:
    hcontent = hcontent.replace(
        """    ggml_tensor * build_ffn(""",
        """    ggml_tensor * build_ffn_local(
         ggml_tensor * cur,
                 int   il) const;

    ggml_tensor * build_ffn("""
    )
    with open("llama.cpp-PoC/src/llama-graph.h", "w") as f:
        f.write(hcontent)
