with open("llama.cpp-PoC/common/common.h", "r") as f:
    hcontent = f.read()

# Add to common_params
if "int ffn_mode = 0;" not in hcontent:
    hcontent = hcontent.replace(
        "    int32_t n_predict             =    -1;",
        "    int ffn_mode                  =     0;\n    std::string ffn_file          =    \"\";\n    int32_t n_predict             =    -1;"
    )
    with open("llama.cpp-PoC/common/common.h", "w") as f:
        f.write(hcontent)
