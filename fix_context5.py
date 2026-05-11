with open("llama.cpp-PoC/include/llama.h", "r") as f:
    hcontent = f.read()

# Did I already add ffn_mode?
# Let's check `llama_model_params`.
print("Before:\n", [line for line in hcontent.split("\n") if "ffn" in line])
