with open("llama.cpp-PoC/src/llama-context.cpp", "r") as f:
    ccontent = f.read()

# Add ggml_backend_tensor_get/set and dispatch
if "if (lctx.ffn_mode == FFN_LOCAL" not in ccontent:
    # Actually, we need to modify decode internal, where is `llm_build_ffn`?
    # It's in `llama-graph.cpp` ! Wait, no, `llama_decode_internal` is somewhere?
    pass

import re
out = subprocess.check_output("grep -n 'build_ffn' llama.cpp-PoC/src/models/* || true", shell=True).decode()
print("Found build_ffn in models:\n", out[:500])
