# Now we need to modify the models to use `build_ffn_local` if `ffn_mode == FFN_LOCAL`.
# e.g., in `models/llama.cpp`

with open("llama.cpp-PoC/src/models/llama.cpp", "r") as f:
    ccontent = f.read()

old = """            cur = build_ffn(cur,
                    model.layers[il].ffn_up,   model.layers[il].ffn_up_b,   model.layers[il].ffn_up_s,
                    model.layers[il].ffn_gate, model.layers[il].ffn_gate_b, model.layers[il].ffn_gate_s,
                    model.layers[il].ffn_down, model.layers[il].ffn_down_b, model.layers[il].ffn_down_s,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, il);"""

new = """            if (cparams.ffn_mode == FFN_LOCAL) {
                cur = build_ffn_local(cur, il);
            } else {
                cur = build_ffn(cur,
                        model.layers[il].ffn_up,   model.layers[il].ffn_up_b,   model.layers[il].ffn_up_s,
                        model.layers[il].ffn_gate, model.layers[il].ffn_gate_b, model.layers[il].ffn_gate_s,
                        model.layers[il].ffn_down, model.layers[il].ffn_down_b, model.layers[il].ffn_down_s,
                        NULL,
                        LLM_FFN_SILU, LLM_FFN_PAR, il);
            }"""

ccontent = ccontent.replace(old, new)

with open("llama.cpp-PoC/src/models/llama.cpp", "w") as f:
    f.write(ccontent)

# Also gemma2, gemma3, etc. but let's just stick to llama for the moment, or wait, gemma2 is what our test uses!
with open("llama.cpp-PoC/src/models/gemma2.cpp", "r") as f:
    ccontent = f.read()

old = """            cur = build_ffn(cur,
                    model.layers[il].ffn_up,   nullptr, nullptr,
                    model.layers[il].ffn_gate, nullptr, nullptr,
                    model.layers[il].ffn_down, nullptr, nullptr,
                    nullptr,
                    LLM_FFN_GELU, LLM_FFN_PAR, il);"""

new = """            if (cparams.ffn_mode == FFN_LOCAL) {
                cur = build_ffn_local(cur, il);
            } else {
                cur = build_ffn(cur,
                        model.layers[il].ffn_up,   nullptr, nullptr,
                        model.layers[il].ffn_gate, nullptr, nullptr,
                        model.layers[il].ffn_down, nullptr, nullptr,
                        nullptr,
                        LLM_FFN_GELU, LLM_FFN_PAR, il);
            }"""
ccontent = ccontent.replace(old, new)

with open("llama.cpp-PoC/src/models/gemma2.cpp", "w") as f:
    f.write(ccontent)
