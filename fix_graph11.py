with open("llama.cpp-PoC/src/llama-graph.cpp", "r") as f:
    ccontent = f.read()

# We need to find where ffn_local is initialized.
# We don't have access to `llama_model` inside `llm_graph_context`?
# Wait! In `llama-graph.h` `llm_graph_context` constructor:
# llm_graph_context(const llama_model & model, ...);
pass
