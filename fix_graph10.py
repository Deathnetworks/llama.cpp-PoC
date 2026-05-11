with open("llama.cpp-PoC/src/llama-graph.cpp", "r") as f:
    ccontent = f.read()

# Since `llm_graph_context` has `llama_model & model` member! Let's check `llama-graph.h`.
