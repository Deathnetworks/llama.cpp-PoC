import subprocess

print("Finding all build_ffn calls inside the per-layer loop.")

# We want to change the graph building to not build the ffn graph,
# but we can't `ggml_backend_graph_compute` in the middle of `build_arch_graph` because `build_arch_graph` is called ONCE to build the graph, and THEN `llama_context::graph_compute` runs the whole graph.
# Wait! In the new llama.cpp architecture, the entire computation is formulated as a single computation graph, which is then scheduled across backends by `ggml_backend_sched_graph_compute_async`.
# So you CANNOT interrupt the graph building with a compute!
# What we can do instead: we can add a custom `ggml_map_custom1` op to the graph!
# `ggml_map_custom1` executes a C callback during the graph evaluation.
