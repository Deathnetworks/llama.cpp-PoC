# The spec Phase 3 says we must intercept the loop inside decode_internal (which is now in build_arch_graph probably or somewhere else)
# Or wait, the spec says:
# "The only change to `llama_decode_internal` is replacing the unconditional FFN graph call with a dispatch:"
# "if (lctx.ffn_mode == FFN_LOCAL) { do_ffn_cpu_path(...) }"
# "The do_ffn_cpu_path function encapsulates §5.3. It must not touch the GGML graph builder — it operates on the already-computed tensor directly."
# So the spec assumes we compute the ATTENTION graph up to `cur`, then pull to CPU, do BLAS, then push to GPU.
# BUT in the new ggml_backend scheduler, we CANNOT stop graph computation in the middle to do our own things, the graph is built first, then executed.
# Wait! In the new llama.cpp, is there `llama_decode_internal`?
# The function `llama_decode` just creates a graph for all layers and schedules it.
# If the spec requires us to interrupt the execution, we have to rethink how to do it in the new ggml_backend.
# Actually, we CAN use a custom op `ggml_map_custom1_inplace` to let the CPU backend execute our FFN directly!
# Let's write the custom op!
pass
