// llama.cpp-PoC/src/llama-ffn-local.h
#pragma once
#include "ggml.h"
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <cstring>
#include <memory>

// ── Mode ─────────────────────────────────────────────────────────────────
enum ffn_mode_t {
    FFN_GPU      = 0,  // default — all weights on GPU (GPU-ONLY mode)
    FFN_LOCAL    = 1,  // FFN weights on CPU RAM, attention on GPU (FNN-RAM-CPU mode)
    FFN_ZERO_CPU = 2,  // FFN weights mmap'd from SSD, attention on GPU (FNN-ZERO-CPU mode)
};

// ── Per-layer weight pointers ────────────────────────────────────────────
// ALL pointer fields are raw mmap addresses into the zero-copy region.
// INVARIANT: never memcpy these before compute — use directly.  spec §13 inv 2
struct ffn_layer_ptrs_t {
    const void*  ffn_norm      = nullptr;  // raw mmap ptr — never memcpy
    enum ggml_type ffn_norm_type = GGML_TYPE_F32;
    const void*  gate          = nullptr;  // raw mmap ptr — never memcpy
    enum ggml_type gate_type     = GGML_TYPE_F32;
    const void*  up            = nullptr;  // raw mmap ptr — never memcpy
    enum ggml_type up_type       = GGML_TYPE_F32;
    const void*  down          = nullptr;  // raw mmap ptr — never memcpy
    enum ggml_type down_type     = GGML_TYPE_F32;
    uint32_t     n_ffn         = 0;
    uint32_t     n_embd        = 0;
};

// ── mmap file handle ─────────────────────────────────────────────────────
struct llama_file;
struct llama_mmap;

struct ffn_mmap_t {
    std::unique_ptr<llama_file> file;
    std::unique_ptr<llama_mmap> mmap;
    void*  base      = nullptr;
    size_t file_size = 0;
    std::vector<ffn_layer_ptrs_t> layers;

    uint32_t n_embd         = 0;
    float    f_norm_rms_eps = 1e-6f;

    bool        is_split_file = false;
    std::string source_sha256;
    uint32_t    layer_first   = 0;
    uint32_t    layer_last    = 0;
};

// ── Split mode ─────────────────────────────────────────────────────────────
// Controls which tensor types go to CPU vs GPU in FFN_LOCAL mode.
// FFN_ALWAYS: FFN tensors always go to CPU (required for VRAM reduction)
// ATTN_ALWAYS: Attention tensors always stay on GPU (required for performance)
// OTHER_GPU: Other tensors (SSM, etc.) stay on GPU (default)
// OTHER_CPU: Other tensors also go to CPU (saves more VRAM, may be slower)
enum split_other_t {
    SPLIT_OTHER_GPU = 0,       // default: SSM/other tensors on GPU
    SPLIT_OTHER_CPU = 1,       // SSM/other tensors also on CPU (excludes embedding/output)
    SPLIT_OTHER_ALL_CPU = 2,   // ALL non-attention tensors on CPU (including embedding/output)
};

// ── Routing predicate ─────────────────────────────────────────────────────
// Single source of truth for FFN tensor routing.  spec §3
// Must match FFN_PATTERNS in tests/unit/test_routing.py exactly.
inline bool is_ffn_tensor(const char* name) {
    if (name == nullptr) return false;
    // MoE router stays on GPU — exclude from CPU offload
    if (strstr(name, "ffn_gate_inp") != nullptr) return false;
    return strstr(name, "ffn_norm")      != nullptr ||
           strstr(name, "ffn_gate")      != nullptr ||
           strstr(name, "ffn_up")        != nullptr ||
           strstr(name, "ffn_down")      != nullptr ||
           strstr(name, "ffn_gate_exps") != nullptr ||
           strstr(name, "ffn_up_exps")   != nullptr ||
           strstr(name, "ffn_down_exps") != nullptr;
}

// Check if a tensor is an attention tensor (should stay on GPU)
inline bool is_attn_tensor(const char* name) {
    if (name == nullptr) return false;
    return strstr(name, "attn_q")              != nullptr ||
           strstr(name, "attn_k")              != nullptr ||
           strstr(name, "attn_v")              != nullptr ||
           strstr(name, "attn_output")         != nullptr ||
           strstr(name, "attn_o")              != nullptr ||
           strstr(name, "attn_qkv")            != nullptr ||
           strstr(name, "attn_norm")           != nullptr ||
           strstr(name, "attn_gate")           != nullptr ||
           strstr(name, "post_attention_norm") != nullptr ||
           strstr(name, "rope_freqs")          != nullptr ||
           strstr(name, "attn_k_norm")         != nullptr ||
           strstr(name, "attn_v_norm")         != nullptr;
}

// Check if a tensor is an "other" tensor (SSM, etc.) — configurable split
inline bool is_other_tensor(const char* name) {
    if (name == nullptr) return false;
    // If it's not FFN and not attention, it's "other"
    return !is_ffn_tensor(name) && !is_attn_tensor(name);
}

// Check if a tensor is embedding or output (should stay on GPU for performance)
inline bool is_embedding_or_output(const char* name) {
    if (name == nullptr) return false;
    return strstr(name, "token_embd") != nullptr ||
           strstr(name, "output_norm") != nullptr ||
           strstr(name, "output") != nullptr ||
           strstr(name, "lm_head") != nullptr ||
           strstr(name, "classifier") != nullptr;
}

// Forward declarations for loader functions (defined in llama.cpp or llama-ffn-local.cpp)
struct gguf_context;

// ── Loader ────────────────────────────────────────────────────────────────
// Open path via mmap and populate per-layer pointer table.
ffn_mmap_t* ffn_mmap_from_full_gguf(const char* path,
                                     const struct gguf_context* gctx,
                                     int n_layers);
void ffn_mmap_prefetch(const ffn_mmap_t* ffn, int il);
void ffn_mmap_free(ffn_mmap_t* ffn);

// ── Compute ───────────────────────────────────────────────────────────────
void llm_compute_ffn_cpu(const ffn_mmap_t* ffn, int layer,
                          float* hidden, int n_tokens, int n_embd,
                          bool skip_norm = false, bool do_add = true, int n_threads = 1);

struct ffn_local_userdata {
    const ffn_mmap_t * ffn;
    int layer;
};

void ffn_local_callback(struct ggml_tensor * dst , const struct ggml_tensor * a, int ith, int nth, void * userdata);

// ── Dynamic swap ──────────────────────────────────────────────────────────
// NOT thread-safe. Must only be called between token decodes.  spec §9
// Phase 1-3: stub implementation (logs warning, returns immediately).
// Phase 4: full implementation.
struct llama_context;
void llama_swap_ffn(struct llama_context* ctx,
                    int layer_first, int layer_last,
                    const char* new_ffn_path);
