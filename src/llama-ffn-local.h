// llama.cpp-PoC/src/llama-ffn-local.h
#pragma once
#include "ggml.h"
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <cstring>
#include <memory>
#include <unordered_map>

// ── Weight offset map for streaming mode ──────────────────────────────────
// Maps tensor name → (file_index, file_offset, size) for FFN weights
// Used by both llama_model (population) and llama_context (consumption)
struct ffn_weight_offset {
    uint16_t file_idx;
    uint64_t file_off;
    size_t   size;
};

// ── Mode ─────────────────────────────────────────────────────────────────
enum ffn_mode_t {
    FFN_GPU      = 0,  // default — all weights on GPU (GPU-ONLY mode)
    FFN_LOCAL    = 1,  // FFN weights on CPU RAM, attention on GPU (FNN-RAM-CPU mode)
    FFN_ZERO_CPU = 2,  // FFN weights mmap'd from SSD, attention on GPU (FNN-ZERO-CPU mode)
};

// ── Per-layer weight storage ──────────────────────────────────────────────
// Supports two backends:
//   1. MMAP (legacy): raw pointers into mmap'd GGUF file
//   2. ASYNC_DBL_BUFFER (new): double-buffered async I/O with O_DIRECT
//      - Two pre-allocated buffers ping-pong between compute and load
//      - Next layer loads via io_uring/Overlapped while current layer computes
//      - Dequantize Q4→F32 on-the-fly during matmul (no separate dequant step)
//      - RAM usage = 2 × max_layer_size (typically 300-600MB total)

// Async buffer alignment requirement for O_DIRECT
constexpr size_t ASYNC_IO_ALIGN = 4096;

// Per-layer weight info — stores either mmap pointers or file offset/size for async read
struct ffn_layer_ptrs_t {
    // MMAP path (legacy)
    const void*  ffn_norm      = nullptr;
    enum ggml_type ffn_norm_type = GGML_TYPE_F32;
    const void*  gate          = nullptr;
    enum ggml_type gate_type     = GGML_TYPE_F32;
    const void*  up            = nullptr;
    enum ggml_type up_type       = GGML_TYPE_F32;
    const void*  down          = nullptr;
    enum ggml_type down_type     = GGML_TYPE_F32;
    uint32_t     n_ffn         = 0;
    uint32_t     n_embd        = 0;

    // Async double-buffer path — file offsets and sizes for O_DIRECT read
    uint64_t     file_off_ffn_norm = 0;  // byte offset in GGUF file
    uint64_t     file_off_gate     = 0;
    uint64_t     file_off_up       = 0;
    uint64_t     file_off_down     = 0;
    size_t       size_ffn_norm     = 0;  // byte size of each tensor
    size_t       size_gate         = 0;
    size_t       size_up           = 0;
    size_t       size_down         = 0;
    bool         async_valid       = false;  // true if file offsets are set
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

// Check if a tensor belongs to an MTP (Multi-Token Predictor) layer.
// MTP layers have "nextn" tensors.
inline bool is_mtp_tensor(const char* name) {
    if (name == nullptr) return false;
    return strstr(name, "nextn.") != nullptr;
}

// Check if an FFN tensor belongs to an MTP predictor layer.
// MTP layers are identified by having "nextn" tensors.
// This function checks if the given FFN tensor is in the last layer (where MTP predictor lives).
inline bool is_ffn_in_mtp_layer(const char* name, int n_layer) {
    if (name == nullptr || !is_ffn_tensor(name)) return false;
    char layer_prefix[32];
    snprintf(layer_prefix, sizeof(layer_prefix), "blk.%d.", n_layer - 1);
    return strstr(name, layer_prefix) != nullptr;
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

// ── Async Double-Buffer Engine ─────────────────────────────────────────────
// Bypasses kernel page cache entirely using O_DIRECT + io_uring (Linux)
// or Overlapped I/O (Windows). Only 2× max_layer_size RAM is needed.
//
// Flow:
//   1. Before layer N FFN: async-read layer N weights into active buffer
//   2. Wait for read to complete (should be ready if pipeline is full)
//   3. Point tensor data pointers into active buffer
//   4. Compute FFN on CPU (dequantize Q4→F32 on-the-fly during matmul)
//   5. Submit async read for layer N+1 into prefetch buffer
//   6. Swap buffers (ping-pong)

struct ffn_async_buffer {
    int          fd              = -1;      // GGUF file descriptor (O_DIRECT)
    uint64_t     data_offset     = 0;       // byte offset of tensor data in file
    size_t       data_size       = 0;       // total bytes for all FFN tensors in layer
    void*        buf[2]          = {nullptr, nullptr};  // two aligned buffers
    int          active_idx      = 0;       // which buffer is currently active
    int          prefetch_idx    = 1;       // which buffer is being prefetched
    bool         initialized     = false;

    // io_uring state (Linux)
#if !defined(_WIN32) && defined(__linux__)
    struct io_uring* ring        = nullptr;
    int              req_layer   = -1;      // layer currently being prefetched (-1 = none)
#endif

    // Overlapped I/O state (Windows)
#if defined(_WIN32)
    HANDLE       h_event         = nullptr; // completion event
    OVERLAPPED   overlapped      = {};      // overlapped structure
    int          req_layer       = -1;      // layer currently being prefetched
#endif
};

// Initialize async buffer engine
// Returns true on success
bool ffn_async_init(ffn_async_buffer* ab, const char* path,
                    const ffn_mmap_t* ffn, int n_layers);

// Shutdown and free buffers
void ffn_async_free(ffn_async_buffer* ab);

// Submit async read for layer N into the prefetch buffer
// Returns immediately (non-blocking)
bool ffn_async_prefetch(ffn_async_buffer* ab, int layer);

// Wait for the prefetch to complete and swap buffers
// After this, the active buffer contains layer N's weights
bool ffn_async_swap(ffn_async_buffer* ab, int layer);

// Get pointer to active buffer data for a specific tensor
const void* ffn_async_get_ptr(const ffn_async_buffer* ab, int layer,
                               const ffn_layer_ptrs_t* lp,
                               const char* tensor_name);

// Load a layer's weights into the active buffer (synchronous)
// This blocks until the read completes
bool ffn_async_load_layer(ffn_async_buffer* ab, int layer, const ffn_mmap_t* ffn);

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
