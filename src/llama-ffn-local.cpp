#include "llama-ffn-local.h"
#include "llama-mmap.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"
#include <cmath>
#include <cassert>
#include <vector>
#include <cstdio>
#include <cstring>

#if defined(__APPLE__) || defined(__FreeBSD__) || defined(__OpenBSD__) || defined(__NetBSD__) || defined(__DragonFly__)
#include <sys/mman.h>
#elif defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
// madvise() requires _GNU_SOURCE on Linux
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <sys/mman.h>
#endif

// Note: BLAS is not required. The FFN compute uses manual dot products
// for portability. BLAS includes are kept for future optimization.
#if 0  // Disable BLAS include — not needed for manual dot product path
extern "C" {
#if defined(GGML_USE_SYCL) || defined(GGML_BLAS_USE_MKL)
#   include <mkl.h>
#elif defined(GGML_BLAS_USE_ACCELERATE)
#   include <Accelerate/Accelerate.h>
#elif defined(GGML_BLAS_USE_BLIS)
#   include <blis.h>
#elif defined(GGML_BLAS_USE_NVPL)
#   include <nvpl_blas.h>
#else
#   include <cblas.h>
#endif
}
#endif

// ── Helpers ───────────────────────────────────────────────────────────────

static void rms_norm(const float* x, const float* w, float* out, int n, float eps) {
    float ss = 0.0f;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    const float scale = 1.0f / sqrtf(ss / n + eps);
    for (int i = 0; i < n; i++) out[i] = x[i] * scale * w[i];
}

static void silu_inplace(float* x, int n) {
    for (int i = 0; i < n; i++) x[i] = x[i] / (1.0f + expf(-x[i]));
}

// ── Compute ───────────────────────────────────────────────────────────────

void llm_compute_ffn_cpu(const ffn_mmap_t* ffn, int layer,
                          float* hidden, int n_tokens, int n_embd,
                          bool skip_norm, bool do_add, int n_threads) {
    (void)n_threads; 
    assert(layer >= 0 && layer < (int)ffn->layers.size());
    const auto& lp = ffn->layers[layer];
    if (!lp.gate || !lp.up || !lp.down) return;

    const int nf = lp.n_ffn;

    // Row dequant helper: dequantize one row of a quantized weight matrix.
    // row_ptr: pointer to the start of the row in the mmap'd quantized data.
    // type:    ggml_type of the weight (e.g. GGML_TYPE_Q4_K).
    // n_cols:  number of output floats.
    // out:     pre-allocated f32 buffer of size n_cols.
    auto dequant_row = [](const void* row_ptr, enum ggml_type type, int n_cols, float* out) {
        if (type == GGML_TYPE_F32) {
            std::memcpy(out, row_ptr, n_cols * sizeof(float));
            return;
        }
        const auto* tt = ggml_get_type_traits(type);
        if (tt && tt->to_float) {
            tt->to_float(row_ptr, out, n_cols);
        }
    };

    // Stride of one quantized row in bytes
    // ggml stores blocks: type_size bytes per block_size elements.
    auto row_bytes = [](enum ggml_type type, int n_cols) -> size_t {
        const auto* tt = ggml_get_type_traits(type);
        if (!tt || tt->blck_size == 0) return (size_t)n_cols * sizeof(float);
        size_t n_blocks = ((size_t)n_cols + tt->blck_size - 1) / tt->blck_size;
        return n_blocks * tt->type_size;
    };

    std::vector<float> normed(n_embd);
    std::vector<float> gate_buf(nf);
    std::vector<float> up_buf(nf);
    std::vector<float> wrow((std::max)(n_embd, nf)); // scratch for one dequantized row

    const size_t gate_row_bytes = row_bytes(lp.gate_type, n_embd);
    const size_t up_row_bytes   = row_bytes(lp.up_type,   n_embd);
    const size_t down_row_bytes = row_bytes(lp.down_type, nf);

    for (int t = 0; t < n_tokens; t++) {
        float* h = hidden + (size_t)t * n_embd;
        
        const float* normed_ptr = h;
        if (!skip_norm && lp.ffn_norm) {
            // ffn_norm weights are always f32 in GGUF
            rms_norm(h, (const float*)lp.ffn_norm, normed.data(), n_embd, ffn->f_norm_rms_eps);
            normed_ptr = normed.data();
        }

        // gate[i] = dot(W_gate[i], normed)  for i in [0, nf)
        for (int i = 0; i < nf; i++) {
            const void* row = (const uint8_t*)lp.gate + (size_t)i * gate_row_bytes;
            dequant_row(row, lp.gate_type, n_embd, wrow.data());
            float acc = 0.0f;
            for (int j = 0; j < n_embd; j++) acc += wrow[j] * normed_ptr[j];
            gate_buf[i] = acc;
        }
        silu_inplace(gate_buf.data(), nf);

        // up[i] = dot(W_up[i], normed)
        for (int i = 0; i < nf; i++) {
            const void* row = (const uint8_t*)lp.up + (size_t)i * up_row_bytes;
            dequant_row(row, lp.up_type, n_embd, wrow.data());
            float acc = 0.0f;
            for (int j = 0; j < n_embd; j++) acc += wrow[j] * normed_ptr[j];
            up_buf[i] = acc;
        }

        // Hadamard product: gate *= up
        for (int i = 0; i < nf; i++) gate_buf[i] *= up_buf[i];

        // down[i] = dot(W_down[i], gate*up) then add into residual h (beta=1.0)
        // ⚠ Must ADD into h (not overwrite) — this is the residual. spec §6.2, §13 invariant 4
        for (int i = 0; i < n_embd; i++) {
            const void* row = (const uint8_t*)lp.down + (size_t)i * down_row_bytes;
            dequant_row(row, lp.down_type, nf, wrow.data());
            float acc = 0.0f;
            for (int j = 0; j < nf; j++) acc += wrow[j] * gate_buf[j];
            h[i] += acc; // residual add (beta=1.0 equivalent)
        }
    }
}


void ffn_local_callback(struct ggml_tensor * dst , const struct ggml_tensor * a, int ith, int nth, void * userdata) {
    GGML_UNUSED(ith);
    const ffn_local_userdata * ud = (const ffn_local_userdata *) userdata;
    const ffn_mmap_t * ffn = ud->ffn;
    int layer = ud->layer;

    int n_embd   = (int) ffn->layers[layer].n_embd;
    if (n_embd == 0) return;

    int n_elements = (int) ggml_nelements(a);
    int n_tokens   = n_elements / n_embd;
    if (n_tokens <= 0) return;

    // Dequantize / convert input tensor to f32.
    std::vector<float> h_cpu(n_tokens * n_embd);
    if (a->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(a, h_cpu.data(), 0, ggml_nbytes(a));
    } else {
        std::vector<uint8_t> raw(ggml_nbytes(a));
        ggml_backend_tensor_get(a, raw.data(), 0, ggml_nbytes(a));
        const auto * tt = ggml_get_type_traits(a->type);
        if (tt && tt->to_float) {
            tt->to_float(raw.data(), h_cpu.data(), n_elements);
        } else {
            fprintf(stderr, "FFN_LOCAL cb: unsupported type %d\n", (int)a->type);
            return; // unsupported type
        }
    }

    llm_compute_ffn_cpu(ffn, layer, h_cpu.data(), n_tokens, n_embd, false, true, nth);

    // Write result back — may need to convert back to f16 if dst is f16
    if (dst->type == GGML_TYPE_F32) {
        ggml_backend_tensor_set(dst, h_cpu.data(), 0, ggml_nbytes(dst));
    } else {
        std::vector<uint8_t> out_raw(ggml_nbytes(dst));
        const auto * tt = ggml_get_type_traits(dst->type);
        if (tt && tt->from_float_ref) {
            tt->from_float_ref(h_cpu.data(), out_raw.data(), n_elements);
        }
        ggml_backend_tensor_set(dst, out_raw.data(), 0, ggml_nbytes(dst));
    }
}


// ── Loader ────────────────────────────────────────────────────────────────

ffn_mmap_t* ffn_mmap_from_full_gguf(const char* path, const struct gguf_context* gctx, int n_layers) {
    auto ffn = new ffn_mmap_t();
    try {
        ffn->file = std::make_unique<llama_file>(path, "rb");
        // spec §5.1: MAP_SHARED | MAP_NORESERVE
        ffn->mmap = std::make_unique<llama_mmap>(ffn->file.get(), 0, false); 
        ffn->base = ffn->mmap->addr();
        ffn->file_size = ffn->mmap->size();
    } catch (...) {
        delete ffn;
        return nullptr;
    }

    size_t data_off = gguf_get_data_offset(gctx);
    
    // Get hyper-params — try multiple architecture key prefixes
    uint32_t n_embd = 0;
    uint32_t n_ffn  = 0;
    float    eps    = 1e-6f;

    int64_t k_embd = gguf_find_key(gctx, "llama.embedding_length");
    if (k_embd == -1) k_embd = gguf_find_key(gctx, "qwen2.embedding_length");
    if (k_embd == -1) k_embd = gguf_find_key(gctx, "qwen3.embedding_length");
    if (k_embd != -1) n_embd = gguf_get_val_u32(gctx, k_embd);

    int64_t k_ffn = gguf_find_key(gctx, "llama.feed_forward_length");
    if (k_ffn == -1) k_ffn = gguf_find_key(gctx, "qwen2.feed_forward_length");
    if (k_ffn == -1) k_ffn = gguf_find_key(gctx, "qwen3.feed_forward_length");
    if (k_ffn != -1) n_ffn = gguf_get_val_u32(gctx, k_ffn);

    int64_t k_eps = gguf_find_key(gctx, "llama.attention.layer_norm_rms_epsilon");
    if (k_eps == -1) k_eps = gguf_find_key(gctx, "qwen2.attention.layer_norm_rms_epsilon");
    if (k_eps == -1) k_eps = gguf_find_key(gctx, "qwen3.attention.layer_norm_rms_epsilon");
    if (k_eps != -1) eps = gguf_get_val_f32(gctx, k_eps);

    ffn->n_embd = n_embd;
    ffn->f_norm_rms_eps = eps;
    ffn->layers.resize(n_layers);
    
    int64_t n_tensors = gguf_get_n_tensors(gctx);
    for (int64_t i = 0; i < n_tensors; i++) {
        const char* name = gguf_get_tensor_name(gctx, i);
        if (!is_ffn_tensor(name)) continue;
        
        size_t off = gguf_get_tensor_offset(gctx, i);
        enum ggml_type type = gguf_get_tensor_type(gctx, i);
        const void* ptr = (const void*)((const uint8_t*)ffn->base + data_off + off);
        
        int layer = -1;
        char tt[64] = {};
        if (sscanf(name, "blk.%d.%63s", &layer, tt) != 2) continue;
        if (layer < 0 || layer >= n_layers) continue;
        
        auto& lp = ffn->layers[layer];
        lp.n_embd = n_embd;
        lp.n_ffn  = n_ffn;
        
        if      (strstr(tt, "ffn_norm")) { lp.ffn_norm = ptr; lp.ffn_norm_type = type; }
        else if (strstr(tt, "ffn_gate")) { lp.gate     = ptr; lp.gate_type     = type; }
        else if (strstr(tt, "ffn_up"))   { lp.up       = ptr; lp.up_type       = type; }
        else if (strstr(tt, "ffn_down")) { lp.down     = ptr; lp.down_type     = type; }
    }
    
    return ffn;
}

void ffn_mmap_prefetch(const ffn_mmap_t* ffn, int il) {
    if (il + 1 >= (int)ffn->layers.size()) return;
    const auto& lp = ffn->layers[il + 1];
    if (!lp.gate || !lp.up) return;
    
    size_t nb = (size_t)lp.n_ffn * lp.n_embd * sizeof(float);
    
#ifdef _WIN32
    // Optimization only, skipping for Windows PoC
    (void)ffn; (void)nb;
#else
    madvise((void*)lp.gate, nb, MADV_SEQUENTIAL);
    madvise((void*)lp.up,   nb, MADV_SEQUENTIAL);
#endif
}

void ffn_mmap_free(ffn_mmap_t* ffn) {
    delete ffn;
}

// ── Dynamic swap ──────────────────────────────────────────────────────────

void llama_swap_ffn(struct llama_context* /*ctx*/,
                    int layer_first, int layer_last,
                    const char* new_ffn_path) {
    // Stub implementation for Phase 1-3.
    // Full body in Phase 4.
    (void)layer_first; (void)layer_last; (void)new_ffn_path;
}
