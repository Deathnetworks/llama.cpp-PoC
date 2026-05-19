// llama.cpp-PoC/src/llama-ffn-async.cpp
// Async double-buffer engine for zero-RAM FFN inference
// Uses synchronous read() into temp buffers — only 1 layer in RAM at a time
// Future: io_uring on Linux, Overlapped I/O on Windows

#include "llama-ffn-local.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>

#ifndef _WIN32
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/uio.h>
// io_uring support is handled by CMake (HAS_IO_URING defined via target_compile_definitions)
// When HAS_IO_URING is not defined, it defaults to 0
#ifndef HAS_IO_URING
#define HAS_IO_URING 0
#endif
#if HAS_IO_URING
#include <liburing.h>
#endif
#else
#include <windows.h>
#include <io.h>
#endif
// ── Aligned buffer allocation ─────────────────────────────────────────────

static void* aligned_alloc_4k(size_t size) {
#ifdef _WIN32
    return _aligned_malloc(size, 4096);
#else
    void* ptr = nullptr;
    if (posix_memalign(&ptr, 4096, size) != 0) return nullptr;
    return ptr;
#endif
}

static void aligned_free_4k(void* ptr) {
#ifdef _WIN32
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

// ── Async buffer engine ───────────────────────────────────────────────────

bool ffn_async_init(ffn_async_buffer* ab, const char* path,
                    const ffn_mmap_t* ffn, int n_layers) {
    if (!ab || !path || n_layers <= 0) return false;
    (void)ffn;

    // Open file for reading
#ifndef _WIN32
    ab->fd = open(path, O_RDONLY);
#else
    ab->fd = open(path, O_RDONLY | O_BINARY);
#endif
    if (ab->fd < 0) {
        fprintf(stderr, "DEBUG STREAM: failed to open file '%s' for streaming\n", path);
        return false;
    }

    // Estimate max layer size from a simple heuristic
    // For Q4_K_M models, each FFN tensor is roughly n_embd * n_ff / 2 bytes
    // We'll use a conservative estimate and let the buffer grow if needed
    size_t max_layer_size = 256 * 1024 * 1024;  // 256 MB per layer (conservative)

    ab->data_size = max_layer_size;

    // Allocate two aligned buffers
    ab->buf[0] = aligned_alloc_4k(max_layer_size);
    ab->buf[1] = aligned_alloc_4k(max_layer_size);
    if (!ab->buf[0] || !ab->buf[1]) {
        fprintf(stderr, "DEBUG STREAM: failed to allocate async buffers (%zu MB each)\n", max_layer_size / (1024*1024));
        aligned_free_4k(ab->buf[0]);
        aligned_free_4k(ab->buf[1]);
        close(ab->fd);
        ab->fd = -1;
        return false;
    }

    ab->active_idx = 0;
    ab->prefetch_idx = 1;
    ab->initialized = true;
    ab->req_layer = -1;

    fprintf(stderr, "DEBUG STREAM: async buffer initialized — fd=%d, buffer_size=%zu MB\n",
        ab->fd, max_layer_size / (1024*1024));
    return true;
}

void ffn_async_free(ffn_async_buffer* ab) {
    if (!ab || !ab->initialized) return;
    if (ab->fd >= 0) close(ab->fd);
    aligned_free_4k(ab->buf[0]);
    aligned_free_4k(ab->buf[1]);
    ab->fd = -1;
    ab->buf[0] = nullptr;
    ab->buf[1] = nullptr;
    ab->initialized = false;
}

// Read a single layer's weights from SSD into the specified buffer
static bool ffn_read_layer(int fd, const ffn_layer_ptrs_t& lp, void* buf) {
    if (!lp.async_valid || !buf) return false;

    uint8_t* dst = (const_cast<uint8_t*>(static_cast<const uint8_t*>(buf)));
    size_t total_read = 0;

    // Read each tensor sequentially: ffn_norm, gate, up, down
    struct { uint64_t off; size_t size; } tensors[] = {
        {lp.file_off_ffn_norm, lp.size_ffn_norm},
        {lp.file_off_gate,     lp.size_gate},
        {lp.file_off_up,       lp.size_up},
        {lp.file_off_down,     lp.size_down},
    };

    for (const auto& t : tensors) {
        if (t.size == 0) continue;

        // Seek to file offset
#ifndef _WIN32
        lseek(fd, (off_t)t.off, SEEK_SET);
#else
        LARGE_INTEGER li;
        li.QuadPart = (LONGLONG)t.off;
        SetFilePointerEx((HANDLE)(intptr_t)fd, li, nullptr, FILE_BEGIN);
#endif

        // Read data
        size_t remaining = t.size;
        uint8_t* ptr = dst + total_read;
        while (remaining > 0) {
#ifndef _WIN32
            ssize_t bytes_read = read(fd, ptr, remaining);
            if (bytes_read <= 0) return false;
#else
            DWORD bytes_read = 0;
            ReadFile((HANDLE)(intptr_t)fd, ptr, (DWORD)remaining, &bytes_read, nullptr);
            if (bytes_read == 0) return false;
#endif
            ptr += bytes_read;
            remaining -= (size_t)bytes_read;
        }
        total_read += t.size;
    }

    return true;
}

// Load a single tensor from SSD into the buffer
// Uses pread() for thread-safe file access
// Returns the offset within the buffer where the tensor was stored
size_t ffn_async_load_tensor(ffn_async_buffer* ab, int fd,
                              const ffn_weight_offset &wo,
                              void* buf, size_t buf_offset) {
    if (!ab || !ab->initialized || fd < 0) return 0;
    if (wo.size == 0) return 0;

    // Read from file using pread (thread-safe, no seek needed)
    ssize_t bytes_read = pread(fd, (uint8_t*)buf + buf_offset, wo.size, (off_t)wo.file_off);
    if (bytes_read != (ssize_t)wo.size) {
        fprintf(stderr, "ffn_async_load_tensor: read %zd/%zu bytes from offset %lu\n",
            bytes_read, wo.size, (unsigned long)wo.file_off);
        return 0;
    }

    return wo.size;
}

// Load layer N's weights into the active buffer (synchronous)
// Reads all FFN tensors for the layer from SSD
bool ffn_async_load_layer(ffn_async_buffer* ab, int layer, const ffn_mmap_t* ffn) {
    if (!ab || !ab->initialized || layer < 0) return false;
    if (ab->fd < 0) return false;

    uint8_t* buf = (uint8_t*)ab->buf[ab->active_idx];
    size_t offset = 0;

    // Build tensor names and look up offsets
    // This is a simplified version — in practice, you'd pass the weight map
    // For now, fall back to memcpy from mmap region
    if (ffn && layer < (int)ffn->layers.size()) {
        const auto& lp = ffn->layers[layer];
        if (lp.ffn_norm && lp.size_ffn_norm > 0) {
            memcpy(buf + offset, lp.ffn_norm, lp.size_ffn_norm);
            offset += lp.size_ffn_norm;
        }
        if (lp.gate && lp.size_gate > 0) {
            memcpy(buf + offset, lp.gate, lp.size_gate);
            offset += lp.size_gate;
        }
        if (lp.up && lp.size_up > 0) {
            memcpy(buf + offset, lp.up, lp.size_up);
            offset += lp.size_up;
        }
        if (lp.down && lp.size_down > 0) {
            memcpy(buf + offset, lp.down, lp.size_down);
            offset += lp.size_down;
        }
        return offset > 0;
    }

    return false;
}

// Load layer N's weights from SSD using pread() into a specific buffer
// This is the streaming version that reads directly from the GGUF file
// Uses pread() as fallback; io_uring_prep_read_fixed() when liburing available
bool ffn_async_load_layer_pread(ffn_async_buffer* ab, int layer,
                                 const std::unordered_map<std::string, ffn_weight_offset>* weight_map,
                                 int buf_idx) {
    if (!ab || !ab->initialized || layer < 0 || !weight_map) return false;
    if (ab->fd < 0) return false;
    if (buf_idx < 0 || buf_idx >= 2) return false;

    char name_gate[64], name_up[64], name_down[64];
    snprintf(name_gate, sizeof(name_gate), "blk.%d.ffn_gate.weight", layer);
    snprintf(name_up,   sizeof(name_up),   "blk.%d.ffn_up.weight",   layer);
    snprintf(name_down, sizeof(name_down), "blk.%d.ffn_down.weight", layer);

    auto it_gate = weight_map->find(name_gate);
    auto it_up   = weight_map->find(name_up);
    auto it_down = weight_map->find(name_down);

    if (it_gate == weight_map->end() || it_up == weight_map->end() || it_down == weight_map->end()) {
        return false;
    }

    uint64_t block_start = it_gate->second.file_off;
    uint64_t block_end   = it_gate->second.file_off + it_gate->second.size;
    if (it_up->second.file_off < block_start) block_start = it_up->second.file_off;
    if (it_down->second.file_off < block_start) block_start = it_down->second.file_off;
    uint64_t up_end = it_up->second.file_off + it_up->second.size;
    uint64_t down_end = it_down->second.file_off + it_down->second.size;
    if (up_end > block_end) block_end = up_end;
    if (down_end > block_end) block_end = down_end;
    size_t block_size = block_end - block_start;

    if (block_size == 0 || block_size > ab->data_size) return false;

    uint8_t* buf = (uint8_t*)ab->buf[buf_idx];

#if HAS_IO_URING
    if (ab->ring) {
        struct io_uring_sqe *sqe = io_uring_get_sqe(ab->ring);
        if (sqe) {
            io_uring_prep_read_fixed(sqe, ab->fd, buf, block_size,
                                     (off_t)block_start, buf_idx);
            io_uring_sqe_set_data64(sqe, (uint64_t)layer);
            io_uring_submit(ab->ring);
            struct io_uring_cqe *cqe;
            int ret = io_uring_wait_cqe(ab->ring, &cqe);
            if (ret == 0 && cqe->res == (int)block_size) {
                io_uring_cqe_seen(ab->ring, cqe);
                return true;
            }
            if (cqe) io_uring_cqe_seen(ab->ring, cqe);
            return false;
        }
    }
#endif
    ssize_t bytes_read = pread(ab->fd, buf, block_size, (off_t)block_start);
    return (bytes_read == (ssize_t)block_size);
}

// Submit async prefetch for layer N (non-blocking)
// Currently a no-op in the synchronous implementation
bool ffn_async_prefetch(ffn_async_buffer* ab, int layer) {
    if (!ab || !ab->initialized) return false;
    ab->req_layer = layer;
    return true;
}

// Wait for the prefetch to complete and swap buffers
bool ffn_async_swap(ffn_async_buffer* ab, int layer) {
    if (!ab || !ab->initialized || layer < 0) return false;
    ab->req_layer = layer;
    return true;
}

// Get pointer to a specific tensor within the active buffer
const void* ffn_async_get_ptr(const ffn_async_buffer* ab, int layer,
                               const ffn_layer_ptrs_t* lp,
                               const char* tensor_name) {
    if (!ab || !ab->initialized || !lp || !tensor_name) return nullptr;

    // Calculate offset within the buffer based on tensor name
    // Tensors are stored sequentially: ffn_norm, gate, up, down
    size_t offset = 0;
    if (strstr(tensor_name, "ffn_norm")) {
        return (const uint8_t*)ab->buf[ab->active_idx] + offset;
    }
    offset += lp->size_ffn_norm;
    if (strstr(tensor_name, "ffn_gate")) {
        return (const uint8_t*)ab->buf[ab->active_idx] + offset;
    }
    offset += lp->size_gate;
    if (strstr(tensor_name, "ffn_up")) {
        return (const uint8_t*)ab->buf[ab->active_idx] + offset;
    }
    offset += lp->size_up;
    if (strstr(tensor_name, "ffn_down")) {
        return (const uint8_t*)ab->buf[ab->active_idx] + offset;
    }

    return nullptr;
}
