// llama.cpp-PoC/src/llama-ffn-async.cpp
// Async double-buffer engine for zero-RAM FFN inference
// Dedicated I/O thread prefetches next layer while CPU computes current layer

#include "llama-ffn-local.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <errno.h>
#include <string.h>

#ifndef _WIN32
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/uio.h>
#include <pthread.h>
#if defined(__linux__)
// liburing may not be available on all systems — guard with __has_include
#if __has_include(<liburing.h>)
#include <liburing.h>
#endif
#endif
#else
#include <windows.h>
#include <io.h>
#endif

// ── Aligned memory allocation ──────────────────────────────────────────────

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

// ── Async buffer engine ────────────────────────────────────────────────────
// Uses io_uring on Linux for truly asynchronous SSD reads.
// Dual-buffer ping-pong: while CPU computes layer N from buffer A,
// io_uring async-loads layer N+1 into buffer B.

bool ffn_async_init(ffn_async_buffer* ab, const char* path, int n_layers) {
    if (!ab || !path || n_layers <= 0) return false;

    // Open file with O_DIRECT for aligned I/O (required for io_uring registered buffers)
#ifndef _WIN32
    ab->fd = open(path, O_RDONLY | O_DIRECT);
    if (ab->fd >= 0) {
        posix_fadvise(ab->fd, 0, 0, POSIX_FADV_SEQUENTIAL);
    }
#else
    ab->fd = open(path, O_RDONLY | O_BINARY);
#endif
    if (ab->fd < 0) {
        fprintf(stderr, "ffn_async: failed to open '%s': %s\n", path, strerror(errno));
        return false;
    }

    // Compute max FFN block size across all layers
    // For now use conservative 256MB — actual size computed during first load
    size_t max_layer_size = 256 * 1024 * 1024;  // 256 MB
    ab->data_size = max_layer_size;
    ab->n_layers = n_layers;

    // Allocate two page-aligned buffers
    ab->buf[0] = aligned_alloc_4k(max_layer_size);
    ab->buf[1] = aligned_alloc_4k(max_layer_size);
    if (!ab->buf[0] || !ab->buf[1]) {
        fprintf(stderr, "ffn_async: failed to allocate buffers (%zu MB each)\n", max_layer_size / (1024*1024));
        aligned_free_4k(ab->buf[0]);
        aligned_free_4k(ab->buf[1]);
        close(ab->fd);
        ab->fd = -1;
        return false;
    }

    ab->active_idx = 0;
    ab->prefetch_idx = 1;
    ab->loaded_layer = -1;
    ab->req_layer = -1;

    // Initialize threading
    mutex_init(&ab->mtx);
    cond_init(&ab->cond);
    ab->io_thread = 0;
    ab->io_running = false;
    ab->io_exit = false;

    // Set up io_uring (Linux with liburing)
#if HAS_IO_URING
    ab->ring = (struct io_uring*)malloc(sizeof(struct io_uring));
    if (!ab->ring) {
        fprintf(stderr, "ffn_async: failed to allocate io_uring\n");
        goto fail;
    }
    struct io_uring_params params = {0};
    int ret = io_uring_queue_init_params(32, ab->ring, &params);
    if (ret < 0) {
        fprintf(stderr, "ffn_async: io_uring_queue_init failed: %s\n", strerror(-ret));
        free(ab->ring);
        ab->ring = nullptr;
        goto fail;
    }

    // Register buffers with io_uring for zero-copy fixed reads
    ab->iov[0].iov_base = ab->buf[0];
    ab->iov[0].iov_len  = max_layer_size;
    ab->iov[1].iov_base = ab->buf[1];
    ab->iov[1].iov_len  = max_layer_size;
    ret = io_uring_register_buffers(ab->ring, ab->iov, 2);
    if (ret < 0) {
        fprintf(stderr, "ffn_async: io_uring_register_buffers failed: %s (falling back to pread)\n", strerror(-ret));
        // Non-fatal — we'll use regular pread fallback
    } else {
        ab->buf_registered = 1;
        fprintf(stderr, "ffn_async: io_uring initialized with %d registered buffers\n", 2);
    }
#endif

    fprintf(stderr, "ffn_async: initialized — fd=%d, buffer_size=%zu MB, layers=%d\n",
        ab->fd, max_layer_size / (1024*1024), n_layers);
    ab->initialized = true;
    return true;

fail:
    aligned_free_4k(ab->buf[0]);
    aligned_free_4k(ab->buf[1]);
    close(ab->fd);
    ab->fd = -1;
    return false;
}

// ── I/O thread: async prefetch via io_uring ────────────────────────────────
// Submits io_uring_prep_read_fixed() requests for whole-layer FFN blocks.
// While one read is in-flight, we can submit the next read immediately.
// This overlaps I/O with GPU Attention computation.

void* io_thread_func(void* arg) {
    io_thread_data* data = (io_thread_data*)arg;
    ffn_async_buffer* ab = data->ab;
    const auto* weight_map = data->weight_map;

    while (true) {
        // Wait for a prefetch request
        mutex_lock(&ab->mtx);
        while (ab->req_layer < 0 && !ab->io_exit) {
            cond_wait(&ab->cond, &ab->mtx);
        }
        if (ab->io_exit) {
            mutex_unlock(&ab->mtx);
            break;
        }
        int layer = ab->req_layer;
        int buf_idx = ab->prefetch_idx;
        ab->req_layer = -1;
        mutex_unlock(&ab->mtx);

        // Look up file offsets for this layer's FFN block
        char name_gate[64], name_up[64], name_down[64];
        snprintf(name_gate, sizeof(name_gate), "blk.%d.ffn_gate.weight", layer);
        snprintf(name_up,   sizeof(name_up),   "blk.%d.ffn_up.weight",   layer);
        snprintf(name_down, sizeof(name_down), "blk.%d.ffn_down.weight", layer);

        auto it_gate = weight_map->find(name_gate);
        auto it_up   = weight_map->find(name_up);
        auto it_down = weight_map->find(name_down);

        if (it_gate == weight_map->end() || it_up == weight_map->end() || it_down == weight_map->end()) {
            LLAMA_LOG_DEBUG("streaming: layer %d missing weight entries\n", layer);
            continue;
        }

        // Compute contiguous block range
        uint64_t block_start = it_gate->second.file_off;
        uint64_t block_end   = it_gate->second.file_off + it_gate->second.size;
        if (it_up->second.file_off < block_start) block_start = it_up->second.file_off;
        if (it_down->second.file_off < block_start) block_start = it_down->second.file_off;
        uint64_t up_end = it_up->second.file_off + it_up->second.size;
        uint64_t down_end = it_down->second.file_off + it_down->second.size;
        if (up_end > block_end) block_end = up_end;
        if (down_end > block_end) block_end = down_end;
        size_t block_size = block_end - block_start;

        if (block_size == 0 || block_size > ab->data_size) {
            fprintf(stderr, "ffn_async: layer %d invalid block_size=%zu\n", layer, block_size);
            continue;
        }

        // Submit async read via io_uring (Linux) or synchronous pread (fallback)
        bool read_ok = false;

#if HAS_IO_URING
        if (ab->ring && ab->buf_registered) {
            // io_uring async read with pre-registered buffer
            struct io_uring_sqe *sqe = io_uring_get_sqe(ab->ring);
            if (sqe) {
                // buf_idx is 0 or 1, matching our registered iov entries
                io_uring_prep_read_fixed(sqe, ab->fd, ab->buf[buf_idx], block_size,
                                         (off_t)block_start, buf_idx);
                io_uring_sqe_set_data64(sqe, (uint64_t)layer);
                io_uring_submit(ab->ring);

                // Wait for completion
                struct io_uring_cqe *cqe;
                int ret = io_uring_wait_cqe(ab->ring, &cqe);
                if (ret == 0 && cqe->res == (int)block_size) {
                    read_ok = true;
                } else {
                    LLAMA_LOG_DEBUG("streaming: io_uring read failed for layer %d: res=%d\n", layer, cqe ? cqe->res : ret);
                }
                if (cqe) io_uring_cqe_seen(ab->ring, cqe);
            }
        }
#endif
        if (!read_ok) {
            // Fallback: synchronous pread
            ssize_t bytes_read = pread(ab->fd, ab->buf[buf_idx], block_size, (off_t)block_start);
            read_ok = (bytes_read == (ssize_t)block_size);
        }

        if (read_ok) {
            mutex_lock(&ab->mtx);
            ab->loaded_layer = layer;
                    cond_broadcast(&ab->cond);  // wake all waiters
                    mutex_unlock(&ab->mtx);

            // Read-ahead hints (non-blocking)
#ifndef _WIN32
            // Drop consumed layer pages
            int consumed = layer - 2;
            if (consumed >= 0) {
                char cg[64], cu[64], cd[64];
                snprintf(cg, sizeof(cg), "blk.%d.ffn_gate.weight", consumed);
                snprintf(cu, sizeof(cu), "blk.%d.ffn_up.weight", consumed);
                snprintf(cd, sizeof(cd), "blk.%d.ffn_down.weight", consumed);
                auto itg = weight_map->find(cg);
                auto itu = weight_map->find(cu);
                auto itd = weight_map->find(cd);
                if (itg != weight_map->end())
                    posix_fadvise(ab->fd, (off_t)itg->second.file_off, (off_t)itg->second.size, POSIX_FADV_DONTNEED);
                if (itu != weight_map->end())
                    posix_fadvise(ab->fd, (off_t)itu->second.file_off, (off_t)itu->second.size, POSIX_FADV_DONTNEED);
                if (itd != weight_map->end())
                    posix_fadvise(ab->fd, (off_t)itd->second.file_off, (off_t)itd->second.size, POSIX_FADV_DONTNEED);
            }
            // Prefetch next 2 layers
            for (int ahead = 1; ahead <= 2; ahead++) {
                int nl = layer + ahead;
                if (nl >= ab->n_layers) continue;
                char ng[64], nu[64], nd[64];
                snprintf(ng, sizeof(ng), "blk.%d.ffn_gate.weight", nl);
                snprintf(nu, sizeof(nu), "blk.%d.ffn_up.weight", nl);
                snprintf(nd, sizeof(nd), "blk.%d.ffn_down.weight", nl);
                auto itg = weight_map->find(ng);
                auto itu = weight_map->find(nu);
                auto itd = weight_map->find(nd);
                if (itg != weight_map->end() && itu != weight_map->end() && itd != weight_map->end()) {
                    uint64_t ns = itg->second.file_off;
                    uint64_t ne = itg->second.file_off + itg->second.size;
                    if (itu->second.file_off < ns) ns = itu->second.file_off;
                    if (itd->second.file_off < ns) ns = itd->second.file_off;
                    uint64_t ue = itu->second.file_off + itu->second.size;
                    uint64_t de = itd->second.file_off + itd->second.size;
                    if (ue > ne) ne = ue;
                    if (de > ne) ne = de;
                    size_t sz = ne - ns;
                    if (sz > 0 && sz < 100*1024*1024)
                        posix_fadvise(ab->fd, (off_t)ns, (off_t)sz, POSIX_FADV_WILLNEED);
                }
            }
#endif
        } else {
            LLAMA_LOG_DEBUG("streaming: failed to load layer %d\n", layer);
        }
    }
    return NULL;
}

void ffn_async_free(ffn_async_buffer* ab) {
    if (!ab || !ab->initialized) return;

    // Shut down I/O thread
    if (ab->io_running) {
        mutex_lock(&ab->mtx);
        ab->io_exit = true;
        cond_signal(&ab->cond);
        mutex_unlock(&ab->mtx);
        thread_join(ab->io_thread);
        ab->io_running = false;
    }

    // Tear down io_uring
#if HAS_IO_URING
    if (ab->ring) {
        io_uring_queue_exit(ab->ring);
        free(ab->ring);
        ab->ring = nullptr;
    }
#endif

    if (ab->fd >= 0) close(ab->fd);
    aligned_free_4k(ab->buf[0]);
    aligned_free_4k(ab->buf[1]);
    ab->fd = -1;
    ab->buf[0] = nullptr;
    ab->buf[1] = nullptr;
    ab->initialized = false;

    mutex_destroy(&ab->mtx);
    cond_destroy(&ab->cond);
}

// ── Synchronous fallback: load a layer into active buffer ──────────────────
// Used by context.cpp for pre-loading layer 0 before graph construction.
// Reads entire FFN block (gate+up+down) as ONE contiguous pread().
bool ffn_async_load_layer(ffn_async_buffer* ab, int layer,
                           const std::unordered_map<std::string, ffn_weight_offset>* weight_map) {
    if (!ab || !ab->initialized || layer < 0 || !weight_map) return false;
    if (ab->fd < 0) return false;

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

    // Load into the prefetch buffer (inactive), then swap
    int buf_idx = ab->prefetch_idx;
    ssize_t bytes_read = pread(ab->fd, ab->buf[buf_idx], block_size, (off_t)block_start);
    if (bytes_read != (ssize_t)block_size) return false;

    // Swap: prefetch buffer becomes active
    int tmp = ab->active_idx;
    ab->active_idx = ab->prefetch_idx;
    ab->prefetch_idx = tmp;
    ab->loaded_layer = layer;

    return true;
}
