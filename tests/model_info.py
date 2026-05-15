#!/usr/bin/env python3
import os, sys
sys.path.insert(0, "/tmp/venv/lib/python3.11/site-packages")
from gguf import GGUFReader, GGUFValueType
import numpy as np

def get_val(reader, key):
    try:
        field = reader.get_field(key)
        if field and field.data:
            idx = field.data[0]
            if idx < len(field.parts):
                part = field.parts[idx]
                if isinstance(part, np.ndarray):
                    return int(part[0])
                return int(part)
    except:
        pass
    return 0

def get_str(reader, key):
    try:
        field = reader.get_field(key)
        if field and field.data:
            idx = field.data[0]
            if idx < len(field.parts):
                part = field.parts[idx]
                if isinstance(part, np.ndarray) and part.dtype == np.uint8:
                    return part.tobytes().decode("utf-8", errors="replace")
                return str(part)
    except:
        pass
    return ""

def analyze(path):
    size = os.path.getsize(path)
    reader = GGUFReader(path)

    arch = get_str(reader, "general.architecture") or "qwen35"
    n_layer = get_val(reader, f"{arch}.block_count")
    ctx_len = get_val(reader, f"{arch}.context_length")
    n_embd = get_val(reader, f"{arch}.embedding_length")
    n_ffn = get_val(reader, f"{arch}.feed_forward_length")
    n_head = get_val(reader, f"{arch}.attention.head_count")
    n_head_kv = get_val(reader, f"{arch}.attention.head_count_kv")
    head_size = get_val(reader, f"{arch}.attention.key_length")

    # Categorize block tensors
    ffn_bytes = attn_bytes = 0
    ffn_count = attn_count = 0
    for t in reader.tensors:
        if "blk." not in t.name: continue
        try:
            sz = t.data.nbytes
        except:
            sz = len(t.data)
        if "ffn" in t.name:
            ffn_bytes += sz
            ffn_count += 1
        else:
            attn_bytes += sz
            attn_count += 1

    print(f"\n{'='*60}")
    print(f"Model: {os.path.basename(path)}")
    print(f"{'='*60}")
    print(f"File size: {size/1024/1024:.0f} MB")
    print(f"Arch: {arch}, Layers: {n_layer}, Embed: {n_embd}, FFN: {n_ffn}")
    print(f"Heads: {n_head} (KV: {n_head_kv}), Head size: {head_size}")
    print(f"Default ctx: {ctx_len}")
    print(f"\nBlock tensors (quantized on-disk):")
    print(f"  FFN:  {ffn_count} tensors, {ffn_bytes/1024/1024:.1f} MB")
    print(f"  Attn: {attn_count} tensors, {attn_bytes/1024/1024:.1f} MB")
    print(f"  Total: {(ffn_bytes+attn_bytes)/1024/1024:.1f} MB")

    if n_layer > 0 and n_head_kv > 0 and head_size > 0:
        print(f"\nKV cache sizes (Q8):")
        for ctx in [2048, 4096, 8192]:
            kv_q8 = 2 * n_layer * n_head_kv * head_size * ctx
            gpu_split = (attn_bytes + kv_q8) / 1024 / 1024
            gpu_full = (ffn_bytes + attn_bytes + kv_q8) / 1024 / 1024
            print(f"  ctx={ctx:>6}: local-gpu={gpu_full:.0f} MB, local-ssd={gpu_split:.0f} MB (FFN {ffn_bytes/1024/1024:.0f} MB on CPU)")

models = [
    "Models/Qwen3.5-2B-Q4_K_M.gguf",
    "Models/Qwen3.5-4B-Q4_K_M.gguf",
    "Models/Qwen3.5-9B-Q4_K_M.gguf",
    "Models/Qwen3.5-27B-Q4_K_M.gguf",
]

for m in models:
    if os.path.exists(m):
        analyze(m)
    else:
        print(f"\nNOT FOUND: {m}")
