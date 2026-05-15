import struct, os

path = "Models/Qwen3-0.6B-Q4_K_M.gguf"
size = os.path.getsize(path)

with open(path, "rb") as f:
    f.read(4)  # magic
    f.read(4)  # version
    f.read(8)  # tensor_count
    kv_count = struct.unpack("<Q", f.read(8))[0]

    # Skip all KV pairs
    for i in range(kv_count):
        kl = struct.unpack("<Q", f.read(8))[0]
        f.read(kl)  # key
        vt = struct.unpack("<I", f.read(4))[0]

        # GGUF v3 value types
        if vt == 0: f.read(1)  # uint8
        elif vt == 1: f.read(1)  # int8
        elif vt == 2: f.read(2)  # uint16
        elif vt == 3: f.read(2)  # int16
        elif vt == 4: f.read(4)  # uint32
        elif vt == 5: f.read(4)  # int32
        elif vt == 6: f.read(4)  # float32
        elif vt == 7: f.read(1)  # bool
        elif vt == 8:  # string
            sl = struct.unpack("<Q", f.read(8))[0]
            f.read(sl)
        elif vt == 9:  # array (GGUF v3)
            at = struct.unpack("<I", f.read(4))[0]  # array element type
            ac = struct.unpack("<Q", f.read(8))[0]  # array count
            for j in range(ac):
                if at == 8:  # string array
                    sl2 = struct.unpack("<Q", f.read(8))[0]
                    f.read(sl2)
                elif at in (0, 1, 7): f.read(1)
                elif at in (2, 3): f.read(2)
                elif at in (4, 5, 6): f.read(4)
                elif at in (10, 11): f.read(8)
                else: f.read(4)
        elif vt == 10: f.read(8)  # uint64
        elif vt == 11: f.read(8)  # int64
        elif vt == 12: f.read(8)  # float64
        else:
            print(f"Unknown type {vt} at KV {i}")
            break

    tensor_start = f.tell()
    print(f"Tensor info starts at: {tensor_start}")

    # Read all 311 tensors
    tensors = []
    for i in range(311):
        nl = struct.unpack("<Q", f.read(8))[0]
        name = f.read(nl).decode("utf-8", errors="replace")
        nd = struct.unpack("<I", f.read(4))[0]
        dims = tuple(struct.unpack("<Q", f.read(8))[0] for _ in range(nd))
        dtype = struct.unpack("<I", f.read(4))[0]
        offset = struct.unpack("<Q", f.read(8))[0]
        elems = 1
        for d in dims: elems *= d
        # Type sizes for GGML types
        ts = {0:1, 1:1, 2:2, 3:2, 4:4, 5:4, 6:4, 7:1, 8:2, 9:2, 10:4, 11:4, 12:8, 13:8, 14:2, 15:2, 16:4, 17:4, 18:8, 19:8, 20:1, 21:1, 22:2, 23:2, 24:4, 25:4, 26:8, 27:8, 28:4, 29:4, 30:8, 31:8}.get(dtype, 4)
        tensors.append((name, dims, dtype, elems * ts))

print(f"Read {len(tensors)} tensors")

# Categorize
attn = sum(s for n,d,t,s in tensors if any(k in n for k in ["attn_norm","attn_q","attn_k","attn_v","attn_output"]))
ffn = sum(s for n,d,t,s in tensors if any(k in n for k in ["ffn_norm","ffn_gate","ffn_up","ffn_down"]))
total = sum(s for n,d,t,s in tensors)
other = total - attn - ffn
attn_n = sum(1 for n,d,t,s in tensors if any(k in n for k in ["attn_norm","attn_q","attn_k","attn_v","attn_output"]))
ffn_n = sum(1 for n,d,t,s in tensors if any(k in n for k in ["ffn_norm","ffn_gate","ffn_up","ffn_down"]))

n_layers = 28
layers = {}
for name, dims, dtype, sz in tensors:
    ln = -1
    if "blk." in name:
        try: ln = int(name.split("blk.")[1].split(".")[0])
        except: pass
    if ln not in layers: layers[ln] = [0, 0, 0]
    if any(k in name for k in ["attn_norm","attn_q","attn_k","attn_v","attn_output"]): layers[ln][0] += sz
    elif any(k in name for k in ["ffn_norm","ffn_gate","ffn_up","ffn_down"]): layers[ln][1] += sz
    else: layers[ln][2] += sz

print(f"\nAttention:  {attn_n} tensors, {attn/1024/1024:.2f} MB ({attn/1024:.0f} KB)")
print(f"FFN:        {ffn_n} tensors, {ffn/1024/1024:.2f} MB ({ffn/1024:.0f} KB)")
print(f"Other:      {len(tensors)-attn_n-ffn_n} tensors, {other/1024/1024:.2f} MB")
print(f"Total:      {len(tensors)} tensors, {total/1024/1024:.2f} MB")
print(f"\nLayers: {n_layers}")
print(f"Per-layer attn avg: {attn/n_layers/1024:.0f} KB")
print(f"Per-layer FFN avg:  {ffn/n_layers/1024:.0f} KB")

# Write output
with open("Models/Qwen3-0.6B-Q4_K_M.LayerSizes.md", "w") as out:
    out.write("# Qwen3-0.6B-Q4_K_M.gguf — Layer Size Analysis\n\n")
    out.write(f"**File size**: {size:,} bytes ({size/1024/1024:.1f} MB)\n")
    out.write(f"**Tensors**: {len(tensors)}, **Layers**: {n_layers}\n\n")
    out.write("## Architecture\n- Embedding length: 1024\n- FFN length: 3072\n- Attention heads: 16 (KV: 8)\n- KV head size: 128\n- Context length: 40960\n\n")
    out.write("## Tensor Size Summary\n\n")
    out.write("| Category | Count | Size (MB) |\n|----------|-------|----------|\n")
    out.write(f"| Attention | {attn_n} | {attn/1024/1024:.2f} |\n")
    out.write(f"| FFN | {ffn_n} | {ffn/1024/1024:.2f} |\n")
    out.write(f"| Other | {len(tensors)-attn_n-ffn_n} | {other/1024/1024:.2f} |\n")
    out.write(f"| **Total** | **{len(tensors)}** | **{total/1024/1024:.2f}** |\n\n")
    out.write("## Per-Layer Breakdown\n\n")
    out.write("| Layer | Attn (KB) | FFN (KB) | Other (KB) | Total (KB) |\n")
    out.write("|-------|-----------|----------|------------|------------|\n")
    for l in range(n_layers):
        a, f, o = layers.get(l, [0, 0, 0])
        out.write(f"| {l} | {a/1024:.0f} | {f/1024:.0f} | {o/1024:.0f} | {(a+f+o)/1024:.0f} |\n")
    out.write(f"| **TOTAL** | **{attn/1024:.0f}** | **{ffn/1024:.0f}** | **{layers.get(-1,[0,0,0])[2]/1024:.0f}** | **{total/1024:.0f}** |\n\n")
    out.write("## VRAM Implications for Decoupled Attention\n\n")
    out.write("### Option A: FFN on CPU (mmap), Attention on GPU\n")
    out.write(f"- GPU VRAM: {attn/1024/1024:.2f} MB (attention weights) + KV cache\n")
    out.write(f"- CPU RAM: {ffn/1024/1024:.2f} MB (FFN weights, mmap'd)\n")
    out.write(f"- PCIe per layer: ~{1024*2/1024:.1f} KB residual (f16)\n\n")
    out.write("### Option B: All on GPU\n")
    out.write(f"- GPU VRAM: {(attn+ffn)/1024/1024:.2f} MB\n\n")
    out.write("### Option C: FFN on GPU via OpenCL, Attention on GPU\n")
    out.write(f"- GPU VRAM: {(attn+ffn)/1024/1024:.2f} MB (all weights, no PCIe transfer)\n\n")
    out.write("## All Tensors (sorted by size)\n\n")
    out.write("| Name | Dims | Type | Size (KB) |\n")
    out.write("|------|------|------|----------|\n")
    for name, dims, dtype, sz in sorted(tensors, key=lambda t: t[3], reverse=True):
        dims_str = "x".join(str(d) for d in dims)
        out.write(f"| {name} | {dims_str} | {dtype} | {sz/1024:.0f} |\n")

print("\nSaved to Models/Qwen3-0.6B-Q4_K_M.LayerSizes.md")
