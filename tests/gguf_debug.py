import struct

f = open("Models/Qwen3-0.6B-Q4_K_M.gguf", "rb")
f.read(4)  # magic
version = struct.unpack("<I", f.read(4))[0]
tensor_count = struct.unpack("<Q", f.read(8))[0]
kv_count = struct.unpack("<Q", f.read(8))[0]
print(f"v{version}, {tensor_count} tensors, {kv_count} KV pairs")

for i in range(kv_count):
    pos = f.tell()
    kl = struct.unpack("<Q", f.read(8))[0]
    key = f.read(kl)
    vt = struct.unpack("<I", f.read(4))[0]
    print(f"  KV {i} @ {pos}: key={key[:60]}... type={vt}")

    if vt == 8:  # string
        sl = struct.unpack("<Q", f.read(8))[0]
        f.read(sl)
    elif vt == 12:  # array
        at = struct.unpack("<I", f.read(4))[0]
        ac = struct.unpack("<Q", f.read(8))[0]
        print(f"    Array: type={at}, count={ac}")
        for j in range(ac):
            if at == 8:
                sl2 = struct.unpack("<Q", f.read(8))[0]
                if sl2 > 1000000:
                    print(f"    WARNING: huge string at array[{j}]: {sl2} bytes")
                    f.read(sl2)
                else:
                    f.read(sl2)
            elif at in (0, 1, 7): f.read(1)
            elif at in (2, 3, 14, 15): f.read(2)
            elif at in (4, 5, 6, 16, 17, 18, 19): f.read(4)
            elif at in (10, 11): f.read(8)
            else: f.read(4)
    elif vt in (0, 1, 7): f.read(1)
    elif vt in (2, 3, 14, 15): f.read(2)
    elif vt in (4, 5, 6, 16, 17, 18, 19): f.read(4)
    elif vt in (10, 11): f.read(8)
    elif vt == 9: f.read(1)
    else:
        print(f"    Unknown type {vt}, skipping 4 bytes")
        f.read(4)

print(f"\nFinal position: {f.tell()}")
f.close()
