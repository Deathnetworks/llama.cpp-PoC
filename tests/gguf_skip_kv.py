import struct

f = open("Models/Qwen3-0.6B-Q4_K_M.gguf", "rb")
f.read(24)  # header: magic(4) + version(4) + tensor_count(8) + kv_count(8)

kv_count = struct.unpack("<Q", f.read(8))[0]  # re-read kv_count from position 16
f.seek(16)
kv_count = struct.unpack("<Q", f.read(8))[0]
print(f"KV count: {kv_count}")

f.seek(24)  # back to start of KV pairs
for i in range(kv_count):
    kl = struct.unpack("<Q", f.read(8))[0]
    key = f.read(kl)
    vt = struct.unpack("<I", f.read(4))[0]
    if vt == 8:  # string
        sl = struct.unpack("<Q", f.read(8))[0]
        val = f.read(sl)
    elif vt in (0, 1, 7): f.read(1)
    elif vt in (2, 3, 14, 15, 20, 21, 22, 23): f.read(2)
    elif vt in (4, 5, 6, 16, 17, 18, 19, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39): f.read(4)
    elif vt in (10, 11): f.read(8)
    elif vt == 12:  # array
        at = struct.unpack("<I", f.read(4))[0]
        ac = struct.unpack("<Q", f.read(8))[0]
        # Skip array elements
        for j in range(ac):
            if at == 8:  # string
                sl2 = struct.unpack("<Q", f.read(8))[0]
                f.read(sl2)
            elif at in (0,1,7): f.read(1)
            elif at in (2,3,14,15): f.read(2)
            elif at in (4,5,6,16,17,18,19): f.read(4)
            elif at in (10,11): f.read(8)
            else: f.read(4)
    else:
        print(f"  Unknown type {vt} at KV {i}, key={key[:40]}")
        break
    if i < 5 or i == kv_count - 1:
        print(f"  KV {i}: key={key[:50]}... pos={f.tell()}")

print(f"\nPosition after all KV pairs: {f.tell()}")
print(f"File size: {f.seek(0, 2)}")
f.close()
