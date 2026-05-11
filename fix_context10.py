import subprocess
out = subprocess.check_output("grep -n 'build_ffn' llama.cpp-PoC/src/models/* || true", shell=True).decode()
print("Found build_ffn in models:\n", out[:500])
