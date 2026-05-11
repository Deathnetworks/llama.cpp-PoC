with open("llama.cpp-PoC/tools/CMakeLists.txt", "r") as f:
    content = f.read()

if "llama-slice" not in content:
    content = content.replace(
        "    add_subdirectory(gguf-split)",
        "    add_subdirectory(gguf-split)\n    add_subdirectory(llama-slice)"
    )
    with open("llama.cpp-PoC/tools/CMakeLists.txt", "w") as f:
        f.write(content)
