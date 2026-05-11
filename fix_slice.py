with open("llama.cpp-PoC/tools/llama-slice/llama-slice.cpp", "w") as f:
    f.write("""#include <iostream>

int main() {
    std::cout << "llama-slice: Not implemented in C++, use Python script." << std::endl;
    return 0;
}
""")
