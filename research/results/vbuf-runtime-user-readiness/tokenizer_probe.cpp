// Diagnostic only: exercise the exact tokenizer used by the measured server.
#define VBUF_COMPAT_SERVER_LIBRARY_ONLY
#include "../../../integrations/ggml/tools/vbuf_compat_server.cpp"
#undef VBUF_COMPAT_SERVER_LIBRARY_ONLY

int main(int argc, char ** argv) {
    if (argc != 3) return 2;
    VbufTokenizer tokenizer(argv[1]);
    const auto tokens = tokenizer.encode_text(argv[2], tokenizer.add_bos());
    std::cout << "add_bos=" << tokenizer.add_bos() << " tokens=";
    for (const auto token : tokens) std::cout << token << ',';
    std::cout << " hash=" << std::hex << token_hash(tokens) << '\n';
}
