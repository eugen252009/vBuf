fn main() {
    if std::env::var_os("CARGO_FEATURE_GGML").is_some() {
        if let Some(path) = std::env::var_os("VBUF_GGML_LIB_DIR") {
            println!("cargo:rustc-link-search=native={}", path.to_string_lossy());
        }
        println!("cargo:rustc-link-lib=dylib=ggml");
    }
}
