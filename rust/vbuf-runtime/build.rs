fn main() {
    if std::env::var_os("CARGO_FEATURE_GGML").is_some() {
        if let Some(path) = std::env::var_os("VBUF_GGML_LIB_DIR") {
            println!("cargo:rustc-link-search=native={}", path.to_string_lossy());
        }
        println!("cargo:rustc-link-lib=dylib=ggml");
    }
    if std::env::var_os("CARGO_FEATURE_CUDA").is_some() {
        let manifest = std::path::PathBuf::from(
            std::env::var_os("CARGO_MANIFEST_DIR").expect("manifest directory"),
        );
        let source = manifest.join("cuda/cuda_backend.cu");
        let out =
            std::path::PathBuf::from(std::env::var_os("OUT_DIR").expect("cargo output directory"));
        let object = out.join("cuda_backend.o");
        let library = out.join("libvbuf_cuda.a");
        let status = std::process::Command::new("nvcc")
            .args([
                "-ccbin",
                "/usr/bin/g++-13",
                "-std=c++17",
                "-O3",
                "-arch=sm_86",
                "-Xcompiler",
                "-fPIC",
                "-c",
            ])
            .arg(&source)
            .arg("-o")
            .arg(&object)
            .status()
            .expect("failed to start nvcc");
        assert!(status.success(), "nvcc failed to compile CUDA backend");
        let status = std::process::Command::new("ar")
            .args(["crus"])
            .arg(&library)
            .arg(&object)
            .status()
            .expect("failed to start ar");
        assert!(status.success(), "ar failed to create CUDA backend library");
        println!("cargo:rustc-link-search=native={}", out.display());
        println!("cargo:rustc-link-lib=static=vbuf_cuda");
        println!("cargo:rustc-link-lib=dylib=stdc++");
        println!("cargo:rustc-link-lib=dylib=cudart");
        println!("cargo:rustc-link-lib=dylib=cublas");
        println!("cargo:rerun-if-changed={}", source.display());
    }
}
