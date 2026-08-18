plugins {
    id("com.android.application")
}

android {
    namespace = "com.eugen.vbufchat"
    compileSdk = 36
    ndkVersion = "27.1.12297006"

    defaultConfig {
        applicationId = "com.eugen.vbufchat"
        minSdk = 29
        targetSdk = 36
        versionCode = 1
        versionName = "0.1"

        ndk {
            abiFilters += "arm64-v8a"
        }

        externalNativeBuild {
            cmake {
                arguments += "-DLLAMA_SRC=/tmp/llama.cpp-step21"
                arguments += "-DVBUF_RUST_LIB=${project.projectDir}/src/main/jniLibs/arm64-v8a/libvbuf_ml.so"
                arguments += "-DGGML_NATIVE=OFF"
                arguments += "-DGGML_CPU_ALL_VARIANTS=OFF"
                arguments += "-DLLAMA_BUILD_COMMON=OFF"
                arguments += "-DLLAMA_BUILD_TESTS=OFF"
                arguments += "-DLLAMA_BUILD_EXAMPLES=OFF"
                arguments += "-DLLAMA_BUILD_SERVER=OFF"
                arguments += "-DGGML_BACKEND_DL=OFF"
                arguments += "-DGGML_OPENMP=OFF"
                arguments += "-DANDROID_STL=c++_shared"
            }
        }
    }

    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
}
