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
        versionName = "0.2"

        buildFeatures {
            buildConfig = true
        }

        val remoteUrl = (project.findProperty("vbufRemoteUrl") as String?) ?: ""
        val residencyBudgetBytes = (project.findProperty("vbufResidencyBudgetBytes") as String?)
            ?: "268435456"
        require(residencyBudgetBytes.matches(Regex("[1-9][0-9]*"))) {
            "vbufResidencyBudgetBytes must be a positive decimal byte count"
        }
        buildTypes {
            getByName("debug") {
                buildConfigField("String", "VBUF_REMOTE_URL", "\"${remoteUrl.replace("\\\"", "\\\\\"")}\"")
            }
        }

        ndk {
            abiFilters += "arm64-v8a"
        }

        externalNativeBuild {
            cmake {
                val ggmlSrc = (project.findProperty("vbufGgmlSrc") as String?)
                    ?: "/tmp/llama.cpp-step21/ggml"
                arguments += "-DGGML_SRC=$ggmlSrc"
                arguments += "-DVBUF_RUST_LIB=${project.projectDir}/src/main/jniLibs/arm64-v8a/libvbuf_ml.so"
                arguments += "-DGGML_NATIVE=OFF"
                arguments += "-DGGML_CPU_ALL_VARIANTS=OFF"
                arguments += "-DGGML_BACKEND_DL=OFF"
                arguments += "-DGGML_OPENMP=OFF"
                arguments += "-DANDROID_STL=c++_shared"
                arguments += "-DVBUF_RESIDENCY_BUDGET_BYTES=$residencyBudgetBytes"
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
