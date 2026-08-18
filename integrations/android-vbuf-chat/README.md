# Android ARM64 vBuf Chat POC

This is a Phase A local-inference consumer for the Pixel 7 Pro. It intentionally
uses a local Qwen3-0.6B vBuf artifact to isolate Android ARM64 native loading,
tokenization, prompt evaluation, and bounded greedy generation from Phase B
HTTP Range loading.

Validated target: Pixel 7 Pro, Android 17, `arm64-v8a`. The successful run opened
the local model in 1680 ms and generated 16 bounded tokens in 20620 ms. The
observed native heap PSS was 736924 KiB and total PSS was 1492923 KiB.

```text
Java Activity
    -> JNI facade
    -> llama_model_load_vbuf_direct
    -> existing vBuf/ML FFI + source-independent llama integration
    -> llama.cpp / ggml CPU backend
```

The model remains a vBuf artifact. Kotlin/Java does not parse vBuf, and the
native backend does not know whether a future materialized span came from SELF,
file, or HTTP RangeSource. Phase B only needs to replace the local source
materialization input; it is not implemented here.

## Build Prerequisites

The workstation must provide:

- Android SDK with platform 36 and build-tools;
- Android NDK `27.1.12297006`;
- CMake 3.22.1;
- Java 17+ and the Gradle wrapper;
- the prepared pinned llama.cpp checkout at `/tmp/llama.cpp-step21`;
- an Android Rust `aarch64-linux-android` cdylib build of `vbuf-ml`.

The Android inference worker uses an explicit 8 MiB stack. The current parser
path exceeded the default Java executor thread stack on Android; this is an
Android runtime requirement and does not change generic vBuf parser policy.
For Qwen3, the vBuf-ML metadata projection derives `rope.dimension_count`
deterministically from the authoritative key-head dimension. No persistent vBuf
format or source metadata change is involved.

Build the Rust library first, then copy its `libvbuf_ml.so` into
`app/src/main/jniLibs/arm64-v8a/`. Build the APK with:

```sh
export ANDROID_HOME=/home/eugen/Android/Sdk
export ANDROID_NDK_HOME="$ANDROID_HOME/ndk/27.1.12297006"
cargo build --manifest-path rust/Cargo.toml --release \
  --target aarch64-linux-android -p vbuf-ml
cp rust/target/aarch64-linux-android/release/libvbuf_ml.so \
  integrations/android-vbuf-chat/app/src/main/jniLibs/arm64-v8a/
integrations/android-vbuf-chat/gradlew \
  -p integrations/android-vbuf-chat assembleDebug
```

Deploy the local Phase A model to the app's private storage during development:

```sh
adb push research-models/Qwen3-0.6B-Q8_0.vbuf /data/local/tmp/
adb shell run-as com.eugen.vbufchat mkdir -p files/models
adb shell run-as com.eugen.vbufchat cp \
  /data/local/tmp/Qwen3-0.6B-Q8_0.vbuf files/models/
adb install -r integrations/android-vbuf-chat/app/build/outputs/apk/debug/app-debug.apk
adb shell monkey -p com.eugen.vbufchat 1
adb logcat -s vbuf-android-chat
```

The verified Phase A evidence is recorded in
`research/results/vbuf-android-arm64-chat-poc/phase-a-local-chat.json`.

## Scope

The UI is deliberately minimal: status, prompt, a bounded Generate button, and
generated text. It does not implement accounts, history, caching, prefetch,
remote loading, GPU backends, or production chat behavior.

Phase B is intentionally not implemented. Its next step is semantic bootstrap
plus a real Wi-Fi HTTP `RangeSource` feeding the same Android ARM64 compute path.
