# Android ARM64 vBuf-ML Direct Runtime Demo

This is the Android user-facing demo for the canonical direct PoC22/vBuf-ML
runtime. It uses the DeepSeek-V2-Lite IQ2_XXS semantic bootstrap, fetches the
validated physical payload ranges over HTTP, keeps residency bounded at 256 MiB,
and executes the borrowed tensors through the GGML CPU backend.

The native JNI library is an adapter around the existing direct runtime. It does
not use `llama_model_loader`, preload the model, or own source acquisition.

```text
Java Activity
    -> JNI facade
    -> PoC22 direct runtime
    -> vBuf-ML consumer / HttpRangeSource / materializer / residency
    -> GGML CPU backend
```

The Java UI does not parse vBuf. Prompt tokenization uses the validated
tokenizer metadata in the semantic bootstrap; generated text is decoded from
the returned runtime token IDs.

## Build Prerequisites

The workstation must provide:

- Android SDK with platform 36 and build-tools;
- Android NDK `27.1.12297006`;
- CMake 3.22.1;
- Java 17+ and the Gradle wrapper;
- the prepared pinned ggml checkout at `/tmp/llama.cpp-step21/ggml`;
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

When the pinned checkout is elsewhere, pass it explicitly with
`-PvbufGgmlSrc=/path/to/ggml`. The app remains a thin adapter over the direct
vBuf-ML runtime; this is only a build-source configuration seam.

Deploy the local Phase A model to the app's private storage during development:

```sh
adb push /tmp/opencode/deepseek-v2-lite-imat/DeepSeek-V2-Lite.IQ2_XXS.semantic.vbuf /data/local/tmp/
adb shell run-as com.eugen.vbufchat mkdir -p files/models
adb shell run-as com.eugen.vbufchat cp \
  /data/local/tmp/DeepSeek-V2-Lite.IQ2_XXS.semantic.vbuf \
  files/models/
adb install -r integrations/android-vbuf-chat/app/build/outputs/apk/debug/app-debug.apk
adb shell monkey -p com.eugen.vbufchat 1
adb logcat -s vbuf-android-direct
```

Build the APK with the endpoint configured as a debug build property:

```sh
integrations/android-vbuf-chat/gradlew \
  -p integrations/android-vbuf-chat \
  -PvbufRemoteUrl=http://127.0.0.1:18124/models/DeepSeek-V2-Lite.IQ2_XXS.vbuf \
  assembleDebug
```

The active vBuf-ML residency budget defaults to `268435456` bytes. A
qualification build may vary only that numeric budget with, for example,
`-PvbufResidencyBudgetBytes=536870912`; the residency algorithm and policy stay
unchanged.

Use `adb reverse tcp:18124 tcp:18124` for the canonical local range server.

## Scope

The UI is deliberately minimal: model status, prompt, bounded Generate and
Cancel controls, generated text, and measured runtime counters. It does not
implement accounts, history, GPU backends, or production chat behavior.

Generation is bounded to the existing direct runtime context and uses the
current required readiness and lease boundaries. `consumer_wait` remains
explicitly `not instrumented`.
