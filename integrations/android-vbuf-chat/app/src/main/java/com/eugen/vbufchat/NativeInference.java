package com.eugen.vbufchat;

final class NativeInference {
    static {
        System.loadLibrary("vbuf_android_chat");
    }

    static native String open(String modelPath, String endpoint);
    static native String generate(String prompt, int maxTokens);
    static native String progress();
    static native String metrics();
    static native void cancel();
    static native void closeModel();

    private NativeInference() {}
}
