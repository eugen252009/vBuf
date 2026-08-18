package com.eugen.vbufchat;

final class NativeInference {
    static {
        System.loadLibrary("vbuf_android_chat");
    }

    static native String open(String modelPath);
    static native String generate(String prompt, int maxTokens);
    static native void closeModel();

    private NativeInference() {}
}
