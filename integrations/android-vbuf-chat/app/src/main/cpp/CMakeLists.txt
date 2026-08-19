cmake_minimum_required(VERSION 3.22.1)
project(vbuf_android_chat LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(VBUF_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../../../../..")
set(LLAMA_SRC "/tmp/llama.cpp-step21" CACHE PATH "Pinned llama.cpp checkout")
set(VBUF_RUST_LIB "${CMAKE_CURRENT_LIST_DIR}/../jniLibs/arm64-v8a/libvbuf_ml.so" CACHE FILEPATH "Android vbuf-ml cdylib")

if(NOT EXISTS "${LLAMA_SRC}/CMakeLists.txt")
    message(FATAL_ERROR "LLAMA_SRC must point at the prepared pinned llama.cpp checkout")
endif()
if(NOT EXISTS "${VBUF_RUST_LIB}")
    message(FATAL_ERROR "Build vbuf-ml for aarch64-linux-android and place libvbuf_ml.so at ${VBUF_RUST_LIB}")
endif()

set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
add_subdirectory("${LLAMA_SRC}" "${CMAKE_CURRENT_BINARY_DIR}/llama")
target_sources(llama PRIVATE
    "${VBUF_ROOT}/integrations/ggml/src/vbuf_d0_1_diagnostics.cpp")
target_include_directories(llama PRIVATE
    "${VBUF_ROOT}/integrations/ggml/include")
target_link_libraries(llama PRIVATE log)

add_library(vbuf_ml SHARED IMPORTED)
set_target_properties(vbuf_ml PROPERTIES
    IMPORTED_LOCATION "${VBUF_RUST_LIB}"
    IMPORTED_NO_SONAME TRUE)

add_library(vbuf_android_chat SHARED
    vbuf_android_chat.cpp
    "${VBUF_ROOT}/integrations/llama.cpp/llama_vbuf_loader.cpp"
    "${VBUF_ROOT}/integrations/llama.cpp/vbuf_direct_source.cpp"
    "${VBUF_ROOT}/integrations/llama.cpp/vbuf_ml_adapter.cpp"
    "${VBUF_ROOT}/integrations/llama.cpp/vbuf_remote_source.cpp"
    "${VBUF_ROOT}/integrations/ggml/src/vbuf_range_source.cpp"
    "${VBUF_ROOT}/integrations/ggml/src/vbuf_materializer.cpp")

target_include_directories(vbuf_android_chat PRIVATE
    "${LLAMA_SRC}"
    "${LLAMA_SRC}/include"
    "${LLAMA_SRC}/common"
    "${LLAMA_SRC}/src"
    "${LLAMA_SRC}/ggml/include"
    "${LLAMA_SRC}/ggml/src"
    "${VBUF_ROOT}/integrations/llama.cpp"
    "${VBUF_ROOT}/integrations/ggml/include")

target_link_libraries(vbuf_android_chat
    llama
    vbuf_ml
    log
    android
    c++_shared)
