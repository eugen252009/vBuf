cmake_minimum_required(VERSION 3.22.1)
project(vbuf_android_chat LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(VBUF_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../../../../..")
set(GGML_SRC "/tmp/llama.cpp-step21/ggml" CACHE PATH "Pinned ggml checkout")
set(VBUF_RUST_LIB "${CMAKE_CURRENT_LIST_DIR}/../jniLibs/arm64-v8a/libvbuf_ml.so" CACHE FILEPATH "Android vbuf-ml cdylib")

if(NOT EXISTS "${GGML_SRC}/CMakeLists.txt")
    message(FATAL_ERROR "GGML_SRC must point at the prepared pinned ggml checkout")
endif()
if(NOT EXISTS "${VBUF_RUST_LIB}")
    message(FATAL_ERROR "Build vbuf-ml for aarch64-linux-android and place libvbuf_ml.so at ${VBUF_RUST_LIB}")
endif()

set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GGML_CPU ON CACHE BOOL "" FORCE)
set(GGML_CPU_REPACK OFF CACHE BOOL "" FORCE)
set(GGML_BLAS OFF CACHE BOOL "" FORCE)
set(GGML_ACCELERATE OFF CACHE BOOL "" FORCE)
set(GGML_LLAMAFILE OFF CACHE BOOL "" FORCE)
add_subdirectory("${GGML_SRC}" "${CMAKE_CURRENT_BINARY_DIR}/ggml" EXCLUDE_FROM_ALL)

add_library(vbuf_ml SHARED IMPORTED)
set_target_properties(vbuf_ml PROPERTIES
    IMPORTED_LOCATION "${VBUF_RUST_LIB}"
    IMPORTED_NO_SONAME TRUE)

add_library(vbuf_region_executor STATIC
    "${VBUF_ROOT}/integrations/ggml/src/vbuf_region_executor.cpp"
    "${VBUF_ROOT}/integrations/ggml/src/vbuf_execution_region.cpp"
    "${VBUF_ROOT}/integrations/ggml/src/vbuf_tensor_wave.cpp"
    "${VBUF_ROOT}/integrations/ggml/src/vbuf_prefetch_planner.cpp"
    "${VBUF_ROOT}/integrations/ggml/src/vbuf_materializer.cpp"
    "${VBUF_ROOT}/integrations/ggml/src/vbuf_d0_1_diagnostics.cpp"
    "${VBUF_ROOT}/integrations/ggml/src/vbuf_range_source.cpp"
    "${VBUF_ROOT}/integrations/ggml/src/vbuf_source_selection.cpp"
    "${VBUF_ROOT}/integrations/ggml/src/vbuf_residency.cpp"
    "${VBUF_ROOT}/integrations/ggml/src/vbuf_tiered_residency.cpp"
    "${VBUF_ROOT}/integrations/ggml/src/vbuf_striped_materializer.cpp"
    "${VBUF_ROOT}/integrations/ggml/src/vbuf_topk.cpp"
    "${VBUF_ROOT}/integrations/ggml/src/vbuf_weighted_merge.cpp"
    "${VBUF_ROOT}/integrations/ggml/src/vbuf_runtime_state.cpp")
target_compile_features(vbuf_region_executor PUBLIC cxx_std_17)
target_include_directories(vbuf_region_executor PUBLIC
    "${VBUF_ROOT}/integrations/ggml/include"
    "${GGML_SRC}/include")
target_link_libraries(vbuf_region_executor PUBLIC ggml vbuf_ml)

add_library(vbuf_android_chat SHARED
    vbuf_android_direct.cpp)

target_include_directories(vbuf_android_chat PRIVATE
    "${VBUF_ROOT}/integrations/ggml/include"
    "${VBUF_ROOT}/integrations/ggml/tools"
    "${GGML_SRC}/include"
    "${GGML_SRC}/src")

target_link_libraries(vbuf_android_chat
    vbuf_region_executor
    vbuf_ml
    log
    android
    c++_shared)
