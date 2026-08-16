cmake_minimum_required(VERSION 3.18)

project(vbuf_ggml_substrate LANGUAGES C CXX)

if(POLICY CMP0169)
    # Populate is intentional here: the upstream repository root is fetched,
    # but its CMake project is added only after the ggml options are set.
    cmake_policy(SET CMP0169 OLD)
endif()

set(VBUF_GGML_COMMIT "2d191b5dee1a591c41ee8a653ce42bfcd9c8716d" CACHE STRING
    "Immutable upstream ggml revision")
set(VBUF_GGML_SOURCE_DIR "" CACHE PATH
    "Optional pre-fetched ggml source directory for offline builds")
set(VBUF_ML_LIBRARY "" CACHE FILEPATH
    "Existing libvbuf_ml.so to expose to native targets")
set(VBUF_ML_INCLUDE_DIR "" CACHE PATH
    "Directory containing vBuf-ML native headers, if needed")
option(VBUF_ENABLE_CUDA "Build the pinned ggml CUDA backend" OFF)
option(VBUF_BUILD_PROBES "Build disposable substrate qualification probes" ON)

include(FetchContent)

if(VBUF_GGML_SOURCE_DIR)
    set(GGML_SOURCE_DIR "${VBUF_GGML_SOURCE_DIR}")
    find_package(Git REQUIRED)
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" rev-parse HEAD
        WORKING_DIRECTORY "${GGML_SOURCE_DIR}"
        OUTPUT_VARIABLE VBUF_GGML_LOCAL_COMMIT
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE VBUF_GGML_LOCAL_COMMIT_RESULT)
    if(NOT VBUF_GGML_LOCAL_COMMIT_RESULT EQUAL 0 OR
       NOT VBUF_GGML_LOCAL_COMMIT STREQUAL VBUF_GGML_COMMIT)
        message(FATAL_ERROR
            "VBUF_GGML_SOURCE_DIR must be ggml commit ${VBUF_GGML_COMMIT}; "
            "found ${VBUF_GGML_LOCAL_COMMIT}")
    endif()
else()
    FetchContent_Declare(ggml_source
        GIT_REPOSITORY https://github.com/ggml-org/ggml.git
        GIT_TAG "${VBUF_GGML_COMMIT}"
        GIT_SHALLOW FALSE)
    FetchContent_GetProperties(ggml_source)
    if(NOT ggml_source_POPULATED)
        FetchContent_Populate(ggml_source)
    endif()
    set(GGML_SOURCE_DIR "${ggml_source_SOURCE_DIR}")
endif()

if(NOT EXISTS "${GGML_SOURCE_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "ggml source is missing: ${GGML_SOURCE_DIR}")
endif()

# Keep ownership explicit: CMake consumes an already-built Rust library and
# never invokes Cargo.
if(VBUF_ML_LIBRARY)
    if(NOT EXISTS "${VBUF_ML_LIBRARY}")
        message(FATAL_ERROR "VBUF_ML_LIBRARY does not exist: ${VBUF_ML_LIBRARY}")
    endif()
    add_library(vbuf_ml SHARED IMPORTED GLOBAL)
    set_target_properties(vbuf_ml PROPERTIES
        IMPORTED_LOCATION "${VBUF_ML_LIBRARY}")
    if(VBUF_ML_INCLUDE_DIR)
        set_property(TARGET vbuf_ml PROPERTY
            INTERFACE_INCLUDE_DIRECTORIES "${VBUF_ML_INCLUDE_DIR}")
    endif()
endif()

# These are the only ggml options needed by the first substrate. The cache
# assignments precede add_subdirectory so the pinned project consumes them.
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
set(GGML_CPU ON CACHE BOOL "" FORCE)
set(GGML_CPU_REPACK OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_BLAS OFF CACHE BOOL "" FORCE)
set(GGML_ACCELERATE OFF CACHE BOOL "" FORCE)
set(GGML_LLAMAFILE OFF CACHE BOOL "" FORCE)
set(GGML_CUDA ${VBUF_ENABLE_CUDA} CACHE BOOL "" FORCE)

add_subdirectory("${GGML_SOURCE_DIR}" "${CMAKE_BINARY_DIR}/ggml" EXCLUDE_FROM_ALL)

add_library(vbuf_region_executor STATIC
    src/vbuf_region_executor.cpp)
target_compile_features(vbuf_region_executor PUBLIC cxx_std_17)
target_include_directories(vbuf_region_executor
    PUBLIC include
    PRIVATE "${GGML_SOURCE_DIR}/include")
target_link_libraries(vbuf_region_executor PUBLIC ggml)
if(TARGET vbuf_ml)
    target_link_libraries(vbuf_region_executor PUBLIC vbuf_ml)
endif()

if(VBUF_BUILD_PROBES)
    add_executable(vbuf_ggml_smoke tests/ggml_smoke.cpp)
    target_compile_features(vbuf_ggml_smoke PRIVATE cxx_std_17)
    target_link_libraries(vbuf_ggml_smoke PRIVATE vbuf_region_executor)

    add_executable(vbuf_borrowed_cpu_buffer_probe
        tests/borrowed_cpu_buffer.cpp)
    target_compile_features(vbuf_borrowed_cpu_buffer_probe PRIVATE cxx_std_17)
    target_link_libraries(vbuf_borrowed_cpu_buffer_probe PRIVATE vbuf_region_executor)

    enable_testing()
    add_test(NAME vbuf_ggml_smoke COMMAND vbuf_ggml_smoke)
    add_test(NAME vbuf_borrowed_cpu_buffer_probe
        COMMAND vbuf_borrowed_cpu_buffer_probe)
endif()

message(STATUS "vBuf native substrate")
message(STATUS "  ggml source: ${GGML_SOURCE_DIR}")
message(STATUS "  ggml commit: ${VBUF_GGML_COMMIT}")
message(STATUS "  CUDA: ${VBUF_ENABLE_CUDA}")
message(STATUS "  CPU_REPACK: ${GGML_CPU_REPACK}")
message(STATUS "  vBuf-ML library: ${VBUF_ML_LIBRARY}")
