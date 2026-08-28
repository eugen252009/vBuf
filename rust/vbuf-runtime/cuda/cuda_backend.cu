#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <new>

extern "C" {

struct VbufCudaDeviceInfo {
    char name[256];
    int major;
    int minor;
    uint64_t total_bytes;
    uint64_t free_bytes;
};

struct VbufCudaContext;
struct VbufCudaTensor;

static thread_local char vbuf_cuda_error[512];

static int fail(const char *message) {
    std::snprintf(vbuf_cuda_error, sizeof(vbuf_cuda_error), "%s", message);
    return 1;
}

static int cuda_fail(cudaError_t error, const char *where) {
    std::snprintf(vbuf_cuda_error, sizeof(vbuf_cuda_error), "%s: %s", where,
                  cudaGetErrorString(error));
    return 1;
}

static int cublas_fail(cublasStatus_t status, const char *where) {
    std::snprintf(vbuf_cuda_error, sizeof(vbuf_cuda_error), "%s: cublas status %d",
                  where, static_cast<int>(status));
    return 1;
}

struct VbufCudaContext {
    int device;
    cublasHandle_t blas;
};

struct VbufCudaTensor {
    VbufCudaContext *context;
    float *data;
    uint64_t elements;
    int rank;
    uint64_t dimensions[4];
};

static int checked_elements(const uint64_t *dimensions, int rank, uint64_t *result) {
    if (!dimensions || rank < 1 || rank > 4) return fail("invalid CUDA tensor rank");
    uint64_t value = 1;
    for (int i = 0; i < rank; ++i) {
        if (!dimensions[i] || value > UINT64_MAX / dimensions[i])
            return fail("CUDA tensor shape overflows");
        value *= dimensions[i];
    }
    *result = value;
    return 0;
}

static int make_tensor(VbufCudaContext *context, int rank, const uint64_t *dimensions,
                       VbufCudaTensor **result) {
    uint64_t elements = 0;
    if (!context || checked_elements(dimensions, rank, &elements)) return 1;
    auto *tensor = new (std::nothrow) VbufCudaTensor{};
    if (!tensor) return fail("CUDA tensor descriptor allocation failed");
    tensor->context = context;
    tensor->elements = elements;
    tensor->rank = rank;
    std::memcpy(tensor->dimensions, dimensions, sizeof(uint64_t) * rank);
    if (elements > UINT64_MAX / sizeof(float)) {
        delete tensor;
        return fail("CUDA tensor allocation size overflows");
    }
    auto error = cudaMalloc(&tensor->data, elements * sizeof(float));
    if (error != cudaSuccess) {
        delete tensor;
        return cuda_fail(error, "cudaMalloc");
    }
    *result = tensor;
    return 0;
}

__global__ static void add_kernel(const float *left, const float *right, float *out,
                                  uint64_t count) {
    uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) out[index] = left[index] + right[index];
}

__global__ static void mul_kernel(const float *left, const float *right, float *out,
                                  uint64_t count) {
    uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) out[index] = left[index] * right[index];
}

__global__ static void scalar_add_kernel(const float *input, const float *bias, float *out,
                                         uint64_t rows, uint64_t width) {
    uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < rows * width) out[index] = input[index] + bias[index % width];
}

__global__ static void activation_kernel(const float *input, float *out, uint64_t count,
                                         int kind) {
    uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        float x = input[index];
        out[index] = kind == 0 ? x / (1.0f + expf(-x)) : 1.0f / (1.0f + expf(-x));
    }
}

__global__ static void gated_kernel(const float *gate, const float *up, float *out,
                                    uint64_t count) {
    uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        float value = gate[index];
        out[index] = value / (1.0f + expf(-value)) * up[index];
    }
}

__global__ static void rms_norm_kernel(const float *input, const float *weight, float *out,
                                       uint64_t rows, uint64_t width, float epsilon) {
    uint64_t row = blockIdx.x;
    if (row >= rows) return;
    float sum = 0.0f;
    for (uint64_t index = threadIdx.x; index < width; index += blockDim.x) {
        float x = input[row * width + index];
        sum += x * x;
    }
    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (unsigned stride = blockDim.x / 2; stride; stride /= 2) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    float scale = rsqrtf(partial[0] / static_cast<float>(width) + epsilon);
    for (uint64_t index = threadIdx.x; index < width; index += blockDim.x)
        out[row * width + index] = input[row * width + index] * scale * weight[index];
}

__global__ static void rotary_kernel(const float *input, float *out, uint64_t sequence,
                                     uint64_t heads, uint64_t head_dim, uint64_t rotary_dim,
                                     float theta, uint64_t position_start) {
    uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    uint64_t total = sequence * heads * head_dim;
    if (index >= total) return;
    uint64_t component = index % head_dim;
    if (component >= rotary_dim) {
        out[index] = input[index];
        return;
    }
    uint64_t position = (index / head_dim) / heads;
    uint64_t pair = component % (rotary_dim / 2);
    uint64_t base = (index / head_dim) * head_dim;
    float inverse = powf(theta, -2.0f * static_cast<float>(pair) /
                                  static_cast<float>(rotary_dim));
    float angle = static_cast<float>(position_start + position) * inverse;
    float cosine = cosf(angle), sine = sinf(angle);
    float left = input[base + pair], right = input[base + rotary_dim / 2 + pair];
    out[index] = component == pair ? left * cosine - right * sine
                                    : component == rotary_dim / 2 + pair
                                          ? right * cosine + left * sine
                                          : input[index];
}

__global__ static void attention_kernel(const float *query, const float *key, const float *value,
                                        float *out, uint64_t sequence, uint64_t q_heads,
                                        uint64_t kv_heads, uint64_t dimension, float scale) {
    uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    uint64_t total = sequence * q_heads * dimension;
    if (index >= total) return;
    uint64_t component = index % dimension;
    uint64_t head_position = index / dimension;
    uint64_t query_head = head_position % q_heads;
    uint64_t query_position = head_position / q_heads;
    uint64_t group = q_heads / kv_heads;
    uint64_t kv_head = query_head / group;
    float maximum = -INFINITY;
    for (uint64_t key_position = 0; key_position <= query_position; ++key_position) {
        float dot = 0.0f;
        for (uint64_t i = 0; i < dimension; ++i) {
            uint64_t q = (query_position * q_heads + query_head) * dimension + i;
            uint64_t k = (key_position * kv_heads + kv_head) * dimension + i;
            dot += query[q] * key[k];
        }
        maximum = fmaxf(maximum, dot * scale);
    }
    float denominator = 0.0f;
    for (uint64_t key_position = 0; key_position <= query_position; ++key_position) {
        float dot = 0.0f;
        for (uint64_t i = 0; i < dimension; ++i) {
            uint64_t q = (query_position * q_heads + query_head) * dimension + i;
            uint64_t k = (key_position * kv_heads + kv_head) * dimension + i;
            dot += query[q] * key[k];
        }
        denominator += expf(dot * scale - maximum);
    }
    float result = 0.0f;
    for (uint64_t key_position = 0; key_position <= query_position; ++key_position) {
        float dot = 0.0f;
        for (uint64_t i = 0; i < dimension; ++i) {
            uint64_t q = (query_position * q_heads + query_head) * dimension + i;
            uint64_t k = (key_position * kv_heads + kv_head) * dimension + i;
            dot += query[q] * key[k];
        }
        uint64_t v = (key_position * kv_heads + kv_head) * dimension + component;
        result += expf(dot * scale - maximum) / denominator * value[v];
    }
    out[index] = result;
}

__global__ static void expert_mask_kernel(float *data, const uint32_t *ids, const float *weights,
                                          uint64_t tokens, uint64_t width, uint64_t top_k,
                                          uint32_t expert) {
    uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= tokens * width) return;
    uint64_t token = index / width;
    float weight = 0.0f;
    for (uint64_t rank = 0; rank < top_k; ++rank)
        if (ids[token * top_k + rank] == expert) weight = weights[token * top_k + rank];
    data[index] *= weight;
}

int vbuf_cuda_device_count() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) return 0;
    return count;
}

int vbuf_cuda_device_info(int index, VbufCudaDeviceInfo *info) {
    if (!info) return fail("device info output is null");
    cudaDeviceProp properties{};
    auto error = cudaGetDeviceProperties(&properties, index);
    if (error != cudaSuccess) return cuda_fail(error, "cudaGetDeviceProperties");
    uint64_t free_bytes = 0, total_bytes = 0;
    error = cudaSetDevice(index);
    if (error != cudaSuccess) return cuda_fail(error, "cudaSetDevice");
    error = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (error != cudaSuccess) return cuda_fail(error, "cudaMemGetInfo");
    std::snprintf(info->name, sizeof(info->name), "%s", properties.name);
    info->major = properties.major;
    info->minor = properties.minor;
    info->free_bytes = free_bytes;
    info->total_bytes = total_bytes;
    return 0;
}

const char *vbuf_cuda_last_error() { return vbuf_cuda_error; }

int vbuf_cuda_synchronize(VbufCudaContext *context) {
    if (!context) return fail("CUDA synchronization context is null");
    auto error = cudaSetDevice(context->device);
    if (error == cudaSuccess) error = cudaDeviceSynchronize();
    return error == cudaSuccess ? 0 : cuda_fail(error, "cudaDeviceSynchronize");
}

int vbuf_cuda_memory_info(VbufCudaContext *context, uint64_t *free_bytes, uint64_t *total_bytes) {
    if (!context || !free_bytes || !total_bytes) return fail("CUDA memory info argument is null");
    auto error = cudaSetDevice(context->device);
    if (error == cudaSuccess) error = cudaMemGetInfo(free_bytes, total_bytes);
    return error == cudaSuccess ? 0 : cuda_fail(error, "cudaMemGetInfo");
}

int vbuf_cuda_context_create(int device, VbufCudaContext **result) {
    if (!result) return fail("CUDA context output is null");
    auto error = cudaSetDevice(device);
    if (error != cudaSuccess) return cuda_fail(error, "cudaSetDevice");
    auto *context = new (std::nothrow) VbufCudaContext{};
    if (!context) return fail("CUDA context allocation failed");
    context->device = device;
    if (cublasCreate(&context->blas) != CUBLAS_STATUS_SUCCESS) {
        delete context;
        return fail("cublasCreate failed");
    }
    if (cublasSetMathMode(context->blas, CUBLAS_PEDANTIC_MATH) != CUBLAS_STATUS_SUCCESS) {
        cublasDestroy(context->blas);
        delete context;
        return fail("cublasSetMathMode(CUBLAS_PEDANTIC_MATH) failed");
    }
    *result = context;
    return 0;
}

int vbuf_cuda_context_destroy(VbufCudaContext *context) {
    if (!context) return 0;
    cudaSetDevice(context->device);
    cublasDestroy(context->blas);
    delete context;
    return 0;
}

int vbuf_cuda_tensor_destroy(VbufCudaTensor *tensor) {
    if (!tensor) return 0;
    cudaSetDevice(tensor->context->device);
    auto error = cudaFree(tensor->data);
    delete tensor;
    return error == cudaSuccess ? 0 : cuda_fail(error, "cudaFree");
}

int vbuf_cuda_tensor_upload(VbufCudaContext *context, const float *values, uint64_t count,
                            int rank, const uint64_t *dimensions, VbufCudaTensor **result) {
    if (!values || !result) return fail("CUDA upload argument is null");
    uint64_t elements = 0;
    if (checked_elements(dimensions, rank, &elements) || elements != count) return 1;
    if (make_tensor(context, rank, dimensions, result)) return 1;
    auto error = cudaMemcpy((*result)->data, values, count * sizeof(float), cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        vbuf_cuda_tensor_destroy(*result);
        *result = nullptr;
        return cuda_fail(error, "cudaMemcpyHostToDevice");
    }
    return 0;
}

int vbuf_cuda_tensor_download(VbufCudaTensor *tensor, float *values, uint64_t count) {
    if (!tensor || !values || count != tensor->elements) return fail("CUDA download geometry is invalid");
    auto error = cudaMemcpy(values, tensor->data, count * sizeof(float), cudaMemcpyDeviceToHost);
    return error == cudaSuccess ? 0 : cuda_fail(error, "cudaMemcpyDeviceToHost");
}

static int allocate_like(VbufCudaTensor *input, VbufCudaTensor **result) {
    return make_tensor(input->context, input->rank, input->dimensions, result);
}

int vbuf_cuda_rms_norm(VbufCudaTensor *input, VbufCudaTensor *weight, float epsilon,
                       VbufCudaTensor **result) {
    if (!input || !weight || input->rank < 1 || weight->rank != 1 ||
        weight->elements != input->dimensions[input->rank - 1]) return fail("CUDA RMSNorm geometry is invalid");
    if (allocate_like(input, result)) return 1;
    uint64_t width = input->dimensions[input->rank - 1];
    uint64_t rows = input->elements / width;
    rms_norm_kernel<<<static_cast<unsigned>(rows), 256>>>(input->data, weight->data, (*result)->data,
                                                          rows, width, epsilon);
    auto error = cudaGetLastError();
    if (error != cudaSuccess) {
        vbuf_cuda_tensor_destroy(*result);
        *result = nullptr;
        return cuda_fail(error, "RMSNorm kernel");
    }
    return 0;
}

int vbuf_cuda_matmul(VbufCudaTensor *input, VbufCudaTensor *weight, VbufCudaTensor **result) {
    if (!input || !weight || weight->rank != 2 || input->rank < 1 ||
        weight->dimensions[1] != input->dimensions[input->rank - 1]) return fail("CUDA MatMul geometry is invalid");
    uint64_t width = input->dimensions[input->rank - 1];
    uint64_t rows = input->elements / width;
    uint64_t output_width = weight->dimensions[0];
    uint64_t dimensions[4]{};
    for (int i = 0; i < input->rank; ++i) dimensions[i] = input->dimensions[i];
    dimensions[input->rank - 1] = output_width;
    if (make_tensor(input->context, input->rank, dimensions, result)) return 1;
    const float alpha = 1.0f, beta = 0.0f;
    auto status = cublasSgemm(input->context->blas, CUBLAS_OP_T, CUBLAS_OP_N,
                              static_cast<int>(output_width), static_cast<int>(rows),
                              static_cast<int>(width), &alpha, weight->data,
                              static_cast<int>(width), input->data, static_cast<int>(width),
                              &beta, (*result)->data, static_cast<int>(output_width));
    if (status != CUBLAS_STATUS_SUCCESS) {
        vbuf_cuda_tensor_destroy(*result);
        *result = nullptr;
        return cublas_fail(status, "cublasSgemm");
    }
    return 0;
}

int vbuf_cuda_add(VbufCudaTensor *left, VbufCudaTensor *right, VbufCudaTensor **result) {
    if (!left || !right || left->elements != right->elements || left->rank != right->rank ||
        std::memcmp(left->dimensions, right->dimensions, sizeof(uint64_t) * left->rank) != 0)
        return fail("CUDA add geometry is invalid");
    if (allocate_like(left, result)) return 1;
    add_kernel<<<static_cast<unsigned>((left->elements + 255) / 256), 256>>>(left->data, right->data,
                                                                               (*result)->data, left->elements);
    auto error = cudaGetLastError();
    if (error != cudaSuccess) {
        vbuf_cuda_tensor_destroy(*result);
        *result = nullptr;
        return cuda_fail(error, "add kernel");
    }
    return 0;
}

int vbuf_cuda_mul(VbufCudaTensor *left, VbufCudaTensor *right, VbufCudaTensor **result) {
    if (!left || !right || left->elements != right->elements || left->rank != right->rank ||
        std::memcmp(left->dimensions, right->dimensions, sizeof(uint64_t) * left->rank) != 0)
        return fail("CUDA multiply geometry is invalid");
    if (allocate_like(left, result)) return 1;
    mul_kernel<<<static_cast<unsigned>((left->elements + 255) / 256), 256>>>(left->data, right->data,
                                                                               (*result)->data, left->elements);
    auto error = cudaGetLastError();
    if (error != cudaSuccess) {
        vbuf_cuda_tensor_destroy(*result);
        *result = nullptr;
        return cuda_fail(error, "multiply kernel");
    }
    return 0;
}

int vbuf_cuda_bias_add(VbufCudaTensor *input, VbufCudaTensor *bias, VbufCudaTensor **result) {
    if (!input || !bias || bias->rank != 1 || bias->elements != input->dimensions[input->rank - 1])
        return fail("CUDA bias geometry is invalid");
    if (allocate_like(input, result)) return 1;
    uint64_t width = input->dimensions[input->rank - 1];
    scalar_add_kernel<<<static_cast<unsigned>((input->elements + 255) / 256), 256>>>(input->data,
                                                                                       bias->data, (*result)->data,
                                                                                       input->elements / width, width);
    auto error = cudaGetLastError();
    if (error != cudaSuccess) {
        vbuf_cuda_tensor_destroy(*result);
        *result = nullptr;
        return cuda_fail(error, "bias kernel");
    }
    return 0;
}

int vbuf_cuda_activation(VbufCudaTensor *input, int kind, VbufCudaTensor **result) {
    if (!input) return fail("CUDA activation input is null");
    if (allocate_like(input, result)) return 1;
    activation_kernel<<<static_cast<unsigned>((input->elements + 255) / 256), 256>>>(input->data,
                                                                                       (*result)->data, input->elements, kind);
    auto error = cudaGetLastError();
    if (error != cudaSuccess) {
        vbuf_cuda_tensor_destroy(*result);
        *result = nullptr;
        return cuda_fail(error, "activation kernel");
    }
    return 0;
}

int vbuf_cuda_reshape(VbufCudaTensor *input, int rank, const uint64_t *dimensions,
                      VbufCudaTensor **result) {
    uint64_t elements = 0;
    if (!input || checked_elements(dimensions, rank, &elements) || elements != input->elements) return fail("CUDA reshape geometry is invalid");
    if (make_tensor(input->context, rank, dimensions, result)) return 1;
    auto error = cudaMemcpy((*result)->data, input->data, elements * sizeof(float), cudaMemcpyDeviceToDevice);
    if (error != cudaSuccess) {
        vbuf_cuda_tensor_destroy(*result);
        *result = nullptr;
        return cuda_fail(error, "reshape copy");
    }
    return 0;
}

int vbuf_cuda_rotary(VbufCudaTensor *input, uint64_t heads, uint64_t head_dim,
                     uint64_t rotary_dim, float theta, uint64_t position_start,
                     VbufCudaTensor **result) {
    if (!input || input->rank != 4 || input->dimensions[2] != heads || input->dimensions[3] != head_dim)
        return fail("CUDA rotary geometry is invalid");
    if (allocate_like(input, result)) return 1;
    rotary_kernel<<<static_cast<unsigned>((input->elements + 255) / 256), 256>>>(input->data, (*result)->data,
                                                                                   input->dimensions[1], heads, head_dim,
                                                                                   rotary_dim, theta, position_start);
    auto error = cudaGetLastError();
    if (error != cudaSuccess) {
        vbuf_cuda_tensor_destroy(*result);
        *result = nullptr;
        return cuda_fail(error, "rotary kernel");
    }
    return 0;
}

int vbuf_cuda_attention(VbufCudaTensor *query, VbufCudaTensor *key, VbufCudaTensor *value,
                        uint64_t q_heads, uint64_t kv_heads, uint64_t head_dim, float scale,
                        VbufCudaTensor **result) {
    if (!query || !key || !value || query->rank != 4 || key->rank != 4 || value->rank != 4 ||
        query->dimensions[1] != key->dimensions[1] || key->dimensions[1] != value->dimensions[1] ||
        query->dimensions[2] != q_heads || key->dimensions[2] != kv_heads ||
        value->dimensions[2] != kv_heads || query->dimensions[3] != head_dim ||
        key->dimensions[3] != head_dim || value->dimensions[3] != head_dim)
        return fail("CUDA attention geometry is invalid");
    if (allocate_like(query, result)) return 1;
    uint64_t sequence = query->dimensions[1];
    attention_kernel<<<static_cast<unsigned>((query->elements + 255) / 256), 256>>>(query->data, key->data,
                                                                                      value->data, (*result)->data,
                                                                                      sequence, q_heads, kv_heads,
                                                                                      head_dim, scale);
    auto error = cudaGetLastError();
    if (error != cudaSuccess) {
        vbuf_cuda_tensor_destroy(*result);
        *result = nullptr;
        return cuda_fail(error, "attention kernel");
    }
    return 0;
}

int vbuf_cuda_expert_dispatch(VbufCudaTensor *input, VbufCudaTensor *gate, VbufCudaTensor *up,
                              VbufCudaTensor *down, const uint32_t *ids, const float *weights,
                              uint64_t token_count, uint64_t top_k, uint32_t expert,
                              VbufCudaTensor **result) {
    if (!input || !gate || !up || !down || !ids || !weights || input->rank != 3 || gate->rank != 2 ||
        up->rank != 2 || down->rank != 2 || input->dimensions[0] * input->dimensions[1] != token_count ||
        gate->dimensions[1] != input->dimensions[2] || up->dimensions[1] != input->dimensions[2] ||
        up->dimensions[0] != gate->dimensions[0] || down->dimensions[1] != gate->dimensions[0] ||
        down->dimensions[0] != input->dimensions[2]) return fail("CUDA expert geometry is invalid");
    VbufCudaTensor *gate_output = nullptr, *up_output = nullptr, *activated = nullptr;
    VbufCudaTensor *activated_output = nullptr, *output = nullptr;
    uint64_t intermediate_dims[2] = {token_count, gate->dimensions[0]};
    uint64_t output_dims[3] = {input->dimensions[0], input->dimensions[1], input->dimensions[2]};
    uint32_t *device_ids = nullptr;
    float *device_weights = nullptr;
    cudaError_t error = cudaSuccess;
    if (make_tensor(input->context, 2, intermediate_dims, &gate_output) ||
        make_tensor(input->context, 2, intermediate_dims, &up_output)) goto expert_error;
    {
        const float alpha = 1.0f, beta = 0.0f;
        auto status = cublasSgemm(input->context->blas, CUBLAS_OP_T, CUBLAS_OP_N,
                                  static_cast<int>(gate->dimensions[0]), static_cast<int>(token_count),
                                  static_cast<int>(input->dimensions[2]), &alpha, gate->data,
                                  static_cast<int>(input->dimensions[2]), input->data,
                                  static_cast<int>(input->dimensions[2]), &beta, gate_output->data,
                                  static_cast<int>(gate->dimensions[0]));
        if (status != CUBLAS_STATUS_SUCCESS) { cublas_fail(status, "expert gate"); goto expert_error; }
        status = cublasSgemm(input->context->blas, CUBLAS_OP_T, CUBLAS_OP_N,
                             static_cast<int>(up->dimensions[0]), static_cast<int>(token_count),
                             static_cast<int>(input->dimensions[2]), &alpha, up->data,
                             static_cast<int>(input->dimensions[2]), input->data,
                             static_cast<int>(input->dimensions[2]), &beta, up_output->data,
                             static_cast<int>(up->dimensions[0]));
        if (status != CUBLAS_STATUS_SUCCESS) { cublas_fail(status, "expert up"); goto expert_error; }
    }
    if (make_tensor(input->context, 2, intermediate_dims, &activated_output)) goto expert_error;
    gated_kernel<<<static_cast<unsigned>((gate_output->elements + 255) / 256), 256>>>(
        gate_output->data, up_output->data, activated_output->data, gate_output->elements);
    error = cudaGetLastError();
    if (error != cudaSuccess) { cuda_fail(error, "expert gated activation"); goto expert_error; }
    vbuf_cuda_tensor_destroy(gate_output); gate_output = nullptr;
    vbuf_cuda_tensor_destroy(up_output); up_output = nullptr;
    if (make_tensor(input->context, 3, output_dims, &output)) goto expert_error;
    {
        const float alpha = 1.0f, beta = 0.0f;
        auto status = cublasSgemm(input->context->blas, CUBLAS_OP_T, CUBLAS_OP_N,
                                  static_cast<int>(down->dimensions[0]), static_cast<int>(token_count),
                                  static_cast<int>(down->dimensions[1]), &alpha, down->data,
                                  static_cast<int>(down->dimensions[1]), activated_output->data,
                                  static_cast<int>(down->dimensions[1]), &beta, output->data,
                                  static_cast<int>(down->dimensions[0]));
        if (status != CUBLAS_STATUS_SUCCESS) { cublas_fail(status, "expert down"); goto expert_error; }
    }
    error = cudaMalloc(&device_ids, token_count * top_k * sizeof(uint32_t));
    if (error != cudaSuccess) { cuda_fail(error, "expert IDs"); goto expert_error; }
    error = cudaMalloc(&device_weights, token_count * top_k * sizeof(float));
    if (error != cudaSuccess) { cuda_fail(error, "expert weights"); goto expert_error; }
    error = cudaMemcpy(device_ids, ids, token_count * top_k * sizeof(uint32_t), cudaMemcpyHostToDevice);
    if (error == cudaSuccess) error = cudaMemcpy(device_weights, weights, token_count * top_k * sizeof(float), cudaMemcpyHostToDevice);
    if (error != cudaSuccess) { cuda_fail(error, "expert selection upload"); goto expert_error; }
    expert_mask_kernel<<<static_cast<unsigned>((output->elements + 255) / 256), 256>>>(output->data, device_ids,
                                                                                         device_weights, token_count,
                                                                                         input->dimensions[2], top_k, expert);
    error = cudaGetLastError();
    cudaFree(device_ids); cudaFree(device_weights);
    device_ids = nullptr;
    device_weights = nullptr;
    if (error != cudaSuccess) { cuda_fail(error, "expert selection kernel"); goto expert_error; }
    vbuf_cuda_tensor_destroy(activated_output);
    activated_output = nullptr;
    *result = output;
    return 0;
expert_error:
    if (gate_output) vbuf_cuda_tensor_destroy(gate_output);
    if (up_output) vbuf_cuda_tensor_destroy(up_output);
    if (activated) vbuf_cuda_tensor_destroy(activated);
    if (activated_output) vbuf_cuda_tensor_destroy(activated_output);
    if (output) vbuf_cuda_tensor_destroy(output);
    if (device_ids) cudaFree(device_ids);
    if (device_weights) cudaFree(device_weights);
    return 1;
}

}
