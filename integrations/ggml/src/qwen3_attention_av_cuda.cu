#include "ggml-cuda/common.cuh"

namespace {

template <typename Value>
__global__ void attention_av_native_kernel(
        const char * value,
        const char * weights,
        const int32_t * positions,
        char * output,
        int64_t n_dim,
        int64_t n_context,
        int64_t n_queries,
        int64_t n_query_heads,
        int64_t n_kv_heads,
        int64_t value_nb0,
        int64_t value_nb1,
        int64_t weights_nb0,
        int64_t weights_nb1,
        int64_t weights_nb2,
        int64_t positions_nb0,
        int64_t output_nb0,
        int64_t output_nb1,
        int64_t output_nb2) {
    const int64_t d = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t h = blockIdx.y;
    const int64_t q = blockIdx.z;
    if (d >= n_dim || h >= n_query_heads || q >= n_queries) return;

    const int64_t kv_h = h / (n_query_heads / n_kv_heads);
    const int32_t last_position = *reinterpret_cast<const int32_t *>(
        reinterpret_cast<const char *>(positions) + q * positions_nb0);
    char * output_element = output + d * output_nb0 + q * output_nb1 + h * output_nb2;
    if (last_position < 0 || last_position >= n_context) {
        *reinterpret_cast<float *>(output_element) = nanf("");
        return;
    }

    float sum = 0.0f;
    for (int64_t p = 0; p <= last_position; ++p) {
        const float weight = *reinterpret_cast<const float *>(
            reinterpret_cast<const char *>(weights) + p * weights_nb0 + q * weights_nb1 + h * weights_nb2);
        const char * value_element = value + d * value_nb0 + (p * n_kv_heads + kv_h) * value_nb1;
        float v;
        if constexpr (std::is_same_v<Value, half>) {
            v = __half2float(*reinterpret_cast<const half *>(value_element));
        } else {
            v = *reinterpret_cast<const float *>(value_element);
        }
        sum += weight * v;
    }
    *reinterpret_cast<float *>(output_element) = sum;
}

} // namespace

void vbuf_ggml_cuda_attention_av(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void vbuf_ggml_cuda_attention_av(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * value = dst->src[0];
    const ggml_tensor * weights = dst->src[1];
    const ggml_tensor * positions = dst->src[2];
    GGML_ASSERT(value->type == GGML_TYPE_F16 || value->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32 && positions->type == GGML_TYPE_I32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    const int64_t n_dim = value->ne[0];
    const int64_t n_context = weights->ne[0];
    const int64_t n_queries = weights->ne[1];
    const int64_t n_query_heads = weights->ne[2];
    GGML_ASSERT(value->ne[1] % n_context == 0);
    const int64_t n_kv_heads = value->ne[1] / n_context;
    GGML_ASSERT(n_query_heads % n_kv_heads == 0);
    GGML_ASSERT(dst->ne[0] == n_dim && dst->ne[1] == n_queries && dst->ne[2] == n_query_heads);
    GGML_ASSERT(positions->ne[0] == n_queries);

    constexpr int block_size = 32;
    const dim3 block_dims(block_size, 1, 1);
    const dim3 grid_dims((n_dim + block_size - 1) / block_size, n_query_heads, n_queries);
    const ggml_cuda_kernel_launch_params launch_params(grid_dims, block_dims, 0, ctx.stream());
    const auto launch = [&](auto value_type_tag) {
        using Value = decltype(value_type_tag);
        ggml_cuda_kernel_launch(attention_av_native_kernel<Value>, launch_params,
            static_cast<const char *>(value->data), static_cast<const char *>(weights->data),
            static_cast<const int32_t *>(positions->data), static_cast<char *>(dst->data),
            n_dim, n_context, n_queries, n_query_heads, n_kv_heads,
            static_cast<int64_t>(value->nb[0]), static_cast<int64_t>(value->nb[1]),
            static_cast<int64_t>(weights->nb[0]), static_cast<int64_t>(weights->nb[1]),
            static_cast<int64_t>(weights->nb[2]), static_cast<int64_t>(positions->nb[0]),
            static_cast<int64_t>(dst->nb[0]), static_cast<int64_t>(dst->nb[1]),
            static_cast<int64_t>(dst->nb[2]));
    };
    if (value->type == GGML_TYPE_F16) launch(half{});
    else launch(float{});
}
