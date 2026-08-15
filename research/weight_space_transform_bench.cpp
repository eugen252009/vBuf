// Research-only activation transform microbenchmark for 5120-element vectors.
#include <immintrin.h>
#include <x86intrin.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

struct Timing { double seconds; uint64_t cycles; };
template<class F> static Timing best(F fn) {
    Timing value{1e9, UINT64_MAX};
    for (int repeat = 0; repeat < 7; ++repeat) {
        auto start = std::chrono::steady_clock::now(); uint64_t ticks = __rdtsc(); fn();
        uint64_t elapsed_ticks = __rdtsc() - ticks;
        double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        if (elapsed < value.seconds) value = {elapsed, elapsed_ticks};
    }
    return value;
}

static void scale_avx(const float * input, const float * scales, float * output, size_t count) {
    for (size_t i = 0; i < count; i += 8) _mm256_storeu_ps(output+i, _mm256_mul_ps(_mm256_loadu_ps(input+i), _mm256_loadu_ps(scales+i)));
}

static void permute_gather(const float * input, const int * permutation, float * output, size_t count) {
    for (size_t i = 0; i < count; i += 8) {
        __m256i indexes = _mm256_loadu_si256((const __m256i *)(permutation+i));
        _mm256_storeu_ps(output+i, _mm256_i32gather_ps(input, indexes, 4));
    }
}

static void signed_permute_gather(const float * input, const int * permutation, const float * signs, float * output, size_t count) {
    for (size_t i = 0; i < count; i += 8) {
        __m256i indexes = _mm256_loadu_si256((const __m256i *)(permutation+i));
        __m256 value = _mm256_i32gather_ps(input, indexes, 4);
        _mm256_storeu_ps(output+i, _mm256_mul_ps(value, _mm256_loadu_ps(signs+i)));
    }
}

static void hadamard_blocks(const float * input, float * output, size_t count, int block) {
    const float normalization = 1.0f/std::sqrt(float(block));
    for (size_t base = 0; base < count; base += block) {
        for (int i = 0; i < block; ++i) output[base+i] = input[base+i];
        for (int width = 1; width < block; width *= 2) for (int first = 0; first < block; first += 2*width) for (int lane = 0; lane < width; ++lane) {
            float a=output[base+first+lane], b=output[base+first+lane+width];
            output[base+first+lane]=a+b; output[base+first+lane+width]=a-b;
        }
        for (int i = 0; i < block; ++i) output[base+i] *= normalization;
    }
}

int main() {
    constexpr size_t count=5120; constexpr int iterations=131072;
    std::vector<float> input(count), output(count), scales(count), signs(count); std::vector<int> permutation(count);
    for (size_t i=0;i<count;++i) { input[i]=std::sin(float(i)*.01f); scales[i]=std::ldexp(1.0f,int(i%9)-4); signs[i]=(i&1)?-1.f:1.f; permutation[i]=int((i*3137)%count); }
    std::puts("transform,block_size,cycles_per_element,cycles_per_vector,g_elements_per_second,temporary_bytes,memory_read_bytes_per_vector,memory_write_bytes_per_vector,in_place,vectorization");
    volatile float sink=0;
    auto emit=[&](const char * name,int block,auto fn,size_t temporary,const char * in_place,const char * vectorization) {
        Timing timing=best([&]{for(int i=0;i<iterations;++i){fn();sink+=output[i&(count-1)];}});
        double elements=double(count)*iterations;
        std::printf("%s,%d,%.9f,%.9f,%.9f,%zu,%zu,%zu,%s,%s\n",name,block,double(timing.cycles)/elements,double(timing.cycles)/iterations,elements/timing.seconds/1e9,temporary,count*4,count*4,in_place,vectorization);
    };
    emit("permutation",0,[&]{permute_gather(input.data(),permutation.data(),output.data(),count);},count*4,"NO","AVX2 gather");
    emit("signed_permutation",0,[&]{signed_permute_gather(input.data(),permutation.data(),signs.data(),output.data(),count);},count*4,"NO","AVX2 gather+multiply");
    emit("diagonal_scale",0,[&]{scale_avx(input.data(),scales.data(),output.data(),count);},0,"YES if alias-safe implementation","AVX2 multiply");
    for(int block:{8,16,32}) emit("hadamard",block,[&]{hadamard_blocks(input.data(),output.data(),count,block);},count*4,"YES with staged implementation","scalar butterfly reference; SIMD-friendly butterflies");
    return sink==123456.f?1:0;
}
