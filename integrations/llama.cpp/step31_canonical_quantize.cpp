#include "ggml.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

struct Format { const char * name; ggml_type type; };

static std::vector<float> read_f32(const char * path, size_t count) {
    std::vector<float> v(count); std::ifstream in(path, std::ios::binary); if (!in) throw std::runtime_error("input open failed");
    in.read(reinterpret_cast<char *>(v.data()), (std::streamsize) (count*sizeof(float))); if ((size_t) in.gcount() != count*sizeof(float)) throw std::runtime_error("input size mismatch");
    char extra; if (in.read(&extra,1)) throw std::runtime_error("input has trailing bytes"); return v;
}
static void write_raw(const std::string & path, const void * data, size_t size) { std::ofstream out(path,std::ios::binary); if (!out) throw std::runtime_error("output open failed"); out.write((const char *)data,(std::streamsize)size); if (!out) throw std::runtime_error("output write failed"); }

int main(int argc, char ** argv) {
    if (argc != 7) return 2;
    const char * input=argv[1]; const char * imatrix_path=argv[2]; const std::string outdir=argv[3]; const int64_t rows=std::stoll(argv[4]); const int64_t cols=std::stoll(argv[5]); const char * wanted=argv[6];
    const Format formats[]={{"Q2_K",GGML_TYPE_Q2_K},{"Q3_K",GGML_TYPE_Q3_K},{"Q4_0",GGML_TYPE_Q4_0},{"Q4_K",GGML_TYPE_Q4_K},{"IQ2_XXS",GGML_TYPE_IQ2_XXS},{"IQ2_XS",GGML_TYPE_IQ2_XS},{"IQ2_S",GGML_TYPE_IQ2_S},{"IQ3_XXS",GGML_TYPE_IQ3_XXS},{"IQ3_S",GGML_TYPE_IQ3_S},{"IQ4_NL",GGML_TYPE_IQ4_NL},{"IQ4_XS",GGML_TYPE_IQ4_XS}};
    try {
        auto src=read_f32(input,(size_t)rows*cols); std::vector<float> imatrix;
        if (std::strcmp(imatrix_path,"-")!=0) imatrix=read_f32(imatrix_path,(size_t)cols);
        for (const auto & f:formats) if (std::strcmp(wanted,f.name)==0) {
            const auto * traits=ggml_get_type_traits(f.type); const bool requires=ggml_quantize_requires_imatrix(f.type); const size_t row_size=ggml_row_size(f.type,cols); const size_t bytes=row_size*(size_t)rows;
            if (cols%traits->blck_size) throw std::runtime_error("row tail is not a complete canonical block");
            if (requires && imatrix.empty()) { std::printf("{\"format\":\"%s\",\"status\":\"BLOCKED\",\"reason\":\"canonical quantizer requires importance matrix\",\"block_size\":%lld,\"bytes_per_block\":%zu}\n",f.name,(long long)traits->blck_size,traits->type_size); return 0; }
            std::vector<unsigned char> q(bytes); const size_t written=ggml_quantize_chunk(f.type,src.data(),q.data(),0,rows,cols,imatrix.empty()?nullptr:imatrix.data());
            if (written!=bytes) throw std::runtime_error("canonical byte count mismatch");
            std::vector<float> restored((size_t)rows*cols);
            if (!traits->to_float) throw std::runtime_error("canonical dequantizer unavailable");
            for (int64_t r=0;r<rows;++r) traits->to_float(q.data()+row_size*(size_t)r,restored.data()+cols*(size_t)r,cols);
            write_raw(outdir+"/"+f.name+".bin",q.data(),q.size()); write_raw(outdir+"/"+f.name+".f32",restored.data(),restored.size()*sizeof(float));
            std::printf("{\"format\":\"%s\",\"status\":\"TESTED\",\"block_size\":%lld,\"bytes_per_block\":%zu,\"row_size\":%zu,\"rows\":%lld,\"cols\":%lld,\"block_count\":%zu,\"serialized_bytes\":%zu,\"true_bpw\":%.9f,\"requires_imatrix\":%s,\"imatrix\":\"%s\",\"quantizer\":\"ggml_quantize_chunk\",\"dequantizer\":\"ggml_type_traits.to_float\"}\n",f.name,(long long)traits->blck_size,traits->type_size,row_size,(long long)rows,(long long)cols,(size_t)rows*cols/traits->blck_size,bytes,bytes*8.0/(rows*cols),requires?"true":"false",imatrix.empty()?"none":"functional-validation mean-square activation"); ggml_quantize_free(); return 0;
        }
        throw std::runtime_error("unknown format");
    } catch(const std::exception & e) { std::fprintf(stderr,"%s\n",e.what()); return 3; }
}
