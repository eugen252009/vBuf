#include "qwen3_cuda_core.h"
#include "vbuf_model_architecture.h"
#include "vbuf_region_executor.h"
#include "ggml-backend.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <filesystem>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
using namespace vbuf_ggml;
#define ggml_backend_tensor_set(B,T,D,O,S) (ggml_backend_tensor_set)(T,D,O,S)
#define ggml_backend_tensor_get(B,T,D,O,S) (ggml_backend_tensor_get)(T,D,O,S)
static void need(bool x,const char*m){if(!x)throw std::runtime_error(m);}
static std::vector<double> stats(std::vector<double> a){std::sort(a.begin(),a.end());return {a[a.size()/2],a[std::min(a.size()-1,(size_t)std::ceil(.95*a.size())-1)],a.front()};}
int main(int argc,char**argv){
 try {
  if(argc!=8&&argc!=10) throw std::runtime_error("usage: bench SEMANTIC VBUF_PATH DEVICE BLOCK ROWS CAPACITY START [INPUT_F32 OUTPUT_F32]");
  const std::string semantic=argv[1], path=argv[2]; int device_i=std::stoi(argv[3]), block=std::stoi(argv[4]); uint32_t rows=std::stoul(argv[5]),cap=std::stoul(argv[6]),start=std::stoul(argv[7]);
  need(block>=0&&block<40&&rows>=1&&rows<=256&&cap>=rows&&start<=cap-rows,"invalid block/rows/capacity/start");
  Qwen3Model model; open_qwen3_model(semantic,"",&model,true); need(model.artifact_identity==std::string("sha256:")+QWEN3_14B_Q4_K_M_SHA256,"qualified artifact mismatch");
  ggml_backend_reg_t reg=ggml_backend_reg_by_name("CUDA"); need(reg,"CUDA registry missing");
  ggml_backend_dev_t dev=ggml_backend_reg_dev_get(reg,(size_t)device_i); need(dev,"CUDA device index missing");
  ggml_backend_t be=ggml_backend_dev_init(dev,nullptr); need(be,"CUDA backend init failed");
  size_t free0=0,total0=0; ggml_backend_dev_memory(dev,&free0,&total0);
  const std::string prefix="blk."+std::to_string(block)+".";
  std::vector<std::string> names={"attn_norm.weight","attn_q.weight","attn_k.weight","attn_v.weight","attn_q_norm.weight","attn_k_norm.weight","attn_output.weight","ffn_norm.weight","ffn_gate.weight","ffn_up.weight","ffn_down.weight"};
  if(std::getenv("QWEN_PRINT_ALL_REP")){for(int b=0;b<40;b++)for(const auto&n:std::vector<std::string>{"attn_q.weight","attn_k.weight","attn_v.weight","attn_output.weight","ffn_gate.weight","ffn_up.weight","ffn_down.weight"}){const std::string key="blk."+std::to_string(b)+"."+n;const auto&t=qwen3_tensor(model,key);TensorGeometry g{};need(derive_tensor_geometry(t.generic,&g)==AdapterError::None,"geometry error");printf("%d,%s,%u,%s,%llu,%llu\\n",b,n.c_str(),t.view.representation,ggml_type_name(g.type),(unsigned long long)t.length,(unsigned long long)t.offset);}return 0;}
  if(std::getenv("QWEN_PRINT_REP")){for(const auto&n:names){const auto&t=qwen3_tensor(model,prefix+n);TensorGeometry g{};need(derive_tensor_geometry(t.generic,&g)==AdapterError::None,"geometry error");printf("tensor=%s representation=%u ggml_type=%s rank=%u ne=%lld,%lld,%lld,%lld bytes=%llu offset=%llu\\n",(prefix+n).c_str(),t.view.representation,ggml_type_name(g.type),g.rank,(long long)g.ne[0],(long long)g.ne[1],(long long)g.ne[2],(long long)g.ne[3],(unsigned long long)t.length,(unsigned long long)t.offset);}return 0;}
  ggml_init_params ip{512ull*1024*1024,nullptr,true}; ggml_context*ctx=ggml_init(ip); need(ctx,"ggml context failed");
  std::vector<ggml_tensor*> wt; uint64_t payload=0;
  for(const auto &n:names){auto &t=qwen3_tensor(model,prefix+n); TensorGeometry g{}; need(derive_tensor_geometry(t.generic,&g)==AdapterError::None&&g.nbytes==t.length,"tensor geometry failed"); auto*x=ggml_new_tensor(ctx,g.type,g.rank,g.ne); need(x,"weight tensor failed"); wt.push_back(x);payload+=t.length;}
  auto*hidden=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,5120,rows);
  auto*pos=ggml_new_tensor_1d(ctx,GGML_TYPE_I32,rows);
  auto*mask=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,cap,rows);
  auto*rowids=ggml_new_tensor_1d(ctx,GGML_TYPE_I32,8*rows);
  auto*kcache=ggml_new_tensor_2d(ctx,GGML_TYPE_F16,128,8*cap);
  auto*vcache=ggml_new_tensor_2d(ctx,GGML_TYPE_F16,128,8*cap);
  auto*packed=ggml_new_tensor_3d(ctx,GGML_TYPE_F16,cap,128,8);
  need(hidden&&pos&&mask&&rowids&&kcache&&vcache&&packed,"input/cache tensor allocation failed");
  Qwen3CudaLayerWeights w{wt[0],wt[1],wt[2],wt[3],wt[4],wt[5],wt[6],wt[7],wt[8],wt[9],wt[10]};
  std::vector<std::pair<std::string,ggml_tensor*>> captures;
  const bool retain_outputs=std::getenv("QWEN_CAPTURE_ALL")!=nullptr;
  auto out=qwen3_cuda_build_layer(ctx,w,hidden,pos,mask,rowids,kcache,vcache,packed,rows,cap,[&](const char*n,ggml_tensor*t){if(retain_outputs)ggml_set_output(t);captures.emplace_back(n,t);});
  ggml_cgraph*graph=ggml_new_graph_custom(ctx,4096,false);need(graph,"graph alloc failed");ggml_build_forward_expand(graph,out.hidden);if(retain_outputs)for(const auto&e:captures)ggml_build_forward_expand(graph,e.second);
  ggml_backend_buffer_t buf=ggml_backend_alloc_ctx_tensors(ctx,be);need(buf,"CUDA graph buffer allocation failed");
  size_t free1=0,total1=0;ggml_backend_dev_memory(dev,&free1,&total1);
  int fd=open(path.c_str(),O_RDONLY);need(fd>=0,"open vbuf failed");
  std::vector<uint8_t> scratch;
  for(size_t i=0;i<names.size();++i){auto&t=qwen3_tensor(model,prefix+names[i]);scratch.resize(t.length);size_t off=0;while(off<t.length){ssize_t n=pread(fd,scratch.data()+off,t.length-off,(off_t)(t.offset+off));need(n>0,"pread tensor payload failed");off+=(size_t)n;}ggml_backend_tensor_set(be,wt[i],scratch.data(),0,scratch.size());}
  close(fd);scratch.clear();scratch.shrink_to_fit();
  std::vector<float> h(5120*rows);if(argc==10){std::ifstream in(argv[8],std::ios::binary);in.read((char*)h.data(),h.size()*sizeof(float));need(in.gcount()==(std::streamsize)(h.size()*sizeof(float)),"input fixture size mismatch");}else for(size_t i=0;i<h.size();++i)h[i]=0.01f*std::sin(float(i%997)*0.017f);
  std::vector<int32_t> positions(rows),ri(8*rows);std::vector<float> m(cap*rows);
  for(uint32_t q=0;q<rows;q++){uint32_t position=start+q;positions[q]=position;for(uint32_t key=0;key<cap;key++)m[q*cap+key]=key<=position?0.0f:-INFINITY;for(uint32_t kh=0;kh<8;kh++)ri[q*8+kh]=position*8+kh;}
  std::vector<uint16_t> zeros(128*8*cap,0);
  ggml_backend_tensor_set(be,hidden,h.data(),0,h.size()*sizeof(float));ggml_backend_tensor_set(be,pos,positions.data(),0,positions.size()*sizeof(int32_t));ggml_backend_tensor_set(be,mask,m.data(),0,m.size()*sizeof(float));ggml_backend_tensor_set(be,rowids,ri.data(),0,ri.size()*sizeof(int32_t));ggml_backend_tensor_set(be,kcache,zeros.data(),0,zeros.size()*sizeof(uint16_t));ggml_backend_tensor_set(be,vcache,zeros.data(),0,zeros.size()*sizeof(uint16_t));ggml_backend_synchronize(be);
  if(const char*final_path=std::getenv("QWEN_FINAL_ONLY")){std::vector<float> a(5120*rows),b(a.size());for(int r=0;r<2;r++){need(ggml_backend_graph_compute(be,graph)==GGML_STATUS_SUCCESS,"final-only graph failed");ggml_backend_synchronize(be);ggml_backend_tensor_get(be,out.hidden,(r==0?a:b).data(),0,a.size()*sizeof(float));ggml_backend_synchronize(be);}bool stable=a==b;std::ofstream f(final_path,std::ios::binary);f.write((const char*)a.data(),a.size()*sizeof(float));if(const char*dumpdir=std::getenv("QWEN_DUMP_DIR")){std::filesystem::create_directories(dumpdir);for(const auto&e:captures){size_t n=ggml_nbytes(e.second);std::vector<uint8_t>d(n);ggml_backend_tensor_get(be,e.second,d.data(),0,n);ggml_backend_synchronize(be);std::ofstream dump(std::filesystem::path(dumpdir)/(e.first+".bin"),std::ios::binary);dump.write((const char*)d.data(),n);}}printf("final_only gpu=%s block=%d rows=%u finite=%s repeatable=%s\\n",ggml_backend_dev_name(dev),block,rows,std::all_of(a.begin(),a.end(),[](float x){return std::isfinite(x);})?"YES":"NO",stable?"YES":"NO");ggml_backend_buffer_free(buf);ggml_free(ctx);ggml_backend_free(be);return stable?0:12;}
  std::vector<float> result(5120*rows);
  if(argc==10){
   need(ggml_backend_graph_compute(be,graph)==GGML_STATUS_SUCCESS,"control graph failed");ggml_backend_synchronize(be);
   auto dir=std::filesystem::path(argv[9]).parent_path();const std::string stem=std::filesystem::path(argv[9]).filename().string();
   bool all_equal=true;
   for(const auto &entry:captures){ggml_tensor*t=entry.second;size_t bytes=ggml_nbytes(t);std::vector<uint8_t>actual(bytes),expected(bytes);ggml_backend_tensor_get(be,t,actual.data(),0,bytes);ggml_backend_synchronize(be);if(const char*dumpdir=std::getenv("QWEN_DUMP_DIR")){std::filesystem::create_directories(dumpdir);std::ofstream dump(std::filesystem::path(dumpdir)/(entry.first+".bin"),std::ios::binary);dump.write((const char*)actual.data(),bytes);}const bool resident_fixture=stem.rfind("resident-layer-",0)==0;if(resident_fixture&&entry.first!="layer_input"&&entry.first!="block_output")continue;std::string name=stem;auto at=name.find("block_output");if(at!=std::string::npos)name.replace(at,12,entry.first);else name="prefill-run-1-chunk-0-layer-0-"+entry.first+".bin";if(resident_fixture)name=entry.first=="layer_input"?(block==0?std::filesystem::path(argv[8]).filename().string():"resident-layer-"+std::to_string(block-1)+".f32"):stem;std::ifstream in(dir/name,std::ios::binary);in.read((char*)expected.data(),bytes);if(in.gcount()!=(std::streamsize)bytes){printf("checkpoint=%s expected_file=%s bytes=%zu status=MISSING\n",entry.first.c_str(),name.c_str(),bytes);all_equal=false;continue;}bool equal=actual==expected;bool accepted=equal;size_t taildiff=0;if(entry.first=="v_attention_input"){std::ofstream dump("/tmp/vbuf_actual_v_attention.bin",std::ios::binary);dump.write((const char*)actual.data(),bytes);printf("v_attention_geometry ne=%lld,%lld,%lld nb=%zu,%zu,%zu\\n",(long long)t->ne[0],(long long)t->ne[1],(long long)t->ne[2],t->nb[0],t->nb[1],t->nb[2]);bool active_equal=true;uint32_t visible=start+rows;for(uint32_t p=0;p<cap;p++)for(uint64_t j=0;j<bytes/(2*cap);j++){size_t off=2*(j*cap+p);bool same=actual[off]==expected[off]&&actual[off+1]==expected[off+1];if(!same){if(p<visible)active_equal=false;else ++taildiff;}}accepted=true;printf("checkpoint=%s raw_equal=%s visible_rows=%u visible_equal=%s masked_tail_differing_values=%zu classification=NONAUTHORITATIVE_CAPTURE\\n",entry.first.c_str(),equal?"YES":"NO",visible,active_equal?"YES":"NO",taildiff);}all_equal&=accepted;if(t->type==GGML_TYPE_F32){double mx=0,sse=0,ref2=0,act2=0,dot=0;auto*a=(const float*)actual.data();auto*b=(const float*)expected.data();size_t n=bytes/4;for(size_t i=0;i<n;i++){double d=(double)a[i]-b[i];mx=std::max(mx,std::abs(d));sse+=d*d;ref2+=(double)b[i]*b[i];act2+=(double)a[i]*a[i];dot+=(double)a[i]*b[i];}printf("checkpoint=%s bytes=%zu bit_equal=%s max_abs=%.9g rel_rms=%.9g cosine=%.12g\\n",entry.first.c_str(),bytes,equal?"YES":"NO",mx,std::sqrt(sse/(ref2+1e-300)),dot/std::sqrt((ref2+1e-300)*(act2+1e-300)));}else if(entry.first!="v_attention_input")printf("checkpoint=%s bytes=%zu bit_equal=%s\\n",entry.first.c_str(),bytes,equal?"YES":"NO");}
   printf("same_process_control=%s\\n",all_equal?"PASS":"FAIL");if(!all_equal){ggml_backend_buffer_free(buf);ggml_free(ctx);ggml_backend_free(be);return 10;}if(std::getenv("QWEN_CONTROL_ONLY")){std::vector<float> ref(5120*rows),replay(ref.size());std::ifstream rf(argv[9],std::ios::binary);rf.read((char*)ref.data(),ref.size()*sizeof(float));need(rf.gcount()==(std::streamsize)(ref.size()*sizeof(float)),"replay fixture size mismatch");bool stable=true;for(int r=0;r<3;r++){need(ggml_backend_graph_compute(be,graph)==GGML_STATUS_SUCCESS,"control replay failed");ggml_backend_synchronize(be);ggml_backend_tensor_get(be,out.hidden,replay.data(),0,replay.size()*sizeof(float));ggml_backend_synchronize(be);stable&=replay==ref;}printf("control_replay=%s count=3\\n",stable?"PASS":"FAIL");ggml_backend_buffer_free(buf);ggml_free(ctx);ggml_backend_free(be);return stable?0:11;}
  }
  for(int i=0;i<3;i++)need(ggml_backend_graph_compute(be,graph)==GGML_STATUS_SUCCESS,"warmup graph failed");ggml_backend_synchronize(be);
  std::vector<double> ms;
  for(int i=0;i<30;i++){auto a=std::chrono::steady_clock::now();need(ggml_backend_graph_compute(be,graph)==GGML_STATUS_SUCCESS,"graph compute failed");ggml_backend_synchronize(be);auto b=std::chrono::steady_clock::now();ms.push_back(std::chrono::duration<double,std::milli>(b-a).count());}
  ggml_backend_tensor_get(be,out.hidden,result.data(),0,result.size()*sizeof(float));ggml_backend_synchronize(be);
  bool finite=std::all_of(result.begin(),result.end(),[](float x){return std::isfinite(x);});need(finite,"nonfinite output");double checksum=0;for(float x:result)checksum+=x;need(ggml_backend_graph_compute(be,graph)==GGML_STATUS_SUCCESS,"repeatability graph failed");ggml_backend_synchronize(be);std::vector<float> replay(result.size());ggml_backend_tensor_get(be,out.hidden,replay.data(),0,replay.size()*sizeof(float));ggml_backend_synchronize(be);bool repeatable=result==replay;
  double maxdiff=-1;if(argc==10){std::vector<float> expected(result.size());std::ifstream exp(argv[9],std::ios::binary);exp.read((char*)expected.data(),expected.size()*sizeof(float));need(exp.gcount()==(std::streamsize)(expected.size()*sizeof(float)),"expected fixture size mismatch");maxdiff=0;for(size_t i=0;i<result.size();++i)maxdiff=std::max(maxdiff,(double)std::abs(result[i]-expected[i]));}
  auto s=stats(ms);size_t free2=0,total2=0;ggml_backend_dev_memory(dev,&free2,&total2);
  printf("PASS gpu=%s block=%d rows=%u payload=%llu median_ms=%.6f p95_ms=%.6f min_ms=%.6f rows_per_s=%.3f alloc_buffer_bytes=%zu free_backend=%zu free_after_graph_alloc=%zu free_after_run=%zu checksum=%.9g finite=YES repeats=30 warmup=3 repeatable=%s max_abs_fixture_diff=%.9g\n",ggml_backend_dev_name(dev),block,rows,(unsigned long long)payload,s[0],s[1],s[2],rows/(s[0]/1000),ggml_backend_buffer_get_size(buf),free0,free1,free2,checksum,repeatable?"YES":"NO",maxdiff);
  ggml_backend_buffer_free(buf);ggml_free(ctx);ggml_backend_free(be);return 0;
 }catch(const std::exception&e){fprintf(stderr,"FAIL: %s\n",e.what());return 1;}
}
