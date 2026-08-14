#include <algorithm>
#include <atomic>
#include <chrono>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <vector>

using Clock = std::chrono::steady_clock;
struct Span { std::string role, label; uint64_t layer = UINT64_MAX, start = 0, end = 0, bytes = 0, useful = 0, gap = 0, tensors = 0; };
struct SpanResult { Span span; double ms = 0; uint64_t requests = 0, bytes = 0, short_reads = 0; int errors = 0; };
struct Stats { long minflt = 0, majflt = 0; long user_us = 0, sys_us = 0; long rss_kb = -1; long long read_bytes = -1, rchar = -1, syscr = -1; };

static Stats stats() {
    rusage u{}; getrusage(RUSAGE_SELF, &u); Stats s{u.ru_minflt,u.ru_majflt,u.ru_utime.tv_sec*1000000L+u.ru_utime.tv_usec,u.ru_stime.tv_sec*1000000L+u.ru_stime.tv_usec};
    std::ifstream status("/proc/self/status"); std::string key; while (status >> key) { if (key == "VmRSS:") { status >> s.rss_kb; break; } std::string ignored; std::getline(status, ignored); }
    std::ifstream io("/proc/self/io"); while (io >> key) { long long v=-1; io >> v; if(key=="read_bytes:")s.read_bytes=v; else if(key=="rchar:")s.rchar=v; else if(key=="syscr:")s.syscr=v; }
    return s;
}
static std::vector<std::string> csv(const std::string & line) { std::vector<std::string> out; size_t at=0,next; while ((next=line.find(',',at)) != std::string::npos) { out.push_back(line.substr(at,next-at)); at=next+1; } out.push_back(line.substr(at)); return out; }
static std::vector<Span> read_plan(const char * path) {
    std::ifstream in(path); std::string line; std::getline(in,line); std::vector<Span> spans;
    while (std::getline(in,line)) { auto f=csv(line); if(f.size()<9) throw std::runtime_error("malformed span plan"); spans.push_back({f[0],f[1],std::stoull(f[2]),std::stoull(f[3]),std::stoull(f[4]),std::stoull(f[5]),std::stoull(f[6]),std::stoull(f[7])}); }
    if(spans.empty()) throw std::runtime_error("empty span plan"); return spans;
}
static void read_exact(int fd, uint64_t offset, size_t len, uint8_t * dst, SpanResult & result) {
    size_t done=0;
    while(done<len) { ssize_t n=pread(fd,dst+done,len-done,(off_t)(offset+done)); ++result.requests; if(n<0){++result.errors; return;} if(n==0){++result.short_reads; return;} if((size_t)n<len-done)++result.short_reads; done+=(size_t)n; result.bytes+=n; }
}
static SpanResult read_span(int fd, const Span & span, size_t width, unsigned qd) {
    SpanResult result; result.span=span; const size_t actual=width ? std::min<uint64_t>(width,span.bytes) : (size_t)span.bytes; const unsigned workers=width ? qd : 1; std::atomic<uint64_t> next{0};
    const auto begin=Clock::now(); std::vector<std::thread> threads; std::vector<SpanResult> locals(workers); for(auto & local:locals)local.span=span;
    for(unsigned worker=0;worker<workers;++worker) threads.emplace_back([&,worker]{ std::vector<uint8_t> buffer(actual); auto & local=locals[worker]; while(true){ uint64_t chunk=next.fetch_add(1); uint64_t offset=chunk*(uint64_t)actual; if(offset>=span.bytes)break; size_t len=(size_t)std::min<uint64_t>(actual,span.bytes-offset); read_exact(fd,span.start+offset,len,buffer.data(),local); } });
    for(auto & t:threads)t.join(); for(const auto & local:locals){result.requests+=local.requests;result.bytes+=local.bytes;result.short_reads+=local.short_reads;result.errors+=local.errors;} result.ms=std::chrono::duration<double,std::milli>(Clock::now()-begin).count(); return result;
}
static SpanResult touch_span(const uint8_t * mapping, const Span & span, size_t page) {
    SpanResult result; result.span=span; auto begin=Clock::now(); uintptr_t first=((uintptr_t)mapping+span.start)&~(uintptr_t)(page-1); uintptr_t last=((uintptr_t)mapping+span.end+page-1)&~(uintptr_t)(page-1); volatile uint8_t sink=0; for(uintptr_t p=first;p<last;p+=page){sink^=*reinterpret_cast<volatile const uint8_t *>(p);++result.requests; result.bytes += page;} (void)sink; result.ms=std::chrono::duration<double,std::milli>(Clock::now()-begin).count(); return result;
}
static void print_stats(const char * key, const Stats & s) { std::printf("\"%s\":{\"minor_faults\":%ld,\"major_faults\":%ld,\"user_us\":%ld,\"sys_us\":%ld,\"rss_kb\":%ld,\"read_bytes\":%lld,\"rchar\":%lld,\"syscr\":%lld}",key,s.minflt,s.majflt,s.user_us,s.sys_us,s.rss_kb,s.read_bytes,s.rchar,s.syscr); }
int main(int argc,char ** argv) {
    if(argc!=6) return 2; const char * path=argv[1], * plan_path=argv[2], * mode=argv[3]; const size_t width=std::strtoull(argv[4],nullptr,10)*1024*1024; const unsigned qd=std::max(1u,(unsigned)std::strtoul(argv[5],nullptr,10));
    int fd=open(path,O_RDONLY); if(fd<0)return 3; auto spans=read_plan(plan_path); std::sort(spans.begin(),spans.end(),[](const Span&a,const Span&b){return a.start<b.start;});
    struct stat st{}; if(fstat(fd,&st)!=0){close(fd);return 4;} for(const auto & s:spans)if(s.end>(uint64_t)st.st_size){close(fd);return 5;}
    const size_t page=(size_t)sysconf(_SC_PAGESIZE); Stats before=stats(); std::vector<SpanResult> results; const auto all_begin=Clock::now();
    if(std::strcmp(mode,"page")==0){ void * map=mmap(nullptr,(size_t)st.st_size,PROT_READ,MAP_PRIVATE,fd,0); if(map==MAP_FAILED){close(fd);return 6;} for(const auto&s:spans)results.push_back(touch_span((const uint8_t*)map,s,page)); munmap(map,(size_t)st.st_size); }
    else if(std::strcmp(mode,"whole")==0){ for(const auto & span:spans) results.push_back(read_span(fd,span,0,1)); }
    else if(std::strcmp(mode,"pread")==0){ for(const auto&s:spans)results.push_back(read_span(fd,s,width,qd)); }
    else {close(fd);return 7;}
    Stats after=stats(); close(fd); double elapsed=std::chrono::duration<double,std::milli>(Clock::now()-all_begin).count(); uint64_t bytes=0,requests=0,shorts=0; int errors=0; size_t reserved=(mode==std::string("pread") || mode==std::string("whole")) ? (width ? width*qd : (size_t)std::max_element(spans.begin(),spans.end(),[](const Span&a,const Span&b){return a.bytes<b.bytes;})->bytes) : 0; for(const auto&r:results){bytes+=r.bytes;requests+=r.requests;shorts+=r.short_reads;errors+=r.errors;}
    std::printf("{\"path\":\"%s\",\"mode\":\"%s\",\"width_bytes\":%zu,\"queue_depth\":%u,\"page_size\":%zu,\"elapsed_ms\":%.6f,\"requested_bytes\":%llu,\"successful_bytes\":%llu,\"effective_GB_per_s\":%.9f,\"requests\":%llu,\"average_request_bytes\":%.3f,\"short_reads\":%llu,\"errors\":%d,\"buffer_reserved_bytes\":%zu,",path,mode,width,qd,page,elapsed,(unsigned long long)std::accumulate(spans.begin(),spans.end(),uint64_t(0),[](uint64_t x,const Span&s){return x+s.bytes;}),(unsigned long long)bytes,(bytes/1e9)/(elapsed/1000.0),(unsigned long long)requests,requests?(double)bytes/requests:0.0,(unsigned long long)shorts,errors,reserved); print_stats("before",before); std::printf(","); print_stats("after",after); std::printf(",\"spans\":["); for(size_t i=0;i<results.size();++i){auto&r=results[i];std::printf("%s{\"role\":\"%s\",\"label\":\"%s\",\"layer_id\":%llu,\"start_offset\":%llu,\"span_bytes\":%llu,\"elapsed_ms\":%.6f,\"requests\":%llu,\"successful_bytes\":%llu,\"effective_GB_per_s\":%.9f}",i?",":"",r.span.role.c_str(),r.span.label.c_str(),(unsigned long long)r.span.layer,(unsigned long long)r.span.start,(unsigned long long)r.span.bytes,r.ms,(unsigned long long)r.requests,(unsigned long long)r.bytes,(r.bytes/1e9)/(r.ms/1000.0));} std::printf("]}\n"); return errors||shorts?8:0;
}
