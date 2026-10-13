// Synchronous diagnostic profiler. Use unprofiled runs for performance claims.
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <string>
#include "impl/Kokkos_Profiling_C_Interface.h"
namespace {
using Clock=std::chrono::steady_clock;
struct Entry { std::string name; Clock::time_point start; };
std::mutex mutex;
uint64_t next_id=0, allocated=0, peak_bytes=0;
std::map<uint64_t,Entry> active;
std::map<std::string,std::pair<uint64_t,double>> totals;
void begin(const char *name, uint64_t *id) {
 std::lock_guard<std::mutex> lock(mutex);
 *id=++next_id; active.emplace(*id,Entry{name,Clock::now()});
}
void end(uint64_t id) {
 auto now=Clock::now(); std::lock_guard<std::mutex> lock(mutex);
 auto it=active.find(id); if(it==active.end()) return;
 auto &v=totals[it->second.name]; ++v.first;
 v.second+=std::chrono::duration<double>(now-it->second.start).count();
 active.erase(it);
}
}
extern "C" void kokkosp_allocate_data(Kokkos_Profiling_SpaceHandle, const char*, const void*, uint64_t size) {
 std::lock_guard<std::mutex> lock(mutex); allocated+=size;
 if(allocated>peak_bytes) peak_bytes=allocated;
 const char *s=std::getenv("KOKKOS_MEMORY_BUDGET_BYTES");
 uint64_t budget=s ? std::strtoull(s,nullptr,10) : 24000000000ULL;
 if(allocated>budget) std::abort();
}
extern "C" void kokkosp_deallocate_data(Kokkos_Profiling_SpaceHandle, const char*, const void*, uint64_t size) {
 std::lock_guard<std::mutex> lock(mutex); allocated-=size;
}
extern "C" void kokkosp_request_tool_settings(uint32_t,Kokkos_Tools_ToolSettings *s) {
 s->requires_global_fencing=true;
}
extern "C" void kokkosp_begin_parallel_for(const char *n,uint32_t,uint64_t *id) {begin(n,id);}
extern "C" void kokkosp_end_parallel_for(uint64_t id) {end(id);}
extern "C" void kokkosp_begin_parallel_reduce(const char *n,uint32_t,uint64_t *id) {begin(n,id);}
extern "C" void kokkosp_end_parallel_reduce(uint64_t id) {end(id);}
extern "C" void kokkosp_begin_parallel_scan(const char *n,uint32_t,uint64_t *id) {begin(n,id);}
extern "C" void kokkosp_end_parallel_scan(uint64_t id) {end(id);}
extern "C" void kokkosp_finalize_library() {
 auto p=std::getenv("KOKKOS_PROFILE_REPORT"); if(!p) return;
 FILE *f=std::fopen(p,"w"); if(!f) std::abort();
 std::fprintf(f,"kernel,calls,synchronous_seconds\n");
 for(auto &v:totals) std::fprintf(f,"%s,%llu,%.9f\n",v.first.c_str(),
 static_cast<unsigned long long>(v.second.first),v.second.second);
 std::fclose(f);
 std::fprintf(stderr,"Profile tracked peak bytes: %llu\n",static_cast<unsigned long long>(peak_bytes));
}
