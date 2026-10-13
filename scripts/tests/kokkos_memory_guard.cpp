// Host-side Kokkos Tools plug-in: bound and report combined tracked allocations.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <string>

#include "impl/Kokkos_Profiling_C_Interface.h"

namespace {
std::mutex lock;
std::map<std::string,uint64_t> current, peaks, kernels;
uint64_t total=0, peak=0, budget=24000000000ULL;
}

extern "C" void kokkosp_init_library(int, uint64_t, uint32_t,
                                     Kokkos_Profiling_KokkosPDeviceInfo*) {
  if (auto p=std::getenv("KOKKOS_MEMORY_BUDGET_BYTES")) budget=std::strtoull(p,nullptr,10);
}

extern "C" void kokkosp_request_tool_settings(uint32_t,
                                               Kokkos_Tools_ToolSettings *settings) {
  settings->requires_global_fencing=false;
}

extern "C" void kokkosp_allocate_data(Kokkos_Profiling_SpaceHandle space,
                                      const char*, const void*, uint64_t size) {
  std::lock_guard<std::mutex> guard(lock);
  total += size;
  current[space.name] += size;
  peaks[space.name]=std::max(peaks[space.name],current[space.name]);
  peak=std::max(peak,total);
  if (total > budget) {
    std::fprintf(stderr,"Kokkos combined allocation budget exceeded: %llu > %llu\n",
                 static_cast<unsigned long long>(total),
                 static_cast<unsigned long long>(budget));
    std::abort();
  }
}

extern "C" void kokkosp_deallocate_data(Kokkos_Profiling_SpaceHandle space,
                                        const char*, const void*, uint64_t size) {
  std::lock_guard<std::mutex> guard(lock);
  current[space.name] -= size;
  total -= size;
}

extern "C" void kokkosp_begin_parallel_for(const char *name, uint32_t,
                                          uint64_t *kernel_id) {
  std::lock_guard<std::mutex> guard(lock);
  ++kernels[name];
  *kernel_id=0;
}

extern "C" void kokkosp_end_parallel_for(uint64_t) {}

extern "C" void kokkosp_finalize_library() {
  const char *path=std::getenv("KOKKOS_MEMORY_REPORT");
  if (!path) return;
  FILE *f=std::fopen(path,"w");
  if (!f) {std::perror(path); std::abort();}
  std::fprintf(f,"{\"combined_peak_bytes\":%llu,\"budget_bytes\":%llu,\"spaces\":{",
               static_cast<unsigned long long>(peak),
               static_cast<unsigned long long>(budget));
  bool first=true;
  for (const auto &entry : peaks) {
    std::fprintf(f,"%s\"%s\":%llu",first ? "" : ",",entry.first.c_str(),
                 static_cast<unsigned long long>(entry.second));
    first=false;
  }
  std::fprintf(f,"},\"metric_evaluations\":%llu}\n",
               static_cast<unsigned long long>(kernels["update_adm_vars_ad"]+
                                                kernels["update_adm_vars_fd"]));
  std::fclose(f);
}
