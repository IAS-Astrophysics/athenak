#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <string>
#include <unordered_map>

namespace {
using Clock = std::chrono::steady_clock;

struct ActiveKernel {
  std::string name;
  Clock::time_point start;
};

struct KernelStats {
  uint64_t calls = 0;
  double seconds = 0.0;
};

std::mutex mutex;
std::unordered_map<uint64_t, ActiveKernel> active;
std::unordered_map<std::string, KernelStats> stats;

void BeginKernel(const char *name, uint64_t *kernel_id) {
  static uint64_t next_id = 1;
  std::lock_guard<std::mutex> lock(mutex);
  *kernel_id = next_id++;
  active[*kernel_id] = {name ? name : "<unnamed>", Clock::now()};
}

void EndKernel(uint64_t kernel_id) {
  const auto stop = Clock::now();
  std::lock_guard<std::mutex> lock(mutex);
  auto it = active.find(kernel_id);
  if (it == active.end()) {
    return;
  }
  const double dt = std::chrono::duration<double>(stop - it->second.start).count();
  auto &entry = stats[it->second.name];
  entry.calls += 1;
  entry.seconds += dt;
  active.erase(it);
}
}  // namespace

extern "C" void kokkosp_init_library(const int, const uint64_t, const uint32_t, void *) {}

extern "C" void kokkosp_finalize_library() {
  std::lock_guard<std::mutex> lock(mutex);
  std::cerr << "KOKKOS_KERNEL_TIMES_BEGIN\n";
  for (const auto &entry : stats) {
    std::cerr << std::setprecision(9) << entry.second.seconds << " "
              << entry.second.calls << " " << entry.first << "\n";
  }
  std::cerr << "KOKKOS_KERNEL_TIMES_END\n";
}

extern "C" void kokkosp_begin_parallel_for(const char *name, const uint32_t, uint64_t *id) {
  BeginKernel(name, id);
}

extern "C" void kokkosp_end_parallel_for(const uint64_t id) {
  EndKernel(id);
}

extern "C" void kokkosp_begin_parallel_reduce(const char *name, const uint32_t, uint64_t *id) {
  BeginKernel(name, id);
}

extern "C" void kokkosp_end_parallel_reduce(const uint64_t id) {
  EndKernel(id);
}

extern "C" void kokkosp_begin_parallel_scan(const char *name, const uint32_t, uint64_t *id) {
  BeginKernel(name, id);
}

extern "C" void kokkosp_end_parallel_scan(const uint64_t id) {
  EndKernel(id);
}
