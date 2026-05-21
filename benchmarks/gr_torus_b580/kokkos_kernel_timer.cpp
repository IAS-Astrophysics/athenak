#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

struct Kokkos_Profiling_KokkosPDeviceInfo;

namespace {

using Clock = std::chrono::steady_clock;

struct ActiveKernel {
  std::string name;
  Clock::time_point start;
};

struct KernelStats {
  uint64_t count = 0;
  double seconds = 0.0;
};

std::mutex timer_mutex;
uint64_t next_kernel_id = 1;
std::unordered_map<uint64_t, ActiveKernel> active;
std::unordered_map<std::string, KernelStats> stats;

void begin_kernel(const char *name, uint64_t *kernel_id) {
  std::lock_guard<std::mutex> lock(timer_mutex);
  const uint64_t id = next_kernel_id++;
  *kernel_id = id;
  active.emplace(id, ActiveKernel{name ? name : "<unnamed>", Clock::now()});
}

void end_kernel(const uint64_t kernel_id) {
  const auto stop = Clock::now();
  std::lock_guard<std::mutex> lock(timer_mutex);
  auto it = active.find(kernel_id);
  if (it == active.end()) {
    return;
  }
  const double seconds = std::chrono::duration<double>(stop - it->second.start).count();
  auto &entry = stats[it->second.name];
  entry.count += 1;
  entry.seconds += seconds;
  active.erase(it);
}

} // namespace

extern "C" void kokkosp_init_library(const int, const uint64_t, const uint32_t,
                                      Kokkos_Profiling_KokkosPDeviceInfo*) {}

extern "C" void kokkosp_finalize_library() {
  std::vector<std::pair<std::string, KernelStats>> rows;
  {
    std::lock_guard<std::mutex> lock(timer_mutex);
    rows.reserve(stats.size());
    for (const auto &entry : stats) {
      rows.push_back(entry);
    }
  }
  std::sort(rows.begin(), rows.end(),
            [](const auto &a, const auto &b) {
              return a.second.seconds > b.second.seconds;
            });
  const char *limit_env = std::getenv("KOKKOS_KERNEL_TIMER_LIMIT");
  const std::size_t limit = limit_env ? std::max(1, std::atoi(limit_env)) : 80;
  std::cerr << "KOKKOS_KERNEL_TIMER_BEGIN\n";
  std::cerr << "seconds count avg_seconds kernel\n";
  for (std::size_t n = 0; n < rows.size() && n < limit; ++n) {
    const auto &name = rows[n].first;
    const auto &entry = rows[n].second;
    std::cerr << entry.seconds << " " << entry.count << " "
              << (entry.count ? entry.seconds/static_cast<double>(entry.count) : 0.0)
              << " " << name << "\n";
  }
  std::cerr << "KOKKOS_KERNEL_TIMER_END\n";
}

extern "C" void kokkosp_begin_parallel_for(const char *name, const uint32_t,
                                           uint64_t *kernel_id) {
  begin_kernel(name, kernel_id);
}

extern "C" void kokkosp_end_parallel_for(const uint64_t kernel_id) {
  end_kernel(kernel_id);
}

extern "C" void kokkosp_begin_parallel_reduce(const char *name, const uint32_t,
                                              uint64_t *kernel_id) {
  begin_kernel(name, kernel_id);
}

extern "C" void kokkosp_end_parallel_reduce(const uint64_t kernel_id) {
  end_kernel(kernel_id);
}

extern "C" void kokkosp_begin_parallel_scan(const char *name, const uint32_t,
                                            uint64_t *kernel_id) {
  begin_kernel(name, kernel_id);
}

extern "C" void kokkosp_end_parallel_scan(const uint64_t kernel_id) {
  end_kernel(kernel_id);
}
