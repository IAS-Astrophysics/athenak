#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mutex>
#include <string>
#include <unordered_map>

struct SpaceHandle {
  char name[64];
};

namespace {
std::mutex mutex;
std::unordered_map<const void *, std::pair<std::string, uint64_t>> allocations;
std::unordered_map<std::string, uint64_t> current_by_space;
std::unordered_map<std::string, uint64_t> peak_by_space;

std::string SpaceName(SpaceHandle handle) {
  return std::string(handle.name, strnlen(handle.name, sizeof(handle.name)));
}

void PrintPeaks() {
  std::cerr << "KOKKOS_MEMORY_PEAK_BEGIN\n";
  for (const auto &entry : peak_by_space) {
    std::cerr << entry.first << " " << entry.second << "\n";
  }
  std::cerr << "KOKKOS_MEMORY_PEAK_END\n";
}
}  // namespace

extern "C" void kokkosp_init_library(const int, const uint64_t, const uint32_t, void *) {}

extern "C" void kokkosp_finalize_library() {
  std::lock_guard<std::mutex> lock(mutex);
  PrintPeaks();
}

extern "C" void kokkosp_allocate_data(SpaceHandle handle, const char *, const void *ptr,
                                      uint64_t size) {
  std::lock_guard<std::mutex> lock(mutex);
  const std::string space = SpaceName(handle);
  allocations[ptr] = {space, size};
  current_by_space[space] += size;
  if (current_by_space[space] > peak_by_space[space]) {
    peak_by_space[space] = current_by_space[space];
  }
}

extern "C" void kokkosp_deallocate_data(SpaceHandle handle, const char *, const void *ptr,
                                        uint64_t size) {
  std::lock_guard<std::mutex> lock(mutex);
  auto it = allocations.find(ptr);
  if (it != allocations.end()) {
    current_by_space[it->second.first] -= it->second.second;
    allocations.erase(it);
  } else {
    const std::string space = SpaceName(handle);
    current_by_space[space] -= size;
  }
}
