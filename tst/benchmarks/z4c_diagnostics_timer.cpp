// Copyright (c) 2026 Hengrui Zhu
// AthenaK: BSD-3-Clause (see LICENSE).
// CPU/Serial Kokkos Tools callback: time only the full curvature kernel.
// Build: c++ -O2 -std=c++17 -shared -fPIC z4c_diagnostics_timer.cpp -o timer.so
// Use: KOKKOS_TOOLS_LIBS=/absolute/path/timer.so ./athena ...
// These host timings are not GPU timings; asynchronous backends need device events.
#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstring>

namespace {
using Clock = std::chrono::steady_clock;
Clock::time_point start;
constexpr uint64_t diagnostic_id = 790;
}

extern "C" void kokkosp_begin_parallel_for(const char *name, uint32_t,
                                           uint64_t *id) {
  *id = 0;
  if (std::strcmp(name, "vacuum curvature diagnostics") == 0) {
    *id = diagnostic_id;
    start = Clock::now();
  }
}

extern "C" void kokkosp_end_parallel_for(uint64_t id) {
  if (id == diagnostic_id) {
    const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        Clock::now()-start).count();
    std::printf("Z4C_DIAGNOSTIC_NS %" PRId64 "\n", static_cast<int64_t>(ns));
  }
}
