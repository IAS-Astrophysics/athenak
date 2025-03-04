#ifndef RADIATION_M1_CLOSURE_H
#define RADIATION_M1_CLOSURE_H

#include "athena.hpp"
#include "radiation_m1/radiation_m1_params.hpp"

namespace radiationm1 {
KOKKOS_INLINE_FUNCTION Real closure_fun(const Real &xi,
                                        const RadiationM1Closure &closure_type) {
  switch (closure_type) {
    case Minerbo:
      return 1.0 / 3.0 + xi * xi * (6.0 - 2.0 * xi + 6.0 * xi * xi) / 15.0;
    case Thin:
      return 1.0;
    case Eddington:
      return 1.0 / 3.0;
    case Kershaw:
      return 1.0 / 3.0 + 2.0 / 3.0 * xi * xi;
    default:
      return 1.0 / 3.0 + xi * xi * (6.0 - 2.0 * xi + 6.0 * xi * xi) / 15.0;
  }
}
}  // namespace radiationm1
#endif  // RADIATION_M1_CLOSURE_H
