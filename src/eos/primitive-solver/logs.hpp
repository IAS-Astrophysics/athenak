#ifndef EOS_PRIMITIVE_SOLVER_LOGS_HPP_
#define EOS_PRIMITIVE_SOLVER_LOGS_HPP_

#include <cmath>
#include <limits>

#include "../../athena.hpp"

namespace Primitive{

class LogPolicy {
  public:
  LogPolicy() = default;
  ~LogPolicy() = default;

  // KOKKOS_INLINE_FUNCTION Real log2(const Real x) const {return std::numeric_limits<Real>::quiet_NaN();}
  // KOKKOS_INLINE_FUNCTION Real exp2(const Real x) const {return std::numeric_limits<Real>::quiet_NaN();}
};

class NormalLogs : public LogPolicy {
  public:
    NormalLogs() = default;
    ~NormalLogs() = default;

    KOKKOS_INLINE_FUNCTION Real log2_(const Real x) const {
      return Kokkos::log2(x);
    }

    KOKKOS_INLINE_FUNCTION Real exp2_(const Real x) const {
      return Kokkos::exp2(x);
    }

};

class NQTLogs : public LogPolicy {
  public:
    NQTLogs() = default;
    ~NQTLogs() = default;

    KOKKOS_INLINE_FUNCTION Real log2_(const Real x) const {
      // Magic numbers constexpr because C++ doesn't constexpr reinterpret casts
      // these are floating point numbers as reinterpreted as integers.
      // as_int(1.0)
      constexpr int64_t one_as_int = 4607182418800017408;
      // 1./static_cast<double>(as_int(2.0) - as_int(1.0))
      constexpr Real scale_down = 2.22044604925031e-16;
      return static_cast<Real>(as_int(x) - one_as_int) * scale_down;
    }

    KOKKOS_INLINE_FUNCTION Real exp2_(const Real x) const {
      // Magic numbers constexpr because C++ doesn't constexpr reinterpret casts
      // these are floating point numbers as reinterpreted as integers.
      // as_int(1.0)
      constexpr int64_t one_as_int = 4607182418800017408;
      // as_int(2.0) - as_int(1.0)
      constexpr Real scale_up = 4503599627370496;
      return as_double(static_cast<int64_t>(x*scale_up) + one_as_int);
    }
  private:
    KOKKOS_INLINE_FUNCTION int64_t as_int(const Real f) const {
      Real f_ = f;
      return *reinterpret_cast<int64_t*>(&f_);
    }

    KOKKOS_INLINE_FUNCTION Real as_double(const int64_t i) const {
      int64_t i_ = i;
      return *reinterpret_cast<Real*>(&i_);
    }
};

} // namespace Primitive

/*
KOKKOS_INLINE_FUNCTION int64_t as_int(Real f) {
  return *reinterpret_cast<int64_t*>(&f);
}

KOKKOS_INLINE_FUNCTION Real as_double(int64_t i) {
  return *reinterpret_cast<Real*>(&i);
}

KOKKOS_INLINE_FUNCTION Real NQT_log2(const Real x) {
  // Magic numbers constexpr because C++ doesn't constexpr reinterpret casts
  // these are floating point numbers as reinterpreted as integers.
  // as_int(1.0)
  constexpr int64_t one_as_int = 4607182418800017408;
  // 1./static_cast<double>(as_int(2.0) - as_int(1.0))
  constexpr Real scale_down = 2.22044604925031e-16;
  return static_cast<Real>(as_int(x) - one_as_int) * scale_down;
}

KOKKOS_INLINE_FUNCTION Real NQT_exp2(const Real x) {
  // Magic numbers constexpr because C++ doesn't constexpr reinterpret casts
  // these are floating point numbers as reinterpreted as integers.
  // as_int(1.0)
  constexpr int64_t one_as_int = 4607182418800017408;
  // as_int(2.0) - as_int(1.0)
  constexpr Real scale_up = 4503599627370496;
  return as_double(static_cast<int64_t>(x*scale_up) + one_as_int);
}

KOKKOS_INLINE_FUNCTION Real log2(Real x, bool use_NQT=false) {
  if (use_NQT) {
    return NQT_log2(x);
  } else {
    return std::log2(x);
  }
}

KOKKOS_INLINE_FUNCTION Real exp2(Real x, bool use_NQT=false) {
  if (use_NQT) {
    return NQT_exp2(x);
  } else {
    return std::exp2(x);
  }
}
*/

#endif