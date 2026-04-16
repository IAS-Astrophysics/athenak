#ifndef EOS_PRIMITIVE_SOLVER_LOGS_HPP_
#define EOS_PRIMITIVE_SOLVER_LOGS_HPP_
//========================================================================================
// Not-Quite-Transcendental log and exp functions
// Reproduced from https://github.com/lanl/not-quite-transcendental/tree/main
//========================================================================================
/*
© (or copyright) 2022. Triad National Security, LLC. All rights
reserved.  This program was produced under U.S. Government contract
89233218CNA000001 for Los Alamos National Laboratory (LANL), which is
operated by Triad National Security, LLC for the U.S.  Department of
Energy/National Nuclear Security Administration. All rights in the
program are reserved by Triad National Security, LLC, and the
U.S. Department of Energy/National Nuclear Security
Administration. The Government is granted for itself and others acting
on its behalf a nonexclusive, paid-up, irrevocable worldwide license
in this material to reproduce, prepare derivative works, distribute
copies to the public, perform publicly and display publicly, and to
permit others to do so.

This program is open source under the BSD-3 License.  Redistribution
and use in source and binary forms, with or without modification, are
permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE
*/

#include <limits>

#include <Kokkos_Core.hpp>

#include "../../athena.hpp"

namespace Primitive {

class NormalLogs {
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

class NQTLogs {
 public:
  NQTLogs() = default;
  ~NQTLogs() = default;
  // Because we are using bit-hacks we are explicit/specific about types
  // These are left here because we could add an option to select the order
  KOKKOS_FORCEINLINE_FUNCTION
  double log2_LANL(const double x) const {
    // Magic numbers constexpr because C++ doesn't constexpr reinterpret casts
    // these are floating point numbers as reinterpreted as integers.
    // as_int(1.0)
    constexpr int64_t one_as_int = 4607182418800017408;
    // 1./static_cast<double>(as_int(2.0) - as_int(1.0))
    constexpr double scale_down = 2.22044604925031e-16;
    return static_cast<double>(as_int(x) - one_as_int) * scale_down;
  }

  KOKKOS_FORCEINLINE_FUNCTION
  double exp2_LANL(const double x) const {
    // Magic numbers constexpr because C++ doesn't constexpr reinterpret casts
    // these are floating point numbers as reinterpreted as integers.
    // as_int(1.0)
    constexpr int64_t one_as_int = 4607182418800017408;
    // as_int(2.0) - as_int(1.0)
    constexpr double scale_up = 4503599627370496;
    return as_real(static_cast<int64_t>(x*scale_up) + one_as_int);
  }

  KOKKOS_FORCEINLINE_FUNCTION
  float log2_LANL(const float x) const {
    // as_int(1.0)
    constexpr int32_t one_as_int = 1065353216;
    // 1./static_cast<double>(as_int(2.0) - as_int(1.0))
    constexpr float scale_down = 1.f/8388608.f;
    return static_cast<float>(as_int(x) - one_as_int) * scale_down;
  }

  KOKKOS_FORCEINLINE_FUNCTION
  float exp2_LANL(const float x) const {
    // as_int(1.0)
    constexpr int32_t one_as_int = 1065353216;
    // as_int(2.0) - as_int(1.0)
    constexpr float scale_up = 8388608.f;
    return as_real(static_cast<int32_t>(x*scale_up) + one_as_int);
  }

  KOKKOS_FORCEINLINE_FUNCTION
  double log2_(const double x) const {
    // as_int(1.0) == 2^62 - 2^52
    constexpr int64_t one_as_int = 4607182418800017408;
    // 1/(as_int(2.0) - as_int(1.0)) == 2^-52
    constexpr double scale_down = 2.220446049250313e-16;
    // 2^52 - 1
    constexpr int64_t mantissa_mask = 4503599627370495;
    // 2^26 - 1
    constexpr int64_t low_mask = 67108863;

    const int64_t x_as_int = as_int(x) - one_as_int;
    const int64_t frac_as_int = x_as_int & mantissa_mask;
    const int64_t frac_high = frac_as_int>>26;
    const int64_t frac_low  = frac_as_int & low_mask;
    const int64_t frac_squared = frac_high*frac_high + ((frac_high*frac_low)>>25);

    return static_cast<double>(x_as_int +
                                  ((frac_as_int - frac_squared)/3)) * scale_down;
    // return static_cast<double>(x_as_int) * scale_down;
  }

  KOKKOS_FORCEINLINE_FUNCTION
  double exp2_(const double x) const {
    // as_int(1.0) == 2^62 - 2^52
    constexpr int64_t one_as_int = 4607182418800017408;
    // as_int(2.0) - as_int(1.0) == 2^52
    constexpr double scale_up = 4503599627370496.;
    constexpr int64_t mantissa_mask = 4503599627370495; // 2^52 - 1
    constexpr int64_t a = 9007199254740992; // 2 * 2^52
    // constexpr double b = 67108864; // 2^26
    constexpr int64_t c = 18014398509481984; // 4 * 2^52

    const int64_t x_as_int = static_cast<int64_t>(x*scale_up);
    const int64_t frac_as_int = x_as_int & mantissa_mask;
    const int64_t frac_sqrt = static_cast<int64_t>(
      Kokkos::sqrt(scale_up*static_cast<double>(c-3*frac_as_int)));

    return as_real(x_as_int + a - frac_sqrt - frac_as_int + one_as_int);
    // return as_float(x_as_int + one_as_int);
  }

  KOKKOS_FORCEINLINE_FUNCTION
  float log2_(const float x) const {
    // as_int(1.0) == 2^30 - 2^23
    constexpr int32_t one_as_int = 1065353216;
    // 1/(as_int(2.0) - as_int(1.0)) == 2^-23
    constexpr float scale_down = 1.f/8388608.f;
    // 2^23 - 1
    constexpr int32_t mantissa_mask = 8388607;
    // 2^11 - 1
    constexpr int32_t low_mask = 2047;

    const int32_t x_as_int = as_int(x) - one_as_int;
    const int32_t frac_as_int = x_as_int & mantissa_mask;
    const int32_t frac_high = frac_as_int>>11;
    const int32_t frac_low  = frac_as_int & low_mask;
    const int32_t frac_squared = (frac_high*frac_high +
                                       ((frac_high*frac_low)>>10))>>1;

    return static_cast<float>(x_as_int +
                                  ((frac_as_int - frac_squared)/3)) * scale_down;
  }

  KOKKOS_FORCEINLINE_FUNCTION
  float exp2_(const float x) const {
    // as_int(1.0) == 2^30 - 2^23
    constexpr int32_t one_as_int = 1065353216;
    // as_int(2.0) - as_int(1.0) == 2^23
    constexpr float scale_up = 8388608.f;
    constexpr int32_t mantissa_mask = 8388607; // 2^23 - 1
    constexpr int32_t a = 16777216; // 2 * 2^23
    // constexpr float b = 2896.309375740099f; // 2^(23/2)
    constexpr int32_t c = 33554432; // 4 * 2^23

    const int32_t x_as_int = static_cast<int32_t>(x*scale_up);
    const int32_t frac_as_int = x_as_int & mantissa_mask;
    const int32_t frac_sqrt = static_cast<int32_t>(
      Kokkos::sqrt(scale_up*static_cast<float>(c-3*frac_as_int)));

    return as_real(x_as_int + a - frac_sqrt - frac_as_int + one_as_int);
  }

 private:
  KOKKOS_FORCEINLINE_FUNCTION int64_t as_int(double f) const {
    return *reinterpret_cast<int64_t*>(&f);
  }

  KOKKOS_FORCEINLINE_FUNCTION double as_real(int64_t i) const {
    return *reinterpret_cast<double*>(&i);
  }

  KOKKOS_FORCEINLINE_FUNCTION int32_t as_int(float f) const {
    return *reinterpret_cast<int32_t*>(&f);
  }

  KOKKOS_FORCEINLINE_FUNCTION float as_real(int32_t i) const {
    return *reinterpret_cast<float*>(&i);
  }
};

} // namespace Primitive
#endif  // EOS_PRIMITIVE_SOLVER_LOGS_HPP_
