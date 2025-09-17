#ifndef UTILS_TOV_TOV_UTILS_HPP_
#define UTILS_TOV_TOV_UTILS_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file tov_utils.hpp

#include "athena.hpp"

namespace tov {

KOKKOS_INLINE_FUNCTION
static Real Interpolate(Real x,
                        const Real x1, const Real x2, const Real y1, const Real y2) {
  //return ((y2 - y1)*x + (y1*x2 - y2*x1))/(x2 - x1);
  Real t = (x - x1)/(x2 - x1);
  return y1*(1. - t) + y2*t;
}

enum class LocationTag {Host, Device};

// UsesYe is a trait which checks if a particular EOS has the `GetYeFromRho` function.
// The construction is a bit opaque thanks to the joy of C++ templates, but the idea is as
// follows: by default, UsesYe is false. It takes two template parameters, but the second
// parameter is void by default. There is a specialization of this template to fill out
// the second parameter. It assumes the existence of a function `GetYeFromRho` which is
// templated over LocationTag and takes one Real as an argument. The compiler attempts
// to "call" it in a sense to check if it's well-formed, and if it is, UsesYe is true. If
// the template is not well-formed because `GetYeFromRho` doesn't exist, the template is
// ignored thanks to SFINAE, and it falls back to the default case.
//
// The reason for doing this is so that pgens using the TOV solver don't need to know
// implementation details about specific equations of state. At the time of writing, only
// EOSCompOSE/TabulatedEOS provide information about Ye, but this may not always be the
// case in the future.
template<class TOVEOS, class = void>
constexpr bool UsesYe = false;

template<class TOVEOS>
constexpr bool UsesYe<
  TOVEOS,
  std::void_t<
    decltype(std::declval<TOVEOS>().
             template GetYeFromRho<LocationTag::Host>(std::declval<Real>()))
  >
> = true;


} // namespace tov

#endif // UTILS_TOV_TOV_UTILS_HPP_
