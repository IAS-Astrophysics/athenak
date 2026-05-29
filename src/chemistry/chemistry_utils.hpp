#ifndef CHEMISTRY_CHEMISTRY_UTILS_HPP_
#define CHEMISTRY_CHEMISTRY_UTILS_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file chemistry_utils.hpp
//  \brief utilities for chemistry

#include "athena.hpp"
#include "utils/register_array.hpp"

namespace chemistry {

/*!
 * \brief A struct to hold the creation and destruction rates
 *
 * \tparam N The size of each array
 */
template <std::size_t N>
struct CDRates {
  RegisterArray<Real, N> creation, destruction;
};

}  // namespace chemistry

#endif  // CHEMISTRY_CHEMISTRY_UTILS_HPP_
