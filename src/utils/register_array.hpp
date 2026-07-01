#ifndef UTILS_REGISTER_ARRAY_HPP_
#define UTILS_REGISTER_ARRAY_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file register_array.hpp
//  \brief contains a struct for an array stored in registers

#include <cstddef>

#include "athena.hpp"

/*!
 * \brief A struct for an array stored in registers without any kokkos view
 * overhead. Permits access via operator() and operator[], also contains the
 * size.
 *
 * \tparam T The type of the elements in the array
 * \tparam N The size of the array
 */
template <typename T, std::size_t N>
struct RegisterArray {
  const std::size_t size = N;

  KOKKOS_FORCEINLINE_FUNCTION
  T& operator[](std::size_t const i) { return raw_array[i]; }

  KOKKOS_FORCEINLINE_FUNCTION
  const T& operator[](std::size_t const i) const { return raw_array[i]; }

  KOKKOS_FORCEINLINE_FUNCTION
  T& operator()(std::size_t const i) { return raw_array[i]; }

  KOKKOS_FORCEINLINE_FUNCTION
  const T& operator()(std::size_t const i) const { return raw_array[i]; }

 private:
  T raw_array[N];
};

#endif  // UTILS_REGISTER_ARRAY_HPP_
