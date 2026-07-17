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
struct CDRates_t {
  RegisterArray<Real, N> creation, destruction;
};

/*!
 * \brief Compute the numerical Jacobian for a chemical network.
 *
 * \tparam network_t The network object
 * \tparam vec_type The type of y_in
 * \tparam mat_type The type of jac
 * \param[in] network The chemical network to compute the Jacobian of
 * \param[in] t Current time
 * \param[in] dt Time step
 * \param[in] y_in The current state
 * \param[out] jac The Jacobian
 */
template <class network_t, class vec_type, class mat_type>
KOKKOS_FUNCTION void numerical_jacobian(const network_t& network, const Real t,
                                        const Real dt, const vec_type& y_in,
                                        const mat_type& jac) {
  RegisterArray<Real, network.neqs> f0, yp, fp;

  // Set yp to to the unperturbed values
  for (int n = 0; n < network.neqs; ++n) {
    yp(n) = y_in(n);
  }

  // Evaluate the unperturbed f0
  network.evaluate_function(t, dt, y_in, f0);

  // The perturbation to add to each element in turn
  const Real perturbation = Kokkos::sqrt(Kokkos::ArithTraits<Real>::epsilon());

  for (int j = 0; j < network.neqs; ++j) {
    // Add the perturbation to the jth element
    yp(j) += perturbation * Kokkos::fmax(Kokkos::abs(y_in(j)), Real(1.0));

    // Compute the perturbed values of fp
    network.evaluate_function(t, dt, yp, fp);

    // Update the Jacobian
    for (int k = 0; k < network.neqs; ++k) {
      jac(k, j) = (fp(k) - f0(k)) / perturbation;
    }

    // Reset the perturbed field
    yp(j) = y_in(j);
  }
}
}  // namespace chemistry

#endif  // CHEMISTRY_CHEMISTRY_UTILS_HPP_
