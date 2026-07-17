#ifndef ODE_SOLVERS_KOKKOS_BDF_HPP_
#define ODE_SOLVERS_KOKKOS_BDF_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file kokkos_bdf.hpp
//  \brief Wrapper for the Kokkos Kernels BDF ODE solver

#include <KokkosODE_BDF.hpp>
#include <string>  // NOLINT(build/include_order)

#include "athena.hpp"

namespace ode_solvers {

struct KokkosBDFSettings {};

/*!
 * \brief Solve a system of ODEs using the BDF solver from Kokkos Kernels
 *
 * \tparam T The type of the ODE system to solve
 */
template <typename ode_t>
class KokkosBDF {
 public:
  // ----- Constructor & Destructor -----
  KOKKOS_FUNCTION
  KokkosBDF(KokkosBDFSettings const settings, ode_t& ode_system,
            Real const t_start, Real const dt)
      : ode_system(ode_system),
        t_start(t_start),
        dt(dt),
        t_end(t_start + dt),
        max_step(dt),
        temp_(&temp_buffer_[0][0], ode_t::neqs, 23 + 2 * ode_t::neqs + 4),
        temp2_(&temp2_buffer_[0][0], 6, 7) {}
  KOKKOS_FUNCTION
  ~KokkosBDF() = default;

  // ----- Variables -----
  /// The system of ODEs to solve
  ode_t& ode_system;
  /// The starting time for this solve
  const Real t_start;
  /// The amount of time to evolve the system of equations
  const Real dt;
  /// Time to integrate to
  const Real t_end;
  /// First time step size, if zero then the solver will decide
  const Real dt0 = 0.0;
  /// The maximum time step, as of Kokkos Kernels 4.4 this is not implemented so
  /// it does nothing
  const Real max_step;

  /*!
   * \brief Get the settings for the  ODE solver from the input file
   *
   * \param pin The ParameterInput object
   * \param module The physics module that this ODE solver is called in. The
   * name should match the block name in the input file for the physics module.
   * \return KokkosBDFSettings The settings for the Kokkos BDF solver
   */
  static KokkosBDFSettings GetSettings(ParameterInput* pin,
                                       std::string module) {
    return KokkosBDFSettings();
  }

  KOKKOS_FUNCTION
  void SolveODE() {
    KokkosODE::Experimental::BDFSolve(ode_system, t_start, t_end, dt0, max_step,
                                      ode_system.y, ode_system.y_new, temp_,
                                      temp2_);
  }

 private:
  // temporary storage for inside the BDF solver
  Real temp_buffer_[ode_t::neqs][23 + 2 * ode_t::neqs + 4];
  Real temp2_buffer_[6][7];
  Kokkos::View<Real**, Kokkos::LayoutRight, Kokkos::MemoryUnmanaged> temp_;
  Kokkos::View<Real**, Kokkos::LayoutRight, Kokkos::MemoryUnmanaged> temp2_;
};
}  // namespace ode_solvers
#endif  // ODE_SOLVERS_KOKKOS_BDF_HPP_
