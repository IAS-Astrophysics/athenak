#ifndef ODE_SOLVERS_FORWARD_EULER_HPP_
#define ODE_SOLVERS_FORWARD_EULER_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file forward_euler.hpp
//  \brief The implementation of a forward euler solver for solving systems of
//  ODEs

#include <string>

#include "athena.hpp"

namespace ode_solvers {

struct FESettings {
  /// The max number of subcycles the Forwared Euler solver should execute
  unsigned int fe_n_subcycle_max;
  /// CFL number for the subcycles
  Real fe_cfl;
  /// floor for calculating subcycling timestep
  Real fe_yfloor;
};

/*!
 * \brief Solve a system of ODEs using an explicit Forward Euler method.
 *
 * \details All ODE classes are required to have a handful of specific things so
 * that the ODE solvers have a common interface to work with. They need:
 *   - A `neqs` variable that specifies the number of equations. That should be
 *     the number of species plus 1 for the internal energy
 *   - `y` and `f` RegisterArray variables to hold the current state and the
 *     result of evaluating the equations respectively.
 *   - An `evaluate_function` method that computes `f` from `y`
 *
 *
 * \tparam T The type of the ODE system to solve
 */
template <typename T>
class ForwardEuler {
 public:
  // ----- Constructor & Destructor -----
  KOKKOS_FUNCTION
  ForwardEuler(FESettings const settings, T& ode_system, Real const t_start,
               Real const dt)
      : ode_system(ode_system),
        fe_cfl(settings.fe_cfl),
        fe_n_subcycle_max(settings.fe_n_subcycle_max),
        fe_yfloor(settings.fe_yfloor),
        t_start(t_start),
        dt(dt) {}
  KOKKOS_FUNCTION
  ~ForwardEuler() = default;

  // ----- Variables -----
  /// A small number approximately equal to
  /// 1024*std::numeric_limits<float>::min()
  static constexpr Real small = 1e-35;
  /// floor for calculating subcycling timestep
  const Real fe_yfloor;
  /// The CFL number for the forward euler subcycling. Lowering this has no
  /// impact on the solution for the H2 network
  const Real fe_cfl;
  /// The maximum number of forward euler iterations
  unsigned int fe_n_subcycle_max;
  /// The system of ODEs to solve
  T& ode_system;
  /// The starting time for this solve
  const Real t_start;
  /// The amount of time to evolve the system of equations
  const Real dt;
  /// If the solver failed to converge within the allocated number of cycles
  bool failed = false;

  /*!
   * \brief Get the settings for the Forward Euler ODE solver from the input
   * file
   *
   * \param pin The ParameterInput object
   * \param module The physics module that this ODE solver is called in. The
   * name should match the block name in the input file for the physics module.
   * \return FESettings The settings for the Forward Euler solver
   */
  static FESettings GetSettings(ParameterInput* pin, std::string module) {
    unsigned int fe_n_subcycle_max =
        pin->GetOrAddInteger(module, "fe_n_subcycle_max", 1e5);
    Real fe_cfl = pin->GetOrAddReal(module, "fe_cfl", 0.1);
    Real fe_yfloor = pin->GetOrAddReal(module, "fe_yfloor", 1.e-3);

    return FESettings{fe_n_subcycle_max, fe_cfl, fe_yfloor};
  }

  KOKKOS_FUNCTION
  void SolveODE() {
    // ------ Solve the ODEs ------
    unsigned int icount = 0;
    Real t_now = t_start;
    Real t_end = t_start + dt;
    while (t_now < t_end) {
      // Evaluate the ODEs
      ode_system.evaluate_function();

      // Compute the subcycle timestep
      Real dt_subcycle;
      {
        // loop through the ODE and evaluated values to find the timestep
        dt_subcycle = Kokkos::reduction_identity<Real>::min();
        for (int i = 0; i < ode_system.neqs; i++) {
          // put floor in y for computing the timestep
          Real const yf = Kokkos::max(ode_system.y(i), fe_yfloor);

          // Compute the value to reduce
          // NOLINTNEXTLINE(build/include_what_you_use)
          dt_subcycle = Kokkos::min(
              dt_subcycle, Kokkos::abs(yf / (ode_system.f(i) + small)));
        }
        dt_subcycle = fe_cfl * dt_subcycle;

        // If t_now + dt_subcycle is greater than t_end then lower the
        // timestep accordingly
        // NOLINTNEXTLINE(build/include_what_you_use)
        dt_subcycle = Kokkos::min(dt_subcycle, t_end - t_now);
      }

      // Advance one subcycle
      {
        for (int i = 0; i < ode_system.neqs; i++) {
          ode_system.y(i) += ode_system.f(i) * dt_subcycle;
        }
      }

      // Update timing
      t_now += dt_subcycle;
      icount++;

      // check if convergence is established within fe_n_subcycle_max.  If not,
      // trigger a failure
      if (icount > fe_n_subcycle_max) {
        failed = true;
        break;
      }
    }
  }
};
}  // namespace ode_solvers
#endif  // ODE_SOLVERS_FORWARD_EULER_HPP_
