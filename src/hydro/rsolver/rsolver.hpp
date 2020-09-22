#ifndef HYDRO_RSOLVER_RSOLVER_HPP_
#define HYDRO_RSOLVER_RSOLVER_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file rsolver.hpp
//  \brief Contains data and functions that implement different Riemann solvers for
//   nonrelativistic hydrodynamics

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/meshblock.hpp"
//#include "hydro/hydro.hpp"

enum class RiemannSolverMethod {advection, llf, hlle, hllc, roe};

//enum HydroEvolution {hydro_static, kinematic, hydro_dynamic, no_evolution};
namespace hydro {

//----------------------------------------------------------------------------------------
//! \class RiemannSolver
//  \brief functions for Riemann solvers

class RiemannSolver
{
 public:
  RiemannSolver(Mesh* pm, ParameterInput* pin, int igid, bool is_adiabatic,
                bool is_dynamic);
  ~RiemannSolver() = default;

  // wrapper function that calls appropriate solver
  KOKKOS_FUNCTION
  void RSolver(TeamMember_t const &member, const int il, const  int iu, const int dir,
       const AthenaScratch2D<Real> &wl, const AthenaScratch2D<Real> &wr,
       AthenaScratch2D<Real> &flx);

  // functions that implement various solvers
  KOKKOS_FUNCTION
  void Advection(TeamMember_t const &member, const int il, const  int iu, const int dir,
       const AthenaScratch2D<Real> &wl, const AthenaScratch2D<Real> &wr,
       AthenaScratch2D<Real> &flx);

  KOKKOS_FUNCTION
  void LLF(TeamMember_t const &member, const int il, const  int iu, const int dir,
       const AthenaScratch2D<Real> &wl, const AthenaScratch2D<Real> &wr,
       AthenaScratch2D<Real> &flx);

  KOKKOS_FUNCTION
  void HLLE(TeamMember_t const &member, const int il, const  int iu, const int dir,
       const AthenaScratch2D<Real> &wl, const AthenaScratch2D<Real> &wr,
       AthenaScratch2D<Real> &flx);

  KOKKOS_FUNCTION
  void HLLC(TeamMember_t const &member, const int il, const  int iu, const int dir,
       const AthenaScratch2D<Real> &wl, const AthenaScratch2D<Real> &wr,
       AthenaScratch2D<Real> &flx);

  KOKKOS_FUNCTION
  void Roe(TeamMember_t const &member, const int il, const  int iu, const int dir,
       const AthenaScratch2D<Real> &wl, const AthenaScratch2D<Real> &wr,
       AthenaScratch2D<Real> &flx);

 private:
  Mesh *pmesh_;
  int my_mbgid_;
  RiemannSolverMethod rsolver_method_;   // enum that selects which solver to use
};

} // namespace hydro

#endif // HYDRO_RSOLVER_RSOLVER_HPP_
