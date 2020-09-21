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
  void RSolver(const int il, const  int iu, const int dir, const AthenaArray2D<Real> &wl,
               const AthenaArray2D<Real> &wr, AthenaArray2D<Real> &flx);

  // functions that implement various solvers
  void Advection(const int il, const  int iu, const int dir, const AthenaArray2D<Real> &wl,
                 const AthenaArray2D<Real> &wr, AthenaArray2D<Real> &flx);

  void LLF(const int il, const  int iu, const int dir, const AthenaArray2D<Real> &wl,
           const AthenaArray2D<Real> &wr, AthenaArray2D<Real> &flx);

  void HLLE(const int il, const  int iu, const int dir, const AthenaArray2D<Real> &wl,
            const AthenaArray2D<Real> &wr, AthenaArray2D<Real> &flx);

  void HLLC(const int il, const  int iu, const int dir, const AthenaArray2D<Real> &wl,
            const AthenaArray2D<Real> &wr, AthenaArray2D<Real> &flx);

  void Roe(const int il, const  int iu, const int dir, const AthenaArray2D<Real> &wl,
           const AthenaArray2D<Real> &wr, AthenaArray2D<Real> &flx);

 private:
  Mesh *pmesh_;
  int my_mbgid_;
  RiemannSolverMethod rsolver_method_;   // enum that selects which solver to use
};

} // namespace hydro

#endif // HYDRO_RSOLVER_RSOLVER_HPP_
