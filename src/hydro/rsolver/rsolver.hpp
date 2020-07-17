#ifndef HYDRO_RSOLVER_RSOLVER_HPP_
#define HYDRO_RSOLVER_RSOLVER_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file rsolver.hpp
//  \brief defines abstract base class RiemannSolver, and various derived classes
//  Each derived class contains data and functions that implement different Riemann
//  solvers for nonrelativistic hydrodynamics

#include "athena.hpp"
#include "athena_arrays.hpp"
#include "mesh/meshblock.hpp"
#include "parameter_input.hpp"

namespace hydro {

//----------------------------------------------------------------------------------------
//! \class RiemannSolver
//  \brief abstract base class for all RiemannSolver classes

class RiemannSolver {
 public:
  RiemannSolver(Hydro *phyd, std::unique_ptr<ParameterInput> &pin);
  virtual ~RiemannSolver() = default;

  Hydro *pmy_hydro;

  virtual void RSolver(const int il, const  int iu, const int dir,
    const AthenaArray<Real> &wl, const AthenaArray<Real> &wr, AthenaArray<Real> &flx) = 0;

 protected:
};

//----------------------------------------------------------------------------------------
//! \class Advection
//  \brief derived RiemannSolver class for pure advection problems

class Advection : public RiemannSolver {
 public:
  Advection(Hydro *phyd, std::unique_ptr<ParameterInput> &pin);

  void RSolver(const int il, const  int iu, const int dir, const AthenaArray<Real> &wl,
    const AthenaArray<Real> &wr, AthenaArray<Real> &flx) override;

 private:
};

//----------------------------------------------------------------------------------------
//! \class LLF
//  \brief derived RiemannSolver class for local Lax-Friedrichs (LLF) hydro solver

class LLF : public RiemannSolver {
 public:
  LLF(Hydro *phyd, std::unique_ptr<ParameterInput> &pin);

  void RSolver(const int il, const  int iu, const int dir, const AthenaArray<Real> &wl,
    const AthenaArray<Real> &wr, AthenaArray<Real> &flx) override;

 private:
};

} // namespace hydro

#endif // HYDRO_RSOLVER_RSOLVER_HPP_
