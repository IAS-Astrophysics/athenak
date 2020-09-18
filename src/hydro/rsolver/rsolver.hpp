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
#include "mesh/meshblock.hpp"
#include "parameter_input.hpp"

namespace hydro {

//----------------------------------------------------------------------------------------
//! \class RiemannSolver
//  \brief abstract base class for all RiemannSolver classes

class RiemannSolver
{
 public:
  RiemannSolver(Mesh* pm, ParameterInput* pin, int igid);
  virtual ~RiemannSolver() = default;

  virtual void RSolver(const int il, const  int iu, const int dir,
                       const AthenaArray2D<Real> &wl, const AthenaArray2D<Real> &wr,
                       AthenaArray2D<Real> &flx) = 0;

 protected:
  Mesh *pmesh_;
  int my_mbgid_;
};

//----------------------------------------------------------------------------------------
//! \class Advection
//  \brief derived RiemannSolver class for pure advection problems

class Advection : public RiemannSolver
{
 public:
  Advection(Mesh* pm, ParameterInput* pin, int igid);
  void RSolver(const int il, const  int iu, const int dir, const AthenaArray2D<Real> &wl,
               const AthenaArray2D<Real> &wr, AthenaArray2D<Real> &flx) override;
};

//----------------------------------------------------------------------------------------
//! \class LLF
//  \brief derived RiemannSolver class for local Lax-Friedrichs (LLF) hydro solver

class LLF : public RiemannSolver
{
 public:
  LLF(Mesh* pm, ParameterInput* pin, int igid);
  void RSolver(const int il, const  int iu, const int dir, const AthenaArray2D<Real> &wl,
               const AthenaArray2D<Real> &wr, AthenaArray2D<Real> &flx) override;
};

//----------------------------------------------------------------------------------------
//! \class HLLE
//  \brief derived RiemannSolver class for Harten-Lax-van Leer hydro solver with Einfeldt
//  fix to prevent unphysical solutions (HLLE)
    
class HLLE : public RiemannSolver
{   
 public:
  HLLE(Mesh* pm, ParameterInput* pin, int igid);
  void RSolver(const int il, const  int iu, const int dir, const AthenaArray2D<Real> &wl,
               const AthenaArray2D<Real> &wr, AthenaArray2D<Real> &flx) override;
};  

//----------------------------------------------------------------------------------------
//! \class HLLC
//  \brief derived RiemannSolver class for HLLC hydro solver
    
class HLLC : public RiemannSolver
{   
 public:
  HLLC(Mesh* pm, ParameterInput* pin, int igid);
  void RSolver(const int il, const  int iu, const int dir, const AthenaArray2D<Real> &wl,
               const AthenaArray2D<Real> &wr, AthenaArray2D<Real> &flx) override;
};  

//----------------------------------------------------------------------------------------
//! \class Roe
//  \brief derived RiemannSolver class for Roe hydro solver

class Roe : public RiemannSolver
{
 public:
  Roe(Mesh* pm, ParameterInput* pin, int igid);
  void RSolver(const int il, const  int iu, const int dir, const AthenaArray2D<Real> &wl,
               const AthenaArray2D<Real> &wr, AthenaArray2D<Real> &flx) override;
};

} // namespace hydro

#endif // HYDRO_RSOLVER_RSOLVER_HPP_
