//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file rsolver.cpp
//  \brief implements ctor and fns for RiemannSolver base class

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
//#include "hydro
#include "hydro/rsolver/rsolver.hpp"

namespace hydro {

//----------------------------------------------------------------------------------------
// RSolver constructor

RiemannSolver::RiemannSolver(Mesh* pm, ParameterInput* pin, int igid, bool is_adiabatic,
  bool is_dynamic) : pmesh_(pm), my_mbgid_(igid)
{
  // select Riemann solver (no default).  Test for compatibility of options
  std::string rsolver = pin->GetString("hydro","rsolver");
    
  if (rsolver.compare("advection") == 0) {
    if (is_dynamic) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ 
                << std::endl << "<hydro>/rsolver = '" << rsolver
                << "' cannot be used with hydrodynamic problems" << std::endl;
      std::exit(EXIT_FAILURE);
    } else {
      rsolver_method_ = RiemannSolverMethod::advection;
    }
  } else if (!is_dynamic) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "<hydro>/rsolver = '" << rsolver
              << "' cannot be used with non-hydrodynamic problems" << std::endl;
    std::exit(EXIT_FAILURE);
  } else if (rsolver.compare("llf") == 0) {
    rsolver_method_ = RiemannSolverMethod::llf;
  } else if (rsolver.compare("hlle") == 0) {
    rsolver_method_ = RiemannSolverMethod::hlle;
  } else if (rsolver.compare("hllc") == 0) {
    if (is_adiabatic) {
      rsolver_method_ = RiemannSolverMethod::hllc;
    } else {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ 
                << std::endl << "<hydro>/rsolver = '" << rsolver
                << "' cannot be used with isothermal EOS" << std::endl;
      std::exit(EXIT_FAILURE);
    }
  } else if (rsolver.compare("roe") == 0) {
    rsolver_method_ = RiemannSolverMethod::roe;
  } else {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "<hydro> rsolver = '" << rsolver << "' not implemented" << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

//----------------------------------------------------------------------------------------
// RSolver() 
  
KOKKOS_FUNCTION
void RiemannSolver::RSolver(TeamMember_t const &member, const int il, const  int iu,
     const int dir, const AthenaScratch2D<Real> &wl, const AthenaScratch2D<Real> &wr,
     AthenaScratch2D<Real> &flx)
{                  
  switch (rsolver_method_) {
    case RiemannSolverMethod::advection:
      Advection(member, il, iu, dir, wl, wr, flx);
      break;
    case RiemannSolverMethod::llf:
      LLF(member, il, iu, dir, wl, wr, flx);
      break;
    case RiemannSolverMethod::hlle:
      HLLE(member, il, iu, dir, wl, wr, flx);
      break;
    case RiemannSolverMethod::hllc:
      HLLC(member, il, iu, dir, wl, wr, flx);
      break;
    case RiemannSolverMethod::roe:
      Roe(member, il, iu, dir, wl, wr, flx);
      break;
    default: 
      break; 
  }
  return;
} 

} // namespace hydro
