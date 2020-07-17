#ifndef HYDRO_HYDRO_HPP_
#define HYDRO_HYDRO_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file hydro.hpp
//  \brief definitions for Hydro class

#include "athena.hpp"
#include "athena_arrays.hpp"
#include "parameter_input.hpp"
#include "hydro/eos/eos.hpp"
#include "reconstruct/reconstruct.hpp"
#include "hydro/rsolver/rsolver.hpp"

namespace hydro {

// constants that enumerate Hydro physics options
//enum class HydroEOS {adiabatic, isothermal};
//enum class HydroRiemannSolver {llf, hlle, hllc, roe};

// constants that determine array index of Hydro variables
//enum ConsIndex {IDN=0, IM1=1, IM2=2, IM3=3, IEN=4};
//enum PrimIndex {IVX=1, IVY=2, IVZ=3, IPR=4};

//----------------------------------------------------------------------------------------
//! \class Hydro

class Hydro {
 public:
  Hydro(MeshBlock *pmb, std::unique_ptr<ParameterInput> &pin);
  ~Hydro();

  // data
  MeshBlock* pmy_mblock;              // ptr to MeshBlock containing this Hydro
  HydroEOS hydro_eos;                 // enum storing choice for EOS
  ReconstructionMethod hydro_recon;   // enum storing choice of reconstruction method
  HydroRiemannSolver hydro_rsolver;   // enum storing choice of Riemann solver

  EquationOfState *peos;      // object that implements chosen EOS
  Reconstruction  *precon;    // object that implements chosen reconstruction methods
  RiemannSolver   *prsolver;  // object that implements chosen Riemann solver

  int nhydro;             // number of conserved variables (5/4 for adiabatic/isothermal)
  AthenaArray<Real> u;    // conserved variables
  AthenaArray<Real> u1;   // conserved variables at intermediate step 

  // functions
  void HydroDivFlux(AthenaArray<Real> &divf);
  void UpdateHydro(AthenaArray<Real> &u0, AthenaArray<Real> &u1, AthenaArray<Real> &divf);

 private:
  AthenaArray<Real> w_,wl_,wr_,uflux_;   // 1 spatial-D scratch vectors
  AthenaArray<Real> divf_;              // divergence of fluxes (3 spatial-D)

};

} // namespace hydro
#endif // HYDRO_HYDRO_HPP_
