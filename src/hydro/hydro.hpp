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
#include "tasklist/task_list.hpp"
#include "hydro/eos/eos.hpp"
#include "reconstruct/reconstruct.hpp"
#include "hydro/rsolver/rsolver.hpp"

class Driver;

namespace hydro {

// constants that enumerate Hydro physics options
enum class HydroEOS {adiabatic, isothermal};
enum class HydroEvolution {hydro_static, kinematic, hydro_dynamic, no_evolution};
enum class HydroReconMethod {donor_cell, piecewise_linear, piecewise_parabolic};
enum class HydroRiemannSolver {advection, llf, hlle, hllc, roe};

// constants that determine array index of Hydro variables
//enum ConsIndex {IDN=0, IM1=1, IM2=2, IM3=3, IEN=4};
//enum PrimIndex {IVX=1, IVY=2, IVZ=3, IPR=4};

//----------------------------------------------------------------------------------------
//! \class Hydro

class Hydro {
 public:
  Hydro(Mesh *pm, ParameterInput *pin, int gid);
  ~Hydro();

  // data
  HydroEOS hydro_eos;                 // enum storing choice of EOS
  HydroEvolution hydro_evol;          // enum storing choice of time evolution
  HydroReconMethod hydro_recon;       // enum storing choice of reconstruction method
  HydroRiemannSolver hydro_rsolver;   // enum storing choice of Riemann solver

  EquationOfState *peos;      // object that implements chosen EOS
  Reconstruction  *precon;    // object that implements chosen reconstruction methods
  RiemannSolver   *prsolver;  // object that implements chosen Riemann solver
  BoundaryValues  *pbvals;

  int nhydro;             // number of conserved variables (5/4 for adiabatic/isothermal)
  AthenaArray<Real> u0;   // conserved variables
  AthenaArray<Real> w0;   // primitive variables

  // following only used for time-evolving flow
  AthenaArray<Real> u1;           // 4D conserved variables at intermediate step 
  AthenaArray<Real> divf;         // 4D divergence of fluxes (3 spatial-D)
  AthenaArray<Real> uflx_x1face;  // 3D fluxes on x1-faces (used in flux correction step)
  AthenaArray<Real> uflx_x2face;  // 3D fluxes on x2-faces
  AthenaArray<Real> uflx_x3face;  // 3D fluxes on x3-faces
  Real dtnew;

  // functions
//  void HydroDivFlux(AthenaArray<Real> &u);
//  void UpdateHydro(AthenaArray<Real> &u0, AthenaArray<Real> &u1, AthenaArray<Real> &divf);
  void HydroAddTasks(TaskList &tl);
  TaskStatus CopyConserved(Driver *d, int stage) {
    if (stage == 1) {
      int size = u0.GetSize();
      for (int n=0; n<size; ++n) { u1(n) = u0(n); }
    }
    return TaskStatus::complete;
  }
  TaskStatus HydroDivFlux(Driver *d, int stage);
  TaskStatus HydroUpdate(Driver *d, int stage);
  TaskStatus HydroSend(Driver *d, int stage); 
  TaskStatus HydroReceive(Driver *d, int stage); 
  TaskStatus ConToPrim(Driver *d, int stage);
  TaskStatus NewTimeStep(Driver *d, int stage);

 private:
  Mesh* pmesh_;                 // ptr to Mesh containing this Hydro
  int my_mbgid_;
  AthenaArray<Real> w_,wl_,wl_jp1,wl_kp1,wr_,uflux_;   // 1-spatialD scratch vectors

};

} // namespace hydro
#endif // HYDRO_HYDRO_HPP_
