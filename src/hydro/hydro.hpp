#ifndef HYDRO_HYDRO_HPP_
#define HYDRO_HYDRO_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file hydro.hpp
//  \brief definitions for Hydro class

#include <vector>
#include "athena.hpp"
#include "athena_arrays.hpp"
#include "parameter_input.hpp"
#include "tasklist/task_list.hpp"
#include "hydro/eos/eos.hpp"
#include "reconstruct/reconstruct.hpp"
#include "hydro/rsolver/rsolver.hpp"

class Driver;

namespace hydro {

// constants that enumerate Hydro dynamics options
enum class HydroEvolution {hydro_static, kinematic, hydro_dynamic, no_evolution};

//----------------------------------------------------------------------------------------
//! \class Hydro

class Hydro
{
 public:
  Hydro(Mesh *pm, ParameterInput *pin, int gid);
  ~Hydro();

  // data
  HydroEvolution hydro_evol;  // enum storing choice of time evolution
  EquationOfState *peos;      // object that implements chosen EOS

  int nhydro;             // number of conserved variables (5/4 for adiabatic/isothermal)
  AthenaArray<Real> u0;   // conserved variables
  AthenaArray<Real> w0;   // primitive variables

  // following only used for time-evolving flow
  Reconstruction  *precon;    // object that implements chosen reconstruction methods
  RiemannSolver   *prsolver;  // object that implements chosen Riemann solver
  AthenaArray<Real> u1;           // 4D conserved variables at intermediate step 
  AthenaArray<Real> divf;         // 4D divergence of fluxes (3 spatial-D)
  AthenaArray<Real> uflx_x1face;  // 3D fluxes on x1-faces (used in flux correction step)
  AthenaArray<Real> uflx_x2face;  // 3D fluxes on x2-faces
  AthenaArray<Real> uflx_x3face;  // 3D fluxes on x3-faces
  Real dtnew;

  // functions
  void HydroAddTasks(TaskList &tl, TaskID start, std::vector<TaskID> &added);
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
