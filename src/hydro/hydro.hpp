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
#include "parameter_input.hpp"
#include "tasklist/task_list.hpp"
#include "bvals/bvals.hpp"

// forward declarations
class Driver;
class EquationOfState;

// constants that enumerate Hydro Riemann Solver options
enum RiemannSolver {advect, llf, hlle, hllc, roe};

namespace hydro {

//----------------------------------------------------------------------------------------
//! \class Hydro

class Hydro
{
 public:
  Hydro(Mesh *pm, ParameterInput *pin, int gid);
  ~Hydro();

  // data
  EquationOfState *peos;    // object that implements chosen EOS

  int nhydro;               // number of hydro variables (5/4 for adiabatic/isothermal)
  int nscalars;             // number of passive scalars
  AthenaArray4D<Real> u0;   // conserved variables
  AthenaArray4D<Real> w0;   // primitive variables

  BBuffer bbuf;    // send/recv buffers and BoundaryStatus flags for Hydro comms.

  // following only used for time-evolving flow
  AthenaArray4D<Real> u1;    // conserved variables at intermediate step 
  AthenaArray4D<Real> divf;   // divergence of fluxes
  AthenaArray3D<Real> uflx_x1face;  // fluxes on x1-faces
  AthenaArray3D<Real> uflx_x2face;  // fluxes on x2-faces
  AthenaArray3D<Real> uflx_x3face;  // fluxes on x3-faces
  Real dtnew;

  // functions
  void HydroStageStartTasks(TaskList &tl, TaskID start, std::vector<TaskID> &added);
  void HydroStageRunTasks(TaskList &tl, TaskID start, std::vector<TaskID> &added);
  void HydroStageEndTasks(TaskList &tl, TaskID start, std::vector<TaskID> &added);
  TaskStatus HydroInitRecv(Driver *d, int stage);
  TaskStatus HydroClearRecv(Driver *d, int stage);
  TaskStatus HydroClearSend(Driver *d, int stage);
  TaskStatus HydroCopyCons(Driver *d, int stage);
  TaskStatus HydroDivFlux(Driver *d, int stage);
  TaskStatus HydroUpdate(Driver *d, int stage);
  TaskStatus HydroSend(Driver *d, int stage); 
  TaskStatus HydroReceive(Driver *d, int stage); 
  TaskStatus ConToPrim(Driver *d, int stage);
  TaskStatus NewTimeStep(Driver *d, int stage);

 private:
  Mesh* pmesh_;   // ptr to Mesh containing this Hydro
  int my_mbgid_;  // GridID of MeshBlock contianing this Hydro
  ReconstructionMethod recon_method_;
  RiemannSolver rsolver_method_;
};

} // namespace hydro
#endif // HYDRO_HYDRO_HPP_
