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
enum Hydro_RSolver {advect, llf, hllc, roe};

namespace hydro {

//----------------------------------------------------------------------------------------
//! \class Hydro

class Hydro
{
 public:
  Hydro(MeshBlockPack *ppack, ParameterInput *pin);
  ~Hydro();

  // data
  EquationOfState *peos;    // object that implements chosen EOS

  int nhydro;               // number of hydro variables (5/4 for adiabatic/isothermal)
  int nscalars;             // number of passive scalars
  AthenaArray5D<Real> u0;   // conserved variables
  AthenaArray5D<Real> w0;   // primitive variables

  // following are vectors of length (#neighbors), stored as a vector of length (#MBs)
  std::vector<std::vector<BoundaryBuffer>> send_buf, recv_buf;  // send/recv buffers

  // following only used for time-evolving flow
  AthenaArray5D<Real> u1;    // conserved variables at intermediate step 
  AthenaArray5D<Real> divf;   // divergence of fluxes
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
  TaskStatus HydroApplyPhysicalBCs(Driver* pdrive, int stage);

  // functions to set physical BCs for Hydro conserved variables, applied to single MB 
  void ReflectInnerX1(int m);
  void ReflectOuterX1(int m);
  void ReflectInnerX2(int m);
  void ReflectOuterX2(int m);
  void ReflectInnerX3(int m);
  void ReflectOuterX3(int m);
  void OutflowInnerX1(int m);
  void OutflowOuterX1(int m);
  void OutflowInnerX2(int m);
  void OutflowOuterX2(int m);
  void OutflowInnerX3(int m);
  void OutflowOuterX3(int m);

 private:
  MeshBlockPack* pmy_pack;  // ptr to MeshBlockPack containing this Hydro
  ReconstructionMethod recon_method_;
  Hydro_RSolver rsolver_method_;
};

} // namespace hydro
#endif // HYDRO_HYDRO_HPP_
