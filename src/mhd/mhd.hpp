#ifndef MHD_MHD_HPP_
#define MHD_MHD_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file mhd.hpp
//  \brief definitions for MHD class

#include "athena.hpp"
#include "parameter_input.hpp"
#include "tasklist/task_list.hpp"
#include "bvals/bvals.hpp"

// forward declarations
class Driver;
class EquationOfState;

// constants that enumerate MHD Riemann Solver options
enum MHD_RSolver {advect, llf, hlld, roe};

namespace mhd {

//----------------------------------------------------------------------------------------
//! \class MHD

class MHD
{
 public:
  MHD(Mesh *pm, ParameterInput *pin, int gid);
  ~MHD();

  // data
  EquationOfState *peos;    // object that implements chosen EOS

  int nhydro;               // number of hydro variables (5/4 for adiabatic/isothermal)
  int nscalars;             // number of passive scalars
  AthenaArray4D<Real> u0;   // conserved variables
  AthenaArray4D<Real> w0;   // primitive variables
  FaceArray3D<Real>   b0;   // face-centered magnetic fields

  std::vector<BoundaryBuffer> send_buf, recv_buf;   // send/recv buffers for MHD comms.

  // following only used for time-evolving flow
  AthenaArray4D<Real> u1;     // conserved variables at intermediate step 
  AthenaArray4D<Real> divf;   // divergence of fluxes
  FaceArray3D<Real>   b1;     // face-centered magnetic fields at intermediate step
  AthenaArray3D<Real> uflx_x1face;  // fluxes on x1-faces
  AthenaArray3D<Real> uflx_x2face;  // fluxes on x2-faces
  AthenaArray3D<Real> uflx_x3face;  // fluxes on x3-faces
  Real dtnew;

  // functions
  void MHDStageStartTasks(TaskList &tl, TaskID start, std::vector<TaskID> &added);
  void MHDStageRunTasks(TaskList &tl, TaskID start, std::vector<TaskID> &added);
  void MHDStageEndTasks(TaskList &tl, TaskID start, std::vector<TaskID> &added);
  TaskStatus MHDInitRecv(Driver *d, int stage);
  TaskStatus MHDClearRecv(Driver *d, int stage);
  TaskStatus MHDClearSend(Driver *d, int stage);
  TaskStatus MHDCopyCons(Driver *d, int stage);
  TaskStatus MHDDivFlux(Driver *d, int stage);
  TaskStatus MHDUpdate(Driver *d, int stage);
  TaskStatus MHDSend(Driver *d, int stage); 
  TaskStatus MHDReceive(Driver *d, int stage); 
  TaskStatus ConToPrim(Driver *d, int stage);
  TaskStatus NewTimeStep(Driver *d, int stage);

 private:
  Mesh* pmesh_;   // ptr to Mesh containing this MHD
  int my_mbgid_;  // GridID of MeshBlock contianing this MHD
  ReconstructionMethod recon_method_;
  MHD_RSolver rsolver_method_;
};

} // namespace mhd
#endif // MHD_MHD_HPP_
