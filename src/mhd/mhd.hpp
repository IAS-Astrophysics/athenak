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
enum class MHD_RSolver {advect, llf, hlld, roe};

namespace mhd {

//----------------------------------------------------------------------------------------
//! \class MHD

class MHD
{
 public:
  MHD(MeshBlockPack *ppack, ParameterInput *pin);
  ~MHD();

  // data
  EquationOfState *peos;    // object that implements chosen EOS

  int nmhd;               // number of cons variables (5/4 for adiabatic/isothermal)
  int nscalars;           // number of passive scalars
  DvceArray5D<Real> u0;   // conserved variables
  DvceArray5D<Real> w0;   // primitive variables
  DvceFaceFld4D<Real> b0;   // face-centered magnetic fields
  DvceArray5D<Real> bcc0; // cell-centered magnetic fields`

  // Objects containing boundary communication buffers and routines for u and b
  BoundaryValueCC *pbval_u;
  BoundaryValueFC *pbval_b;

  // following only used for time-evolving flow
  DvceArray5D<Real> u1;           // conserved variables, second register
  DvceFaceFld4D<Real> b1;         // face-centered magnetic fields, second register
  DvceArray5D<Real> bcc1;         // cell-centered magnetic fields, second register
  DvceArray5D<Real> divf;         // divergence of fluxes
  DvceArray3D<Real> uflx_x1face;  // fluxes on x1-faces
  DvceArray3D<Real> uflx_x2face;  // fluxes on x2-faces
  DvceArray3D<Real> uflx_x3face;  // fluxes on x3-faces
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
  TaskStatus MHDSendU(Driver *d, int stage); 
  TaskStatus MHDRecvU(Driver *d, int stage); 
  TaskStatus MHDSendB(Driver *d, int stage); 
  TaskStatus MHDRecvB(Driver *d, int stage); 
  TaskStatus ConToPrim(Driver *d, int stage);
  TaskStatus NewTimeStep(Driver *d, int stage);
  TaskStatus MHDApplyPhysicalBCs(Driver* pdrive, int stage);

  // functions to set physical BCs for Hydro conserved variables, applied to single MB
  // specified by argument 'm'. 
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
  MeshBlockPack* pmy_pack;   // ptr to MeshBlockPack containing this MHD
  ReconstructionMethod recon_method_;
  MHD_RSolver rsolver_method_;
};

} // namespace mhd
#endif // MHD_MHD_HPP_
