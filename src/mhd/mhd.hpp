#ifndef MHD_MHD_HPP_
#define MHD_MHD_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file mhd.hpp
//  \brief definitions for MHD class

#include <map>
#include "athena.hpp"
#include "parameter_input.hpp"
#include "tasklist/task_list.hpp"
#include "bvals/bvals.hpp"

// forward declarations
class EquationOfState;
class Driver;

// constants that enumerate MHD Riemann Solver options
enum class MHD_RSolver {advect, llf, hlld, roe};

// constants that enumerate Hydro tasks
enum class MHDTaskName {undef=0, init_recv, copy_cons, calc_flux, update,
  send_u, recv_u, corner_emf, ct, send_b, recv_b, phys_bcs, cons2prim, newdt,
  clear_send};


namespace mhd {

//----------------------------------------------------------------------------------------
//! \class MHD

class MHD
{
 public:
  MHD(MeshBlockPack *ppack, ParameterInput *pin);
  ~MHD();

  // data
  EquationOfState *peos;   // chosen EOS

  int nmhd;                // number of cons variables (5/4 for adiabatic/isothermal)
  int nscalars;            // number of passive scalars
  DvceArray5D<Real> u0;    // conserved variables
  DvceArray5D<Real> w0;    // primitive variables
  DvceFaceFld4D<Real> b0;  // face-centered magnetic fields
  DvceArray5D<Real> bcc0;  // cell-centered magnetic fields`

  // Objects containing boundary communication buffers and routines for u and b
  BoundaryValueCC *pbval_u;
  BoundaryValueFC *pbval_b;

  // following only used for time-evolving flow
  DvceArray5D<Real> u1;       // conserved variables, second register
  DvceFaceFld4D<Real> b1;     // face-centered magnetic fields, second register
  DvceFaceFld5D<Real> uflx;   // fluxes of conserved quantities on cell faces
  DvceEdgeFld4D<Real> efld;   // edge-centered electric fields (fluxes of B)
  Real dtnew;

  // map for associating MHDTaskName with TaskID
  std::map<MHDTaskName, TaskID> mhd_tasks;

  // functions
  void AssembleStageStartTasks(TaskList &tl, TaskID start);
  void AssembleStageRunTasks(TaskList &tl, TaskID start);
  void AssembleStageEndTasks(TaskList &tl, TaskID start);
  TaskStatus InitRecv(Driver *d, int stage);
  TaskStatus ClearRecv(Driver *d, int stage);
  TaskStatus ClearSend(Driver *d, int stage);
  TaskStatus CopyCons(Driver *d, int stage);
  TaskStatus CalcFluxes(Driver *d, int stage);
  TaskStatus CornerE(Driver *d, int stage);
  TaskStatus CT(Driver *d, int stage);
  TaskStatus Update(Driver *d, int stage);
  TaskStatus SendU(Driver *d, int stage); 
  TaskStatus RecvU(Driver *d, int stage); 
  TaskStatus SendB(Driver *d, int stage); 
  TaskStatus RecvB(Driver *d, int stage); 
  TaskStatus ConToPrim(Driver *d, int stage);
  TaskStatus NewTimeStep(Driver *d, int stage);
  TaskStatus ApplyPhysicalBCs(Driver* pdrive, int stage); // in file mhd/bvals dir

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
  void ShearInnerX1(int m);
  void ShearOuterX1(int m);

 private:
  MeshBlockPack* pmy_pack;   // ptr to MeshBlockPack containing this MHD
  ReconstructionMethod recon_method_;
  MHD_RSolver rsolver_method_;
  // temporary variables used to store face-centered electric fields returned by RS
  DvceArray4D<Real> e3x1, e2x1;
  DvceArray4D<Real> e1x2, e3x2;
  DvceArray4D<Real> e2x3, e1x3;
  DvceArray4D<Real> e1_cc, e2_cc, e3_cc;
};

} // namespace mhd
#endif // MHD_MHD_HPP_
