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
class EquationOfState;
class Coordinates;
class Viscosity;
class Resistivity;
class Conduction;
class SourceTerms;
class Driver;

// function ptr for user-defined MHD boundary functions enrolled in problem generator
namespace mhd {
using MHDBoundaryFnPtr = void (*)(int m, Mesh* pm, MHD* pmhd, DvceArray5D<Real> &u);
}

// constants that enumerate MHD Riemann Solver options
enum class MHD_RSolver {advect, llf, hlle, hlld, roe,   // non-relativistic
                        llf_sr, hlle_sr,                // SR
                        hlle_gr};                       // GR

//----------------------------------------------------------------------------------------
//! \struct MHDTaskIDs
//  \brief container to hold TaskIDs of all mhd tasks

struct MHDTaskIDs {
  TaskID irecv;
  TaskID copyu;
  TaskID flux;
  TaskID sendf;
  TaskID recvf;
  TaskID expl;
  TaskID restu;
  TaskID sendu;
  TaskID recvu;
  TaskID efld;
  TaskID sende;
  TaskID recve;
  TaskID ct;
  TaskID restb;
  TaskID sendb;
  TaskID recvb;
  TaskID bcs;
  TaskID c2p;
  TaskID newdt;
  TaskID clear;
};

namespace mhd {

//----------------------------------------------------------------------------------------
//! \class MHD

class MHD {
 public:
  MHD(MeshBlockPack *ppack, ParameterInput *pin);
  ~MHD();

  // data
  // flags to denote relativistic dynamics
  bool is_special_relativistic = false;
  bool is_general_relativistic = false;

  ReconstructionMethod recon_method;
  MHD_RSolver rsolver_method;
  EquationOfState *peos;   // chosen EOS

  int nmhd;                // number of mhd variables (5/4 for ideal/isothermal EOS)
  int nscalars;            // number of passive scalars
  DvceArray5D<Real> u0;    // conserved variables
  DvceArray5D<Real> w0;    // primitive variables
  DvceFaceFld4D<Real> b0;  // face-centered magnetic fields
  DvceArray5D<Real> bcc0;  // cell-centered magnetic fields`

  DvceArray5D<Real> coarse_u0;    // conserved variables on 2x coarser grid (for SMR/AMR)
  DvceFaceFld4D<Real> coarse_b0;  // face-centered B-field on 2x coarser grid

  // Objects containing boundary communication buffers and routines for u and b
  BoundaryValuesCC *pbval_u;
  BoundaryValuesFC *pbval_b;
  MHDBoundaryFnPtr MHDBoundaryFunc[6];

  // Object(s) for extra physics (viscosity, resistivity, thermal conduction, srcterms)
  Viscosity *pvisc = nullptr;
  Resistivity *presist = nullptr;
  Conduction *pcond = nullptr;
  SourceTerms *psrc = nullptr;

  // following only used for time-evolving flow
  DvceArray5D<Real> u1;       // conserved variables, second register
  DvceFaceFld4D<Real> b1;     // face-centered magnetic fields, second register
  DvceFaceFld5D<Real> uflx;   // fluxes of conserved quantities on cell faces
  DvceEdgeFld4D<Real> efld;   // edge-centered electric fields (fluxes of B)
  Real dtnew;

  // container to hold names of TaskIDs
  MHDTaskIDs id;

  // functions
  void AssembleMHDTasks(TaskList &start, TaskList &run, TaskList &end);
  TaskStatus InitRecv(Driver *d, int stage);
  TaskStatus ClearRecv(Driver *d, int stage);
  TaskStatus ClearSend(Driver *d, int stage);
  TaskStatus CopyCons(Driver *d, int stage);
  TaskStatus SendU(Driver *d, int stage);
  TaskStatus RecvU(Driver *d, int stage);
  TaskStatus SendE(Driver *d, int stage);
  TaskStatus RecvE(Driver *d, int stage);
  TaskStatus SendB(Driver *d, int stage);
  TaskStatus RecvB(Driver *d, int stage);
  TaskStatus SendFlux(Driver *d, int stage);
  TaskStatus RecvFlux(Driver *d, int stage);
  TaskStatus RestrictU(Driver *d, int stage);
  TaskStatus RestrictB(Driver *d, int stage);
  TaskStatus ConToPrim(Driver *d, int stage);
  TaskStatus CornerE(Driver *d, int stage);
  TaskStatus CT(Driver *d, int stage);
  TaskStatus ExpRKUpdate(Driver *d, int stage);
  TaskStatus NewTimeStep(Driver *d, int stage);
  TaskStatus ApplyPhysicalBCs(Driver* pdrive, int stage); // file in mhd/bvals dir

  // CalculateFluxes function templated over Riemann Solvers
  template <MHD_RSolver T>
  TaskStatus CalcFluxes(Driver *d, int stage);

 private:
  MeshBlockPack* pmy_pack;   // ptr to MeshBlockPack containing this MHD
  // temporary variables used to store face-centered electric fields returned by RS
  DvceArray4D<Real> e3x1, e2x1;
  DvceArray4D<Real> e1x2, e3x2;
  DvceArray4D<Real> e2x3, e1x3;
  DvceArray4D<Real> e1_cc, e2_cc, e3_cc;
};

} // namespace mhd
#endif // MHD_MHD_HPP_
