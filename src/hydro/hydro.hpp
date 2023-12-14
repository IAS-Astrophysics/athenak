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
class EquationOfState;
class Coordinates;
class Viscosity;
class Conduction;
class SourceTerms;
class Driver;

// constants that enumerate Hydro Riemann Solver options
enum class Hydro_RSolver {advect, llf, hlle, hllc, roe,    // non-relativistic
                          llf_sr, hlle_sr, hllc_sr,        // SR
                          llf_gr, hlle_gr};                // GR

//----------------------------------------------------------------------------------------
//! \struct HydroTaskIDs
//  \brief container to hold TaskIDs of all hydro tasks

struct HydroTaskIDs {
  TaskID irecv;
  TaskID copyu;
  TaskID flux;
  TaskID sendf;
  TaskID recvf;
  TaskID expl;
  TaskID restu;
  TaskID sendu;
  TaskID recvu;
  TaskID bcs;
  TaskID prol;
  TaskID c2p;
  TaskID newdt;
  TaskID csend;
  TaskID crecv;
};

namespace hydro {

//----------------------------------------------------------------------------------------
//! \class Hydro

class Hydro {
 public:
  Hydro(MeshBlockPack *ppack, ParameterInput *pin);
  ~Hydro();

  // data
  ReconstructionMethod recon_method;
  Hydro_RSolver rsolver_method;
  EquationOfState *peos;  // chosen EOS

  int nhydro;             // number of hydro variables (5/4 for ideal/isothermal EOS)
  int nscalars;           // number of passive scalars
  DvceArray5D<Real> u0;   // conserved variables
  DvceArray5D<Real> w0;   // primitive variables

  DvceArray5D<Real> coarse_u0;  // conserved variables on 2x coarser grid (for SMR/AMR)
  DvceArray5D<Real> coarse_w0;  // primitive variables on 2x coarser grid (for SMR/AMR)

  // Boundary communication buffers and functions for u
  BoundaryValuesCC *pbval_u;

  // Object(s) for extra physics (viscosity, thermal conduction, srcterms)
  Viscosity *pvisc = nullptr;
  Conduction *pcond = nullptr;
  SourceTerms *psrc = nullptr;

  // following only used for time-evolving flow
  DvceArray5D<Real> u1;       // conserved variables at intermediate step
  DvceFaceFld5D<Real> uflx;   // fluxes of conserved quantities on cell faces
  Real dtnew;

  // following used for FOFC
  DvceArray4D<bool> fofc;  // flag for each cell to indicate if FOFC is needed
  bool use_fofc = false;   // flag to enable FOFC

  // container to hold names of TaskIDs
  HydroTaskIDs id;

  // functions...
  void AssembleHydroTasks(TaskList &start, TaskList &run, TaskList &end);
  // ...in start task list
  TaskStatus InitRecv(Driver *d, int stage);
  // ...in run task list
  TaskStatus CopyCons(Driver *d, int stage);
  TaskStatus Fluxes(Driver *d, int stage);
  TaskStatus SendFlux(Driver *d, int stage);
  TaskStatus RecvFlux(Driver *d, int stage);
  TaskStatus ExpRKUpdate(Driver *d, int stage);
  TaskStatus RestrictU(Driver *d, int stage);
  TaskStatus SendU(Driver *d, int stage);
  TaskStatus RecvU(Driver *d, int stage);
  TaskStatus ApplyPhysicalBCs(Driver* pdrive, int stage);
  TaskStatus Prolongate(Driver* pdrive, int stage);
  TaskStatus ConToPrim(Driver *d, int stage);
  TaskStatus NewTimeStep(Driver *d, int stage);
  // ...in end task list
  TaskStatus ClearSend(Driver *d, int stage);
  TaskStatus ClearRecv(Driver *d, int stage);  // also in Driver::Initialize

  // CalculateFluxes function templated over Riemann Solvers
  template <Hydro_RSolver T>
  void CalculateFluxes(Driver *d, int stage);

  // first-order flux correction
  void FOFC(Driver *d, int stage);

 private:
  MeshBlockPack* pmy_pack;  // ptr to MeshBlockPack containing this Hydro
  DvceArray5D<Real> utest;  // scratch array for FOFC
};

} // namespace hydro
#endif // HYDRO_HYDRO_HPP_
