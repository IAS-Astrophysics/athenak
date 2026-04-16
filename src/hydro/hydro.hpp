#ifndef HYDRO_HYDRO_HPP_
#define HYDRO_HYDRO_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file hydro.hpp
//  \brief definitions for Hydro class

#include <map>
#include <memory>
#include <string>

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
class OrbitalAdvectionCC;
class ShearingBoxCC;
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
  TaskID rkupdt;
  TaskID srctrms;
  TaskID sendu_oa;
  TaskID recvu_oa;
  TaskID restu;
  TaskID sendu;
  TaskID recvu;
  TaskID sendu_shr;
  TaskID recvu_shr;
  TaskID bcs;
  TaskID prol;
  TaskID c2p;
  TaskID newdt;
  TaskID csend;
  TaskID crecv;
};

namespace hydro {

enum class DiffusionSelection {explicit_only, sts_only};

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
  MeshBoundaryValuesCC *pbval_u;

  // Orbital advection and shearing box BCs
  OrbitalAdvectionCC *porb_u = nullptr;
  ShearingBoxCC *psbox_u = nullptr;

  // Object(s) for extra physics (viscosity, thermal conduction, srcterms)
  Viscosity *pvisc = nullptr;
  Conduction *pcond = nullptr;
  SourceTerms *psrc = nullptr;

  // following only used for time-evolving flow
  DvceArray5D<Real> u1;       // conserved variables at intermediate step
  DvceArray5D<Real> u_sts0;   // conserved variables at start of STS sweep
  DvceArray5D<Real> u_sts1;   // previous STS stage state
  DvceArray5D<Real> u_sts2;   // second previous STS stage state
  DvceArray5D<Real> u_sts_rhs;  // cached stage-1 RKL2 operator contribution
  DvceFaceFld5D<Real> uflx;   // fluxes of conserved quantities on cell faces
  Real dtnew;

  bool has_explicit_viscosity = false;
  bool has_explicit_conduction = false;
  bool has_sts_viscosity = false;
  bool has_sts_conduction = false;
  bool has_any_sts_diffusion = false;

  // following used for FOFC
  DvceArray4D<bool> fofc;  // flag for each cell to indicate if FOFC is needed
  bool use_fofc = false;   // flag to enable FOFC
  DvceArray5D<Real> utest;  // scratch array for FOFC

  // container to hold names of TaskIDs
  HydroTaskIDs id;

  // functions...
  void AssembleHydroTasks(std::map<std::string, std::shared_ptr<TaskList>> tl);
  // ...in "before_stagen_tl" list
  TaskStatus InitRecv(Driver *d, int stage);
  TaskStatus InitRecvParabolic(Driver *d, int stage);
  // ...in "stagen_tl" list
  TaskStatus CopyCons(Driver *d, int stage);
  TaskStatus Fluxes(Driver *d, int stage);
  TaskStatus SendFlux(Driver *d, int stage);
  TaskStatus RecvFlux(Driver *d, int stage);
  TaskStatus RKUpdate(Driver *d, int stage);
  TaskStatus HydroSrcTerms(Driver *d, int stage);
  TaskStatus SendU_OA(Driver *d, int stage);
  TaskStatus RecvU_OA(Driver *d, int stage);
  TaskStatus SendU_OA_Parabolic(Driver *d, int stage);
  TaskStatus RecvU_OA_Parabolic(Driver *d, int stage);
  TaskStatus RestrictU(Driver *d, int stage);
  TaskStatus SendU(Driver *d, int stage);
  TaskStatus RecvU(Driver *d, int stage);
  TaskStatus SendU_Shr(Driver *d, int stage);
  TaskStatus RecvU_Shr(Driver *d, int stage);
  TaskStatus SendU_Shr_Parabolic(Driver *d, int stage);
  TaskStatus RecvU_Shr_Parabolic(Driver *d, int stage);
  TaskStatus ApplyPhysicalBCs(Driver* pdrive, int stage);
  TaskStatus Prolongate(Driver* pdrive, int stage);
  TaskStatus ConToPrim(Driver *d, int stage);
  TaskStatus NewTimeStep(Driver *d, int stage);
  TaskStatus ClearSTSFlux(Driver *d, int stage);
  TaskStatus STSFluxes(Driver *d, int stage);
  TaskStatus STSUpdate(Driver *d, int stage);
  TaskStatus STSRefreshTimeStep(Driver *d, int stage);
  // ...in "after_stagen_tl" list
  TaskStatus ClearSend(Driver *d, int stage);
  TaskStatus ClearRecv(Driver *d, int stage);  // also in Driver::Initialize
  TaskStatus ClearSendParabolic(Driver *d, int stage);
  TaskStatus ClearRecvParabolic(Driver *d, int stage);

  // CalculateFluxes function templated over Riemann Solvers
  template <Hydro_RSolver T>
  void CalculateFluxes(Driver *d, int stage);

  // first-order flux correction
  void FOFC(Driver *d, int stage);

 private:
  void AddSelectedDiffusionFluxes(DiffusionSelection selection);
  void RecomputeTimeStepFromCurrentState(Driver *pdrive);
  MeshBlockPack* pmy_pack;  // ptr to MeshBlockPack containing this Hydro
};

} // namespace hydro
#endif // HYDRO_HYDRO_HPP_
