#ifndef RADIATION_M1_HPP
#define RADIATION_M1_HPP
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_m1.hpp
//  \brief definitions for Grey M1 radiation class

#include <map>
#include <memory>
#include <string>

#include "athena.hpp"
#include "athena_tensor.hpp"
#include "bvals/bvals.hpp"
#include "parameter_input.hpp"
#include "radiation_m1/radiation_m1_helpers.hpp"
#include "radiation_m1/radiation_m1_macro.hpp"
#include "radiation_m1/radiation_m1_params.hpp"
#include "radiation_m1/radiation_m1_roots_brent.hpp"
#include "radiation_m1/radiation_m1_tensors.hpp"
#include "tasklist/task_list.hpp"

namespace radiationm1 {

//----------------------------------------------------------------------------------------
//! \struct RadiationTaskIDs
//  \brief container to hold TaskIDs of all radiation M1 tasks
struct RadiationM1TaskIDs {
  TaskID irecv;
  TaskID copyu;
  TaskID calcclosure;
  TaskID flux;
  TaskID sendf;
  TaskID recvf;
  TaskID rkupdt;
  TaskID srctrms;
  TaskID restu;
  TaskID sendu;
  TaskID recvu;
  TaskID bcs;
  TaskID prol;
  TaskID newdt;
  TaskID csend;
  TaskID crecv;
};

//----------------------------------------------------------------------------------------
//! \class RadiationM1
//  \brief class for grey M1
class RadiationM1 {
public:
  BrentFunctor BrentFunc;

  RadiationM1(MeshBlockPack *ppack, ParameterInput *pin);
  ~RadiationM1();

  int nvars;                // no. of evolved variables per species
  int nspecies;             // no. of species
  int source_limiter;       // src limiter param to avoid non-physical states
  int nvarstot;             // total no. of evolved variables
  RadiationM1Params params; // user parameters for grey M1

  DvceArray5D<Real> u0;             // evolved variables
  DvceArray5D<Real> coarse_u0;      // evolved variables on 2x coarser grid
  DvceArray5D<Real> P_dd;           // lab radiation pressure
  DvceArray4D<bool> radiation_mask; // radiation mask
  DvceArray5D<Real> u1;             // evolved variables at intermediate step
  DvceFaceFld5D<Real> uflx;         // fluxes of evo. quantities on cell faces
  DvceArray5D<Real> u_mu_data;      // fluid velocity
  AthenaTensor<Real, TensorSymm::NONE, 4, 1> u_mu;

  MeshBoundaryValuesCC *pbval_u; // Communication buffers and functions for u
  RadiationM1TaskIDs id;         // container to hold names of TaskIDs
  Real dtnew;

  DvceArray1D<Real> beam_source_vals; // values of 1d beams

  // functions...
  void
  AssembleRadiationM1Tasks(std::map<std::string, std::shared_ptr<TaskList>> tl);
  // ...in "before_stagen_tl" list
  TaskStatus InitRecv(Driver *d, int stage);
  // ...in "stagen_tl" list
  TaskStatus CopyCons(Driver *d, int stage);
  TaskStatus CalcClosure(Driver *d, int stage);
  TaskStatus CalculateFluxes(Driver *d, int stage);
  TaskStatus SendFlux(Driver *d, int stage);
  TaskStatus RecvFlux(Driver *d, int stage);
  TaskStatus TimeUpdate(Driver *d, int stage);
  TaskStatus RadiationM1SrcTerms(Driver *d, int stage);
  TaskStatus RestrictU(Driver *d, int stage);
  TaskStatus SendU(Driver *d, int stage);
  TaskStatus RecvU(Driver *d, int stage);
  TaskStatus ApplyPhysicalBCs(Driver *pdrive, int stage);
  TaskStatus Prolongate(Driver *pdrive, int stage);
  TaskStatus NewTimeStep(Driver *d, int stage);
  // ...in "after_stagen_tl" list
  TaskStatus ClearSend(Driver *d, int stage);
  TaskStatus ClearRecv(Driver *d, int stage); // also in Driver::Initialize

  void
  calc_closure(const AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> &g_dd,
               const AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> &g_uu,
               const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &n_d,
               const Real &w_lorentz,
               const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &u_u,
               const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &v_d,
               const AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> &proj_ud,
               const Real &E,
               const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &F_d,
               Real &chi, AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> &P_dd,
               const RadiationM1Params &params);

private:
  MeshBlockPack *pmy_pack; // ptr to MeshBlockPack
};

// 1d beam boundary conditions
void ApplyBeamSources1D(Mesh *pmesh);

} // namespace radiationm1

#endif // RADIATION_M1_HPP
