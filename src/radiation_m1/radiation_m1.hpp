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
#include "radiation_m1_roots_hybridsj.hpp"
// #include "radiation_m1_sources.hpp"
#include "radiation_m1_roots_fns.hpp"
#include "tasklist/task_list.hpp"

namespace radiationm1 {
struct SrcParams;
using ToyOpacityFn = void (*)(Real x1, Real x2, Real x3, Real dx, Real dy, Real dz,
                              Real nuidx, Real& eta_0, Real& abs_0, Real& eta_1,
                              Real& abs_1, Real& scat_1);

//----------------------------------------------------------------------------------------
//! \struct RadiationTaskIDs
//  \brief container to hold TaskIDs of all radiation M1 tasks
struct RadiationM1TaskIDs {
  TaskID irecv;
  TaskID copyu;
  TaskID closure;
  TaskID flux;
  TaskID sendf;
  TaskID recvf;
  TaskID rkupdt;
  TaskID mattersrc;
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
  RadiationM1(MeshBlockPack* ppack, ParameterInput* pin);
  ~RadiationM1();

  static BrentFunctor BrentFunc;
  static HybridsjFunctor HybridsjFunc;
  static BrentFunctorInv BrentFuncInv;
  ToyOpacityFn toy_opacity_fn = nullptr;

  int nvars;                 // no. of evolved variables per species
  int nspecies;              // no. of species
  int nvarstot;              // total no. of evolved variables
  RadiationM1Params params;  // user parameters for grey M1

  DvceArray5D<Real> u0;              // evolved variables
  DvceArray5D<Real> coarse_u0;       // evolved variables on 2x coarser grid
  DvceArray5D<Real> chi;             // Eddington factor
  DvceArray4D<bool> radiation_mask;  // radiation mask
  DvceArray5D<Real> u1;              // evolved variables at intermediate step
  DvceFaceFld5D<Real> uflx;          // fluxes of evo. quantities on cell faces
  DvceArray5D<Real> u_mu_data;       // fluid velocity
  DvceArray5D<Real> eta_0;           // number emissivity coefficient
  DvceArray5D<Real> abs_0;           // number absorptivity coefficient
  DvceArray5D<Real> eta_1;           // energy emissivity coefficient
  DvceArray5D<Real> abs_1;           // energy absorptivity coefficient
  DvceArray5D<Real> scat_1;          // energy scattering coefficient
  AthenaTensor<Real, TensorSymm::NONE, 4, 1> u_mu;  // fluid 4-velocity

  MeshBoundaryValuesCC* pbval_u;  // Communication buffers and functions for u
  RadiationM1TaskIDs id;          // container to hold names of TaskIDs
  Real dtnew{};

  DvceArray1D<Real> beam_source_vals;  // values of 1d beams

  // functions...
  void AssembleRadiationM1Tasks(std::map<std::string, std::shared_ptr<TaskList>> tl);
  // ...in "before_stagen_tl" list
  TaskStatus InitRecv(Driver* d, int stage);
  // ...in "stagen_tl" list
  TaskStatus CopyCons(Driver* d, int stage);
  TaskStatus CalcClosure(Driver* d, int stage);
  TaskStatus CalculateFluxes(Driver* d, int stage);
  TaskStatus SendFlux(Driver* d, int stage);
  TaskStatus RecvFlux(Driver* d, int stage);
  TaskStatus TimeUpdate(Driver* d, int stage);
  TaskStatus CalcOpacityNurates(Driver* pdrive, int stage);
  TaskStatus CalcOpacityToy(Driver* pdrive, int stage);
  TaskStatus RestrictU(Driver* d, int stage);
  TaskStatus SendU(Driver* d, int stage);
  TaskStatus RecvU(Driver* d, int stage);
  TaskStatus ApplyPhysicalBCs(Driver* pdrive, int stage);
  TaskStatus Prolongate(Driver* pdrive, int stage);
  TaskStatus NewTimeStep(Driver* d, int stage);
  // ...in "after_stagen_tl" list
  TaskStatus ClearSend(Driver* d, int stage);
  TaskStatus ClearRecv(Driver* d, int stage);  // also in Driver::Initialize

  KOKKOS_INLINE_FUNCTION static void calc_closure(
      const AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2>& g_dd,
      const AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2>& g_uu,
      const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1>& n_d, const Real& w_lorentz,
      const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1>& u_u,
      const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1>& v_d,
      const AthenaPointTensor<Real, TensorSymm::NONE, 4, 2>& proj_ud, const Real& E,
      const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1>& F_d, Real& chi,
      AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2>& P_dd,
      const RadiationM1Params& params);

  KOKKOS_INLINE_FUNCTION void calc_inv_closure(
      const AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2>& g_uu,
      const AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2>& g_dd,
      const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1>& n_u,
      const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1>& n_d,
      const AthenaPointTensor<Real, TensorSymm::NONE, 4, 2>& gamma_ud,
      const Real& w_lorentz, const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1>& u_u,
      const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1>& u_d,
      const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1>& v_d,
      const AthenaPointTensor<Real, TensorSymm::NONE, 4, 2>& proj_ud, const Real& chi,
      const Real& J, const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1>& H_d, Real& E,
      AthenaPointTensor<Real, TensorSymm::NONE, 4, 1>& F_d,
      const RadiationM1Params& params);

  KOKKOS_INLINE_FUNCTION
  static HybridsjSignal prepare_closure(const Real q[4], SrcParams& p,
                                 const RadiationM1Params& param);

  KOKKOS_INLINE_FUNCTION
  static HybridsjSignal prepare_sources(const Real q[4], SrcParams& p);

  KOKKOS_INLINE_FUNCTION
  static HybridsjSignal prepare(const Real q[4], SrcParams& p, const RadiationM1Params& params);

  KOKKOS_INLINE_FUNCTION SrcSignal source_update(
    const Real &cdt, const Real &alp,
    const AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> &g_dd,
    const AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> &g_uu,
    const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &n_d,
    const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &n_u,
    const AthenaPointTensor<Real, TensorSymm::NONE, 4, 2> &gamma_ud,
    const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &u_d,
    const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &u_u,
    const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &v_d,
    const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &v_u,
    const AthenaPointTensor<Real, TensorSymm::NONE, 4, 2> &proj_ud, const Real &W,
    const Real &Eold, const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &Fold_d,
    const Real &Estar, const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &Fstar_d,
    const Real &eta, const Real &kabs, const Real &kscat, Real &chi, Real &Enew,
    AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &Fnew_d,
    const RadiationM1Params &params_);

 private:
  MeshBlockPack* pmy_pack;  // ptr to MeshBlockPack
};

// 1d beam boundary conditions
void ApplyBeamSources1D(Mesh* pmesh);
}  // namespace radiationm1

#endif  // RADIATION_M1_HPP
