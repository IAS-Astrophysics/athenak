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
  TaskID c2p;
  TaskID newdt;
  TaskID csend;
  TaskID crecv;
};

//----------------------------------------------------------------------------------------
//! \enum RadiationM1Closure
//  \brief choice of M1 closure
enum RadiationM1Closure {
  Minerbo,
  Eddington,
  Thin,
};

//----------------------------------------------------------------------------------------
//! \struct RadiationM1Params
//  \brief parameters for the Grey M1 class
struct RadiationM1Params {
  Real rad_E_floor;
  Real rad_eps;
  RadiationM1Closure closure_fun;
};

class RadiationM1 {
public:
  RadiationM1(MeshBlockPack *ppack, ParameterInput *pin);
  ~RadiationM1();

  int nvars;                        // no. of evolved variables per species
  int nspecies;                     // no. of species
  int source_limiter;               // Limit the source terms to avoid nonphysical states
  int nvarstot;                     // total no. of evolved variables
  RadiationM1Params params;         // user parameters for grey M1
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
  Real dtnew{};

  DvceArray1D<Real> beam_source_vals; // values of 1d beams

  // functions...
  void
  AssembleRadiationM1Tasks(std::map<std::string, std::shared_ptr<TaskList>> tl);
  // ...in "before_stagen_tl" list
  TaskStatus InitRecv(Driver *d, int stage);
  // ...in "stagen_tl" list
  TaskStatus CopyCons(Driver *d, int stage);
  TaskStatus CalcClosure(Driver *d, int stage);
  TaskStatus Fluxes(Driver *d, int stage);
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

private:
  MeshBlockPack *pmy_pack; // ptr to MeshBlockPack
};

//----------------------------------------------------------------------------------------
//! \fn radiationm1::tensor_dot
//  \brief function to compute g^ab F_a G_b
template <typename T>
T tensor_dot(const AthenaPointTensor<T, TensorSymm::SYM2, 4, 2> g_uu,
             const AthenaPointTensor<T, TensorSymm::NONE, 4, 1> F_d,
             const AthenaPointTensor<T, TensorSymm::NONE, 4, 1> G_d) {
  T F2 = 0.;
  for (int a = 0; a < 4; ++a) {
    for (int b = 0; b < 4; ++b) {
      F2 += g_uu(a, b) * F_d(a) * G_d(b);
    }
  }
  return F2;
}

//----------------------------------------------------------------------------------------
//! \fn radiationm1::tensor_dot
//  \brief function to compute eta^ab F_a G_b
template <typename T>
T tensor_dot(const AthenaPointTensor<T, TensorSymm::NONE, 4, 1> F_d,
             const AthenaPointTensor<T, TensorSymm::NONE, 4, 1> G_d) {
  return -F_d(0) * G_d(0) + F_d(1) * G_d(1) + F_d(2) * G_d(2) + F_d(3) * G_d(3);
}

//----------------------------------------------------------------------------------------
//! \fn radiationm1::tensor_contract
//  \brief find F_a = g_ab F^b
template <typename T>
void tensor_contract(const AthenaPointTensor<T, TensorSymm::SYM2, 4, 2> g_dd,
                     const AthenaPointTensor<T, TensorSymm::NONE, 4, 1> F_u,
                     AthenaPointTensor<T, TensorSymm::NONE, 4, 1> F_d) {
  for (int a = 0; a < 4; ++a) {
    F_d(a) = 0;
    for (int b = 0; b < 4; ++b) {
      F_d(a) += g_dd(a, b) * F_u(b);
    }
  }
}
//----------------------------------------------------------------------------------------
//! \fn radiationm1::minmod2
//  \brief double minmod
KOKKOS_INLINE_FUNCTION
Real minmod2(Real rl, Real rp, Real th) {
  return Kokkos::min(1.0, Kokkos::min(th * rl, th * rp));
}

//----------------------------------------------------------------------------------------
//! \fn int radiationm1::CombinedIdx
//  \brief given flavor index, variable index and total no. of vars compute
//  combined idx
KOKKOS_INLINE_FUNCTION
int CombinedIdx(int nuidx, int varidx, int nvars) {
  return varidx + nuidx * nvars;
}

} // namespace radiationm1

#endif // RADIATION_M1_HPP
