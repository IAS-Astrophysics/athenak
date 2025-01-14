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

//#include "athena.hpp"
//#include "athena_tensor.hpp"
#include "bvals/bvals.hpp"
//#include "parameter_input.hpp"
//#include "tasklist/task_list.hpp"

namespace radiationm1 {

//----------------------------------------------------------------------------------------
//! \struct RadiationTaskIDs
//  \brief container to hold TaskIDs of all radiation M1 tasks

struct RadiationM1TaskIDs {
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

struct RadiationM1Params {
  Real rad_E_floor;
  Real rad_eps;
};

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
template <typename T>
T tensor_dot(const AthenaPointTensor<T, TensorSymm::NONE, 4, 1> F_d,
             const AthenaPointTensor<T, TensorSymm::NONE, 4, 1> G_d) {
  return -F_d(0) * G_d(0) + F_d(1) * G_d(1) + F_d(2) * G_d(2) + F_d(3) * G_d(3);
}

class RadiationM1 {

public:
  RadiationM1(MeshBlockPack *ppack, ParameterInput *pin);
  ~RadiationM1();

  int nvars;
  DvceArray5D<Real> u0; // conserved variables
  DvceArray5D<Real>
      coarse_u0; // conserved variables on 2x coarser grid (for SMR/AMR)
  Real rad_E_floor;

  // Boundary communication buffers and functions for u
  MeshBoundaryValuesCC *pbval_u;

  // following only used for time-evolving flow
  DvceArray5D<Real> u1;     // conserved variables at intermediate step
  DvceFaceFld5D<Real> uflx; // fluxes of conserved quantities on cell faces
  Real dtnew;

  // container to hold names of TaskIDs
  RadiationM1TaskIDs id;

  // functions...
  void
  AssembleRadiationM1Tasks(std::map<std::string, std::shared_ptr<TaskList>> tl);
  // ...in "before_stagen_tl" list
  TaskStatus InitRecv(Driver *d, int stage);
  // ...in "stagen_tl" list
  TaskStatus CopyCons(Driver *d, int stage);
  TaskStatus Fluxes(Driver *d, int stage);
  TaskStatus SendFlux(Driver *d, int stage);
  TaskStatus RecvFlux(Driver *d, int stage);
  TaskStatus RKUpdate(Driver *d, int stage);
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
  MeshBlockPack *pmy_pack; // ptr to MeshBlockPack containing this Hydro
};
} // namespace radiationm1

#endif // RADIATION_M1_HPP
