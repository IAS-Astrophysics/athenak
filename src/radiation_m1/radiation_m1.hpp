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
#include "radiation_m1/radiation_m1_calc_closure.hpp"
#include "radiation_m1/radiation_m1_macro.hpp"
#include "radiation_m1/radiation_m1_roots.hpp"
#include "radiation_m1/radiation_m1_tensors.hpp"
#include "radiation_m1/radiation_m1_params.hpp"
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

class BrentFunctor {
public:
  KOKKOS_INLINE_FUNCTION
  Real operator()(
      Real xi, const AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> &g_dd,
      const AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> &g_uu,
      const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &n_d,
      const Real &w_lorentz,
      const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &u_u,
      const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &v_d,
      const AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> &proj_ud,
      const Real &E, const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &F_d,
      const RadiationM1Params &params) {
    AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> P_dd;
    apply_closure(g_dd, g_uu, n_d, w_lorentz, u_u, v_d, proj_ud, E, F_d,
                  minerbo(xi), P_dd, params);

    AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> rT_dd;
    assemble_rT(n_d, E, F_d, P_dd, rT_dd);

    const Real J = calc_J_from_rT(rT_dd, u_u);

    AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> H_d;
    calc_H_from_rT(rT_dd, u_u, proj_ud, H_d);

    const Real H2 = tensor_dot(g_uu, H_d, H_d);
    return SQ(J * xi) - H2;
  }
};

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

void ApplyBeamSources1D(Mesh *pmesh);
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

// Computes the closure in the lab frame with a rootfinding procedure
KOKKOS_INLINE_FUNCTION
void RadiationM1::calc_closure(
    const AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> &g_dd,
    const AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> &g_uu,
    const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &n_d,
    const Real &w_lorentz,
    const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &u_u,
    const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &v_d,
    const AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> &proj_ud,
    const Real &E, const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &F_d,
    Real &chi, AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> &P_dd,
    const RadiationM1Params &params) {
  // These are special cases for which no root finding is needed
  if (params.closure_fun == Eddington) {
    chi = 1. / 3.;
    apply_closure(g_dd, g_uu, n_d, w_lorentz, u_u, v_d, proj_ud, E, F_d, chi,
                  P_dd, params);
    return;
  }
  if (params.closure_fun == Thin) {
    chi = 1.0;
    apply_closure(g_dd, g_uu, n_d, w_lorentz, u_u, v_d, proj_ud, E, F_d, chi,
                  P_dd, params);
    return;
  }
  if (params.closure_fun == Minerbo) {
    Real x_lo = 0.;
    Real x_md = 0.5;
    Real x_hi = 1.;
    Real root{};
    BrentState state{};

    // Initialize rootfinder
    BrentDekkerRoot brent_dekker;
    int closure_maxiter = 100;
    Real closure_epsilon = 1e-6;
    BrentSignal ierr = brent_dekker.BrentInitialize(
        BrentFunc, x_lo, x_hi, root, state, g_dd, g_uu, n_d, w_lorentz, u_u,
        v_d, proj_ud, E, F_d, params);

    if (ierr == BRENT_EINVAL) {
      double const z_ed = BrentFunc(0., g_dd, g_uu, n_d, w_lorentz, u_u, v_d,
                                    proj_ud, E, F_d, params);
      double const z_th = BrentFunc(1., g_dd, g_uu, n_d, w_lorentz, u_u, v_d,
                                    proj_ud, E, F_d, params);
      if (Kokkos::abs(z_th) < Kokkos::abs(z_ed)) {
        chi = 1.0;
      } else {
        chi = 1. / 3.;
      }
      apply_closure(g_dd, g_uu, n_d, w_lorentz, u_u, v_d, proj_ud, E, F_d, chi,
                    P_dd, params);
      return;
    }
    if (ierr != BRENT_SUCCESS) {
      printf("Unexpected error in BrentInitialize.\n");
      exit(EXIT_FAILURE);
    }

    // Rootfinding
    int iter = 0;
    do {
      ++iter;
      ierr = brent_dekker.BrentIterate(BrentFunc, x_lo, x_hi, root, state, g_dd,
                                       g_uu, n_d, w_lorentz, u_u, v_d, proj_ud,
                                       E, F_d, params);

      // Some nans in the evaluation. This should not happen.
      if (ierr != BRENT_SUCCESS) {
        printf("Unexpected error in BrentIterate.\n");
        exit(EXIT_FAILURE);
      }
      x_md = root;
      ierr = BrentTestInterval(x_lo, x_hi, closure_epsilon, 0);
    } while (ierr == BRENT_CONTINUE && iter < closure_maxiter);

    chi = minerbo(x_md);

    if (ierr != BRENT_SUCCESS) {
      printf("Maximum number of iterations exceeded when computing the M1 "
             "closure\n");
    }
  }
}

} // namespace radiationm1

#endif // RADIATION_M1_HPP
