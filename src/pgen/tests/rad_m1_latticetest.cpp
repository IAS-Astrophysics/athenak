//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file rad_m1_latticetest.cpp
//  \brief 2D lattice test for grey M1

// C++ headers

// Athena++ headers
#include "athena.hpp"
#include "coordinates/adm.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "pgen/pgen.hpp"
#include "radiation_m1/radiation_m1.hpp"
#include "radiation_m1/radiation_m1_helpers.hpp"

KOKKOS_INLINE_FUNCTION
void LatticeOpacities(Real x1, Real x2, Real x3, Real nuidx, Real &eta_0,
                      Real &abs_0, Real &eta_1, Real &abs_1, Real &scat_1) {
  eta_0 = 0;
  abs_0 = 0;
  eta_1 = 0;
  abs_1 = 0;
  scat_1 = 0;

  if (x1 >= 3 && x1 <= 4 && x2 >= 3 && x2 <= 4) {
    eta_1 = 1.;
  }

  if ((x1 >= 1 && x1 <= 2 && x2 >= 1 && x2 <= 2) ||
      (x1 >= 3 && x1 <= 4 && x2 >= 1 && x2 <= 2) ||
      (x1 >= 5 && x1 <= 6 && x2 >= 1 && x2 <= 2) ||
      (x1 >= 2 && x1 <= 3 && x2 >= 2 && x2 <= 3) ||
      (x1 >= 4 && x1 <= 5 && x2 >= 2 && x2 <= 3) ||
      (x1 >= 1 && x1 <= 2 && x2 >= 3 && x2 <= 4) ||
      (x1 >= 5 && x1 <= 6 && x2 >= 3 && x2 <= 4) ||
      (x1 >= 2 && x1 <= 3 && x2 >= 4 && x2 <= 5) ||
      (x1 >= 4 && x1 <= 5 && x2 >= 4 && x2 <= 5) ||
      (x1 >= 1 && x1 <= 2 && x2 >= 5 && x2 <= 6) ||
      (x1 >= 5 && x1 <= 6 && x2 >= 5 && x2 <= 6)) {
    abs_1 = 10.;
  } else {
    abs_1 = 1.;
  }
}
//----------------------------------------------------------------------------------------
//! \fn void MeshBlock::UserProblem(ParameterInput *pin)
//  \brief Sets initial conditions for radiation M1 beams test

void ProblemGenerator::RadiationM1LatticeTest(ParameterInput *pin,
                                              const bool restart) {
  if (restart) return;

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;

  if (pmbp->pradm1 == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "The 2d lattice problem generator can only be run with "
                 "radiation-m1, but no "
              << "<radiation_m1> block in input file" << std::endl;
    exit(EXIT_FAILURE);
  }
  if (!pmbp->pmesh->two_d) {
    std::cout
        << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
        << std::endl
        << "The 2d lattice test problem generator can only be run with one "
           "dimension, but parfile"
        << "grid setup is not in 1d" << std::endl;
    exit(EXIT_FAILURE);
  }
  if (pmbp->pradm1->nspecies != 1) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "The 2d lattice test problem generator can only be run with "
                 "one neutrino species only!"
              << std::endl;
    exit(EXIT_FAILURE);
  }

  pmbp->pradm1->toy_opacity_fn = LatticeOpacities;

  // capture variables for kernel
  auto &indcs = pmy_mesh_->mb_indcs;
  auto &size = pmbp->pmb->mb_size;
  int &is = indcs.is;
  int &ie = indcs.ie;
  int &js = indcs.js;
  int &je = indcs.je;
  int &ks = indcs.ks;
  int &ke = indcs.ke;

  int isg = is - indcs.ng;
  int ieg = ie + indcs.ng;
  int jsg = (indcs.nx2 > 1) ? js - indcs.ng : js;
  int jeg = (indcs.nx2 > 1) ? je + indcs.ng : je;
  int ksg = (indcs.nx3 > 1) ? ks - indcs.ng : ks;
  int keg = (indcs.nx3 > 1) ? ke + indcs.ng : ke;
  int nmb = pmbp->nmb_thispack;
  auto &u_mu_ = pmbp->pradm1->u_mu;
  adm::ADM::ADM_vars &adm = pmbp->padm->adm;
  auto &beam_vals = pmbp->pradm1->beam_source_vals;

  // set user boundary conditions to true (needed for beams)
  user_bcs = true;
  user_bcs_func = radiationm1::ApplyBeamSources1D;

  Kokkos::realloc(beam_vals, 4);
  HostArray1D<Real> beam_vals_host;
  Kokkos::realloc(beam_vals_host, 4);
  Real E = 1;
  AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> F_d{};
  AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> g_uu{};
  g_uu(0, 0) = -1;
  g_uu(1, 1) = g_uu(2, 2) = g_uu(3, 3) = 1;
  Real Fx = E;
  Real Fy = 0;
  Real Fz = 0;
  pack_F_d(0, 0, 0, Fx, Fy, Fz, F_d);
  apply_floor(g_uu, E, F_d, pmbp->pradm1->params);
  beam_vals_host(M1_E_IDX) = E;
  beam_vals_host(M1_FX_IDX) = F_d(1);
  beam_vals_host(M1_FY_IDX) = F_d(2);
  beam_vals_host(M1_FZ_IDX) = F_d(3);
  printf("Beam values initialized: E = %lf, F = [%lf, %lf, %lf]\n",
         beam_vals_host(M1_E_IDX), beam_vals_host(M1_FX_IDX),
         beam_vals_host(M1_FY_IDX), beam_vals_host(M1_FZ_IDX));
  Kokkos::deep_copy(beam_vals, beam_vals_host);

  // set metric to minkowski, initialize velocity to zero
  par_for(
      "pgen_diffusiontest_metric_initialize", DevExeSpace(), 0, nmb - 1, ksg,
      keg, jsg, jeg, isg, ieg,
      KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
        for (int a = 0; a < 3; ++a)
          for (int b = a; b < 3; ++b) {
            adm.g_dd(m, a, b, k, j, i) = (a == b ? 1. : 0.);
          }

        adm.psi4(m, k, j, i) = 1.;  // adm.psi4

        adm.alpha(m, k, j, i) = 1.;

        u_mu_(m, 0, k, j, i) = 1.;
        u_mu_(m, 1, k, j, i) = 0.;
        u_mu_(m, 2, k, j, i) = 0.;
        u_mu_(m, 3, k, j, i) = 0.;
      });

  return;
}
