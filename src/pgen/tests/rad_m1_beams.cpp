//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file rad_m1_beams.cpp
//  \brief 1D beam for grey M1

// C++ headers

// Athena++ headers
#include <coordinates/cell_locations.hpp>

#include "athena.hpp"
#include "coordinates/adm.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "pgen/pgen.hpp"
#include "radiation_m1/radiation_m1.hpp"
#include "radiation_m1/radiation_m1_helpers.hpp"

//----------------------------------------------------------------------------------------
//! \fn void MeshBlock::UserProblem(ParameterInput *pin)
//  \brief Sets initial conditions for radiation M1 beams test

void ProblemGenerator::RadiationM1BeamTest(ParameterInput *pin, const bool restart) {
  if (restart) return;

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;

  if (pmbp->pradm1 == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "The 1d diffusion problem generator can only be run with "
                 "radiation-m1, but no "
              << "<radiation_m1> block in input file" << std::endl;
    exit(EXIT_FAILURE);
  }

  if (pmbp->pmesh->three_d) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "The beam test problem generator can only be run in one/two "
                 "dimension, but parfile"
              << "grid setup is in 3d" << std::endl;
    exit(EXIT_FAILURE);
  }

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
  auto &beam_vals = pmbp->pradm1->rad_m1_beam.beam_source_vals;

  Real adm_mass = pin->GetOrAddReal("adm", "bh_mass", 1.);
  auto metric = pin->GetOrAddString("adm", "metric", "minkowski");

  // set user boundary conditions to true (needed for beams)
  user_bcs = true;
  pmbp->pradm1->params.beam_sources = true;
  if (pmbp->pmesh->one_d) {
    user_bcs_func = radiationm1::ApplyBeamSources1D;
  }
  if (pmbp->pmesh->two_d) {
    if (pmbp->pradm1->params.beam_sources) {
      pmbp->pradm1->rad_m1_beam.beam_ymin =
          pin->GetOrAddReal("problem", "beam_ymin", 3.0);
      pmbp->pradm1->rad_m1_beam.beam_ymax =
          pin->GetOrAddReal("problem", "beam_ymax", 4.0);
    }    
    if (metric == "minkowski") {
      user_bcs_func = radiationm1::ApplyBeamSources2D;
    } else {
      user_bcs_func = radiationm1::ApplyBeamSourcesBlackHole;  
    }
  }

  // set metric to minkowski, initialize velocity to zero
  par_for(
      "pgen_diffusiontest_metric_initialize", DevExeSpace(), 0, nmb - 1, ksg, keg, jsg,
      jeg, isg, ieg, KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
        if (metric == "minkowski") {
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
        } else if (metric == "isotropic") {
          Real &x1min = size.d_view(m).x1min;
          Real &x1max = size.d_view(m).x1max;
          int nx1 = indcs.nx1;
          Real x1v = CellCenterX(i - is, nx1, x1min, x1max);

          Real &x2min = size.d_view(m).x2min;
          Real &x2max = size.d_view(m).x2max;
          int nx2 = indcs.nx2;
          Real x2v = CellCenterX(j - js, nx2, x2min, x2max);

          Real r = std::sqrt(std::pow(x2v, 2) + std::pow(x1v, 2));

          // set metric
          for (int a = 0; a < 3; ++a) {
            for (int b = a; b < 3; ++b) {
              Real eta_ij = (a == b ? 1. : 0.);
              adm.g_dd(m, a, b, k, j, i) = eta_ij;
            }
          }
          adm.psi4(m, k, j, i) = std::pow(1.0 + 0.5 * adm_mass / r, 4);
          for (int a = 0; a < 3; ++a) {
            for (int b = a; b < 3; ++b) {
              adm.g_dd(m, a, b, k, j, i) *= adm.psi4(m, k, j, i);
            }
          }
          adm.alpha(m, k, j, i) = (1.0 - 0.5 * adm_mass / r) / (1.0 + 0.5 * adm_mass / r);
          adm.beta_u(m, 0, k, j, i) = 0;
          adm.beta_u(m, 1, k, j, i) = 0;
          adm.beta_u(m, 2, k, j, i) = 0;

          // set fluid velocity
          u_mu_(m, 0, k, j, i) =
              1. / adm.alpha(m, k, j, i);  // for isotropic coordinates only
          u_mu_(m, 1, k, j, i) = 0.;
          u_mu_(m, 2, k, j, i) = 0.;
          u_mu_(m, 3, k, j, i) = 0.;
        } else {
          Real &x1min = size.d_view(m).x1min;
          Real &x1max = size.d_view(m).x1max;
          int nx1 = indcs.nx1;
          Real x1v = CellCenterX(i - is, nx1, x1min, x1max);

          Real &x2min = size.d_view(m).x2min;
          Real &x2max = size.d_view(m).x2max;
          int nx2 = indcs.nx2;
          Real x2v = CellCenterX(j - js, nx2, x2min, x2max);

          Real r = std::sqrt(std::pow(x2v, 2) + std::pow(x1v, 2));
          Real lvec[3] = {x1v / r, x2v / r, 0};  // x3v = 0 in 2d

          // set metric
          for (int a = 0; a < 3; ++a)
            for (int b = a; b < 3; ++b) {
              Real eta_ij = (a == b) ? 1. : 0.;
              adm.g_dd(m, a, b, k, j, i) = eta_ij + 2. * adm_mass * lvec[a] * lvec[b] / r;
            }
          adm.psi4(m, k, j, i) = 1.;
          adm.alpha(m, k, j, i) = sqrt(r / (r + 2. * adm_mass));
          adm.beta_u(m, 0, k, j, i) =
              2. * adm_mass * adm.alpha(m, k, j, i) * adm.alpha(m, k, j, i) * lvec[0] / r;
          adm.beta_u(m, 1, k, j, i) =
              2. * adm_mass * adm.alpha(m, k, j, i) * adm.alpha(m, k, j, i) * lvec[1] / r;
          adm.beta_u(m, 2, k, j, i) =
              2. * adm_mass * adm.alpha(m, k, j, i) * adm.alpha(m, k, j, i) * lvec[2] / r;

          // set fluid velocity
          u_mu_(m, 0, k, j, i) = 1. / adm.alpha(m, k, j, i);
          u_mu_(m, 1, k, j, i) =
              -(1. / adm.alpha(m, k, j, i)) * adm.beta_u(m, 0, k, j, i);
          u_mu_(m, 2, k, j, i) =
              -(1. / adm.alpha(m, k, j, i)) * adm.beta_u(m, 1, k, j, i);
          u_mu_(m, 3, k, j, i) =
              -(1. / adm.alpha(m, k, j, i)) * adm.beta_u(m, 2, k, j, i);
        }
      });

  Kokkos::realloc(beam_vals, 4);
  HostArray1D<Real> beam_vals_host;
  Kokkos::realloc(beam_vals_host, 4);
  if (pmbp->pmesh->one_d) {
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
    Kokkos::printf("Beam values initialized: E = %lf, F = [%lf, %lf, %lf]\n",
           beam_vals_host(M1_E_IDX), beam_vals_host(M1_FX_IDX), beam_vals_host(M1_FY_IDX),
           beam_vals_host(M1_FZ_IDX));
    Kokkos::deep_copy(beam_vals, beam_vals_host);
  }
  return;
}
