//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file rad_m1_beams.cpp
//  \brief 1D beam for grey M1

// C++ headers

// Athena++ headers
#include "athena.hpp"
#include "coordinates/adm.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "pgen/pgen.hpp"
#include "radiation_m1/radiation_m1.hpp"

//----------------------------------------------------------------------------------------
//! \fn void MeshBlock::UserProblem(ParameterInput *pin)
//  \brief Sets initial conditions for radiation M1 beams test

void ProblemGenerator::RadiationM1BeamTest(ParameterInput *pin,
                                           const bool restart) {
  if (restart)
    return;

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;

  if (pmbp->pradm1 == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "The 1d diffusion problem generator can only be run with "
                 "radiation-m1, but no "
              << "<radiation_m1> block in input file" << std::endl;
    exit(EXIT_FAILURE);
  }

  if (!pmbp->pmesh->one_d) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "The 1d beam test problem generator can only be run with one "
                 "dimension, but parfile"
              << "grid setup is not in 1d" << std::endl;
    exit(EXIT_FAILURE);
  }

  if (pmbp->pradm1->nspecies != 1) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "The 1d diffusion problem generator can only be run with "
                 "one neutrino species only!"
              << std::endl;
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
  auto &beam_vals = pmbp->pradm1->beam_source_vals;

  // set user boundary conditions to true (needed for beams)
  user_bcs = true;
  user_bcs_func = radiationm1::ApplyBeamSources1D;

  Kokkos::realloc(beam_vals, 4);
  HostArray1D<Real> beam_vals_host;
  Kokkos::realloc(beam_vals_host, 4);
  beam_vals_host(0) = 1.0;
  beam_vals_host(1) = 1.0;
  beam_vals_host(2) = 0;
  beam_vals_host(3) = 0;

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

        adm.psi4(m, k, j, i) = 1.; // adm.psi4

        adm.alpha(m, k, j, i) = 1.;

        u_mu_(m, 0, k, j, i) = 1.;
        u_mu_(m, 1, k, j, i) = 0.;
        u_mu_(m, 2, k, j, i) = 0.;
        u_mu_(m, 3, k, j, i) = 0.;
      });

  return;
}
