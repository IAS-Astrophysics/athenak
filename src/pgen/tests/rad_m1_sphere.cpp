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
void SphereOpacities(Real x1, Real x2, Real x3, 
                      Real dx, Real dy, Real dz, Real nuidx, Real &eta_0,
                      Real &abs_0, Real &eta_1, Real &abs_1, Real &scat_1) {
  eta_0 = 0;
  abs_0 = 0;
  eta_1 = 0;
  abs_1 = 0;
  scat_1 = 0;

  Real const R = 1.0;  // Sphere radius
  if (x1 * x1 + x2 * x2 + x3 * x3 <= R*R) { // Inside the sphere
    // eta_0 = 10.0;
    // abs_0 = 10.0;
    eta_1 = 10.0;
    abs_1 = 10.0;
  } else {
    abs_0 = 0;  // Outside the sphere
  }
  int const NPOINTS = 10;
  int inside = 0;
  int count = 0;
  for (int ii = 0; ii < NPOINTS; ++ii) {
      Real const myx = (x1 - dx/2.) + (ii + 0.5)*(dx/NPOINTS);
      for (int jj = 0; jj < NPOINTS; ++jj) {
          Real const myy = (x2 - dy/2.) + (jj + 0.5)*(dy/NPOINTS);
          for (int kk = 0; kk < NPOINTS; ++kk) {
              Real const myz = (x3 - dz/2.) + (kk + 0.5)*(dz/NPOINTS);
              count++;
              if (myx*myx + myy*myy + myz*myz <= R*R) {
                  inside++;
              }
          }
      }
  }

  Real fraction_inside_sphere = static_cast<Real>(inside) / static_cast<Real>(count);

  // Apply the coefficient to kappa_a and eta
  abs_1 = fraction_inside_sphere * 10.0;
  eta_1 = fraction_inside_sphere * 10.0;
}
//----------------------------------------------------------------------------------------
//! \fn void MeshBlock::UserProblem(ParameterInput *pin)
//  \brief Sets initial conditions for radiation M1 beams test

void ProblemGenerator::RadiationM1SphereTest(ParameterInput *pin,
                                              const bool restart) {
  if (restart) return;

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;

  if (pmbp->pradm1 == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "The sphere test problem generator can only be run with "
                 "radiation-m1, but no "
              << "<radiation_m1> block in input file" << std::endl;
    exit(EXIT_FAILURE);
  }

  if (pmbp->pradm1->nspecies != 1) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "The sphere test problem generator can only be run with "
                 "one neutrino species only!"
              << std::endl;
    exit(EXIT_FAILURE);
  }

  pmbp->pradm1->toy_opacity_fn = SphereOpacities;

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

  // set metric to minkowski, initialize velocity to zero
  par_for(
      "pgen_metric_initialize", DevExeSpace(), 0, nmb - 1, ksg,
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
