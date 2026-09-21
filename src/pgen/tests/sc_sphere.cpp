//========================================================================================
// AthenaK astrophysical plasma code
// Copyright(C) 2024 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file sc_sphere.cpp
//! \brief Homogeneous emitting/absorbing sphere test for the multi-D SC formal solution.
//!
//! Classic short-characteristics benchmark (Davis, Stone & Jiang 2012, ApJS 199, 9): a
//! uniform sphere of radius R with constant absorption sigma_a and constant source S = b
//! sits in vacuum (sigma_a = S = 0 outside, no incoming radiation). For a point at radius
//! r < R the exact specific intensity along a ray whose direction makes cosine
//! mu_r = n.rhat with the outward radial is
//!     I(r, mu_r) = b [ 1 - exp(-sigma_a * s) ],
//!     s = r*mu_r + sqrt(R^2 - r^2 (1 - mu_r^2)),
//! s being the chord length from the sphere entry point to the cell (all inside, since
//! the sphere is convex). The moments J = sum_n w_n I_n, H_i = sum_n w_n n_i I_n and
//! K_ij = sum_n w_n n_i n_j I_n of that intensity are compared to the numeric moments.
//! Because the analytic moments use the SAME angular quadrature (octant-indexed mu/wmu,
//! as ComputeJ / ComputeHK), the angle integration cancels and this isolates the
//! transport + bilinear-interpolation error.
//!
//! There is no fluid, so the pgen enrolls an opacity and an emission hook that write the
//! sphere's sigma_a and brad at every solve (see sc_hooks.hpp). The run goes through the
//! ordinary driver for one cycle, so the solve, the ghost exchange, and the moments are
//! the production path; with several meshblocks the iteration converges to the
//! single-block answer. Interior cells (r < r_mask*R) are checked, avoiding the
//! staircased sphere surface.
//!
//! Relative L2 errors of J, H and K (H and K normalised by J, since H vanishes at the
//! centre) and the L-infty error of J are written to sc_sphere-errs.dat, so the pytest
//! wrapper runs a ladder of resolutions and checks that the errors CONVERGE (the SC
//! bilinear sweep is ~1st order here, limited by the sigma_a discontinuity at the
//! staircased surface) rather than testing a single-resolution threshold. A loose in-pgen
//! tol on J only guards against a grossly broken (or NaN) sweep.

#include <cmath>
#include <cstdio>
#include <iostream>
#include <cstdlib>
#include <string>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/cell_locations.hpp"
#include "pgen/pgen.hpp"
#include "nr_radiation/nr_radiation.hpp"

namespace {
struct SphereVars {
  Real rsph, sigma, b;    // radius, absorption coefficient, source inside the sphere
  Real cx, cy, cz;        // center
  Real r_mask;            // error check restricted to r < r_mask * rsph
  Real tol;               // crash-guard on the L2 error of J
};
SphereVars sphvars;

//----------------------------------------------------------------------------------------
//! \fn void SphereOpacity() / SphereEmission()
//! \brief Uniform sphere in vacuum, written over the full arrays including ghosts (the
//! sweep reads both at ghost footpoints). Called by the module at the start of a solve.

void SphereOpacity(MeshBlockPack *pmbp) {
  auto &indcs = pmbp->pmesh->mb_indcs;
  const int is = indcs.is, js = indcs.js, ks = indcs.ks;
  const int ng = indcs.ng;
  const int n1 = indcs.nx1 + 2*ng;
  const int n2 = (indcs.nx2 > 1) ? indcs.nx2 + 2*ng : 1;
  const int n3 = (indcs.nx3 > 1) ? indcs.nx3 + 2*ng : 1;
  const int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  const int nmb1 = pmbp->nmb_thispack - 1;
  auto sigma_a = pmbp->pnrrad->sigma_a;
  auto &size = pmbp->pmb->mb_size;
  const SphereVars sv = sphvars;
  const Real rsph2 = sv.rsph*sv.rsph;
  par_for("sc_sph_sigma", DevExeSpace(), 0,nmb1, 0,n3-1, 0,n2-1, 0,n1-1,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real x = CellCenterX(i-is, nx1, size.d_view(m).x1min, size.d_view(m).x1max) - sv.cx;
    Real y = (nx2 > 1) ? CellCenterX(j-js, nx2, size.d_view(m).x2min,
                                     size.d_view(m).x2max) - sv.cy : 0.0;
    Real z = (nx3 > 1) ? CellCenterX(k-ks, nx3, size.d_view(m).x3min,
                                     size.d_view(m).x3max) - sv.cz : 0.0;
    sigma_a(m,0,k,j,i) = ((x*x + y*y + z*z) < rsph2) ? sv.sigma : 0.0;
  });
}

void SphereEmission(MeshBlockPack *pmbp) {
  auto &indcs = pmbp->pmesh->mb_indcs;
  const int is = indcs.is, js = indcs.js, ks = indcs.ks;
  const int ng = indcs.ng;
  const int n1 = indcs.nx1 + 2*ng;
  const int n2 = (indcs.nx2 > 1) ? indcs.nx2 + 2*ng : 1;
  const int n3 = (indcs.nx3 > 1) ? indcs.nx3 + 2*ng : 1;
  const int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  const int nmb1 = pmbp->nmb_thispack - 1;
  auto brad = pmbp->pnrrad->brad;
  auto &size = pmbp->pmb->mb_size;
  const SphereVars sv = sphvars;
  const Real rsph2 = sv.rsph*sv.rsph;
  par_for("sc_sph_brad", DevExeSpace(), 0,nmb1, 0,n3-1, 0,n2-1, 0,n1-1,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real x = CellCenterX(i-is, nx1, size.d_view(m).x1min, size.d_view(m).x1max) - sv.cx;
    Real y = (nx2 > 1) ? CellCenterX(j-js, nx2, size.d_view(m).x2min,
                                     size.d_view(m).x2max) - sv.cy : 0.0;
    Real z = (nx3 > 1) ? CellCenterX(k-ks, nx3, size.d_view(m).x3min,
                                     size.d_view(m).x3max) - sv.cz : 0.0;
    brad(m,k,j,i) = ((x*x + y*y + z*z) < rsph2) ? sv.b : 0.0;
  });
}

//----------------------------------------------------------------------------------------
//! \fn void SCSphereErrors()
//! \brief Compare the converged J(r) to the analytic angular quadrature over interior
//! cells and append the relative L2 / L-infty errors to sc_sphere-errs.dat. Single rank.

void SCSphereErrors(ParameterInput *pin, Mesh *pm) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  nr_radiation::SC *psc = pmbp->pnrrad;
  auto &indcs = pm->mb_indcs;
  const int is = indcs.is, ie = indcs.ie, js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  const int nmb1 = pmbp->nmb_thispack - 1;
  const int nang = psc->pang->nang;
  const int nang_tot = psc->nang_tot;
  auto &size = pmbp->pmb->mb_size;
  const SphereVars sv = sphvars;
  const Real rsph2 = sv.rsph*sv.rsph;

  auto mom_h = Kokkos::create_mirror_view(psc->moments);
  Kokkos::deep_copy(mom_h, psc->moments);
  auto mu_h  = psc->pang->mu.h_view;
  auto wmu_h = psc->pang->wmu.h_view;

  const Real r_mask = sv.r_mask;
  const Real rchk2 = (r_mask*sv.rsph)*(r_mask*sv.rsph);

  Real max_err = 0.0, l2_num = 0.0, l2_den = 0.0;
  Real h_num = 0.0, k_num = 0.0;   // sum |H - H_ana|^2, |K - K_ana|^2 (all components)
  int ncheck = 0;
  for (int m=0; m<=nmb1; ++m) {
    for (int k=ks; k<=ke; ++k) for (int j=js; j<=je; ++j) for (int i=is; i<=ie; ++i) {
      Real x = CellCenterX(i-is, nx1, size.h_view(m).x1min, size.h_view(m).x1max) - sv.cx;
      Real y = (nx2 > 1) ? CellCenterX(j-js, nx2, size.h_view(m).x2min,
                                       size.h_view(m).x2max) - sv.cy : 0.0;
      Real z = (nx3 > 1) ? CellCenterX(k-ks, nx3, size.h_view(m).x3min,
                                       size.h_view(m).x3max) - sv.cz : 0.0;
      Real r2 = x*x + y*y + z*z;
      if (r2 >= rchk2) continue;                       // interior mask
      Real r = std::sqrt(r2);

      // analytic mean intensity: quadrature of I(r, mu_r) over the SAME octant-indexed
      // angles as ComputeHK (wmu[a], mu[oct,a,:], angg = oct*nang + a), so the angle
      // integration cancels. mu is a full 3D unit vector even in 2D (|mu| = 1), but the
      // sweep only marches the ACTIVE spatial dims, so the transport path scales with the
      // in-plane speed sqrt(mux^2+muy^2). Hence mnorm and mu_r are taken over the active
      // dims only (mu_z excluded when nx3==1); in 3D all dims are active and this reduces
      // to the plain sphere. tau = sigma_a * (unit-dir chord) / mnorm, the sweep path.
      Real j_ana = 0.0, h_ana[3] = {0.0, 0.0, 0.0};
      Real k_ana[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
      for (int angg=0; angg<nang_tot; ++angg) {
        int oct = angg / nang;
        int a   = angg - oct * nang;
        Real nx = mu_h(oct, a, 0);
        Real ny = (nx2 > 1) ? mu_h(oct, a, 1) : 0.0;
        Real nz = (nx3 > 1) ? mu_h(oct, a, 2) : 0.0;
        Real w  = wmu_h(a);
        Real mnorm = std::sqrt(nx*nx + ny*ny + nz*nz);  // norm over ACTIVE spatial dims
        Real mu_r = (r > 1.0e-12) ? (nx*x + ny*y + nz*z)/(r*mnorm) : 0.0;
        Real s = r*mu_r + std::sqrt(rsph2 - r2*(1.0 - mu_r*mu_r));  // unit-dir chord
        // sweep path = chord/mnorm
        Real tau = sv.sigma*s/mnorm;
        Real wI = w * sv.b*(1.0 - std::exp(-tau));
        // the moments use the full 3D direction cosines, as ComputeHK does
        Real mx = mu_h(oct, a, 0), my = mu_h(oct, a, 1), mz = mu_h(oct, a, 2);
        j_ana += wI;
        h_ana[0] += mx*wI;  h_ana[1] += my*wI;  h_ana[2] += mz*wI;
        k_ana[0] += mx*mx*wI;  k_ana[1] += my*my*wI;  k_ana[2] += mz*mz*wI;
        k_ana[3] += mx*my*wI;  k_ana[4] += mx*mz*wI;  k_ana[5] += my*mz*wI;
      }
      Real j_num = mom_h(m,0,k,j,i);
      Real err = std::fabs(j_num - j_ana)/j_ana;
      if (err > max_err) max_err = err;
      l2_num += (j_num - j_ana)*(j_num - j_ana);
      l2_den += j_ana*j_ana;
      for (int c=0; c<3; ++c) { Real d = mom_h(m,1+c,k,j,i) - h_ana[c]; h_num += d*d; }
      for (int c=0; c<6; ++c) { Real d = mom_h(m,4+c,k,j,i) - k_ana[c]; k_num += d*d; }
      ncheck++;
    }
  }
  Real l2_err = (l2_den > 0.0) ? std::sqrt(l2_num/l2_den) : 0.0;
  Real h_err  = (l2_den > 0.0) ? std::sqrt(h_num/l2_den) : 0.0;   // relative to J
  Real k_err  = (l2_den > 0.0) ? std::sqrt(k_num/l2_den) : 0.0;   // relative to J

  std::cout << "SC sphere: checked " << ncheck << " interior cells (r < " << r_mask
            << "R), rel L2 err  J = " << l2_err << "  H = " << h_err << "  K = " << k_err
            << "  (J Linf " << max_err << "; " << psc->niter_last << " sweeps, residual "
            << psc->resid_last << ")" << std::endl;

  // write relative L2 (RMS-L1 column) and L-infty errors for the convergence ladder
  if (global_variable::my_rank == 0) {
    std::string fname = "sc_sphere-errs.dat";
    FILE *pf = std::fopen(fname.c_str(), "r");
    if (pf != nullptr) {                       // exists -> append
      pf = std::freopen(fname.c_str(), "a", pf);
    } else {                                   // new -> write header
      pf = std::fopen(fname.c_str(), "w");
      std::fprintf(pf, "# Nx1  Nx2  Nx3   Ncycle   RMS-L1(J)    L-infty(J)   "
                       "RMS(H)/J     RMS(K)/J\n");
    }
    std::fprintf(pf, "%04d  %04d  %04d  %05d  %e %e %e %e\n",
                 pm->mesh_indcs.nx1, pm->mesh_indcs.nx2, pm->mesh_indcs.nx3, pm->ncycle,
                 l2_err, max_err, h_err, k_err);
    std::fclose(pf);
  }

  // loose crash-guard only; the real convergence check lives in the pytest wrapper
  const Real tol = sv.tol;
  if (!(l2_err < tol)) {                        // also fails on NaN
    std::cout << "### SC sphere test FAILED (crash-guard): L2 = " << l2_err
              << " not < " << tol << std::endl;
    std::exit(EXIT_FAILURE);
  }
}
}  // namespace

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::SCSphere()
//! \brief Read the sphere parameters, enroll the opacity/emission hooks (before any
//! restart early-return: hooks are not checkpointed), and register the error check.

void ProblemGenerator::SCSphere(ParameterInput *pin, const bool restart) {
  (void)restart;
  pgen_final_func = SCSphereErrors;

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->pnrrad == nullptr) {
    std::cout << "### FATAL ERROR in sc_sphere: requires a <nr_radiation> block"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (pmbp->phydro != nullptr || pmbp->pmhd != nullptr) {
    std::cout << "### FATAL ERROR in sc_sphere: takes no fluid (the hooks supply "
              << "sigma_a and emission)" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  sphvars.rsph  = pin->GetOrAddReal("problem", "radius", 0.6);
  sphvars.sigma = pin->GetOrAddReal("problem", "sigma_a", 5.0);
  sphvars.b     = pin->GetOrAddReal("problem", "source", 1.0);
  sphvars.cx    = pin->GetOrAddReal("problem", "x0", 0.0);
  sphvars.cy    = pin->GetOrAddReal("problem", "y0", 0.0);
  sphvars.cz    = pin->GetOrAddReal("problem", "z0", 0.0);
  sphvars.r_mask = pin->GetOrAddReal("problem", "r_mask", 0.6);
  sphvars.tol    = pin->GetOrAddReal("problem", "tol", 1.0e-1);

  pmbp->pnrrad->EnrollOpacityFunction(SphereOpacity);
  pmbp->pnrrad->EnrollEmissionFunction(SphereEmission);
  return;
}
