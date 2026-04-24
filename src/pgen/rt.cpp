//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file rt.cpp
//! \brief Problem generator for RT instabilty.
//!
//! Note the gravitational acceleration is hardwired to be 0.1. Density difference is
//! hardwired to be 3.0 and is set by the input parameter `problem/drat`.
//! To reproduces 2D results of Liska & Wendroff set it to 2.0,
//! while for the 3D results of Dimonte et al use 3.0.
//!
//! FOR 2D HYDRO:
//! Problem domain should be -1/6 < x < 1/6; -0.5 < y < 0.5 with gamma=1.4 to match Liska
//! & Wendroff. Interface is at y=0; perturbation added to Vy. Gravity acts in y-dirn.
//! Special reflecting boundary conditions added in x2 to improve hydrostatic eqm
//! (prevents launching of weak waves) Atwood number A=(d2-d1)/(d2+d1)=1/3. Options:
//!    - iprob = 1  -- Perturb V2 using single mode
//!    - iprob != 1 -- Perturb V2 using multiple mode
//!
//! FOR 3D:
//! Problem domain should be -0.5 < x < 0.5; -0.5 < y < 0.5, -1. < z < 1., gamma=5/3 to
//! match Dimonte et al.  Interface is at z=0; perturbation added to Vz. Gravity acts in
//! z-dirn. Special reflecting boundary conditions added in x3.  A=1/2.  Options:
//!    - iprob = 1 -- Perturb V3 using single mode
//!    - iprob = 2 -- Perturb V3 using multiple mode
//!    - iprob = 3 -- B rotated by "angle" at interface, multimode perturbation
//!
//! REFERENCE: R. Liska & B. Wendroff, SIAM J. Sci. Comput., 25, 995 (2003)

// C++ headers
#include <cmath>
#include <iostream> // cout

// Athena++ headers
#include "athena.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "srcterms/srcterms.hpp"
#include "utils/random.hpp"
#include "pgen.hpp"

#include <Kokkos_Random.hpp>

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::UserProblem()
//  \brief Problem Generator for the Rayleigh-Taylor instability test

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  if (restart) return;
  if (pmy_mesh_->one_d) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "rti problem generator only works in 2D/3D" << std::endl;
    exit(EXIT_FAILURE);
  }

  Real kx = 2.0*(M_PI)/(pmy_mesh_->mesh_size.x1max - pmy_mesh_->mesh_size.x1min);
  Real ky = 2.0*(M_PI)/(pmy_mesh_->mesh_size.x2max - pmy_mesh_->mesh_size.x2min);
  Real kz = 2.0*(M_PI)/(pmy_mesh_->mesh_size.x3max - pmy_mesh_->mesh_size.x3min);

  // Read perturbation amplitude, problem switch, density ratio
  Real amp = pin->GetReal("problem","amp");
  int iprob = pin->GetInteger("problem","iprob");
  Real drat = pin->GetOrAddReal("problem","drat",3.0);
  bool smooth_interface = pin->GetOrAddBoolean("problem","smooth_interface",false);

  // capture variables for kernel
  auto &indcs = pmy_mesh_->mb_indcs;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  auto &size = pmbp->pmb->mb_size;

  // Select either Hydro or MHD
  DvceArray5D<Real> u0_;
  Real gm1, p0;
  Real grav_acc;
  if (pmbp->phydro != nullptr) {
    grav_acc = pin->GetReal("hydro","const_accel_val");
    u0_ = pmbp->phydro->u0;
    gm1 = (pmbp->phydro->peos->eos_data.gamma) - 1.0;
    p0 = 1.0/(pmbp->phydro->peos->eos_data.gamma);
    p0 = pin->GetOrAddReal("problem", "p0", p0);
  } else if (pmbp->pmhd != nullptr) {
    grav_acc = pin->GetReal("mhd","const_accel_val");
    u0_ = pmbp->pmhd->u0;
    gm1 = (pmbp->pmhd->peos->eos_data.gamma) - 1.0;
    p0 = 1.0/(pmbp->pmhd->peos->eos_data.gamma);
    p0 = pin->GetOrAddReal("problem", "p0", p0);
  }

  // Ensure that p0 is sufficiently large to avoid negative pressures
  if (pmbp->pmesh->two_d) {
    p0 -= grav_acc*pmy_mesh_->mesh_size.x2max;
  } else {
    p0 -= grav_acc*pmy_mesh_->mesh_size.x3max;
  }

  // 2D PROBLEM ----------------------------------------------------------------

  if (pmbp->pmesh->two_d) {
    Kokkos::Random_XorShift64_Pool<> rand_pool64(pmbp->gids);
    par_for("rt2d", DevExeSpace(), 0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real &x1min = size.d_view(m).x1min;
      Real &x1max = size.d_view(m).x1max;
      int nx1 = indcs.nx1;
      Real x1v = CellCenterX(i-is, nx1, x1min, x1max);

      Real &x2min = size.d_view(m).x2min;
      Real &x2max = size.d_view(m).x2max;
      int nx2 = indcs.nx2;
      Real x2v = CellCenterX(j-js, nx2, x2min, x2max);

      Real den=1.0;
      Real sigma = 0.01;
      if (smooth_interface) {
        den = 0.5*((drat + 1.0) + (drat - 1.0)*tanh(x2v/sigma));
      } else {
        if (x2v > 0.0) den *= drat;
      }

      if (iprob == 1) {
        u0_(m,IM2,k,j,i) = (1.0 + cos(kx*x1v))*(1.0 + cos(ky*x2v))/4.0;
      } else {
        auto rand_gen = rand_pool64.get_state();  // get random number state this thread
        u0_(m,IM2,k,j,i) = (rand_gen.frand()-0.5)*(1.0 + cos(ky*x2v))/4.0;
        rand_pool64.free_state(rand_gen);  // free state for use by other threads
      }

      u0_(m,IDN,k,j,i) = den;
      u0_(m,IM1,k,j,i) = 0.0;
      u0_(m,IM2,k,j,i) *= (den*amp);
      u0_(m,IM3,k,j,i) = 0.0;
      u0_(m,IEN,k,j,i) = (p0 + grav_acc*den*x2v)/gm1 + 0.5*SQR(u0_(m,IM2,k,j,i))/den;
    });

    // initialize magnetic fields if MHD
    if (pmbp->pmhd != nullptr) {
      // Read magnetic field strength
      Real bx = pin->GetReal("problem","b0");
      auto &b0 = pmbp->pmhd->b0;
      par_for("pgen_b0", DevExeSpace(), 0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
        b0.x1f(m,k,j,i) = bx;
        b0.x2f(m,k,j,i) = 0.0;
        b0.x3f(m,k,j,i) = 0.0;
        if (i==ie) b0.x1f(m,k,j,i+1) = bx;
        if (j==je) b0.x2f(m,k,j+1,i) = 0.0;
        if (k==ke) b0.x3f(m,k+1,j,i) = 0.0;
        u0_(m,IEN,k,j,i) += 0.5*bx*bx;
      });
    }

  // 3D PROBLEM ----------------------------------------------------------------

  } else {
    Kokkos::Random_XorShift64_Pool<> rand_pool64(pmbp->gids);
    par_for("rt3d", DevExeSpace(), 0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real &x1min = size.d_view(m).x1min;
      Real &x1max = size.d_view(m).x1max;
      int nx1 = indcs.nx1;
      Real x1v = CellCenterX(i-is, nx1, x1min, x1max);

      Real &x2min = size.d_view(m).x2min;
      Real &x2max = size.d_view(m).x2max;
      int nx2 = indcs.nx2;
      Real x2v = CellCenterX(j-js, nx2, x2min, x2max);

      Real &x3min = size.d_view(m).x3min;
      Real &x3max = size.d_view(m).x3max;
      int nx3 = indcs.nx3;
      Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);

      Real den=1.0;
      if (x3v > 0.0) den *= drat;

      if (iprob == 1) {
        u0_(m,IM3,k,j,i) = (1.0+cos(kx*x1v))*(1.0+cos(ky*x2v))*(1.0+cos(kz*x3v))/8.0;
      } else {
        auto rand_gen = rand_pool64.get_state();  // get random number state this thread
        Real r = 2.0*static_cast<Real>(rand_gen.frand()) - 1.0;
        u0_(m,IM3,k,j,i) = r * (1.0 + cos(kz*x3v))/2.0;
        rand_pool64.free_state(rand_gen);  // free state for use by other threads
      }

      u0_(m,IDN,k,j,i) = den;
      u0_(m,IM1,k,j,i) = 0.0;
      u0_(m,IM2,k,j,i) = 0.0;
      u0_(m,IM3,k,j,i) *= (den*amp);
      u0_(m,IEN,k,j,i) = (p0 + grav_acc*den*x3v)/gm1 + 0.5*SQR(u0_(m,IM3,k,j,i))/den;
    });
  }

  return;
}
