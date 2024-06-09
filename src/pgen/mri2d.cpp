//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file mri2d.cpp
//! \brief Problem generator for 2D MRI simulations using the shearing sheet based on
//!  "A powerful local shear instability in weakly magnetized disks. III - Long-term
//!  evolution in a shearing sheet" by Hawley & Balbus.  Based on the hgb.cpp problem
//!  generator in Athena++
//! REFERENCE: Hawley, J. F. & Balbus, S. A., ApJ 400, 595-609 (1992).
//!
//! Two different field configurations are possible:
//! - ifield = 1 - Bz=B0 sin(x1) field with zero-net-flux [default]
//! - ifield = 2 - uniform Bz

// C++ headers
#include <cmath>      // sqrt()
#include <iostream>   // cout, endl
#include <sstream>    // stringstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()

// Athena++ headers
#include "athena.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "shearing_box/shearing_box.hpp"
#include "pgen.hpp"

#include <Kokkos_Random.hpp>

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::_()
//  \brief

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  if (restart) return;

  if (pmy_mesh_->three_d) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "mri2d problem generator only works in 2D (nx3=1)" << std::endl;
    exit(EXIT_FAILURE);
  }

  // initialize problem variables
  Real amp   = pin->GetReal("problem","amp");
  Real beta  = pin->GetReal("problem","beta");
  int nwx    = pin->GetOrAddInteger("problem","nwx",1);
  int ifield = pin->GetOrAddInteger("problem","ifield",1);

  // background density, pressure, and magnetic field
  Real d0 = 1.0;
  Real p0 = 1.0;
  Real binit = std::sqrt(2.0*p0/beta);

  Real x1size = pmy_mesh_->mesh_size.x1max - pmy_mesh_->mesh_size.x1min;
  Real kx = 2.0*(M_PI/x1size)*(static_cast<Real>(nwx));

  // capture variables for kernel
  auto &indcs = pmy_mesh_->mb_indcs;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  auto &size = pmbp->pmb->mb_size;

  if (pmbp->pmhd != nullptr) {
    // First, do some error checks
    if (pmbp->pmhd->psrc == nullptr) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl
                << "Shearing box source terms not enabled for mri2d problem" << std::endl;
      exit(EXIT_FAILURE);
    }
    if (!pmbp->pmhd->shearing_box || pmbp->pmhd->psb->shearing_box_r_phi) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl
                << "hb3 problem generator only works in 2D (x-z) shearing box"
                << std::endl;
      exit(EXIT_FAILURE);
    }

    // Initialize magnetic field first, so entire arrays are initialized before adding
    // magnetic energy to conserved variables in next loop.  For 2D shearing box
    // B1=Bx, B2=Bz, B3=By
    // ifield = 1 - Bz=binit sin(kx*xav1) field with zero-net-flux [default]
    // ifield = 2 - uniform Bz
    auto b0 = pmbp->pmhd->b0;
    par_for("mri2d-b", DevExeSpace(), 0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real &x1min = size.d_view(m).x1min;
      Real &x1max = size.d_view(m).x1max;
      int nx1 = indcs.nx1;
      Real x1v = CellCenterX(i-is, nx1, x1min, x1max);

      if (ifield == 1) {
        b0.x1f(m,k,j,i) = 0.0;
        b0.x2f(m,k,j,i) = binit*sin(kx*x1v);
        b0.x3f(m,k,j,i) = 0.0;
        if (i==ie) b0.x1f(m,k,j,ie+1) = 0.0;
        if (j==je) b0.x2f(m,k,je+1,i) = binit*sin(kx*x1v);
        if (k==ke) b0.x3f(m,ke+1,j,i) = 0.0;
      } else if (ifield == 2) {
        b0.x1f(m,k,j,i) = 0.0;
        b0.x2f(m,k,j,i) = binit;
        b0.x3f(m,k,j,i) = 0.0;
        if (i==ie) b0.x1f(m,k,j,ie+1) = 0.0;
        if (j==je) b0.x2f(m,k,je+1,i) = binit;
        if (k==ke) b0.x3f(m,ke+1,j,i) = 0.0;
      }
    });
  }

  // Initialize conserved variables in Hydro
  // Only sets up uniform motion in x1-direction -- epicycle test
  if (pmbp->phydro != nullptr) {
    // First, do some error checks
    if (pmbp->phydro->psrc == nullptr) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl
                << "Shearing box source terms not enabled for mri2d problem" << std::endl;
      exit(EXIT_FAILURE);
    }
    if (!pmbp->phydro->shearing_box || pmbp->phydro->psb->shearing_box_r_phi) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl
                << "hb3 problem generator only works in 2D (x-z) shearing box"
                << std::endl;
      exit(EXIT_FAILURE);
    }
    EOS_Data &eos = pmbp->phydro->peos->eos_data;
    Real gm1 = eos.gamma - 1.0;
    auto u0 = pmbp->phydro->u0;
    par_for("mri2d-u", DevExeSpace(), 0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      u0(m,IDN,k,j,i) = d0;
      u0(m,IM1,k,j,i) = d0*amp;
      u0(m,IM2,k,j,i) = 0.0;
      u0(m,IM3,k,j,i) = 0.0;
      if (eos.is_ideal) { u0(m,IEN,k,j,i) = p0/gm1 + 0.5*d0*amp*amp; }
    });
  }

  // Initialize conserved variables in MHD
  // Only sets up random perturbations in pressure to seed MRI
  if (pmbp->pmhd != nullptr) {
    EOS_Data &eos = pmbp->pmhd->peos->eos_data;
    Real gm1 = eos.gamma - 1.0;
    auto b0 = pmbp->pmhd->b0;
    auto u0 = pmbp->pmhd->u0;
    Kokkos::Random_XorShift64_Pool<> rand_pool64(pmbp->gids);
    par_for("mri2d-u", DevExeSpace(), 0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real rd = d0;
      Real rp = p0;
      auto rand_gen = rand_pool64.get_state();  // get random number state this thread
      Real rval = 1.0 + amp*(rand_gen.frand() - 0.5);
      if (eos.is_ideal) {
        rp = rval*p0;
      } else {
        rd = rval*d0;
      }
      u0(m,IDN,k,j,i) = rd;
      u0(m,IM1,k,j,i) = 0.0;
      u0(m,IM2,k,j,i) = 0.0;
      u0(m,IM3,k,j,i) = 0.0;
      if (eos.is_ideal) {
        u0(m,IEN,k,j,i) = rp/gm1 + 0.5*SQR(0.5*(b0.x2f(m,k,j,i) + b0.x2f(m,k,j+1,i)));
      }
      rand_pool64.free_state(rand_gen);  // free state for use by other threads
    });
  }

  return;
}
