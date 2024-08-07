//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file mri3d.cpp
//! \brief Problem generator for 3D MRI.
//!
//! PURPOSE:  Problem generator for 3D MRI. Based on the initial conditions described in
//! "Local Three-dimensional Magnetohydrodynamic Simulations of Accretion Disks" by
//! Hawley, Gammie & Balbus, or HGB.  AthanK version based on pgen/hgb.cpp in Athena++.
//!
//! Several different field configurations are possible:
//! - ifield = 1 - Bz=B0 sin(nwx*kx*x1) field with zero-net-flux [default] (nwx input)
//! - ifield = 2 - uniform Bz
//! - ifield = 3 - uniform By
//! Random perturbations to the pressure are added in the initial conditions to seed MRI
//!
//! REFERENCE: Hawley, J. F., Gammie, C.F. & Balbus, S. A., ApJ 440, 742-763 (1995).

// C++ headers
#include <cmath>      // sqrt()
#include <iostream>   // endl

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

  // First, do some error checks
  if (!(pmy_mesh_->three_d)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "mri3d problem generator only works in 2D (nx3=1)" << std::endl;
    exit(EXIT_FAILURE);
  }
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->pmhd != nullptr) {
    if (pmbp->pmhd->psrc == nullptr) {
      std::cout <<"### FATAL ERROR in "<< __FILE__ <<" at line " <<__LINE__ << std::endl
                << "Shearing box source terms not enabled for mri3d problem" << std::endl;
      exit(EXIT_FAILURE);
    }
  } else {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "mri3d problem generator only works in mhd" << std::endl;
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
  auto &size = pmbp->pmb->mb_size;

  // Initialize magnetic field first, so entire arrays are initialized before adding
  // magnetic energy to conserved variables in next loop.  For 3D shearing box
  // B1=Bx, B2=By, B3=Bz
  // ifield = 1 - Bz=binit sin(kx*xav1) field with zero-net-flux [default]
  // ifield = 2 - uniform Bz
  // ifield = 3 - uniform By
  auto b0 = pmbp->pmhd->b0;
  par_for("mri3d", DevExeSpace(), 0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    int nx1 = indcs.nx1;
    Real x1v = CellCenterX(i-is, nx1, x1min, x1max);

    if (ifield == 1) {
      b0.x1f(m,k,j,i) = 0.0;
      b0.x2f(m,k,j,i) = 0.0;
      b0.x3f(m,k,j,i) = binit*sin(kx*x1v);
      if (i==ie) b0.x1f(m,k,j,ie+1) = 0.0;
      if (j==je) b0.x2f(m,k,je+1,i) = 0.0;
      if (k==ke) b0.x3f(m,ke+1,j,i) = binit*sin(kx*x1v);
    } else if (ifield == 2) {
      b0.x1f(m,k,j,i) = 0.0;
      b0.x2f(m,k,j,i) = 0.0;
      b0.x3f(m,k,j,i) = binit;
      if (i==ie) b0.x1f(m,k,j,ie+1) = 0.0;
      if (j==je) b0.x2f(m,k,je+1,i) = 0.0;
      if (k==ke) b0.x3f(m,ke+1,j,i) = binit;
    } else if (ifield == 3) {
      b0.x1f(m,k,j,i) = 0.0;
      b0.x2f(m,k,j,i) = binit;
      b0.x3f(m,k,j,i) = 0.0;
      if (i==ie) b0.x1f(m,k,j,ie+1) = 0.0;
      if (j==je) b0.x2f(m,k,je+1,i) = binit;
      if (k==ke) b0.x3f(m,ke+1,j,i) = 0.0;
    }
  });

  // Initialize conserved variables
  // Only sets up random perturbations in pressure to seed MRI
  EOS_Data &eos = pmbp->pmhd->peos->eos_data;
  Real gm1 = eos.gamma - 1.0;
  auto u0 = pmbp->pmhd->u0;
  Kokkos::Random_XorShift64_Pool<> rand_pool64(pmbp->gids);
  par_for("mri3d-u", DevExeSpace(), 0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
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
      u0(m,IEN,k,j,i) = rp/gm1 + 0.5*SQR(0.5*(b0.x2f(m,k,j,i) + b0.x2f(m,k,j+1,i))) +
                                 0.5*SQR(0.5*(b0.x3f(m,k,j,i) + b0.x3f(m,k+1,j,i)));
    }
    rand_pool64.free_state(rand_gen);  // free state for use by other threads
  });

  return;
}
