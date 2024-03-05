//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file shear_test.cpp
//! \brief Simple test of advection in the shearing box

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
#include "srcterms/srcterms.hpp"
#include "pgen.hpp"

#include <Kokkos_Random.hpp>

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::_()
//  \brief

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  if (restart) return;

  // background density, pressure, and magnetic field
  Real d0 = 1.0;
  Real p0 = 1.0;
  Real beta = 1000.0;
  Real B0 = std::sqrt(2.0*p0/beta);
  Real amp = 0.01;
  auto &msize = pmy_mesh_->mesh_size;
  Real lx = msize.x1max - msize.x1min;
  Real ly = msize.x2max - msize.x2min;
  Real x0 = msize.x1min, y0 = msize.x2min;

  Real xpt[5] = {(x0+0.25*lx), (x0+0.25*lx), (x0+0.5*lx), (x0+0.75*lx), (x0+0.75*lx)};
  Real ypt[5] = {(y0+0.25*ly), (y0+0.75*ly), (y0+0.5*ly), (y0+0.25*ly), (y0+0.75*ly)};

  // capture variables for kernel
  auto &indcs = pmy_mesh_->mb_indcs;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  auto &size = pmbp->pmb->mb_size;

  // Initialize conserved variables in Hydro
  if (pmbp->phydro != nullptr) {
    EOS_Data &eos = pmbp->phydro->peos->eos_data;
    Real gm1 = eos.gamma - 1.0;
    auto u0 = pmbp->phydro->u0;
    par_for("shear0", DevExeSpace(), 0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real &x1min = size.d_view(m).x1min;
      Real &x1max = size.d_view(m).x1max;
      int nx1 = indcs.nx1;
      Real x1v = CellCenterX(i-is, nx1, x1min, x1max);

      Real &x2min = size.d_view(m).x2min;
      Real &x2max = size.d_view(m).x2max;
      int nx2 = indcs.nx2;
      Real x2v = CellCenterX(j-js, nx2, x2min, x2max);

      Real dens = d0;
      for (int n=0; n<5; ++n) {
        if (sqrt(SQR(x1v-xpt[n])+SQR(x2v-ypt[n])) <= (0.1*lx)) {dens += amp;}
      }

      u0(m,IDN,k,j,i) = dens;
      u0(m,IM1,k,j,i) = 0.0;
      u0(m,IM2,k,j,i) = 0.0;
      u0(m,IM3,k,j,i) = 0.0;
      if (eos.is_ideal) { u0(m,IEN,k,j,i) = p0/gm1; }
    });
  }

  // Initialize conserved variables in MHD
  if (pmbp->pmhd != nullptr) {
    EOS_Data &eos = pmbp->pmhd->peos->eos_data;
    Real gm1 = eos.gamma - 1.0;
    auto u0 = pmbp->pmhd->u0;
    par_for("shear1", DevExeSpace(), 0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real &x1min = size.d_view(m).x1min;
      Real &x1max = size.d_view(m).x1max;
      int nx1 = indcs.nx1;
      Real x1v = CellCenterX(i-is, nx1, x1min, x1max);

      Real &x2min = size.d_view(m).x2min;
      Real &x2max = size.d_view(m).x2max;
      int nx2 = indcs.nx2;
      Real x2v = CellCenterX(j-js, nx2, x2min, x2max);

      Real dens = d0;
      for (int n=0; n<5; ++n) {
        if (sqrt(SQR(x1v-xpt[n])+SQR(x2v-ypt[n])) <= (0.1*lx)) { dens += amp; }
      }

      u0(m,IDN,k,j,i) = dens;
      u0(m,IM1,k,j,i) = 0.0;
      u0(m,IM2,k,j,i) = 0.0;
      u0(m,IM3,k,j,i) = 0.0;
      if (eos.is_ideal) { u0(m,IEN,k,j,i) = p0/gm1; }
    });
  }

  // Initialize magnetic field in MHD
  if (pmbp->pmhd != nullptr) {
    // compute vector potential
    DvceArray4D<Real> a3;
    int ncells1 = indcs.nx1 + 2*(indcs.ng);
    int ncells2 = indcs.nx2 + 2*(indcs.ng);
    int ncells3 = indcs.nx3 + 2*(indcs.ng);
    Kokkos::realloc(a3,(pmbp->nmb_thispack),ncells3,ncells2,ncells1);

    par_for("shear3", DevExeSpace(), 0,(pmbp->nmb_thispack-1),ks,ke,js,je+1,is,ie+1,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real &x1min = size.d_view(m).x1min;
      Real &x1max = size.d_view(m).x1max;
      Real x1f    = LeftEdgeX(i  -is, nx1, x1min, x1max);

      Real &x2min = size.d_view(m).x2min;
      Real &x2max = size.d_view(m).x2max;
      Real x2f   = LeftEdgeX(j  -js, nx2, x2min, x2max);

      Real &x3min = size.d_view(m).x3min;
      Real &x3max = size.d_view(m).x3max;
      Real x3v    = CellCenterX(k-ks, nx3, x3min, x3max);

      a3(m,k,j,i) = 0.0;
      for (int n=0; n<5; ++n) {
        if (sqrt(SQR(x1f-xpt[n])+SQR(x2f-ypt[n])) <= (0.1*lx)) {
          a3(m,k,j,i) += 1.0e-3*(0.1*lx - sqrt(SQR(x1f-xpt[n])+SQR(x2f-ypt[n])));
        }
      }
    });

    EOS_Data &eos = pmbp->pmhd->peos->eos_data;
    auto u0 = pmbp->pmhd->u0;
    auto b0 = pmbp->pmhd->b0;
    par_for("shear4", DevExeSpace(), 0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      // Compute face-centered fields from curl(A).
      Real dx1 = size.d_view(m).dx1;
      Real dx2 = size.d_view(m).dx2;

      b0.x1f(m,k,j,i) =   (a3(m,k,j+1,i) - a3(m,k,j,i))/dx2;
      b0.x2f(m,k,j,i) = - (a3(m,k,j,i+1) - a3(m,k,j,i))/dx1;
      b0.x3f(m,k,j,i) = 0.0;

      if (eos.is_ideal) {
        u0(m,IEN,k,j,i) += 0.5*(SQR(0.5*((a3(m,k,j+1,i+1) - a3(m,k,j,i+1)) +
                                         (a3(m,k,j+1,i  ) - a3(m,k,j,i  )))/dx2) +
                                SQR(0.5*((a3(m,k,j+1,i+1) - a3(m,k,j+1,i  )) +
                                         (a3(m,k,j  ,i+1) - a3(m,k,j,i  )))/dx1) );
      }

      // Include extra face-component at edge of block in each direction
      if (i==ie) {
        b0.x1f(m,k,j,i+1) = (a3(m,k,j+1,i+1) - a3(m,k,j,i+1))/dx2;
      }
      if (j==je) {
        b0.x2f(m,k,j+1,i) =  - (a3(m,k,j+1,i+1) - a3(m,k,j+1,i))/dx1;
      }
      if (k==ke) {
        b0.x3f(m,k+1,j,i) = 0.0;
      }
    });
  }
  return;
}
