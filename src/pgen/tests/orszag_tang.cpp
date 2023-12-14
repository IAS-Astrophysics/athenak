//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file orszag-tang.c
//  \brief Problem generator for Orszag-Tang vortex problem.
//
// REFERENCE: For example, see: G. Toth,  "The div(B)=0 constraint in shock capturing
//   MHD codes", JCP, 161, 605 (2000)
//========================================================================================

// C++ headers
#include <math.h>
#include <iostream>   // endl
#include <sstream>    // stringstream

// Athena++ headers
#include "athena.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "mhd/mhd.hpp"
#include "pgen/pgen.hpp"

//----------------------------------------------------------------------------------------
//! \fn Real A3(const Real x1,const Real x2,const Real x3)
//  \brief A3: 3-component of vector potential

KOKKOS_INLINE_FUNCTION
Real A3(const Real x1, const Real x2, const Real B0) {
  return (B0/(4.0*M_PI))*(std::cos(4.0*M_PI*x1) - 2.0*std::cos(2.0*M_PI*x2));
}

//----------------------------------------------------------------------------------------
//! \fn void MeshBlock::OrszagTang_(ParameterInput *pin)
//  \brief Problem Generator for the Orszag-Tang test.  The initial conditions are
//  constructed assuming the domain extends over [-0.5x0.5, -0.5x0.5], so that exact
//  symmetry can be enforced across x=0 and y=0.

void ProblemGenerator::OrszagTang(ParameterInput *pin, const bool restart) {
  if (restart) return;

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->pmhd == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Orszag-Tang test can only be run in MHD, but no <mhd> block "
              << "in input file" << std::endl;
    exit(EXIT_FAILURE);
  }

  Real B0 = 1.0/std::sqrt(4.0*M_PI);
  Real d0 = 25.0/(36.0*M_PI);
  Real v0 = 1.0;
  Real p0 = 5.0/(12.0*M_PI);

  // capture variables for kernel
  auto &indcs = pmy_mesh_->mb_indcs;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;

  EOS_Data &eos = pmbp->pmhd->peos->eos_data;
  Real gm1 = eos.gamma - 1.0;
  auto &u0 = pmbp->pmhd->u0;
  auto &b0 = pmbp->pmhd->b0;
  auto &size = pmbp->pmb->mb_size;

  par_for("pgen_ot1", DevExeSpace(), 0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    int nx1 = indcs.nx1;
    Real x1v = CellCenterX(i-is, nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    int nx2 = indcs.nx2;
    Real x2v = CellCenterX(j-js, nx2, x2min, x2max);

    // compute cell-centered conserved variables
    u0(m,IDN,k,j,i) = d0;
    u0(m,IM1,k,j,i) =  d0*v0*std::sin(2.0*M_PI*x2v);
    u0(m,IM2,k,j,i) = -d0*v0*std::sin(2.0*M_PI*x1v);
    u0(m,IM3,k,j,i) = 0.0;

    // Compute face-centered fields from curl(A).
    Real x1f   = LeftEdgeX(i  -is, nx1, x1min, x1max);
    Real x1fp1 = LeftEdgeX(i+1-is, nx1, x1min, x1max);
    Real x2f   = LeftEdgeX(j  -js, nx2, x2min, x2max);
    Real x2fp1 = LeftEdgeX(j+1-js, nx2, x2min, x2max);
    Real dx1 = size.d_view(m).dx1;
    Real dx2 = size.d_view(m).dx2;

    b0.x1f(m,k,j,i) =  (A3(x1f,  x2fp1,B0) - A3(x1f,x2f,B0))/dx2;
    b0.x2f(m,k,j,i) = -(A3(x1fp1,x2f  ,B0) - A3(x1f,x2f,B0))/dx1;
    b0.x3f(m,k,j,i) = 0.0;

    // Include extra face-component at edge of block in each direction
    if (i==ie) {
      b0.x1f(m,k,j,i+1) =  (A3(x1fp1,x2fp1,B0) - A3(x1fp1,x2f,B0))/dx2;
    }
    if (j==je) {
      b0.x2f(m,k,j+1,i) = -(A3(x1fp1,x2fp1,B0) - A3(x1f,x2fp1,B0))/dx1;
    }
    if (k==ke) {
      b0.x3f(m,k+1,j,i) = 0.0;
    }
  });

  // initialize total energy (requires B to be defined across entire grid first)
  par_for("pgen_ot2", DevExeSpace(), 0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    u0(m,IEN,k,j,i) = p0/gm1 + (0.5/u0(m,IDN,k,j,i))*
         (SQR(u0(m,IM1,k,j,i)) + SQR(u0(m,IM2,k,j,i)) + SQR(u0(m,IM3,k,j,i))) +
          0.5*(SQR(0.5*(b0.x1f(m,k,j,i) + b0.x1f(m,k,j,i+1))) +
               SQR(0.5*(b0.x2f(m,k,j,i) + b0.x2f(m,k,j+1,i))) +
               SQR(0.5*(b0.x3f(m,k,j,i) + b0.x3f(m,k+1,j,i))));
  });

  return;
}
