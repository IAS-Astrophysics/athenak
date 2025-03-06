//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file z4c_Sbc.cpp
//! \brief placeholder for Sommerfeld boundary condition

#include <algorithm>
#include <cinttypes>
#include <iostream>
#include <limits>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "z4c/z4c.hpp"
#include "coordinates/cell_locations.hpp"

namespace z4c {

//----------------------------------------------------------------------------------------
//! \fn void Z4c::Z4cSommerfeld
//! \brief apply Sommerfeld BCs to the given set of points
KOKKOS_INLINE_FUNCTION
static void Z4cSommerfeld(const Z4c::Z4c_vars& z4c, const Z4c::Z4c_vars& rhs,
    const RegionIndcs &indcs, const DualArray1D<RegionSize> &size,
    const int m, const int k, const int j, const int i) {
  // -------------------------------------------------------------------------------------
  // Scratch data
  //

  // First derivatives
  // Scalars
  AthenaPointTensor<Real, TensorSymm::NONE, 3, 1> dKhat_d;
  AthenaPointTensor<Real, TensorSymm::NONE, 3, 1> dTheta_d;

  // Vectors
  AthenaPointTensor<Real, TensorSymm::NONE, 3, 2> dGam_du;

  // Tensors
  AthenaPointTensor<Real, TensorSymm::SYM2, 3, 3> dA_ddd;


  // Psuedoradial vector
  AthenaPointTensor<Real, TensorSymm::NONE, 3, 1> s_u;

  Real idx[] = {1./size.d_view(m).dx1, 1./size.d_view(m).dx2, 1./size.d_view(m).dx3};

  // -------------------------------------------------------------------------------------
  // First derivatives
  // We force all derivatives to be calculated at second-order, as this was found to
  // be necessary for stability in Athena++.
  //
  for (int a = 0; a < 3; a++) {
    dKhat_d(a) = Dx<2>(a, idx, z4c.vKhat, m, k, j, i);
    dTheta_d(a) = Dx<2>(a, idx, z4c.vTheta, m, k, j, i);
  }
  for (int a = 0; a < 3; a++) {
    for (int b = 0; b < 3; b++) {
      dGam_du(b,a) = Dx<2>(b, idx, z4c.vGam_u, m, a, k, j, i);
    }
  }
  for (int a = 0; a < 3; a++) {
    for (int b = a; b < 3; b++) {
      for (int c = 0; c < 3; c++) {
        dA_ddd(c, a, b) = Dx<2>(c, idx, z4c.vA_dd, m, a, b, k, j, i);
      }
    }
  }

  // -------------------------------------------------------------------------------------
  // Compute psuedo-radial vector
  //
  Real &x1min = size.d_view(m).x1min;
  Real &x1max = size.d_view(m).x1max;
  Real &x2min = size.d_view(m).x2min;
  Real &x2max = size.d_view(m).x2max;
  Real &x3min = size.d_view(m).x3min;
  Real &x3max = size.d_view(m).x3max;

  Real x1v = CellCenterX(i-indcs.is, indcs.nx1, x1min, x1max);
  Real x2v = CellCenterX(j-indcs.js, indcs.nx2, x2min, x2max);
  Real x3v = CellCenterX(k-indcs.ks, indcs.nx3, x3min, x3max);

  Real r = sqrt(SQR(x1v) + SQR(x2v) + SQR(x3v));
  s_u(0) = x1v/r;
  s_u(1) = x2v/r;
  s_u(2) = x3v/r;

  // -------------------------------------------------------------------------------------
  // Boundary RHS for scalars
  //
  rhs.vTheta(m,k,j,i) = - z4c.vTheta(m,k,j,i)/r;
  rhs.vKhat(m,k,j,i) = - sqrt(2.) * z4c.vKhat(m,k,j,i)/r;
  for (int a = 0; a < 3; a++) {
    rhs.vTheta(m,k,j,i) -= s_u(a) * dTheta_d(a);
    rhs.vKhat(m,k,j,i) -= sqrt(2.) * s_u(a) * dKhat_d(a);
  }

  // -------------------------------------------------------------------------------------
  // Boundary RHS for Gamma
  //
  for (int a = 0; a < 3; a++) {
    rhs.vGam_u(m,a,k,j,i) = - z4c.vGam_u(m, a, k, j, i)/r;
    for (int b = 0; b < 3; b++) {
      rhs.vGam_u(m,a,k,j,i) -= s_u(b) * dGam_du(b,a);
    }
  }

  // -------------------------------------------------------------------------------------
  // Boundary RHS for A_ab
  //
  for (int a = 0; a < 3; a++) {
    for (int b = a; b < 3; b++) {
      rhs.vA_dd(m,a,b,k,j,i) = - z4c.vA_dd(m,a,b,k,j,i)/r;
      for (int c = 0; c < 3; c++) {
        rhs.vA_dd(m,a,b,k,j,i) -= s_u(c) * dA_ddd(c,a,b);
      }
    }
  }
}


//---------------------------------------------------------------------------------------
//! \fn TaskStatus Z4c::Z4cBoundaryRHS
//! \brief placeholder for the Sommerfield Boundary conditions for z4c
TaskStatus Z4c::Z4cBoundaryRHS(Driver *pdriver, int stage) {
  auto &pm = pmy_pack->pmesh;
  auto &mb_bcs = pmy_pack->pmb->mb_bcs;
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  auto &size = pmy_pack->pmb->mb_size;

  int nmb = pmy_pack->nmb_thispack;
  int is = indcs.is;
  int ie = indcs.ie;
  int js = indcs.js;
  int je = indcs.je;
  int ks = indcs.ks;
  int ke = indcs.ke;

  auto &z4c_ = z4c;
  auto &rhs_ = rhs;
  bool &user_Sbc = opt.user_Sbc;

  // We only need to apply this condition for outflow boundaries
  if (pm->mesh_bcs[BoundaryFace::inner_x1] == BoundaryFlag::outflow
      || pm->mesh_bcs[BoundaryFace::inner_x1] == BoundaryFlag::diode
      || pm->mesh_bcs[BoundaryFace::inner_x1] == BoundaryFlag::user
      || pm->mesh_bcs[BoundaryFace::outer_x1] == BoundaryFlag::outflow
      || pm->mesh_bcs[BoundaryFace::outer_x1] == BoundaryFlag::diode
      || pm->mesh_bcs[BoundaryFace::outer_x1] == BoundaryFlag::user) {
    par_for("z4crhs_bc_x1", DevExeSpace(), 0, (nmb-1), ks, ke, js, je,
    KOKKOS_LAMBDA(int m, int k, int j) {
      // Inner boundary
      switch(mb_bcs.d_view(m,BoundaryFace::inner_x1)) {
        case BoundaryFlag::diode:
        case BoundaryFlag::outflow:
            Z4cSommerfeld(z4c_, rhs_, indcs, size, m, k, j, is);
          break;
        case BoundaryFlag::user:
            if (user_Sbc) {
              Z4cSommerfeld(z4c_, rhs_, indcs, size, m, k, j, is);
            }
          break;
        default:
          break;
      }
      // Outer boundary
      switch (mb_bcs.d_view(m,BoundaryFace::outer_x1)) {
        case BoundaryFlag::diode:
        case BoundaryFlag::outflow:
            Z4cSommerfeld(z4c_, rhs_, indcs, size, m, k, j, ie);
          break;
        case BoundaryFlag::user:
            if (user_Sbc) {
              Z4cSommerfeld(z4c_, rhs_, indcs, size, m, k, j, ie);
            }
          break;
        default:
          break;
      }
    });
  }
  if (pm->mesh_bcs[BoundaryFace::inner_x2] == BoundaryFlag::outflow
      || pm->mesh_bcs[BoundaryFace::inner_x2] == BoundaryFlag::diode
      || pm->mesh_bcs[BoundaryFace::inner_x2] == BoundaryFlag::user
      || pm->mesh_bcs[BoundaryFace::outer_x2] == BoundaryFlag::outflow
      || pm->mesh_bcs[BoundaryFace::outer_x2] == BoundaryFlag::diode
      || pm->mesh_bcs[BoundaryFace::outer_x2] == BoundaryFlag::user) {
    par_for("z4crhs_bc_x2", DevExeSpace(), 0, (nmb-1), ks, ke, is, ie,
    KOKKOS_LAMBDA(int m, int k, int i) {
      // Inner boundary
      switch(mb_bcs.d_view(m,BoundaryFace::inner_x2)) {
        case BoundaryFlag::diode:
        case BoundaryFlag::outflow:
            Z4cSommerfeld(z4c_, rhs_, indcs, size, m, k, js, i);
          break;
        case BoundaryFlag::user:
            if (user_Sbc) {
              Z4cSommerfeld(z4c_, rhs_, indcs, size, m, k, js, i);
            }
          break;
        default:
          break;
      }
      // Outer boundary
      switch (mb_bcs.d_view(m,BoundaryFace::outer_x2)) {
        case BoundaryFlag::diode:
        case BoundaryFlag::outflow:
            Z4cSommerfeld(z4c_, rhs_, indcs, size, m, k, je, i);
          break;
        case BoundaryFlag::user:
            if (user_Sbc) {
              Z4cSommerfeld(z4c_, rhs_, indcs, size, m, k, je, i);
            }
          break;
        default:
          break;
      }
    });
  }
  if (pm->mesh_bcs[BoundaryFace::inner_x3] == BoundaryFlag::outflow
      || pm->mesh_bcs[BoundaryFace::inner_x3] == BoundaryFlag::diode
      || pm->mesh_bcs[BoundaryFace::inner_x3] == BoundaryFlag::user
      || pm->mesh_bcs[BoundaryFace::outer_x3] == BoundaryFlag::outflow
      || pm->mesh_bcs[BoundaryFace::outer_x3] == BoundaryFlag::diode
      || pm->mesh_bcs[BoundaryFace::outer_x3] == BoundaryFlag::user) {
    par_for("z4crhs_bc_x3", DevExeSpace(), 0, (nmb-1), js, je, is, ie,
    KOKKOS_LAMBDA(int m, int j, int i) {
      // Inner boundary
      switch(mb_bcs.d_view(m,BoundaryFace::inner_x3)) {
        case BoundaryFlag::diode:
        case BoundaryFlag::outflow:
            Z4cSommerfeld(z4c_, rhs_, indcs, size, m, ks, j, i);
          break;
        case BoundaryFlag::user:
            if (user_Sbc) {
              Z4cSommerfeld(z4c_, rhs_, indcs, size, m, ks, j, i);
            }
          break;
        default:
          break;
      }
      // Outer boundary
      switch (mb_bcs.d_view(m,BoundaryFace::outer_x3)) {
        case BoundaryFlag::diode:
        case BoundaryFlag::outflow:
            Z4cSommerfeld(z4c_, rhs_, indcs, size, m, ke, j, i);
          break;
        case BoundaryFlag::user:
            if (user_Sbc) {
              Z4cSommerfeld(z4c_, rhs_, indcs, size, m, ke, j, i);
            }
          break;
        default:
          break;
      }
    });
  }


  return TaskStatus::complete;
}

} // end namespace z4c
