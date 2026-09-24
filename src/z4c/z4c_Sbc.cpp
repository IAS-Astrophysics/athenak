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
  AthenaPointTensor<Real, TensorSymm::NONE, 3, 1> dchi_d;

  // Vectors
  AthenaPointTensor<Real, TensorSymm::NONE, 3, 2> dGam_du;

  // Tensors
  AthenaPointTensor<Real, TensorSymm::SYM2, 3, 3> dA_ddd;
  AthenaPointTensor<Real, TensorSymm::SYM2, 3, 3> dg_ddd;


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
  for (int a = 0; a < 3; a++) {
    dchi_d(a) = Dx<2>(a, idx, z4c.chi, m, k, j, i);
  }
  for (int a = 0; a < 3; a++) {
    for (int b = a; b < 3; b++) {
      for (int c = 0; c < 3; c++) {
        dg_ddd(c, a, b) = Dx<2>(c, idx, z4c.g_dd, m, a, b, k, j, i);
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

  // -------------------------------------------------------------------------------------
  // Boundary RHS for the metric sector: chi and g_ab
  //
  // The four groups above are the only ones that carry a radiative condition in the
  // unmodified code; chi, g~_ab, alpha and beta^i are left to integrate their interior RHS
  // off extrapolated ghosts.  That is an inconsistent treatment of the WAVE PAIR
  // (g~_ab, A_ab): schematically d_t g~ = -2 alpha A and d_t A ~ -(1/2) alpha d^2 g~, so
  // constraining A alone does not make the pair outgoing.  g~ then accumulates the time
  // integral of whatever A does on the boundary, and that drift feeds back into A through
  // the Ricci term, closing an unstable loop.
  //
  // Measured in a long BNS postmerger (VVLR_eq, box +/-4096): at the +x face centre both
  // g~_xx and A_xx grow with tau ~ 3900-4000 M while beta^x, Gam^x, chi and alpha grow
  // ~6x slower (tau ~ 24000-27000 M), and the ratio |g~_xx - 1| / |A_xx| = 7778 matches
  // 2*tau = 7834 to 0.7% -- i.e. g~ is exactly the time integral of A, as the slaving
  // relation predicts.  The deformation is trace-free (det g~ held at 1, off-diagonal
  // g~_xy growing at the same rate), so it is the wave sector, not a gauge or conformal
  // mode.  Standard BSSN practice applies the radiative condition to g~_ij as well.
  //
  // Note this is not over-determination in any harmful sense: for genuinely outgoing
  // content the conditions on g~ and A are satisfied by the same solution; they conflict
  // only for the non-outgoing part, which is exactly what should be removed.  The
  // asymptotic values are unambiguous -- chi -> 1, g~_ab -> delta_ab -- and
  // the 1/r form preserves a static Coulomb-like tail as well as radiation, since
  // u - u0 ~ 1/r is a stationary solution of the condition.  EnforceAlgConstr restores
  // det g~ = 1 at the end of every stage, so a component-wise condition on g~ is safe.
  //
  // The GAUGE variables alpha and beta^i are deliberately NOT included, matching CTGamma,
  // whose set is {chi, g_ik, Gam^i, A_ik, Theta, Khat}
  rhs.chi(m,k,j,i) = - (z4c.chi(m,k,j,i) - 1.0)/r;
  for (int a = 0; a < 3; a++) {
    rhs.chi(m,k,j,i) -= s_u(a) * dchi_d(a);
  }
  for (int a = 0; a < 3; a++) {
    for (int b = a; b < 3; b++) {
      Real const flat = (a == b) ? 1.0 : 0.0;
      rhs.g_dd(m,a,b,k,j,i) = - (z4c.g_dd(m,a,b,k,j,i) - flat)/r;
      for (int c = 0; c < 3; c++) {
        rhs.g_dd(m,a,b,k,j,i) -= s_u(c) * dg_ddd(c,a,b);
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
      || pm->mesh_bcs[BoundaryFace::inner_x1] == BoundaryFlag::vacuum
      || pm->mesh_bcs[BoundaryFace::inner_x1] == BoundaryFlag::user
      || pm->mesh_bcs[BoundaryFace::outer_x1] == BoundaryFlag::outflow
      || pm->mesh_bcs[BoundaryFace::outer_x1] == BoundaryFlag::diode
      || pm->mesh_bcs[BoundaryFace::outer_x1] == BoundaryFlag::vacuum
      || pm->mesh_bcs[BoundaryFace::outer_x1] == BoundaryFlag::user) {
    par_for("z4crhs_bc_x1", DevExeSpace(), 0, (nmb-1), ks, ke, js, je,
    KOKKOS_LAMBDA(int m, int k, int j) {
      // Inner boundary
      switch(mb_bcs.d_view(m,BoundaryFace::inner_x1)) {
        case BoundaryFlag::vacuum:
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
        case BoundaryFlag::vacuum:
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
      || pm->mesh_bcs[BoundaryFace::inner_x2] == BoundaryFlag::vacuum
      || pm->mesh_bcs[BoundaryFace::inner_x2] == BoundaryFlag::user
      || pm->mesh_bcs[BoundaryFace::outer_x2] == BoundaryFlag::outflow
      || pm->mesh_bcs[BoundaryFace::outer_x2] == BoundaryFlag::diode
      || pm->mesh_bcs[BoundaryFace::outer_x2] == BoundaryFlag::vacuum
      || pm->mesh_bcs[BoundaryFace::outer_x2] == BoundaryFlag::user) {
    par_for("z4crhs_bc_x2", DevExeSpace(), 0, (nmb-1), ks, ke, is, ie,
    KOKKOS_LAMBDA(int m, int k, int i) {
      // Inner boundary
      switch(mb_bcs.d_view(m,BoundaryFace::inner_x2)) {
        case BoundaryFlag::vacuum:
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
        case BoundaryFlag::vacuum:
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
      || pm->mesh_bcs[BoundaryFace::inner_x3] == BoundaryFlag::vacuum
      || pm->mesh_bcs[BoundaryFace::inner_x3] == BoundaryFlag::user
      || pm->mesh_bcs[BoundaryFace::outer_x3] == BoundaryFlag::outflow
      || pm->mesh_bcs[BoundaryFace::outer_x3] == BoundaryFlag::diode
      || pm->mesh_bcs[BoundaryFace::outer_x3] == BoundaryFlag::vacuum
      || pm->mesh_bcs[BoundaryFace::outer_x3] == BoundaryFlag::user) {
    par_for("z4crhs_bc_x3", DevExeSpace(), 0, (nmb-1), js, je, is, ie,
    KOKKOS_LAMBDA(int m, int j, int i) {
      // Inner boundary
      switch(mb_bcs.d_view(m,BoundaryFace::inner_x3)) {
        case BoundaryFlag::vacuum:
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
        case BoundaryFlag::vacuum:
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
