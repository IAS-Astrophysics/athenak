//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file prolongation.cpp
//! \brief functions to prolongate data at boundaries for cell-centered and face-centered
//! variables. Functions are members of MeshBoundaryValuesCC or MeshBoundaryValuesFC
//! classes.

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <iomanip>    // std::setprecision()

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "mesh/nghbr_index.hpp"
#include "bvals.hpp"
#include "mesh/prolongation.hpp" // implements prolongation operators
#include "mesh/restriction.hpp" // implements restriction operators

#include "coordinates/cell_locations.hpp"

namespace {

KOKKOS_INLINE_FUNCTION
void NeighborOffsetFromIndex(const int n, int &ox1, int &ox2, int &ox3) {
  ox1 = 0;
  ox2 = 0;
  ox3 = 0;
  for (int iz = -1; iz <= 1; ++iz) {
    for (int iy = -1; iy <= 1; ++iy) {
      for (int ix = -1; ix <= 1; ++ix) {
        if ((ix == 0) && (iy == 0) && (iz == 0)) continue;
        for (int n1 = 0; n1 <= 1; ++n1) {
          for (int n2 = 0; n2 <= 1; ++n2) {
            if (NeighborIndex(ix, iy, iz, n1, n2) == n) {
              ox1 = ix;
              ox2 = iy;
              ox3 = iz;
              return;
            }
          }
        }
      }
    }
  }
}

KOKKOS_INLINE_FUNCTION
int MaxNeighborLevelAtOffset(const int m, const int nnghbr, const int ox1,
                             const int ox2, const int ox3,
                             const DualArray2D<NeighborBlock> &nghbr) {
  int max_lev = -1;
  for (int n1 = 0; n1 <= 1; ++n1) {
    for (int n2 = 0; n2 <= 1; ++n2) {
      const int idx = NeighborIndex(ox1, ox2, ox3, n1, n2);
      if ((idx >= 0) && (idx < nnghbr) && (nghbr.d_view(m, idx).gid >= 0)) {
        max_lev =
            (nghbr.d_view(m, idx).lev > max_lev) ? nghbr.d_view(m, idx).lev : max_lev;
      }
    }
  }
  return max_lev;
}

KOKKOS_INLINE_FUNCTION
bool IsActiveFCFace(const int v, const int k, const int j, const int i,
                    const RegionIndcs &indcs) {
  if (v == 0) {
    return (i >= indcs.is) && (i <= indcs.ie + 1) &&
           (j >= indcs.js) && (j <= indcs.je) &&
           (k >= indcs.ks) && (k <= indcs.ke);
  } else if (v == 1) {
    return (i >= indcs.is) && (i <= indcs.ie) &&
           (j >= indcs.js) && (j <= indcs.je + 1) &&
           (k >= indcs.ks) && (k <= indcs.ke);
  } else {
    return (i >= indcs.is) && (i <= indcs.ie) &&
           (j >= indcs.js) && (j <= indcs.je) &&
           (k >= indcs.ks) && (k <= indcs.ke + 1);
  }
}

KOKKOS_INLINE_FUNCTION
bool CanProlongateFCFace(const int m, const int nnghbr, const int v,
                         const int k, const int j, const int i,
                         const int ox1, const int ox2, const int ox3,
                         const int my_lev, const RegionIndcs &indcs,
                         const DualArray2D<NeighborBlock> &nghbr) {
  if (!IsActiveFCFace(v, k, j, i, indcs)) {
    return true;
  }

  // Coarse-neighbor prolongation may write active faces only at the physical
  // fine/coarse interface normal to the face component. Active interior faces and active
  // boundary faces owned by same-level or finer neighbors are left untouched.
  if (v == 0) {
    int normal_ox = 0;
    if (i == indcs.is) {
      normal_ox = -1;
    } else if (i == indcs.ie + 1) {
      normal_ox = 1;
    } else {
      return false;
    }
    return (ox1 == normal_ox) && (ox2 == 0) && (ox3 == 0) &&
           (MaxNeighborLevelAtOffset(m, nnghbr, normal_ox, 0, 0, nghbr) < my_lev);
  } else if (v == 1) {
    int normal_ox = 0;
    if (j == indcs.js) {
      normal_ox = -1;
    } else if (j == indcs.je + 1) {
      normal_ox = 1;
    } else {
      return false;
    }
    return (ox1 == 0) && (ox2 == normal_ox) && (ox3 == 0) &&
           (MaxNeighborLevelAtOffset(m, nnghbr, 0, normal_ox, 0, nghbr) < my_lev);
  } else {
    int normal_ox = 0;
    if (k == indcs.ks) {
      normal_ox = -1;
    } else if (k == indcs.ke + 1) {
      normal_ox = 1;
    } else {
      return false;
    }
    return (ox1 == 0) && (ox2 == 0) && (ox3 == normal_ox) &&
           (MaxNeighborLevelAtOffset(m, nnghbr, 0, 0, normal_ox, nghbr) < my_lev);
  }
}

KOKKOS_INLINE_FUNCTION
void StoreProlongatedFCFace(const int writer, const int m, const int nnghbr,
                            const int v, const int k, const int j, const int i,
                            const int ox1, const int ox2, const int ox3,
                            const int my_lev, const RegionIndcs &indcs,
                            const DualArray2D<NeighborBlock> &nghbr,
                            const Real value, const DvceArray4D<Real> &bf) {
  if (CanProlongateFCFace(m, nnghbr, v, k, j, i, ox1, ox2, ox3,
                          my_lev, indcs, nghbr)) {
    bf(m,k,j,i) = value;
#ifdef ATHENAK_DEBUG_FC_AMR_OWNERSHIP
  } else {
    Kokkos::printf("FC AMR ownership blocked writer=%d m=%d v=%d kji=(%d,%d,%d) "
                   "ox=(%d,%d,%d) lev=%d\n",
                   writer, m, v, k, j, i, ox1, ox2, ox3, my_lev);
#endif
  }
}

KOKKOS_INLINE_FUNCTION
void ProlongFCSharedX1FaceOwned(const int m, const int nnghbr,
                                const int k, const int j, const int i,
                                const int fk, const int fj, const int fi,
                                const int ox1, const int ox2, const int ox3,
                                const int my_lev, const bool multi_d,
                                const bool three_d, const RegionIndcs &indcs,
                                const DualArray2D<NeighborBlock> &nghbr,
                                const DvceArray4D<Real> &cbx1f,
                                const DvceArray4D<Real> &bx1f) {
  Real dvar2 = 0.0;
  if (multi_d) {
    Real dl = cbx1f(m,k,j  ,i) - cbx1f(m,k,j-1,i);
    Real dr = cbx1f(m,k,j+1,i) - cbx1f(m,k,j  ,i);
    dvar2 = 0.125*(SIGN(dl) + SIGN(dr))*fmin(fabs(dl), fabs(dr));
  }

  Real dvar3 = 0.0;
  if (three_d) {
    Real dl = cbx1f(m,k  ,j,i) - cbx1f(m,k-1,j,i);
    Real dr = cbx1f(m,k+1,j,i) - cbx1f(m,k  ,j,i);
    dvar3 = 0.125*(SIGN(dl) + SIGN(dr))*fmin(fabs(dl), fabs(dr));
  }

  StoreProlongatedFCFace(1, m, nnghbr, 0, fk, fj, fi, ox1, ox2, ox3,
                         my_lev, indcs, nghbr, cbx1f(m,k,j,i) - dvar2 - dvar3, bx1f);
  if (multi_d) {
    StoreProlongatedFCFace(1, m, nnghbr, 0, fk, fj+1, fi, ox1, ox2, ox3,
                           my_lev, indcs, nghbr, cbx1f(m,k,j,i) + dvar2 - dvar3, bx1f);
  }
  if (three_d) {
    StoreProlongatedFCFace(1, m, nnghbr, 0, fk+1, fj, fi, ox1, ox2, ox3,
                           my_lev, indcs, nghbr, cbx1f(m,k,j,i) - dvar2 + dvar3, bx1f);
    StoreProlongatedFCFace(1, m, nnghbr, 0, fk+1, fj+1, fi, ox1, ox2, ox3,
                           my_lev, indcs, nghbr, cbx1f(m,k,j,i) + dvar2 + dvar3, bx1f);
  }
}

KOKKOS_INLINE_FUNCTION
void ProlongFCSharedX2FaceOwned(const int m, const int nnghbr,
                                const int k, const int j, const int i,
                                const int fk, const int fj, const int fi,
                                const int ox1, const int ox2, const int ox3,
                                const int my_lev, const bool three_d,
                                const RegionIndcs &indcs,
                                const DualArray2D<NeighborBlock> &nghbr,
                                const DvceArray4D<Real> &cbx2f,
                                const DvceArray4D<Real> &bx2f) {
  Real dl = cbx2f(m,k,j,i  ) - cbx2f(m,k,j,i-1);
  Real dr = cbx2f(m,k,j,i+1) - cbx2f(m,k,j,i  );
  Real dvar1 = 0.125*(SIGN(dl) + SIGN(dr))*fmin(fabs(dl), fabs(dr));

  Real dvar3 = 0.0;
  if (three_d) {
    dl = cbx2f(m,k  ,j,i) - cbx2f(m,k-1,j,i);
    dr = cbx2f(m,k+1,j,i) - cbx2f(m,k  ,j,i);
    dvar3 = 0.125*(SIGN(dl) + SIGN(dr))*fmin(fabs(dl), fabs(dr));
  }

  StoreProlongatedFCFace(1, m, nnghbr, 1, fk, fj, fi, ox1, ox2, ox3,
                         my_lev, indcs, nghbr, cbx2f(m,k,j,i) - dvar1 - dvar3, bx2f);
  StoreProlongatedFCFace(1, m, nnghbr, 1, fk, fj, fi+1, ox1, ox2, ox3,
                         my_lev, indcs, nghbr, cbx2f(m,k,j,i) + dvar1 - dvar3, bx2f);
  if (three_d) {
    StoreProlongatedFCFace(1, m, nnghbr, 1, fk+1, fj, fi, ox1, ox2, ox3,
                           my_lev, indcs, nghbr, cbx2f(m,k,j,i) - dvar1 + dvar3, bx2f);
    StoreProlongatedFCFace(1, m, nnghbr, 1, fk+1, fj, fi+1, ox1, ox2, ox3,
                           my_lev, indcs, nghbr, cbx2f(m,k,j,i) + dvar1 + dvar3, bx2f);
  }
}

KOKKOS_INLINE_FUNCTION
void ProlongFCSharedX3FaceOwned(const int m, const int nnghbr,
                                const int k, const int j, const int i,
                                const int fk, const int fj, const int fi,
                                const int ox1, const int ox2, const int ox3,
                                const int my_lev, const bool multi_d,
                                const RegionIndcs &indcs,
                                const DualArray2D<NeighborBlock> &nghbr,
                                const DvceArray4D<Real> &cbx3f,
                                const DvceArray4D<Real> &bx3f) {
  Real dl = cbx3f(m,k,j,i  ) - cbx3f(m,k,j,i-1);
  Real dr = cbx3f(m,k,j,i+1) - cbx3f(m,k,j,i  );
  Real dvar1 = 0.125*(SIGN(dl) + SIGN(dr))*fmin(fabs(dl), fabs(dr));

  Real dvar2 = 0.0;
  if (multi_d) {
    dl = cbx3f(m,k,j  ,i) - cbx3f(m,k,j-1,i);
    dr = cbx3f(m,k,j+1,i) - cbx3f(m,k,j  ,i);
    dvar2 = 0.125*(SIGN(dl) + SIGN(dr))*fmin(fabs(dl), fabs(dr));
  }

  StoreProlongatedFCFace(1, m, nnghbr, 2, fk, fj, fi, ox1, ox2, ox3,
                         my_lev, indcs, nghbr, cbx3f(m,k,j,i) - dvar1 - dvar2, bx3f);
  StoreProlongatedFCFace(1, m, nnghbr, 2, fk, fj, fi+1, ox1, ox2, ox3,
                         my_lev, indcs, nghbr, cbx3f(m,k,j,i) + dvar1 - dvar2, bx3f);
  if (multi_d) {
    StoreProlongatedFCFace(1, m, nnghbr, 2, fk, fj+1, fi, ox1, ox2, ox3,
                           my_lev, indcs, nghbr, cbx3f(m,k,j,i) - dvar1 + dvar2, bx3f);
    StoreProlongatedFCFace(1, m, nnghbr, 2, fk, fj+1, fi+1, ox1, ox2, ox3,
                           my_lev, indcs, nghbr, cbx3f(m,k,j,i) + dvar1 + dvar2, bx3f);
  }
}

KOKKOS_INLINE_FUNCTION
void ProlongFCInternalOwned(const int m, const int nnghbr, const int fk, const int fj,
                            const int fi, const int ox1, const int ox2, const int ox3,
                            const int my_lev, const bool three_d,
                            const RegionIndcs &indcs,
                            const DualArray2D<NeighborBlock> &nghbr,
                            const DvceFaceFld4D<Real> &b) {
  if (three_d) {
    Real Uxx  = 0.0, Vyy  = 0.0, Wzz  = 0.0;
    Real Uxyz = 0.0, Vxyz = 0.0, Wxyz = 0.0;
    for (int jj=0; jj<2; jj++) {
      int jsgn = 2*jj - 1;
      int fjj  = fj + jj, fjp = fj + 2*jj;
      for (int ii=0; ii<2; ii++) {
        int isgn = 2*ii - 1;
        int fii = fi + ii, fip = fi + 2*ii;
        Uxx += isgn*(jsgn*(b.x2f(m,fk  ,fjp,fii) + b.x2f(m,fk+1,fjp,fii)) +
                          (b.x3f(m,fk+2,fjj,fii) - b.x3f(m,fk  ,fjj,fii)));

        Vyy += jsgn*(     (b.x3f(m,fk+2,fjj,fii) - b.x3f(m,fk  ,fjj,fii)) +
                     isgn*(b.x1f(m,fk  ,fjj,fip) + b.x1f(m,fk+1,fjj,fip)));

        Wzz +=       isgn*(b.x1f(m,fk+1,fjj,fip) - b.x1f(m,fk  ,fjj,fip)) +
                     jsgn*(b.x2f(m,fk+1,fjp,fii) - b.x2f(m,fk  ,fjp,fii));

        Uxyz += isgn*jsgn*(b.x1f(m,fk+1,fjj,fip) - b.x1f(m,fk  ,fjj,fip));
        Vxyz += isgn*jsgn*(b.x2f(m,fk+1,fjp,fii) - b.x2f(m,fk  ,fjp,fii));
        Wxyz += isgn*jsgn*(b.x3f(m,fk+2,fjj,fii) - b.x3f(m,fk  ,fjj,fii));
      }
    }
    Uxx *= 0.125;  Vyy *= 0.125;  Wzz *= 0.125;
    Uxyz *= 0.0625; Vxyz *= 0.0625; Wxyz *= 0.0625;

    StoreProlongatedFCFace(2, m, nnghbr, 0, fk, fj, fi+1, ox1, ox2, ox3, my_lev,
                           indcs, nghbr,
                           0.5*(b.x1f(m,fk,fj,fi) + b.x1f(m,fk,fj,fi+2))
                           + Uxx - Vxyz - Wxyz, b.x1f);
    StoreProlongatedFCFace(2, m, nnghbr, 0, fk, fj+1, fi+1, ox1, ox2, ox3, my_lev,
                           indcs, nghbr,
                           0.5*(b.x1f(m,fk,fj+1,fi) + b.x1f(m,fk,fj+1,fi+2))
                           + Uxx - Vxyz + Wxyz, b.x1f);
    StoreProlongatedFCFace(2, m, nnghbr, 0, fk+1, fj, fi+1, ox1, ox2, ox3, my_lev,
                           indcs, nghbr,
                           0.5*(b.x1f(m,fk+1,fj,fi) + b.x1f(m,fk+1,fj,fi+2))
                           + Uxx + Vxyz - Wxyz, b.x1f);
    StoreProlongatedFCFace(2, m, nnghbr, 0, fk+1, fj+1, fi+1, ox1, ox2, ox3, my_lev,
                           indcs, nghbr,
                           0.5*(b.x1f(m,fk+1,fj+1,fi) + b.x1f(m,fk+1,fj+1,fi+2))
                           + Uxx + Vxyz + Wxyz, b.x1f);

    StoreProlongatedFCFace(2, m, nnghbr, 1, fk, fj+1, fi, ox1, ox2, ox3, my_lev,
                           indcs, nghbr,
                           0.5*(b.x2f(m,fk,fj,fi) + b.x2f(m,fk,fj+2,fi))
                           + Vyy - Uxyz - Wxyz, b.x2f);
    StoreProlongatedFCFace(2, m, nnghbr, 1, fk, fj+1, fi+1, ox1, ox2, ox3, my_lev,
                           indcs, nghbr,
                           0.5*(b.x2f(m,fk,fj,fi+1) + b.x2f(m,fk,fj+2,fi+1))
                           + Vyy - Uxyz + Wxyz, b.x2f);
    StoreProlongatedFCFace(2, m, nnghbr, 1, fk+1, fj+1, fi, ox1, ox2, ox3, my_lev,
                           indcs, nghbr,
                           0.5*(b.x2f(m,fk+1,fj,fi) + b.x2f(m,fk+1,fj+2,fi))
                           + Vyy + Uxyz - Wxyz, b.x2f);
    StoreProlongatedFCFace(2, m, nnghbr, 1, fk+1, fj+1, fi+1, ox1, ox2, ox3, my_lev,
                           indcs, nghbr,
                           0.5*(b.x2f(m,fk+1,fj,fi+1) + b.x2f(m,fk+1,fj+2,fi+1))
                           + Vyy + Uxyz + Wxyz, b.x2f);

    StoreProlongatedFCFace(2, m, nnghbr, 2, fk+1, fj, fi, ox1, ox2, ox3, my_lev,
                           indcs, nghbr,
                           0.5*(b.x3f(m,fk+2,fj,fi) + b.x3f(m,fk,fj,fi))
                           + Wzz - Uxyz - Vxyz, b.x3f);
    StoreProlongatedFCFace(2, m, nnghbr, 2, fk+1, fj, fi+1, ox1, ox2, ox3, my_lev,
                           indcs, nghbr,
                           0.5*(b.x3f(m,fk+2,fj,fi+1) + b.x3f(m,fk,fj,fi+1))
                           + Wzz - Uxyz + Vxyz, b.x3f);
    StoreProlongatedFCFace(2, m, nnghbr, 2, fk+1, fj+1, fi, ox1, ox2, ox3, my_lev,
                           indcs, nghbr,
                           0.5*(b.x3f(m,fk+2,fj+1,fi) + b.x3f(m,fk,fj+1,fi))
                           + Wzz + Uxyz - Vxyz, b.x3f);
    StoreProlongatedFCFace(2, m, nnghbr, 2, fk+1, fj+1, fi+1, ox1, ox2, ox3, my_lev,
                           indcs, nghbr,
                           0.5*(b.x3f(m,fk+2,fj+1,fi+1) + b.x3f(m,fk,fj+1,fi+1))
                           + Wzz + Uxyz + Vxyz, b.x3f);
  } else {
    Real tmp1 = 0.25*(b.x2f(m,fk,fj+2,fi+1) - b.x2f(m,fk,fj,  fi+1)
                    - b.x2f(m,fk,fj+2,fi  ) + b.x2f(m,fk,fj,  fi  ));
    Real tmp2 = 0.25*(b.x1f(m,fk,fj,  fi  ) - b.x1f(m,fk,fj,  fi+2)
                    - b.x1f(m,fk,fj+1,fi  ) + b.x1f(m,fk,fj+1,fi+2));
    StoreProlongatedFCFace(2, m, nnghbr, 0, fk, fj, fi+1, ox1, ox2, ox3, my_lev,
                           indcs, nghbr,
                           0.5*(b.x1f(m,fk,fj,fi) + b.x1f(m,fk,fj,fi+2)) + tmp1,
                           b.x1f);
    StoreProlongatedFCFace(2, m, nnghbr, 0, fk, fj+1, fi+1, ox1, ox2, ox3, my_lev,
                           indcs, nghbr,
                           0.5*(b.x1f(m,fk,fj+1,fi) + b.x1f(m,fk,fj+1,fi+2)) + tmp1,
                           b.x1f);
    StoreProlongatedFCFace(2, m, nnghbr, 1, fk, fj+1, fi, ox1, ox2, ox3, my_lev,
                           indcs, nghbr,
                           0.5*(b.x2f(m,fk,fj,fi) + b.x2f(m,fk,fj+2,fi)) + tmp2,
                           b.x2f);
    StoreProlongatedFCFace(2, m, nnghbr, 1, fk, fj+1, fi+1, ox1, ox2, ox3, my_lev,
                           indcs, nghbr,
                           0.5*(b.x2f(m,fk,fj,fi+1) + b.x2f(m,fk,fj+2,fi+1)) + tmp2,
                           b.x2f);
  }
}

} // namespace
//----------------------------------------------------------------------------------------
//! \fn void FillCoarseInBndryCC()
//! \brief To ensure that the coarse array is up-to-date in all neighboring cells touched
//! by the prolongation interpolation stencil, data is restricted to coarse array in
//! boundaries between MeshBlocks at the same level.

void MeshBoundaryValuesCC::FillCoarseInBndryCC(DvceArray5D<Real> &a,
                                               DvceArray5D<Real> &ca,
                                               bool is_z4c) {
  // create local references for variables in kernel
  int nmb = pmy_pack->nmb_thispack;
  int nnghbr = pmy_pack->pmb->nnghbr;
  //bool not_z4c = (pmbp->pz4c == nullptr)? true : false;

  int nvar = a.extent_int(1);  // TODO(@user): 2nd index from L of in array must be NVAR
  int nmnv = nmb*nnghbr*nvar;
  auto &nghbr = pmy_pack->pmb->nghbr;
  auto &mblev = pmy_pack->pmb->mb_lev;
  auto &rbuf = recvbuf;
  auto &indcs  = pmy_pack->pmesh->mb_indcs;
  const bool multi_d = pmy_pack->pmesh->multi_d;
  const bool three_d = pmy_pack->pmesh->three_d;
  auto &nx1 = pmy_pack->pmesh->mb_indcs.nx1;
  auto &nx2 = pmy_pack->pmesh->mb_indcs.nx2;
  auto &nx3 = pmy_pack->pmesh->mb_indcs.nx3;
  auto& restrict_2nd = pmy_pack->pmesh->pmr->weights.restrict_2nd;
  auto& restrict_4th = pmy_pack->pmesh->pmr->weights.restrict_4th;
  auto& restrict_4th_edge = pmy_pack->pmesh->pmr->weights.restrict_4th_edge;

  // Restrict data into coarse array in any boundary filled with data from the same
  // level.  This ensures data in the coarse array at corners where one direction is a
  // coarser level and the other the same level is filled properly.
  // (Only needed in multidimensions)

  if (multi_d) {
    auto &cis = indcs.cis;
    auto &cjs = indcs.cjs;
    auto &cks = indcs.cks;
    // Outer loop over (# of MeshBlocks)*(# of buffers)*(# of variables)
    Kokkos::TeamPolicy<> policy(DevExeSpace(), nmnv, Kokkos::AUTO);
    Kokkos::parallel_for("ProlCCSame", policy, KOKKOS_LAMBDA(TeamMember_t tmember) {
      const int m = (tmember.league_rank())/(nnghbr*nvar);
      const int n = (tmember.league_rank() - m*(nnghbr*nvar))/nvar;
      const int v = (tmember.league_rank() - m*(nnghbr*nvar) - n*nvar);

      // only restrict when neighbor exists and is at SAME level
      if ((nghbr.d_view(m,n).gid >= 0) && (nghbr.d_view(m,n).lev == mblev.d_view(m))) {
        // loop over indices for receives at same level, but convert loop limits to
        // coarse array
        int il = (rbuf[n].isame[0].bis + cis)/2;
        int iu = (rbuf[n].isame[0].bie + cis)/2;
        int jl = (rbuf[n].isame[0].bjs + cjs)/2;
        int ju = (rbuf[n].isame[0].bje + cjs)/2;
        int kl = (rbuf[n].isame[0].bks + cks)/2;
        int ku = (rbuf[n].isame[0].bke + cks)/2;

        const int ni = iu - il + 1;
        const int nj = ju - jl + 1;
        const int nk = ku - kl + 1;
        const int nkji = nk*nj*ni;
        const int nji  = nj*ni;

        // Middle loop over k,j,i
        Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nkji),[&](const int idx) {
          int k = idx/nji;
          int j = (idx - k*nji)/ni;
          int i = (idx - k*nji - j*ni) + il;
          j += jl;
          k += kl;

          // indices refer to coarse array.  So must compute indices for fine array
          int finei = (i - indcs.cis)*2 + indcs.is;
          int finej = (j - indcs.cjs)*2 + indcs.js;
          int finek = (k - indcs.cks)*2 + indcs.ks;

          // restrict in 2D
          if (!(three_d)) {
            ca(m,v,kl,j,i) = 0.25*(a(m,v,kl,finej  ,finei) + a(m,v,kl,finej  ,finei+1)
                                 + a(m,v,kl,finej+1,finei) + a(m,v,kl,finej+1,finei+1));
          // restrict in 3D
          } else {
            if (!is_z4c) {
              ca(m,v,k,j,i) = 0.125*(
                  a(m,v,finek  ,finej  ,finei) + a(m,v,finek  ,finej  ,finei+1)
                + a(m,v,finek  ,finej+1,finei) + a(m,v,finek  ,finej+1,finei+1)
                + a(m,v,finek+1,finej,  finei) + a(m,v,finek+1,finej,  finei+1)
                + a(m,v,finek+1,finej+1,finei) + a(m,v,finek+1,finej+1,finei+1));
            } else {
                switch (indcs.ng) {
                  case 2: ca(m,v,k,j,i) = RestrictInterpolation<2>(m,v,finek,finej,finei,
                              nx1,nx2,nx3,a,restrict_2nd,restrict_4th,restrict_4th_edge);
                          break;
                  case 4: ca(m,v,k,j,i) = RestrictInterpolation<4>(m,v,finek,finej,finei,
                              nx1,nx2,nx3,a,restrict_2nd,restrict_4th,restrict_4th_edge);
                          break;
                }
            }
          }
        });
      }
      tmember.team_barrier();
    });
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void ProlongateCC()
//! \brief Prolongate data at boundaries for cell-centered data.
//! Code here is based on MeshRefinement::ProlongateCellCenteredValues() in C++ version

void MeshBoundaryValuesCC::ProlongateCC(DvceArray5D<Real> &a, DvceArray5D<Real> &ca,
    bool is_z4c) {
  // create local references for variables in kernel
  int nmb = pmy_pack->nmb_thispack;
  int nnghbr = pmy_pack->pmb->nnghbr;

  // ptr to z4c, which requires different prolongation/restriction scheme
  //bool not_z4c = (pmbp->pz4c == nullptr)? true : false;

  int nvar = a.extent_int(1);  // TODO(@user): 2nd index from L of in array must be NVAR
  int nmnv = nmb*nnghbr*nvar;
  auto &nghbr = pmy_pack->pmb->nghbr;
  auto &mblev = pmy_pack->pmb->mb_lev;
  auto &rbuf = recvbuf;
  auto &indcs  = pmy_pack->pmesh->mb_indcs;
  const bool multi_d = pmy_pack->pmesh->multi_d;
  const bool three_d = pmy_pack->pmesh->three_d;
  auto &nx1 = indcs.nx1;
  auto &nx2 = indcs.nx2;
  auto &nx3 = indcs.nx3;
  auto& prolong_2nd = pmy_pack->pmesh->pmr->weights.prolong_2nd;
  auto& prolong_4th = pmy_pack->pmesh->pmr->weights.prolong_4th;

  // Outer loop over (# of MeshBlocks)*(# of buffers)*(# of variables)
  Kokkos::TeamPolicy<> policy(DevExeSpace(), nmnv, Kokkos::AUTO);
  Kokkos::parallel_for("ProlCC", policy, KOKKOS_LAMBDA(TeamMember_t tmember) {
    const int m = (tmember.league_rank())/(nnghbr*nvar);
    const int n = (tmember.league_rank() - m*(nnghbr*nvar))/nvar;
    const int v = (tmember.league_rank() - m*(nnghbr*nvar) - n*nvar);

    // only prolongate when neighbor exists and is at coarser level
    if ((nghbr.d_view(m,n).gid >= 0) && (nghbr.d_view(m,n).lev < mblev.d_view(m))) {
      // loop over indices for prolongation on this buffer
      int il = rbuf[n].iprol[0].bis;
      int iu = rbuf[n].iprol[0].bie;
      int jl = rbuf[n].iprol[0].bjs;
      int ju = rbuf[n].iprol[0].bje;
      int kl = rbuf[n].iprol[0].bks;
      int ku = rbuf[n].iprol[0].bke;
      const int ni = iu - il + 1;
      const int nj = ju - jl + 1;
      const int nk = ku - kl + 1;
      const int nkji = nk*nj*ni;
      const int nji  = nj*ni;

      // Middle loop over k,j,i
      Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nkji), [&](const int idx) {
        int k = idx/nji;
        int j = (idx - k*nji)/ni;
        int i = (idx - k*nji - j*ni) + il;
        j += jl;
        k += kl;

        // indices for prolongation refer to coarse array.  So must compute
        // indices for fine array
        int fi = (i - indcs.cis)*2 + indcs.is;
        int fj = (j - indcs.cjs)*2 + indcs.js;
        int fk = (k - indcs.cks)*2 + indcs.ks;
        // call inlined prolongation operator for CC variables
        if (!is_z4c) {
          ProlongCC(m,v,k,j,i,fk,fj,fi,multi_d,three_d,ca,a);
        } else {
          switch (indcs.ng) {
            case 2: HighOrderProlongCC<2>(m,v,k,j,i,fk,fj,fi,nx1,nx2,nx3,
                                          ca,a,prolong_2nd);
                    break;
            case 4: HighOrderProlongCC<4>(m,v,k,j,i,fk,fj,fi,nx1,nx2,nx3,
                                          ca,a,prolong_4th);
                    break;
          }
        }
      });
    }
    tmember.team_barrier();
  });
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void FillCoarseInBndryFC()
//! \brief As in the case of cell-centered variables, to ensure that the coarse field is
//! up-to-date in all neighboring cells touched by the prolongation interpolation stencil,
//! data is also restricted to coarse array in boundaries between MeshBlocks at the same
//! level.

void MeshBoundaryValuesFC::FillCoarseInBndryFC(DvceFaceFld4D<Real> &b,
                                           DvceFaceFld4D<Real> &cb) {
  // create local references for variables in kernel
  int nmb = pmy_pack->nmb_thispack;
  int nnghbr = pmy_pack->pmb->nnghbr;

  auto &nghbr = pmy_pack->pmb->nghbr;
  auto &indcs  = pmy_pack->pmesh->mb_indcs;
  auto &mblev = pmy_pack->pmb->mb_lev;
  bool &multi_d = pmy_pack->pmesh->multi_d;
  bool &three_d = pmy_pack->pmesh->three_d;

  // Restrict data into coarse array in any boundary filled with data from the same
  // level. (Only needed in multidimensions)

  if (multi_d) {
    int nmnv = 3*nmb*nnghbr;
    auto &rbuf = recvbuf;
    auto &cis = indcs.cis;
    auto &cjs = indcs.cjs;
    auto &cks = indcs.cks;
    // Outer loop over (# of MeshBlocks)*(# of buffers)*(# of variables)
    Kokkos::TeamPolicy<> policy(DevExeSpace(), nmnv, Kokkos::AUTO);
    Kokkos::parallel_for("ProlFCSame", policy, KOKKOS_LAMBDA(TeamMember_t tmember) {
      const int m = (tmember.league_rank())/(3*nnghbr);
      const int n = (tmember.league_rank() - m*(3*nnghbr))/3;
      const int v = (tmember.league_rank() - m*(3*nnghbr) - 3*n);

      // only restrict when neighbor exists and is at SAME level
      if ((nghbr.d_view(m,n).gid >= 0) && (nghbr.d_view(m,n).lev == mblev.d_view(m))) {
        // loop over indices for receives at same level, but convert loop limits to
        // coarse array
        int il = (rbuf[n].isame[v].bis + cis)/2;
        int iu = (rbuf[n].isame[v].bie + cis)/2;
        int jl = (rbuf[n].isame[v].bjs + cjs)/2;
        int ju = (rbuf[n].isame[v].bje + cjs)/2;
        int kl = (rbuf[n].isame[v].bks + cks)/2;
        int ku = (rbuf[n].isame[v].bke + cks)/2;

        const int ni = iu - il + 1;
        const int nj = ju - jl + 1;
        const int nk = ku - kl + 1;
        const int nkji = nk*nj*ni;
        const int nji  = nj*ni;

        // Middle loop over k,j,i
        Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nkji),[&](const int idx) {
          int k = idx/nji;
          int j = (idx - k*nji)/ni;
          int i = (idx - k*nji - j*ni) + il;
          j += jl;
          k += kl;

          // indices refer to coarse array.  So must compute indices for fine array
          int fk = (k - indcs.cks)*2 + indcs.ks;
          int fj = (j - indcs.cjs)*2 + indcs.js;
          int fi = (i - indcs.cis)*2 + indcs.is;

          // restrict in 2D
          if (!(three_d)) {
            if (v==0) {
              cb.x1f(m,kl,j,i) = 0.5*(b.x1f(m,kl,fj,fi) + b.x1f(m,kl,fj+1,fi));
            } else if (v==1) {
              cb.x2f(m,kl,j,i) = 0.5*(b.x2f(m,kl,fj,fi) + b.x2f(m,kl,fj,fi+1));
            } else {
              Real b3c = 0.25*(b.x3f(m,kl,fj  ,fi) + b.x3f(m,kl,fj  ,fi+1)
                             + b.x3f(m,kl,fj+1,fi) + b.x3f(m,kl,fj+1,fi+1));
              cb.x3f(m,kl  ,j,i) = b3c;
              cb.x3f(m,kl+1,j,i) = b3c;
            }

          // restrict in 3D
          } else {
            if (v==0) {
              cb.x1f(m,k,j,i) = 0.25*(b.x1f(m,fk  ,fj,fi) + b.x1f(m,fk  ,fj+1,fi)
                                    + b.x1f(m,fk+1,fj,fi) + b.x1f(m,fk+1,fj+1,fi));
            } else if (v==1) {
              cb.x2f(m,k,j,i) = 0.25*(b.x2f(m,fk  ,fj,fi) + b.x2f(m,fk  ,fj,fi+1)
                                    + b.x2f(m,fk+1,fj,fi) + b.x2f(m,fk+1,fj,fi+1));
            } else {
              cb.x3f(m,k,j,i) = 0.25*(b.x3f(m,fk,fj  ,fi) + b.x3f(m,fk,fj  ,fi+1)
                                    + b.x3f(m,fk,fj+1,fi) + b.x3f(m,fk,fj+1,fi+1));
            }
          }
        });
      }
      tmember.team_barrier();
    });
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void ProlongateFC()
//! \brief Prolongate data at boundaries for face-centered data (e.g. magnetic fields).

void MeshBoundaryValuesFC::ProlongateFC(DvceFaceFld4D<Real> &b, DvceFaceFld4D<Real> &cb) {
  // create local references for variables in kernel
  int nmb = pmy_pack->nmb_thispack;
  int nnghbr = pmy_pack->pmb->nnghbr;

  auto &nghbr = pmy_pack->pmb->nghbr;
  auto &indcs  = pmy_pack->pmesh->mb_indcs;
  auto &mblev = pmy_pack->pmb->mb_lev;
  bool &multi_d = pmy_pack->pmesh->multi_d;
  bool &three_d = pmy_pack->pmesh->three_d;

  // Prolongate b.x1f/b.x2f/b.x3f at all shared coarse/fine cell edges
  // Code here is based on MeshRefinement::ProlongateSharedFieldX1/2/3() and
  // MeshRefinement::ProlongateInternalField() in C++ version

  // Outer loop over (# of MeshBlocks)*(# of buffers)*(three field components)
  {int nmnv = 3*nmb*nnghbr;
  auto &rbuf = recvbuf;
  Kokkos::TeamPolicy<> policy(DevExeSpace(), nmnv, Kokkos::AUTO);
  Kokkos::parallel_for("ProFC-2d-shared", policy, KOKKOS_LAMBDA(TeamMember_t tmember) {
    const int m = (tmember.league_rank())/(3*nnghbr);
    const int n = (tmember.league_rank() - m*(3*nnghbr))/3;
    const int v = (tmember.league_rank() - m*(3*nnghbr) - 3*n);

    // only prolongate when neighbor exists and is at coarser level
    if ((nghbr.d_view(m,n).gid >= 0) && (nghbr.d_view(m,n).lev < mblev.d_view(m))) {
      int ox1, ox2, ox3;
      NeighborOffsetFromIndex(n, ox1, ox2, ox3);
      const int my_lev = mblev.d_view(m);

      int il = rbuf[n].iprol[v].bis;
      int iu = rbuf[n].iprol[v].bie;
      int jl = rbuf[n].iprol[v].bjs;
      int ju = rbuf[n].iprol[v].bje;
      int kl = rbuf[n].iprol[v].bks;
      int ku = rbuf[n].iprol[v].bke;
      const int ni = iu - il + 1;
      const int nj = ju - jl + 1;
      const int nk = ku - kl + 1;
      const int nkji = nk*nj*ni;
      const int nji  = nj*ni;

      // Middle loop over k,j,i
      Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember,nkji),[&](const int idx) {
        int k = idx/nji;
        int j = (idx - k*nji)/ni;
        int i = (idx - k*nji - j*ni) + il;
        j += jl;
        k += kl;

        int fi = (i - indcs.cis)*2 + indcs.is;                   // fine i
        int fj = (multi_d)? ((j - indcs.cjs)*2 + indcs.js) : j;  // fine j
        int fk = (three_d)? ((k - indcs.cks)*2 + indcs.ks) : k;  // fine k

        // Prolongate face-centered fields at shared faces betwen fine and coarse cells
        // by calling inlined prolongation operator for FC variables
        if (v==0) {
          ProlongFCSharedX1FaceOwned(m,nnghbr,k,j,i,fk,fj,fi,ox1,ox2,ox3,my_lev,
                                     multi_d,three_d,indcs,nghbr,cb.x1f,b.x1f);
        } else if (v==1) {
          ProlongFCSharedX2FaceOwned(m,nnghbr,k,j,i,fk,fj,fi,ox1,ox2,ox3,my_lev,
                                     three_d,indcs,nghbr,cb.x2f,b.x2f);
        } else {
          ProlongFCSharedX3FaceOwned(m,nnghbr,k,j,i,fk,fj,fi,ox1,ox2,ox3,my_lev,
                                     multi_d,indcs,nghbr,cb.x3f,b.x3f);
        }
      });
    }
    tmember.team_barrier();
  });}

  // Now prolongate b.x1f/b.x2f/b.x3f at interior fine cells using the 2nd-order
  // divergence-preserving interpolation scheme of Toth & Roe, JCP 180, 736 (2002).
  // Note prolongation at shared coarse/fine cell edges must be completed first as
  // interpolation formulae use these values.

  // Outer loop over (# of MeshBlocks)*(# of buffers)
  {int nmn = nmb*nnghbr;
  bool &one_d = pmy_pack->pmesh->one_d;
  auto &rbuf = recvbuf;
  Kokkos::TeamPolicy<> policy(DevExeSpace(), nmn, Kokkos::AUTO);
  Kokkos::parallel_for("ProFC-2d-int", policy, KOKKOS_LAMBDA(TeamMember_t tmember) {
    const int m = (tmember.league_rank())/(nnghbr);
    const int n = (tmember.league_rank() - m*(nnghbr));

    // only prolongate when neighbor exists and is at coarser level
    if ((nghbr.d_view(m,n).gid >= 0) && (nghbr.d_view(m,n).lev < mblev.d_view(m))) {
      int ox1, ox2, ox3;
      NeighborOffsetFromIndex(n, ox1, ox2, ox3);
      const int my_lev = mblev.d_view(m);

      // use prolongation indices of different field components for interior fine cells
      int il = rbuf[n].iprol[2].bis;
      int iu = rbuf[n].iprol[2].bie;
      int jl = rbuf[n].iprol[0].bjs;
      int ju = rbuf[n].iprol[0].bje;
      int kl = rbuf[n].iprol[1].bks;
      int ku = rbuf[n].iprol[1].bke;
      const int ni = iu - il + 1;
      const int nj = ju - jl + 1;
      const int nk = ku - kl + 1;
      const int nkji = nk*nj*ni;
      const int nji  = nj*ni;

      // Middle loop over k,j,i
      Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember,nkji),[&](const int idx) {
        int k = idx/nji;
        int j = (idx - k*nji)/ni;
        int i = (idx - k*nji - j*ni) + il;
        j += jl;
        k += kl;

        int fi = (i - indcs.cis)*2 + indcs.is;   // fine i
        int fj = (j - indcs.cjs)*2 + indcs.js;   // fine j
        int fk = (k - indcs.cks)*2 + indcs.ks;   // fine k

        if (one_d) {
          // In 1D, interior face field is trivial
          StoreProlongatedFCFace(2, m, nnghbr, 0, fk, fj, fi+1, ox1, ox2, ox3,
                                 my_lev, indcs, nghbr,
                                 0.5*(b.x1f(m,fk,fj,fi) + b.x1f(m,fk,fj,fi+2)),
                                 b.x1f);
        } else {
          // in multi-D call inlined prolongation operator for FC fields at internal faces
          ProlongFCInternalOwned(m,nnghbr,fk,fj,fi,ox1,ox2,ox3,my_lev,
                                 three_d,indcs,nghbr,b);
        }
      });
    }
    tmember.team_barrier();
  });}

  return;
}
