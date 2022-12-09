//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file prolongation.cpp
//! \brief functions to prolongate data at boundaries for cell-centered and face-centered
//! values.  Functions are members of either BValCC or BValFC classes.

#include <cstdlib>
#include <iostream>
#include <iomanip>    // std::setprecision()

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "bvals.hpp"

//----------------------------------------------------------------------------------------
//! \fn void ProlongCC()
//! \brief Prolongate data at boundaries for cell-centered data. To ensure that the
//! coarse array is up-to-date in all neighboring cells touched by the prolongation
//! interpolation stencil, data is also restricted to coarse array in boundaries
//! between MeshBlocks at the same level.

void BoundaryValuesCC::ProlongCC(DvceArray5D<Real> &a, DvceArray5D<Real> &ca) {
  // create local references for variables in kernel
  int nmb = pmy_pack->nmb_thispack;
  int nnghbr = pmy_pack->pmb->nnghbr;

  int nvar = a.extent_int(1);  // TODO(@user): 2nd index from L of in array must be NVAR
  int nmnv = nmb*nnghbr*nvar;
  auto &nghbr = pmy_pack->pmb->nghbr;
  auto &mblev = pmy_pack->pmb->mb_lev;
  auto &rbuf = recv_buf;
  auto &indcs  = pmy_pack->pmesh->mb_indcs;
  bool &multi_d = pmy_pack->pmesh->multi_d;
  bool &two_d = pmy_pack->pmesh->two_d;
  bool &three_d = pmy_pack->pmesh->three_d;

  // First restrict data into coarse array in any boundary filled with data from the same
  // level.  This ensures data in the coarse array at corners where one direction is a
  // coarser level and the other the same level is filled properly.
  // (Only needed in multidimensions)

  if (multi_d) {
    auto &cis = indcs.cis;
    auto &cie = indcs.cie;
    auto &cjs = indcs.cjs;
    auto &cje = indcs.cje;
    auto &cks = indcs.cks;
    auto &cke = indcs.cke;
    // Outer loop over (# of MeshBlocks)*(# of buffers)*(# of variables)
    Kokkos::TeamPolicy<> policy(DevExeSpace(), nmnv, Kokkos::AUTO);
    Kokkos::parallel_for("ProlSame", policy, KOKKOS_LAMBDA(TeamMember_t tmember) {
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
          if (two_d) {
            ca(m,v,kl,j,i) = 0.25*(a(m,v,kl,finej  ,finei) + a(m,v,kl,finej  ,finei+1)
                                 + a(m,v,kl,finej+1,finei) + a(m,v,kl,finej+1,finei+1));
          // restrict in 3D
          } else {
            ca(m,v,k,j,i) = 0.125*(
                 a(m,v,finek  ,finej  ,finei) + a(m,v,finek  ,finej  ,finei+1)
               + a(m,v,finek  ,finej+1,finei) + a(m,v,finek  ,finej+1,finei+1)
               + a(m,v,finek+1,finej,  finei) + a(m,v,finek+1,finej,  finei+1)
               + a(m,v,finek+1,finej+1,finei) + a(m,v,finek+1,finej+1,finei+1));
          }
        });
      }
    });
  }

  // Now prolongate cell-centered variables.
  // Code here is based on MeshRefinement::ProlongateCellCenteredValues() in C++ version

  // Outer loop over (# of MeshBlocks)*(# of buffers)*(# of variables)
  Kokkos::TeamPolicy<> policy(DevExeSpace(), nmnv, Kokkos::AUTO);
  Kokkos::parallel_for("RecvBuff", policy, KOKKOS_LAMBDA(TeamMember_t tmember) {
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
        int finei = (i - indcs.cis)*2 + indcs.is;
        int finej = (j - indcs.cjs)*2 + indcs.js;
        int finek = (k - indcs.cks)*2 + indcs.ks;

        // calculate x1-gradient using the min-mod limiter
        Real dl = ca(m,v,k,j,i  ) - ca(m,v,k,j,i-1);
        Real dr = ca(m,v,k,j,i+1) - ca(m,v,k,j,i  );
        Real dvar1 = 0.125*(SIGN(dl) + SIGN(dr))*fmin(fabs(dl), fabs(dr));

        // calculate x2-gradient using the min-mod limiter
        Real dvar2 = 0.0;
        if (multi_d) {
          dl = ca(m,v,k,j  ,i) - ca(m,v,k,j-1,i);
          dr = ca(m,v,k,j+1,i) - ca(m,v,k,j  ,i);
          dvar2 = 0.125*(SIGN(dl) + SIGN(dr))*fmin(fabs(dl), fabs(dr));
        }

        // calculate x1-gradient using the min-mod limiter
        Real dvar3 = 0.0;
        if (three_d) {
          dl = ca(m,v,k  ,j,i) - ca(m,v,k-1,j,i);
          dr = ca(m,v,k+1,j,i) - ca(m,v,k  ,j,i);
          dvar3 = 0.125*(SIGN(dl) + SIGN(dr))*fmin(fabs(dl), fabs(dr));
        }

        // interpolate to the finer grid
        a(m,v,finek,finej,finei  ) = ca(m,v,k,j,i) - dvar1 - dvar2 - dvar3;
        a(m,v,finek,finej,finei+1) = ca(m,v,k,j,i) + dvar1 - dvar2 - dvar3;
        if (multi_d) {
          a(m,v,finek,finej+1,finei  ) = ca(m,v,k,j,i) - dvar1 + dvar2 - dvar3;
          a(m,v,finek,finej+1,finei+1) = ca(m,v,k,j,i) + dvar1 + dvar2 - dvar3;
        }
        if (three_d) {
          a(m,v,finek+1,finej  ,finei  ) = ca(m,v,k,j,i) - dvar1 - dvar2 + dvar3;
          a(m,v,finek+1,finej  ,finei+1) = ca(m,v,k,j,i) + dvar1 - dvar2 + dvar3;
          a(m,v,finek+1,finej+1,finei  ) = ca(m,v,k,j,i) - dvar1 + dvar2 + dvar3;
          a(m,v,finek+1,finej+1,finei+1) = ca(m,v,k,j,i) + dvar1 + dvar2 + dvar3;
        }
      });
    }
  });

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void ProlongFC()
//! \brief Prolongate data at boundaries for face-centered data (e.g. magnetic fields).
//! This also requires restriction of all three field components in boundaries between
//! MeshBlocks at the same level.

void BoundaryValuesFC::ProlongFC(DvceFaceFld4D<Real> &b, DvceFaceFld4D<Real> &cb) {
  // create local references for variables in kernel
  int nmb = pmy_pack->nmb_thispack;
  int nnghbr = pmy_pack->pmb->nnghbr;

  auto &nghbr = pmy_pack->pmb->nghbr;
  auto &indcs  = pmy_pack->pmesh->mb_indcs;
  auto &mblev = pmy_pack->pmb->mb_lev;
  bool &multi_d = pmy_pack->pmesh->multi_d;
  bool &two_d = pmy_pack->pmesh->two_d;
  bool &three_d = pmy_pack->pmesh->three_d;

  // First restrict data into coarse array in any boundary filled with data from the same
  // level. (Only needed in multidimensions)

  if (multi_d) {
    int nmnv = 3*nmb*nnghbr;
    auto &rbuf = recv_buf;
    auto &cis = indcs.cis;
    auto &cie = indcs.cie;
    auto &cjs = indcs.cjs;
    auto &cje = indcs.cje;
    auto &cks = indcs.cks;
    auto &cke = indcs.cke;
    // Outer loop over (# of MeshBlocks)*(# of buffers)*(# of variables)
    Kokkos::TeamPolicy<> policy(DevExeSpace(), nmnv, Kokkos::AUTO);
    Kokkos::parallel_for("ProlSame", policy, KOKKOS_LAMBDA(TeamMember_t tmember) {
      const int m = (tmember.league_rank())/(3*nnghbr);
      const int n = (tmember.league_rank() - m*(3*nnghbr))/3;
      const int v = (tmember.league_rank() - m*(3*nnghbr) - 3*n);

      // only restrict when neighbor exists and is at SAME level
      if ((nghbr.d_view(m,n).gid >= 0) && (nghbr.d_view(m,n).lev == mblev.d_view(m))) {
//      if (((n==0) || (n==4) || (n==8) || (n==12) || (n==24) || (n==28)) &&
//          (nghbr.d_view(m,n).gid >= 0) && (nghbr.d_view(m,n).lev == mblev.d_view(m))) {
        // loop over indices for receives at same level, but convert loop limits to
        // coarse array
        int il = (rbuf[n].isame[v].bis + cis)/2;
        int iu = (rbuf[n].isame[v].bie + cis)/2;
        int jl = (rbuf[n].isame[v].bjs + cjs)/2;
        int ju = (rbuf[n].isame[v].bje + cjs)/2;
        int kl = (rbuf[n].isame[v].bks + cks)/2;
        int ku = (rbuf[n].isame[v].bke + cks)/2;

//        int il = cis, iu = cie;
//        if (n==0) {il = cis-1, iu = cis-1;}
//        if (n==4) {il = cie+1, iu = cie+1;}
//        int jl = cjs, ju = cje;
//        if (n==8) {jl = cjs-1, ju = cjs-1;}
//        if (n==12) {jl = cje+1, ju = cje+1;}
//        int kl = cks, ku = cke;
//        if (n==24) {kl = cks-1, ku = cks-1;}
//        if (n==28) {kl = cke+1, ku = cke+1;}
//
//        if (v==0) {iu+=1;}
//        if (v==1) {ju+=1;}
//        if (v==2 && three_d) {ku+=1;}

//        int il = rbuf[n].iprol[v].bis;
//        int iu = rbuf[n].iprol[v].bie;
//        int jl = rbuf[n].iprol[v].bjs;
//        int ju = rbuf[n].iprol[v].bje;
//        int kl = rbuf[n].iprol[v].bks;
//        int ku = rbuf[n].iprol[v].bke;
        const int nj = ju - jl + 1;
        const int nk = ku - kl + 1;
        const int nkj  = nk*nj;

        // Middle loop over k,j
        Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nkj), [&](const int idx) {
          int k = idx / nj;
          int j = (idx - k * nj) + jl;
          k += kl;

          // indices refer to coarse array.  So must compute indices for fine array
          int fj = (j - indcs.cjs)*2 + indcs.js;
          int fk = (k - indcs.cks)*2 + indcs.ks;

          if (two_d) {
            if (v==0) {
              // restrict B1
              Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il,iu+1),
              [&](const int i) {
                int fi = (i - indcs.cis)*2 + indcs.is;
                cb.x1f(m,kl,j,i) = 0.5*(b.x1f(m,kl,fj,fi) + b.x1f(m,kl,fj+1,fi));
              });
            } else if (v==1) {
            // restrict B2
              Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il,iu+1),
              [&](const int i) {
                int fi = (i - indcs.cis)*2 + indcs.is;
                cb.x2f(m,kl,j,i) = 0.5*(b.x2f(m,kl,fj,fi) + b.x2f(m,kl,fj,fi+1));
              });
            } else {
              // restrict B3
              Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il,iu+1),
              [&](const int i) {
                int fi = (i - indcs.cis)*2 + indcs.is;
                Real b3c = 0.25*(b.x3f(m,kl,fj  ,fi) + b.x3f(m,kl,fj  ,fi+1)
                               + b.x3f(m,kl,fj+1,fi) + b.x3f(m,kl,fj+1,fi+1));
                cb.x3f(m,kl  ,j,i) = b3c;
                cb.x3f(m,kl+1,j,i) = b3c;
              });
            }

          // restrict in 3D
          } else {
            if (v==0) {
            // restrict B1
              Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il,iu+1),
              [&](const int i) {
                int fi = (i - indcs.cis)*2 + indcs.is;
                cb.x1f(m,k,j,i) = 0.25*(b.x1f(m,fk  ,fj,fi) + b.x1f(m,fk  ,fj+1,fi)
                                      + b.x1f(m,fk+1,fj,fi) + b.x1f(m,fk+1,fj+1,fi));
              });
            } else if (v==1) {
            // restrict B2
              Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il,iu+1),
              [&](const int i) {
                int fi = (i - indcs.cis)*2 + indcs.is;
                cb.x2f(m,k,j,i) = 0.25*(b.x2f(m,fk  ,fj,fi) + b.x2f(m,fk  ,fj,fi+1)
                                      + b.x2f(m,fk+1,fj,fi) + b.x2f(m,fk+1,fj,fi+1));
              });
            } else {
            // restrict B3
              Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il,iu+1),
              [&](const int i) {
                int fi = (i - indcs.cis)*2 + indcs.is;
                cb.x3f(m,k,j,i) = 0.25*(b.x3f(m,fk,fj  ,fi) + b.x3f(m,fk,fj  ,fi+1)
                                      + b.x3f(m,fk,fj+1,fi) + b.x3f(m,fk,fj+1,fi+1));
              });
            }
          }
        });
      }
    });
  }

  // Now prolongate face-fields when neighbor at coarser level
  // Code here is based on MeshRefinement::ProlongateSharedFieldX1/2/3() and
  // MeshRefinement::ProlongateInternalField() in C++ version

  //----- 1D PROBLEM: -----
  if (pmy_pack->pmesh->one_d) {
    int js = indcs.js;
    int ks = indcs.ks;
    auto &rbuf = recv_buf;

    // Outer loop over (# of MeshBlocks)*(# of buffers)
    par_for_outer("ProlongFC-1d",DevExeSpace(), 0, 0, 0, (nmb-1), 0, (nnghbr-1),
    KOKKOS_LAMBDA(TeamMember_t member, const int m, const int n) {
      // only prolongate when neighbor exists and is at coarser level
      if ((nghbr.d_view(m,n).gid >= 0) && (nghbr.d_view(m,n).lev < mblev.d_view(m))) {
        // prolongate b.x1f (v=0)
        int il = rbuf[n].iprol[0].bis;
        int iu = rbuf[n].iprol[0].bie;
        par_for_inner(member, il, iu, [&](const int i) {
          int finei = (i - indcs.cis)*2 + indcs.is;
          // prolongate B1 at shared coarse/fine cell edges
          // equivalent to code for 1D in MeshRefinement::ProlongateSharedFieldX1()
          b.x1f(m,ks,js,finei) = cb.x1f(m,ks,js,i);
          // prolongate B1 at interior cell edges on fine mesh
          // equivalent to code for 1D in MeshRefinement::ProlongateInternalField()
          if (finei > indcs.is) {
            // oib: interior cell to left of shared edge
            b.x1f(m,ks,js,finei-1) = cb.x1f(m,ks,js,i);
          } else {
            // iib: interior cell to right of shared edge
            b.x1f(m,ks,js,finei+1) = cb.x1f(m,ks,js,i);
          }
        });

        // prolongate b.x2f (v=1)
        il = rbuf[n].iprol[1].bis;
        iu = rbuf[n].iprol[1].bie;
        par_for_inner(member, il, iu, [&](const int i) {
          int fi = (i - indcs.cis)*2 + indcs.is;
          // interpolate B2 in x1 to fine cell locations
          // equivalent to code for 1D in MeshRefinement::ProlongateSharedFieldX2()
          Real dl = cb.x2f(m,ks,js,i  ) - cb.x2f(m,ks,js,i-1);
          Real dr = cb.x2f(m,ks,js,i+1) - cb.x2f(m,ks,js,i  );
          Real dvar1 = 0.125*(SIGN(dl) + SIGN(dr))*fmin(fabs(dl), fabs(dr));
          b.x2f(m,ks,js  ,fi  ) = cb.x2f(m,ks,js,i) - dvar1;
          b.x2f(m,ks,js  ,fi+1) = cb.x2f(m,ks,js,i) + dvar1;
          b.x2f(m,ks,js+1,fi  ) = cb.x2f(m,ks,js,i) - dvar1;
          b.x2f(m,ks,js+1,fi+1) = cb.x2f(m,ks,js,i) + dvar1;
        });

        // prolongate b.x3f (v=2)
        il = rbuf[n].iprol[2].bis;
        iu = rbuf[n].iprol[2].bie;
        par_for_inner(member, il, iu, [&](const int i) {
          int fi = (i - indcs.cis)*2 + indcs.is;
          // interpolate B3 in x1 to fine cell locations
          // equivalent to code for 1D in MeshRefinement::ProlongateSharedFieldX3()
          Real dl = cb.x3f(m,ks,js,i  ) - cb.x3f(m,ks,js,i-1);
          Real dr = cb.x3f(m,ks,js,i+1) - cb.x3f(m,ks,js,i  );
          Real dvar1 = 0.125*(SIGN(dl) + SIGN(dr))*fmin(fabs(dl), fabs(dr));
          b.x3f(m,ks  ,js,fi  ) = cb.x3f(m,ks,js,i) - dvar1;
          b.x3f(m,ks  ,js,fi+1) = cb.x3f(m,ks,js,i) + dvar1;
          b.x3f(m,ks+1,js,fi  ) = cb.x3f(m,ks,js,i) - dvar1;
          b.x3f(m,ks+1,js,fi+1) = cb.x3f(m,ks,js,i) + dvar1;
        });
      }
    });

  //----- 2D PROBLEM: -----
  } else if (pmy_pack->pmesh->two_d) {
    int ks = indcs.ks;
    auto &rbuf = recv_buf;

    // Prolongate b.x1f/b.x2f/b.x3f at all shared coarse/fine cell edges
    // equivalent to code for 2D in MeshRefinement::ProlongateSharedFieldX1/2/3()

    // Outer loop over (# of MeshBlocks)*(# of buffers)*(three field components)
    {int nmnv = 3*nmb*nnghbr;
    Kokkos::TeamPolicy<> policy(DevExeSpace(), nmnv, Kokkos::AUTO);
    Kokkos::parallel_for("ProFC-2d-shared", policy, KOKKOS_LAMBDA(TeamMember_t tmember) {
      const int m = (tmember.league_rank())/(3*nnghbr);
      const int n = (tmember.league_rank() - m*(3*nnghbr))/3;
      const int v = (tmember.league_rank() - m*(3*nnghbr) - 3*n);

      // only prolongate when neighbor exists and is at coarser level
      if ((nghbr.d_view(m,n).gid >= 0) && (nghbr.d_view(m,n).lev < mblev.d_view(m))) {
        int il = rbuf[n].iprol[v].bis;
        int iu = rbuf[n].iprol[v].bie;
        int jl = rbuf[n].iprol[v].bjs;
        int ju = rbuf[n].iprol[v].bje;
        const int nj = ju - jl + 1;

        // Prolongate b.x1f (v=0) by interpolating in x2
        if (v==0) {
          // Middle loop over j
          Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nj),
          [&](const int idx) {
            int j = idx + jl;
            int fj = (j - indcs.cjs)*2 + indcs.js;  // fine j
            // inner vector loop
            Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il,iu+1),
            [&](const int i) {
              int fi = (i - indcs.cis)*2 + indcs.is;  // fine i
              Real dl = cb.x1f(m,ks,j  ,i) - cb.x1f(m,ks,j-1,i);
              Real dr = cb.x1f(m,ks,j+1,i) - cb.x1f(m,ks,j  ,i);
              Real dvar2 = 0.125*(SIGN(dl) + SIGN(dr))*fmin(fabs(dl), fabs(dr));
              b.x1f(m,ks,fj  ,fi) = cb.x1f(m,ks,j,i) - dvar2;
              b.x1f(m,ks,fj+1,fi) = cb.x1f(m,ks,j,i) + dvar2;
            });
          });

        // Prolongate b.x2f (v=1) by interpolating in x1
        } else if (v==1) {
          // Middle loop over j
          Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nj),
          [&](const int idx) {
            int j = idx + jl;
            int fj = (j - indcs.cjs)*2 + indcs.js;  // fine j
            // inner vector loop
            Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il,iu+1),
            [&](const int i) {
              int fi = (i - indcs.cis)*2 + indcs.is;  // fine i
              Real dl = cb.x2f(m,ks,j,i  ) - cb.x2f(m,ks,j,i-1);
              Real dr = cb.x2f(m,ks,j,i+1) - cb.x2f(m,ks,j,i  );
              Real dvar1 = 0.125*(SIGN(dl) + SIGN(dr))*fmin(fabs(dl), fabs(dr));
              b.x2f(m,ks,fj,fi  ) = cb.x2f(m,ks,j,i) - dvar1;
              b.x2f(m,ks,fj,fi+1) = cb.x2f(m,ks,j,i) + dvar1;
            });
          });

        // Prolongate b.x3f (v=2) by interpolating in x1/x2
        } else {
          // Middle loop over j
          Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nj),
          [&](const int idx) {
            int j = idx + jl;
            int fj = (j - indcs.cjs)*2 + indcs.js;  // fine j
            // inner vector loop
            Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il,iu+1),
            [&](const int i) {
              int fi = (i - indcs.cis)*2 + indcs.is;  // fine i
              Real dl = cb.x3f(m,ks,j,i  ) - cb.x3f(m,ks,j,i-1);
              Real dr = cb.x3f(m,ks,j,i+1) - cb.x3f(m,ks,j,i  );
              Real dvar1 = 0.125*(SIGN(dl) + SIGN(dr))*fmin(fabs(dl), fabs(dr));
              dl = cb.x3f(m,ks,j  ,i) - cb.x3f(m,ks,j-1,i);
              dr = cb.x3f(m,ks,j+1,i) - cb.x3f(m,ks,j  ,i);
              Real dvar2 = 0.125*(SIGN(dl) + SIGN(dr))*fmin(fabs(dl), fabs(dr));
              b.x3f(m,ks,fj  ,fi  ) = cb.x3f(m,ks,j,i) - dvar1 - dvar2;
              b.x3f(m,ks,fj  ,fi+1) = cb.x3f(m,ks,j,i) + dvar1 - dvar2;
              b.x3f(m,ks,fj+1,fi  ) = cb.x3f(m,ks,j,i) - dvar1 + dvar2;
              b.x3f(m,ks,fj+1,fi+1) = cb.x3f(m,ks,j,i) + dvar1 + dvar2;

              b.x3f(m,ks+1,fj  ,fi  ) = b.x3f(m,ks,fj  ,fi  );
              b.x3f(m,ks+1,fj  ,fi+1) = b.x3f(m,ks,fj  ,fi+1);
              b.x3f(m,ks+1,fj+1,fi  ) = b.x3f(m,ks,fj+1,fi  );
              b.x3f(m,ks+1,fj+1,fi+1) = b.x3f(m,ks,fj+1,fi+1);
            });
          });
        }
      }
    });
    }

    // Now prolongate b.x1f/b.x2f at interior fine cells
    // equivalent to code for 2D in MeshRefinement::ProlongateInternalField()
    // Note prolongation at shared coarse/fine cell edges must be completed first as
    // interpolation formulae use these values.

    // Outer loop over (# of MeshBlocks)*(# of buffers)
    {int nmn = nmb*nnghbr;
    Kokkos::TeamPolicy<> policy(DevExeSpace(), nmn, Kokkos::AUTO);
    Kokkos::parallel_for("ProFC-2d-int", policy, KOKKOS_LAMBDA(TeamMember_t tmember) {
      const int m = (tmember.league_rank())/(nnghbr);
      const int n = (tmember.league_rank() - m*(nnghbr));

      // only prolongate when neighbor exists and is at coarser level
      if ((nghbr.d_view(m,n).gid >= 0) && (nghbr.d_view(m,n).lev < mblev.d_view(m))) {
        // use prolongation indices of different field components for interior fine cells
        int il = rbuf[n].iprol[2].bis;
        int iu = rbuf[n].iprol[2].bie;
        int jl = rbuf[n].iprol[0].bjs;
        int ju = rbuf[n].iprol[0].bje;
        const int nj = ju - jl + 1;

        // Middle loop over j
        Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nj), [&](const int idx) {
          int j = idx + jl;
          int fj = (j - indcs.cjs)*2 + indcs.js;   // fine j

          Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il,iu+1),
          [&](const int i) {
            int fi = (i - indcs.cis)*2 + indcs.is;   // fine i

            Real tmp1 = 0.25*(b.x2f(m,ks,fj+2,fi+1) - b.x2f(m,ks,fj,  fi+1)
                            - b.x2f(m,ks,fj+2,fi  ) + b.x2f(m,ks,fj,  fi  ));
            Real tmp2 = 0.25*(b.x1f(m,ks,fj,  fi  ) - b.x1f(m,ks,fj,  fi+2)
                            - b.x1f(m,ks,fj+1,fi  ) + b.x1f(m,ks,fj+1,fi+2));
            b.x1f(m,ks,fj  ,fi+1)=0.5*(b.x1f(m,ks,fj,  fi  )+b.x1f(m,ks,fj,  fi+2)) +tmp1;
            b.x1f(m,ks,fj+1,fi+1)=0.5*(b.x1f(m,ks,fj+1,fi  )+b.x1f(m,ks,fj+1,fi+2)) +tmp1;
            b.x2f(m,ks,fj+1,fi  )=0.5*(b.x2f(m,ks,fj,  fi  )+b.x2f(m,ks,fj+2,fi  )) +tmp2;
            b.x2f(m,ks,fj+1,fi+1)=0.5*(b.x2f(m,ks,fj,  fi+1)+b.x2f(m,ks,fj+2,fi+1)) +tmp2;
          });
        });
      }
    });
    }

  //------ 3D PROBLEM: -----
  } else {
    auto &rbuf = recv_buf;

    // Prolongate b.x1f/b.x2f/b.x3f at all shared coarse/fine cell edges
    // equivalent to code for 3D in MeshRefinement::ProlongateSharedFieldX1/2/3()

    // Outer loop over (# of MeshBlocks)*(# of buffers)*(three field components)
    {int nmnv = 3*nmb*nnghbr;
    Kokkos::TeamPolicy<> policy(DevExeSpace(), nmnv, Kokkos::AUTO);
    Kokkos::parallel_for("ProFC-2d-shared", policy, KOKKOS_LAMBDA(TeamMember_t tmember) {
      const int m = (tmember.league_rank())/(3*nnghbr);
      const int n = (tmember.league_rank() - m*(3*nnghbr))/3;
      const int v = (tmember.league_rank() - m*(3*nnghbr) - 3*n);

      // only prolongate when neighbor exists and is at coarser level
      if ((nghbr.d_view(m,n).gid >= 0) && (nghbr.d_view(m,n).lev < mblev.d_view(m))) {
        int il = rbuf[n].iprol[v].bis;
        int iu = rbuf[n].iprol[v].bie;
        int jl = rbuf[n].iprol[v].bjs;
        int ju = rbuf[n].iprol[v].bje;
        int kl = rbuf[n].iprol[v].bks;
        int ku = rbuf[n].iprol[v].bke;
        const int nj = ju - jl + 1;
        const int nk = ku - kl + 1;
        const int nkj  = nk*nj;

        // Prolongate b.x1f (v=0) by interpolating in x2/x3
        if (v==0) {
          // Middle loop over k,j
          Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nkj),
          [&](const int idx) {
            int k = idx / nj;
            int j = (idx - k * nj) + jl;
            k += kl;
            // inner vector loop
            Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il,iu+1),
            [&](const int i) {
              int fi = (i - indcs.cis)*2 + indcs.is;  // fine i
              int fj = (j - indcs.cjs)*2 + indcs.js;  // fine j
              int fk = (k - indcs.cks)*2 + indcs.ks;  // fine k
              Real dl = cb.x1f(m,k,j  ,i) - cb.x1f(m,k,j-1,i);
              Real dr = cb.x1f(m,k,j+1,i) - cb.x1f(m,k,j  ,i);
              Real dvar2 = 0.125*(SIGN(dl) + SIGN(dr))*fmin(fabs(dl), fabs(dr));
              dl = cb.x1f(m,k  ,j,i) - cb.x1f(m,k-1,j,i);
              dr = cb.x1f(m,k+1,j,i) - cb.x1f(m,k  ,j,i);
              Real dvar3 = 0.125*(SIGN(dl) + SIGN(dr))*fmin(fabs(dl), fabs(dr));
              b.x1f(m,fk  ,fj  ,fi) = cb.x1f(m,k,j,i) - dvar2 - dvar3;
              b.x1f(m,fk  ,fj+1,fi) = cb.x1f(m,k,j,i) + dvar2 - dvar3;
              b.x1f(m,fk+1,fj  ,fi) = cb.x1f(m,k,j,i) - dvar2 + dvar3;
              b.x1f(m,fk+1,fj+1,fi) = cb.x1f(m,k,j,i) + dvar2 + dvar3;
            });
          });

        // Prolongate b.x2f (v=1) by interpolating in x1/x3
        } else if (v==1) {
          // Middle loop over k,j
          Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nkj),
          [&](const int idx) {
            int k = idx / nj;
            int j = (idx - k * nj) + jl;
            k += kl;
            // inner vector loop
            Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il,iu+1),
            [&](const int i) {
              int fi = (i - indcs.cis)*2 + indcs.is;  // fine i
              int fj = (j - indcs.cjs)*2 + indcs.js;  // fine j
              int fk = (k - indcs.cks)*2 + indcs.ks;  // fine k
              Real dl = cb.x2f(m,k,j,i  ) - cb.x2f(m,k,j,i-1);
              Real dr = cb.x2f(m,k,j,i+1) - cb.x2f(m,k,j,i  );
              Real dvar1 = 0.125*(SIGN(dl) + SIGN(dr))*fmin(fabs(dl), fabs(dr));
              dl = cb.x2f(m,k  ,j,i) - cb.x2f(m,k-1,j,i);
              dr = cb.x2f(m,k+1,j,i) - cb.x2f(m,k  ,j,i);
              Real dvar3 = 0.125*(SIGN(dl) + SIGN(dr))*fmin(fabs(dl), fabs(dr));
              b.x2f(m,fk  ,fj,fi  ) = cb.x2f(m,k,j,i) - dvar1 - dvar3;
              b.x2f(m,fk  ,fj,fi+1) = cb.x2f(m,k,j,i) + dvar1 - dvar3;
              b.x2f(m,fk+1,fj,fi  ) = cb.x2f(m,k,j,i) - dvar1 + dvar3;
              b.x2f(m,fk+1,fj,fi+1) = cb.x2f(m,k,j,i) + dvar1 + dvar3;
            });
          });

        // Prolongate b.x3f (v=2) by interpolating in x1/x2
        } else {
          // Middle loop over k,j
          Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nkj),
          [&](const int idx) {
            int k = idx / nj;
            int j = (idx - k * nj) + jl;
            k += kl;
            // inner vector loop
            Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il,iu+1),
            [&](const int i) {
              int fi = (i - indcs.cis)*2 + indcs.is;  // fine i
              int fj = (j - indcs.cjs)*2 + indcs.js;  // fine j
              int fk = (k - indcs.cks)*2 + indcs.ks;  // fine k
              Real dl = cb.x3f(m,k,j,i  ) - cb.x3f(m,k,j,i-1);
              Real dr = cb.x3f(m,k,j,i+1) - cb.x3f(m,k,j,i  );
              Real dvar1 = 0.125*(SIGN(dl) + SIGN(dr))*fmin(fabs(dl), fabs(dr));
              dl = cb.x3f(m,k,j  ,i) - cb.x3f(m,k,j-1,i);
              dr = cb.x3f(m,k,j+1,i) - cb.x3f(m,k,j  ,i);
              Real dvar2 = 0.125*(SIGN(dl) + SIGN(dr))*fmin(fabs(dl), fabs(dr));
              b.x3f(m,fk,fj  ,fi  ) = cb.x3f(m,k,j,i) - dvar1 - dvar2;
              b.x3f(m,fk,fj  ,fi+1) = cb.x3f(m,k,j,i) + dvar1 - dvar2;
              b.x3f(m,fk,fj+1,fi  ) = cb.x3f(m,k,j,i) - dvar1 + dvar2;
              b.x3f(m,fk,fj+1,fi+1) = cb.x3f(m,k,j,i) + dvar1 + dvar2;
            });
          });
        }
      }
    });
    }

    // Now prolongate b.x1f/b.x2f/b.x3f at interior fine cells
    // equivalent to code for 3D in MeshRefinement::ProlongateInternalField()
    // Note prolongation at shared coarse/fine cell edges must be completed first as
    // interpolation formulae use these values.

    // Outer loop over (# of MeshBlocks)*(# of buffers)
    {int nmn = nmb*nnghbr;
    Kokkos::TeamPolicy<> policy(DevExeSpace(), nmn, Kokkos::AUTO);
    Kokkos::parallel_for("ProFC-2d-int", policy, KOKKOS_LAMBDA(TeamMember_t tmember) {
      const int m = (tmember.league_rank())/(nnghbr);
      const int n = (tmember.league_rank() - m*(nnghbr));

      // only prolongate when neighbor exists and is at coarser level
      if ((nghbr.d_view(m,n).gid >= 0) && (nghbr.d_view(m,n).lev < mblev.d_view(m))) {
        // use prolongation indices of different field components for interior fine cells
        int il = rbuf[n].iprol[2].bis;
        int iu = rbuf[n].iprol[2].bie;
        int jl = rbuf[n].iprol[0].bjs;
        int ju = rbuf[n].iprol[0].bje;
        int kl = rbuf[n].iprol[1].bks;
        int ku = rbuf[n].iprol[1].bke;
        const int nj = ju - jl + 1;
        const int nk = ku - kl + 1;
        const int nkj  = nk*nj;

        // Middle loop over k,j
        Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nkj),
        [&](const int idx) {
          int k = idx / nj;
          int j = (idx - k * nj) + jl;
          k += kl;

          Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il,iu+1),
          [&](const int i) {
            int fi = (i - indcs.cis)*2 + indcs.is;   // fine i
            int fj = (j - indcs.cjs)*2 + indcs.js;   // fine j
            int fk = (k - indcs.cks)*2 + indcs.ks;  // fine k

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

            b.x1f(m,fk  ,fj  ,fi+1)=0.5*(b.x1f(m,fk  ,fj  ,fi  )+b.x1f(m,fk  ,fj  ,fi+2))
                                    + Uxx - Vxyz - Wxyz;
            b.x1f(m,fk  ,fj+1,fi+1)=0.5*(b.x1f(m,fk  ,fj+1,fi  )+b.x1f(m,fk  ,fj+1,fi+2))
                                    + Uxx - Vxyz + Wxyz;
            b.x1f(m,fk+1,fj  ,fi+1)=0.5*(b.x1f(m,fk+1,fj  ,fi  )+b.x1f(m,fk+1,fj  ,fi+2))
                                    + Uxx + Vxyz - Wxyz;
            b.x1f(m,fk+1,fj+1,fi+1)=0.5*(b.x1f(m,fk+1,fj+1,fi  )+b.x1f(m,fk+1,fj+1,fi+2))
                                    + Uxx + Vxyz + Wxyz;
            b.x2f(m,fk  ,fj+1,fi  )=0.5*(b.x2f(m,fk  ,fj  ,fi  )+b.x2f(m,fk  ,fj+2,fi  ))
                                    + Vyy - Uxyz - Wxyz;
            b.x2f(m,fk  ,fj+1,fi+1)=0.5*(b.x2f(m,fk  ,fj  ,fi+1)+b.x2f(m,fk  ,fj+2,fi+1))
                                    + Vyy - Uxyz + Wxyz;
            b.x2f(m,fk+1,fj+1,fi  )=0.5*(b.x2f(m,fk+1,fj  ,fi  )+b.x2f(m,fk+1,fj+2,fi  ))
                                    + Vyy + Uxyz - Wxyz;
            b.x2f(m,fk+1,fj+1,fi+1)=0.5*(b.x2f(m,fk+1,fj  ,fi+1)+b.x2f(m,fk+1,fj+2,fi+1))
                                    + Vyy + Uxyz + Wxyz;
            b.x3f(m,fk+1,fj  ,fi  )=0.5*(b.x3f(m,fk+2,fj  ,fi  )+b.x3f(m,fk  ,fj  ,fi  ))
                                    + Wzz - Uxyz - Vxyz;
            b.x3f(m,fk+1,fj  ,fi+1)=0.5*(b.x3f(m,fk+2,fj  ,fi+1)+b.x3f(m,fk  ,fj  ,fi+1))
                                    + Wzz - Uxyz + Vxyz;
            b.x3f(m,fk+1,fj+1,fi  )=0.5*(b.x3f(m,fk+2,fj+1,fi  )+b.x3f(m,fk  ,fj+1,fi  ))
                                    + Wzz + Uxyz - Vxyz;
            b.x3f(m,fk+1,fj+1,fi+1)=0.5*(b.x3f(m,fk+2,fj+1,fi+1)+b.x3f(m,fk  ,fj+1,fi+1))
                                    + Wzz + Uxyz + Vxyz;
          });
        });
      }
    });
    }
  }

  return;
}
