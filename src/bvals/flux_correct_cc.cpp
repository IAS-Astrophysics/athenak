//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file flux_correction_cc.cpp
//! \brief functions to pack/send and recv/unpack fluxes for cell-centered variables at
//! fine/coarse boundaries for the flux correction step.

#include <cstdlib>
#include <iostream>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "bvals.hpp"

//----------------------------------------------------------------------------------------
//! \fn void BoundaryValuesCC::PackAndSendFlux()
//! \brief Pack restricted fluxes of cell-centered variables at fine/coarse boundaries
//! into boundary buffers and send to neighbors for flux-correction step.  These fluxes
//! (e.g. for the conserved hydro variables) live at cell faces.
//!
//! This routine packs ALL the buffers on ALL the faces simultaneously for ALL the
//! MeshBlocks. Buffer data are then sent (via MPI) or copied directly for periodic or
//! block boundaries.

TaskStatus BoundaryValuesCC::PackAndSendFluxCC(DvceFaceFld5D<Real> &flx) {
  // create local references for variables in kernel
  int nmb = pmy_pack->pmb->nmb;
  int nnghbr = pmy_pack->pmb->nnghbr;
  int nvar = flx.x1f.extent_int(1);  // TODO(@user): 2nd idx from L of in arr must be NVAR

  auto &cis = pmy_pack->pmesh->mb_indcs.cis;
  auto &cjs = pmy_pack->pmesh->mb_indcs.cjs;
  auto &cks = pmy_pack->pmesh->mb_indcs.cks;

  int &my_rank = global_variable::my_rank;
  auto &nghbr = pmy_pack->pmb->nghbr;
  auto &mbgid = pmy_pack->pmb->mb_gid;
  auto &mblev = pmy_pack->pmb->mb_lev;
  auto &sbuf = send_buf;
  auto &rbuf = recv_buf;
  auto &one_d = pmy_pack->pmesh->one_d;
  auto &two_d = pmy_pack->pmesh->two_d;

  // Outer loop over (# of MeshBlocks)*(# of neighbors)*(# of variables)
  Kokkos::TeamPolicy<> policy(DevExeSpace(), (nmb*nnghbr*nvar), Kokkos::AUTO);
  Kokkos::parallel_for("RecvBuff", policy, KOKKOS_LAMBDA(TeamMember_t tmember) {
    const int m = (tmember.league_rank())/(nnghbr*nvar);
    const int n = (tmember.league_rank() - m*(nnghbr*nvar))/nvar;
    const int v = (tmember.league_rank() - m*(nnghbr*nvar) - n*nvar);

    // Note send buffer flux indices are for the coarse mesh
    int il = sbuf[n].iflux[0].bis;
    int iu = sbuf[n].iflux[0].bie;
    int jl = sbuf[n].iflux[0].bjs;
    int ju = sbuf[n].iflux[0].bje;
    int kl = sbuf[n].iflux[0].bks;
    int ku = sbuf[n].iflux[0].bke;
    const int ni = iu - il + 1;
    const int nj = ju - jl + 1;
    const int nk = ku - kl + 1;
    const int nkj  = nk*nj;

    // indices of recv'ing (destination) MB and buffer: MB IDs are stored sequentially
    // in MeshBlockPacks, so array index equals (target_id - first_id)
    int dm = nghbr.d_view(m,n).gid - mbgid.d_view(0);
    int dn = nghbr.d_view(m,n).dest;

    // only pack buffers when neighbor is at coarser level
    if ((nghbr.d_view(m,n).gid >=0) && (nghbr.d_view(m,n).lev < mblev.d_view(m))) {
      // x1faces
      if (n<8) {
        Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nkj), [&](const int idx) {
          int k = idx / nj;
          int j = (idx - k * nj) + jl;
          k += kl;
          int fj = 2*j - cjs;
          int fk = 2*k - cks;
          Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il,iu+1),
          [&](const int i) {
            int fi = 2*i - cis;
            Real rflx;
            if (one_d) {
              rflx = flx.x1f(m,v,0,0,fi);
            } else if (two_d) {
              rflx = 0.5*(flx.x1f(m,v,0,fj,fi) + flx.x1f(m,v,0,fj+1,fi));
            } else {
              rflx = 0.25*(flx.x1f(m,v,fk  ,fj,fi) + flx.x1f(m,v,fk  ,fj+1,fi) +
                           flx.x1f(m,v,fk+1,fj,fi) + flx.x1f(m,v,fk+1,fj+1,fi));
            }
            // copy directly into recv buffer if MeshBlocks on same rank
            if (nghbr.d_view(m,n).rank == my_rank) {
              rbuf[dn].flux(dm, (i-il + ni*(j-jl + nj*(k-kl + nk*v))) ) = rflx;
            // else copy into send buffer for MPI communication below
            } else {
              sbuf[n].flux(m, (i-il + ni*(j-jl + nj*(k-kl + nk*v))) ) = rflx;
            }
          });
        });

      // x2faces
      } else if (n<16) {
        Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nkj), [&](const int idx) {
          int k = idx / nj;
          int j = (idx - k * nj) + jl;
          k += kl;
          int fj = 2*j - cjs;
          int fk = 2*k - cks;
          Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il,iu+1),
          [&](const int i) {
            int fi = 2*i - cis;
            Real rflx;
            if (two_d) {
              rflx = 0.5*(flx.x2f(m,v,0,fj,fi) + flx.x2f(m,v,0,fj,fi+1));
            } else {
              rflx = 0.25*(flx.x2f(m,v,fk  ,fj,fi) + flx.x2f(m,v,fk  ,fj,fi+1) +
                           flx.x2f(m,v,fk+1,fj,fi) + flx.x2f(m,v,fk+1,fj,fi+1));
            }
            // copy directly into recv buffer if MeshBlocks on same rank
            if (nghbr.d_view(m,n).rank == my_rank) {
              rbuf[dn].flux(dm, (i-il + ni*(j-jl + nj*(k-kl + nk*v))) ) = rflx;
            // else copy into send buffer for MPI communication below
            } else {
              sbuf[n].flux(m, (i-il + ni*(j-jl + nj*(k-kl + nk*v))) ) = rflx;
            }
          });
        });

      // x3faces
      } else if ((n>=24) && (n<32)) {
        Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nkj), [&](const int idx) {
          int k = idx / nj;
          int j = (idx - k * nj) + jl;
          k += kl;
          int fj = 2*j - cjs;
          int fk = 2*k - cks;
          Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il,iu+1),
          [&](const int i) {
            int fi = 2*i - cis;
            Real rflx = 0.25*(flx.x3f(m,v,fk,fj  ,fi) + flx.x3f(m,v,fk,fj  ,fi+1) +
                              flx.x3f(m,v,fk,fj+1,fi) + flx.x3f(m,v,fk,fj+1,fi+1));
            // copy directly into recv buffer if MeshBlocks on same rank
            if (nghbr.d_view(m,n).rank == my_rank) {
              rbuf[dn].flux(dm, (i-il + ni*(j-jl + nj*(k-kl + nk*v))) ) = rflx;
            // else copy into send buffer for MPI communication below
            } else {
              sbuf[n].flux(m, (i-il + ni*(j-jl + nj*(k-kl + nk*v))) ) = rflx;
            }
          });
        });
      }
    }  // end if-neighbor-exists block
  });  // end par_for_outer

  // Send boundary buffer to neighboring MeshBlocks using MPI
  // Sends only occur to neighbors on faces at a COARSER level

  bool no_errors=true;
  for (int m=0; m<nmb; ++m) {
    for (int n=0; n<nnghbr; ++n) {
      if ( (nghbr.h_view(m,n).gid >=0) && (nghbr.h_view(m,n).lev < mblev.h_view(m)) &&
           ((n<16) || ((n>=24) && (n<32))) ) {
        // index and rank of destination Neighbor
        int dn = nghbr.h_view(m,n).dest;
        int drank = nghbr.h_view(m,n).rank;

        // if MeshBlocks are on same rank, data already copied into receive buffer above
        // So simply set communication status tag as received.
        if (drank == my_rank) {
          int dm = nghbr.h_view(m,n).gid - pmy_pack->gids;
          rbuf[dn].flux_stat[dm] = BoundaryCommStatus::received;
#if MPI_PARALLEL_ENABLED
        // Send boundary data using MPI
        } else {
          // create tag using local ID and buffer index of *receiving* MeshBlock
          int lid = nghbr.h_view(m,n).gid - pmy_pack->pmesh->gidslist[drank];
          int tag = CreateMPITag(lid, dn);

          // get ptr to send buffer for fluxes
          int data_size = nvar*(send_buf[n].iflux_ndat);
          void* send_ptr = &(send_buf[n].flux(m,0));

          int ierr = MPI_Isend(send_ptr, data_size, MPI_ATHENA_REAL, drank, tag,
                               flux_comm, &(send_buf[n].flux_req[m]));
          if (ierr != MPI_SUCCESS) {no_errors=false;}
#endif
        }
      }
    }
  }
  if (no_errors) return TaskStatus::complete;

  return TaskStatus::fail;
}

//----------------------------------------------------------------------------------------
// \!fn void RecvBuffers()
// \brief Unpack boundary buffers

TaskStatus BoundaryValuesCC::RecvAndUnpackFluxCC(DvceFaceFld5D<Real> &flx) {
  // create local references for variables in kernel
  int nmb = pmy_pack->pmb->nmb;
  int nnghbr = pmy_pack->pmb->nnghbr;

  bool bflag = false;
  auto &nghbr = pmy_pack->pmb->nghbr;
  auto &rbuf = recv_buf;
  auto &mblev = pmy_pack->pmb->mb_lev;

#if MPI_PARALLEL_ENABLED
  // probe MPI communications.  This is a bit of black magic that seems to promote
  // communications to top of stack and gets them to complete more quickly
  int test;
  int ierr = MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, flux_comm, &test, MPI_STATUS_IGNORE);
  if (ierr != MPI_SUCCESS) {return TaskStatus::incomplete;}
#endif

  //----- STEP 1: check that recv boundary buffer communications have all completed
  // receives only occur for neighbors on faces at a FINER level

  for (int m=0; m<nmb; ++m) {
    for (int n=0; n<nnghbr; ++n) {
      if ( (nghbr.h_view(m,n).gid >=0) && (nghbr.h_view(m,n).lev > mblev.h_view(m)) &&
           ((n<16) || ((n>=24) && (n<32))) ) {
        if (nghbr.h_view(m,n).rank == global_variable::my_rank) {
          if (rbuf[n].flux_stat[m] == BoundaryCommStatus::waiting) {bflag = true;}
#if MPI_PARALLEL_ENABLED
        } else {
          MPI_Test(&(rbuf[n].flux_req[m]), &test, MPI_STATUS_IGNORE);
          if (static_cast<bool>(test)) {
            rbuf[n].flux_stat[m] = BoundaryCommStatus::received;
          } else {
            bflag = true;
          }
#endif
        }
      }
    }
  }

  // exit if recv boundary buffer communications have not completed
  if (bflag) {return TaskStatus::incomplete;}

  //----- STEP 2: buffers have all completed, so unpack

  int nvar = flx.x1f.extent_int(1); // TODO(@user): 2nd idx from L of in arr must be NVAR

  // Outer loop over (# of MeshBlocks)*(# of neighbors)*(# of variables)
  Kokkos::TeamPolicy<> policy(DevExeSpace(), (nmb*nnghbr*nvar), Kokkos::AUTO);
  Kokkos::parallel_for("RecvBuff", policy, KOKKOS_LAMBDA(TeamMember_t tmember) {
    const int m = (tmember.league_rank())/(nnghbr*nvar);
    const int n = (tmember.league_rank() - m*(nnghbr*nvar))/nvar;
    const int v = (tmember.league_rank() - m*(nnghbr*nvar) - n*nvar);

    // Recv buffer flux indices are for the regular mesh
    int il = rbuf[n].iflux[0].bis;
    int iu = rbuf[n].iflux[0].bie;
    int jl = rbuf[n].iflux[0].bjs;
    int ju = rbuf[n].iflux[0].bje;
    int kl = rbuf[n].iflux[0].bks;
    int ku = rbuf[n].iflux[0].bke;
    const int ni = iu - il + 1;
    const int nj = ju - jl + 1;
    const int nk = ku - kl + 1;
    const int nkj  = nk*nj;

    // only unpack buffers for faces when neighbor is at finer level
    if ((nghbr.d_view(m,n).gid >=0) && (nghbr.d_view(m,n).lev > mblev.d_view(m))) {
      //x1 faces
      if (n<8) {
        Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nkj), [&](const int idx) {
          int k = idx / nj;
          int j = (idx - k * nj) + jl;
          k += kl;
          Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il,iu+1),
          [&](const int i) {
            flx.x1f(m,v,k,j,i) = rbuf[n].flux(m,(i-il + ni*(j-jl + nj*(k-kl + nk*v))));
          });
        });
      // x2faces
      } else if (n<16) {
        Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nkj), [&](const int idx) {
          int k = idx / nj;
          int j = (idx - k * nj) + jl;
          k += kl;
          Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il,iu+1),
          [&](const int i) {
            flx.x2f(m,v,k,j,i) = rbuf[n].flux(m,(i-il + ni*(j-jl + nj*(k-kl + nk*v))));
          });
        });
      // x3faces
      } else if ((n>=24) && (n<32)) {
        Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nkj), [&](const int idx) {
          int k = idx / nj;
          int j = (idx - k * nj) + jl;
          k += kl;
          Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il,iu+1),
          [&](const int i) {
            flx.x3f(m,v,k,j,i) = rbuf[n].flux(m,(i-il + ni*(j-jl + nj*(k-kl + nk*v))));
          });
        });
      }
    }  // end if-neighbor-exists block
  });  // end par_for_outer

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  void BoundaryValuesCC::InitRecvFlux
//! \brief Posts non-blocking receives (with MPI), and initialize all boundary receive
//! status flags to waiting (with or without MPI) for boundary communications of fluxes.

TaskStatus BoundaryValuesCC::InitFluxRecv(const int nvar) {
  int &nmb = pmy_pack->nmb_thispack;
  int &nnghbr = pmy_pack->pmb->nnghbr;
  auto &nghbr = pmy_pack->pmb->nghbr;
  auto &mblev = pmy_pack->pmb->mb_lev;

  // Initialize communications of fluxes
  bool no_errors=true;
  for (int m=0; m<nmb; ++m) {
    for (int n=0; n<nnghbr; ++n) {
      // only post receives for neighbors on faces at FINER level
      // this is the only thing different from BoundaryValuesFC::InitRecvFlux()
      if ( (nghbr.h_view(m,n).gid >=0) && (nghbr.h_view(m,n).lev > mblev.h_view(m)) &&
           ((n<16) || ((n>=24) && (n<32))) ) {
#if MPI_PARALLEL_ENABLED
        // rank of destination buffer
        int drank = nghbr.h_view(m,n).rank;

        // post non-blocking receive if neighboring MeshBlock on a different rank
        if (drank != global_variable::my_rank) {
          // create tag using local ID and buffer index of *receiving* MeshBlock
          int tag = CreateMPITag(m, n);

          // get ptr to recv buffer when neighbor is at coarser/same/fine level
          int data_size = nvar*(recv_buf[n].iflux_ndat);
          void* recv_ptr = &(recv_buf[n].flux(m,0));

          // Post non-blocking receive for this buffer on this MeshBlock
          int ierr = MPI_Irecv(recv_ptr, data_size, MPI_ATHENA_REAL, drank, tag,
                               flux_comm, &(recv_buf[n].flux_req[m]));
          if (ierr != MPI_SUCCESS) {no_errors=false;}
        }
#endif
        // initialize boundary receive status flags
        recv_buf[n].flux_stat[m] = BoundaryCommStatus::waiting;
      }
    }
  }
  if (no_errors) return TaskStatus::complete;

  return TaskStatus::fail;
}

//----------------------------------------------------------------------------------------
//! \fn  void BoundaryValuesCC::ClearFluxRecv
//  \brief Waits for all MPI receives associated with boundary communcations for fluxes
//  to complete before allowing execution to continue

TaskStatus BoundaryValuesCC::ClearFluxRecv() {
  bool no_errors=true;
#if MPI_PARALLEL_ENABLED
  int &nmb = pmy_pack->nmb_thispack;
  int &nnghbr = pmy_pack->pmb->nnghbr;
  auto &nghbr = pmy_pack->pmb->nghbr;

  // wait for all non-blocking receives for fluxes to finish before continuing
  for (int m=0; m<nmb; ++m) {
    for (int n=0; n<nnghbr; ++n) {
      if ( (nghbr.h_view(m,n).gid >= 0) &&
           (nghbr.h_view(m,n).rank != global_variable::my_rank) &&
           (recv_buf[n].flux_req[m] != MPI_REQUEST_NULL) ) {
        int ierr = MPI_Wait(&(recv_buf[n].flux_req[m]), MPI_STATUS_IGNORE);
        if (ierr != MPI_SUCCESS) {no_errors=false;}
      }
    }
  }
#endif
  if (no_errors) return TaskStatus::complete;

  return TaskStatus::fail;
}

//----------------------------------------------------------------------------------------
//! \fn  void BoundaryValuesCC::ClearFluxSend
//  \brief Waits for all MPI sends associated with boundary communcations for fluxes to
//   complete before allowing execution to continue

TaskStatus BoundaryValuesCC::ClearFluxSend() {
  bool no_errors=true;
#if MPI_PARALLEL_ENABLED
  int &nmb = pmy_pack->nmb_thispack;
  int &nnghbr = pmy_pack->pmb->nnghbr;
  auto &nghbr = pmy_pack->pmb->nghbr;

  // wait for all non-blocking sends for fluxes to finish before continuing
  for (int m=0; m<nmb; ++m) {
    for (int n=0; n<nnghbr; ++n) {
      if ( (nghbr.h_view(m,n).gid >= 0) &&
           (nghbr.h_view(m,n).rank != global_variable::my_rank) &&
           (send_buf[n].flux_req[m] != MPI_REQUEST_NULL) ) {
        int ierr = MPI_Wait(&(send_buf[n].flux_req[m]), MPI_STATUS_IGNORE);
        if (ierr != MPI_SUCCESS) {no_errors=false;}
      }
    }
  }
#endif
  if (no_errors) return TaskStatus::complete;

  return TaskStatus::fail;
}
