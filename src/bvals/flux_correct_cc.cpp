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
//! \fn void MeshBoundaryValuesCC::PackAndSendFlux()
//! \brief Pack restricted fluxes of cell-centered variables at fine/coarse boundaries
//! into boundary buffers and send to neighbors for flux-correction step.  These fluxes
//! (e.g. for the conserved hydro variables) live at cell faces.
//!
//! This routine packs ALL the buffers on ALL the faces simultaneously for ALL the
//! MeshBlocks. Buffer data are then sent (via MPI) or copied directly for periodic or
//! block boundaries.

TaskStatus MeshBoundaryValuesCC::PackAndSendFluxCC(DvceFaceFld5D<Real> &flx) {
  // create local references for variables in kernel
  int nmb = pmy_pack->nmb_thispack;
  int nnghbr = pmy_pack->pmb->nnghbr;
  int nvar = flx.x1f.extent_int(1);  // TODO(@user): 2nd idx from L of in arr must be NVAR

  auto &cis = pmy_pack->pmesh->mb_indcs.cis;
  auto &cjs = pmy_pack->pmesh->mb_indcs.cjs;
  auto &cks = pmy_pack->pmesh->mb_indcs.cks;

  int my_rank = global_variable::my_rank;
  auto &nghbr = pmy_pack->pmb->nghbr;
  auto &mbgid = pmy_pack->pmb->mb_gid;
  auto &mblev = pmy_pack->pmb->mb_lev;
  auto &sbuf = sendbuf;
  auto &rbuf = recvbuf;
  auto &one_d = pmy_pack->pmesh->one_d;
  auto &two_d = pmy_pack->pmesh->two_d;

  // Outer loop over (# of MeshBlocks)*(# of neighbors)*(# of variables)
  Kokkos::TeamPolicy<> policy(DevExeSpace(), (nmb*nnghbr*nvar), Kokkos::AUTO);
  Kokkos::parallel_for("RecvBuff", policy, KOKKOS_LAMBDA(TeamMember_t tmember) {
    const int m = (tmember.league_rank())/(nnghbr*nvar);
    const int n = (tmember.league_rank() - m*(nnghbr*nvar))/nvar;
    const int v = (tmember.league_rank() - m*(nnghbr*nvar) - n*nvar);

    // Note send buffer flux indices are for the coarse mesh
    int il = sbuf[n].iflux_coar[0].bis;
    int iu = sbuf[n].iflux_coar[0].bie;
    int jl = sbuf[n].iflux_coar[0].bjs;
    int ju = sbuf[n].iflux_coar[0].bje;
    int kl = sbuf[n].iflux_coar[0].bks;
    int ku = sbuf[n].iflux_coar[0].bke;
    const int ni = iu - il + 1;
    const int nj = ju - jl + 1;
    const int nk = ku - kl + 1;
    const int nji  = nj*ni;
    const int nkj  = nk*nj;
    const int nki  = nk*ni;

    // indices of recv'ing (destination) MB and buffer: MB IDs are stored sequentially
    // in MeshBlockPacks, so array index equals (target_id - first_id)
    int dm = nghbr.d_view(m,n).gid - mbgid.d_view(0);
    int dn = nghbr.d_view(m,n).dest;

    // only pack buffers when neighbor is at coarser level
    if ((nghbr.d_view(m,n).gid >=0) && (nghbr.d_view(m,n).lev < mblev.d_view(m))) {
      // x1faces
      if (n<8) {
        // i-index is fixed for flux correction on x1faces
        int fi = 2*il - cis;
        Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nkj), [&](const int idx) {
          int k = idx / nj;
          int j = (idx - k * nj) + jl;
          k += kl;
          int fj = 2*j - cjs;
          int fk = 2*k - cks;
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
            rbuf[dn].flux(dm, (j-jl + nj*(k-kl + nk*v)) ) = rflx;
          // else copy into send buffer for MPI communication below
          } else {
            sbuf[n].flux(m, (j-jl + nj*(k-kl + nk*v)) ) = rflx;
          }
        });

      // x2faces
      } else if (n<16) {
        // j-index is fixed for flux correction on x2faces
        int fj = 2*jl - cjs;
        Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nki), [&](const int idx) {
          int k = idx / ni;
          int i = (idx - k * ni) + il;
          k += kl;
          int fi = 2*i - cis;
          int fk = 2*k - cks;
          Real rflx;
          if (two_d) {
            rflx = 0.5*(flx.x2f(m,v,0,fj,fi) + flx.x2f(m,v,0,fj,fi+1));
          } else {
            rflx = 0.25*(flx.x2f(m,v,fk  ,fj,fi) + flx.x2f(m,v,fk  ,fj,fi+1) +
                         flx.x2f(m,v,fk+1,fj,fi) + flx.x2f(m,v,fk+1,fj,fi+1));
          }
          // copy directly into recv buffer if MeshBlocks on same rank
          if (nghbr.d_view(m,n).rank == my_rank) {
            rbuf[dn].flux(dm, (i-il + ni*(k-kl + nk*v)) ) = rflx;
          // else copy into send buffer for MPI communication below
          } else {
            sbuf[n].flux(m, (i-il + ni*(k-kl + nk*v)) ) = rflx;
          }
        });

      // x3faces
      } else if ((n>=24) && (n<32)) {
        // k-index is fixed for flux correction on x3faces
        int fk = 2*kl - cks;
        Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nji), [&](const int idx) {
          int j = idx / ni;
          int i = (idx - j * ni) + il;
          j += jl;
          int fi = 2*i - cis;
          int fj = 2*j - cjs;
          Real rflx = 0.25*(flx.x3f(m,v,fk,fj  ,fi) + flx.x3f(m,v,fk,fj  ,fi+1) +
                            flx.x3f(m,v,fk,fj+1,fi) + flx.x3f(m,v,fk,fj+1,fi+1));
          // copy directly into recv buffer if MeshBlocks on same rank
          if (nghbr.d_view(m,n).rank == my_rank) {
            rbuf[dn].flux(dm, (i-il + ni*(j-jl + nj*v)) ) = rflx;
          // else copy into send buffer for MPI communication below
          } else {
            sbuf[n].flux(m, (i-il + ni*(j-jl + nj*v)) ) = rflx;
          }
        });
      }
    }  // end if-neighbor-exists block
    tmember.team_barrier();
  });  // end par_for_outer

#if MPI_PARALLEL_ENABLED
  // Send boundary buffer to neighboring MeshBlocks using MPI
  // Sends only occur to neighbors on FACES at a COARSER level
  Kokkos::fence();
  bool no_errors=true;
  for (int m=0; m<nmb; ++m) {
    for (int n=0; n<nnghbr; ++n) {
      if ( (nghbr.h_view(m,n).gid >=0) &&
           (nghbr.h_view(m,n).lev < mblev.h_view(m)) &&
           ((n<16) || ((n>=24) && (n<32))) ) {
        // index and rank of destination Neighbor
        int dn = nghbr.h_view(m,n).dest;
        int drank = nghbr.h_view(m,n).rank;

        if (drank != my_rank) {
          // create tag using local ID and buffer index of *receiving* MeshBlock
          int lid = nghbr.h_view(m,n).gid - pmy_pack->pmesh->gids_eachrank[drank];
          int tag = CreateBvals_MPI_Tag(lid, dn);

          // get ptr to send buffer for fluxes
          int data_size = nvar*(sendbuf[n].iflxc_ndat);
          auto send_ptr = Kokkos::subview(sendbuf[n].flux, m, Kokkos::ALL);

          int ierr = MPI_Isend(send_ptr.data(), data_size, MPI_ATHENA_REAL, drank, tag,
                               comm_flux, &(sendbuf[n].flux_req[m]));
          if (ierr != MPI_SUCCESS) {no_errors=false;}
        }
      }
    }
  }
  // Quit if MPI error detected
  if (!(no_errors)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
       << std::endl << "MPI error in posting sends" << std::endl;
    std::exit(EXIT_FAILURE);
  }
#endif
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn void RecvBuffers()
//! \brief Unpack boundary buffers for flux correction of CC variables.

TaskStatus MeshBoundaryValuesCC::RecvAndUnpackFluxCC(DvceFaceFld5D<Real> &flx) {
  // create local references for variables in kernel
  int nmb = pmy_pack->nmb_thispack;
  int nnghbr = pmy_pack->pmb->nnghbr;
  auto &nghbr = pmy_pack->pmb->nghbr;
  auto &mblev = pmy_pack->pmb->mb_lev;
  auto &rbuf = recvbuf;
#if MPI_PARALLEL_ENABLED
  //----- STEP 1: check that recv boundary buffer communications have all completed
  // receives only occur for neighbors on faces at a FINER level

  bool bflag = false;
  bool no_errors=true;
  for (int m=0; m<nmb; ++m) {
    for (int n=0; n<nnghbr; ++n) {
      if ( (nghbr.h_view(m,n).gid >=0) &&
           (nghbr.h_view(m,n).lev > mblev.h_view(m)) &&
           ((n<16) || ((n>=24) && (n<32))) ) {
        if (nghbr.h_view(m,n).rank != global_variable::my_rank) {
          int test;
          int ierr = MPI_Test(&(rbuf[n].flux_req[m]), &test, MPI_STATUS_IGNORE);
          if (ierr != MPI_SUCCESS) {no_errors=false;}
          if (!(static_cast<bool>(test))) {
            bflag = true;
          }
        }
      }
    }
  }
  // Quit if MPI error detected
  if (!(no_errors)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "MPI error in testing non-blocking receives"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  // exit if recv boundary buffer communications have not completed
  if (bflag) {return TaskStatus::incomplete;}
#endif

  //----- STEP 2: buffers have all completed, so unpack

  int nvar = flx.x1f.extent_int(1); // TODO(@user): 2nd idx from L of in arr must be NVAR

  // Outer loop over (# of MeshBlocks)*(# of neighbors)*(# of variables)
  Kokkos::TeamPolicy<> policy(DevExeSpace(), (nmb*nnghbr*nvar), Kokkos::AUTO);
  Kokkos::parallel_for("RecvBuff", policy, KOKKOS_LAMBDA(TeamMember_t tmember) {
    const int m = (tmember.league_rank())/(nnghbr*nvar);
    const int n = (tmember.league_rank() - m*(nnghbr*nvar))/nvar;
    const int v = (tmember.league_rank() - m*(nnghbr*nvar) - n*nvar);

    // Recv buffer flux indices are for the regular mesh
    int il = rbuf[n].iflux_coar[0].bis;
    int iu = rbuf[n].iflux_coar[0].bie;
    int jl = rbuf[n].iflux_coar[0].bjs;
    int ju = rbuf[n].iflux_coar[0].bje;
    int kl = rbuf[n].iflux_coar[0].bks;
    int ku = rbuf[n].iflux_coar[0].bke;
    const int ni = iu - il + 1;
    const int nj = ju - jl + 1;
    const int nk = ku - kl + 1;
    const int nji  = nj*ni;
    const int nkj  = nk*nj;
    const int nki  = nk*ni;

    // only unpack buffers for faces when neighbor is at finer level
    if ((nghbr.d_view(m,n).gid >=0) && (nghbr.d_view(m,n).lev > mblev.d_view(m))) {
      //x1 faces
      if (n<8) {
        Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nkj), [&](const int idx) {
          int k = idx / nj;
          int j = (idx - k * nj) + jl;
          k += kl;
          flx.x1f(m,v,k,j,il) = rbuf[n].flux(m,(j-jl + nj*(k-kl + nk*v)));
        });
      // x2faces
      } else if (n<16) {
        Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nki), [&](const int idx) {
          int k = idx / ni;
          int i = (idx - k * ni) + il;
          k += kl;
          flx.x2f(m,v,k,jl,i) = rbuf[n].flux(m,(i-il + ni*(k-kl + nk*v)));
        });
      // x3faces
      } else if ((n>=24) && (n<32)) {
        Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nji), [&](const int idx) {
          int j = idx / ni;
          int i = (idx - j * ni) + il;
          j += jl;
          flx.x3f(m,v,kl,j,i) = rbuf[n].flux(m,(i-il + ni*(j-jl + nj*v)));
        });
      }
    }  // end if-neighbor-exists block
    tmember.team_barrier();
  });  // end par_for_outer

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  void BoundaryValuesCC::InitRecvFlux
//! \brief Posts non-blocking receives (with MPI) for boundary communication of fluxes of
//! cell-centered variables, which are communicated at FACES of MeshBlocks at the SAME
//! levels.  This is different than for fluxes of face-centered vars.

TaskStatus MeshBoundaryValuesCC::InitFluxRecv(const int nvars) {
#if MPI_PARALLEL_ENABLED
  int &nmb = pmy_pack->nmb_thispack;
  int &nnghbr = pmy_pack->pmb->nnghbr;
  auto &nghbr = pmy_pack->pmb->nghbr;

  // Initialize communications of fluxes
  bool no_errors=true;
  for (int m=0; m<nmb; ++m) {
    for (int n=0; n<nnghbr; ++n) {
      // only post receives for neighbors on FACES at FINER level
      // this is the only thing different from BoundaryValuesFC::InitRecvFlux()
      if ( (nghbr.h_view(m,n).gid >=0) &&
           (nghbr.h_view(m,n).lev > pmy_pack->pmb->mb_lev.h_view(m)) &&
           ((n<16) || ((n>=24) && (n<32))) ) {
        // rank of destination buffer
        int drank = nghbr.h_view(m,n).rank;

        // post non-blocking receive if neighboring MeshBlock on a different rank
        if (drank != global_variable::my_rank) {
          // create tag using local ID and buffer index of *receiving* MeshBlock
          int tag = CreateBvals_MPI_Tag(m, n);

          // calculate amount of data to be passed, get pointer to variables
          int data_size = nvars*(recvbuf[n].iflxc_ndat);
          auto recv_ptr = Kokkos::subview(recvbuf[n].flux, m, Kokkos::ALL);

          // Post non-blocking receive for this buffer on this MeshBlock
          int ierr = MPI_Irecv(recv_ptr.data(), data_size, MPI_ATHENA_REAL, drank, tag,
                               comm_flux, &(recvbuf[n].flux_req[m]));
          if (ierr != MPI_SUCCESS) {no_errors=false;}
        }
      }
    }
  }
  // Quit if MPI error detected
  if (!(no_errors)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
       << std::endl << "MPI error in posting non-blocking receives" << std::endl;
    std::exit(EXIT_FAILURE);
  }
#endif
  return TaskStatus::complete;
}
