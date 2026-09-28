//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file bvals_cc.cpp
//! \brief functions to pack/send and recv/unpack boundary values for cell-centered (CC)
//! Mesh variables.
//! Prolongation of CC variables  occurs in ProlongateCC() function called from task list

#include <cstdlib>
#include <iostream>
#include <algorithm>
#include <utility>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "bvals.hpp"

//----------------------------------------------------------------------------------------
// BValCC constructor:

MeshBoundaryValuesCC::MeshBoundaryValuesCC(MeshBlockPack *pp, ParameterInput *pin,
                                           bool z4c) :
  MeshBoundaryValues(pp, pin, z4c) {
}

//----------------------------------------------------------------------------------------
//! \fn void MeshBoundaryValuesCC::PackAndSendCC()
//! \brief Pack cell-centered variables into boundary buffers and send to neighbors.
//!
//! This routine packs ALL the buffers on ALL the faces, edges, and corners simultaneously
//! for ALL the MeshBlocks. This reduces the number of kernel launches when there are a
//! large number of MeshBlocks per MPI rank. Buffer data are then sent (via MPI) or copied
//! directly for periodic or block boundaries.
//!
//! Input arrays must be 5D Kokkos View dimensioned (nmb, nvar, nx3, nx2, nx1)
//! 5D Kokkos View of coarsened (restricted) array data also required with SMR/AMR

TaskStatus MeshBoundaryValuesCC::PackAndSendCC(DvceArray5D<Real> &a,
                                               DvceArray5D<Real> &ca) {
  // create local references for variables in kernel
  int nmb = pmy_pack->nmb_thispack;
  int nnghbr = pmy_pack->pmb->nnghbr;
  int nvar = a.extent_int(1);  // TODO(@user): 2nd index from L of in array must be NVAR

  {int my_rank = global_variable::my_rank;
  auto &nghbr = pmy_pack->pmb->nghbr;
  auto &mbgid = pmy_pack->pmb->mb_gid;
  auto &mblev = pmy_pack->pmb->mb_lev;
  auto &sbuf = sendbuf;
  auto &rbuf = recvbuf;
  auto &is_z4c = is_z4c_;
  auto &multilevel = pmy_pack->pmesh->multilevel;
#if MPI_PARALLEL_ENABLED
  // Build (or refresh) the rank-packed metadata BEFORE packing so the kernel can
  // write off-rank data straight into the aggregate buffer (fused pack/aggregate).
  if (rank_packed_bvals_nvars_ != nvar ||
      rank_packed_mesh_seq_ != pmy_pack->pmesh->GetAMRLoadBalanceUpdateSeq()) {
    BuildRankPackedVarMetadata(nvar);
  }
  auto aggsbuf = rank_sendbuf_vars_;
  auto sendoff = send_agg_offset_;
#endif
  // Outer loop over (# of MeshBlocks)*(# of buffers)*(# of variables)
  int nmnv = nmb*nnghbr*nvar;
  Kokkos::TeamPolicy<> policy(DevExeSpace(), nmnv, Kokkos::AUTO);
  Kokkos::parallel_for("SendBuff", policy, KOKKOS_LAMBDA(TeamMember_t tmember) {
    const int m = (tmember.league_rank())/(nnghbr*nvar);
    const int n = (tmember.league_rank() - m*(nnghbr*nvar))/nvar;
    const int v = (tmember.league_rank() - m*(nnghbr*nvar) - n*nvar);

    // only load buffers when neighbor exists
    if (nghbr.d_view(m,n).gid >= 0) {
      // if neighbor is at coarser level, use coar indices to pack buffer
      int il, iu, jl, ju, kl, ku;
      if (nghbr.d_view(m,n).lev < mblev.d_view(m)) {
        il = sbuf[n].icoar[0].bis;
        iu = sbuf[n].icoar[0].bie;
        jl = sbuf[n].icoar[0].bjs;
        ju = sbuf[n].icoar[0].bje;
        kl = sbuf[n].icoar[0].bks;
        ku = sbuf[n].icoar[0].bke;
      // if neighbor is at same level, use same indices to pack buffer
      } else if (nghbr.d_view(m,n).lev == mblev.d_view(m)) {
        il = sbuf[n].isame[0].bis;
        iu = sbuf[n].isame[0].bie;
        jl = sbuf[n].isame[0].bjs;
        ju = sbuf[n].isame[0].bje;
        kl = sbuf[n].isame[0].bks;
        ku = sbuf[n].isame[0].bke;
      // if neighbor is at finer level, use fine indices to pack buffer
      } else {
        il = sbuf[n].ifine[0].bis;
        iu = sbuf[n].ifine[0].bie;
        jl = sbuf[n].ifine[0].bjs;
        ju = sbuf[n].ifine[0].bje;
        kl = sbuf[n].ifine[0].bks;
        ku = sbuf[n].ifine[0].bke;
      }
      int ni = iu - il + 1;
      int nj = ju - jl + 1;
      int nk = ku - kl + 1;
      int nkj  = nk*nj;

      // indices of recv'ing (destination) MB and buffer: MB IDs are stored sequentially
      // in MeshBlockPacks, so array index equals (target_id - first_id)
      int dm = nghbr.d_view(m,n).gid - mbgid.d_view(0);
      int dn = nghbr.d_view(m,n).dest;

      // Middle loop over k,j
      Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nkj), [&](const int idx) {
        int k = idx / nj;
        int j = (idx - k * nj) + jl;
        k += kl;

        // Inner (vector) loop over i
        // copy directly into recv buffer if MeshBlocks on same rank

        if (nghbr.d_view(m,n).rank == my_rank) {
          // if neighbor is at same or finer level, load data from u0
          if (nghbr.d_view(m,n).lev >= mblev.d_view(m)) {
            Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il,iu+1),
            [&](const int i) {
              rbuf[dn].vars(dm, (i-il + ni*(j-jl + nj*(k-kl + nk*v))) ) = a(m,v,k,j,i);
            });
          // if neighbor is at coarser level, load data from coarse_u0
          } else {
            Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il,iu+1),
            [&](const int i) {
              rbuf[dn].vars(dm, (i-il + ni*(j-jl + nj*(k-kl + nk*v))) ) = ca(m,v,k,j,i);
            });
          }

        // else copy into send buffer for MPI communication below

        } else {
#if MPI_PARALLEL_ENABLED
          // off-rank: write straight into the rank-packed aggregate send buffer
          // at this entry's base offset (fuses the former RankPackAgg kernel).
          int base = sendoff(m*nnghbr + n);
          if (nghbr.d_view(m,n).lev >= mblev.d_view(m)) {
            Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il,iu+1),
            [&](const int i) {
              aggsbuf(base + (i-il + ni*(j-jl + nj*(k-kl + nk*v))) ) = a(m,v,k,j,i);
            });
          } else {
            Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il,iu+1),
            [&](const int i) {
              aggsbuf(base + (i-il + ni*(j-jl + nj*(k-kl + nk*v))) ) = ca(m,v,k,j,i);
            });
          }
#else
          // if neighbor is at same or finer level, load data from u0
          if (nghbr.d_view(m,n).lev >= mblev.d_view(m)) {
            Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il,iu+1),
            [&](const int i) {
              sbuf[n].vars(m, (i-il + ni*(j-jl + nj*(k-kl + nk*v))) ) = a(m,v,k,j,i);
            });
          // if neighbor is at coarser level, load data from coarse_u0
          } else {
            Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il,iu+1),
            [&](const int i) {
              sbuf[n].vars(m, (i-il + ni*(j-jl + nj*(k-kl + nk*v))) ) = ca(m,v,k,j,i);
            });
          }
#endif
        }
      });
    } // end if-neighbor-exists block
    tmember.team_barrier();
  }); // end par_for_outer

  Kokkos::parallel_for("SendBuff", policy, KOKKOS_LAMBDA(TeamMember_t tmember) {
    const int m = (tmember.league_rank())/(nnghbr*nvar);
    const int n = (tmember.league_rank() - m*(nnghbr*nvar))/nvar;
    const int v = (tmember.league_rank() - m*(nnghbr*nvar) - n*nvar);

    // only load buffers when neighbor exists
    if (nghbr.d_view(m,n).gid >= 0) {
      int il, iu, jl, ju, kl, ku;
      // If neighbor is at same level and data is for Z4c module, append data from coarse
      // array for higher-order prolongation
      if ((nghbr.d_view(m,n).lev == mblev.d_view(m)) && (is_z4c) && (multilevel)) {
        il = sbuf[n].isame_z4c.bis;
        iu = sbuf[n].isame_z4c.bie;
        jl = sbuf[n].isame_z4c.bjs;
        ju = sbuf[n].isame_z4c.bje;
        kl = sbuf[n].isame_z4c.bks;
        ku = sbuf[n].isame_z4c.bke;
        int ni = iu - il + 1;
        int nj = ju - jl + 1;
        int nk = ku - kl + 1;
        int nkj  = nk*nj;
        int ndat = nvar*sbuf[n].isame_ndat; // size of same level data already in buff

        // indices of recv'ing (destination) MB and buffer: MB IDs are stored sequentially
        // in MeshBlockPacks, so array index equals (target_id - first_id)
        int dm = nghbr.d_view(m,n).gid - mbgid.d_view(0);
        int dn = nghbr.d_view(m,n).dest;

        // Middle loop over k,j
        Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nkj), [&](const int idx) {
          int k = idx / nj;
          int j = (idx - k * nj) + jl;
          k += kl;

          // Inner (vector) loop over i
          // copy directly into recv buffer if MeshBlocks on same rank
          if (nghbr.d_view(m,n).rank == my_rank) {
            // load data from coarse_u0
            Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il,iu+1),
            [&](const int i) {
              rbuf[dn].vars(dm,ndat+ (i-il + ni*(j-jl + nj*(k-kl + nk*v))))=ca(m,v,k,j,i);
            });

          // else copy into send buffer for MPI communication below
          } else {
#if MPI_PARALLEL_ENABLED
            int base = sendoff(m*nnghbr + n);
            Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il,iu+1),
            [&](const int i) {
              aggsbuf(base+ndat+ (i-il + ni*(j-jl + nj*(k-kl + nk*v))) )=ca(m,v,k,j,i);
            });
#else
            // load data from coarse_u0
            Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il,iu+1),
            [&](const int i) {
              sbuf[n].vars(m,ndat+ (i-il + ni*(j-jl + nj*(k-kl + nk*v))) )=ca(m,v,k,j,i);
            });
#endif
          }
        });
      }
    } // end if-neighbor-exists block
    tmember.team_barrier();
  }); // end par_for_outer
  }

#if MPI_PARALLEL_ENABLED
  // Send boundary buffer to neighboring MeshBlocks using MPI. The SendBuff kernel
  // above already wrote every off-rank payload directly into rank_sendbuf_vars_,
  // so this single fence (which guarantees that kernel has completed) is all that
  // is needed before posting the sends.
  Kokkos::fence();
  bool no_errors = true;
  std::fill(send_var_reqs_.begin(), send_var_reqs_.end(), MPI_REQUEST_NULL);

  // Payload-only Isend: the receiver already knows the layout of this payload
  // from the one-shot header exchange done during BuildRankPackedVarMetadata.
  for (std::size_t i = 0; i < send_var_msgs_.size(); ++i) {
    const auto &msg = send_var_msgs_[i];
    int ierr = MPI_Isend(rank_sendbuf_vars_.data() + msg.offset, msg.data_size,
                     MPI_ATHENA_REAL, msg.rank, 1, comm_vars, &send_var_reqs_[i]);
    if (ierr != MPI_SUCCESS) no_errors = false;
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
// \!fn void RecvBuffers()
// \brief Unpack boundary buffers

TaskStatus MeshBoundaryValuesCC::RecvAndUnpackCC(DvceArray5D<Real> &a,
                                                 DvceArray5D<Real> &ca) {
  // create local references for variables in kernel
  int nmb = pmy_pack->nmb_thispack;
  int nnghbr = pmy_pack->pmb->nnghbr;
  auto &nghbr = pmy_pack->pmb->nghbr;
  auto &rbuf = recvbuf;
  auto &is_z4c = is_z4c_;
  auto &multilevel = pmy_pack->pmesh->multilevel;
#if MPI_PARALLEL_ENABLED
  //----- STEP 1: check that recv boundary buffer communications have all completed

  bool bflag = false;
  bool no_errors=true;
  for (std::size_t i = 0; i < recv_var_reqs_.size(); ++i) {
    int test;
    int ierr = MPI_Test(&recv_var_reqs_[i], &test, MPI_STATUS_IGNORE);
    if (ierr != MPI_SUCCESS) {no_errors=false;}
    if (!(static_cast<bool>(test))) {
      bflag = true;
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
  // Off-rank payloads are read directly out of the rank-packed aggregate recv
  // buffer (fuses the former RankUnpackScatter kernel); on-rank payloads remain
  // in their per-neighbour recv buffers, written there directly by the sender.

  int nvar = a.extent_int(1);  // TODO(@user): 2nd index from L of in array must be NVAR
  auto &mblev = pmy_pack->pmb->mb_lev;
#if MPI_PARALLEL_ENABLED
  auto aggrbuf = rank_recvbuf_vars_;
  auto recvoff = recv_agg_offset_;
#endif

  // Outer loop over (# of MeshBlocks)*(# of buffers)*(# of variables)
  Kokkos::TeamPolicy<> policy(DevExeSpace(), (nmb*nnghbr*nvar), Kokkos::AUTO);
  Kokkos::parallel_for("RecvBuff", policy, KOKKOS_LAMBDA(TeamMember_t tmember) {
    const int m = (tmember.league_rank())/(nnghbr*nvar);
    const int n = (tmember.league_rank() - m*(nnghbr*nvar))/nvar;
    const int v = (tmember.league_rank() - m*(nnghbr*nvar) - n*nvar);

    // only unpack buffers when neighbor exists
    if (nghbr.d_view(m,n).gid >= 0) {
      int il, iu, jl, ju, kl, ku;
      // if neighbor is at coarser level, use coar indices to unpack buffer
      if (nghbr.d_view(m,n).lev < mblev.d_view(m)) {
        il = rbuf[n].icoar[0].bis;
        iu = rbuf[n].icoar[0].bie;
        jl = rbuf[n].icoar[0].bjs;
        ju = rbuf[n].icoar[0].bje;
        kl = rbuf[n].icoar[0].bks;
        ku = rbuf[n].icoar[0].bke;
      // if neighbor is at same level, use same indices to unpack buffer
      } else if (nghbr.d_view(m,n).lev == mblev.d_view(m)) {
        il = rbuf[n].isame[0].bis;
        iu = rbuf[n].isame[0].bie;
        jl = rbuf[n].isame[0].bjs;
        ju = rbuf[n].isame[0].bje;
        kl = rbuf[n].isame[0].bks;
        ku = rbuf[n].isame[0].bke;
      // if neighbor is at finer level, use fine indices to unpack buffer
      } else {
        il = rbuf[n].ifine[0].bis;
        iu = rbuf[n].ifine[0].bie;
        jl = rbuf[n].ifine[0].bjs;
        ju = rbuf[n].ifine[0].bje;
        kl = rbuf[n].ifine[0].bks;
        ku = rbuf[n].ifine[0].bke;
      }
      int ni = iu - il + 1;
      int nj = ju - jl + 1;
      int nk = ku - kl + 1;
      int nkj  = nk*nj;

      // base offset of this (m,n) payload in the aggregate buffer; -1 == on-rank
#if MPI_PARALLEL_ENABLED
      const int base = recvoff(m*nnghbr + n);
#endif

      // Middle loop over k,j
      Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nkj), [&](const int idx) {
        int k = idx / nj;
        int j = (idx - k * nj) + jl;
        k += kl;

        // if neighbor is at same or finer level, load data directly into u0
        if (nghbr.d_view(m,n).lev >= mblev.d_view(m)) {
          Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il,iu+1),
          [&](const int i) {
            const int bi = (i-il + ni*(j-jl + nj*(k-kl + nk*v)));
#if MPI_PARALLEL_ENABLED
            a(m,v,k,j,i) = (base >= 0) ? aggrbuf(base + bi) : rbuf[n].vars(m, bi);
#else
            a(m,v,k,j,i) = rbuf[n].vars(m, bi);
#endif
          });

        // if neighbor is at coarser level, load data into coarse_u0
        } else {
          Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il,iu+1),
          [&](const int i) {
            const int bi = (i-il + ni*(j-jl + nj*(k-kl + nk*v)));
#if MPI_PARALLEL_ENABLED
            ca(m,v,k,j,i) = (base >= 0) ? aggrbuf(base + bi) : rbuf[n].vars(m, bi);
#else
            ca(m,v,k,j,i) = rbuf[n].vars(m, bi);
#endif
          });
        }
      });
    }  // end if-neighbor-exists block
    tmember.team_barrier();
  });  // end par_for_outer

  // Outer loop over (# of MeshBlocks)*(# of buffers)*(# of variables)
  Kokkos::parallel_for("RecvBuff", policy, KOKKOS_LAMBDA(TeamMember_t tmember) {
    const int m = (tmember.league_rank())/(nnghbr*nvar);
    const int n = (tmember.league_rank() - m*(nnghbr*nvar))/nvar;
    const int v = (tmember.league_rank() - m*(nnghbr*nvar) - n*nvar);
    // only unpack buffers when neighbor exists
    if (nghbr.d_view(m,n).gid >= 0) {
      int il, iu, jl, ju, kl, ku;
      // If neighbor is at same level and data is for Z4c module, unpack data from coarse
      // array for higher-order prolongation
      if ((nghbr.d_view(m,n).lev == mblev.d_view(m)) && (is_z4c) && (multilevel)) {
        il = rbuf[n].isame_z4c.bis;
        iu = rbuf[n].isame_z4c.bie;
        jl = rbuf[n].isame_z4c.bjs;
        ju = rbuf[n].isame_z4c.bje;
        kl = rbuf[n].isame_z4c.bks;
        ku = rbuf[n].isame_z4c.bke;
        int ni = iu - il + 1;
        int nj = ju - jl + 1;
        int nk = ku - kl + 1;
        int nkj  = nk*nj;
        int ndat = nvar*rbuf[n].isame_ndat; // size of same level data packed in buff
#if MPI_PARALLEL_ENABLED
        const int base = recvoff(m*nnghbr + n);
#endif

        // Middle loop over k,j
        Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nkj), [&](const int idx) {
          int k = idx / nj;
          int j = (idx - k * nj) + jl;
          k += kl;

          // load data into coarse_u0
          Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il,iu+1),
          [&](const int i) {
            const int bi = ndat + (i-il + ni*(j-jl + nj*(k-kl + nk*v)));
#if MPI_PARALLEL_ENABLED
            ca(m,v,k,j,i) = (base >= 0) ? aggrbuf(base + bi) : rbuf[n].vars(m, bi);
#else
            ca(m,v,k,j,i) = rbuf[n].vars(m, bi);
#endif
          });
        });
      }
    }  // end if-neighbor-exists block
    tmember.team_barrier();
  });  // end par_for_outer

  return TaskStatus::complete;
}
