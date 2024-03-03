//========================================================================================
// AthenaK astrophysical fluid dynamics code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file orbital_advection.cpp
//! \brief functions to pack/send and recv/unpack boundary values for cell-centered (CC)
//! variables in the orbital advection step used with the shearing box. Data is shifted
//! by the appropriate offset during the recv/unpack step, so these functions both
//! communicate the data and perform the shift. Based on BoundaryValues send/recv funcs.

#include <cstdlib>
#include <iostream>
#include <utility>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/cell_locations.hpp"
#include "shearing_box.hpp"
#include "remap_fluxes.hpp"

//----------------------------------------------------------------------------------------
//! \fn void ShearingBox::PackAndSendCC_Orb()
//! \brief Pack cell-centered variables into boundary buffers and send to neighbors for
//! the orbital advection step.  Only ghost zones on the x2-faces (Y-faces) are passed.
//! Since fine/coarse boundaries parallel to x2-faces are not allowed in the shearing box
//! (i.e. the mesh resolution must be constant in the x2-direction), communication of
//! coarse arrays is not needed.
//!
//! Input arrays must be 5D Kokkos View dimensioned (nmb, nvar, nx3, nx2, nx1)

TaskStatus ShearingBox::PackAndSendCC_Orb(DvceArray5D<Real> &a) {
  // create local references for variables in kernel
  int nmb = pmy_pack->nmb_thispack;
  int nvar = a.extent_int(1);  // TODO(@user): 2nd index from L of in array must be NVAR

  int my_rank = global_variable::my_rank;
  auto &nghbr = pmy_pack->pmb->nghbr;
  auto &mbgid = pmy_pack->pmb->mb_gid;
  auto &mblev = pmy_pack->pmb->mb_lev;
  auto &sbuf = sendbuf_orb;
  auto &rbuf = recvbuf_orb;

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  auto &is = indcs.is, &ie = indcs.ie;
  auto &js = indcs.js, &je = indcs.je;
  auto &ks = indcs.ks, &ke = indcs.ke;
  auto &ng = indcs.ng;

  // Outer loop over (# of MeshBlocks)*(# of buffers)*(# of variables)
  int nmnv = nmb*2*nvar;  // only consider 2 neighbors (x2-faces)
  Kokkos::TeamPolicy<> policy(DevExeSpace(), nmnv, Kokkos::AUTO);
  Kokkos::parallel_for("SendBuff", policy, KOKKOS_LAMBDA(TeamMember_t tmember) {
    const int m = (tmember.league_rank())/(2*nvar);
    const int n = (tmember.league_rank() - m*(2*nvar))/nvar;
    const int v = (tmember.league_rank() - m*(2*nvar) - n*nvar);

    // indices of x2-face buffers in nghbr view
    int nnghbr;
    if (n==0) {nnghbr=8;} else {nnghbr=12;}

    // only load buffers when neighbor exists
    if (nghbr.d_view(m,nnghbr).gid >= 0) {
      // neighbor must always be at same level, so use same indices to pack buffer
      // Note j-range of indices extended by shear
      int il = is;
      int iu = ie;
      int jl, ju;
      if (n==0) {
        int jl = js;
        int ju = js + ng + maxjshift;;
      } else {
        int jl = je - ng - maxjshift;
        int ju = je;
      }
      int kl = ks;
      int ku = ke;
      int ni = iu - il + 1;
      int nj = ju - jl + 1;
      int nk = ku - kl + 1;
      int nji = nj*ni;
      int nkji = nk*nj*ni;

      // index of recv'ing (destination) MB and buffer [0,1]: MB IDs are stored
      // sequentially in MeshBlockPacks, so array index equals (target_id - first_id)
      int dm = nghbr.d_view(m,nnghbr).gid - mbgid.d_view(0);
      int dn = (n+1) % 2;

      // Middle loop over k,j,i
      Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nkji), [&](const int idx) {
        int k = (idx)/nji;
        int j = (idx - k*nji)/ni;
        int i = (idx - k*nji - j*ni) + il;
        k += kl;
        j += jl;

        // copy directly into recv buffer if MeshBlocks on same rank
        if (nghbr.d_view(m,nnghbr).rank == my_rank) {
          rbuf[dn].vars(dm,v,k-kl,j-jl,i-il) = a(m,v,k,j,i);

        // else copy into send buffer for MPI communication below
        } else {
          sbuf[n].vars(m,v,k-kl,j-jl,i-il) = a(m,v,k,j,i);
        }
      });
    } // end if-neighbor-exists block
  }); // end par_for_outer

#if MPI_PARALLEL_ENABLED
  // Send boundary buffer to neighboring MeshBlocks using MPI
  Kokkos::fence();
  bool no_errors=true;
  for (int m=0; m<nmb; ++m) {
    for (int n=8; n<=12; n+=4) {
      if (nghbr.h_view(m,n).gid >= 0) {  // neighbor exists and not a physical boundary
        // index and rank of destination Neighbor
        int dn = nghbr.h_view(m,n).dest;
        int drank = nghbr.h_view(m,n).rank;
        if (drank != my_rank) {
          // create tag using local ID and buffer index of *receiving* MeshBlock
          int lid = nghbr.h_view(m,n).gid - pmy_pack->pmesh->gids_eachrank[drank];
          int tag = CreateBvals_MPI_Tag(lid, dn);

          // get ptr to send buffer when neighbor is at coarser/same/fine level
          auto send_ptr = Kokkos::subview(sbuf[n].vars, m, Kokkos::ALL, Kokkos::ALL,
                                          Kokkos::ALL, Kokkos::ALL);
          int data_size = send_ptr.size();

          int ierr = MPI_Isend(send_ptr.data(), data_size, MPI_ATHENA_REAL, drank, tag,
                               comm_orb, &(sbuf[n].vars_req[m]));
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
// \!fn void RecvAndUnpackCC_Orb()
// \brief Receive and unpack boundary buffers for CC variables
//! Remaps cell-centered variables in input array u0 using orbital advection during unpack

TaskStatus ShearingBox::RecvAndUnpackCC_Orb(DvceArray5D<Real> &u0,
                                            ReconstructionMethod rcon){
  // create local references for variables in kernel
  int nmb = pmy_pack->nmb_thispack;
  int nnghbr = pmy_pack->pmb->nnghbr;
  auto &nghbr = pmy_pack->pmb->nghbr;
  auto &rbuf = recvbuf_orb;
#if MPI_PARALLEL_ENABLED
  //----- STEP 1: check that recv boundary buffer communications have all completed

  bool bflag = false;
  bool no_errors=true;
  for (int m=0; m<nmb; ++m) {
    for (int n=0; n<nnghbr; ++n) {
      if (nghbr.h_view(m,n).gid >= 0) { // neighbor exists and not a physical boundary
        if (nghbr.h_view(m,n).rank != global_variable::my_rank) {
          int test;
          int ierr = MPI_Test(&(rbuf[n].vars_req[m]), &test, MPI_STATUS_IGNORE);
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

  int nvar = u0.extent_int(1);  // TODO(@user): 2nd index from L of in array must be NVAR

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  auto &is = indcs.is, &ie = indcs.ie;
  auto &js = indcs.js, &je = indcs.je;
  auto &ks = indcs.ks, &ke = indcs.ke;
  auto &ng = indcs.ng;
  int ncells2 = indcs.nx2 + 2*(indcs.ng);

  auto &mb_size = pmy_pack->pmb->mb_size;
  auto &mesh_size = pmy_pack->pmesh->mesh_size;
  Real &time = pmy_pack->pmesh->time;
  Real qom = qshear*omega0;
  Real ly = (mesh_size.x2max - mesh_size.x2min);

  int scr_lvl=0;
  size_t scr_size = ScrArray1D<Real>::shmem_size(ncells2) * 2;
  par_for_outer("oadv",DevExeSpace(),scr_size,scr_lvl,0,(nmb-1),0,(nvar-1),ks,ke,is,ie,
  KOKKOS_LAMBDA(TeamMember_t member, const int m, const int n, const int k, const int i) {
    ScrArray1D<Real> u0_(member.team_scratch(scr_lvl), ncells2);
    ScrArray1D<Real> flx(member.team_scratch(scr_lvl), ncells2);

    Real &x1min = mb_size.d_view(m).x1min;
    Real &x1max = mb_size.d_view(m).x1max;
    int nx1 = indcs.nx1;
    Real x1v = CellCenterX(i-is, nx1, x1min, x1max);

    Real yshear = -qom*x1v*time;
    Real deltay = fmod(yshear, ly);
    int joffset = static_cast<int>(deltay/(mb_size.d_view(m).dx2));

    // Load scratch array.  Index with shift:  jj = j + jshift
    par_for_inner(member, 0, (ncells2-1), [&](const int jj) {
      if (jj < (js + joffset)) {
        // Load scratch arrays from L boundary buffer with offset
        u0_(jj) = rbuf[0].vars(m,n,k-ks,jj,i-is);
      } else if (jj < (je + joffset)) {
        // Load from array itself with offset
        u0_(jj) = u0(m,n,k,(jj+joffset),i);
      } else {
        // Load scratch arrays from R boundary buffer with offset
        u0_(jj) = rbuf[1].vars(m,n,k-ks,(jj-(je+1)),i-is);
      }
    });


    // Compute x2-fluxes from fractional offset, including in ghost zones
    Real epsi = fmod(deltay,(mb_size.d_view(m).dx2))/(mb_size.d_view(m).dx2);
    switch (rcon) {
      case ReconstructionMethod::dc:
        DonorCellOrbAdvFlx(member, js, je+1, epsi, u0_, flx);
        break;
      case ReconstructionMethod::plm:
        PiecewiseLinearOrbAdvFlx(member, js, je+1, epsi, u0_, flx);
        break;
//      case ReconstructionMethod::ppm4:
//      case ReconstructionMethod::ppmx:
//          PiecewiseParabolicOrbAdvFlx(member,eos_,extrema,true,m,k,j,il,iu, w0_, wl_jp1, wr);
//        break;
      default:
        break;
    }
    member.team_barrier();

    // Update CC variables (including ghost zones) with orbital advection fluxes
    par_for_inner(member, js, je, [&](const int j) {
      u0(m,n,k,j,i) = u0_(j) + (flx(j+1) - flx(j));
    });
  });

  return TaskStatus::complete;
}
