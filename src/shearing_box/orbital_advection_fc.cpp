//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file orbital_advection_fc.cpp
//! \brief functions to pack/send and recv/unpack boundary values for face-centered (FC)
//! variables in the orbital advection step used with the shearing box. Data is shifted
//! by the appropriate offset during the recv/unpack step, so these functions both
//! communicate the data and perform the shift.

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <utility>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/cell_locations.hpp"
#include "shearing_box.hpp"
#include "orbital_advection.hpp"
#include "mhd/mhd.hpp"
#include "remap_fluxes.hpp"

//----------------------------------------------------------------------------------------
// OrbitalAdvectionFC derived class constructor:

OrbitalAdvectionFC::OrbitalAdvectionFC(MeshBlockPack *pp, ParameterInput *pin) :
  OrbitalAdvection(pp, pin) {
  // Initialize boundary buffers
  int nmb = std::max((pp->nmb_thispack), (pp->pmesh->nmb_maxperrank));
  auto &indcs = pp->pmesh->mb_indcs;
  int ncells3 = indcs.nx3 + 1;
  int ncells2 = indcs.ng + maxjshift;
  int ncells1 = indcs.nx1 + 1;
  for (int n=0; n<2; ++n) {
    Kokkos::realloc(sendbuf[n].vars,nmb,2,ncells3,ncells2,ncells1);
    Kokkos::realloc(recvbuf[n].vars,nmb,2,ncells3,ncells2,ncells1);
  }

  // Allocate memory for electric fields
  nmb = pmy_pack->nmb_thispack;
  ncells1 = indcs.nx1 + 2*(indcs.ng);
  ncells2 = indcs.nx2 + 2*(indcs.ng);
  ncells3 = indcs.nx3 + 2*(indcs.ng);
  Kokkos::realloc(emfx,nmb,ncells3,ncells2,ncells1);
  Kokkos::realloc(emfz,nmb,ncells3,ncells2,ncells1);
}

//----------------------------------------------------------------------------------------
//! \fn void OrbitalAdvectionFC::PackAndSendFC()
//! \brief Pack face-centered fields into boundary buffers and send to neighbors for
//! the orbital advection step. Only ghost zones on the x2-faces (Y-faces) are passed.
//! Note only B3 and B1 need be passed.

TaskStatus OrbitalAdvectionFC::PackAndSendFC(DvceFaceFld4D<Real> &b) {
  // create local references for variables in kernel
  int nmb = pmy_pack->nmb_thispack;

  int my_rank = global_variable::my_rank;
  auto &nghbr = pmy_pack->pmb->nghbr;
  auto &mbgid = pmy_pack->pmb->mb_gid;
  auto &sbuf = sendbuf;
  auto &rbuf = recvbuf;

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  auto &is = indcs.is, &ie = indcs.ie;
  auto &js = indcs.js, &je = indcs.je;
  auto &ks = indcs.ks, &ke = indcs.ke;
  auto &ng = indcs.ng;
  const int &maxjshift_ = maxjshift;

  // Outer loop over (# of MeshBlocks)*(# of buffers)*(# of variables)
  int nmnv = nmb*2;  // only consider 2 neighbors (x2-faces) and only 2 vars
  Kokkos::TeamPolicy<> policy(DevExeSpace(), nmnv, Kokkos::AUTO);
  Kokkos::parallel_for("oa-packB", policy, KOKKOS_LAMBDA(TeamMember_t tmember) {
    const int m = tmember.league_rank()/2;
    const int n = tmember.league_rank()%2;

    // indices of x2-face buffers in nghbr view
    int nnghbr;
    if (n==0) {nnghbr=8;} else {nnghbr=12;}

    // only load buffers when neighbor exists
    if (nghbr.d_view(m,nnghbr).gid >= 0) {
      // neighbor must always be at same level, so use same indices to pack buffer
      // Note j-range of indices extended by shear
      int il = is;
      int iu = ie+1;
      int jl, ju;
      if (n==0) {
        jl = js;
        ju = js + (ng + maxjshift_ - 1);;
      } else {
        jl = je - (ng + maxjshift_ - 1);
        ju = je;
      }
      int kl = ks;
      int ku = ke+1;
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

        // copy B1/B3 directly into recv buffer if MeshBlocks on same rank
        if (nghbr.d_view(m,nnghbr).rank == my_rank) {
          rbuf[dn].vars(dm,0,(k-kl),(j-jl),(i-il)) = b.x3f(m,k,j,i);
          rbuf[dn].vars(dm,1,(k-kl),(j-jl),(i-il)) = b.x1f(m,k,j,i);
        // else copy B1/B3 into send buffer for MPI communication below
        } else {
          sbuf[n].vars(m,0,(k-kl),(j-jl),(i-il)) = b.x3f(m,k,j,i);
          sbuf[n].vars(m,1,(k-kl),(j-jl),(i-il)) = b.x1f(m,k,j,i);
        }
      });
    } // end if-neighbor-exists block
  }); // end par_for_outer

#if MPI_PARALLEL_ENABLED
  // Send boundary buffer to neighboring MeshBlocks using MPI
  Kokkos::fence();
  bool no_errors=true;
  for (int m=0; m<nmb; ++m) {
    for (int n=0; n<2; ++n) {
      // indices of x2-face buffers in nghbr view
      int nnghbr;
      if (n==0) {nnghbr=8;} else {nnghbr=12;}
      if (nghbr.h_view(m,nnghbr).gid >= 0) {  // neighbor exists and not a physical bndry
        // index and rank of destination Neighbor
        int dn = nghbr.h_view(m,nnghbr).dest;
        int drank = nghbr.h_view(m,nnghbr).rank;
        if (drank != my_rank) {
          // create tag using local ID and buffer index of *receiving* MeshBlock
          int lid = nghbr.h_view(m,nnghbr).gid - pmy_pack->pmesh->gids_eachrank[drank];
          int tag = CreateBvals_MPI_Tag(lid, dn);

          // get ptr to send buffer when neighbor is at coarser/same/fine level
          using Kokkos::ALL;
          auto send_ptr = Kokkos::subview(sbuf[n].vars, m, ALL, ALL, ALL, ALL);
          int data_size = send_ptr.size();

          int ierr = MPI_Isend(send_ptr.data(), data_size, MPI_ATHENA_REAL, drank, tag,
                               comm_orb_advect, &(sbuf[n].vars_req[m]));
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
//! \!fn void OrbitalAdvectionFC::RecvAndUnpackFC()
//! \brief Receive and unpack boundary buffers for FC fields with orbital advection, and
//! apply shift in x2- (y-) direction across entire MeshBlock. Since CT is required to
//! update fields, the algorithm used here is different from that used for CC variables in
//! RecvAndUnpackCC(). Here an effective electric field is computed including both the
//! integer and fractional cell shifts. These fields are then used to update B using CT.
//! The fields themselves are not directly remapped like the CC variables.

TaskStatus OrbitalAdvectionFC::RecvAndUnpackFC(DvceFaceFld4D<Real> &b0,
                                             ReconstructionMethod rcon) {
  int nmb = pmy_pack->nmb_thispack;
  auto &rbuf = recvbuf;
#if MPI_PARALLEL_ENABLED
  auto &nghbr = pmy_pack->pmb->nghbr;
  //----- STEP 1: check that recv boundary buffer communications have all completed

  bool bflag = false;
  bool no_errors=true;
  for (int m=0; m<nmb; ++m) {
    for (int n=0; n<2; ++n) {
      // indices of x2-face buffers in nghbr view
      int nnghbr;
      if (n==0) {nnghbr=8;} else {nnghbr=12;}
      if (nghbr.h_view(m,nnghbr).gid >= 0) { // neighbor exists and not a physical bndry
        if (nghbr.h_view(m,nnghbr).rank != global_variable::my_rank) {
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

  //----- STEP 2: buffers have all completed, so unpack and compute effective EMF

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  auto &is = indcs.is, &ie = indcs.ie;
  auto &js = indcs.js, &je = indcs.je;
  auto &ks = indcs.ks, &ke = indcs.ke;
  auto &ng = indcs.ng;
  int jfs = ng + maxjshift;
  int jfe = jfs + indcs.nx2 - 1;
  int nfx = indcs.nx2 + 2*(ng + maxjshift);

  auto &mbsize = pmy_pack->pmb->mb_size;
  auto &mesh_size = pmy_pack->pmesh->mesh_size;
  Real &dt = pmy_pack->pmesh->dt;
  Real ly = (mesh_size.x2max - mesh_size.x2min);
  Real qo = qshear*omega0;

  int scr_lvl=0;
  size_t scr_size = ScrArray1D<Real>::shmem_size(nfx) * 2;
  auto &emfx_ = emfx;
  auto &emfz_ = emfz;
  par_for_outer("oa-unB",DevExeSpace(),scr_size,scr_lvl,0,(nmb-1),0,1,ks,ke+1,is,ie+1,
  KOKKOS_LAMBDA(TeamMember_t member, const int m, const int v, const int k, const int i) {
    ScrArray1D<Real> b0_(member.team_scratch(scr_lvl), nfx); // 1D slice of data
    ScrArray1D<Real> flx(member.team_scratch(scr_lvl), nfx); // "flux" at faces

    Real &x1min = mbsize.d_view(m).x1min;
    Real &x1max = mbsize.d_view(m).x1max;
    int nx1 = indcs.nx1;

    Real x1;
    if (v==0) {
      // B3 located at x1-cell centers
      x1 = CellCenterX(i-is, nx1, x1min, x1max);
    } else if (v==1) {
      // B1 located at x1-cell faces
      x1 = LeftEdgeX(i-is, nx1, x1min, x1max);
    }
    Real yshear = -(qo)*x1*dt;
    int joffset = static_cast<int>(yshear/(mbsize.d_view(m).dx2));

    // Load scratch array with no shift
    par_for_inner(member, 0, (nfx-1), [&](const int jf) {
      if (jf < jfs) {
        // Load from L boundary buffer
        b0_(jf) = rbuf[0].vars(m,v,(k-ks),jf,(i-is));
      } else if (jf <= jfe) {
        // Load from array itself (addressed with j=jf-jfs+js)
        if (v==0) {
          b0_(jf) = b0.x3f(m,k,(jf-jfs+js),i);
        } else if (v==1) {
          b0_(jf) = b0.x1f(m,k,(jf-jfs+js),i);
        }
      } else {
        // Load scratch arrays from R boundary buffer
        b0_(jf) = rbuf[1].vars(m,v,(k-ks),jf-(jfe+1),(i-is));
      }
    });
    member.team_barrier();

    // Compute x2-fluxes at shifted cell faces
    Real epsi = fmod(yshear,(mbsize.d_view(m).dx2))/(mbsize.d_view(m).dx2);
    switch (rcon) {
      case ReconstructionMethod::dc:
        DC_RemapFlx(member, (jfs-joffset), (jfe+1-joffset), epsi, b0_, flx);
        break;
      case ReconstructionMethod::plm:
        PLM_RemapFlx(member, (jfs-joffset), (jfe+1-joffset), epsi, b0_, flx);
        break;
      case ReconstructionMethod::ppm4:
      case ReconstructionMethod::ppmx:
      case ReconstructionMethod::wenoz:
        PPMX_RemapFlx(member, (jfs-joffset), (jfe+1-joffset), epsi, b0_, flx);
        break;
      default:
        break;
    }
    member.team_barrier();

    // Compute emfx = -VyBz, which is at cell-center in x1-direction
    if (v==0) {
      par_for_inner(member, js, je+1, [&](const int j) {
        int jf = j-js + jfs;
        emfx_(m,k,j,i) = -flx(jf-joffset);
        // Sum integer offsets into effective EMFs
        for (int jj=1; jj<=joffset; jj++) {
          emfx_(m,k,j,i) -= b0_(jf-jj);
        }
        for (int jj=(joffset+1); jj<=0; jj++) {
          emfx_(m,k,j,i) += b0_(jf-jj);
        }
      });
      member.team_barrier();

    // Compute emfz =  VyBx, which is at cell-face in x1-direction
    } else if (v==1) {
      par_for_inner(member, js, je+1, [&](const int j) {
        int jf = j-js + jfs;
        emfz_(m,k,j,i) = flx(jf-joffset);
        // Sum integer offsets into effective EMFs
        for (int jj=1; jj<=joffset; jj++) {
          emfz_(m,k,j,i) += b0_(jf-jj);
        }
        for (int jj=(joffset+1); jj<=0; jj++) {
          emfz_(m,k,j,i) -= b0_(jf-jj);
        }
      });
      member.team_barrier();
    }
  });

  // Update face-centered fields using CT
  //---- update B1 (only for 2D/3D problems)
  if (pmy_pack->pmesh->multi_d) {
    par_for("oaCT-b1", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie+1,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      b0.x1f(m,k,j,i) -= (emfz_(m,k,j+1,i) - emfz_(m,k,j,i));
    });
  }

  //---- update B2 (curl terms in 1D and 3D problems)
  const bool &three_d_ = pmy_pack->pmesh->three_d;
  par_for("oaCT-b2", DevExeSpace(), 0, nmb-1, ks, ke, js, je+1, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real dydx = mbsize.d_view(m).dx2/mbsize.d_view(m).dx1;
    b0.x2f(m,k,j,i) += dydx*(emfz_(m,k,j,i+1) - emfz_(m,k,j,i));
    if (three_d_) {
      Real dydz = mbsize.d_view(m).dx2/mbsize.d_view(m).dx3;
      b0.x2f(m,k,j,i) -= dydz*(emfx_(m,k+1,j,i) - emfx_(m,k,j,i));
    }
  });

  //---- update B3 (curl terms in 1D and 2D/3D problems)
  if (pmy_pack->pmesh->multi_d) {
    par_for("oaCT-b3", DevExeSpace(), 0, nmb-1, ks, ke+1, js, je, is, ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      b0.x3f(m,k,j,i) += (emfx_(m,k,j+1,i) - emfx_(m,k,j,i));
    });
  }

  return TaskStatus::complete;
}
