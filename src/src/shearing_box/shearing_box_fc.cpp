//========================================================================================
// AthenaK astrophysical fluid dynamics code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file shearing_box_fc.cpp
//! \brief functions to pack/send and recv/unpack boundary values for face-centered (FC)
//! variables (magnetic fields) with shearing box boundaries.

#include <iostream>
#include <string>
#include <algorithm>
#include <vector>
#include <utility>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "bvals/bvals.hpp"
#include "shearing_box.hpp"
#include "remap_fluxes.hpp"

//----------------------------------------------------------------------------------------
// ShearingBoxFC derived class constructor:

ShearingBoxFC::ShearingBoxFC(MeshBlockPack *pp, ParameterInput *pin) :
    ShearingBox(pp, pin) {
  // Allocate boundary buffers
  auto &indcs = pp->pmesh->mb_indcs;
  int ncells3 = indcs.nx3 + 2*indcs.ng;
  int ncells2 = indcs.nx2 + 2*indcs.ng;
  int ncells1 = indcs.ng;
  for (int n=0; n<2; ++n) {
    int nmb = std::max(1,nmb_x1bndry(n));
    Kokkos::realloc(sendbuf[n].vars,nmb,ncells2,3,ncells3,ncells1);
    Kokkos::realloc(recvbuf[n].vars,nmb,ncells2,3,ncells3,ncells1);
  }
}

//----------------------------------------------------------------------------------------
//! \fn void ShearingBox::PackAndSendFC()
//! \brief Apply shearing sheet BCs to cell-centered variables, including MPI
//! MPI communications. Both the inner_x1 and outer_x1 boundaries are updated.
//! Called on the physics_bcs task after purely periodic BC communication is finished.

TaskStatus ShearingBoxFC::PackAndSendFC(DvceFaceFld4D<Real> &b,
                                        ReconstructionMethod rcon) {
  const auto &indcs = pmy_pack->pmesh->mb_indcs;
  const auto &ie = indcs.ie;
  const auto &js = indcs.js, &je = indcs.je;
  const auto &ks = indcs.ks, &ke = indcs.ke;
  const auto &ng = indcs.ng;

  // copy ghost zones at x1-faces into send buffer view
  // apply fractional cell offset to data in send buffers using conservative remap
  const auto &mbsize = pmy_pack->pmb->mb_size;
  int kl=ks, ku=ke;
  if (pmy_pack->pmesh->three_d) {kl -= ng; ku += ng;}
  int nj = indcs.nx2 + 2*ng;
  const int &gids_ = pmy_pack->gids;
  const Real &yshear_ = yshear;
  const auto &x1bndry_mbgid_ = x1bndry_mbgid;
  auto &sbuf = sendbuf;
  int scr_lvl=0;
  size_t scr_size = ScrArray1D<Real>::shmem_size(nj) * 2;
  for (int n=0; n<2; ++n) {
    int nmb1 = nmb_x1bndry(n) - 1;
    par_for_outer("shrcc",DevExeSpace(),scr_size,scr_lvl,0,nmb1,0,2,kl,ku,0,(ng-1),
    KOKKOS_LAMBDA(TeamMember_t member,const int m,const int v,const int k,const int i) {
      ScrArray1D<Real> a_(member.team_scratch(scr_lvl), nj); // 1D slice of data
      ScrArray1D<Real> flx(member.team_scratch(scr_lvl), nj); // "flux" at faces
      int mm = x1bndry_mbgid_.d_view(n,m) - gids_;

      // Load scratch array
      if (n==0) {
        if (v==0) {
          par_for_inner(member, 0, nj, [&](const int j) {
            a_(j) = b.x1f(mm,k,j,i);
          });
        } else if (v==1) {
          par_for_inner(member, 0, nj, [&](const int j) {
            a_(j) = b.x2f(mm,k,j,i);
          });
        } else if (v==2) {
          par_for_inner(member, 0, nj, [&](const int j) {
            a_(j) = b.x3f(mm,k,j,i);
          });
        }
      } else if (n==1) {
        if (v==0) {
          par_for_inner(member, 0, nj, [&](const int j) {
            a_(j) = b.x1f(mm,k,j,(ie+2)+i);
          });
        } else if (v==1) {
          par_for_inner(member, 0, nj, [&](const int j) {
            a_(j) = b.x2f(mm,k,j,(ie+1)+i);
          });
        } else if (v==2) {
          par_for_inner(member, 0, nj, [&](const int j) {
            a_(j) = b.x3f(mm,k,j,(ie+1)+i);
          });
        }
      }
      member.team_barrier();

      // compute fractional offset
      Real eps = fmod(yshear_,(mbsize.d_view(mm).dx2))/(mbsize.d_view(mm).dx2);
      if (n == 1) {eps *= -1.0;}

      // Compute "fluxes" at shifted cell faces
      switch (rcon) {
        case ReconstructionMethod::dc:
          DC_RemapFlx(member, js, (je+1), eps, a_, flx);
          break;
        case ReconstructionMethod::plm:
          PLM_RemapFlx(member, js, (je+1), eps, a_, flx);
          break;
        case ReconstructionMethod::ppm4:
        case ReconstructionMethod::ppmx:
        case ReconstructionMethod::wenoz:
          PPMX_RemapFlx(member, js, (je+1), eps, a_, flx);
          break;
        default:
          break;
      }
      member.team_barrier();

      // update data in send buffer with fracational shift
      par_for_inner(member, js, je, [&](const int j) {
        sbuf[n].vars(m,j,v,k,i) = a_(j) - (flx(j+1) - flx(j));
      });
    });
  }

  // shift data at x1 boundaries by integer number of cells.
  // Algorithm is broken into three steps: case1/2/3.
  //  * Case1 and case3 are when the integer shift (jr<ng), so that the sending MB
  //    overlaps the ghost cells of the two neighbors, and so requires copy/send
  //    to three separate target MBs.
  //  * Case2 is when the sending MB straddles the boundary between MBs, and so requires
  //    copy/send to only two target MBs.
  // Use deep copy if target MB on same rank, or MPI sends if not
  Kokkos::fence();
  const int &nx2 = indcs.nx2;
  bool no_errors=true;
  for (int n=0; n<2; ++n) {
    for (int m=0; m<nmb_x1bndry(n); ++m) {
      int gid = x1bndry_mbgid.h_view(n,m);
      int mm = gid - pmy_pack->gids;
      // Find integer and fractional number of grids over which offset extends.
      // This assumes every grid has same number of cells in x2-direction!
      int joffset  = static_cast<int>(yshear/(mbsize.h_view(mm).dx2));
      int ji = joffset/nx2;
      int jr = joffset - ji*nx2;

      if (jr < ng) {               //--- CASE 1 (in my nomenclature)
        int tgid, trank;
        std::pair<int,int> jsrc[3],jdst[3];
        if (n==0) {
          jsrc[0] = std::make_pair(js,js+ng-jr);
          jsrc[1] = std::make_pair(js,je+1);
          jsrc[2] = std::make_pair(je-(ng-1)-jr,je+1);
          jdst[0] = std::make_pair(je+1+jr,je+ng+1);
          jdst[1] = std::make_pair(js+jr,je+jr+1);
          jdst[2] = std::make_pair(js-ng,js+jr);
        } else {
          jsrc[0] = std::make_pair(js,js+ng+jr);
          jsrc[1] = std::make_pair(js,je+1);
          jsrc[2] = std::make_pair(je-(ng-1)+jr,je+1);
          jdst[0] = std::make_pair(je+1-jr,je+ng+1);
          jdst[1] = std::make_pair(js-jr,je-jr+1);
          jdst[2] = std::make_pair(js-ng,js-jr);
        }
        // ix1 boundary: send to (target-1) through (target+1)
        // ox1 boundary: send to (target-1) through (target+1)
        for (int l=0; l<3; ++l) {
          int jshift;
          if (n==0) {jshift = ji+l-1;} else {jshift = l-1-ji;} // offset of target
          FindTargetMB(gid,jshift,tgid,trank);
          if (trank == global_variable::my_rank) {
            int tm = TargetIndex(n,tgid);
            using Kokkos::ALL;
            auto src = subview(sendbuf[n].vars,m, jsrc[l],ALL,ALL,ALL);
            auto dst = subview(recvbuf[n].vars,tm,jdst[l],ALL,ALL,ALL);
            deep_copy(DevExeSpace(), dst, src);
#if MPI_PARALLEL_ENABLED
          } else {
            using Kokkos::ALL;
            auto send_ptr = subview(sendbuf[n].vars,m,jsrc[l],ALL,ALL,ALL);
            // create tag using GID of *receiving* MeshBlock
            int tag = CreateBvals_MPI_Tag(tgid, ((n<<2) | l));
            int data_size = send_ptr.size();
            int ierr = MPI_Isend(send_ptr.data(), data_size, MPI_ATHENA_REAL, trank, tag,
                                 comm_sbox, &(sendbuf[n].vars_req[3*m + l]));
            if (ierr != MPI_SUCCESS) {no_errors=false;}
#endif
          }
        }
      } else if (jr < (nx2-ng)) {  //--- CASE 2
        int tgid, trank;
        std::pair<int,int> jsrc[2],jdst[2];
        if (n==0) {
          jsrc[0] = std::make_pair(js,je+ng-jr+1);
          jsrc[1] = std::make_pair(je-(ng-1)-jr,je+1);
          jdst[0] = std::make_pair(js+jr,je+ng+1);
          jdst[1] = std::make_pair(js-ng,js+jr);
        } else {
          jsrc[0] = std::make_pair(js,js+ng+jr);
          jsrc[1] = std::make_pair(js-ng+jr,je+1);
          jdst[0] = std::make_pair(je-jr+1,je+ng+1);
          jdst[1] = std::make_pair(js-ng,je-jr+1);
        }
        // ix1 boundary: send to (target  ) through (target+1)
        // ox1 boundary: send to (target-1) through (target  )
        for (int l=0; l<2; ++l) {
          int jshift;
          if (n==0) {jshift = ji+l;} else {jshift = l-1-ji;}
          FindTargetMB(gid,jshift,tgid,trank);
          if (trank == global_variable::my_rank) {
            int tm = TargetIndex(n,tgid);
            using Kokkos::ALL;
            auto src = subview(sendbuf[n].vars,m, jsrc[l],ALL,ALL,ALL);
            auto dst = subview(recvbuf[n].vars,tm,jdst[l],ALL,ALL,ALL);
            deep_copy(DevExeSpace(), dst, src);
#if MPI_PARALLEL_ENABLED
          } else {
            using Kokkos::ALL;
            auto send_ptr = subview(sendbuf[n].vars,m,jsrc[l],ALL,ALL,ALL);
            // create tag using GID of *receiving* MeshBlock
            int tag = CreateBvals_MPI_Tag(tgid, ((n<<2) | l));
            int data_size = send_ptr.size();
            int ierr = MPI_Isend(send_ptr.data(), data_size, MPI_ATHENA_REAL, trank, tag,
                                 comm_sbox, &(sendbuf[n].vars_req[3*m + l]));
            if (ierr != MPI_SUCCESS) {no_errors=false;}
#endif
          }
        }
      } else {                     //--- CASE 3
        int tgid, trank;
        std::pair<int,int> jsrc[3],jdst[3];
        if (n==0) {
          jsrc[0] = std::make_pair(js,js+ng+(nx2-jr));
          jsrc[1] = std::make_pair(js,je+1);
          jsrc[2] = std::make_pair(je-(ng-1)+(nx2-jr),je+1);
          jdst[0] = std::make_pair(je+1-(nx2-jr),je+ng+1);
          jdst[1] = std::make_pair(js-(nx2-jr),je-(nx2-jr)+1);
          jdst[2] = std::make_pair(js-ng,js-(nx2-jr));
        } else {
          jsrc[0] = std::make_pair(js,js+ng-(nx2-jr));
          jsrc[1] = std::make_pair(js,je+1);
          jsrc[2] = std::make_pair(je-(ng-1)-(nx2-jr),je+1);
          jdst[0] = std::make_pair(je+1+(nx2-jr),je+ng+1);
          jdst[1] = std::make_pair(js+(nx2-jr),je+(nx2-jr)+1);
          jdst[2] = std::make_pair(js-ng,js+(nx2-jr));
        }
        // ix1 boundary: send to (target  ) through (target+2)
        // ox1 boundary: send to (target-2) through (target  )
        for (int l=0; l<3; ++l) {
          int jshift;
          if (n==0) {jshift = ji+l;} else {jshift = l-2-ji;}
          FindTargetMB(gid,jshift,tgid,trank);
          if (trank == global_variable::my_rank) {
            int tm = TargetIndex(n,tgid);
            using Kokkos::ALL;
            auto src = subview(sendbuf[n].vars,m, jsrc[l],ALL,ALL,ALL);
            auto dst = subview(recvbuf[n].vars,tm,jdst[l],ALL,ALL,ALL);
            deep_copy(DevExeSpace(), dst, src);
#if MPI_PARALLEL_ENABLED
          } else {
            using Kokkos::ALL;
            auto send_ptr = subview(sendbuf[n].vars,m,jsrc[l],ALL,ALL,ALL);
            // create tag using GID of *receiving* MeshBlock
            int tag = CreateBvals_MPI_Tag(tgid, ((n<<2) | l));
            int data_size = send_ptr.size();
            int ierr = MPI_Isend(send_ptr.data(), data_size, MPI_ATHENA_REAL, trank, tag,
                                 comm_sbox, &(sendbuf[n].vars_req[3*m + l]));
#endif
          }
        }
      }
    }
  }
  if (!(no_errors)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
       << std::endl << "MPI error in posting sends" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \!fn void ShearingBoxFC::RecvAndUnpackFC()
//! \brief Check MPI communication of boundary buffers for FC variables have finished,
//! then copy buffers into ghost zones. Shift has already been performed in
//! PackAndSendFC() function

TaskStatus ShearingBoxFC::RecvAndUnpackFC(DvceFaceFld4D<Real> &b) {
  // create local references for variables in kernel
  const auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int &ng = indcs.ng;
#if MPI_PARALLEL_ENABLED
  //----- STEP 1: check that recv boundary buffer communications have all completed
  const int &nx2 = indcs.nx2;
  bool bflag = false;
  bool no_errors=true;
  for (int n=0; n<2; ++n) {
    for (int m=0; m<nmb_x1bndry(n); ++m) {
      int gid = x1bndry_mbgid.h_view(n,m);
      int mm = gid - pmy_pack->gids;
      // Find integer and fractional number of grids over which offset extends.
      // This assumes every grid has same number of cells in x2-direction!
      int joffset  = static_cast<int>(yshear/(pmy_pack->pmb->mb_size.h_view(mm).dx2));
      int ji = joffset/nx2;
      int jr = joffset - ji*nx2;

      if (jr < ng) {               //--- CASE 1 (in my nomenclature)
        // ix1 boundary: receive from (target+1) through (target-1)
        // ox1 boundary: receive from (target+1) through (target-1)
        for (int l=0; l<3; ++l) {
          int jshift;
          if (n==0) {jshift = -(ji+l-1);} else {jshift = -(l-1-ji);} // offset of sender
          int sgid, srank;
          FindTargetMB(gid,jshift,sgid,srank);
          if (srank != global_variable::my_rank) {
            int test;
            int ierr = MPI_Test(&(recvbuf[n].vars_req[3*m + l]),&test,MPI_STATUS_IGNORE);
            if (ierr != MPI_SUCCESS) {no_errors=false;}
            if (!(static_cast<bool>(test))) {bflag = true;}
          }
        }
      } else if (jr < (nx2-ng)) {  //--- CASE 2
        // ix1 boundary: receive from (target  ) through (target-1)
        // ox1 boundary: receive from (target+1) through (target  )
        for (int l=0; l<2; ++l) {
          int jshift;
          if (n==0) {jshift = -(ji+l);} else {jshift = -(l-1-ji);} // offset of sender
          int sgid, srank;
          FindTargetMB(gid,jshift,sgid,srank);
          if (srank != global_variable::my_rank) {
            int test;
            int ierr = MPI_Test(&(recvbuf[n].vars_req[3*m + l]),&test,MPI_STATUS_IGNORE);
            if (ierr != MPI_SUCCESS) {no_errors=false;}
            if (!(static_cast<bool>(test))) {bflag = true;}
          }
        }
      } else {                     //--- CASE 3
        // ix1 boundary: send to (target  ) through (target+2)
        // ox1 boundary: send to (target-2) through (target  )
        for (int l=0; l<3; ++l) {
          int jshift;
          if (n==0) {jshift = -(ji+l);} else {jshift = -(l-2-ji);} // offset of sender
          int sgid, srank;
          FindTargetMB(gid,jshift,sgid,srank);
          if (srank != global_variable::my_rank) {
            int test;
            int ierr = MPI_Test(&(recvbuf[n].vars_req[3*m + l]),&test,MPI_STATUS_IGNORE);
            if (ierr != MPI_SUCCESS) {no_errors=false;}
            if (!(static_cast<bool>(test))) {bflag = true;}
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

  //----- STEP 2: communications have all completed, so unpack and apply shift
  // copy recv buffer view into ghost zones at x1-faces
  const int &ie = indcs.ie;
  int kl=indcs.ks, ku=indcs.ke;
  if (pmy_pack->pmesh->three_d) {kl -= ng; ku += ng;}
  int nj = indcs.nx2 + 2*ng;
  const int &gids_ = pmy_pack->gids;
  const auto &x1bndry_mbgid_ = x1bndry_mbgid;
  auto &rbuf = recvbuf;
  int scr_lvl=0;
  size_t scr_size = ScrArray1D<Real>::shmem_size(nj) * 3;
  for (int n=0; n<2; ++n) {
    int nmb1 = nmb_x1bndry(n) - 1;
    par_for_outer("shrcc",DevExeSpace(),scr_size,scr_lvl,0,nmb1,0,2,kl,ku,0,(ng-1),
    KOKKOS_LAMBDA(TeamMember_t member,const int m,const int v,const int k,const int i) {
      int mm = x1bndry_mbgid_.d_view(n,m) - gids_;
      if (n==0) {
        if (v==0) {
          par_for_inner(member, 0, nj, [&](const int j) {
            b.x1f(mm,k,j,i) = rbuf[n].vars(m,j,v,k,i);
          });
          member.team_barrier();
        } else if (v==1) {
          par_for_inner(member, 0, nj, [&](const int j) {
            b.x2f(mm,k,j,i) = rbuf[n].vars(m,j,v,k,i);
          });
          member.team_barrier();
        } else if (v==2) {
          par_for_inner(member, 0, nj, [&](const int j) {
            b.x3f(mm,k,j,i) = rbuf[n].vars(m,j,v,k,i);
          });
          member.team_barrier();
        }
      } else {
        if (v==0) {
          par_for_inner(member, 0, nj, [&](const int j) {
            b.x1f(mm,k,j,(ie+2)+i) = rbuf[n].vars(m,j,v,k,i);
          });
          member.team_barrier();
        } else if (v==1) {
          par_for_inner(member, 0, nj, [&](const int j) {
            b.x2f(mm,k,j,(ie+1)+i) = rbuf[n].vars(m,j,v,k,i);
          });
          member.team_barrier();
        } else if (v==2) {
          par_for_inner(member, 0, nj, [&](const int j) {
            b.x3f(mm,k,j,(ie+1)+i) = rbuf[n].vars(m,j,v,k,i);
          });
          member.team_barrier();
        }
      }
    });
  }

  return TaskStatus::complete;
}
