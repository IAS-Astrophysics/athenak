//========================================================================================
// AthenaK astrophysical fluid dynamics code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file shearing_box_cc.cpp
//! \brief constructor for ShearingBoxBoundary abstract base class, as well as functions
//! to pack/send and recv/unpack boundary values for cell-centered (CC) variables with
//! shearing box boundaries. Data is shifted by the appropriate offset during the
//! recv/unpack step, so these functions both communicate the data and perform the shift.
//! Based on BoundaryValues send/recv funcs.

#include <iostream>
#include <string>
#include <algorithm>
#include <vector>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "bvals/bvals.hpp"
#include "shearing_box.hpp"
#include "remap_fluxes.hpp"

//----------------------------------------------------------------------------------------
//! ShearingBoxBoundary base class constructor

ShearingBoxBoundary::ShearingBoxBoundary(MeshBlockPack *ppack, ParameterInput *pin,
                                         int nvar) :
    nmb_x1bndry("nmbx1",2),
    x1bndry_mbgid("x1gid",1,1),
    pmy_pack(ppack) {
  // Work out how many MBs on this rank are at ix1/ox1 shearing-box boundaries
  std::vector<int> tmp_ix1bndry_gid, tmp_ox1bndry_gid;
  auto &mbbcs = ppack->pmb->mb_bcs;
  for (int m=0; m<(ppack->nmb_thispack); ++m) {
    if (mbbcs.h_view(m,BoundaryFace::inner_x1) == BoundaryFlag::shear_periodic) {
      tmp_ix1bndry_gid.push_back(m + ppack->gids);
    }
    if (mbbcs.h_view(m,BoundaryFace::outer_x1) == BoundaryFlag::shear_periodic) {
      tmp_ox1bndry_gid.push_back(m + ppack->gids);
    }
  }
  nmb_x1bndry(0) = tmp_ix1bndry_gid.size();
  nmb_x1bndry(1) = tmp_ox1bndry_gid.size();

  // allocate mbgid array and initialize GIDs to -1
  int nmb = std::max(nmb_x1bndry(0),nmb_x1bndry(1));
  Kokkos::realloc(x1bndry_mbgid, 2, nmb);
  for (int n=0; n<2; ++n) {
    for (int m=0; m<nmb; ++m) {
      x1bndry_mbgid(n,m) = -1;
    }
  }
  // load GIDs of meshblocks at x1 boundaries into DualArray
  for (int m=0; m<nmb_x1bndry(0); ++m) {
    x1bndry_mbgid(0,m) = tmp_ix1bndry_gid[m];
  }
  for (int m=0; m<nmb_x1bndry(1); ++m) {
    x1bndry_mbgid(1,m) = tmp_ox1bndry_gid[m];
  }

#if MPI_PARALLEL_ENABLED
  // initialize vectors of MPI requests for ix1/ox1 boundaries in fixed length arrays
  // each MB on x1-face can communicate with up to 3 nghbrs
  for (int n=0; n<2; ++n) {
    if (nmb_x1bndry(n) > 0) {
      sendbuf[n].vars_req = new MPI_Request[3*nmb_x1bndry(n)];
      recvbuf[n].vars_req = new MPI_Request[3*nmb_x1bndry(n)];
      for (int m=0; m<nmb_x1bndry(0); ++m) {
        for (int n=0; n<3; ++n) {
          sendbuf[n].vars_req[3*m + n] = MPI_REQUEST_NULL;
          recvbuf[n].vars_req[3*m + n] = MPI_REQUEST_NULL;
        }
      }
    }
  }
  // create unique communicators for shearing box
  MPI_Comm_dup(MPI_COMM_WORLD, &comm_sbox);
#endif
}

//----------------------------------------------------------------------------------------
// ShearingBoxBoundary base class destructor

ShearingBoxBoundary::~ShearingBoxBoundary() {
#if MPI_PARALLEL_ENABLED
  for (int n=0; n<2; ++n) {
    if (nmb_x1bndry(n) > 0) {
      delete [] sendbuf[n].vars_req;
      delete [] recvbuf[n].vars_req;
    }
  }
#endif
}

//----------------------------------------------------------------------------------------
// ShearingBoxBoundaryCC derived class constructor:

ShearingBoxBoundaryCC::ShearingBoxBoundaryCC(MeshBlockPack *pp, ParameterInput *pin,
                                             int nvar) :
    ShearingBoxBoundary(pp, pin, nvar) {
  // Allocate boundary buffers
  auto &indcs = pp->pmesh->mb_indcs;
  int ncells3 = indcs.nx3 + 2*indcs.ng;
  int ncells2 = indcs.nx2 + 2*indcs.ng;
  int ncells1 = indcs.ng;
  for (int n=0; n<2; ++n) {
    int nmb = std::max(1,nmb_x1bndry(n));
    Kokkos::realloc(sendbuf[n].vars,nmb,nvar,ncells3,ncells2,ncells1);
    Kokkos::realloc(recvbuf[n].vars,nmb,nvar,ncells3,ncells2,ncells1);
  }
}

//----------------------------------------------------------------------------------------
//! \fn void ShearingBoxBoundary::FindTargetMB()
//! \brief  function to find target MB offset by shear.  Returns GID and rank

void ShearingBoxBoundary::FindTargetMB(const int igid, const int jshift, int &gid,
                                       int &rank){
  Mesh *pm = pmy_pack->pmesh;
  // find lloc of input MB
  LogicalLocation lloc = pm->lloc_eachmb[igid];
  // find number of MBs in x2 direction at this level
  std::int32_t nmbx2 = pm->nmb_rootx2 << (lloc.level - pm->root_level);
  // apply shift by input number of blocks
  lloc.lx2 = static_cast<std::int32_t>((lloc.lx2 + jshift)/nmbx2);
  // find target GID and rank
  gid = (pm->ptree->FindMeshBlock(lloc))->GetGID();
  rank = pm->rank_eachmb[gid];
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void ShearingBoxBoundary::PackAndSendCC()
//! \brief Apply shearing sheet BCs to cell-centered variables, including MPI
//! MPI communications. Both the inner_x1 and outer_x1 boundaries are updated.
//! Called on the physics_bcs task after purely periodic BC communication is finished.

TaskStatus ShearingBoxBoundaryCC::PackAndSendCC(DvceArray5D<Real> &a,
                                                ReconstructionMethod rcon, Real qom) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  auto &is = indcs.is, &ie = indcs.ie;
  auto &js = indcs.js, &je = indcs.je;
  auto &ks = indcs.ks, &ke = indcs.ke;
  auto &ng = indcs.ng;
  // copy ghost zones at x1-faces into send buffer view
  for (int n=0; n<2; ++n) {
    std::pair<int,int> isrc;
    if (n==0) {
      isrc = std::make_pair(0,ng);
    } else {
      isrc = std::make_pair(ie+1,ie+1+ng);
    }
    for (int m=0; m<nmb_x1bndry(n); ++m) {
      int mm = x1bndry_mbgid(n,m) - pmy_pack->gids;
      using namespace Kokkos;
      auto src = subview(a,mm,ALL,ALL,ALL,isrc);
      auto dst = subview(sendbuf[n].vars,m,ALL,ALL,ALL,ALL);
      deep_copy(DevExeSpace(), dst, src);
    }
  }

  // figure out distance boundaries are sheared
  auto &mesh_size = pmy_pack->pmesh->mesh_size;
  Real &time = pmy_pack->pmesh->time;
  Real lx = (mesh_size.x1max - mesh_size.x1min);
  Real yshear = qom*lx*time;

  // apply fractional cell offset to data in send buffers using conservative remap
  int nvar = a.extent_int(1);  // TODO(@user): 2nd index from L of in array must be NVAR
  auto &mbsize = pmy_pack->pmb->mb_size;
  int nj = indcs.nx2 + 2*ng;
  auto sbuf = sendbuf;
  int scr_lvl=0;
  size_t scr_size = ScrArray1D<Real>::shmem_size(nj) * 3;
  for (int n=0; n<2; ++n) {
    int nmb1 = nmb_x1bndry(n) - 1;
    par_for_outer("shrcc",DevExeSpace(),scr_size,scr_lvl,0,nmb1,0,(nvar-1),ks,ke,is,ie,
    KOKKOS_LAMBDA(TeamMember_t member, const int m, const int v, const int k, const int i)
    {
      ScrArray1D<Real> u0_(member.team_scratch(scr_lvl), nj); // 1D slice of data
      ScrArray1D<Real> flx(member.team_scratch(scr_lvl), nj); // "flux" at faces
      ScrArray1D<Real> q1_(member.team_scratch(scr_lvl), nj); // scratch array

      // Load scratch array
      par_for_inner(member, 0, nj, [&](const int j) {
        u0_(j) = sbuf[n].vars(m,v,k,j,i);
      });
      member.team_barrier();

      // compute fractional offset
      Real eps = fmod(yshear,(mbsize.d_view(m).dx2))/(mbsize.d_view(m).dx2);
      if (n == 1) {eps *= -1.0;}

      // Compute "fluxes" at shifted cell faces
      switch (rcon) {
        case ReconstructionMethod::dc:
          DCRemapFlx(member, js, (je+1), eps, u0_, q1_, flx);
          break;
        case ReconstructionMethod::plm:
          PLMRemapFlx(member, js, (je+1), eps, u0_, q1_, flx);
          break;
//      case ReconstructionMethod::ppm4:
//      case ReconstructionMethod::ppmx:
//          PPMRemapFlx(member,eos_,extrema,true,m,k,j,il,iu, w0_, wl_jp1, wr);
//        break;
        default:
          break;
      }
      member.team_barrier();

      // update data in send buffer with fracational shift
      par_for_inner(member, js, je, [&](const int j) {
        sbuf[n].vars(m,v,k,j,i) -= (flx(j+1) - flx(j));
      });
    });
  }

  // shift data at x1 boundaries by integer number of cells.
  // Algorithm is broken into three steps: case1/2/3.
  //  * Case1 and case3 are when the integer shift (jr<ng), so that the sending MB
  //    overlaps the ghost cells of the two neighbors, and so requires copies
  //    to three seperate target MB.
  //  * Case2 is when the sending MB straddles the boundary between MBs, and so requires
  //    copies to only two target MBs.
  // j-indices of domain to be copied are stored in std::pair in each case.
  // Use deep copy if target MB on same rank, or MPI sends if not
  for (int n=0; n<2; ++n) {
    int &nx2 = indcs.nx2;
    for (int m=0; m<nmb_x1bndry(n); ++m) {
      int gid = x1bndry_mbgid(n,m);
      int mm = gid - pmy_pack->gids;
      // Find integer and fractional number of grids over which offset extends.
      // This assumes every grid has same number of cells in x2-direction!
      int joffset  = static_cast<int>(yshear/(mbsize.h_view(mm).dx2));
      int ji = joffset/(pmy_pack->pmesh->mb_indcs.nx2);
      int jr = joffset - ji*(pmy_pack->pmesh->mb_indcs.nx2);

      if (jr < ng) {  //-------------------------------------- CASE 1 (in my nomenclature)
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
        // send to (target-1) through (target+1) [ix1 boundary]
        // send to (target-1) through (target+1) [ox1 boundary]
        for (int l=0; l<=2; ++l) {
          int jshift = l-1;
          FindTargetMB(gid,(ji+jshift),tgid,trank);
          if (trank == global_variable::my_rank) {
            int tm = TargetIndex(n,tgid);
            using namespace Kokkos;
            auto src = subview(sendbuf[n].vars,m, ALL,ALL,jsrc[l],ALL);
            auto dst = subview(recvbuf[n].vars,tm,ALL,ALL,jdst[l],ALL);
            deep_copy(DevExeSpace(), dst, src);
#if MPI_PARALLEL_ENABLED
          } else {
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
          jsrc[0] = std::make_pair(js,je+ng+jr+1);
          jsrc[1] = std::make_pair(js-ng+jr,je+1);
          jdst[0] = std::make_pair(je-jr+1,je+ng+1);
          jdst[1] = std::make_pair(js-ng,je-jr+1);
        }
        // send to (target  ) through (target+1) [ix1 boundary]
        // send to (target-1) through (target  ) [ox1 boundary]
        for (int l=0; l<=1; ++l) {
          int jshift;
          if (n==0) {jshift = l;} else {jshift = l-1;}
          FindTargetMB(gid,(ji+jshift),tgid,trank);
          if (trank == global_variable::my_rank) {
            int tm = TargetIndex(n,tgid);
            using namespace Kokkos;
            auto src = subview(sendbuf[n].vars,m, ALL,ALL,jsrc[l],ALL);
            auto dst = subview(recvbuf[n].vars,tm,ALL,ALL,jdst[l],ALL);
            deep_copy(DevExeSpace(), dst, src);
#if MPI_PARALLEL_ENABLED
          } else {
#endif
          }
        }
      } else {  //--------------------------------------------------- CASE 3
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
        // send to (target  ) through (target+2) [ix1 boundary]
        // send to (target-2) through (target  ) [ox1 boundary]
        for (int l=0; l<=2; ++l) {
          int jshift;
          if (n==0) {jshift = l;} else {jshift = l-2;}
          FindTargetMB(gid,(ji+jshift),tgid,trank);
          if (trank == global_variable::my_rank) {
            int tm = TargetIndex(n,tgid);
            using namespace Kokkos;
            auto src = subview(sendbuf[n].vars,m, ALL,ALL,jsrc[l],ALL);
            auto dst = subview(recvbuf[n].vars,tm,ALL,ALL,jdst[l],ALL);
            deep_copy(DevExeSpace(), dst, src);
#if MPI_PARALLEL_ENABLED
          } else {
#endif
          }
        }
      }
    }
  }

/*

#if MPI_PARALLEL_ENABLED
  // Send boundary buffer to neighboring MeshBlocks using MPI
  Kokkos::fence();
  bool no_errors=true;
  // Quit if MPI error detected
  if (!(no_errors)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
       << std::endl << "MPI error in posting sends" << std::endl;
    std::exit(EXIT_FAILURE);
  }
#endif
*/
  return TaskStatus::complete;
}
