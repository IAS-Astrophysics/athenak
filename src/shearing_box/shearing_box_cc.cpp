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
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "diffusion/viscosity.hpp"
#include "diffusion/conduction.hpp"
#include "srcterms/srcterms.hpp"
#include "bvals/bvals.hpp"
#include "shearing_box.hpp"

//----------------------------------------------------------------------------------------
//! ShearingBoxBoundary base class constructor

ShearingBoxBoundary::ShearingBoxBoundary(MeshBlockPack *ppack, ParameterInput *pin,
                                         int nvar) :
    nmb_ix1bndry(0),
    nmb_ox1bndry(0),
    pmy_pack(ppack) {
  // Work out how many MBs on this rank are at ix1/ox1 shearing-box boundaries
  std::vector<int> tmp_ix1bndry_mbs, tmp_ox1bndry_mbs;
  auto &mbbcs = ppack->pmb->mb_bcs;
  for (int m=0; m<(ppack->nmb_thispack); ++m) {
    if (mbbcs.h_view(m,BoundaryFace::inner_x1) == BoundaryFlag::shear_periodic) {
      tmp_ix1bndry_mbs.push_back(m + ppack->gids);
    }
    if (mbbcs.h_view(m,BoundaryFace::outer_x1) == BoundaryFlag::shear_periodic) {
      tmp_ox1bndry_mbs.push_back(m + ppack->gids);
    }
  }
  nmb_ix1bndry = tmp_ix1bndry_mbs.size();
  nmb_ox1bndry = tmp_ox1bndry_mbs.size();

  // load GIDs of meshblocks at ix1 boundaries into DualArray
  if (nmb_ix1bndry > 0) {
    Kokkos::realloc(ix1bndry_mbgid, nmb_ix1bndry);
    for (int m=0; m<nmb_ix1bndry; ++m) {
      ix1bndry_mbgid.h_view(m) = tmp_ix1bndry_mbs[m];
    }
    // sync device array
    ix1bndry_mbgid.template modify<HostMemSpace>();
    ix1bndry_mbgid.template sync<DevExeSpace>();
  }
  // load GIDs of meshblocks at ox1 boundaries into DualArray
  if (nmb_ox1bndry > 0) {
    Kokkos::realloc(ox1bndry_mbgid, nmb_ox1bndry);
    for (int m=0; m<nmb_ox1bndry; ++m) {
      ox1bndry_mbgid.h_view(m) = tmp_ox1bndry_mbs[m];
    }
    // sync device array
    ox1bndry_mbgid.template modify<HostMemSpace>();
    ox1bndry_mbgid.template sync<DevExeSpace>();
  }

#if MPI_PARALLEL_ENABLED
  // initialize vectors of MPI requests for ix1/ox1 boundaries in fixed length arrays
  if (nmb_ix1bndry > 0) {
    for (int n=0; n<3; ++n) {
      sendbuf_ix1[n].vars_req = new MPI_Request[nmb_ix1bndry];
      recvbuf_ix1[n].vars_req = new MPI_Request[nmb_ix1bndry];
      for (int m=0; m<nmb_ix1bndry; ++m) {
        sendbuf_ix1[n].vars_req[m] = MPI_REQUEST_NULL;
        recvbuf_ix1[n].vars_req[m] = MPI_REQUEST_NULL;
      }
    }
  }
  if (nmb_ox1bndry > 0) {
    for (int n=0; n<3; ++n) {
      sendbuf_ox1[n].vars_req = new MPI_Request[nmb_ox1bndry];
      recvbuf_ox1[n].vars_req = new MPI_Request[nmb_ox1bndry];
      for (int m=0; m<nmb_ox1bndry; ++m) {
        sendbuf_ox1[n].vars_req[m] = MPI_REQUEST_NULL;
        recvbuf_ox1[n].vars_req[m] = MPI_REQUEST_NULL;
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
  if (nmb_ix1bndry > 0) {
    for (int n=0; n<3; ++n) {
      delete [] sendbuf_ix1[n].vars_req;
      delete [] recvbuf_ix1[n].vars_req;
    }
  }
  if (nmb_ox1bndry > 0) {
    for (int n=0; n<3; ++n) {
      delete [] sendbuf_ox1[n].vars_req;
      delete [] recvbuf_ox1[n].vars_req;
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
  // Note only [0] index element of buffer variables is used to send data to neighboring
  // Meshblocks. The [1] and [2] elements are therefore not allocated any memory
  if (nmb_ix1bndry > 0) {
    Kokkos::realloc(sendbuf_ix1[0].vars,nmb_ix1bndry,nvar,ncells3,ncells2,ncells1);
    Kokkos::realloc(recvbuf_ix1[0].vars,nmb_ix1bndry,nvar,ncells3,ncells2,ncells1);
  }
  if (nmb_ox1bndry > 0) {
    Kokkos::realloc(sendbuf_ox1[0].vars,nmb_ox1bndry,nvar,ncells3,ncells2,ncells1);
    Kokkos::realloc(recvbuf_ox1[0].vars,nmb_ox1bndry,nvar,ncells3,ncells2,ncells1);
  }
}

//----------------------------------------------------------------------------------------
//! \fn void ShearingBoxBoundary::PackAndSendCC()
//! \brief Apply shearing sheet BCs to cell-centered variables, including MPI
//! MPI communications. Both the inner_x1 and outer_x1 boundaries are updated.
//! Called on the physics_bcs task after purely periodic BC communication is finished.

TaskStatus ShearingBoxBoundaryCC::PackAndSendCC(DvceArray5D<Real> &a) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  auto &is = indcs.is, &ie = indcs.ie;
  auto &js = indcs.js, &je = indcs.je;
  auto &ks = indcs.ks, &ke = indcs.ke;
  auto &ng = indcs.ng;
  // copy ix1 and ox1 ghost zones into send buffer view
  // Note only variables in [0] index of buffer array are used here and throughout
  std::pair<int,int> isrc, idst;
  isrc = std::make_pair(0,ng);
  idst = std::make_pair(0,ng);
  for (int m=0; m<nmb_ix1bndry; ++m) {
    using namespace Kokkos;
    auto src = subview(a,m,ALL,ALL,ALL,isrc);
    auto dst = subview(sendbuf_ix1[0].vars,m,ALL,ALL,ALL,idst);
    deep_copy(DevExeSpace(), dst, src);
  }
  isrc = std::make_pair(ie+1,ie+1+ng);
  idst = std::make_pair(0,ng);
  for (int m=0; m<nmb_ox1bndry; ++m) {
    using namespace Kokkos;
    auto src = subview(a,m,ALL,ALL,ALL,isrc);
    auto dst = subview(sendbuf_ox1[0].vars,m,ALL,ALL,ALL,idst);
    deep_copy(DevExeSpace(), dst, src);
  }

  // figure out distance boundaries are sheared
  auto &mesh_size = pmy_pack->pmesh->mesh_size;
  Real &time = pmy_pack->pmesh->time;
  Real lx = (mesh_size.x1max - mesh_size.x1min);
  Real yshear = qshear*omega0*lx*time;
/*

  // apply fractional cell offset to data in send buffer using conservative remap
  int nvar = a.extent_int(1);  // TODO(@user): 2nd index from L of in array must be NVAR
  auto &mbsize = pmy_pack->pmb->mb_size;
  int nj = indcs.nx2 + 2*ng;
  int nmb1 = nmb_x1bndry-1;
  auto sbuf = sendcc_shr;
  int scr_lvl=0;
  size_t scr_size = ScrArray1D<Real>::shmem_size(nj) * 3;
  par_for_outer("shrcc0",DevExeSpace(),scr_size,scr_lvl,0,nmb1,0,(nvar-1),ks,ke,is,ie,
  KOKKOS_LAMBDA(TeamMember_t member, const int m, const int n, const int k, const int i) {
    ScrArray1D<Real> u0_(member.team_scratch(scr_lvl), nfx); // 1D slice of data
    ScrArray1D<Real> flx(member.team_scratch(scr_lvl), nfx); // "flux" at faces
    ScrArray1D<Real> q1_(member.team_scratch(scr_lvl), nfx); // scratch array

    // Load scratch array
    par_for_inner(member, 0, nj, [&](const int j) {
      u0_(j) = a(m,n,k,j,i);
    });
    member.team_barrier();

    // compute fractional offset
    Real eps = fmod(yshear,(mbsize.d_view(m).dx2))/(mbsize.d_view(m).dx2);
    if (x1bndry_mbs.d_view(m,0) != 0) {eps *= -1.0;}

    // Compute "fluxes" at shifted cell faces
    switch (rcon) {
      case ReconstructionMethod::dc:
        DonorCellOrbAdvFlx(member, js, (je+1), eps, u0_, q1_, flx);
        break;
      case ReconstructionMethod::plm:
        PcwsLinearOrbAdvFlx(member, js, (je+1), eps, u0_, q1_, flx);
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
      sbuf(m,n,k,j,i) -= (flx(j+1) - flx(j));
    });
  });

  // copy data into recieve buffer for any MBs on the same rank
  for (int m=0; m<nmb_x1bndry; ++m) {
    // Find integer and fractional number of grids over which offset extends.
    // This assumes every grid has same number of cells in x2-direction!
    int jshear  = static_cast<int>(yshear/(mbsize.h_view(m).dx2));
    int jsheari = jshear/mbsize.h_view(m).nx2;
    int jshearf = jshear - jsheari*mbsize.h_view(m).nx2;

    Mesh *pm = pmy_pack->pmesh;
    LogicalLocation lloc = pm->lloc_eachmb[pmy_pack->gids + m];
    std::int32_t nmbx2 = pm->nmb_rootx2 << (lloc.level - pm->root_level);
    lloc.lx2 = static_cast<std::int32_t>((lloc.lx2 + jsheari)/nmbx2);
//    MeshBlockTree *bt = pm->ptree->FindMeshBlock(lloc);
    int tgid = (pm->ptree->FindMeshBlock(lloc))->GetGID();
    int trank = pm->rank_eachmb[tgid];
    int tm = tgid - gids_eachrank[trank];

    if (jshearf < ng) {
      // send to (dm-1)
      std::pair<int,int> isrc = std::make_pair(0,ng);
      std::pair<int,int> idst = std::make_pair(0,ng);
      using Kokkos;
      auto src = subview(a,m,ALL,ALL,ALL,isrc);
      auto dst = subview(sendcc_shr.vars,tm,ALL,ALL,ALL,isrc);
      deep_copyaDevExeSpace(), dst, src)
      // send to dm

      // send to (dm+1)
    }
  }


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
