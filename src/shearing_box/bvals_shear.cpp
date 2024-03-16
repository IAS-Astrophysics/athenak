//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file bvals_shear.cpp
//! \brief functions to apply shearing sheet BCs in both 2D (r_phi) and 3D coordinates.
//! This requires pack/send and recv/unpack steps for ix1 and ox1 ghost zones with MPI.

#include <cstdlib>
#include <iostream>
#include <utility>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/cell_locations.hpp"
#include "shearing_box.hpp"
#include "mhd/mhd.hpp"
#include "remap_fluxes.hpp"

//----------------------------------------------------------------------------------------
//! \fn void ShearingBox::ShearPeriodic_CC()
//! \brief Apply shearing sheet BCs to cell-centered variables, including MPI
//! MPI communications. Both the inner_x1 and outer_x1 boundaries are updated.
//! Called on the physics_bcs task after purely periodic BC communication is finished.

TaskStatus ShearingBox::ShearPeriodic_CC(DvceArray5D<Real> &a) {
/*
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  auto &is = indcs.is, &ie = indcs.ie;
  auto &js = indcs.js, &je = indcs.je;
  auto &ks = indcs.ks, &ke = indcs.ke;
  auto &ng = indcs.ng;
  // copy ix1 and ox1 ghost zones into send buffer view
  for (int m=0; m<nmb_x1bndry; ++m) {
    std::pair<int,int> isrc, idst;
    if (x1bndry_mbs.h_view(m,0)==0) {
      isrc = std::make_pair(0,ng);
      idst = std::make_pair(0,ng);
    } else {
      isrc = std::make_pair(ie+1,ie+1+ng);
      idst = std::make_pair(0,ng);
    }
    using namespace Kokkos;
    auto src = subview(a,m,ALL,ALL,ALL,isrc);
    auto dst = subview(sendcc_shr.vars,m,ALL,ALL,ALL,isrc);
    deep_copy(DevExeSpace(), dst, src);
  }

  // figure out distance boundaries are sheared
  auto &mesh_size = pmy_pack->pmesh->mesh_size;
  Real &time = pmy_pack->pmesh->time;
  Real lx = (mesh_size.x1max - mesh_size.x1min);
  Real yshear = qshear*omega0*lx*time;

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
