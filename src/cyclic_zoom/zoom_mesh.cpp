//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file zoom_mesh.cpp
//  \brief implementation of constructor and functions in CyclicZoom class

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "cyclic_zoom/cyclic_zoom.hpp"

//----------------------------------------------------------------------------------------
// constructor, initializes data structures and parameters

ZoomMesh::ZoomMesh(CyclicZoom *pz, ParameterInput *pin) :
    pzoom(pz)
  {
  max_level = pzoom->zamr.max_level;
  min_level = pzoom->zamr.min_level;
  nlevels = max_level - min_level + 1;
  nleaf = 2;
  if (pzoom->pmesh->two_d) nleaf = 4;
  if (pzoom->pmesh->three_d) nleaf = 8;
  // mzoom = nleaf*(max_level - min_level + 1);
  nzmb_total = 0;
  nzmb_thisdvce = 0;
  nzmb_thishost = 0;
  // TODO(@mhguo): let's use a large default value for now, may tune it later
  // TODO(@mhguo): error check if the number of zoom MBs exceeds maximum when storing variables
  nzmb_max_perdvce = pin->GetOrAddInteger(pzoom->block_name,"max_nzmb_per_dvce",
                     pzoom->pmesh->nmb_maxperrank);
  nzmb_max_perhost = pin->GetOrAddInteger(pzoom->block_name,"max_nzmb_per_host",
                     pzoom->pmesh->nmb_maxperrank);
  gids_eachlevel = new int[nlevels]();
  nzmb_eachlevel = new int[nlevels]();
  gids_eachdvce = new int[global_variable::nranks]();
  nzmb_eachdvce = new int[global_variable::nranks]();
  rank_eachzmb.resize(1);
  lid_eachzmb.resize(1);
  rank_eachmb.resize(1);
  lid_eachmb.resize(1);
  lloc_eachzmb.resize(1);
  return;
}

//----------------------------------------------------------------------------------------
// destructor

ZoomMesh::~ZoomMesh() {
  delete[] gids_eachlevel;
  delete[] nzmb_eachlevel;
  delete[] gids_eachdvce;
  delete[] nzmb_eachdvce;
}

//----------------------------------------------------------------------------------------
//! \fn void ZoomMesh::GatherZMB()
//! \brief Sync meta data of zoom MeshBlocks across all ranks

void ZoomMesh::GatherZMB(int zm_count, int zone) {
  // Get total number across all ranks (inclusive scan or Allreduce)
  // int zm_total = zm_count;
  nzmb_thisdvce = zm_count;
#if MPI_PARALLEL_ENABLED
  // Gather counts and displacements
  MPI_Allgather(&nzmb_thisdvce, 1, MPI_INT, nzmb_eachdvce, 1, MPI_INT, MPI_COMM_WORLD);
  // MPI_Exscan(&zm, &zm_offset, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  // sum up number of stored MeshBlocks over all ranks
  // MPI_Allreduce(MPI_IN_PLACE, &zm_total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
#endif
  // assign starting global ID for each rank/device
  gids_eachdvce[0] = gids_eachlevel[zone];
  for (int i = 1; i < global_variable::nranks; ++i) {
    gids_eachdvce[i] = gids_eachdvce[i-1] + nzmb_eachdvce[i-1];
  }
  return;
}
//----------------------------------------------------------------------------------------
//! \fn void ZoomMesh::UpdateMeshData()
//! \brief Assign meta data of zoom MeshBlocks across all ranks

void ZoomMesh::UpdateMeshData() {
  // Get starting global index for this rank (exclusive scan)
  // int zm_offset = 0;
  // Get total number across all ranks (inclusive scan or Allreduce)
  int nzmb_thislevel = 0;
  for (int i = 0; i < global_variable::nranks; ++i) {
    nzmb_thislevel += nzmb_eachdvce[i];
  }
  nzmb_total = gids_eachlevel[pzoom->zstate.zone-1] + nzmb_thislevel;
  if (nzmb_total > nzmb_max_perhost * global_variable::nranks) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "CyclicZoom:: Total number of zoom MeshBlocks exceed maximum "
              << "allowed on host: " << nzmb_total << " > "
              << nzmb_max_perhost * global_variable::nranks << std::endl;
    std::exit(EXIT_FAILURE);
  }
  nzmb_eachlevel[pzoom->zstate.zone-1] = nzmb_thislevel;
  gids_eachlevel[pzoom->zstate.zone] = nzmb_total;
  // gids_eachdvce[global_variable::my_rank] = gids_eachlevel[zstate.zone-1] + zm_offset;
  // resize arrays to hold all zoom MeshBlocks across all levels/ranks
  rank_eachzmb.resize(nzmb_total);
  lid_eachzmb.resize(nzmb_total);
  rank_eachmb.resize(nzmb_total);
  lid_eachmb.resize(nzmb_total);
  lloc_eachzmb.resize(nzmb_total);
  // simultaneously assign rank and local ID of each zoom MB on host
  // using round-robin for load balancing
  int lmbs = gids_eachlevel[pzoom->zstate.zone-1];
  for (int lm=0; lm<nzmb_eachlevel[pzoom->zstate.zone-1]; ++lm) {
    rank_eachzmb[lm+lmbs] = (lm+lmbs) % global_variable::nranks;
    lid_eachzmb[lm+lmbs] = (lm+lmbs) / global_variable::nranks;
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void ZoomMesh::SyncMBLists()
//! \brief Sync rank_eachmb and lid_eachmb across all ranks

void ZoomMesh::SyncMBLists() {
#if MPI_PARALLEL_ENABLED
  // Gather rank_eachmb
  MPI_Allgatherv(MPI_IN_PLACE, nzmb_thisdvce, MPI_INT, rank_eachmb.data(),
                 nzmb_eachdvce, gids_eachdvce, MPI_INT, MPI_COMM_WORLD);
  // Gather lid_eachmb
  MPI_Allgatherv(MPI_IN_PLACE, nzmb_thisdvce, MPI_INT, lid_eachmb.data(),
                 nzmb_eachdvce, gids_eachdvce, MPI_INT, MPI_COMM_WORLD);
#endif
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void ZoomMesh::SyncLogicalLocations()
//! \brief Sync lloc_eachzmb across all ranks

void ZoomMesh::SyncLogicalLocations() {
#if MPI_PARALLEL_ENABLED
  // Create MPI datatype for LogicalLocation
  MPI_Datatype lloc_type;
  MPI_Type_contiguous(4, MPI_INT32_T, &lloc_type);
  MPI_Type_commit(&lloc_type);
  // Gather lloc_eachzmb (using the custom datatype, no byte conversion needed)
  MPI_Allgatherv(MPI_IN_PLACE, nzmb_thisdvce, lloc_type, lloc_eachzmb.data(),
                 nzmb_eachdvce, gids_eachdvce, lloc_type, MPI_COMM_WORLD);
  MPI_Type_free(&lloc_type);
#endif
  return;
}

//----------------------------------------------------------------------------------------
//! \fn int ZoomMesh::FindMB()
//! \brief find the meshblock that cover this zoom meshblock

int ZoomMesh::FindMB(int gzm) {
  int nmb = pzoom->pmesh->pmb_pack->nmb_thispack;
  int mbs = pzoom->pmesh->gids_eachrank[global_variable::my_rank];
  auto &zlloc = lloc_eachzmb[gzm];
  // check current level (zoom MBs at same level as current AMR level)
  for (int m = 0; m < nmb; ++m) {
    auto &lloc = pzoom->pmesh->lloc_eachmb[m+mbs];
    int level_diff = zlloc.level - lloc.level;
    // if zoom MB is the same as or a child of this MB
    if (level_diff >= 0) {
      if ( (zlloc.lx1 >> level_diff == lloc.lx1) &&
          (zlloc.lx2 >> level_diff == lloc.lx2) &&
          (zlloc.lx3 >> level_diff == lloc.lx3) ) {
        return m;
      }
    }
    // }
  }
  return -1;
}

//----------------------------------------------------------------------------------------
//! \fn void ZoomMesh::FindRegion()
//! \brief Find meshblocks for certain zone

void ZoomMesh::FindRegion(int zone) {
  int nmb = pzoom->pmesh->pmb_pack->nmb_thispack;
  int mbs = pzoom->pmesh->gids_eachrank[global_variable::my_rank];
  int nlmb = nzmb_eachlevel[zone]; // number of zoom MBs on previous level
  int lmbs = gids_eachlevel[zone]; // starting gid of zoom MBs on previous level
  // note that now the zoom state has been updated
  // use the updated zoom region parameters
  int zm_count = 0;
  for (int lm=0; lm<nlmb; ++lm) {
    int m = FindMB(lm+lmbs);
    if (m >= 0) {
      rank_eachmb[lm+lmbs] = global_variable::my_rank;
      lid_eachmb[lm+lmbs] = m;
      std::cout << "  Rank " << global_variable::my_rank
                << " Find MeshBlock " << m+mbs << " for zoom MeshBlock "
                << lm+lmbs << std::endl;
      ++zm_count;
    }
  }
  std::cout << "  Rank " << global_variable::my_rank << " total zoom MBs to be applied: "
            << zm_count << std::endl;
  // TODO(@mhguo): you probably don't need to sync, as lloc_eachmb includes all MBs
  // TODO(@mhguo): you can loop over all meshblocks though it may be slower
  GatherZMB(zm_count, zone);
  int lm_total = 0;
  for (int i = 0; i < global_variable::nranks; ++i) {
    lm_total += nzmb_eachdvce[i];
  }
  if (lm_total != nlmb) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "CyclicZoom::GatherZMB(): inconsistent total number of zoom MeshBlocks "
              << "across all ranks: found " << lm_total << " vs. stored "
              << nlmb << std::endl;
    std::exit(EXIT_FAILURE);
  }
  return;
}