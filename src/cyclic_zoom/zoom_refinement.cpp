//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file zoom_refinement.cpp
//! \brief Functions to handle cyclic zoom mesh refinement

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "cyclic_zoom/cyclic_zoom.hpp"
#include "coordinates/cartesian_ks.hpp"
#include "coordinates/cell_locations.hpp"
#include "eos/eos.hpp"
#include "eos/ideal_c2p_hyd.hpp"
#include "eos/ideal_c2p_mhd.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
// TODO(@mhguo): check whehther all above includes are necessary

//----------------------------------------------------------------------------------------
//! \fn void CyclicZoom::WorkBeforeAMR()
//! \brief Work before zooming: store zoom region

void CyclicZoom::WorkBeforeAMR() {
  // zoom state has been updated in SetRefinementFlags()
  if (zamr.zooming_out) {
    StoreVariables();
    // sync rank_eachmb and lid_eachmb to all ranks
    pzmesh->SyncMBLists();
    pzmesh->SyncLogicalLocations();
    pzdata->PackBuffer();
    pzdata->SaveToStorage(zstate.zone-1);
    // TODO(@mhguo): remove previous functions
    // UpdateVariables();
    // // TODO(@mhguo): only store the data needed on this rank instead of holding all
    // SyncVariables();
    // UpdateGhostVariables();
    if (global_variable::my_rank == 0) {
      std::cout << "CyclicZoom: Stored variables before zooming" << std::endl;
    }
    if (global_variable::my_rank == 0) {
      std::cout << "CyclicZoom: Done Work Before AMR" << std::endl;
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void CyclicZoom::WorkAfterAMR()
//! \brief Work after zooming: initialize zoom region, store emf if needed

void CyclicZoom::WorkAfterAMR(Driver *pdriver) {
  if (zamr.zooming_in) {
    FindReinitRegion();
    pzmesh->SyncMBLists();
    pzdata->LoadFromStorage(zstate.zone);
    pzdata->UnpackBuffer();
    ReinitVariables();
    // ApplyVariables();
    if (global_variable::my_rank == 0) {
      std::cout << "CyclicZoom: Apply variables after zooming" << std::endl;
    }
  }
  // Set up mask region
  // TODO(@mhguo): data may already be on the correct device, probably try to avoid unnecessary transfer
  if (zstate.zone > 0) {
    FindMaskRegion();
    pzmesh->SyncMBLists();
    pzdata->LoadFromStorage(zstate.zone-1);
    pzdata->UnpackBuffer();
    MaskVariables();
  }
  if (zamr.zooming_out) {
    if (pmesh->pmb_pack->pmhd != nullptr) {
      UpdateElectricFields(pdriver);
      // TODO(@mhguo): this is inefficient, may optimize to only transfer emf data
      pzdata->PackBuffer();
      pzdata->SaveToStorage(zstate.zone-1);
    }
  }
  // reset zooming flags
  zamr.zooming_out = false;
  zamr.zooming_in = false;
  if (global_variable::my_rank == 0) {
    std::cout << "CyclicZoom: Done Work After AMR" << std::endl;
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void CyclicZoom::StoreVariables()
//! \brief Store variables before zooming (out)

void CyclicZoom::StoreVariables() {
  if (global_variable::my_rank == 0) {
    std::cout << "CyclicZoom: Storing variables before zooming" << std::endl;
  }
  int nmb = pmesh->pmb_pack->nmb_thispack;
  int mbs = pmesh->gids_eachrank[global_variable::my_rank];
  int zm_count = 0;
  // TODO(@mhguo): now CheckStoreFlag is called multiple times, can be optimized
  for (int m=0; m<nmb; ++m) {
    if (CheckStoreFlag(m)) {
      if (zm_count >= pzmesh->nzmb_max_perdvce) {
        std::cerr << "CyclicZoom::StoreVariables ERROR: exceed maximum number of "
                  << "stored MeshBlocks per device: " << pzmesh->nzmb_max_perdvce
                  << std::endl;
        exit(1);
      }
      ++zm_count;
    }
  }
  // TODO(@mhguo): move some of the following code to ZoomMesh functions
  pzmesh->GatherZMB(zm_count, zstate.zone-1);
  pzmesh->UpdateMeshData();
  // TODO(@mhguo): may create a list of stored MBs to avoid this loop?
  // assign rank and local ID of each MB that contains the zoom MBs
  int zmbs = pzmesh->gids_eachdvce[global_variable::my_rank];
  int zm = 0;
  for (int m=0; m<nmb; ++m) {
    if (CheckStoreFlag(m)) {
      pzmesh->rank_eachmb[zmbs + zm] = global_variable::my_rank;
      pzmesh->lid_eachmb[zmbs + zm] = m;
      // copy LogicalLocation of stored MeshBlocks
      pzmesh->lloc_eachzmb[zmbs + zm] = pmesh->lloc_eachmb[m + mbs];
      pzdata->StoreDataToZoomData(zm, m);
      ++zm;
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void CyclicZoom::ReinitVariables()
//! \brief Reinitialize variables after zooming (in)

void CyclicZoom::ReinitVariables() {
  int zmbs = pzmesh->gids_eachdvce[global_variable::my_rank];
  for (int zm=0; zm<pzmesh->nzmb_thisdvce; ++zm) {
    int m = pzmesh->lid_eachmb[zm+zmbs];
    std::cout << "  Rank " << global_variable::my_rank
              << " Reinitializing MeshBlock " << m + pmesh->gids_eachrank[global_variable::my_rank]
              << " using zoom MeshBlock " << zm + zmbs
              << std::endl;
    pzdata->LoadDataFromZoomData(m, zm);
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void CyclicZoom::MaskVariables()
//! \brief Mask variables in the zoom region

void CyclicZoom::MaskVariables() {
  int zmbs = pzmesh->gids_eachdvce[global_variable::my_rank];
  for (int zm=0; zm<pzmesh->nzmb_thisdvce; ++zm) {
    int m = pzmesh->lid_eachmb[zm+zmbs];
    pzdata->MaskDataInZoomRegion(m, zm);
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void CyclicZoom::UpdateElectricFields()
//! \brief Update electric fields after masking

void CyclicZoom::UpdateElectricFields(Driver *pdriver) {
  // step 1: call MHD functions to update electric fields in all MeshBlocks
  // clear delta_efld first
  pzdata->ResetDataEC(pzdata->delta_efld);
  // call MHD functions
  mhd::MHD *pmhd = pmesh->pmb_pack->pmhd;
  (void) pmhd->InitRecv(pdriver, 0);  // stage = 0 
  (void) pmhd->CopyCons(pdriver, 1);  // stage = 1: copy u0 to u1
  (void) pmhd->Fluxes(pdriver, 0);
  // (void) pmhd->RestrictU(this, 0);
  // TODO(@mhguo): think about the order
  // TODO(@mhguo): this is redundant, should only send/recv electric fields
  (void) pmhd->SendFlux(pdriver, 0);  // stage = 0
  (void) pmhd->RecvFlux(pdriver, 0);  // stage = 0
  (void) pmhd->SendU(pdriver, 0);
  (void) pmhd->RecvU(pdriver, 0);
  (void) pmhd->CornerE(pdriver, 0);
  (void) pmhd->EFieldSrc(pdriver, 0);
  (void) pmhd->SendE(pdriver, 0);
  (void) pmhd->RecvE(pdriver, 0);
  (void) pmhd->SendB(pdriver, 0);
  (void) pmhd->RecvB(pdriver, 0);
  (void) pmhd->ClearSend(pdriver, 0); // stage = 0
  (void) pmhd->ClearRecv(pdriver, 0); // stage = 0
  std::cout << " Rank " << global_variable::my_rank 
            << " Calculated electric fields after AMR" << std::endl;

  // step 2: update electric fields in zoom region
  // TODO(@mhguo): only stored the emf, may need to limit de to emin/max
  int zmbs = pzmesh->gids_eachdvce[global_variable::my_rank];
  for (int zm=0; zm<pzmesh->nzmb_thisdvce; ++zm) {
    int m = pzmesh->lid_eachmb[zm+zmbs];
    // pzdata->UpdateElectricFieldsInZoomRegion(m, zm);
    auto efld = pmesh->pmb_pack->pmhd->efld;
    pzdata->StoreEFieldsAfterAMR(zm, m, efld);
  }
  pzdata->LimitEFields();
  if (global_variable::my_rank == 0) {
    std::cout << "CyclicZoom: Updated electric fields in zoom region" << std::endl;
  }
  return;
}


//----------------------------------------------------------------------------------------
//! \fn bool CyclicZoom::CheckStoreFlag(int m)
//! \brief Check whether to store variables for MeshBlock with given global ID

bool CyclicZoom::CheckStoreFlag(int m) {
  auto &size = pmesh->pmb_pack->pmb->mb_size;
  int mbs = pmesh->gids_eachrank[global_variable::my_rank];
  // note that now the zoom state has been updated
  // use the updated zoom region parameters
  Real r_zoom = zregion.radius;
  Real x1c = zregion.x1c, x2c = zregion.x2c, x3c = zregion.x3c;
  // check previous level (finer level)
  if (pmesh->lloc_eachmb[m+mbs].level == zamr.level + 1) {
    // extract bounds of MeshBlock
    Real x1min = size.h_view(m).x1min;
    Real x1max = size.h_view(m).x1max;
    Real x2min = size.h_view(m).x2min;
    Real x2max = size.h_view(m).x2max;
    Real x3min = size.h_view(m).x3min;
    Real x3max = size.h_view(m).x3max;
    // Find the closest point on the box to the sphere center
    Real closest_x1 = fmax(x1min, fmin(x1c, x1max));
    Real closest_x2 = fmax(x2min, fmin(x2c, x2max));
    Real closest_x3 = fmax(x3min, fmin(x3c, x3max));
    // Calculate the distance from sphere center to this closest point
    Real r_sq = SQR(x1c - closest_x1) + SQR(x2c - closest_x2) + SQR(x3c - closest_x3);
    if (r_sq < SQR(r_zoom)) {
      return true;
    }
  }
  return false;
}

//----------------------------------------------------------------------------------------
//! \fn void CyclicZoom::FindMaskRegion()
//! \brief Find meshblocks to be masked

void CyclicZoom::FindMaskRegion() {
  int nmb = pmesh->pmb_pack->nmb_thispack;
  int mbs = pmesh->gids_eachrank[global_variable::my_rank];
  int nlmb = pzmesh->nzmb_eachlevel[zstate.zone-1]; // number of zoom MBs on previous level
  int lmbs = pzmesh->gids_eachlevel[zstate.zone-1]; // starting gid of zoom MBs on previous level
  // note that now the zoom state has been updated
  // use the updated zoom region parameters
  int zm_count = 0;
  for (int lm=0; lm<nlmb; ++lm) {
    int m = FindMaskMB(lm);
    if (m >= 0) {
      // now map the zoom MB to this MB for masking
      pzmesh->rank_eachmb[lm+lmbs] = global_variable::my_rank;
      pzmesh->lid_eachmb[lm+lmbs] = m;
      std::cout << "  Rank " << global_variable::my_rank
                << " Masking MeshBlock " << m+mbs << " for zoom MeshBlock "
                << lm+lmbs << std::endl;
      ++zm_count;
    }
  }
  std::cout << "  Rank " << global_variable::my_rank << " total zoom MBs to be masked: "
            << zm_count << std::endl;
  // TODO(@mhguo): you probably don't need to sync, as lloc_eachmb includes all MBs
  pzmesh->GatherZMB(zm_count, zstate.zone-1);
  int lm_total = 0;
  for (int i = 0; i < global_variable::nranks; ++i) {
    lm_total += pzmesh->nzmb_eachdvce[i];
  }
  if (lm_total != pzmesh->nzmb_eachlevel[zstate.zone-1]) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "CyclicZoom::GatherZMB(): inconsistent total number of zoom MeshBlocks "
              << "across all ranks: found " << lm_total << " vs. stored "
              << pzmesh->nzmb_eachlevel[zstate.zone-1] << std::endl;
    std::exit(EXIT_FAILURE);
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void CyclicZoom::FindReinitRegion()
//! \brief Find meshblocks to be re-initialized

// TODO(@mhguo): similar to FindMaskRegion, can be optimized later, also may rename
void CyclicZoom::FindReinitRegion() {
  int nmb = pmesh->pmb_pack->nmb_thispack;
  int mbs = pmesh->gids_eachrank[global_variable::my_rank];
  int nlmb = pzmesh->nzmb_eachlevel[zstate.zone]; // number of zoom MBs on previous level
  int lmbs = pzmesh->gids_eachlevel[zstate.zone]; // starting gid of zoom MBs on previous level
  // note that now the zoom state has been updated
  // use the updated zoom region parameters
  int zm_count = 0;
  for (int lm=0; lm<nlmb; ++lm) {
    int m = FindReinitMB(lm);
    if (m >= 0) {
      pzmesh->rank_eachmb[lm+lmbs] = global_variable::my_rank;
      pzmesh->lid_eachmb[lm+lmbs] = m;
      std::cout << "  Rank " << global_variable::my_rank
                << " Reinit MeshBlock " << m+mbs << " using zoom MeshBlock "
                << lm+lmbs << std::endl;
      ++zm_count;
    }
  }
  std::cout << "  Rank " << global_variable::my_rank << " total zoom MBs to be applied: "
            << zm_count << std::endl;
  // TODO(@mhguo): you probably don't need to sync, as lloc_eachmb includes all MBs
  pzmesh->GatherZMB(zm_count, zstate.zone);
  int lm_total = 0;
  for (int i = 0; i < global_variable::nranks; ++i) {
    lm_total += pzmesh->nzmb_eachdvce[i];
  }
  if (lm_total != pzmesh->nzmb_eachlevel[zstate.zone]) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "CyclicZoom::GatherZMB(): inconsistent total number of zoom MeshBlocks "
              << "across all ranks: found " << lm_total << " vs. stored "
              << pzmesh->nzmb_eachlevel[zstate.zone] << std::endl;
    std::exit(EXIT_FAILURE);
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn int CyclicZoom::FindMaskMB()
//! \brief Check which meshblocks need to be masked

int CyclicZoom::FindMaskMB(int lm) {
  int nmb = pmesh->pmb_pack->nmb_thispack;
  int mbs = pmesh->gids_eachrank[global_variable::my_rank];
  int lmbs = pzmesh->gids_eachlevel[zstate.zone-1]; // starting gid of zoom MBs on previous level
  auto &zlloc = pzmesh->lloc_eachzmb[lm+lmbs];
  // check previous level (finer level)
  for (int m = 0; m < nmb; ++m) {
    auto &lloc = pmesh->lloc_eachmb[m+mbs];
    // if (lloc.level == zamr.level) {
    // if zoom MB is child of this MB
    if ( (zlloc.level == lloc.level + 1) && (zlloc.lx1>>1 == lloc.lx1) &&
          (zlloc.lx2>>1 == lloc.lx2) && (zlloc.lx3>>1 == lloc.lx3) ) {
      return m;
    }
    // }
  }
  return -1;
}

//----------------------------------------------------------------------------------------
//! \fn int CyclicZoom::FindReinitMB()
//! \brief Check which meshblocks need to be applied

int CyclicZoom::FindReinitMB(int lm) {
  int nmb = pmesh->pmb_pack->nmb_thispack;
  int mbs = pmesh->gids_eachrank[global_variable::my_rank];
  int lmbs = pzmesh->gids_eachlevel[zstate.zone]; // starting gid of zoom MBs on this level
  auto &zlloc = pzmesh->lloc_eachzmb[lm+lmbs];
  // check current level (zoom MBs at same level as current AMR level)
  for (int m = 0; m < nmb; ++m) {
    auto &lloc = pmesh->lloc_eachmb[m+mbs];
    // if (lloc.level == zamr.level) {
    // if zoom MB is the same as this MB
    if ( (zlloc.level == lloc.level) && (zlloc.lx1 == lloc.lx1) &&
          (zlloc.lx2 == lloc.lx2) && (zlloc.lx3 == lloc.lx3) ) {
      return m;
    }
    // }
  }
  return -1;
}
