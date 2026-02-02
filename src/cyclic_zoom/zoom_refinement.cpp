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
//! \fn void CyclicZoom::StoreZoomRegion()
//! \brief Work before zooming: store zoom region

void CyclicZoom::StoreZoomRegion() {
  // zoom state has been updated in SetRefinementFlags()
  if (zamr.zooming_out) {
    StoreVariables();
    // sync rank_eachmb and lid_eachmb to all ranks
    pzmesh->SyncMBLists();
    pzmesh->SyncLogicalLocations();
    pzdata->PackBuffer();
    pzdata->SaveToStorage(zstate.zone-1);
    if (global_variable::my_rank == 0) {
      std::cout << "CyclicZoom: Stored zoom region before zooming" << std::endl;
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void CyclicZoom::ApplyZoomRegion()
//! \brief Apply zoom region: reinitialize and mask variables if needed

void CyclicZoom::ApplyZoomRegion(Driver *pdriver) {
  if (zamr.zooming_in) {
    pzmesh->FindRegion(zstate.zone);
    pzmesh->SyncMBLists();
    pzdata->LoadFromStorage(zstate.zone);
    pzdata->UnpackBuffer();
    ReinitVariables();
    if (global_variable::my_rank == 0) {
      std::cout << "CyclicZoom: Apply variables after zooming" << std::endl;
    }
  }
  // Set up mask region
  // TODO(@mhguo): data may already be on the correct device, probably try to avoid unnecessary transfer
  if (zstate.zone > 0) {
    pzmesh->FindRegion(zstate.zone-1);
    pzmesh->SyncMBLists();
    pzdata->LoadFromStorage(zstate.zone-1);
    pzdata->UnpackBuffer();
    MaskVariables();
  }
  // Initialize boundary values and primitive variables after reinitialization and masking
  pdriver->InitBoundaryValuesAndPrimitives(pmesh);
  if (zamr.zooming_out) {
    // update electric fields after masking
    if (pmesh->pmb_pack->pmhd != nullptr) {
      pzdata->ResetDataEC(pzdata->delta_efld); // clear delta_efld first
      UpdateFluxes(pdriver);
      StoreFluxes();
      // TODO(@mhguo): this is inefficient, may optimize to only transfer emf data
      pzdata->PackBuffer();
      pzdata->SaveToStorage(zstate.zone-1);
    }
  }
  // reset zooming flags
  zamr.zooming_out = false;
  zamr.zooming_in = false;
  if (global_variable::my_rank == 0) {
    std::cout << "CyclicZoom: Applied zoom region" << std::endl;
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
  // store efld_pre for future use
  if (pmesh->pmb_pack->pmhd != nullptr && zstate.zone > 1) {
    // copy pzdata->efld_pre to pzdata->efld_buf
    Kokkos::deep_copy(pzdata->efld_buf.x1e, pzdata->efld_pre.x1e);
    Kokkos::deep_copy(pzdata->efld_buf.x2e, pzdata->efld_pre.x2e);
    Kokkos::deep_copy(pzdata->efld_buf.x3e, pzdata->efld_pre.x3e);
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
      pzdata->StoreDataToZoomData(zm_count, m);
      ++zm_count;
    }
  }
  CorrectVariables(); // correct variables after zoom data update but before zoom mesh update
  // TODO(@mhguo): move some of the following code to ZoomMesh functions
  pzmesh->GatherZMB(zm_count, zstate.zone-1);
  pzmesh->UpdateMeshStructure();
  // TODO(@mhguo): may create a list of stored MBs to avoid this loop?
  // assign rank and local ID of each MB that contains the zoom MBs
  int zmbs = pzmesh->gzms_eachdvce[global_variable::my_rank];
  int zm = 0;
  for (int m=0; m<nmb; ++m) {
    if (CheckStoreFlag(m)) {
      pzmesh->rank_eachmb[zmbs + zm] = global_variable::my_rank;
      pzmesh->lid_eachmb[zmbs + zm] = m;
      // copy LogicalLocation of stored MeshBlocks
      pzmesh->lloc_eachzmb[zmbs + zm] = pmesh->lloc_eachmb[m + mbs];
      ++zm;
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void CyclicZoom::CorrectVariables()
//! \brief Correct physical variables before storing (e.g., electric fields)

void CyclicZoom::CorrectVariables() {
  if (pmesh->pmb_pack->pmhd != nullptr && zstate.zone > 1) {
    int zmbs = pzmesh->gzms_eachdvce[global_variable::my_rank];
    int zm_count = 0;
    int m0 = pzmesh->lid_eachmb[zmbs];
    for (int zm=0; zm<pzmesh->nzmb_thisdvce; ++zm) {
      int m = pzmesh->lid_eachmb[zm+zmbs];
      if (m > m0) { // now move to next MeshBlock
        m0 = m;
        ++zm_count;
      }
      // print diagnostic info
      std::cout << "CyclicZoom: Correcting variables for zoom MeshBlock " << zm_count
                << " using zoom MeshBlock " << zm
                << " on MeshBlock " << m + pmesh->gids_eachrank[global_variable::my_rank]
                << std::endl;
      // correct electric fields
      pzdata->StoreFinerEFields(zm_count, zm, pzdata->efld_buf);
    }
    if (global_variable::my_rank == 0) {
      std::cout << "CyclicZoom: Corrected variables before zooming" << std::endl;
    }
  }
  return;
}


//----------------------------------------------------------------------------------------
//! \fn void CyclicZoom::ReinitVariables()
//! \brief Reinitialize variables after zooming (in)

void CyclicZoom::ReinitVariables() {
  int zmbs = pzmesh->gzms_eachdvce[global_variable::my_rank];
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
  int zmbs = pzmesh->gzms_eachdvce[global_variable::my_rank];
  for (int zm=0; zm<pzmesh->nzmb_thisdvce; ++zm) {
    int m = pzmesh->lid_eachmb[zm+zmbs];
    pzdata->MaskDataInZoomRegion(m, zm);
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void CyclicZoom::UpdateFluxes()
//! \brief Update electric fields after masking

void CyclicZoom::UpdateFluxes(Driver *pdriver) {
  // call MHD functions to update electric fields in all MeshBlocks
  mhd::MHD *pmhd = pmesh->pmb_pack->pmhd;
  (void) pmhd->InitRecv(pdriver, 1);  // stage = 1 
  (void) pmhd->CopyCons(pdriver, 1);  // stage = 1: copy u0 to u1
  (void) pmhd->Fluxes(pdriver, 1);
  // (void) pmhd->RestrictU(this, 0);
  // TODO(@mhguo): think about the order
  // TODO(@mhguo): this is redundant, should only send/recv electric fields
  (void) pmhd->SendFlux(pdriver, 1);  // stage = 1
  (void) pmhd->RecvFlux(pdriver, 1);  // stage = 1
  (void) pmhd->SendU(pdriver, 1);
  (void) pmhd->RecvU(pdriver, 1);
  (void) pmhd->CornerE(pdriver, 1);
  (void) pmhd->EFieldSrc(pdriver, 1);
  (void) pmhd->SendE(pdriver, 1);
  (void) pmhd->RecvE(pdriver, 1);
  (void) pmhd->SendB(pdriver, 1);
  (void) pmhd->RecvB(pdriver, 1);
  (void) pmhd->ClearSend(pdriver, 1); // stage = 1
  (void) pmhd->ClearRecv(pdriver, 1); // stage = 1
  std::cout << " Rank " << global_variable::my_rank 
            << " Calculated electric fields after AMR" << std::endl;
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void CyclicZoom::StoreFluxes()
//! \brief Update electric fields after masking

void CyclicZoom::StoreFluxes() {
  // update electric fields in zoom region
  // TODO(@mhguo): only stored the emf, may need to limit de to emin/max
  int zmbs = pzmesh->gzms_eachdvce[global_variable::my_rank];
  for (int zm=0; zm<pzmesh->nzmb_thisdvce; ++zm) {
    int m = pzmesh->lid_eachmb[zm+zmbs];
    // pzdata->UpdateElectricFieldsInZoomRegion(m, zm);
    auto efld = pmesh->pmb_pack->pmhd->efld;
    pzdata->StoreEFieldsAfterAMR(zm, m, efld);
  }

  // limit electric fields if needed
  pzdata->LimitEFields();
  if (global_variable::my_rank == 0) {
    std::cout << "CyclicZoom: Updated electric fields in zoom region" << std::endl;
  }
  return;
}
