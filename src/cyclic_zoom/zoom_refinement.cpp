//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file zoom_refinement.cpp
//! \brief Functions to handle cyclic zoom mesh refinement

#include <iostream>

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "cyclic_zoom/cyclic_zoom.hpp"

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
    if (verbose && global_variable::my_rank == 0) {
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
    if (verbose && global_variable::my_rank == 0) {
      std::cout << "CyclicZoom: Apply variables after zooming" << std::endl;
    }
  }
  // Set up mask region
  // TODO(@mhguo): data may already be on the correct device,
  // TODO(@mhguo): may try to avoid unnecessary transfer later
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
  if (verbose && global_variable::my_rank == 0) {
    std::cout << "CyclicZoom: Applied zoom region" << std::endl;
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void CyclicZoom::StoreVariables()
//! \brief Store variables before zooming (out)

void CyclicZoom::StoreVariables() {
  if (verbose && global_variable::my_rank == 0) {
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
      pzdata->StoreData(zm_count, m);
      ++zm_count;
    }
  }
  // correct variables after zoom data update but before zoom mesh update
  CorrectVariables();
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
    // TODO(@mhguo): think whether this is ok if with multiple levels
    for (int zm = 0; zm < pzmesh->nzmb_thisdvce; ++zm) {
      int m = pzmesh->lid_eachmb[zm+zmbs];
      if (m > m0) { // now move to next MeshBlock
        m0 = m;
        ++zm_count;
      }
      // print diagnostic info
      if (verbose) {
        std::cout << "CyclicZoom: Correcting variables for zoom MeshBlock " << zm_count
                  << " using zoom MeshBlock " << zm + zmbs << " on MeshBlock "
                  << m + pmesh->gids_eachrank[global_variable::my_rank]
                  << " on rank " << global_variable::my_rank
                  << std::endl;
      }
      // correct electric fields
      pzdata->StoreEFieldsFromFiner(zm_count, zm, pzdata->efld_buf);
    }
    if (verbose && global_variable::my_rank == 0) {
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
  int mbs = pmesh->gids_eachrank[global_variable::my_rank];
  if (verbose && global_variable::my_rank == 0) {
    std::cout << " Apply zoom region radius: " << old_zregion.radius << std::endl;
  }
  for (int zm = 0; zm < pzmesh->nzmb_thisdvce; ++zm) {
    auto &zlloc = pzmesh->lloc_eachzmb[zm+zmbs];
    int m = pzmesh->lid_eachmb[zm+zmbs];
    auto &lloc = pmesh->lloc_eachmb[m+mbs];
    if (verbose && global_variable::my_rank == 0) {
      std::cout << "  Rank " << global_variable::my_rank
                << " Reinitializing MeshBlock " << m + mbs
                << " using zoom MeshBlock " << zm + zmbs
                << std::endl;
    }
    // Reinitialize variables in the zoom region using the same level zoom data
    if (zlloc.level == lloc.level) {
      pzdata->ApplyDataSameLevel(m, zm, old_zregion);
    } else {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << "zoom meshblock level is different from MeshBlock level"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void CyclicZoom::MaskVariables()
//! \brief Mask variables in the zoom region

void CyclicZoom::MaskVariables() {
  int zmbs = pzmesh->gzms_eachdvce[global_variable::my_rank];
  int mbs = pmesh->gids_eachrank[global_variable::my_rank];
  for (int zm = 0; zm < pzmesh->nzmb_thisdvce; ++zm) {
    auto &zlloc = pzmesh->lloc_eachzmb[zm+zmbs];
    int m = pzmesh->lid_eachmb[zm+zmbs];
    auto &lloc = pmesh->lloc_eachmb[m+mbs];
    if (zlloc.level == lloc.level) {
      pzdata->ApplyDataSameLevel(m, zm, zregion);
    } else if (zlloc.level - lloc.level == 1) {
      pzdata->ApplyDataFromFiner(m, zm, zregion);
    } else {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << "zoom MeshBlock is more than 1 level finer than MeshBlock level"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void CyclicZoom::ApplyMask()
//! \brief wrapper function to call MaskVariables()

void CyclicZoom::ApplyMask() {
  if (zstate.zone > 0 && !zamr.zooming_out && !zamr.zooming_in) {
    MaskVariables();
  }
  return;
}
