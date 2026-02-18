//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file zoom_refinement.cpp
//! \brief Functions to handle cyclic zoom mesh refinement logic for zoom region

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
    int zm_count = pzmesh->CountMBsToStore(zstate.zone-1);
    StoreVariables();
    // correct variables after zoom data update but before zoom mesh update
    CorrectVariables();
    pzmesh->GatherNZMB(zm_count, zstate.zone-1);
    pzmesh->UpdateMeshStructure();
    pzmesh->AssignMBLists();
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
    LoadZoomData(zstate.zone);
    ReinitVariables();
    if (verbose && global_variable::my_rank == 0) {
      std::cout << "CyclicZoom: Apply variables after zooming" << std::endl;
    }
  }
  // Set up mask region
  if (zstate.zone > 0) {
    LoadZoomData(zstate.zone-1);
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
//! \fn void CyclicZoom::LoadZoomData()
//! \brief Load zoom data from storage for a given zone

void CyclicZoom::LoadZoomData(int zone) {
  pzmesh->FindRegion(zone);
  pzmesh->SyncMBLists();
  pzdata->LoadFromStorage(zone);
  pzdata->UnpackBuffer();
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
  for (int m=0; m<nmb; ++m) {
    if (pzmesh->zm_eachmb[m] >= 0) {
      pzdata->StoreData(pzmesh->zm_eachmb[m], m);
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void CyclicZoom::CorrectVariables()
//! \brief Correct physical variables before storing (e.g., electric fields)

void CyclicZoom::CorrectVariables() {
  // now zoom data is updated, but zoom mesh is still old with data stored in the buffer
  // so we correct the zoom data using the data buffer from the finer zoom data
  if (pmesh->pmb_pack->pmhd != nullptr && zstate.zone > 1) {
    int zmbs = pzmesh->gzms_eachdvce[global_variable::my_rank];
    for (int zmf = 0; zmf < pzmesh->nzmb_thisdvce; ++zmf) {
      int m = pzmesh->mblid_eachzmb[zmf+zmbs];
      int zmc = pzmesh->zm_eachmb[m];
      // print diagnostic info
      if (verbose) {
        std::cout << "CyclicZoom: Correcting variables for zoom MeshBlock " << zmc
                  << " using zoom MeshBlock " << zmf + zmbs << " on MeshBlock "
                  << m + pmesh->gids_eachrank[global_variable::my_rank]
                  << " on rank " << global_variable::my_rank
                  << std::endl;
      }
      // correct electric fields
      pzdata->StoreEFieldsFromFiner(zmc, zmf, pzdata->efld_buf);
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
    int m = pzmesh->mblid_eachzmb[zm+zmbs];
    auto &lloc = pmesh->lloc_eachmb[m+mbs];
    if (verbose) {
      std::cout << " Rank " << global_variable::my_rank
                << " Reinitializing MeshBlock " << m + mbs
                << " using zoom MeshBlock " << zm + zmbs
                << std::endl;
    }
    // Reinitialize variables in the zoom region using the same level zoom data
    if (zlloc.level == lloc.level) {
      pzdata->ApplyDataSameLevel(m, zm, old_zregion);
    } else {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "zoom meshblock level " << zlloc.level
                << " is different from MeshBlock level " << lloc.level
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
    int m = pzmesh->mblid_eachzmb[zm+zmbs];
    auto &lloc = pmesh->lloc_eachmb[m+mbs];
    if (zlloc.level == lloc.level) {
      pzdata->ApplyDataSameLevel(m, zm, zregion);
    } else if (zlloc.level - lloc.level == 1) {
      pzdata->ApplyDataFromFiner(m, zm, zregion);
    } else {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "zoom MeshBlock level " << zlloc.level
                << " is more than 1 level finer than MeshBlock level " << lloc.level
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
