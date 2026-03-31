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
    AdjustExcisionForZoom();
    LoadZoomData(zstate.zone-1);
    MaskVariables();
  }
  // Initialize boundary values and primitive variables after reinitialization and masking
  if (zamr.zooming_in || zstate.zone > 0) {
    pdriver->InitBoundaryValuesAndPrimitives(pmesh);
  }
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
