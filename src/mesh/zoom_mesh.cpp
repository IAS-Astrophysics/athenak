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
#include "mesh/cyclic_zoom.hpp"

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
  gids_eachrank = new int[global_variable::nranks]();
  nzmb_eachrank = new int[global_variable::nranks]();
  // TODO(@mhguo): set these dynamically
  rank_eachzmb.resize(1);
  lid_eachzmb.resize(1);
  rank_eachmb.resize(1);
  lid_eachmb.resize(1);
  lloc_eachzmb.resize(1);
  return;
}
