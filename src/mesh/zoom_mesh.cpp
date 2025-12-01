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
#include "eos/eos.hpp"
#include "eos/ideal_c2p_hyd.hpp"
#include "eos/ideal_c2p_mhd.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
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
  // nzmb_total
  nzmb_thishost = 0;
  nzmb_thisdvce = 0;
  // TODO(@mhguo): let's use a large default value for now, may tune it later
  // TODO(@mhguo): error check if the number of zoom MBs exceeds maximum when storing variables
  nzmb_max_perdvce = pin->GetOrAddInteger("cyclic_zoom","max_nzmb_per_dvce",256);
  nzmb_max_perhost = pin->GetOrAddInteger("cyclic_zoom","max_nzmb_per_host",512);
  return;
}
