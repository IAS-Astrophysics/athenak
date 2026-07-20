//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file flux_generalized.hpp
//! \brief Declares generalized flux integration.

#ifndef UTILS_FLUX_GENERALIZED_HPP_
#define UTILS_FLUX_GENERALIZED_HPP_

#include <vector>
#include "athena.hpp"
#include "outputs/outputs.hpp"
#include "utils/surface_grid.hpp"

// Forward declaration of the main integration function
void TorusFluxes_General(HistoryData *pdata,
                         MeshBlockPack *pmbp,
                         const std::vector<SphericalSurfaceGrid*>& surfs);

#endif  // UTILS_FLUX_GENERALIZED_HPP_
