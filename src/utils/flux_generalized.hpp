// flux_generalized.hpp

#pragma once

#include <vector>
#include "athena.hpp" // For Real

// Forward declarations
class HistoryData;
class MeshBlockPack;
class SphericalSurfaceGrid; // The function uses this type

// --- Function Declaration ---
// The default argument = nullptr is ONLY specified here.
// The signature now takes the MeshBlockPack and the vector of surfaces directly.
void TorusFluxes_General(HistoryData *pdata,
                         MeshBlockPack *pmbp,
                         const std::vector<SphericalSurfaceGrid*>& surfs,
                         const Real axis_n[3] = nullptr);