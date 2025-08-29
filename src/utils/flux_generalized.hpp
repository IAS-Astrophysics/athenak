#ifndef UTILS_FLUX_GENERALIZED_HPP_
#define UTILS_FLUX_GENERALIZED_HPP_
//========================================================================================
// flux_generalized.hpp
//
// DECLARATION FOR THE GENERALIZED FLUX INTEGRATION FUNCTION
//========================================================================================

#include <vector>
#include "athena.hpp"
#include "outputs/outputs.hpp" // <--- FIX: Added this include for HistoryData
#include "utils/surface_grid.hpp"

// Forward declaration of the main integration function
void TorusFluxes_General(HistoryData *pdata,
                         MeshBlockPack *pmbp,
                         const std::vector<SphericalSurfaceGrid*>& surfs,
                         const Real axis_n[3] = nullptr);

#endif // UTILS_FLUX_GENERALIZED_HPP_