#ifndef UTILS_FLUX_GENERALIZED_HPP_
#define UTILS_FLUX_GENERALIZED_HPP_

#include <vector>

#include "outputs/outputs.hpp"
#include "utils/surface_grid.hpp"

void TorusFluxes_General(HistoryData *pdata, MeshBlockPack *pmbp,
                         const std::vector<SphericalSurfaceGrid*> &surfaces);

#endif // UTILS_FLUX_GENERALIZED_HPP_
