#include "flux_generalized.hpp"

#include <cstdlib>
#include <iostream>

#include "coordinates/adm.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"
#include "mesh/mesh.hpp"
#include "mhd/mhd.hpp"

namespace {

struct SurfaceFluxSum {
  Real mdot = 0.0;
  Real area = 0.0;

  KOKKOS_INLINE_FUNCTION SurfaceFluxSum &operator+=(const SurfaceFluxSum &other) {
    mdot += other.mdot;
    area += other.area;
    return *this;
  }
};

} // namespace

namespace Kokkos {
template<>
struct reduction_identity<SurfaceFluxSum> {
  KOKKOS_INLINE_FUNCTION static SurfaceFluxSum sum() { return SurfaceFluxSum{}; }
};
} // namespace Kokkos

void TorusFluxes_General(HistoryData *pdata, MeshBlockPack *pmbp,
                         const std::vector<SphericalSurfaceGrid*> &surfaces) {
  if (pmbp == nullptr || pmbp->padm == nullptr || pmbp->pmhd == nullptr ||
      pmbp->pdyngr == nullptr) {
    std::cerr << "### FATAL: dynbbh surface fluxes require ADM, MHD, and DynGRMHD"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  constexpr int values_per_surface = 2;
  pdata->nhist = values_per_surface*static_cast<int>(surfaces.size());
  if (pdata->nhist > NHISTORY_VARIABLES) {
    std::cerr << "### FATAL: too many dynbbh surface history variables" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  for (std::size_t n = 0; n < surfaces.size(); ++n) {
    pdata->label[values_per_surface*n] = "mdot_" + surfaces[n]->Label();
    pdata->label[values_per_surface*n + 1] = "area_" + surfaces[n]->Label();
  }

  for (std::size_t n = 0; n < surfaces.size(); ++n) {
    SphericalSurfaceGrid *surface = surfaces[n];
    if (pmbp->pmesh->adaptive) surface->RebuildAll();
    const int nvars = pmbp->pmhd->nmhd + pmbp->pmhd->nscalars;
    DualArray2D<Real> primitive = surface->InterpolateToSurface(
        pmbp->pmhd->w0, 0, nvars);
    DualArray2D<Real> gauge = surface->InterpolateToSurface(
        pmbp->padm->u_adm, adm::ADM::I_ADM_ALPHA,
        adm::ADM::I_ADM_BETAZ + 1);
    surface->InterpolateMetric();
    DualArray2D<Real> dsigma;
    surface->BuildSurfaceCovectors(dsigma);

    auto primitive_d = primitive.d_view;
    auto gauge_d = gauge.d_view;
    auto metric_d = surface->Metric().d_view;
    auto indices_d = surface->InterpIndices().d_view;
    auto area_d = surface->ProperAreaElement().d_view;
    auto dsigma_d = dsigma.d_view;
    SurfaceFluxSum result;
    Kokkos::parallel_reduce(
        "dynbbh_surface_mass_flux",
        Kokkos::RangePolicy<DevExeSpace>(0, surface->Npts()),
        KOKKOS_LAMBDA(const int p, SurfaceFluxSum &sum) {
          if (indices_d(p, 0) < 0) return;
          const Real q1 = primitive_d(p, IVX);
          const Real q2 = primitive_d(p, IVY);
          const Real q3 = primitive_d(p, IVZ);
          const Real norm2 = metric_d(p, 0)*q1*q1 + metric_d(p, 3)*q2*q2 +
              metric_d(p, 5)*q3*q3 + 2.0*(metric_d(p, 1)*q1*q2 +
              metric_d(p, 2)*q1*q3 + metric_d(p, 4)*q2*q3);
          const Real lorentz = sqrt(1.0 + norm2);
          const Real alpha = gauge_d(p, 0);
          const Real transport1 = q1 - lorentz*gauge_d(p, 1)/alpha;
          const Real transport2 = q2 - lorentz*gauge_d(p, 2)/alpha;
          const Real transport3 = q3 - lorentz*gauge_d(p, 3)/alpha;
          const Real flux = alpha*primitive_d(p, IDN)*
              (transport1*dsigma_d(p, 0) + transport2*dsigma_d(p, 1) +
               transport3*dsigma_d(p, 2));
          sum.mdot -= flux;
          sum.area += area_d(p);
        }, Kokkos::Sum<SurfaceFluxSum>(result));
    Kokkos::fence();
    pdata->hdata[values_per_surface*n] = result.mdot;
    pdata->hdata[values_per_surface*n + 1] = result.area;
  }
}
