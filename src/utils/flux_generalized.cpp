//========================================================================================
// flux_generalized.cpp — integrate fluxes on arbitrary r(θ,φ) surfaces (Valencia MHD)
//----------------------------------------------------------------------------------------
#include "flux_generalized.hpp"
#include <iomanip>
#include <sstream>
#include <iostream>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "mhd/mhd.hpp"
#include "eos/eos.hpp"
#include "coordinates/adm.hpp"
#include "utils/surface_grid.hpp"
#include "outputs/outputs.hpp"
#include "z4c/z4c.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"

#include "eos/primitive_solver_hyd.hpp"
#include "eos/primitive-solver/idealgas.hpp"
#include "eos/primitive-solver/piecewise_polytrope.hpp"
#include "eos/primitive-solver/reset_floor.hpp"

// Expanded accumulator for separated fluid/EM fluxes and all axes
struct FluxAccumulator {
  Real mdot=0, edot_fluid=0, edot_em=0, phiB=0, area=0;
  // Linear momentum fluxes (fluid, EM)
  Real pdot_x_fluid=0, pdot_y_fluid=0, pdot_z_fluid=0;
  Real pdot_x_em=0, pdot_y_em=0, pdot_z_em=0;
  // Angular momentum fluxes (fluid, EM for x, y, z axes)
  Real ldot_x_fluid=0, ldot_y_fluid=0, ldot_z_fluid=0;
  Real ldot_x_em=0, ldot_y_em=0, ldot_z_em=0;

  KOKKOS_INLINE_FUNCTION
  FluxAccumulator& operator+=(const FluxAccumulator& src) {
    mdot += src.mdot; edot_fluid += src.edot_fluid; edot_em += src.edot_em;
    phiB += src.phiB; area += src.area;
    pdot_x_fluid += src.pdot_x_fluid; pdot_y_fluid += src.pdot_y_fluid;
    pdot_z_fluid += src.pdot_z_fluid;
    pdot_x_em += src.pdot_x_em; pdot_y_em += src.pdot_y_em; pdot_z_em += src.pdot_z_em;
    ldot_x_fluid += src.ldot_x_fluid; ldot_y_fluid += src.ldot_y_fluid;
    ldot_z_fluid += src.ldot_z_fluid;
    ldot_x_em += src.ldot_x_em; ldot_y_em += src.ldot_y_em; ldot_z_em += src.ldot_z_em;
    return *this;
  }
};

namespace Kokkos {
  template<> struct reduction_identity<FluxAccumulator> {
    KOKKOS_INLINE_FUNCTION static FluxAccumulator sum() { return FluxAccumulator{}; }
  };
} // namespace Kokkos


namespace { // anonymous namespace for internal helper functions

// --- Templated Helper Function ---
// This function contains the core logic and is templated on the specific
// DynGRMHDPS type. This allows compile-time access to the 'eos' member.
template <typename DynGRMHDPS_t>
void CalculateFluxesForEOS(HistoryData *pdata, MeshBlockPack *pmbp,
                           const std::vector<SphericalSurfaceGrid*>& surfs,
                           DynGRMHDPS_t* pdyngr_ps) {
  int nvars_mhd = pmbp->pmhd->nmhd + pmbp->pmhd->nscalars;
  auto eos_policy = pdyngr_ps->eos.ps.GetEOS(); // Capture by value for the kernel
  const Real mb = eos_policy.GetBaryonMass();
  const int nmhd = pmbp->pmhd->nmhd;
  const int nscalars = pmbp->pmhd->nscalars;
  const int nflux = 17;

  DualArray2D<Real> dSigma;

  for (int g=0; g<surfs.size(); ++g) {
    auto current_surf = surfs[g];

    // reconstruct surface geometry, interpolation etc if using AMR
    if (pmbp->pmesh->adaptive) {
      current_surf->RebuildAll();
    }

    int np = current_surf->Npts();

    //--- 2. Interpolate all required grid quantities to the surface ---
    DualArray2D<Real> w_vals = current_surf->InterpolateToSurface(pmbp->pmhd->w0, 0, nvars_mhd);
    auto w_vals_d = w_vals.d_view;
    DualArray2D<Real> B_vals = current_surf->InterpolateToSurface(pmbp->pmhd->bcc0, 0, 3);
    auto B_vals_d = B_vals.d_view;
    auto gauge_source_array = pmbp->padm->u_adm;
    int ab_start = adm::ADM::I_ADM_ALPHA;
    int ab_end = adm::ADM::I_ADM_BETAZ + 1;
    if (pmbp->pz4c != nullptr) {
      gauge_source_array = pmbp->pz4c->u0;
      ab_start = z4c::Z4c::I_Z4C_ALPHA;
      ab_end = z4c::Z4c::I_Z4C_BETAZ + 1;
    }
    DualArray2D<Real> ab_vals = current_surf->InterpolateToSurface(gauge_source_array, ab_start, ab_end);
    auto ab_vals_d = ab_vals.d_view;

    //--- 3. Prepare surface geometry (interpolates g_ij internally) ---
    current_surf->InterpolateMetric();
    current_surf->BuildSurfaceCovectors(dSigma);

    //--- 4. Integrate fluxes over the surface ---
    auto g_surf_d = current_surf->Metric().d_view;
    FluxAccumulator fluxes;
    Kokkos::parallel_reduce("flux_integral_from_prims", Kokkos::RangePolicy<DevExeSpace>(0, np),
      KOKKOS_LAMBDA(const int p, FluxAccumulator& update) {
        //--- A. Get interpolated primitive variables and geometry at point p ---
        const Real rho = w_vals_d(p,IDN);
        const Real pgas = w_vals_d(p,IPR);
        const Real v_prim_u[3] = {w_vals_d(p,IVX), w_vals_d(p,IVY), w_vals_d(p,IVZ)};

        const Real alp = ab_vals_d(p,0);
        const Real beta_u[3] = {ab_vals_d(p,1), ab_vals_d(p,2), ab_vals_d(p,3)};
        const Real g_dd[3][3] = {
            {g_surf_d(p,0), g_surf_d(p,1), g_surf_d(p,2)},
            {g_surf_d(p,1), g_surf_d(p,3), g_surf_d(p,4)},
            {g_surf_d(p,2), g_surf_d(p,4), g_surf_d(p,5)}
        };
        const Real dS_d[3] = {dSigma.d_view(p,0), dSigma.d_view(p,1), dSigma.d_view(p,2)};

        //--- B. Calculate derived quantities from primitives ---
        const Real det_g = adm::SpatialDet(g_dd[0][0],g_dd[0][1],g_dd[0][2],g_dd[1][1],g_dd[1][2],g_dd[2][2]);
        const Real sqrt_det_g = sqrt(det_g);
        const Real ivol = 1.0/sqrt_det_g;
        const Real B_u[3] = {B_vals_d(p,0)*ivol, B_vals_d(p,1)*ivol, B_vals_d(p,2)*ivol};

        const Real g_dd_1d[6] = {g_dd[0][0], g_dd[0][1], g_dd[0][2],
                                 g_dd[1][1], g_dd[1][2], g_dd[2][2]};

        Real v_sq = Primitive::SquareVector(v_prim_u, g_dd_1d);
        if (v_sq >= 1.0) v_sq = 1.0 - 1.0e-16;
        const Real W = 1.0/sqrt(1.0 - v_sq);
        const Real Wsq = W * W;

        // General EOS-based enthalpy calculation
        Real prim_pt[NPRIM] = {0.0};
        prim_pt[PRH] = rho / mb;
        prim_pt[PVX] = v_prim_u[0];
        prim_pt[PVY] = v_prim_u[1];
        prim_pt[PVZ] = v_prim_u[2];
        prim_pt[PPR] = pgas;

        for (int s = 0; s < nscalars; ++s) {
            prim_pt[PYF + s] = w_vals_d(p, nmhd + s);
        }

        prim_pt[PTM] = eos_policy.GetTemperatureFromP(prim_pt[PRH], prim_pt[PPR],
                                                 nullptr);
        const Real H_enthalpy = rho * eos_policy.GetEnthalpy(prim_pt[PRH], prim_pt[PTM],
                                                        nullptr);

        //--- C1. Calculate helper quantities from source-term formulation ---
        const Real B_sq = Primitive::SquareVector(B_u, g_dd_1d);

        Real v_prim_d[3]={0.};
        Real B_d[3]={0.};
        for(int i=0; i<3; ++i) {
          for(int j=0; j<3; ++j) {
            v_prim_d[i] += g_dd[i][j] * v_prim_u[j];
            B_d[i]      += g_dd[i][j] * B_u[j];
          }
        }

        const Real B_dot_v_prim = B_u[0]*v_prim_d[0] + B_u[1]*v_prim_d[1] + B_u[2]*v_prim_d[2];
        const Real b_sq = (B_dot_v_prim * B_dot_v_prim) + B_sq/Wsq;

        //--- C2. Separate Energy Density (E) by term ---
        const Real E_fluid = H_enthalpy * Wsq - pgas;
        const Real E_em = B_sq - 0.5*b_sq;

        //--- C3. Separate Momentum Density (S_i) by term ---
        Real S_d_fluid[3];
        Real S_d_em[3];
        for(int i=0; i<3; ++i) {
            S_d_fluid[i] = H_enthalpy * Wsq * v_prim_d[i];
            S_d_em[i] = b_sq * v_prim_d[i] - (B_dot_v_prim * B_d[i] / Wsq);
        }

        //--- C4. Separate Spatial Stress (S_ij) by term ---
        Real g_u[3][3];
        {
            Real g_uu[6];
            adm::SpatialInv(1.0/det_g, g_dd[0][0],g_dd[0][1],g_dd[0][2],g_dd[1][1],g_dd[1][2],g_dd[2][2],
                            &g_uu[0], &g_uu[1], &g_uu[2], &g_uu[3], &g_uu[4], &g_uu[5]);
            g_u[0][0]=g_uu[0]; g_u[0][1]=g_uu[1]; g_u[0][2]=g_uu[2];
            g_u[1][0]=g_uu[1]; g_u[1][1]=g_uu[3]; g_u[1][2]=g_uu[4];
            g_u[2][0]=g_uu[2]; g_u[2][1]=g_uu[4]; g_u[2][2]=g_uu[5];
        }

        Real S_uu_fluid[3][3];
        Real S_uu_em[3][3];
        for (int i = 0; i < 3; i++) {
          for (int j = 0; j < 3; j++) {
            S_uu_fluid[i][j] = H_enthalpy * Wsq * v_prim_u[i] * v_prim_u[j] + pgas * g_u[i][j];

            S_uu_em[i][j] = b_sq * v_prim_u[i] * v_prim_u[j]
                          - (B_u[i]*B_u[j]/Wsq)
                          - (B_dot_v_prim/W)*(B_u[i]*v_prim_u[j] + B_u[j]*v_prim_u[i])
                          + (0.5*b_sq)*g_u[i][j];
          }
        }

        Real S_dd_fluid[3][3] = {{0.}};
        Real S_dd_em[3][3] = {{0.}};
        for (int i=0; i<3; ++i) {
          for (int j=0; j<3; ++j) {
            for (int a=0; a<3; ++a) {
              for (int b=0; b<3; ++b) {
                S_dd_fluid[i][j] += g_dd[i][a] * g_dd[j][b] * S_uu_fluid[a][b];
                S_dd_em[i][j] += g_dd[i][a] * g_dd[j][b] * S_uu_em[a][b];
              }
            }
          }
        }
        
        //--- D. Calculate all fluxes ---
        const Real v_coord_x = alp * v_prim_u[0] - beta_u[0];
        const Real v_coord_y = alp * v_prim_u[1] - beta_u[1];
        const Real v_coord_z = alp * v_prim_u[2] - beta_u[2];
        const Real u_transport_u[3] = {(W/alp)*v_coord_x, (W/alp)*v_coord_y, (W/alp)*v_coord_z};
        
        const Real alpha_sqrt_gamma = alp * sqrt_det_g;
        const Real mass_flux_x = alpha_sqrt_gamma * rho * u_transport_u[0];
        const Real mass_flux_y = alpha_sqrt_gamma * rho * u_transport_u[1];
        const Real mass_flux_z = alpha_sqrt_gamma * rho * u_transport_u[2];
        update.mdot += -(mass_flux_x * dS_d[0] + mass_flux_y * dS_d[1] + mass_flux_z * dS_d[2]);
        
        Real S_u_fluid[3] = {0.}, S_u_em[3] = {0.};
        for(int i=0; i<3; ++i) {
          for(int j=0; j<3; ++j) {
            S_u_fluid[i] += g_u[i][j] * S_d_fluid[j];
            S_u_em[i]    += g_u[i][j] * S_d_em[j];
          }
        }
        Real FE_fluid[3], FE_em[3];
        for(int i=0; i<3; ++i) {
          FE_fluid[i] = alp * S_u_fluid[i] - beta_u[i] * E_fluid;
          FE_em[i]    = alp * S_u_em[i]    - beta_u[i] * E_em;
        }
        update.edot_fluid += -(FE_fluid[0]*dS_d[0] + FE_fluid[1]*dS_d[1] + FE_fluid[2]*dS_d[2]);
        update.edot_em    += -(FE_em[0]*dS_d[0]    + FE_em[1]*dS_d[1]    + FE_em[2]*dS_d[2]);

        Real FS_fluid[3][3], FS_em[3][3];
        for(int i=0; i<3; ++i) {
          for(int j=0; j<3; ++j) {
            Real S_ui_fluid_j = 0., S_ui_em_j = 0.;
            for(int k=0; k<3; ++k) {
              S_ui_fluid_j += g_u[i][k] * S_dd_fluid[k][j];
              S_ui_em_j    += g_u[i][k] * S_dd_em[k][j];
            }
            FS_fluid[i][j] = alp * S_ui_fluid_j - beta_u[i] * S_d_fluid[j];
            FS_em[i][j]    = alp * S_ui_em_j    - beta_u[i] * S_d_em[j];
          }
        }
        
        Real d_pdot_fluid[3]={0.}, d_pdot_em[3]={0.};
        for (int j=0; j<3; ++j) {
          for (int i=0; i<3; ++i) {
            d_pdot_fluid[j] += FS_fluid[i][j] * dS_d[i];
            d_pdot_em[j]    += FS_em[i][j]    * dS_d[i];
          }
        }
        
        update.pdot_x_fluid += -d_pdot_fluid[0];
        update.pdot_y_fluid += -d_pdot_fluid[1];
        update.pdot_z_fluid += -d_pdot_fluid[2];
        update.pdot_x_em += -d_pdot_em[0];
        update.pdot_y_em += -d_pdot_em[1];
        update.pdot_z_em += -d_pdot_em[2];

        const Real x = current_surf->Coords().d_view(p,0);
        const Real y = current_surf->Coords().d_view(p,1);
        const Real z = current_surf->Coords().d_view(p,2);
        update.ldot_x_fluid += -(y * d_pdot_fluid[2] - z * d_pdot_fluid[1]);
        update.ldot_y_fluid += -(z * d_pdot_fluid[0] - x * d_pdot_fluid[2]);
        update.ldot_z_fluid += -(x * d_pdot_fluid[1] - y * d_pdot_fluid[0]);
        update.ldot_x_em += -(y * d_pdot_em[2] - z * d_pdot_em[1]);
        update.ldot_y_em += -(z * d_pdot_em[0] - x * d_pdot_em[2]);
        update.ldot_z_em += -(x * d_pdot_em[1] - y * d_pdot_em[0]);

        {
          const Real u_u_t = W / alp;
          const Real u_u[3] = {W*v_prim_u[0], W*v_prim_u[1], W*v_prim_u[2]};
          const Real b_u_t = B_dot_v_prim / alp;
          Real b_u[3];
          for (int i=0; i<3; ++i) {
            b_u[i] = (B_u[i] / W) + (B_dot_v_prim * v_prim_u[i]);
          }
          const Real M_u[3] = {
            u_u_t * b_u[0] - u_u[0] * b_u_t,
            u_u_t * b_u[1] - u_u[1] * b_u_t,
            u_u_t * b_u[2] - u_u[2] * b_u_t
          };
          const Real flux_dot_dS = M_u[0]*dS_d[0] + M_u[1]*dS_d[1] + M_u[2]*dS_d[2];
          update.phiB += 0.5 * Kokkos::fabs(flux_dot_dS);
        }
        
        Real dS_u[3] = {0.};
        for(int i=0; i<3; ++i) {
            for(int j=0; j<3; ++j) { dS_u[i] += g_u[i][j] * dS_d[j]; }
        }
        const Real dS_sq = dS_d[0]*dS_u[0] + dS_d[1]*dS_u[1] + dS_d[2]*dS_u[2];
        update.area += (dS_sq > 0.0) ? sqrt(dS_sq) : 0.0;

      }, Kokkos::Sum<FluxAccumulator>(fluxes));
    
    // Store the final integrated values in the history data array
    int i = nflux*g;
    pdata->hdata[i++] = fluxes.mdot;
    pdata->hdata[i++] = fluxes.edot_fluid;
    pdata->hdata[i++] = fluxes.edot_em;
    pdata->hdata[i++] = fluxes.pdot_x_fluid;
    pdata->hdata[i++] = fluxes.pdot_y_fluid;
    pdata->hdata[i++] = fluxes.pdot_z_fluid;
    pdata->hdata[i++] = fluxes.pdot_x_em;
    pdata->hdata[i++] = fluxes.pdot_y_em;
    pdata->hdata[i++] = fluxes.pdot_z_em;
    pdata->hdata[i++] = fluxes.ldot_x_fluid;
    pdata->hdata[i++] = fluxes.ldot_y_fluid;
    pdata->hdata[i++] = fluxes.ldot_z_fluid;
    pdata->hdata[i++] = fluxes.ldot_x_em;
    pdata->hdata[i++] = fluxes.ldot_y_em;
    pdata->hdata[i++] = fluxes.ldot_z_em;
    pdata->hdata[i++] = fluxes.phiB;
    pdata->hdata[i++] = fluxes.area;
  }
}

} // end anonymous namespace


// --- Main Function Definition ---
// This function now acts as a dispatcher. It determines the runtime EOS type
// and calls the appropriate templated helper function.
void TorusFluxes_General(HistoryData *pdata,
                         MeshBlockPack *pmbp,
                         const std::vector<SphericalSurfaceGrid*>& surfs) {
  //--- 1. Validation and Setup ---
  if (pmbp->padm == nullptr || pmbp->pmhd == nullptr || pmbp->pdyngr == nullptr) {
    std::cerr << "### FATAL: TorusFluxes_General requires ADM, MHD, and DynGRMHD objects." << std::endl;
    std::exit(EXIT_FAILURE);
  }

  if (surfs.empty()) { pdata->nhist = 0; return; }
  const int nflux = 17;
  pdata->nhist = surfs.size() * nflux;

  if (pdata->nhist > NHISTORY_VARIABLES) {
    std::cerr << "### FATAL: pdata->nhist exceeds NHISTORY_VARIABLES" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // Set new history output labels
  for (int g=0; g<surfs.size(); ++g) {
    std::string s_name = surfs[g]->Label();
    int i = nflux*g;
    pdata->label[i++] = "mdot_" + s_name;
    pdata->label[i++] = "edot_fluid_" + s_name;
    pdata->label[i++] = "edot_em_" + s_name;
    pdata->label[i++] = "pdot_x_fluid_" + s_name;
    pdata->label[i++] = "pdot_y_fluid_" + s_name;
    pdata->label[i++] = "pdot_z_fluid_" + s_name;
    pdata->label[i++] = "pdot_x_em_" + s_name;
    pdata->label[i++] = "pdot_y_em_" + s_name;
    pdata->label[i++] = "pdot_z_em_" + s_name;
    pdata->label[i++] = "ldot_x_fluid_" + s_name;
    pdata->label[i++] = "ldot_y_fluid_" + s_name;
    pdata->label[i++] = "ldot_z_fluid_" + s_name;
    pdata->label[i++] = "ldot_x_em_" + s_name;
    pdata->label[i++] = "ldot_y_em_" + s_name;
    pdata->label[i++] = "ldot_z_em_" + s_name;
    pdata->label[i++] = "phiB_" + s_name;
    pdata->label[i++] = "area_" + s_name;
  }

  // Dispatch to the correct templated implementation based on the runtime EOS policy.
  // We assume the error policy is always ResetFloor as it's the only one
  // instantiated in the factory function.
  switch (pmbp->pdyngr->eos_policy) {
    case DynGRMHD_EOS::eos_ideal: {
      auto pdyngr_ps = static_cast<dyngr::DynGRMHDPS<Primitive::IdealGas, Primitive::ResetFloor>*>(pmbp->pdyngr);
      CalculateFluxesForEOS(pdata, pmbp, surfs, pdyngr_ps);
      break;
    }
    case DynGRMHD_EOS::eos_piecewise_poly: {
      auto pdyngr_ps = static_cast<dyngr::DynGRMHDPS<Primitive::PiecewisePolytrope, Primitive::ResetFloor>*>(pmbp->pdyngr);
      CalculateFluxesForEOS(pdata, pmbp, surfs, pdyngr_ps);
      break;
    }
    // Add other EOS cases here if they are added to the factory function
    // case DynGRMHD_EOS::eos_compose: {
    //   auto pdyngr_ps = static_cast<dyngr::DynGRMHDPS<Primitive::EOSCompOSE, Primitive::ResetFloor>*>(pmbp->pdyngr);
    //   CalculateFluxesForEOS(pdata, pmbp, surfs, pdyngr_ps);
    //   break;
    // }
    default:
      std::cerr << "### FATAL: EOS policy in TorusFluxes_General is not supported." << std::endl;
      std::exit(EXIT_FAILURE);
  }

  for (int n=pdata->nhist; n<NHISTORY_VARIABLES; ++n) pdata->hdata[n] = 0.0;
}