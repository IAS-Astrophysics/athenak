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


// --- Main Function Definition ---
void TorusFluxes_General(HistoryData *pdata,
                         MeshBlockPack *pmbp,
                         const std::vector<SphericalSurfaceGrid*>& surfs) {
  //--- 1. Validation and Setup ---
  if (pmbp->padm == nullptr || pmbp->pmhd == nullptr) {
    std::cerr << "### FATAL: TorusFluxes_General requires ADM and MHD objects." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  int nvars_mhd = pmbp->pmhd->nmhd + pmbp->pmhd->nscalars;
  const Real gamma_eos = pmbp->pmhd->peos->eos_data.gamma;

  if (surfs.empty()) { pdata->nhist = 0; return; }
  int nsurf = surfs.size();
  const int nflux = 17;
  pdata->nhist = nsurf * nflux;

  if (pdata->nhist > NHISTORY_VARIABLES) {
    std::cerr << "### FATAL: pdata->nhist exceeds NHISTORY_VARIABLES" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // Set new history output labels
  for (int g=0; g<nsurf; ++g) {
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

  DualArray2D<Real> dSigma;

  for (int g=0; g<nsurf; ++g) {
    auto current_surf = surfs[g];
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

        Real v_prim_d[3]={0.};
        Real v_sq = 0.;
        for(int i=0; i<3; ++i) {
          for(int j=0; j<3; ++j) { v_prim_d[i] += g_dd[i][j] * v_prim_u[j]; }
          v_sq += v_prim_u[i] * v_prim_d[i];
        }
        if (v_sq >= 1.0) v_sq = 1.0 - 1.0e-16;
        const Real W = 1.0/sqrt(1.0 - v_sq);
        const Real h = 1.0 + (pgas/(gamma_eos-1.0) + pgas)/rho;

        //--- C. Calculate 3+1 components of the stress-energy tensor ---
        // 1. Fluid (advective) part
        const Real E_fluid = rho * h * W * W - pgas;
        Real S_d_fluid[3];
        for(int i=0; i<3; ++i) { S_d_fluid[i] = rho * h * W * W * v_prim_d[i]; }
        Real S_dd_fluid[3][3];
        for(int i=0; i<3; ++i) {
          for(int j=0; j<3; ++j) {
            S_dd_fluid[i][j] = S_d_fluid[i] * v_prim_d[j] + pgas * g_dd[i][j];
          }
        }
        
        // 2. EM part
        Real B_d[3]={0.};
        Real B_sq = 0.;
        for(int i=0; i<3; ++i) {
          for(int j=0; j<3; ++j) { B_d[i] += g_dd[i][j] * B_u[j]; }
          B_sq += B_u[i] * B_d[i];
        }
        const Real E_d[3] = {-sqrt_det_g * (v_prim_u[1]*B_u[2] - v_prim_u[2]*B_u[1]),
                             -sqrt_det_g * (v_prim_u[2]*B_u[0] - v_prim_u[0]*B_u[2]),
                             -sqrt_det_g * (v_prim_u[0]*B_u[1] - v_prim_u[1]*B_u[0])};

        Real g_uu[6], g_u[3][3];
        adm::SpatialInv(1.0/det_g, g_dd[0][0],g_dd[0][1],g_dd[0][2],g_dd[1][1],g_dd[1][2],g_dd[2][2],
                        &g_uu[0], &g_uu[1], &g_uu[2], &g_uu[3], &g_uu[4], &g_uu[5]);
        g_u[0][0]=g_uu[0]; g_u[0][1]=g_uu[1]; g_u[0][2]=g_uu[2];
        g_u[1][0]=g_uu[1]; g_u[1][1]=g_uu[3]; g_u[1][2]=g_uu[4];
        g_u[2][0]=g_uu[2]; g_u[2][1]=g_uu[4]; g_u[2][2]=g_uu[5];
        
        Real E_sq = 0.;
        for(int i=0; i<3; ++i) {
          for(int j=0; j<3; ++j) { E_sq += g_u[i][j] * E_d[i] * E_d[j]; }
        }
        const Real E_em = 0.5 * (B_sq + E_sq);
        
        Real E_u[3]={0.};
        for(int i=0; i<3; ++i) {
          for(int j=0; j<3; ++j) { E_u[i] += g_u[i][j] * E_d[j]; }
        }
        const Real S_d_em[3] = {sqrt_det_g * (E_u[1]*B_u[2] - E_u[2]*B_u[1]),
                                sqrt_det_g * (E_u[2]*B_u[0] - E_u[0]*B_u[2]),
                                sqrt_det_g * (E_u[0]*B_u[1] - E_u[1]*B_u[0])};
        Real S_dd_em[3][3];
        for(int i=0; i<3; ++i) {
          for(int j=0; j<3; ++j) {
            S_dd_em[i][j] = -E_d[i]*E_d[j] - B_d[i]*B_d[j] + E_em*g_dd[i][j];
          }
        }
        
        //--- D. Calculate all fluxes ---
        // 1. Mass Flux (purely fluid)
        const Real v_coord_x = alp * v_prim_u[0] - beta_u[0];
        const Real v_coord_y = alp * v_prim_u[1] - beta_u[1];
        const Real v_coord_z = alp * v_prim_u[2] - beta_u[2];
        const Real u_transport_u[3] = {(W/alp)*v_coord_x, (W/alp)*v_coord_y, (W/alp)*v_coord_z};
        
        const Real alpha_sqrt_gamma = alp * sqrt_det_g;
        const Real mass_flux_x = alpha_sqrt_gamma * rho * u_transport_u[0];
        const Real mass_flux_y = alpha_sqrt_gamma * rho * u_transport_u[1];
        const Real mass_flux_z = alpha_sqrt_gamma * rho * u_transport_u[2];
        update.mdot += -(mass_flux_x * dS_d[0] + mass_flux_y * dS_d[1] + mass_flux_z * dS_d[2]);
        
        // 2. Energy Flux (fluid, EM)
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

        // ----- START FIX: Corrected momentum and angular momentum flux calculation -----
        // 3. Momentum and Angular Momentum Fluxes
        Real FS_fluid[3][3], FS_em[3][3]; // Momentum flux density T^i_j
        for(int i=0; i<3; ++i) { // contravariant index i
          for(int j=0; j<3; ++j) { // covariant index j
            Real S_ui_fluid_j = 0., S_ui_em_j = 0.;
            for(int k=0; k<3; ++k) {
              S_ui_fluid_j += g_u[i][k] * S_dd_fluid[k][j];
              S_ui_em_j    += g_u[i][k] * S_dd_em[k][j];
            }
            FS_fluid[i][j] = alp * S_ui_fluid_j - beta_u[i] * S_d_fluid[j];
            FS_em[i][j]    = alp * S_ui_em_j    - beta_u[i] * S_d_em[j];
          }
        }
        
        // Calculate momentum flux through the surface element dS_d at this point
        Real d_pdot_fluid[3]={0.}, d_pdot_em[3]={0.};
        for (int j=0; j<3; ++j) { // j-th component of momentum
          for (int i=0; i<3; ++i) { // flux through i-th surface direction
            d_pdot_fluid[j] += FS_fluid[i][j] * dS_d[i];
            d_pdot_em[j]    += FS_em[i][j]    * dS_d[i];
          }
        }
        
        // Update total linear momentum flux (negative sign for flux *into* volume)
        update.pdot_x_fluid += -d_pdot_fluid[0];
        update.pdot_y_fluid += -d_pdot_fluid[1];
        update.pdot_z_fluid += -d_pdot_fluid[2];
        update.pdot_x_em += -d_pdot_em[0];
        update.pdot_y_em += -d_pdot_em[1];
        update.pdot_z_em += -d_pdot_em[2];

        // Update total angular momentum flux using the local coordinates
        const Real x = current_surf->Coords().d_view(p,0);
        const Real y = current_surf->Coords().d_view(p,1);
        const Real z = current_surf->Coords().d_view(p,2);
        update.ldot_x_fluid += -(y * d_pdot_fluid[2] - z * d_pdot_fluid[1]);
        update.ldot_y_fluid += -(z * d_pdot_fluid[0] - x * d_pdot_fluid[2]);
        update.ldot_z_fluid += -(x * d_pdot_fluid[1] - y * d_pdot_fluid[0]);
        update.ldot_x_em += -(y * d_pdot_em[2] - z * d_pdot_em[1]);
        update.ldot_y_em += -(z * d_pdot_em[0] - x * d_pdot_em[2]);
        update.ldot_z_em += -(x * d_pdot_em[1] - y * d_pdot_em[0]);
        // ----- END FIX -----

        // ----- START FIX: Corrected magnetic flux calculation -----
        // 4. Unsigned Magnetic Flux (purely EM)
        {
          // The magnetic flux is the integral of the normal component of the Eulerian
          // magnetic field B^i over the surface.
          // B_u is the contravariant magnetic field component B^i.
          // dS_d is the covariant surface element dS_i.
          // The scalar product B^i dS_i is the flux through the surface element.
          const Real B_dot_dS = B_u[0] * dS_d[0] + B_u[1] * dS_d[1] + B_u[2] * dS_d[2];

          // We sum the absolute value of the flux through each element to get the
          // total unsigned magnetic flux piercing the surface.
          update.phiB += Kokkos::fabs(B_dot_dS);
        }
        // ----- END FIX -----
        
        // 5. Surface Area
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

  for (int n=pdata->nhist; n<NHISTORY_VARIABLES; ++n) pdata->hdata[n] = 0.0;
}