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
#include "z4c/z4c.hpp" // Now needed for Z4c variable indices
#include "z4c/tmunu.hpp"

// Custom Kokkos reducer for summing multiple flux quantities in one kernel
struct FluxAccumulator {
  Real mdot=0, edot=0, ldot=0, phiB=0;
  Real pdot_x=0, pdot_y=0, pdot_z=0;
  Real area=0;

  KOKKOS_INLINE_FUNCTION
  FluxAccumulator& operator+=(const FluxAccumulator& src) {
    mdot += src.mdot; edot += src.edot; ldot += src.ldot; phiB += src.phiB;
    pdot_x += src.pdot_x; pdot_y += src.pdot_y; pdot_z += src.pdot_z;
    area += src.area;
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
                         const std::vector<SphericalSurfaceGrid*>& surfs,
                         const Real axis_n[3]) {
  //--- 1. Validation and Setup ---
  if (pmbp->padm == nullptr || pmbp->pmhd == nullptr || pmbp->ptmunu == nullptr) {
    std::cerr << "### FATAL: TorusFluxes_General requires ADM, MHD and Tmunu objects." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  int nvars_mhd = pmbp->pmhd->nmhd + pmbp->pmhd->nscalars;
  auto u_tmunu = pmbp->ptmunu->u_tmunu;

  if (surfs.empty()) { pdata->nhist = 0; return; }
  int nsurf = surfs.size();
  const int nflux = 8;
  pdata->nhist = nsurf * nflux;

  for (int g=0; g<nsurf; ++g) {
    std::string s_name = surfs[g]->Label();
    pdata->label[nflux*g+0] = "mdot_" + s_name; pdata->label[nflux*g+1] = "edot_" + s_name;
    pdata->label[nflux*g+2] = "ldot_" + s_name; pdata->label[nflux*g+3] = "phiB_" + s_name;
    pdata->label[nflux*g+4] = "pdot_x_" + s_name; pdata->label[nflux*g+5] = "pdot_y_" + s_name;
    pdata->label[nflux*g+6] = "pdot_z_" + s_name;
    pdata->label[nflux*g+7] = "area_" + s_name;
  }

  Real n_ax[3] = {0.0,0.0,1.0};
  if (axis_n) { n_ax[0]=axis_n[0]; n_ax[1]=axis_n[1]; n_ax[2]=axis_n[2]; }

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
    DualArray2D<Real> tmunu_vals = current_surf->InterpolateToSurface(u_tmunu, 0, Tmunu::N_Tmunu);
    auto tmunu_vals_d = tmunu_vals.d_view;

    //--- 3. Prepare surface geometry (interpolates g_ij internally) ---
    current_surf->InterpolateMetric();
    current_surf->BuildSurfaceCovectors(dSigma);

    //--- 4. Integrate fluxes over the surface ---
    auto g_surf_d = current_surf->Metric().d_view;
    FluxAccumulator total_fluxes;
    Kokkos::parallel_reduce("flux_integral", Kokkos::RangePolicy<DevExeSpace>(0, np),
      KOKKOS_LAMBDA(const int p, FluxAccumulator& update) {
        const Real rho = w_vals_d(p,IDN);
        const Real v_prim_x = w_vals_d(p,IVX), v_prim_y = w_vals_d(p,IVY), v_prim_z = w_vals_d(p,IVZ);
        const Real Bx = B_vals_d(p,0), By = B_vals_d(p,1), Bz = B_vals_d(p,2);
        const Real alp = ab_vals_d(p,0);
        const Real beta_u[3] = {ab_vals_d(p,1), ab_vals_d(p,2), ab_vals_d(p,3)};
        const Real gxx= g_surf_d(p,0), gxy= g_surf_d(p,1), gxz= g_surf_d(p,2),
                   gyy= g_surf_d(p,3), gyz= g_surf_d(p,4), gzz= g_surf_d(p,5);
        const Real E_surf = tmunu_vals_d(p, Tmunu::I_Tmunu_E);
        const Real S_d[3] = { tmunu_vals_d(p, Tmunu::I_Tmunu_Sx),
                              tmunu_vals_d(p, Tmunu::I_Tmunu_Sy),
                              tmunu_vals_d(p, Tmunu::I_Tmunu_Sz) };
        const Real S_dd[6] = { tmunu_vals_d(p, Tmunu::I_Tmunu_Sxx),
                               tmunu_vals_d(p, Tmunu::I_Tmunu_Sxy),
                               tmunu_vals_d(p, Tmunu::I_Tmunu_Sxz),
                               tmunu_vals_d(p, Tmunu::I_Tmunu_Syy),
                               tmunu_vals_d(p, Tmunu::I_Tmunu_Syz),
                               tmunu_vals_d(p, Tmunu::I_Tmunu_Szz) };
        const Real dSx = dSigma.d_view(p,0), dSy = dSigma.d_view(p,1), dSz = dSigma.d_view(p,2);

        Real g_uu[6]; Real uxx, uxy, uxz, uyy, uyz, uzz;
        const Real det_g = adm::SpatialDet(gxx, gxy, gxz, gyy, gyz, gzz);
        adm::SpatialInv(1.0/det_g, gxx, gxy, gxz, gyy, gyz, gzz, &uxx, &uxy, &uxz, &uyy, &uyz, &uzz);
        g_uu[0]=uxx; g_uu[1]=uxy; g_uu[2]=uxz; g_uu[3]=uyy; g_uu[4]=uyz; g_uu[5]=uzz;
        
        const Real v_coord_x = alp * v_prim_x - beta_u[0];
        const Real v_coord_y = alp * v_prim_y - beta_u[1];
        const Real v_coord_z = alp * v_prim_z - beta_u[2];

        const Real v_coord_d[3] = {gxx*v_coord_x + gxy*v_coord_y + gxz*v_coord_z,
                                   gxy*v_coord_x + gyy*v_coord_y + gyz*v_coord_z,
                                   gxz*v_coord_x + gyz*v_coord_y + gzz*v_coord_z};
        const Real v2 = v_coord_x*v_coord_d[0] + v_coord_y*v_coord_d[1] + v_coord_z*v_coord_d[2];
        Real one_minus_v2 = 1.0 - v2; if (one_minus_v2 < 1e-16) one_minus_v2 = 1e-16;
        const Real W = 1.0/sqrt(one_minus_v2);

        // 1. Mass Flux (mdot)
        const Real u_u[3] = {(W/alp)*v_coord_x, (W/alp)*v_coord_y, (W/alp)*v_coord_z};
        update.mdot += -(rho*u_u[0]*dSx + rho*u_u[1]*dSy + rho*u_u[2]*dSz);

        // 2. Energy Flux (edot)
        const Real S_u[3] = {g_uu[0]*S_d[0] + g_uu[1]*S_d[1] + g_uu[2]*S_d[2],
                             g_uu[1]*S_d[0] + g_uu[3]*S_d[1] + g_uu[4]*S_d[2],
                             g_uu[2]*S_d[0] + g_uu[4]*S_d[1] + g_uu[5]*S_d[2]};
        const Real FE_x = alp*S_u[0] - beta_u[0]*E_surf;
        const Real FE_y = alp*S_u[1] - beta_u[1]*E_surf;
        const Real FE_z = alp*S_u[2] - beta_u[2]*E_surf;
        update.edot += -(FE_x*dSx + FE_y*dSy + FE_z*dSz);

        // 3. Linear Momentum Flux (pdot)
        Real S_ud[3][3];
        S_ud[0][0] = g_uu[0]*S_dd[0] + g_uu[1]*S_dd[1] + g_uu[2]*S_dd[2]; S_ud[0][1] = g_uu[0]*S_dd[1] + g_uu[1]*S_dd[3] + g_uu[2]*S_dd[4]; S_ud[0][2] = g_uu[0]*S_dd[2] + g_uu[1]*S_dd[4] + g_uu[2]*S_dd[5];
        S_ud[1][0] = g_uu[1]*S_dd[0] + g_uu[3]*S_dd[1] + g_uu[4]*S_dd[2]; S_ud[1][1] = g_uu[1]*S_dd[1] + g_uu[3]*S_dd[3] + g_uu[4]*S_dd[4]; S_ud[1][2] = g_uu[1]*S_dd[2] + g_uu[3]*S_dd[4] + g_uu[4]*S_dd[5];
        S_ud[2][0] = g_uu[2]*S_dd[0] + g_uu[4]*S_dd[1] + g_uu[5]*S_dd[2]; S_ud[2][1] = g_uu[2]*S_dd[1] + g_uu[4]*S_dd[3] + g_uu[5]*S_dd[4]; S_ud[2][2] = g_uu[2]*S_dd[2] + g_uu[4]*S_dd[4] + g_uu[5]*S_dd[5];
        Real FS[3][3];
        for (int i=0; i<3; ++i) for (int j=0; j<3; ++j) FS[i][j] = alp*S_ud[i][j] - beta_u[i]*S_d[j];
        update.pdot_x += -(FS[0][0]*dSx + FS[1][0]*dSy + FS[2][0]*dSz);
        update.pdot_y += -(FS[0][1]*dSx + FS[1][1]*dSy + FS[2][1]*dSz);
        update.pdot_z += -(FS[0][2]*dSx + FS[1][2]*dSy + FS[2][2]*dSz);

        // 4. Angular Momentum Flux (ldot)
        const Real x = current_surf->Coords().d_view(p,0), y = current_surf->Coords().d_view(p,1), z = current_surf->Coords().d_view(p,2);
        const Real ph_d[3] = {n_ax[1]*z - n_ax[2]*y, n_ax[2]*x - n_ax[0]*z, n_ax[0]*y - n_ax[1]*x};
        const Real FL_x = ph_d[0]*FS[0][0] + ph_d[1]*FS[0][1] + ph_d[2]*FS[0][2];
        const Real FL_y = ph_d[0]*FS[1][0] + ph_d[1]*FS[1][1] + ph_d[2]*FS[1][2];
        const Real FL_z = ph_d[0]*FS[2][0] + ph_d[1]*FS[2][1] + ph_d[2]*FS[2][2];
        update.ldot += -(FL_x*dSx + FL_y*dSy + FL_z*dSz);

        // 5. Magnetic Flux (phiB)
        // <--- MODIFIED: Replaced simple B.dS with the paper's formula for unsigned flux.
        { // Start new scope for magnetic flux variables
          // We need u^0 and b^mu to compute the integrand |b^r u^0 - b^0 u^r| from the paper.
          // This is a component of the dual Faraday tensor *F^{mu,nu}.
          
          // Calculate fluid 4-velocity u^mu = (u^0, u^i)
          const Real u0 = W / alp;
          // u_u[3] is u^i, already calculated above

          // Calculate magnetic 4-vector b^mu = (b^0, b^i)
          // We need v_i from the primitive velocity for this
          const Real v_prim_d[3] = {gxx*v_prim_x + gxy*v_prim_y + gxz*v_prim_z,
                                    gxy*v_prim_x + gyy*v_prim_y + gyz*v_prim_z,
                                    gxz*v_prim_x + gyz*v_prim_y + gzz*v_prim_z};
          const Real B_dot_v_prim = Bx * v_prim_d[0] + By * v_prim_d[1] + Bz * v_prim_d[2];
          
          const Real b0 = W * B_dot_v_prim / alp;
          const Real b_u_x = (Bx + u_u[0] * b0) / W;
          const Real b_u_y = (By + u_u[1] * b0) / W;
          const Real b_u_z = (Bz + u_u[2] * b0) / W;
          
          // The paper's formula is a coordinate-specific version of integrating |*F^{0i} dSigma_i|.
          // We generalize it by computing the flux vector *F^{0i} = b^0 u^i - b^i u^0
          const Real flux_vec_x = b0 * u_u[0] - b_u_x * u0;
          const Real flux_vec_y = b0 * u_u[1] - b_u_y * u0;
          const Real flux_vec_z = b0 * u_u[2] - b_u_z * u0;

          // Contract with the surface element dSigma_i
          const Real flux_dot_dS = flux_vec_x * dSx + flux_vec_y * dSy + flux_vec_z * dSz;
          
          // Add the absolute value with 1/2 factor, as per the paper's formula
          update.phiB += 0.5 * Kokkos::fabs(flux_dot_dS);
        } // End scope for magnetic flux variables

        // 6. Surface Area (area)
        const Real dSu[3] = {g_uu[0]*dSx + g_uu[1]*dSy + g_uu[2]*dSz,
                             g_uu[1]*dSx + g_uu[3]*dSy + g_uu[4]*dSz,
                             g_uu[2]*dSx + g_uu[4]*dSy + g_uu[5]*dSz};
        const Real dS_sq = dSx*dSu[0] + dSy*dSu[1] + dSz*dSu[2];
        update.area += (dS_sq > 0.0) ? sqrt(dS_sq) : 0.0;

      }, Kokkos::Sum<FluxAccumulator>(total_fluxes));

    pdata->hdata[nflux*g+0] = total_fluxes.mdot;   pdata->hdata[nflux*g+1] = total_fluxes.edot;
    pdata->hdata[nflux*g+2] = total_fluxes.ldot;   pdata->hdata[nflux*g+3] = total_fluxes.phiB;
    pdata->hdata[nflux*g+4] = total_fluxes.pdot_x; pdata->hdata[nflux*g+5] = total_fluxes.pdot_y;
    pdata->hdata[nflux*g+6] = total_fluxes.pdot_z;
    pdata->hdata[nflux*g+7] = total_fluxes.area;
  }

  for (int n=pdata->nhist; n<NHISTORY_VARIABLES; ++n) pdata->hdata[n] = 0.0;
}