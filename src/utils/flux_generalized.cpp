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
#include "coordinates/coordinates.hpp"
#include "dyn_grmhd/valencia_stress_energy.hpp"

// Expanded accumulator for separated fluid/EM fluxes and all axes
struct FluxAccumulator {
  Real mdot = 0, edot_fluid = 0, edot_em = 0, phiB = 0, area = 0;
  // Linear momentum fluxes (fluid, EM)
  Real pdot_x_fluid = 0, pdot_y_fluid = 0, pdot_z_fluid = 0;
  Real pdot_x_em    = 0, pdot_y_em    = 0, pdot_z_em    = 0;
  // Angular momentum fluxes (fluid, EM for x, y, z axes)
  Real ldot_x_fluid = 0, ldot_y_fluid = 0, ldot_z_fluid = 0;
  Real ldot_x_em    = 0, ldot_y_em    = 0, ldot_z_em    = 0;

  KOKKOS_INLINE_FUNCTION
  FluxAccumulator& operator+=(const FluxAccumulator& src) {
    mdot        += src.mdot;
    edot_fluid  += src.edot_fluid;
    edot_em     += src.edot_em;
    phiB        += src.phiB;
    area        += src.area;
    pdot_x_fluid += src.pdot_x_fluid;
    pdot_y_fluid += src.pdot_y_fluid;
    pdot_z_fluid += src.pdot_z_fluid;
    pdot_x_em    += src.pdot_x_em;
    pdot_y_em    += src.pdot_y_em;
    pdot_z_em    += src.pdot_z_em;
    ldot_x_fluid += src.ldot_x_fluid;
    ldot_y_fluid += src.ldot_y_fluid;
    ldot_z_fluid += src.ldot_z_fluid;
    ldot_x_em    += src.ldot_x_em;
    ldot_y_em    += src.ldot_y_em;
    ldot_z_em    += src.ldot_z_em;
    return *this;
  }
};

namespace Kokkos {
  template<>
  struct reduction_identity<FluxAccumulator> {
    KOKKOS_INLINE_FUNCTION
    static FluxAccumulator sum() { return FluxAccumulator{}; }
  };
} // namespace Kokkos

//----------------------------------------------------------------------------------------
// Main Function Definition (no EOS templates; gamma-law assumed)
//----------------------------------------------------------------------------------------
void TorusFluxes_General(HistoryData *pdata,
                         MeshBlockPack *pmbp,
                         const std::vector<SphericalSurfaceGrid*>& surfs) {
  //--- 1. Validation and Setup ---
  if (pmbp->padm == nullptr || pmbp->pmhd == nullptr || pmbp->pdyngr == nullptr) {
    std::cerr << "### FATAL: TorusFluxes_General requires ADM, MHD, and DynGRMHD objects."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }

  if (surfs.empty()) {
    pdata->nhist = 0;
    return;
  }

  const int nflux = 17; // mdot, edot_f, edot_em, px_f,py_f,pz_f, px_em,py_em,pz_em,
                        // lxdot_f,lydot_f,lzdot_f, lxdot_em,lydot_em,lzdot_em, phiB, area
  pdata->nhist = static_cast<int>(surfs.size()) * nflux;

  if (pdata->nhist > NHISTORY_VARIABLES) {
    std::cerr << "### FATAL: pdata->nhist exceeds NHISTORY_VARIABLES" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // Set history output labels
  for (int g = 0; g < static_cast<int>(surfs.size()); ++g) {
    std::string s_name = surfs[g]->Label();
    int i = nflux*g;
    pdata->label[i++] = "mdot_"     + s_name;
    pdata->label[i++] = "edot_f_"   + s_name;
    pdata->label[i++] = "edot_em_"  + s_name;
    pdata->label[i++] = "pxdot_f_"  + s_name;
    pdata->label[i++] = "pydot_f_"  + s_name;
    pdata->label[i++] = "pzdot_f_"  + s_name;
    pdata->label[i++] = "pxdot_em_" + s_name;
    pdata->label[i++] = "pydot_em_" + s_name;
    pdata->label[i++] = "pzdot_em_" + s_name;
    pdata->label[i++] = "lxdot_f_"  + s_name;
    pdata->label[i++] = "lydot_f_"  + s_name;
    pdata->label[i++] = "lzdot_f_"  + s_name;
    pdata->label[i++] = "lxdot_em_" + s_name;
    pdata->label[i++] = "lydot_em_" + s_name;
    pdata->label[i++] = "lzdot_em_" + s_name;
    pdata->label[i++] = "phiB_"     + s_name;
    pdata->label[i++] = "area_"     + s_name;
  }

  // Gamma-law EOS parameter; must match valencia_stress_energy.hpp usage
  const Real gamma_gas = pmbp->pmhd->peos->eos_data.gamma;

  DualArray2D<Real> dSigma;

  //--- 2. Loop over surfaces ---
  for (int g = 0; g < static_cast<int>(surfs.size()); ++g) {
    auto current_surf = surfs[g];

    if (pmbp->pmesh->adaptive) {
      current_surf->RebuildAll();
    }

    const int np = current_surf->Npts();

    // Interpolate primitives and densitized B to the surface
    const int nvars_mhd = pmbp->pmhd->nmhd + pmbp->pmhd->nscalars;
    DualArray2D<Real> w_vals =
        current_surf->InterpolateToSurface(pmbp->pmhd->w0, 0, nvars_mhd);
    auto w_vals_d = w_vals.d_view;

    DualArray2D<Real> B_vals =
        current_surf->InterpolateToSurface(pmbp->pmhd->bcc0, 0, 3);
    auto B_vals_d = B_vals.d_view;

    // Lapse/shift from ADM or Z4C
    auto gauge_source_array = pmbp->padm->u_adm;
    int ab_start = adm::ADM::I_ADM_ALPHA;
    int ab_end   = adm::ADM::I_ADM_BETAZ + 1;
    if (pmbp->pz4c != nullptr) {
      gauge_source_array = pmbp->pz4c->u0;
      ab_start = z4c::Z4c::I_Z4C_ALPHA;
      ab_end   = z4c::Z4c::I_Z4C_BETAZ + 1;
    }
    DualArray2D<Real> ab_vals =
        current_surf->InterpolateToSurface(gauge_source_array, ab_start, ab_end);
    auto ab_vals_d = ab_vals.d_view;

    // Cartesian coordinates and interpolation indices
    auto coords     = current_surf->Coords().d_view;
    auto interp_ind = current_surf->InterpIndices().d_view;

    // Surface metric and dSigma_i
    current_surf->InterpolateMetric();
    current_surf->BuildSurfaceCovectors(dSigma);
    auto g_surf_d = current_surf->Metric().d_view;
    auto dSigma_d = dSigma.d_view;

    FluxAccumulator fluxes;

    //--- 3. Parallel integration on the surface ---
    Kokkos::parallel_reduce("flux_integral_from_prims",
      Kokkos::RangePolicy<DevExeSpace>(0, np),
      KOKKOS_LAMBDA(const int p, FluxAccumulator& update) {
        if (interp_ind(p,0) == -1) return;

        // --- A. Local primitives and geometry ---
        const Real rho  = w_vals_d(p, IDN);
        const Real pgas = w_vals_d(p, IPR);

        Real prim_Wv_u[3] = {
          w_vals_d(p, IVX),
          w_vals_d(p, IVY),
          w_vals_d(p, IVZ)
        };

        const Real alp = ab_vals_d(p, 0);
        const Real beta_u[3] = {
          ab_vals_d(p, 1),
          ab_vals_d(p, 2),
          ab_vals_d(p, 3)
        };

        // Spatial metric gamma_ij on the surface: store as 1D symmetric array
        Real g_dd_1d[6];
        g_dd_1d[0] = g_surf_d(p,0);
        g_dd_1d[1] = g_surf_d(p,1);
        g_dd_1d[2] = g_surf_d(p,2);
        g_dd_1d[3] = g_surf_d(p,3);
        g_dd_1d[4] = g_surf_d(p,4);
        g_dd_1d[5] = g_surf_d(p,5);

        // det(gamma)  (use ADM helper; minimal change requested)
        Real detg = adm::SpatialDet(g_dd_1d[0], g_dd_1d[1], g_dd_1d[2],
                                    g_dd_1d[3], g_dd_1d[4], g_dd_1d[5]);
        if (detg <= 0.0) return;

        // dS_i (covariant surface element)
        const Real dS_d[3] = {
          dSigma_d(p,0),
          dSigma_d(p,1),
          dSigma_d(p,2)
        };

        // Inverse metric g^{ij}
        Real g_uu_1d[6];
        adm::SpatialInv(1.0/detg,
                        g_dd_1d[0], g_dd_1d[1], g_dd_1d[2],
                        g_dd_1d[3], g_dd_1d[4], g_dd_1d[5],
                        &g_uu_1d[0], &g_uu_1d[1], &g_uu_1d[2],
                        &g_uu_1d[3], &g_uu_1d[4], &g_uu_1d[5]);

        Real g_u[3][3];
        g_u[0][0] = g_uu_1d[0];
        g_u[0][1] = g_uu_1d[1];
        g_u[0][2] = g_uu_1d[2];
        g_u[1][0] = g_uu_1d[1];
        g_u[1][1] = g_uu_1d[3];
        g_u[1][2] = g_uu_1d[4];
        g_u[2][0] = g_uu_1d[2];
        g_u[2][1] = g_uu_1d[4];
        g_u[2][2] = g_uu_1d[5];

        // --- B. Eulerian velocity and Eulerian B from Valencia helpers ---
        Real W, v_u[3], v_d[3];
        ValenciaPrimsToEulerianVelocity(prim_Wv_u, g_dd_1d, W, v_u, v_d);

        Real B_tilde_u[3] = {
          B_vals_d(p,0),
          B_vals_d(p,1),
          B_vals_d(p,2)
        };

        Real sqrt_gamma_surf, B_u[3], B_d[3];
        ValenciaEulerianMagneticFromDensitized(B_tilde_u, g_dd_1d,
                                               sqrt_gamma_surf, B_u, B_d);

        // Comoving magnetic field
        Real Bv, b0, b_u[3], b_d[3], b2;
        ValenciaComovingB(v_u, v_d, B_u, B_d, W, Bv, b0, b_u, b_d, b2);

        // --- C. Decomposed stress-energy (fluid + EM) from one source of truth ---
        Real E_fluid, E_em;
        Real S_fluid_d[3], S_em_d[3];
        Real S_fluid_dd[3][3], S_em_dd[3][3];

        ValenciaStressEnergyDecomposed(rho, pgas, gamma_gas,
                                       v_u, v_d, W,
                                       B_u, B_d,
                                       b0, b_d, b2,
                                       g_dd_1d,
                                       E_fluid, E_em,
                                       S_fluid_d, S_em_d,
                                       S_fluid_dd, S_em_dd);

        const Real E_tot = E_fluid + E_em;

        // Raise S_j to S^i = g^{ij} S_j
        Real S_fluid_u[3] = {0.0,0.0,0.0};
        Real S_em_u[3]    = {0.0,0.0,0.0};
        for (int i = 0; i < 3; ++i) {
          for (int j = 0; j < 3; ++j) {
            S_fluid_u[i] += g_u[i][j] * S_fluid_d[j];
            S_em_u[i]    += g_u[i][j] * S_em_d[j];
          }
        }

        // Mixed stresses S^i_j = g^{ik} S_kj
        Real S_fluid_ud[3][3], S_em_ud[3][3];
        for (int i = 0; i < 3; ++i) {
          for (int j = 0; j < 3; ++j) {
            Real val_f = 0.0;
            Real val_em = 0.0;
            for (int k = 0; k < 3; ++k) {
              val_f  += g_u[i][k] * S_fluid_dd[k][j];
              val_em += g_u[i][k] * S_em_dd[k][j];
            }
            S_fluid_ud[i][j] = val_f;
            S_em_ud[i][j]    = val_em;
          }
        }

        // --- D. Mass flux ---
        Real u_u[3] = {
          W * (v_u[0] - beta_u[0]/alp),
          W * (v_u[1] - beta_u[1]/alp),
          W * (v_u[2] - beta_u[2]/alp)
        };
        const Real mass_flux_vec[3] = {
          rho * u_u[0],
          rho * u_u[1],
          rho * u_u[2]
        };

        // Negative sign for inward accretion convention
        update.mdot -= (mass_flux_vec[0]*dS_d[0]
                      + mass_flux_vec[1]*dS_d[1]
                      + mass_flux_vec[2]*dS_d[2]);

        // --- E. Energy fluxes (fluid + EM) ---
        Real FE_fluid[3], FE_em[3];
        for (int i = 0; i < 3; ++i) {
          FE_fluid[i] = alp * S_fluid_u[i] - beta_u[i] * E_fluid;
          FE_em[i]    = alp * S_em_u[i]    - beta_u[i] * E_em;
        }

        update.edot_fluid -= (FE_fluid[0]*dS_d[0]
                            + FE_fluid[1]*dS_d[1]
                            + FE_fluid[2]*dS_d[2]);
        update.edot_em    -= (FE_em[0]*dS_d[0]
                            + FE_em[1]*dS_d[1]
                            + FE_em[2]*dS_d[2]);

        // --- F. Momentum fluxes ---
        Real FS_fluid[3][3], FS_em[3][3];
        for (int i = 0; i < 3; ++i) {
          for (int j = 0; j < 3; ++j) {
            FS_fluid[i][j] = alp * S_fluid_ud[i][j] - beta_u[i] * S_fluid_d[j];
            FS_em[i][j]    = alp * S_em_ud[i][j]    - beta_u[i] * S_em_d[j];
          }
        }

        Real d_pdot_fluid[3] = {0.0,0.0,0.0};
        Real d_pdot_em[3]    = {0.0,0.0,0.0};
        for (int j = 0; j < 3; ++j) {
          for (int i = 0; i < 3; ++i) {
            d_pdot_fluid[j] += FS_fluid[i][j]*dS_d[i];
            d_pdot_em[j]    += FS_em[i][j]   *dS_d[i];
          }
        }

        // Store linear momentum fluxes
        update.pdot_x_fluid += d_pdot_fluid[0];
        update.pdot_y_fluid += d_pdot_fluid[1];
        update.pdot_z_fluid += d_pdot_fluid[2];

        update.pdot_x_em    += d_pdot_em[0];
        update.pdot_y_em    += d_pdot_em[1];
        update.pdot_z_em    += d_pdot_em[2];

        // --- G. Angular-momentum fluxes (MINIMAL FIX)
        // Angular momentum density uses J_a = ε_{aℓm} x^ℓ S_m (covariant S_m).
        // Therefore flux through surface is
        //   Ldot_a = ∮ ε_{aℓm} x^ℓ (F^j{}_m dΣ_j)
        // and we already computed A_m := (F^j{}_m dΣ_j) as d_pdot_*[m] above.
        {
          const Real x = coords(p,0);
          const Real y = coords(p,1);
          const Real z = coords(p,2);

          // A_m = F^j{}_m dΣ_j  (covariant in m)
          const Real A_fluid_d[3] = { d_pdot_fluid[0], d_pdot_fluid[1], d_pdot_fluid[2] };
          const Real A_em_d[3]    = { d_pdot_em[0],    d_pdot_em[1],    d_pdot_em[2]    };

          // Ldot = r × A  (same convention as your angular_momentum density block)
          update.ldot_x_fluid += (y * A_fluid_d[2] - z * A_fluid_d[1]);
          update.ldot_y_fluid += (z * A_fluid_d[0] - x * A_fluid_d[2]);
          update.ldot_z_fluid += (x * A_fluid_d[1] - y * A_fluid_d[0]);

          update.ldot_x_em    += (y * A_em_d[2] - z * A_em_d[1]);
          update.ldot_y_em    += (z * A_em_d[0] - x * A_em_d[2]);
          update.ldot_z_em    += (x * A_em_d[1] - y * A_em_d[0]);
        }

        // --- H. Phi_B (magnetic flux) ---
        {
          // b^μ: using comoving B pieces from ValenciaComovingB
          const Real u_u_t = W / alp;  // u^t (Eulerian)
          const Real b_u_t = Bv / alp; // b^t

          Real b_u_fields[3];
          for (int i = 0; i < 3; ++i) {
            b_u_fields[i] = B_u[i]/W + Bv * v_u[i];
          }

          const Real M_u[3] = {
            u_u_t * b_u_fields[0] - u_u[0] * b_u_t,
            u_u_t * b_u_fields[1] - u_u[1] * b_u_t,
            u_u_t * b_u_fields[2] - u_u[2] * b_u_t
          };

          const Real flux_dot_dS =
            M_u[0]*dS_d[0] + M_u[1]*dS_d[1] + M_u[2]*dS_d[2];
          update.phiB += 0.5 * Kokkos::fabs(flux_dot_dS);
        }

        // --- I. Surface area ---
        Real dS_u[3] = {0.0,0.0,0.0};
        for (int i = 0; i < 3; ++i) {
          for (int j = 0; j < 3; ++j) {
            dS_u[i] += g_u[i][j]*dS_d[j];
          }
        }
        const Real dS_sq = dS_d[0]*dS_u[0] + dS_d[1]*dS_u[1] + dS_d[2]*dS_u[2];
        update.area += (dS_sq > 0.0 ? sqrt(dS_sq) : 0.0);
      },
      Kokkos::Sum<FluxAccumulator>(fluxes));

    //--- 4. Store the final integrated values in the history data array ---
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

  // Zero out unused history slots
  for (int n = pdata->nhist; n < NHISTORY_VARIABLES; ++n) {
    pdata->hdata[n] = 0.0;
  }
}
