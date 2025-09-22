//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file srcterms_newdt.cpp
//! \brief function to compute timestep for source terms across all MeshBlock(s) in a
//! MeshBlockPack

#include <float.h>

#include <limits>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "coordinates/cell_locations.hpp"
#include "ismcooling.hpp"
#include "srcterms.hpp"
#include "units/units.hpp"
#include "cooling_tables.hpp"

//----------------------------------------------------------------------------------------
//! \fn void SourceTerms::NewTimeStep()
//! \brief Compute new timestep for source terms.

void SourceTerms::NewTimeStep(const DvceArray5D<Real> &w0, const EOS_Data &eos_data) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, nx1 = indcs.nx1;
  int js = indcs.js, nx2 = indcs.nx2;
  int ks = indcs.ks, nx3 = indcs.nx3;
  
  // Get nscalars and nhydro/nmhd from the appropriate physics module
  int nscalars = 0;
  int nhydro = 0;
  if (pmy_pack->phydro != nullptr) {
    nscalars = pmy_pack->phydro->nscalars;
    nhydro = pmy_pack->phydro->nhydro;
  } else if (pmy_pack->pmhd != nullptr) {
    nscalars = pmy_pack->pmhd->nscalars;
    nhydro = pmy_pack->pmhd->nmhd;  // For MHD, use nmhd instead of nhydro
  }

  const int nmkji = (pmy_pack->nmb_thispack)*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1;
  const int nji  = nx2*nx1;
  dtnew = static_cast<Real>(std::numeric_limits<float>::max());

  if (ism_cooling) {
    Real use_e = eos_data.use_e;
    Real gamma = eos_data.gamma;
    Real gm1 = gamma - 1.0;
    Real heating_rate = hrate;
    Real temp_unit = pmy_pack->punit->temperature_cgs();
    Real n_unit = pmy_pack->punit->density_cgs()/pmy_pack->punit->mu()
                  / pmy_pack->punit->atomic_mass_unit_cgs;
    Real cooling_unit = pmy_pack->punit->pressure_cgs()/pmy_pack->punit->time_cgs()
                        / n_unit/n_unit;
    Real heating_unit = pmy_pack->punit->pressure_cgs()/pmy_pack->punit->time_cgs()
                        / n_unit;

    // find smallest (e/cooling_rate) in each cell
    Kokkos::parallel_reduce("srcterms_cooling_newdt",
                            Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
    KOKKOS_LAMBDA(const int &idx, Real &min_dt) {
      // compute m,k,j,i indices of thread and call function
      int m = (idx)/nkji;
      int k = (idx - m*nkji)/nji;
      int j = (idx - m*nkji - k*nji)/nx1;
      int i = (idx - m*nkji - k*nji - j*nx1) + is;
      k += ks;
      j += js;

      // temperature in cgs unit
      Real temp = 1.0;
      Real eint = 1.0;
      if (use_e) {
        temp = temp_unit*w0(m,IEN,k,j,i)/w0(m,IDN,k,j,i)*gm1;
        eint = w0(m,IEN,k,j,i);
      } else {
        temp = temp_unit*w0(m,ITM,k,j,i);
        eint = w0(m,ITM,k,j,i)*w0(m,IDN,k,j,i)/gm1;
      }

      Real lambda_cooling = ISMCoolFn(temp)/cooling_unit;
      Real gamma_heating = heating_rate/heating_unit;

      // add a tiny number
      Real cooling_heating = FLT_MIN + fabs(w0(m,IDN,k,j,i) *
                             (w0(m,IDN,k,j,i) * lambda_cooling - gamma_heating));

      min_dt = fmin((eint/cooling_heating), min_dt);
    }, Kokkos::Min<Real>(dtnew));
  }
 
  if (cgm_cooling) {
    auto &size = pmy_pack->pmb->mb_size;
    int nmb1 = pmy_pack->nmb_thispack - 1;
    int nx1 = indcs.nx1;
    int nx2 = indcs.nx2;
    int nx3 = indcs.nx3;

    Real gamma = eos_data.gamma;
    Real gm1 = gamma - 1.0;

    auto &units = pmy_pack->punit;
    Real temp_unit = units->temperature_cgs();
    Real nH_unit = units->density_cgs()/units->atomic_mass_unit_cgs;
    Real cooling_unit = units->pressure_cgs()/units->time_cgs()/nH_unit/nH_unit;
    Real heating_unit = units->pressure_cgs()/units->time_cgs()/nH_unit;
    Real length_unit = units->length_cgs();

    auto Tbins_ = Tbins.d_view;
    auto nHbins_ = nHbins.d_view;
    auto Metal_Cooling_ = Metal_Cooling.d_view;
    auto H_He_Cooling_ = H_He_Cooling.d_view;
    auto Metal_Cooling_CIE_ = Metal_Cooling_CIE.d_view;
    auto H_He_Cooling_CIE_ = H_He_Cooling_CIE.d_view;

    auto Tfloor  = Tbins_ARR[0];
    auto Tceil   = Tbins_ARR[Tbins_DIM_0 - 1];
    auto nHfloor = nHbins_ARR[0];
    auto nHceil  = nHbins_ARR[nHbins_DIM_0 - 1];

    Real X = 0.75; // Hydrogen mass fraction
    Real Zsol = 0.02; // Solar metallicity

    Real h_rate = hrate;
    Real h_norm = hscale_norm;
    Real h_height = hscale_height;
    Real h_radius = hscale_radius;
    Real T_max_ = T_max;

    // find smallest (e/cooling_rate) in each cell
    Kokkos::parallel_reduce("srcterms_cooling_newdt",
                            Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
    KOKKOS_LAMBDA(const int &idx, Real &min_dt) {
      // compute m,k,j,i indices of thread and call function
      int m = (idx)/nkji;
      int k = (idx - m*nkji)/nji;
      int j = (idx - m*nkji - k*nji)/nx1;
      int i = (idx - m*nkji - k*nji - j*nx1) + is;
      k += ks;
      j += js;

      const Real rho = w0(m,IDN,k,j,i);
      const Real egy = w0(m,IEN,k,j,i);
      const Real temp = temp_unit * egy / rho * gm1;

      const Real m_lowT = (temp <  Tfloor) ? 1.0 : 0.0;
      const Real m_Tin  = (temp >= Tfloor && temp <= Tceil) ? 1.0 : 0.0;

      const Real nH = X * nH_unit * rho; // density in cgs units
      const Real Z = w0(m,nhydro,k,j,i) / Zsol; // Assumes Z is the first passive scalar
      const Real log_temp = log10(temp);
      const Real log_nH = log10(nH);

      const Real m_Nin  = (nH >= nHfloor && nH <= nHceil) ? 1.0 : 0.0;
      const Real m_PIE  = m_Tin * m_Nin;   // 1 only if both in-range
      const Real m_CIE  = m_Tin;           // 1 if T in-range

      // Indices
      int iT = 0, jN = 0;
      while (iT < Tbins_DIM_0 - 2 && Tbins_(iT + 1) < log_temp) ++iT;
      while (jN < nHbins_DIM_0 - 2 && nHbins_(jN + 1) < log_nH) ++jN;

      // --- Weights (well-defined even if masks are 0)
      const Real log_T0 = Tbins_(iT);
      const Real log_T1 = Tbins_(iT + 1);
      const Real inv_dT = 1.0 / (log_T1 - log_T0);
      const Real t      = (log_temp - log_T0) * inv_dT;
      const Real omt    = 1.0 - t;

      const Real log_n0 = nHbins_(jN);
      const Real log_n1 = nHbins_(jN + 1);
      const Real inv_dn = 1.0 / (log_n1 - log_n0);
      const Real u      = (log_nH - log_n0) * inv_dn;
      const Real omu    = 1.0 - u;

      // --- PIE bilinear interpolation
      // WiersmaCooling at redshift z = 0 taken from Wiersma et al (2009)
      const Real prim_PIE =
          omt*omu*H_He_Cooling_(iT,   jN  ) +
          t  *omu*H_He_Cooling_(iT+1, jN  ) +
          omt*u  *H_He_Cooling_(iT,   jN+1) +
          t  *u  *H_He_Cooling_(iT+1, jN+1);

      const Real metal_PIE =
          omt*omu*Metal_Cooling_(iT,   jN  ) +
          t  *omu*Metal_Cooling_(iT+1, jN  ) +
          omt*u  *Metal_Cooling_(iT,   jN+1) +
          t  *u  *Metal_Cooling_(iT+1, jN+1);

      const Real lambda_PIE = m_PIE * fma(Z, metal_PIE, prim_PIE);

      // --- CIE linear interpolation
      const Real C0 = H_He_Cooling_CIE_(iT);
      const Real M0 = Metal_Cooling_CIE_(iT);
      const Real prim_CIE  = fma(t, H_He_Cooling_CIE_(iT+1) - C0, C0);
      const Real metal_CIE = fma(t, Metal_Cooling_CIE_(iT+1) - M0, M0);
      const Real lambda_CIE_tab = fma(Z, metal_CIE, prim_CIE);

      // --- Low-T analytic fit
      // for temperatures less than 100 K, use Koyama & Inutsuka (2002)
      const Real e1 = exp(-1.184e5 / (temp + 1.0e3));
      const Real e2 = exp(-92.0 / temp);
      const Real lambda_lowT = Z * (2.0e-19 * e1 + 2.8e-28 * sqrt(temp) * e2);

      // Blend CIE regimes without branches:
      // if T in-table -> use lambda_CIE_tab, else if lowT -> use lambda_lowT, else 0
      const Real lambda_CIE = m_CIE * lambda_CIE_tab + (1.0 - m_CIE) * (m_lowT * lambda_lowT);

      // --- Heating profile
      const Real x1min = size.d_view(m).x1min, x1max = size.d_view(m).x1max;
      const Real x2min = size.d_view(m).x2min, x2max = size.d_view(m).x2max;
      const Real x3min = size.d_view(m).x3min, x3max = size.d_view(m).x3max;

      const Real x1v = CellCenterX(i - is, nx1, x1min, x1max);
      const Real x2v = CellCenterX(j - js, nx2, x2min, x2max);
      const Real x3v = CellCenterX(k - ks, nx3, x3min, x3max);

      const Real R2 = fma(x1v, x1v, x2v*x2v);
      const Real R  = sqrt(R2);

      const Real horz_falloff = exp(-R / h_radius);
      const Real vert_scale2  = h_height*h_height*(1 + R2 / (h_radius*h_radius));
      const Real vert_falloff = exp(-(x3v*x3v) / vert_scale2);

      Real gamma_heating = h_rate * h_norm * X * nH_unit * horz_falloff * vert_falloff;

      // power cutoff: (temp>1e4) ? *= (1e4/temp)^8 : *= 1
      const Real m_hot = (temp > 1.0e4) ? 1.0 : 0.0;
      const Real inv_ratio = 1.0e4 / temp;
      const Real damp_factor =
          m_hot * inv_ratio*inv_ratio*inv_ratio*inv_ratio *
                  inv_ratio*inv_ratio*inv_ratio*inv_ratio
        + (1.0 - m_hot); // equals 1 if not hot
      gamma_heating *= damp_factor;

      // --- Shielding mix
      const Real dx_cgs = size.d_view(m).dx1 * length_unit;
      const Real neutral_frac = 1.0 - 0.5 * (1.0 + tanh((temp - 8e3) / 1.5e3));
      const Real tau  = neutral_frac * nH * 1.0e-17 * dx_cgs;
      const Real frac = exp(-tau);
      const Real lambda_cooling = (1.0 - frac) * lambda_CIE + frac * lambda_PIE;
      //gamma_heating *= (1.0 - frac);

      // --- Timestep calculation
      Real cooling_heating =
        FLT_MIN + fabs(X * rho * ( X * rho * (lambda_cooling / cooling_unit)
                                            - (gamma_heating / heating_unit)));

      min_dt = fmin(0.25 * (egy/cooling_heating), min_dt);
    }, Kokkos::Min<Real>(dtnew));
    
  }

  if (rel_cooling) {
    Real use_e = eos_data.use_e;
    Real gamma = eos_data.gamma;
    Real gm1 = gamma - 1.0;
    Real cooling_rate = crate_rel;
    Real cooling_power = cpower_rel;

    // find smallest (e/cooling_rate) in each cell
    Kokkos::parallel_reduce("srcterms_cooling_newdt",
                            Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
    KOKKOS_LAMBDA(const int &idx, Real &min_dt) {
      // compute m,k,j,i indices of thread and call function
      int m = (idx)/nkji;
      int k = (idx - m*nkji)/nji;
      int j = (idx - m*nkji - k*nji)/nx1;
      int i = (idx - m*nkji - k*nji - j*nx1) + is;
      k += ks;
      j += js;

      // temperature in cgs unit
      Real temp = 1.0;
      Real eint = 1.0;
      if (use_e) {
        temp = w0(m,IEN,k,j,i)/w0(m,IDN,k,j,i)*gm1;
        eint = w0(m,IEN,k,j,i);
      } else {
        temp = w0(m,ITM,k,j,i);
        eint = w0(m,ITM,k,j,i)*w0(m,IDN,k,j,i)/gm1;
      }

      auto &ux = w0(m, IVX, k, j, i);
      auto &uy = w0(m, IVY, k, j, i);
      auto &uz = w0(m, IVZ, k, j, i);

      auto ut = 1. + ux * ux + uy * uy + uz * uz;
      ut = sqrt(ut);

      // The following should be approximately correct
      // add a tiny number
      Real cooling_heating = FLT_MIN + fabs(w0(m,IDN,k,j,i) * ut *
                             pow(temp*cooling_rate, cooling_power));

      min_dt = fmin((eint/cooling_heating), min_dt);
    }, Kokkos::Min<Real>(dtnew));
  }

  return;
}
