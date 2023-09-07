//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_source.cpp

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "driver/driver.hpp"
#include "coordinates/cartesian_ks.hpp"
#include "coordinates/coordinates.hpp"
#include "coordinates/cell_locations.hpp"
#include "geodesic-grid/geodesic_grid.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "units/units.hpp"
#include "radiation.hpp"

#include "radiation/radiation_tetrad.hpp"
#include "radiation/radiation_opacities.hpp"

namespace radiation {

KOKKOS_INLINE_FUNCTION
bool FourthPolyRoot(const Real coef4, const Real tconst, Real &root);

//----------------------------------------------------------------------------------------
//! \fn TaskStatus Radiation::AddRadiationSourceTerm(Driver *pdriver, int stage)
// \brief Add implicit radiation source term.  Based off of @c-white and @yanfeij's gr_rad
// branch, radiation/coupling/emission.cpp commit be7f84565b.

TaskStatus Radiation::AddRadiationSourceTerm(Driver *pdriver, int stage) {
  // Return if radiation source term disabled
  if (!(rad_source)) {
    return TaskStatus::complete;
  }

  // Extract indices, size data, hydro/mhd/units flags, and coupling flags
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int &is = indcs.is, &ie = indcs.ie;
  int &js = indcs.js, &je = indcs.je;
  int &ks = indcs.ks, &ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  int nang1 = prgeo->nangles - 1;
  auto &size = pmy_pack->pmb->mb_size;
  bool &is_hydro_enabled_ = is_hydro_enabled;
  bool &is_mhd_enabled_ = is_mhd_enabled;
  bool &are_units_enabled_ = are_units_enabled;
  bool &is_compton_enabled_ = is_compton_enabled;
  bool &fixed_fluid_ = fixed_fluid;
  bool &affect_fluid_ = affect_fluid;

  // Extract coordinate/excision data
  auto &coord = pmy_pack->pcoord->coord_data;
  bool &flat = coord.is_minkowski;
  Real &spin = coord.bh_spin;
  bool &excise = pmy_pack->pcoord->coord_data.bh_excise;
  auto &rad_mask_ = pmy_pack->pcoord->excision_floor;
  Real &n_0_floor_ = n_0_floor;

  // Extract radiation constant and units
  Real &arad_ = arad;
  Real density_scale_ = 1.0, temperature_scale_ = 1.0, length_scale_ = 1.0;
  Real mean_mol_weight_ = 1.0;
  Real rosseland_coef_ = 1.0, planck_minus_rosseland_coef_ = 0.0;
  Real inv_t_electron_ = 1.0;
  if (are_units_enabled_) {
    density_scale_ = pmy_pack->punit->density_cgs();
    temperature_scale_ = pmy_pack->punit->temperature_cgs();
    length_scale_ = pmy_pack->punit->length_cgs();
    mean_mol_weight_ = pmy_pack->punit->mu();
    rosseland_coef_ = pmy_pack->punit->rosseland_coef_cgs;
    planck_minus_rosseland_coef_ = pmy_pack->punit->planck_minus_rosseland_coef_cgs;
    inv_t_electron_ = temperature_scale_/pmy_pack->punit->electron_rest_mass_energy_cgs;
  }

  // Extract adiabatic index
  Real gm1;
  if (is_hydro_enabled_) {
    gm1 = pmy_pack->phydro->peos->eos_data.gamma - 1.0;
  } else if (is_mhd_enabled_) {
    gm1 = pmy_pack->pmhd->peos->eos_data.gamma - 1.0;
  }

  // Extract radiation, radiation frame, and radiation angular mesh data
  auto &i0_ = i0;
  Real &kappa_a_ = kappa_a;
  Real &kappa_s_ = kappa_s;
  Real &kappa_p_ = kappa_p;
  bool &power_opacity_ = power_opacity;
  auto &nh_c_ = nh_c;
  auto &tt = tet_c;
  auto &tc = tetcov_c;
  auto &norm_to_tet_ = norm_to_tet;
  auto &solid_angles_ = prgeo->solid_angles;

  // Extract hydro/mhd quantities
  DvceArray5D<Real> u0_, w0_;
  if (is_hydro_enabled_) {
    u0_ = pmy_pack->phydro->u0;
    w0_ = pmy_pack->phydro->w0;
  } else if (is_mhd_enabled_) {
    u0_ = pmy_pack->pmhd->u0;
    w0_ = pmy_pack->pmhd->w0;
  }

  // Extract timestep
  Real dt_ = (pdriver->beta[stage-1])*(pmy_pack->pmesh->dt);

  // Call ConsToPrim over active zones prior to source term application
  if (!(fixed_fluid_)) {
    if (is_hydro_enabled_) {
      pmy_pack->phydro->peos->ConsToPrim(u0_,w0_,false,is,ie,js,je,ks,ke);
    } else if (is_mhd_enabled_) {
      auto &b0_ = pmy_pack->pmhd->b0;
      auto &bcc0_ = pmy_pack->pmhd->bcc0;
      pmy_pack->pmhd->peos->ConsToPrim(u0_,b0_,w0_,bcc0_,false,is,ie,js,je,ks,ke);
    }
  }

  // compute implicit source term
  par_for("radiation_source",DevExeSpace(),0,nmb1,ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);

    // compute metric and inverse
    Real glower[4][4], gupper[4][4];
    ComputeMetricAndInverse(x1v,x2v,x3v,flat,spin,glower,gupper);
    Real alpha = sqrt(-1.0/gupper[0][0]);

    // fluid state
    Real &wdn = w0_(m,IDN,k,j,i);
    Real &wvx = w0_(m,IVX,k,j,i);
    Real &wvy = w0_(m,IVY,k,j,i);
    Real &wvz = w0_(m,IVZ,k,j,i);
    Real &wen = w0_(m,IEN,k,j,i);

    // derived quantities
    Real pgas = gm1*wen;
    Real tgas = pgas/wdn;
    Real q = glower[1][1]*wvx*wvx + 2.0*glower[1][2]*wvx*wvy + 2.0*glower[1][3]*wvx*wvz
           + glower[2][2]*wvy*wvy + 2.0*glower[2][3]*wvy*wvz
           + glower[3][3]*wvz*wvz;
    Real gamma = sqrt(1.0 + q);
    Real u0 = gamma/alpha;

    // set opacities
    Real sigma_a, sigma_s, sigma_p;
    OpacityFunction(wdn, density_scale_,
                    tgas, temperature_scale_,
                    length_scale_, gm1, mean_mol_weight_,
                    power_opacity_, rosseland_coef_, planck_minus_rosseland_coef_,
                    kappa_a_, kappa_s_, kappa_p_,
                    sigma_a, sigma_s, sigma_p);
    Real dtcsiga = dt_*sigma_a;
    Real dtcsigs = dt_*sigma_s;
    Real dtcsigp = dt_*sigma_p;
    Real dtaucsiga = dtcsiga/u0;
    Real dtaucsigs = dtcsigs/u0;
    Real dtaucsigp = dtcsigp/u0;

    // compute fluid velocity in tetrad frame
    Real u_tet[4];
    u_tet[0] = (norm_to_tet_(m,0,0,k,j,i)*gamma + norm_to_tet_(m,0,1,k,j,i)*wvx +
                norm_to_tet_(m,0,2,k,j,i)*wvy   + norm_to_tet_(m,0,3,k,j,i)*wvz);
    u_tet[1] = (norm_to_tet_(m,1,0,k,j,i)*gamma + norm_to_tet_(m,1,1,k,j,i)*wvx +
                norm_to_tet_(m,1,2,k,j,i)*wvy   + norm_to_tet_(m,1,3,k,j,i)*wvz);
    u_tet[2] = (norm_to_tet_(m,2,0,k,j,i)*gamma + norm_to_tet_(m,2,1,k,j,i)*wvx +
                norm_to_tet_(m,2,2,k,j,i)*wvy   + norm_to_tet_(m,2,3,k,j,i)*wvz);
    u_tet[3] = (norm_to_tet_(m,3,0,k,j,i)*gamma + norm_to_tet_(m,3,1,k,j,i)*wvx +
                norm_to_tet_(m,3,2,k,j,i)*wvy   + norm_to_tet_(m,3,3,k,j,i)*wvz);

    // coordinate component n^0
    Real n0 = tt(m,0,0,k,j,i);

    // Calculate polynomial coefficients
    Real wght_sum = 0.0;
    Real suma1 = 0.0;
    Real suma2 = 0.0;
    for (int n=0; n<=nang1; ++n) {
      Real n_0 = tc(m,0,0,k,j,i)*nh_c_.d_view(n,0) + tc(m,1,0,k,j,i)*nh_c_.d_view(n,1) +
                 tc(m,2,0,k,j,i)*nh_c_.d_view(n,2) + tc(m,3,0,k,j,i)*nh_c_.d_view(n,3);
      Real n0_cm = (u_tet[0]*nh_c_.d_view(n,0) - u_tet[1]*nh_c_.d_view(n,1) -
                    u_tet[2]*nh_c_.d_view(n,2) - u_tet[3]*nh_c_.d_view(n,3));
      Real omega_cm = solid_angles_.d_view(n)/SQR(n0_cm);
      Real intensity_cm = 4.0*M_PI*(i0_(m,n,k,j,i)/(n0*n_0))*SQR(SQR(n0_cm));
      Real vncsigma = 1.0/(n0 + (dtcsiga + dtcsigs)*n0_cm);
      Real vncsigma2 = n0_cm*vncsigma;
      Real ir_weight = intensity_cm*omega_cm;
      wght_sum += omega_cm;
      suma1 += omega_cm*vncsigma2;
      suma2 += ir_weight*n0*vncsigma;
    }
    suma1 /= wght_sum;
    suma2 /= wght_sum;
    Real suma3 = suma1*(dtcsigs - dtcsigp);
    suma1 *= (dtcsiga + dtcsigp);

    // compute coefficients
    Real coef[2];
    coef[1] = (dtaucsiga+dtaucsigp-(dtaucsiga+dtaucsigp)*suma1/(1.0-suma3))*arad_*gm1/wdn;
    coef[0] = -tgas-(dtaucsiga+dtaucsigp)*suma2*gm1/(wdn*(1.0-suma3));

    // Calculate new gas temperature
    Real tgasnew = tgas;
    bool badcell = false;
    if (fabs(coef[1]) > 1.0e-20) {
      bool flag = FourthPolyRoot(coef[1], coef[0], tgasnew);
      if (!(flag) || !(isfinite(tgasnew))) {
        badcell = true;
        tgasnew = tgas;
      }
    } else {
      tgasnew = -coef[0];
    }

    // Update the specific intensity
    if (!(badcell)) {
      // Calculate emission coefficient and updated jr_cm
      Real emission = arad_*SQR(SQR(tgasnew));
      Real jr_cm = (suma1*emission + suma2)/(1.0 - suma3);
      Real m_old[4] = {0.0}; Real m_new[4] = {0.0};
      for (int n=0; n<=nang1; ++n) {
        // compute coordinate normal components
        Real n_0 = tc(m,0,0,k,j,i)*nh_c_.d_view(n,0) + tc(m,1,0,k,j,i)*nh_c_.d_view(n,1)
                 + tc(m,2,0,k,j,i)*nh_c_.d_view(n,2) + tc(m,3,0,k,j,i)*nh_c_.d_view(n,3);
        Real n_1 = tc(m,0,1,k,j,i)*nh_c_.d_view(n,0) + tc(m,1,1,k,j,i)*nh_c_.d_view(n,1)
                 + tc(m,2,1,k,j,i)*nh_c_.d_view(n,2) + tc(m,3,1,k,j,i)*nh_c_.d_view(n,3);
        Real n_2 = tc(m,0,2,k,j,i)*nh_c_.d_view(n,0) + tc(m,1,2,k,j,i)*nh_c_.d_view(n,1)
                 + tc(m,2,2,k,j,i)*nh_c_.d_view(n,2) + tc(m,3,2,k,j,i)*nh_c_.d_view(n,3);
        Real n_3 = tc(m,0,3,k,j,i)*nh_c_.d_view(n,0) + tc(m,1,3,k,j,i)*nh_c_.d_view(n,1)
                 + tc(m,2,3,k,j,i)*nh_c_.d_view(n,2) + tc(m,3,3,k,j,i)*nh_c_.d_view(n,3);

        // compute moments before coupling
        m_old[0] += (    i0_(m,n,k,j,i)    *solid_angles_.d_view(n));
        m_old[1] += (n_1*i0_(m,n,k,j,i)/n_0*solid_angles_.d_view(n));
        m_old[2] += (n_2*i0_(m,n,k,j,i)/n_0*solid_angles_.d_view(n));
        m_old[3] += (n_3*i0_(m,n,k,j,i)/n_0*solid_angles_.d_view(n));

        // update intensity
        Real n0_cm = (u_tet[0]*nh_c_.d_view(n,0) - u_tet[1]*nh_c_.d_view(n,1) -
                      u_tet[2]*nh_c_.d_view(n,2) - u_tet[3]*nh_c_.d_view(n,3));
        Real intensity_cm = 4.0*M_PI*(i0_(m,n,k,j,i)/(n0*n_0))*SQR(SQR(n0_cm));
        Real vncsigma = 1.0/(n0 + (dtcsiga + dtcsigs)*n0_cm);
        Real vncsigma2 = n0_cm*vncsigma;
        Real di_cm = ( ((dtcsigs-dtcsigp)*jr_cm
                      + (dtcsiga+dtcsigp)*emission
                      - (dtcsigs+dtcsiga)*intensity_cm)*vncsigma2 );
        i0_(m,n,k,j,i) = n0*n_0*fmax(i0_(m,n,k,j,i)/(n0*n_0) +
                                     di_cm/(4.0*M_PI*SQR(SQR(n0_cm))), 0.0);

        // compute moments after coupling
        m_new[0] += (    i0_(m,n,k,j,i)    *solid_angles_.d_view(n));
        m_new[1] += (n_1*i0_(m,n,k,j,i)/n_0*solid_angles_.d_view(n));
        m_new[2] += (n_2*i0_(m,n,k,j,i)/n_0*solid_angles_.d_view(n));
        m_new[3] += (n_3*i0_(m,n,k,j,i)/n_0*solid_angles_.d_view(n));

        // handle excision
        // NOTE(@pdmullen): The below zeroes all intensities within rks <= r_excision and
        // zeroes intensities within angles where n_0 is about zero. When Compton is
        // enabled, we delay the n_0_floor excision so that intensites updated via
        // absorption and scattering inform the Compton update
        if (excise) {
          bool apply_excision = (rad_mask_(m,k,j,i) ||
                                 (!(is_compton_enabled_) && fabs(n_0) < n_0_floor_));
          if (apply_excision) { i0_(m,n,k,j,i) = 0.0; }
        }
      }
      // update conserved fluid variables
      if (affect_fluid_) {
        u0_(m,IEN,k,j,i) += (m_old[0] - m_new[0]);
        u0_(m,IM1,k,j,i) += (m_old[1] - m_new[1]);
        u0_(m,IM2,k,j,i) += (m_old[2] - m_new[2]);
        u0_(m,IM3,k,j,i) += (m_old[3] - m_new[3]);
      }
    }

    // compton scattering
    if (is_compton_enabled_) {
      // use partially updated gas temperature
      tgas = tgasnew;

      // compute polynomial coefficients using partially updated gas temp and intensity
      suma1 = 0.0;
      Real jr_cm = 0.0;
      for (int n=0; n<=nang1; ++n) {
        Real n_0 = tc(m,0,0,k,j,i)*nh_c_.d_view(n,0) + tc(m,1,0,k,j,i)*nh_c_.d_view(n,1) +
                   tc(m,2,0,k,j,i)*nh_c_.d_view(n,2) + tc(m,3,0,k,j,i)*nh_c_.d_view(n,3);
        Real n0_cm = (u_tet[0]*nh_c_.d_view(n,0) - u_tet[1]*nh_c_.d_view(n,1) -
                      u_tet[2]*nh_c_.d_view(n,2) - u_tet[3]*nh_c_.d_view(n,3));
        Real wght_cm = solid_angles_.d_view(n)/SQR(n0_cm)/wght_sum;
        Real intensity_cm = 4.0*M_PI*(i0_(m,n,k,j,i)/(n0*n_0))*SQR(SQR(n0_cm));
        Real ir_weight = intensity_cm*wght_cm;
        jr_cm += ir_weight;
        suma1 += (n0_cm/n0)*4.0*dtcsigs*inv_t_electron_*wght_cm;
      }
      suma2 = 4.0*dtaucsigs*inv_t_electron_*gm1/wdn;

      // compute partially updated radiation temperature
      Real trad = sqrt(sqrt(jr_cm/arad_));
      const bool temp_equil = (fabs(trad - tgas) < 1.0e-12);

      // Calculate new gas temperature due to Compton
      Real tradnew = trad;
      badcell = false;
      if (!(temp_equil)) {
        coef[1] = (1.0 + suma2*jr_cm)/(suma1*jr_cm)*arad_;
        coef[0] = -(1.0 + suma2*jr_cm)/suma1 - tgas;
        bool flag = FourthPolyRoot(coef[1], coef[0], tradnew);
        if (!(flag) || !(isfinite(tradnew))) {
          badcell = true;
        }
      }

      // Update the specific intensity
      if (!(badcell) && !(temp_equil)) {
        // Compute updated gas temperature
        tgasnew = (arad_*SQR(SQR(tradnew)) - jr_cm)/(suma1*jr_cm) + tradnew;
        Real m_old[4] = {0.0}; Real m_new[4] = {0.0};
        for (int n=0; n<=nang1; ++n) {
          // compute coordinate normal components
          Real n_0 = tc(m,0,0,k,j,i)*nh_c_.d_view(n,0)+tc(m,1,0,k,j,i)*nh_c_.d_view(n,1)
                   + tc(m,2,0,k,j,i)*nh_c_.d_view(n,2)+tc(m,3,0,k,j,i)*nh_c_.d_view(n,3);
          Real n_1 = tc(m,0,1,k,j,i)*nh_c_.d_view(n,0)+tc(m,1,1,k,j,i)*nh_c_.d_view(n,1)
                   + tc(m,2,1,k,j,i)*nh_c_.d_view(n,2)+tc(m,3,1,k,j,i)*nh_c_.d_view(n,3);
          Real n_2 = tc(m,0,2,k,j,i)*nh_c_.d_view(n,0)+tc(m,1,2,k,j,i)*nh_c_.d_view(n,1)
                   + tc(m,2,2,k,j,i)*nh_c_.d_view(n,2)+tc(m,3,2,k,j,i)*nh_c_.d_view(n,3);
          Real n_3 = tc(m,0,3,k,j,i)*nh_c_.d_view(n,0)+tc(m,1,3,k,j,i)*nh_c_.d_view(n,1)
                   + tc(m,2,3,k,j,i)*nh_c_.d_view(n,2)+tc(m,3,3,k,j,i)*nh_c_.d_view(n,3);

          // compute moments before coupling
          m_old[0] += (    i0_(m,n,k,j,i)    *solid_angles_.d_view(n));
          m_old[1] += (n_1*i0_(m,n,k,j,i)/n_0*solid_angles_.d_view(n));
          m_old[2] += (n_2*i0_(m,n,k,j,i)/n_0*solid_angles_.d_view(n));
          m_old[3] += (n_3*i0_(m,n,k,j,i)/n_0*solid_angles_.d_view(n));

          // update intensity
          Real n0_cm = (u_tet[0]*nh_c_.d_view(n,0) - u_tet[1]*nh_c_.d_view(n,1) -
                        u_tet[2]*nh_c_.d_view(n,2) - u_tet[3]*nh_c_.d_view(n,3));
          Real di_cm = (n0_cm/n0)*dtcsigs*4.0*jr_cm*inv_t_electron_*(tgasnew - tradnew);
          i0_(m,n,k,j,i) = n0*n_0*fmax(i0_(m,n,k,j,i)/(n0*n_0) +
                                       di_cm/(4.0*M_PI*SQR(SQR(n0_cm))), 0.0);

          // compute moments after coupling
          m_new[0] += (    i0_(m,n,k,j,i)    *solid_angles_.d_view(n));
          m_new[1] += (n_1*i0_(m,n,k,j,i)/n_0*solid_angles_.d_view(n));
          m_new[2] += (n_2*i0_(m,n,k,j,i)/n_0*solid_angles_.d_view(n));
          m_new[3] += (n_3*i0_(m,n,k,j,i)/n_0*solid_angles_.d_view(n));

          // handle excision (see notes above)
          if (excise) {
            if (rad_mask_(m,k,j,i) || fabs(n_0) < n_0_floor_) { i0_(m,n,k,j,i) = 0.0; }
          }
        }

        // feedback on fluid
        if (affect_fluid_) {
          u0_(m,IEN,k,j,i) += (m_old[0] - m_new[0]);
          u0_(m,IM1,k,j,i) += (m_old[1] - m_new[1]);
          u0_(m,IM2,k,j,i) += (m_old[2] - m_new[2]);
          u0_(m,IM3,k,j,i) += (m_old[3] - m_new[3]);
        }
      } else {
        // NOTE(@pdmullen): At this point, it is possible that excision has not been
        // entirely applied if Compton is enabled and a badcell or temperature equilibrium
        // was encountered.. apply excision
        if (excise) {
          for (int n=0; n<=nang1; ++n) {
            Real n_0 = tc(m,0,0,k,j,i)*nh_c_.d_view(n,0)+
                       tc(m,1,0,k,j,i)*nh_c_.d_view(n,1)+
                       tc(m,2,0,k,j,i)*nh_c_.d_view(n,2)+
                       tc(m,3,0,k,j,i)*nh_c_.d_view(n,3);
            if (rad_mask_(m,k,j,i) || fabs(n_0) < n_0_floor_) { i0_(m,n,k,j,i) = 0.0; }
          }
        }
      }
    }
  });

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  bool FourthPolyRoot
//  \brief Exact solution for fourth order polynomial of
//  the form coef4 * x^4 + x + tconst = 0.

KOKKOS_INLINE_FUNCTION
bool FourthPolyRoot(const Real coef4, const Real tconst, Real &root) {
  // Calculate real root of z^3 - 4*tconst/coef4 * z - 1/coef4^2 = 0
  Real ccubic = tconst * tconst * tconst;
  Real delta1 = 0.25 - 64.0 * ccubic * coef4 / 27.0;
  if (delta1 < 0.0) {
    return false;
  }
  delta1 = sqrt(delta1);
  if (delta1 < 0.5) {
    return false;
  }
  Real zroot;
  if (delta1 > 1.0e11) {  // to avoid small number cancellation
    zroot = pow(delta1, -2.0/3.0) / 3.0;
  } else {
    zroot = pow(0.5 + delta1, 1.0/3.0) - pow(-0.5 + delta1, 1.0/3.0);
  }
  if (zroot < 0.0) {
    return false;
  }
  zroot *= pow(coef4, -2.0/3.0);

  // Calculate quartic root using cubic root
  Real rcoef = sqrt(zroot);
  Real delta2 = -zroot + 2.0 / (coef4 * rcoef);
  if (delta2 < 0.0) {
    return false;
  }
  delta2 = sqrt(delta2);
  root = 0.5 * (delta2 - rcoef);
  if (root < 0.0) {
    return false;
  }
  return true;
}

} // namespace radiation
