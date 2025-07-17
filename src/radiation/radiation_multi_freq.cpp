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
#include "radiation_multi_freq.hpp"

#include "radiation/radiation_tetrad.hpp"
#include "radiation/radiation_opacities.hpp"

namespace radiation {
//----------------------------------------------------------------------------------------
//! \fn  void Radiation::SetFrequencyGrid()
//! \brief Set frequency grid for radiation

void Radiation::SetFrequencyGrid() {
  bool &are_units_enabled_ = are_units_enabled;
  Real nu_unit = 1.0;
  if (are_units_enabled_) {
    Real h_p = 6.62607015e-27; // Planck constant
    Real k_b = 1.380649e-16;   // Boltzman constant
    Real temp_unit = pmy_pack->punit->temperature_cgs();
    nu_unit = k_b*temp_unit/h_p;
  }

  // Extract auxiliary quantities for frequency grid
  int  &nfreq_ = nfreq;
  int  &freq_scale = flag_fscale;
  Real &nu_min_ = nu_min;
  Real &nu_max_ = nu_max;
  auto &freq_grid_ = freq_grid;

  // convert frequency from cgs into sim unit
  Real freq_min = nu_min_/nu_unit; // from cgs to sim unit
  Real freq_max = nu_max_/nu_unit; // from cgs to sim unit

  // assign freq_min and freq_max
  int nfreq_grid = nfreq_; // frequency grid is defined starting from 0 but with inf excluded
  freq_grid_(0) = 0.0;
  freq_grid_(1) = freq_min;
  freq_grid_(nfreq_grid-1) = freq_max;

  // partition frequency domain within [freq_min, freq_max]
  if (nfreq_grid > 2) {
    if (freq_scale == 0) { // linear frequency grid
      Real del_freq = (freq_max-freq_min) / (nfreq_grid-2);
      for (int f=2; f<nfreq_grid-1; ++f)
        freq_grid_(f) = (f-1)*del_freq + freq_min;
    } else if (freq_scale == 1) { // log frequency grid
      Real log_freq_max = log(freq_max);
      Real log_freq_min = log(freq_min);
      Real del_log_freq = (log_freq_max-log_freq_min) / (nfreq_grid-2);
      for (int f=2; f<nfreq_grid-1; ++f) {
        Real log_freq = (f-1)*del_log_freq + log_freq_min;
        freq_grid_(f) = exp(log_freq);
      } // endfor f
    } // endelse freq_scale
  } // endif (nfreq_ > 2)

  return;
}


//----------------------------------------------------------------------------------------
//! \fn TaskStatus Radiation::AddMultiFreqRadSrcTerm(Driver *pdriver, int stage)
// \brief Add implicit multi-frequency radiation source term.

TaskStatus Radiation::AddMultiFreqRadSrcTerm(Driver *pdriver, int stage) {
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
  int &nang = prgeo->nangles;
  const int nang1 = nang - 1;
  const int nfreq1 = nfreq - 1;
  const int nfr_ang = nang*nfreq;
  auto &size  = pmy_pack->pmb->mb_size;
  auto &nu_tet = freq_grid;
  bool &is_hydro_enabled_ = is_hydro_enabled;
  bool &is_mhd_enabled_ = is_mhd_enabled;
  // bool &are_units_enabled_ = are_units_enabled;
  // bool &is_compton_enabled_ = is_compton_enabled;
  bool &fixed_fluid_ = fixed_fluid;
  // bool &affect_fluid_ = affect_fluid;

  // Parameters for intensity reconstruction
  int order = 2;
  int limiter = 2;

  // Extract coordinate/excision data
  auto &coord = pmy_pack->pcoord->coord_data;
  bool &flat = coord.is_minkowski;
  Real &spin = coord.bh_spin;
  // bool &excise = pmy_pack->pcoord->coord_data.bh_excise;
  // auto &rad_mask_ = pmy_pack->pcoord->excision_floor;
  // Real &n_0_floor_ = n_0_floor;

  // Extract radiation constant and units
  Real temp_unit = 1.0;
  Real &arad_ = arad;
  // Real density_scale_ = 1.0, temperature_scale_ = 1.0, length_scale_ = 1.0;
  // Real mean_mol_weight_ = 1.0;
  // Real rosseland_coef_ = 1.0, planck_minus_rosseland_coef_ = 0.0;
  // Real inv_t_electron_ = 1.0;
  // if (are_units_enabled_) {
  //   density_scale_ = pmy_pack->punit->density_cgs();
  //   temperature_scale_ = pmy_pack->punit->temperature_cgs();
  //   length_scale_ = pmy_pack->punit->length_cgs();
  //   mean_mol_weight_ = pmy_pack->punit->mu();
  //   rosseland_coef_ = pmy_pack->punit->rosseland_coef_cgs;
  //   planck_minus_rosseland_coef_ = pmy_pack->punit->planck_minus_rosseland_coef_cgs;
  //   inv_t_electron_ = temperature_scale_/pmy_pack->punit->electron_rest_mass_energy_cgs;
  // }

  // Extract adiabatic index
  Real gm1;
  if (is_hydro_enabled_) {
    gm1 = pmy_pack->phydro->peos->eos_data.gamma - 1.0;
  } else if (is_mhd_enabled_) {
    gm1 = pmy_pack->pmhd->peos->eos_data.gamma - 1.0;
  }

  // Extract radiation, radiation frame, and radiation angular mesh data
  auto &i0_ = i0;
  auto &nh_c_ = nh_c;
  auto &tt = tet_c;
  auto &tc = tetcov_c;
  auto &norm_to_tet_ = norm_to_tet;
  auto &solid_angles_ = prgeo->solid_angles;
  // Real &kappa_a_ = kappa_a;
  // Real &kappa_s_ = kappa_s;
  // Real &kappa_p_ = kappa_p;
  // bool &power_opacity_ = power_opacity;

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

  /********** Compute Implicit Source Term **********/
  par_for("multi_freq_radsrc",DevExeSpace(),0,nmb1,ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    // extract spatial position
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

    // compute solid angle weight sum
    Real wght_sum = 0.0;
    for (int iang=0; iang<=nang1; ++iang) {
      Real n0_cm = (u_tet[0]*nh_c_.d_view(iang,0) - u_tet[1]*nh_c_.d_view(iang,1) -
                    u_tet[2]*nh_c_.d_view(iang,2) - u_tet[3]*nh_c_.d_view(iang,3));
      Real domega_cm = solid_angles_.d_view(iang)/SQR(n0_cm);
      wght_sum += domega_cm;
    }

    // compute frequency-dependent coefficients
    Real sum_a[nfreq1+1], sum_b[nfreq1+1], jr_cm_old[nfreq1+1];
    Real eps_em[nfreq1+1], chi_p[nfreq1+1], chi_r[nfreq1+1], chi_s[nfreq1+1];
    for (int ifr=0; ifr<=nfreq1; ++ifr) {
      Real &nu_f = nu_tet(ifr);

      // emissivity
      Real eps_f = 0;
      if (ifr < nfreq1) {
        Real &nu_fp1 = nu_tet(ifr+1);
        eps_f = 1./(4*M_PI) * BBIntegral(nu_f, nu_fp1, tgas, arad_);
      } else {
        eps_f = 1./(4*M_PI) * arad_*SQR(SQR(teff)); // from 0 to inf
        eps_f = eps_f - 1./(4*M_PI) * BBIntegral(0, nu_f, tgas, arad_);
      } // endelse
      eps_em[ifr] = fmax(0., eps_f);

      // opacities
      chi_p[ifr] = 1; // TODO
      chi_r[ifr] = 1; // TODO
      chi_s[ifr] = 1; // TODO

      // initialize coefficients for solving temperature update
      sum_a[ifr] = 0;
      sum_b[ifr] = 0;
      jr_cm_old[ifr] = 0;

    } // endfor ifr




    // Step 1: map fluid-frame intensity into tetrad-frame frequency grid
    Real matrix_imap[nfreq1+1][nfreq1+1];
    Real &nu_e = nu_tet(nfreq1); // last tetrad-frame frequency
    for (int n=0; n<=nfr_ang-1; ++n) {
      // frequency and angle indices
      int ifr, iang;
      getFreqAngIndices(n, nang, ifr, iang);

      // variables for frame transformation
      Real n_0 = tc(m,0,0,k,j,i)*nh_c_.d_view(iang,0) + tc(m,1,0,k,j,i)*nh_c_.d_view(iang,1) +
                 tc(m,2,0,k,j,i)*nh_c_.d_view(iang,2) + tc(m,3,0,k,j,i)*nh_c_.d_view(iang,3);
      Real n0_cm = (u_tet[0]*nh_c_.d_view(iang,0) - u_tet[1]*nh_c_.d_view(iang,1) -
                    u_tet[2]*nh_c_.d_view(iang,2) - u_tet[3]*nh_c_.d_view(iang,3));

      // define auxiliary variables for intensity mapping
      Real nu0, nu1, nu2, nu3, nu1h, nu3h, nu5h;
      Real inu1, inu2, inu1h, inu3h, inu5h;
      Real ir_cm_star_1; int boundary;

      // target frequency and intensity
      Real &nu_f = nu_tet(ifr);
      Real ir_cm_f = 0.0; // value to be assigned

      // get effective temperature at last frequency bin
      int ne = getFreqAngIndex(nfreq1, iang, nang);
      Real ir_cm_star_e = SQR(SQR(n0_cm))*i0_(m,ne,k,j,i)/(n0*n_0);
      Real teff = GetEffTemperature(ir_cm_star_e, n0_cm*nu_e);

      // locate left & right fluid-frame frequency bins (if exist)
      // -1 <= idx_l   <= N-1
      //  0 <= idx_l+1 <= N
      int idx_lp1=0;
      while ((n0_cm*nu_tet(idx_lp1) < nu_f) && (idx_lp1 <= nfreq1+1)) idx_lp1++;
      int idx_l = idx_lp1-1;
      // -1 <= idx_r   <= N-1
      //  0 <= idx_r+1 <= N
      int idx_r=-1, idx_rp1=-1;
      if (ifr+1 <= nfreq1) {
        idx_rp1=fmax(idx_l,0);
        Real &nu_fp1 = nu_tet(ifr+1);
        while ((n0_cm*nu_tet(idx_rp1) <= nu_fp1) && (idx_rp1 <= nfreq1+1)) idx_rp1++;
        idx_r = idx_rp1-1;
      }

      // get mapping coefficients
      if (nu_f >= n0_cm*nu_e) { // only happen when (n0_cm < 1)
        // This covers the rightmost corner case (f = N-1) && (n0_cm < 1)
        //
        // Corner Case 4: Rightmost (nu_tet[f] >= nu_cm[N-1] && n0_cm < 1)
        //
        // When f <= N-2,                        When f = N-1,
        //       (nu_f)          (nu_fp1)               (nu_f)             (nu_fp1)
        //     nu_tet[N-2]      nu_tet[N-1]           nu_tet[N-1]
        // --------|----------------|---> inf     --------|----------------> inf
        //         | (ir_cm_star_e) |                     | (ir_cm_star_e)
        //         | I_cm_star[N-1] |                     | I_cm_star[N-1]
        // -----|---================----> inf     -----|---================> inf
        //  nu_cm[N-1]                             nu_cm[N-1]
        //    (nu1)                                  (nu1)
        Real integral_f = 0.0;
        if (ifr+1 <= nfreq1) { // ifr <= N-2
          Real &nu_fp1 = nu_tet(ifr+1);
          integral_f = 1./(4*M_PI) * BBIntegral(nu_f, nu_fp1, teff, arad_);
        } else { // ifr == N-1
          integral_f = 1./(4*M_PI) * arad_*SQR(SQR(teff)); // from 0 to inf
          integral_f = integral_f - 1./(4*M_PI) * BBIntegral(0, nu_f, teff, arad_);
        }
        Real frac_f = integral_f/ir_cm_star_e;
        ir_cm_f += integral_f;
        matrix_imap[ifr][nfreq1] = frac_f;

      } else if ((ifr == nfreq1) && (n0_cm > 1)) { // (f = N-1) && (n0_cm > 1)
        // (f = N-1) && (n0_cm < 1) is covered at the beginning of the if statement
        //
        // Corner Case 3: Rightmost (ifr=N-1 && n0_cm > 1)
        //                          0 <= L <= N-2
        //                           (nu_f)
        //         nu_tet[N-2]     nu_tet[N-1]
        // ------------|---------------|--------------------------------------------------------> inf
        //           (inu1h)     (inu1)   (inu3h)     (inu2)   (inu5h)
        //    |         *          |                    |                    |
        //    |                    |         *          |         *          |
        //    |                    |                    |                    |         *
        //    |   I_cm_star[N-4]   |   I_cm_star[N-3]   |   I_cm_star[N-2]   |   I_cm_star[N-1]
        // ---|--------------------|----================|====================|==================> inf
        // nu_cm[N-4]           nu_cm[N-3]           nu_cm[N-2]           nu_cm[N-1]
        //    nu0    (nu1h)      (nu1)    (nu3h)      (nu2)     (nu5h)      nu3

        // prepare left frequency and intensity
        bool left_bin_assigned = AssignFreqIntensity(idx_l, nu_tet, i0_, m, k, j, i, iang,
                                                     nang, nfreq1, n0_cm, n0, n_0, arad_, teff,
                                                     nu0, nu1, nu2, nu3, nu1h, nu3h, nu5h,
                                                     inu1h, inu3h, inu5h, inu1, inu2,
                                                     ir_cm_star_1, boundary);
        // compute left fractional contribution
        Real frac_l = 0;
        if (left_bin_assigned) {
          bool leftbin = true;
          Real nu_fp1 = nu2 * 1e12; // set nu_fp1 in this way so it is not invoked in 'IntensityFraction'
          frac_l = IntensityFraction(nu_f, nu_fp1,
                                     nu1h, nu1, nu3h, nu2, nu5h,
                                     inu1h, inu1, inu3h, inu2, inu5h,
                                     order, limiter, boundary, leftbin);
          ir_cm_f += frac_l * ir_cm_star_1;
          matrix_imap[ifr][idx_l] = frac_l;
        }

        // add the rest intensity contribution
        for (int f=idx_l+1; f<=nfreq1; ++f) {
          int nf = getFreqAngIndex(f, iang, nang);
          ir_cm_f += SQR(SQR(n0_cm))*i0_(m,nf,k,j,i)/(n0*n_0);
          matrix_imap[ifr][f] = 1;
        }

      } else { // (f <= N-2)
        Real &nu_fp1 = nu_tet(ifr+1);

        // General Case:
        // Given frequency bin [nu_tet(f), nu_tet(f+1)],
        // we can always find nu_cm(L) <  nu_tet(f)   <= nu_cm(L+1)
        //                and nu_cm(R) <= nu_tet(f+1) <  nu_cm(R+1)
        //
        //                               (nu_f)                                                                (nu_fp1)
        //                              nu_tet[f]                                                             nu_tet[f+1]
        // ............................ ---|----------------------------------------------------------------------|--- ............................ --->
        //                            I_cm_star[L]    I_cm_star[L+1]    I_cm_star[...]    I_cm_star[R-1]    I_cm_star[R]
        // ... ---|----------------|--------========|================|=== .......... ===|================|========--------|----------------|--- ... --->
        //     nu_cm[L-1]       nu_cm[L]         nu_cm[L+1]       nu_cm[L+2]         nu_cm[R-1]       nu_cm[R]         nu_cm[R+1]       nu_cm[R+2]

        // add the contribution from L+1 to R-1
        for (int f=idx_l+1; f<=idx_r-1; ++f) {
          int nf = getFreqAngIndex(f, iang, nang);
          ir_cm_f += SQR(SQR(n0_cm))*i0_(m,nf,k,j,i)/(n0*n_0);
          matrix_imap[ifr][f] = 1;
        }

        // Left:                            (nu_f)
        //                                 nu_tet[f]
        // ................................ ---|--- ......................... --->
        //             (inu1h)   (inu1) (inu3h)|  (inu2) (inu5h)
        //        |       *        |       *   |    |                |
        //        |                |           |    |       *        |
        //        | I_cm_star[L-1] |  I_cm_star[L]  | I_cm_star[L+1] |
        // ... ---|----------------|------------====|----------------|--- ... --->
        //     nu_cm[L-1]       nu_cm[L]         nu_cm[L+1]       nu_cm[L+2]
        //       nu0   (nu1h)    (nu1)   (nu3h)   (nu2)   (nu5h)    nu3

        // prepare left frequency and intensity
        bool left_bin_assigned = AssignFreqIntensity(idx_l, nu_tet, i0_, m, k, j, i, iang,
                                                     nang, nfreq1, n0_cm, n0, n_0, arad_, teff,
                                                     nu0, nu1, nu2, nu3, nu1h, nu3h, nu5h,
                                                     inu1h, inu3h, inu5h, inu1, inu2,
                                                     ir_cm_star_1, boundary);
        // compute left fractional contribution
        Real frac_l = 0;
        if (left_bin_assigned) {
          bool leftbin = true;
          frac_l = IntensityFraction(nu_f, nu_fp1,
                                     nu1h, nu1, nu3h, nu2, nu5h,
                                     inu1h, inu1, inu3h, inu2, inu5h,
                                     order, limiter, boundary, leftbin);
          ir_cm_f += frac_l * ir_cm_star_1;
          matrix_imap[ifr][idx_l] = frac_l;
        }

        // Right:                          (nu_fp1)
        //                                nu_tet[f+1]
        // ................................ ---|--- ......................... --->
        //             (inu1h)   (inu1) (inu3h)|  (inu2) (inu5h)
        //        |       *        |       *   |    |                |
        //        |                |           |    |       *        |
        //        | I_cm_star[R-1] |  I_cm_star[R]  | I_cm_star[R+1] |
        // ... ---|----------------|===========-----|----------------|--- ... --->
        //     nu_cm[R-1]       nu_cm[R]         nu_cm[R+1]       nu_cm[R+2]
        //             (nu1h)    (nu1)   (nu3h)   (nu2)   (nu5h)

        // prepare right frequency and intensity
        bool right_bin_assigned = AssignFreqIntensity(idx_r, nu_tet, i0_, m, k, j, i, iang,
                                                      nang, nfreq1, n0_cm, n0, n_0, arad_, teff,
                                                      nu0, nu1, nu2, nu3, nu1h, nu3h, nu5h,
                                                      inu1h, inu3h, inu5h, inu1, inu2,
                                                      ir_cm_star_1, boundary);
        // compute right fractional contribution
        Real frac_r = 0;
        if (right_bin_assigned) {
          bool leftbin = false;
          frac_r = IntensityFraction(nu_f, nu_fp1,
                                     nu1h, nu1, nu3h, nu2, nu5h,
                                     inu1h, inu1, inu3h, inu2, inu5h,
                                     order, limiter, boundary, leftbin);
          ir_cm_f += frac_r * ir_cm_star_1;
          matrix_imap[ifr][idx_r] = frac_r;
        }

        // Corner Case 4 (continue): Rightmost (nu_tet[f+1] >= nu_cm[N-1] >= nu_tet[f] && n0_cm < 1)
        //    (nu_f)                 (nu_fp1)
        //  nu_tet[N-2]           nu_tet[N-1]
        // -----|------------------------|----> inf
        //      |                        |
        //      |       | I_cm_star[N-1] |
        // -------------|================-----> inf
        //          nu_cm[N-1]
        //            (nu1)
        // R = N-1 (right_bin_assigned==false):
        if (idx_r == nfreq1) {
          nu1 = n0_cm*nu_tet(idx_r);
          int nr = getFreqAngIndex(idx_r, iang, nang);
          Real ir_cm_star_r = SQR(SQR(n0_cm))*i0_(m,nr,k,j,i)/(n0*n_0);
          Real integral_r = 1./(4*M_PI) * BBIntegral(nu1, nu_fp1, teff, arad_);
          frac_r = integral_r/ir_cm_star_r;
          ir_cm_f += integral_r;
          matrix_imap[ifr][idx_r] = frac_r;
        }

        // Corner Case 1: Leftmost (ifr=0 && n0_cm > 1)
        // (nu_f)    (nu_fp1)
        //    0      nu_tet[1] nu_tet[2] nu_tet[3] nu_tet[4] nu_tet[5]
        //    |---------|---------|---------|---------|---------|------> inf
        //  (inu1)      (inu3h)       (inu2)      (inu5h)
        //    |                         |            *            |
        //    |            *            |                         |
        //    |       I_cm_star[0]      |       I_cm_star[1]      |
        //    |=========----------------|-------------------------|----> inf
        // nu_cm[0]                  nu_cm[1]                  nu_cm[2]
        //  (nu1)        (nu3h)       (nu2)        (nu5h)        nu3
        // (L,L+1)=(-1,0) && (R,R+1)=(0,1)
        // L = -1: left_bin_assigned=false
        // R = 0: Do nothing. Algorithm is self-consistent.

        // Corner Case 2: Leftmost (ifr=0 && n0_cm < 1)
        // (nu_f)                                    (nu_fp1)
        //    0                                      nu_tet[1]
        //    |-----------------------------------------|------------------------------> inf
        //                          (inu1h)   (inu1) (inu3h)   (inu2) (inu5h)
        //    |                |       *        |       *        |       *        |
        //    |       *        | ir_cm_star_rm1 |  ir_cm_star_r  | ir_cm_star_rp1 |
        //    |  I_cm_star[0]  |  I_cm_star[1]  |  I_cm_star[2]  |  I_cm_star[3]  |
        //    |================|================|=======---------|----------------|----> inf
        // nu_cm[0]         nu_cm[1]         nu_cm[2]         nu_cm[3]         nu_cm[4]
        //                    nu0   (nu1h)    (nu1)   (nu3h)   (nu2)   (nu5h)    nu3
        // (L,L+1)=(-1,0) && R>=1 && R+1>=2
        // L = -1: left_bin_assigned=false
        // R <= N-1: Do nothing. Algorithm is self-consistent.

      } // endelse

      // set coefficients for solving temperature update
      Real &chf_p=chi_p[ifr], &chf_r=chi_r[ifr], &chf_s=chi_s[ifr];
      Real chf_f = chf_r + chf_s;
      Real domega_cm = solid_angles_.d_view(iang)/SQR(n0_cm);
      domega_cm /= wght_sum; // normalize fluid-frame weight
      sum_a[ifr] += n0/(n0+n0_cm*chf_f*dt_)*ir_cm_f*domega_cm/(4*M_PI);
      sum_b[ifr] += n0_cm*dt_/(n0+n0_cm*chf_f*dt_)*domega_cm/(4*M_PI);
      jr_cm_old[ifr] += ir_cm_f*domega_cm/(4*M_PI);

    } // endfor n

    // Step 2: compute source terms and solve gas temperature update

    // TODO: make initial guess of gas temperature

    // iterate to find temperature update
    for (int ifr=0; ifr<=nfreq1; ++ifr) {
      // Real &nu_f = nu_tet(ifr);
      Real &j0f_old = jr_cm_old[ifr];
      Real &sum_fa = sum_a[ifr], &sum_fb = sum_b[ifr];
      Real &eps_f=eps_em[ifr], &chf_p=chi_p[ifr];
      Real &chf_r=chi_r[ifr],  &chf_s=chi_s[ifr];
      Real chf_f = chf_r + chf_s;




    } // endfor ifr


    // Step 3: update frequency-dependent intensity

    // Step 4: update fluid variables










  });


  return TaskStatus::complete;
}


} // namespace radiation
