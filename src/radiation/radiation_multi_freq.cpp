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
  if (nfreq_grid > 3) {
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
  } // endif (nfreq_ > 3)

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
  int &nfrq = nfreq;
  const int nang1 = nang - 1;
  const int nfreq1 = nfrq - 1;
  const int nfr_ang = nang*nfrq;
  auto &size  = pmy_pack->pmb->mb_size;
  auto &nu_tet = freq_grid;
  bool &is_hydro_enabled_ = is_hydro_enabled;
  bool &is_mhd_enabled_ = is_mhd_enabled;
  bool &are_units_enabled_ = are_units_enabled;
  bool &is_compton_enabled_ = is_compton_enabled;
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
  Real &arad_ = arad;
  Real den_unit = 1.0, temp_unit = 1.0, l_unit = 1.0;
  Real mu_molecular = 1.0;
  Real coeff_r = 1.0, coeff_pmr = 0.0, inv_t_elec = 1.0;
  if (are_units_enabled_) {
    den_unit  = pmy_pack->punit->density_cgs();
    temp_unit = pmy_pack->punit->temperature_cgs();
    l_unit    = pmy_pack->punit->length_cgs();
    mu_molecular = pmy_pack->punit->mu();
    coeff_r   = pmy_pack->punit->rosseland_coef_cgs;
    coeff_pmr = pmy_pack->punit->planck_minus_rosseland_coef_cgs;
    inv_t_elec = temp_unit/pmy_pack->punit->electron_rest_mass_energy_cgs;
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
  auto &nh_c_ = nh_c;
  auto &tt = tet_c;
  auto &tc = tetcov_c;
  auto &norm_to_tet_ = norm_to_tet;
  auto &solid_angles_ = prgeo->solid_angles;
  Real &kappa_a_ = kappa_a;
  Real &kappa_s_ = kappa_s;
  Real &kappa_p_ = kappa_p;
  bool &power_opacity_ = power_opacity;

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

  // Scratch memory
  size_t scr_size = ScrArray2D<Real>::shmem_size(nfrq, nfrq)
                  + ScrArray2D<Real>::shmem_size(nang, nfrq) * 2
                  + ScrArray1D<Real>::shmem_size(nfr_ang)
                  + ScrArray1D<Real>::shmem_size(nang) * 4
                  + ScrArray1D<Real>::shmem_size(nfrq) * 5;

  if (is_compton_enabled_) {
    scr_size += ScrArray1D<Real>::shmem_size(nfrq) * 7;
  }
  int scr_level = 0;

  /********** Compute Implicit Source Term **********/
  par_for_outer("multi_freq_radsrc",DevExeSpace(),scr_size,scr_level,0,nmb1,ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(TeamMember_t member, const int m, const int k, const int j, const int i) {
    // temporary variables assigned on scratch memory
    ScrArray2D<Real> matrix_map(member.team_scratch(scr_level), nfrq, nfrq);
    ScrArray2D<Real> ir_cm_update(member.team_scratch(scr_level), nang, nfrq);
    ScrArray2D<Real> ir_cm_star_update(member.team_scratch(scr_level), nang, nfrq);
    ScrArray1D<Real> ir_cm_n(member.team_scratch(scr_level), nfr_ang);
    ScrArray1D<Real> fac_norm(member.team_scratch(scr_level), nang);
    ScrArray1D<Real> ir_cm_grey(member.team_scratch(scr_level), nang);
    ScrArray1D<Real> n_0_iang(member.team_scratch(scr_level), nang);
    ScrArray1D<Real> n0_cm_iang(member.team_scratch(scr_level), nang);
    ScrArray1D<Real> jr_cm_old(member.team_scratch(scr_level), nfrq);
    ScrArray1D<Real> chi_p(member.team_scratch(scr_level), nfrq);
    ScrArray1D<Real> chi_r(member.team_scratch(scr_level), nfrq);
    ScrArray1D<Real> sum_a(member.team_scratch(scr_level), nfrq);
    ScrArray1D<Real> sum_b(member.team_scratch(scr_level), nfrq);
    ScrArray1D<Real> komp_mat_u, komp_mat_c, komp_mat_d, komp_coeff;
    ScrArray1D<Real> coeff_delta, jr_cm_new, n_soln;
    if (is_compton_enabled_) {
      coeff_delta = ScrArray1D<Real>(member.team_scratch(scr_level), nfrq);
      komp_mat_u  = ScrArray1D<Real>(member.team_scratch(scr_level), nfrq);
      komp_mat_c  = ScrArray1D<Real>(member.team_scratch(scr_level), nfrq);
      komp_mat_d  = ScrArray1D<Real>(member.team_scratch(scr_level), nfrq);
      komp_coeff  = ScrArray1D<Real>(member.team_scratch(scr_level), nfrq);
      jr_cm_new   = ScrArray1D<Real>(member.team_scratch(scr_level), nfrq);
      n_soln      = ScrArray1D<Real>(member.team_scratch(scr_level), nfrq);
    }

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

    // angle-dependent coefficients and variables
    Real wght_sum = 0.0;
    for (int iang=0; iang<=nang1; ++iang) {
      // coordinate-frame normal components
      Real n_0 = tc(m,0,0,k,j,i)*nh_c_.d_view(iang,0) + tc(m,1,0,k,j,i)*nh_c_.d_view(iang,1)
               + tc(m,2,0,k,j,i)*nh_c_.d_view(iang,2) + tc(m,3,0,k,j,i)*nh_c_.d_view(iang,3);
      n_0_iang(iang) = n_0;

      // fluid-frame time component
      Real n0_cm = (u_tet[0]*nh_c_.d_view(iang,0) - u_tet[1]*nh_c_.d_view(iang,1)
                  - u_tet[2]*nh_c_.d_view(iang,2) - u_tet[3]*nh_c_.d_view(iang,3));
      n0_cm_iang(iang) = n0_cm;

      // fluid-frame angular weight sum
      Real domega_cm = solid_angles_.d_view(iang)/SQR(n0_cm);
      wght_sum += domega_cm;

      // initialize variables for later use when guessing temperature update
      fac_norm(iang) = 0;
      ir_cm_grey(iang) = 0;
    } // endfor iang

    // frequency-dependent coefficients
    // TODO: implement a better default opacity function for multi-frequency radiation
    Real chi_abs, chi_s, chi_pmr;
    OpacityFunction(wdn, den_unit, tgas, temp_unit, l_unit,
                    gm1, mu_molecular, power_opacity_, coeff_r, coeff_pmr,
                    kappa_a_, kappa_s_, kappa_p_, chi_abs, chi_s, chi_pmr);
    for (int ifr=0; ifr<=nfreq1; ++ifr) {
      // opacities
      // TODO: interpolate frequency-dependent opacity table
      chi_p(ifr) = chi_pmr + chi_abs;
      chi_r(ifr) = chi_abs;

      // initialize coefficients for later use when solving temperature update
      sum_a(ifr) = 0;
      sum_b(ifr) = 0;
      jr_cm_old(ifr) = 0;
    } // endfor ifr

    // Step 1: map intensity from fluid-frame to coordinate-frame frequency bins
    Real &nu_e = nu_tet(nfreq1); // last tetrad-frame frequency
    for (int n=0; n<=nfr_ang-1; ++n) {
      // frequency and angle indices
      int ifr, iang;
      getFreqAngIndices(n, nang, ifr, iang);

      // variables for frame transformation
      Real &n_0 = n_0_iang(iang);
      Real &n0_cm = n0_cm_iang(iang);

      // compute mapped intensity
      Real ir_cm_f = MapIntensity(ifr, nu_tet, i0_, m, k, j, i, iang,
                                  n0_cm, n0, n_0, arad_, order, limiter,
                                  matrix_map, false);
      ir_cm_n(n) = ir_cm_f;

      // sum for normalization
      fac_norm(iang) += SQR(SQR(n0_cm))*i0_(m,n,k,j,i)/(n0*n_0);
      ir_cm_grey(iang) += ir_cm_f;

    } // endfor n

    // compute normalization factors
    for (int iang=0; iang<=nang1; ++iang) {
      fac_norm(iang) *= 1./ir_cm_grey(iang);
      ir_cm_grey(iang) = 0.; // reset for assignment after normalization
    } // endfor iang

    // compute normalized quantities
    for (int n=0; n<=nfr_ang-1; ++n) {
      // frequency and angle indices
      int ifr, iang;
      getFreqAngIndices(n, nang, ifr, iang);

      // variables for frame transformation
      Real &n0_cm = n0_cm_iang(iang);

      // normalization
      ir_cm_n(n) *= fac_norm(iang);
      Real &ir_cm_f = ir_cm_n(n);

      // set coefficients for solving temperature update
      Real &chf_p=chi_p(ifr), &chf_r=chi_r(ifr), &chf_s=chi_s;
      Real chf_f = chf_r + chf_s;
      Real domega_cm = solid_angles_.d_view(iang)/SQR(n0_cm);
      domega_cm /= wght_sum; // normalize fluid-frame weight
      sum_a(ifr) += n0/(n0+n0_cm*chf_f*dt_)*ir_cm_f*domega_cm/(4*M_PI);
      sum_b(ifr) += n0_cm*dt_/(n0+n0_cm*chf_f*dt_)*domega_cm/(4*M_PI);
      jr_cm_old(ifr) += ir_cm_f*domega_cm/(4*M_PI);
      ir_cm_grey(iang) += ir_cm_f; // used in guessing gas temperature
    } // endfor n


    // Step 2: compute source terms and solve gas temperature update
    bool guess_grey_tgas = true;
    Real tgas_new = tgas;

    // make initial guess of gas temperature
    if (guess_grey_tgas) {
      Real jr_cm_old_grey=0, eps_sum=0;
      Real chi_p_grey=0, chi_r_grey=0;
      for (int ifr=0; ifr<=nfreq1; ++ifr) {
        // compute coefficients
        Real eps_f = ComputeEmissivity(nu_tet, ifr, tgas_new, arad_);
        jr_cm_old_grey += jr_cm_old(ifr);
        chi_p_grey += eps_f*chi_p(ifr);
        chi_r_grey += eps_f/chi_r(ifr);
        eps_sum += eps_f;
      }
      chi_p_grey = chi_p_grey/eps_sum;
      chi_r_grey = eps_sum/chi_r_grey;
      Real chi_f_grey = chi_r_grey + chi_s;

      Real sum_a_grey=0, sum_b_grey=0;
      for (int iang=0; iang<=nang1; ++iang) {
        Real &n_0 = n_0_iang(iang);
        Real &n0_cm = n0_cm_iang(iang);
        Real domega_cm = solid_angles_.d_view(iang)/SQR(n0_cm);
        domega_cm /= wght_sum; // normalize fluid-frame weight
        sum_a_grey += n0/(n0+n0_cm*chi_f_grey*dt_)*ir_cm_grey(iang)*domega_cm/(4*M_PI);
        sum_b_grey += n0_cm*dt_/(n0+n0_cm*chi_f_grey*dt_)*domega_cm/(4*M_PI);
      }

      Real denom_ = 1. - sum_b_grey*(chi_f_grey-chi_p_grey);
      Real coeff4 = gm1/wdn*arad_ * sum_b_grey*chi_p_grey/denom_;
      Real coeff0 = 4*M_PI*gm1/wdn * (sum_a_grey/denom_ - jr_cm_old_grey) - tgas;

      // solve polynomial
      if (fabs(coeff4) > 1.0e-20) {
        bool flag = FourthPolyRoot(coeff4, coeff0, tgas_new);
        if (!(flag) || !(isfinite(tgas_new))) {
          tgas_new = tgas;
        }
      } else tgas_new = -coeff0;
    } // endif guess_grey_tgas

    // iterate to find temperature update
    int num_itr_max = 100; Real tol = 1e-12;
    for (int m=1; m<=num_itr_max; ++m) {
      // compute Newton-Raphson coefficients
      Real f_tar=0, df_tar=0;
      for (int ifr=0; ifr<=nfreq1; ++ifr) {
        // compute coefficients
        Real eps_f = ComputeEmissivity(nu_tet, ifr, tgas_new, arad_);
        Real deps_f = ComputeEmDerivative(nu_tet, ifr, tgas_new, arad_);
        Real &j0f_old = jr_cm_old(ifr);
        Real &sum_fa = sum_a(ifr), &sum_fb = sum_b(ifr);
        Real &chf_p=chi_p(ifr), &chf_r=chi_r(ifr), &chf_s=chi_s;
        Real chf_f = chf_r + chf_s;
        Real denom_ = 1.-sum_fb*(chf_f-chf_p);
        f_tar += (sum_fa + sum_fb*chf_p*eps_f)/denom_ - j0f_old;
        df_tar += sum_fb*chf_p/denom_ * deps_f;
      } // endfor ifr
      f_tar = wdn/gm1*(tgas_new-tgas) + 4*M_PI*f_tar;
      df_tar = wdn/gm1 + 4*M_PI*df_tar;
      Real diff_temp = -f_tar/df_tar;

      // update temperature
      tgas_new += diff_temp;
      if (fabs(diff_temp/tgas_new) < tol) break;

    } // endfor m
    bool badcell=false;
    if (!(isfinite(tgas_new)) || (tgas_new < 0)) badcell = true;

    // Step 3: update intensity and fluid variables
    if (!(badcell)) {
      // Step 4: update frequency-dependent intensity in the fluid frame
      for (int n=0; n<=nfr_ang-1; ++n) {
        // frequency and angle indices
        int ifr, iang;
        getFreqAngIndices(n, nang, ifr, iang);

        // variables for frame transformation
        Real &n_0 = n_0_iang(iang);
        Real &n0_cm = n0_cm_iang(iang);

        // compute coefficients
        Real eps_f = ComputeEmissivity(nu_tet, ifr, tgas_new, arad_);
        Real &sum_fa = sum_a(ifr), &sum_fb = sum_b(ifr);
        Real &chf_p=chi_p(ifr), &chf_r=chi_r(ifr), &chf_s=chi_s;
        Real chf_f = chf_r + chf_s;
        Real denom_ = 1.-sum_fb*(chf_f-chf_p);
        Real jr_cm_ = (sum_fa + sum_fb*chf_p*eps_f) / denom_;

        // update fluid-frame intensity
        Real p1_ = n0/(n0+n0_cm*chf_f*dt_) * ir_cm_n(n);
        Real p2_ = n0_cm*dt_/(n0+n0_cm*chf_f*dt_);
        p2_ *= chf_p*eps_f + (chf_f-chf_p)*jr_cm_;
        ir_cm_update(iang,ifr) = p1_+p2_;

      } // endfor n

      // Step 5: map fluid-frame intensity back to coordinate-frame
      for (int iang=0; iang<=nang1; ++iang) {
        // variables for frame transformation
        Real &n_0 = n_0_iang(iang);
        Real &n0_cm = n0_cm_iang(iang);

        // reset mapping matrices
        for (int nn=0; nn<=nfreq1; nn++) {
            for (int mm=0; mm<=nfreq1; mm++) {
              matrix_map(nn,mm) = 0.0;
            } // endfor mm
        } // endfor nn

        // compute mapping matrix
        for (int ifr=0; ifr<=nfreq1; ++ifr) {
          // compute ifr-th row of mapping matrix
          Real _ = MapIntensity(ifr, nu_tet, i0_, m, k, j, i, iang,
                                n0_cm, n0, n_0, arad_, order, limiter,
                                matrix_map, true);
        } // endfor ifr

        // update intensity through inverse mapping
        bool inv_success = SolveTriLinearSystem(nfrq, matrix_map, ir_cm_update, iang, n0_cm, ir_cm_star_update);

        // if inverse-mapping failed, piecewise linear reconstruct the intensity update
        if (!inv_success) {
          for (int ifr=0; ifr<=nfreq1; ++ifr) {
            ir_cm_star_update(iang,ifr) = InvMapIntensity(ifr, nu_tet, ir_cm_update, iang, n0_cm, arad_, order, limiter);
          } // endfor ifr
        } // endif !inverse_success

        // sum for normalization
        fac_norm(iang) = 0.0;
        Real ir_cm_star_grey = 0.0;
        for (int ifr=0; ifr<=nfreq1; ++ifr) {
         fac_norm(iang)  += ir_cm_update(iang,ifr);
         ir_cm_star_grey += ir_cm_star_update(iang,ifr);
        } // endfor ifr

        // compute normalization factor
        fac_norm(iang) *= 1./ir_cm_star_grey;
      } // endfor iang

      // Step 6: update tetrad-frame intensity and compute moment differences
      Real m_old[4] = {0.0}; Real m_new[4] = {0.0};
      for (int iang=0; iang<=nang1; ++iang) {
        // variables for frame transformation
        Real &n_0 = n_0_iang(iang);
        Real &n0_cm = n0_cm_iang(iang);

        // coordinate normal components
        Real n_1 = tc(m,0,1,k,j,i)*nh_c_.d_view(iang,0) + tc(m,1,1,k,j,i)*nh_c_.d_view(iang,1)
                 + tc(m,2,1,k,j,i)*nh_c_.d_view(iang,2) + tc(m,3,1,k,j,i)*nh_c_.d_view(iang,3);
        Real n_2 = tc(m,0,2,k,j,i)*nh_c_.d_view(iang,0) + tc(m,1,2,k,j,i)*nh_c_.d_view(iang,1)
                 + tc(m,2,2,k,j,i)*nh_c_.d_view(iang,2) + tc(m,3,2,k,j,i)*nh_c_.d_view(iang,3);
        Real n_3 = tc(m,0,3,k,j,i)*nh_c_.d_view(iang,0) + tc(m,1,3,k,j,i)*nh_c_.d_view(iang,1)
                 + tc(m,2,3,k,j,i)*nh_c_.d_view(iang,2) + tc(m,3,3,k,j,i)*nh_c_.d_view(iang,3);
        Real &domega = solid_angles_.d_view(iang);

        for (int ifr=0; ifr<=nfreq1; ++ifr) {
            int n = getFreqAngIndex(ifr, iang, nang);

            // compute moments before coupling
            m_old[0] += (    i0_(m,n,k,j,i)    *domega);
            m_old[1] += (n_1*i0_(m,n,k,j,i)/n_0*domega);
            m_old[2] += (n_2*i0_(m,n,k,j,i)/n_0*domega);
            m_old[3] += (n_3*i0_(m,n,k,j,i)/n_0*domega);

            // normalize ir_cm_star_update
            Real ir_cm_star_update_f = fac_norm(iang)*ir_cm_star_update(iang,ifr);

            // update lab-frame intensity
            i0_(m,n,k,j,i) = (n0*n_0)*fmax(ir_cm_star_update_f, 0.0)/SQR(SQR(n0_cm));

            // compute moments after coupling
            m_new[0] += (    i0_(m,n,k,j,i)    *domega);
            m_new[1] += (n_1*i0_(m,n,k,j,i)/n_0*domega);
            m_new[2] += (n_2*i0_(m,n,k,j,i)/n_0*domega);
            m_new[3] += (n_3*i0_(m,n,k,j,i)/n_0*domega);

        } // endfor ifr
      } // endfor iang

      // Step 7: update fluid variables
      u0_(m,IEN,k,j,i) += (m_old[0] - m_new[0]);
      u0_(m,IM1,k,j,i) += (m_old[1] - m_new[1]);
      u0_(m,IM2,k,j,i) += (m_old[2] - m_new[2]);
      u0_(m,IM3,k,j,i) += (m_old[3] - m_new[3]);
    } // endif (!(badcell))


    // Compton process
    if (!(badcell) && is_compton_enabled_) {
      tgas = tgas_new; // use partially updated gas temperature

      // Step 1: compute coefficients
      Real jr_cm_old_grey = 0; // total zeroth-order moment
      Real int_n_nu4=0, int_n2_nu4=0; // integrations used to compute equilibrium temperature
      Real n_tot = 0; // total photon number
      for (int ifr=0; ifr<=nfreq1; ++ifr) {
        // angular integration for zeroth-order moments of each frequency group
        jr_cm_old(ifr) = 0;
        for (int iang=0; iang<=nang1; ++iang) {
          Real &n0_cm = n0_cm_iang(iang);
          Real domega_cm = solid_angles_.d_view(iang)/SQR(n0_cm);
          domega_cm /= wght_sum; // normalize fluid-frame weight
          jr_cm_old(ifr) += ir_cm_update(iang,ifr)*domega_cm/(4*M_PI);
        } // endfor iang

        // integration for total zeroth-order moment
        jr_cm_old_grey += jr_cm_old(ifr);

        // integrations where the last bin needs to be handled separately
        if (ifr < nfreq1) {
          Real &nu_f = nu_tet(ifr);
          Real &nu_fp1 = nu_tet(ifr+1);
          Real nu_fp1h = (nu_fp1+nu_f)/2;
          Real del_nu = nu_fp1-nu_f;
          // used to solve equilibrium temperature
          int_n_nu4  += jr_cm_old(ifr)/arad_*nu_fp1h;
          int_n2_nu4 += SQR(jr_cm_old(ifr)/arad_/nu_fp1h)/del_nu;
          // used to solve Kompaneets equation
          n_tot += jr_cm_old(ifr)/arad_/nu_fp1h;
        } // endif (ifr < nfreq1)
      } // endfor ifr

      // integrate over the last frequency bin for n_tot, int_n_nu4 and int_n2_nu4
      Real &jr_cm_e = jr_cm_old(nfreq1);
      Real int_n_nu2_e, int_n_nu4_e, int_n2_nu4_e;
      WienInt(nu_e, jr_cm_e, tgas, arad_, int_n_nu2_e, int_n_nu4_e, int_n2_nu4_e);
      int_n2_nu4 += int_n2_nu4_e;
      int_n_nu4 += int_n_nu4_e;
      n_tot += int_n_nu2_e;
      n_tot *= SQR(SQR(M_PI))/15; // normalized total photon number for Kompaneets equation
      Real int_n_nu3 = jr_cm_old_grey/arad_;

      // Step 2: esimate gas temperature
      Real tcomp = (0.25*int_n_nu4 + SQR(SQR(M_PI))/60*int_n2_nu4) / int_n_nu3;
      Real coeff_a = 4*chi_s*dt_*inv_t_elec/u0;
      Real coeff_b = 4*M_PI*gm1/wdn*jr_cm_old_grey;

      Real coeff5 = coeff_a*tcomp;
      Real coeff4 = coeff_a*coeff_b + 1;
      Real coeff0 = -(coeff4 + coeff_a*tgas);
      if (!((coeff5 >= 0) && (coeff4 >= 1) && (coeff0 <= -1))) badcell = true;
      Real rat_ini = 1;

      // solve 4th-order polynomial for initial guess and backup
      bool backup_tgas_compton = true;
      if ((!badcell) && backup_tgas_compton) {
        Real a4 = coeff4/coeff5;
        Real a0 = coeff0/coeff5;
        if (fabs(a4) > 1.0e-20) {
          bool flag = FourthPolyRoot(a4, a0, rat_ini);
          if (!(flag) || !(isfinite(rat_ini))) {
            rat_ini = 1;
          }
        } else rat_ini = -a0;
      } // endif

      // solve 5th-order polynomial for accurate estimation
      Real rat_soln = rat_ini;
      if (!badcell) {
        int getSoln = SolCompTempEst(coeff5, coeff4, coeff0, rat_ini, rat_soln); // 0: success; 1: maximum iteration
        if (getSoln < 0) badcell = true;
      }

      if (!badcell) {
        // update gas temperature
        tgas_new = tgas + coeff_b * (1.0 - SQR(SQR(rat_soln)));

        // Step 3: construct interface coefficients for discretized Kompaneets equation
        // find the normalization factor lamda for quasi-steady solution n_star
        // solve Li_3(1/lambda) = 0.5*n_tot/tgas_new^3
        // TODO: implement a more accurate solver for lambda
        Real coeff_lmd = QuasiSteadySol(n_tot, tgas_new);

        // compute reconstruction coefficients
        // TODO: confirm the algorithms below
        coeff_delta(0) = 0.5;
        coeff_delta(nfreq1) = 0.5;
        if (coeff_lmd > 1.e16) {
          for (int ifr=1; ifr<nfreq1; ++ifr) {
            Real &nu_fm1 = nu_tet(ifr-1);
            Real &nu_f = nu_tet(ifr);
            Real &nu_fp1 = nu_tet(ifr+1);
            Real nu_fm1h = (nu_fm1+nu_f)/2;
            Real nu_fp1h = (nu_fp1+nu_f)/2;
            coeff_delta(ifr) = (nu_fp1h - nu_f) / (nu_fp1h - nu_fm1h);
          } // endfor ifr
        } else {
          for (int ifr=1; ifr<nfreq1; ++ifr) {
            Real &nu_fm1 = nu_tet(ifr-1);
            Real &nu_f = nu_tet(ifr);
            Real &nu_fp1 = nu_tet(ifr+1);
            Real nu_fm1h = (nu_fm1+nu_f)/2;
            Real nu_fp1h = (nu_fp1+nu_f)/2;

            Real n_star_fp1h = 1.0 / (coeff_lmd*exp(nu_fp1h)/tgas_new - 1);
            Real n_star_fm1h = 1.0 / (coeff_lmd*exp(nu_fm1h)/tgas_new - 1);

            Real dn_dnu = (n_star_fp1h - n_star_fm1h) / (nu_fp1h - nu_fm1h);
            Real delta_ = 0.5*(sqrt(1-4.0*tgas_new*dn_dnu) - 1.0);
            delta_ = (delta_ - n_star_fp1h) / (n_star_fm1h - n_star_fp1h);
            coeff_delta(ifr) = fmax(0.0, delta_);
          } // endfor ifr
        } // endelse

        // Step 4: solve the linear system for implicit update
        // construct tridiagonal coefficient matrix
        Real nu_em1h = (nu_e+nu_tet(nfreq1-1))/2;
        Real int_n_nu3_partial = tgas_new * (6*tgas_new*SQR(tgas_new) + 6*SQR(tgas_new)*nu_e + 3*tgas_new*SQR(nu_e) + nu_e*SQR(nu_e));
        Real ne_old = SQR(SQR(M_PI))/15 * jr_cm_old(nfreq1)/arad_ / int_n_nu3_partial;
        Real coeff_astar = u0/(inv_t_elec*chi_s*dt_) * tgas_new*(2*SQR(tgas_new) + 2*tgas_new*nu_e + SQR(nu_e));
        Real coeff_bstar = SQR(SQR(nu_e))*tgas_new/(nu_e-nu_em1h);
        Real coeff_cstar = SQR(SQR(nu_e)) * (1+ne_old);

        for (int ifr=0; ifr<=nfreq1; ++ifr) {
          Real &nu_f = nu_tet(ifr);
          Real &delta_f = coeff_delta(ifr);

          Real c1_f, c2_f, c3_f, c4_f;
          if (ifr == nfreq1) { // last frequency bin
            c1_f = -coeff_bstar;
            c2_f = coeff_astar + coeff_bstar + coeff_cstar;
            c3_f = 0;
            c4_f = -coeff_astar * ne_old;

          } else { // (ifr < nfreq1)
            Real &delta_fp1 = coeff_delta(ifr+1);
            Real &nu_fp1 = nu_tet(ifr+1);
            Real &nu_fp2 = nu_tet(ifr+2);
            Real nu_fp1h = (nu_fp1+nu_f)/2;
            Real nu_fp3h = (nu_fp2+nu_fp1)/2;
            Real n_fp1h = jr_cm_old(ifr)  /arad_/(nu_fp1-nu_f)  /(nu_fp1h*SQR(nu_fp1h)) * SQR(SQR(M_PI))/15;
            Real n_fp3h = jr_cm_old(ifr+1)/arad_/(nu_fp2-nu_fp1)/(nu_fp3h*SQR(nu_fp3h)) * SQR(SQR(M_PI))/15;
            Real nfp1_old = delta_fp1*n_fp1h + (1-delta_fp1)*n_fp3h;

            Real coeff_af   = u0/(inv_t_elec*chi_s*dt_) * SQR(nu_fp1h)*(nu_fp1-nu_f);
            Real coeff_bfp1 = SQR(SQR(nu_fp1))*tgas_new/(nu_fp3h-nu_fp1h);
            Real coeff_cfp1 = SQR(SQR(nu_fp1)) * (1+nfp1_old);

            if (ifr == 0) {
              c1_f = 0;
              c2_f = coeff_af + coeff_bfp1 - delta_fp1*coeff_cfp1;
            } else { // ifr > 0
              Real &nu_fm1 = nu_tet(ifr-1);
              Real nu_fm1h = (nu_fm1+nu_f)/2;
              Real n_fm1h = jr_cm_old(ifr-1)/arad_/(nu_f-nu_fm1)/(nu_fm1h*SQR(nu_fm1h)) * SQR(SQR(M_PI))/15;
              Real nf_old = delta_f*n_fm1h + (1-delta_f)*n_fp1h;

              Real coeff_bf = SQR(SQR(nu_f))*tgas_new/(nu_fp1h-nu_fm1h);
              Real coeff_cf = SQR(SQR(nu_f)) * (1+nf_old);

              c1_f = -coeff_bf + delta_f*coeff_cf;
              c2_f = (ifr==nfreq1-1) ? coeff_af + coeff_bf + coeff_bstar + (1-delta_f)*coeff_cf
                                     : coeff_af + coeff_bf + coeff_bfp1 + (1-delta_f)*coeff_cf - delta_fp1*coeff_cfp1;
            } // endelse

            Real c3_f = (ifr==nfreq1-1) ? -coeff_bstar - coeff_cstar : -coeff_bfp1 - (1-delta_fp1)*coeff_cfp1;
            Real c4_f = -coeff_af*n_fp1h;
          } // endelse

          komp_mat_d(ifr) = c1_f;
          komp_mat_c(ifr) = c2_f;
          komp_mat_u(ifr) = c3_f;
          komp_coeff(ifr) = -c4_f;
        } // endfor

        // coefficient matrix is tridiagonal, use Thomas algorithm
        bool komp_update = SolveTridiag(komp_mat_d, komp_mat_c, komp_mat_u, komp_coeff, nfrq, sum_a, sum_b, n_soln);
        // TODO: make sure sum_a and sum_b are not used in the following calculations

        if (komp_update) {
          // Step 5: update radiation energy and gas temperature
          // radiation energy
          Real jr_cm_new_grey = 0;
          for (int ifr=0; ifr<nfreq1; ++ifr) {
            Real &nu_f = nu_tet(ifr);
            Real &nu_fp1 = nu_tet(ifr+1);
            Real nu_fp1h = (nu_fp1+nu_f)/2;
            jr_cm_new(ifr) = n_soln(ifr) * 15/SQR(SQR(M_PI))*arad_ * (nu_fp1-nu_f) * nu_fp1h*SQR(nu_fp1h);
            jr_cm_new_grey += jr_cm_new(ifr);
          } // endfor ifr
          jr_cm_new(nfreq1) = 15/SQR(SQR(M_PI))*arad_ * n_soln(nfreq1) * int_n_nu3_partial;
          jr_cm_new_grey += jr_cm_new(nfreq1);

          // gas temperature
          Real tgas_update = -4*M_PI*(jr_cm_new_grey-jr_cm_old_grey)*gm1/wdn + tgas;
          if (tgas_update < FLT_MIN) tgas_update = tgas_new;

          // Step 6: update radiation intensity
          // update fluid-frame intensity
          for (int n=0; n<=nfr_ang-1; ++n) {
            int ifr, iang;
            getFreqAngIndices(n, nang, ifr, iang);
            // ir_cm_update(iang,ifr) += jr_cm_new(ifr) - jr_cm_old(ifr); // isotropically
            ir_cm_update(iang,ifr) *= jr_cm_new(ifr)/jr_cm_old(ifr); // maintain original angular distribution
          } // endfor n

          // map fluid-frame intensity back to coordinate-frame
          for (int iang=0; iang<=nang1; ++iang) {
            // variables for frame transformation
            Real &n_0 = n_0_iang(iang);
            Real &n0_cm = n0_cm_iang(iang);

            // reset mapping matrices
            for (int nn=0; nn<=nfreq1; nn++) {
                for (int mm=0; mm<=nfreq1; mm++) {
                  matrix_map(nn,mm) = 0.0;
                } // endfor mm
            } // endfor nn

            // compute mapping matrix
            for (int ifr=0; ifr<=nfreq1; ++ifr) {
              // compute ifr-th row of mapping matrix
              Real _ = MapIntensity(ifr, nu_tet, i0_, m, k, j, i, iang,
                                    n0_cm, n0, n_0, arad_, order, limiter,
                                    matrix_map, true);
            } // endfor ifr

            // update intensity through inverse mapping
            bool inv_success = SolveTriLinearSystem(nfrq, matrix_map, ir_cm_update, iang, n0_cm, ir_cm_star_update);

            // if inverse-mapping failed, piecewise linear reconstruct the intensity update
            if (!inv_success) {
              for (int ifr=0; ifr<=nfreq1; ++ifr) {
                ir_cm_star_update(iang,ifr) = InvMapIntensity(ifr, nu_tet, ir_cm_update, iang, n0_cm, arad_, order, limiter);
              } // endfor ifr
            } // endif !inverse_success

            // sum for normalization
            fac_norm(iang) = 0.0;
            Real ir_cm_star_grey = 0.0;
            for (int ifr=0; ifr<=nfreq1; ++ifr) {
             fac_norm(iang)  += ir_cm_update(iang,ifr);
             ir_cm_star_grey += ir_cm_star_update(iang,ifr);
             if (ir_cm_star_update(iang,ifr) < 0.0) badcell = true;
            } // endfor ifr

            // compute normalization factor
            fac_norm(iang) *= 1./ir_cm_star_grey;
          } // endfor iang

          if (!badcell) {
            // update tetrad-frame intensity
            Real m_old[4] = {0.0}; Real m_new[4] = {0.0};
            for (int iang=0; iang<=nang1; ++iang) {
              // variables for frame transformation
              Real &n_0 = n_0_iang(iang);
              Real &n0_cm = n0_cm_iang(iang);

              // coordinate normal components
              Real n_1 = tc(m,0,1,k,j,i)*nh_c_.d_view(iang,0) + tc(m,1,1,k,j,i)*nh_c_.d_view(iang,1)
                       + tc(m,2,1,k,j,i)*nh_c_.d_view(iang,2) + tc(m,3,1,k,j,i)*nh_c_.d_view(iang,3);
              Real n_2 = tc(m,0,2,k,j,i)*nh_c_.d_view(iang,0) + tc(m,1,2,k,j,i)*nh_c_.d_view(iang,1)
                       + tc(m,2,2,k,j,i)*nh_c_.d_view(iang,2) + tc(m,3,2,k,j,i)*nh_c_.d_view(iang,3);
              Real n_3 = tc(m,0,3,k,j,i)*nh_c_.d_view(iang,0) + tc(m,1,3,k,j,i)*nh_c_.d_view(iang,1)
                       + tc(m,2,3,k,j,i)*nh_c_.d_view(iang,2) + tc(m,3,3,k,j,i)*nh_c_.d_view(iang,3);
              Real &domega = solid_angles_.d_view(iang);

              for (int ifr=0; ifr<=nfreq1; ++ifr) {
                  int n = getFreqAngIndex(ifr, iang, nang);

                  // compute moments before coupling
                  m_old[0] += (    i0_(m,n,k,j,i)    *domega);
                  m_old[1] += (n_1*i0_(m,n,k,j,i)/n_0*domega);
                  m_old[2] += (n_2*i0_(m,n,k,j,i)/n_0*domega);
                  m_old[3] += (n_3*i0_(m,n,k,j,i)/n_0*domega);

                  // normalize ir_cm_star_update
                  Real ir_cm_star_update_f = fac_norm(iang)*ir_cm_star_update(iang,ifr);

                  // update lab-frame intensity
                  i0_(m,n,k,j,i) = (n0*n_0)*fmax(ir_cm_star_update_f, 0.0)/SQR(SQR(n0_cm));

                  // compute moments after coupling
                  m_new[0] += (    i0_(m,n,k,j,i)    *domega);
                  m_new[1] += (n_1*i0_(m,n,k,j,i)/n_0*domega);
                  m_new[2] += (n_2*i0_(m,n,k,j,i)/n_0*domega);
                  m_new[3] += (n_3*i0_(m,n,k,j,i)/n_0*domega);
              } // endfor ifr
            } // endfor iang

            // Step 7: update fluid variables
            u0_(m,IEN,k,j,i) += (m_old[0] - m_new[0]);
            u0_(m,IM1,k,j,i) += (m_old[1] - m_new[1]);
            u0_(m,IM2,k,j,i) += (m_old[2] - m_new[2]);
            u0_(m,IM3,k,j,i) += (m_old[3] - m_new[3]);

          } // endif (!badcell)
        } // endif komp_update

      } // endif (!badcell)

    } // endif is_compton_enabled_

  });

  return TaskStatus::complete;
}




} // namespace radiation
