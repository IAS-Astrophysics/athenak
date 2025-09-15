//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file srcterms.cpp
//  Implements various (physics) source terms to be added to the Hydro or MHD eqns.
//  Source terms objects are stored in the respective fluid class, so that
//  Hydro/MHD can have different source terms

#include "srcterms.hpp"

#include <iostream>
#include <string> // string

#include "athena.hpp"
#include "coordinates/cartesian_ks.hpp"
#include "coordinates/cell_locations.hpp"
#include "eos/eos.hpp"
#include "geodesic-grid/geodesic_grid.hpp"
#include "hydro/hydro.hpp"
#include "ismcooling.hpp"
#include "mesh/mesh.hpp"
#include "mhd/mhd.hpp"
#include "parameter_input.hpp"
#include "radiation/radiation.hpp"
#include "turb_driver.hpp"
#include "units/units.hpp"
#include "cooling_tables.hpp"

//----------------------------------------------------------------------------------------
// constructor, parses input file and initializes data structures and parameters
// Only source terms specified in input file are initialized.

SourceTerms::SourceTerms(std::string block, MeshBlockPack *pp, ParameterInput *pin) :
  pmy_pack(pp),
  Tbins("Tbins",1), nHbins("nHbins",1),
  Metal_Cooling("Metal_Cooling",1,1), H_He_Cooling("H_He_Cooling",1,1),
  Metal_Cooling_CIE("Metal_Cooling_CIE",1), H_He_Cooling_CIE("H_He_Cooling_CIE",1),
  shearing_box_r_phi(false) {
  // (1) (constant) gravitational acceleration
  const_accel = pin->GetOrAddBoolean(block, "const_accel", false);
  if (const_accel) {
    const_accel_val = pin->GetReal(block, "const_accel_val");
    const_accel_dir = pin->GetInteger(block, "const_accel_dir");
    if (const_accel_dir < 1 || const_accel_dir > 3) {
      std::cout << "### FATAL ERROR in "<< __FILE__ <<" at line " << __LINE__ << std::endl
                << "const_accle_dir must be 1,2, or 3" << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }

  // (2) Optically thin (ISM) cooling
  ism_cooling = pin->GetOrAddBoolean(block, "ism_cooling", false);
  if (ism_cooling) {
    hrate = pin->GetReal(block, "hrate");
  }

  // (2b) CGM cooling
  cgm_cooling = pin->GetOrAddBoolean(block, "cgm_cooling", false);
  if (cgm_cooling) {
    // Ensure that ISM cooling is not enabled 
    if (ism_cooling) {
      std::cout << "### FATAL ERROR in "<< __FILE__ <<" at line " << __LINE__ << std::endl
                << "CGM cooling and ISM cooling are incompatible" << std::endl;
      std::exit(EXIT_FAILURE);
    };

    // User input parameters for ISM heating
    hrate = pin->GetOrAddReal(block, "hrate", 0.0); // Heating rate in cgs units
    // Normalization factor in density code units 
    hscale_norm = pin->GetOrAddReal(block, "hscale_norm", 0.0); 
    hscale_height = pin->GetOrAddReal(block, "hscale_height", 0.0); // Scale height in code units
    hscale_radius = pin->GetOrAddReal(block, "hscale_radius", 0.0); // Scale radius in code units

    // Set temperature ceiling
    T_max = pin->GetOrAddReal(block, "T_max", 1e10); // Temperature Ceiling in cgs

    // Initialize Cooling Tables to the right dimensions from cooling_tables.hpp
    Kokkos::realloc(Tbins, Tbins_TOTAL_SIZE);
    Kokkos::realloc(nHbins, nHbins_TOTAL_SIZE);
    Kokkos::realloc(Metal_Cooling, Metal_Cooling_DIM_0, Metal_Cooling_DIM_1);
    Kokkos::realloc(H_He_Cooling, H_He_Cooling_DIM_0, H_He_Cooling_DIM_1);
    Kokkos::realloc(Metal_Cooling_CIE, Metal_Cooling_CIE_DIM_0);
    Kokkos::realloc(H_He_Cooling_CIE, H_He_Cooling_CIE_DIM_0);
  }

  // (3) beam source (radiation)
  beam = pin->GetOrAddBoolean(block, "beam_source", false);
  if (beam) {
    dii_dt = pin->GetReal(block, "dii_dt");
  }

  // (4) cooling (relativistic)
  rel_cooling = pin->GetOrAddBoolean(block, "rel_cooling", false);
  if (rel_cooling) {
    crate_rel = pin->GetReal(block, "crate_rel");
    cpower_rel = pin->GetOrAddReal(block, "cpower_rel", 1.);
  }

  // (5) shearing box
  if (pin->DoesBlockExist("shearing_box")) {
    shearing_box = true;
    qshear = pin->GetReal("shearing_box","qshear");
    omega0 = pin->GetReal("shearing_box","omega0");
  } else {
    shearing_box = false;
  }

  Initialize();
}

//----------------------------------------------------------------------------------------
// destructor

SourceTerms::~SourceTerms() {
}

//----------------------------------------------------------------------------------------
//! \fn
//  \brief Function to initialize

void SourceTerms::Initialize() {
  if (cgm_cooling) {
    // Load Cooling Tables from cooling_tables.hpp into host device
    for (int i = 0; i < Tbins_DIM_0; ++i) Tbins.h_view(i) = Tbins_ARR[i];
    for (int i = 0; i < nHbins_DIM_0; ++i) nHbins.h_view(i) = nHbins_ARR[i];

    for (int i = 0; i < Metal_Cooling_DIM_0; ++i) {
      for (int j = 0; j < Metal_Cooling_DIM_1; ++j) {
        Metal_Cooling.h_view(i, j) = Metal_Cooling_ARR[i * Metal_Cooling_DIM_1 + j];
    }}

    for (int i = 0; i < H_He_Cooling_DIM_0; ++i) {
      for (int j = 0; j < H_He_Cooling_DIM_1; ++j) {
        H_He_Cooling.h_view(i, j) = H_He_Cooling_ARR[i * H_He_Cooling_DIM_1 + j];
    }}

    for (int i = 0; i < Metal_Cooling_CIE_DIM_0; ++i) {
      Metal_Cooling_CIE.h_view(i) = Metal_Cooling_CIE_ARR[i];
    }

    for (int i = 0; i < H_He_Cooling_CIE_DIM_0; ++i) {
      H_He_Cooling_CIE.h_view(i) = H_He_Cooling_CIE_ARR[i];
    }

    // Synchronize Cooling Tables to device memory
    Tbins.template modify<HostMemSpace>();
    Tbins.template sync<DevExeSpace>();
    nHbins.template modify<HostMemSpace>();
    nHbins.template sync<DevExeSpace>();
    Metal_Cooling.template modify<HostMemSpace>();
    Metal_Cooling.template sync<DevExeSpace>();
    H_He_Cooling.template modify<HostMemSpace>();
    H_He_Cooling.template sync<DevExeSpace>();
    Metal_Cooling_CIE.template modify<HostMemSpace>();
    Metal_Cooling_CIE.template sync<DevExeSpace>();
    H_He_Cooling_CIE.template modify<HostMemSpace>();
    H_He_Cooling_CIE.template sync<DevExeSpace>();
  }
}

//----------------------------------------------------------------------------------------
//! \fn
// Add constant acceleration
// NOTE source terms must be computed using primitive (w0) and NOT conserved (u0) vars

void SourceTerms::ConstantAccel(const DvceArray5D<Real> &w0, const EOS_Data &eos_data,
                                const Real bdt, DvceArray5D<Real> &u0) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;

  Real &g = const_accel_val;
  int &dir = const_accel_dir;

  par_for("const_acc", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real src = bdt*g*w0(m,IDN,k,j,i);
    u0(m,dir,k,j,i) += src;
    if (eos_data.is_ideal) { u0(m,IEN,k,j,i) += src*w0(m,dir,k,j,i); }
  });

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void SourceTerms::ISMCooling()
//! \brief Add explict ISM cooling and heating source terms in the energy equations.
// NOTE source terms must be computed using primitive (w0) and NOT conserved (u0) vars

void SourceTerms::ISMCooling(const DvceArray5D<Real> &w0, const EOS_Data &eos_data,
                             const Real bdt, DvceArray5D<Real> &u0) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  Real use_e = eos_data.use_e;
  Real gamma = eos_data.gamma;
  Real gm1 = gamma - 1.0;
  Real heating_rate = hrate;
  Real temp_unit = pmy_pack->punit->temperature_cgs();
  Real n_unit = pmy_pack->punit->density_cgs()/pmy_pack->punit->mu()
                /pmy_pack->punit->atomic_mass_unit_cgs;
  Real cooling_unit = pmy_pack->punit->pressure_cgs()/pmy_pack->punit->time_cgs()
                      /n_unit/n_unit;
  Real heating_unit = pmy_pack->punit->pressure_cgs()/pmy_pack->punit->time_cgs()/n_unit;

  par_for("ism_cooling", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    // temperature in cgs unit
    Real temp = 1.0;
    if (use_e) {
      temp = temp_unit*w0(m,IEN,k,j,i)/w0(m,IDN,k,j,i)*gm1;
    } else {
      temp = temp_unit*w0(m,ITM,k,j,i);
    }

    Real lambda_cooling = ISMCoolFn(temp)/cooling_unit;
    Real gamma_heating = heating_rate/heating_unit;

    u0(m,IEN,k,j,i) -= bdt * w0(m,IDN,k,j,i) *
                        (w0(m,IDN,k,j,i) * lambda_cooling - gamma_heating);
  });

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void SourceTerms::CGMCooling()
//! \brief Add explict CGM cooling source term in the energy equations.
// NOTE source terms must be computed using primitive (w0) and NOT conserved (u0) vars
void SourceTerms::CGMCooling(const DvceArray5D<Real> &w0, const EOS_Data &eos_data,
                             const Real bdt, DvceArray5D<Real> &u0) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nx1 = indcs.nx1;
  int nx2 = indcs.nx2;
  int nx3 = indcs.nx3;
  auto &size = pmy_pack->pmb->mb_size;
  int nscalars = pmy_pack->phydro->nscalars;
  int nhydro = pmy_pack->phydro->nhydro;
  int nmb1 = pmy_pack->nmb_thispack - 1;
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

  auto Tfloor  = std::pow(10, Tbins_ARR[0]);
  auto Tceil   = std::pow(10, Tbins_ARR[Tbins_DIM_0 - 1]);
  auto nHfloor = std::pow(10, nHbins_ARR[0]);
  auto nHceil  = std::pow(10, nHbins_ARR[nHbins_DIM_0 - 1]);

  Real X = 0.75; // Hydrogen mass fraction
  Real Zsol = 0.02; // Solar metallicity

  Real h_rate = hrate;
  Real h_norm = hscale_norm;
  Real h_height = hscale_height;
  Real h_radius = hscale_radius;
  Real T_max_ = T_max;

  par_for("cgm_cooling", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    const Real rho = w0(m,IDN,k,j,i);
    const Real egy = w0(m,IEN,k,j,i);
    const Real temp = temp_unit * egy / rho * gm1;

    const Real m_cap = (temp >= T_max_) ? 1.0 : 0.0;
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

    // --- Energy update
    // First, normal source term:
    const Real dE_source =
        - bdt * X * rho * ( X * rho * (lambda_cooling / cooling_unit)
                                     - (gamma_heating / heating_unit) );

    // Temperature ceiling (applied only when m_cap=1)
    const Real dE_cap = - ((temp - T_max_) / temp_unit) * rho / gm1;
    
    const Real dE_total = (1.0 - m_cap) * dE_source + m_cap * dE_cap;

    u0(m,IEN,k,j,i) += dE_total;
  });

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void SourceTerms::RelCooling()
//! \brief Add explict relativistic cooling in the energy and momentum equations.
// NOTE source terms must be computed using primitive (w0) and NOT conserved (u0) vars

void SourceTerms::RelCooling(const DvceArray5D<Real> &w0, const EOS_Data &eos_data,
                             const Real bdt, DvceArray5D<Real> &u0) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  Real use_e = eos_data.use_e;
  Real gamma = eos_data.gamma;
  Real gm1 = gamma - 1.0;
  Real cooling_rate = crate_rel;
  Real cooling_power = cpower_rel;

  par_for("cooling", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    // temperature in cgs unit
    Real temp = 1.0;
    if (use_e) {
      temp = w0(m,IEN,k,j,i)/w0(m,IDN,k,j,i)*gm1;
    } else {
      temp = w0(m,ITM,k,j,i);
    }

    auto &ux = w0(m,IVX,k,j,i);
    auto &uy = w0(m,IVY,k,j,i);
    auto &uz = w0(m,IVZ,k,j,i);

    auto ut = 1.0 + ux*ux + uy*uy + uz*uz;
    ut = sqrt(ut);

    u0(m,IEN,k,j,i) -= bdt*w0(m,IDN,k,j,i)*ut*pow((temp*cooling_rate), cooling_power);
    u0(m,IM1,k,j,i) -= bdt*w0(m,IDN,k,j,i)*ux*pow((temp*cooling_rate), cooling_power);
    u0(m,IM2,k,j,i) -= bdt*w0(m,IDN,k,j,i)*uy*pow((temp*cooling_rate), cooling_power);
    u0(m,IM3,k,j,i) -= bdt*w0(m,IDN,k,j,i)*uz*pow((temp*cooling_rate), cooling_power);
  });

  return;
}

//----------------------------------------------------------------------------------------
//! \fn SourceTerms::BeamSource()
// \brief Add beam of radiation

void SourceTerms::BeamSource(DvceArray5D<Real> &i0, const Real bdt) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = (pmy_pack->nmb_thispack-1);
  int nang1 = (pmy_pack->prad->prgeo->nangles-1);

  auto &nh_c_ = pmy_pack->prad->nh_c;
  auto &tt = pmy_pack->prad->tet_c;
  auto &tc = pmy_pack->prad->tetcov_c;

  auto &excise = pmy_pack->pcoord->coord_data.bh_excise;
  auto &rad_mask_ = pmy_pack->pcoord->excision_floor;
  Real &n_0_floor_ = pmy_pack->prad->n_0_floor;

  auto &beam_mask_ = pmy_pack->prad->beam_mask;
  Real &dii_dt_ = dii_dt;
  par_for("beam_source",DevExeSpace(),0,nmb1,0,nang1,ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(int m, int n, int k, int j, int i) {
    if (beam_mask_(m,n,k,j,i)) {
      Real n0 = tt(m,0,0,k,j,i);
      Real n_0 = tc(m,0,0,k,j,i)*nh_c_.d_view(n,0) + tc(m,1,0,k,j,i)*nh_c_.d_view(n,1)
               + tc(m,2,0,k,j,i)*nh_c_.d_view(n,2) + tc(m,3,0,k,j,i)*nh_c_.d_view(n,3);
      i0(m,n,k,j,i) += n0*n_0*dii_dt_*bdt;
      // handle excision
      // NOTE(@pdmullen): exicision criterion are not finalized.  The below zeroes all
      // intensities within rks <= 1.0 and zeroes intensities within angles where n_0
      // is about zero.  This needs future attention.
      if (excise) {
        if (rad_mask_(m,k,j,i) || fabs(n_0) < n_0_floor_) { i0(m,n,k,j,i) = 0.0; }
      }
    }
  });

  return;
}
