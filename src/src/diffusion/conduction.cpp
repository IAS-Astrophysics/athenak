//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file conduction.cpp
//! \brief Implements functions for Conduction class. This includes isotropic thermal
//! conduction, in which heat flux is proportional to negative local temperature gradient.
//! Conduction may be added to Hydro and/or MHD independently.

#include <float.h>
#include <algorithm>
#include <limits>
#include <string>
#include <iostream> // cout

// Athena++ headers
#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "eos/eos.hpp"
#include "conduction.hpp"
#include "units/units.hpp"

KOKKOS_INLINE_FUNCTION
Real VanLeerLimiter(const Real a, const Real b) {
  if (a*b > 0) {
    return 2.0*a*b/(a+b);
  } else {
    return 0.0;
  }
}

KOKKOS_INLINE_FUNCTION
Real VL4Limiter(const Real a, const Real b, const Real c, const Real d) {
  return VanLeerLimiter(VanLeerLimiter(a,b),VanLeerLimiter(c,d));
}

//----------------------------------------------------------------------------------------
//! \fn Real KappaTemp()
//! \brief Temperature-dependent conductivity given by Parker (1953) and Spitzer (1962)
KOKKOS_INLINE_FUNCTION
Real KappaTemp(Real temp, Real ceiling) {
  if (temp < 6.5e4) {
    return 2.5e3 * pow(temp, 0.5);
  } else {
    return fmin(6e-7 * pow(temp, 2.5),ceiling);
  }
}

//----------------------------------------------------------------------------------------
//! \brief Conduction constructor
// Note that the coefficient of thermal conduction, kappa, corresponds to conductivity,
// not diffusivity. This is different from the coefficient used in Athena++.

Conduction::Conduction(std::string block, MeshBlockPack *pp, ParameterInput *pin) :
  pmy_pack(pp) {
  // Check that EOS is ideal
  if (pmy_pack->phydro != nullptr) {
    const bool &is_ideal = pmy_pack->phydro->peos->eos_data.is_ideal;
    if (is_ideal == false) {
      std::cout << "### FATAL ERROR in "<< __FILE__ <<" at line " << __LINE__ << std::endl
                << "Thermal conduction only works for ideal gas" << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
  if (pmy_pack->pmhd != nullptr) {
    const bool &is_ideal = pmy_pack->pmhd->peos->eos_data.is_ideal;
    if (is_ideal == false) {
      std::cout << "### FATAL ERROR in "<< __FILE__ <<" at line " << __LINE__ << std::endl
                << "Thermal conduction only works for ideal gas" << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }

  // Read parameters for thermal conduction (if any)
  kappa = pin->GetOrAddReal(block,"conductivity",0.0);
  tdep_kappa = pin->GetOrAddBoolean(block,"tdep_conductivity",false);
  kappa_ceiling = pin->GetOrAddReal(block,"cond_ceiling",
                  static_cast<Real>(std::numeric_limits<float>::max()));
  sat_hflux = pin->GetOrAddBoolean(block,"sat_hflux",false);
}

//----------------------------------------------------------------------------------------
//! \brief Conduction destructor

Conduction::~Conduction() {
}

//----------------------------------------------------------------------------------------
//! \fn void AddHeatFlux()
//! \brief Adds heat flux to face-centered fluxes of conserved variables

void Conduction::AddHeatFlux(const DvceArray5D<Real> &w0, const EOS_Data &eos,
  DvceFaceFld5D<Real> &flx) {
  if (tdep_kappa) {
    TempDependentHeatFlux(w0, eos, flx);
  } else if (kappa > 0.0) {
    IsotropicHeatFlux(w0, eos, flx);
  } else {
    return;
  }
  return;
}
//----------------------------------------------------------------------------------------
//! \fn void IsotropicHeatFlux()
//! \brief Adds isotropic heat flux to face-centered fluxes of conserved variables

void Conduction::IsotropicHeatFlux(const DvceArray5D<Real> &w0, const EOS_Data &eos,
  DvceFaceFld5D<Real> &flx) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  auto size = pmy_pack->pmb->mb_size;
  const bool &use_e = eos.use_e;
  Real gm1 = eos.gamma-1.0;
  Real kappa_ = kappa;

  //--------------------------------------------------------------------------------------
  // fluxes in x1-direction

  auto &flx1 = flx.x1f;

  par_for("conduct1", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie+1,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    // Add heat fluxes into fluxes of conserved variables: energy
    Real dtempdx = 0.0;
    if (use_e) {
      dtempdx = (w0(m,IEN,k,j,i)/w0(m,IDN,k,j,i) - w0(m,IEN,k,j,i-1)/w0(m,IDN,k,j,i-1))
                * gm1 / size.d_view(m).dx1;
    } else {
      dtempdx = (w0(m,ITM,k,j,i) - w0(m,ITM,k,j,i-1)) / size.d_view(m).dx1;
    }
    flx1(m,IEN,k,j,i) -= kappa_ * dtempdx;
  });
  if (pmy_pack->pmesh->one_d) {return;}

  //--------------------------------------------------------------------------------------
  // fluxes in x2-direction

  auto &flx2 = flx.x2f;

  par_for("conduct2",DevExeSpace(), 0, nmb1, ks, ke, js, je+1, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    // Add heat fluxes into fluxes of conserved variables: energy
    Real dtempdx = 0.0;
    if (use_e) {
      dtempdx = (w0(m,IEN,k,j,i)/w0(m,IDN,k,j,i) - w0(m,IEN,k,j-1,i)/w0(m,IDN,k,j-1,i))
                * gm1 / size.d_view(m).dx2;
    } else {
      dtempdx = (w0(m,ITM,k,j,i) - w0(m,ITM,k,j-1,i)) / size.d_view(m).dx2;
    }
    flx2(m,IEN,k,j,i) -= kappa_ * dtempdx;
  });
  if (pmy_pack->pmesh->two_d) {return;}

  //--------------------------------------------------------------------------------------
  // fluxes in x3-direction

  auto &flx3 = flx.x3f;

  par_for("conduct3",DevExeSpace(), 0, nmb1, ks, ke+1, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    // Add heat fluxes into fluxes of conserved variables: energy
    Real dtempdx = 0.0;
    if (use_e) {
      dtempdx = (w0(m,IEN,k,j,i)/w0(m,IDN,k,j,i) - w0(m,IEN,k-1,j,i)/w0(m,IDN,k-1,j,i))
                * gm1 / size.d_view(m).dx3;
    } else {
      dtempdx = (w0(m,ITM,k,j,i) - w0(m,ITM,k-1,j,i)) / size.d_view(m).dx3;
    }
    flx3(m,IEN,k,j,i) -= kappa_ * dtempdx;
  });

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void TempDependentHeatFlux()
//! \brief Adds heat flux to face-centered fluxes of conserved variables with
//! temperature-dependent conductivity

void Conduction::TempDependentHeatFlux(const DvceArray5D<Real> &w0, const EOS_Data &eos,
  DvceFaceFld5D<Real> &flx) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  auto size = pmy_pack->pmb->mb_size;
  const bool &use_e = eos.use_e;
  const bool &sat_hflux_ = sat_hflux;
  bool &multi_d = pmy_pack->pmesh->multi_d;
  bool &three_d = pmy_pack->pmesh->three_d;
  Real gm1 = eos.gamma-1.0;
  Real kappaceil = kappa_ceiling;
  Real temp_unit = pmy_pack->punit->temperature_cgs();
  Real kappa_unit = pmy_pack->punit->pressure_cgs()*pmy_pack->punit->velocity_cgs()*
                    pmy_pack->punit->length_cgs()/pmy_pack->punit->temperature_cgs();

  //--------------------------------------------------------------------------------------
  // fluxes in x1-direction

  auto &flx1 = flx.x1f;

  par_for("conduct1", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie+1,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    // Add heat fluxes into fluxes of conserved variables: energy
    Real temp_l = 0.0, temp_r = 0.0, pres_l = 0.0, pres_r = 0.0;
    Real temp_ll = 0.0, temp_lr = 0.0, temp_rl = 0.0, temp_rr = 0.0;
    if (use_e) {
      temp_l = w0(m,IEN,k,j,i-1)/w0(m,IDN,k,j,i-1)*gm1;
      temp_r = w0(m,IEN,k,j,i)/w0(m,IDN,k,j,i)*gm1;
      pres_l = w0(m,IEN,k,j,i-1)*gm1;
      pres_r = w0(m,IEN,k,j,i)*gm1;
    } else {
      temp_l = w0(m,ITM,k,j,i-1);
      temp_r = w0(m,ITM,k,j,i);
      pres_l = w0(m,ITM,k,j,i-1)*w0(m,IDN,k,j,i-1);
      pres_r = w0(m,ITM,k,j,i)*w0(m,IDN,k,j,i);
    }
    Real kappaf = 0.5*(KappaTemp(temp_unit*temp_l,kappaceil)+
                  KappaTemp(temp_unit*temp_r,kappaceil))/kappa_unit;
    Real dtempdx1 = (temp_r-temp_l)/size.d_view(m).dx1;
    Real hflx = kappaf*dtempdx1;
    // Saturation of thermal conduction by harmonic mean
    if (sat_hflux_) {
      Real dtempdx2 = 0.0, dtempdx3 = 0.0;
      if (multi_d) {
        if (use_e) {
          temp_ll = w0(m,IEN,k,j-1,i-1)/w0(m,IDN,k,j-1,i-1)*gm1;
          temp_lr = w0(m,IEN,k,j+1,i-1)/w0(m,IDN,k,j+1,i-1)*gm1;
          temp_rl = w0(m,IEN,k,j-1,i)/w0(m,IDN,k,j-1,i)*gm1;
          temp_rr = w0(m,IEN,k,j+1,i)/w0(m,IDN,k,j+1,i)*gm1;
        } else {
          temp_ll = w0(m,ITM,k,j-1,i-1);
          temp_lr = w0(m,ITM,k,j+1,i-1);
          temp_rl = w0(m,ITM,k,j-1,i);
          temp_rr = w0(m,ITM,k,j+1,i);
        }
        dtempdx2 = VL4Limiter(temp_rr-temp_r,temp_r-temp_rl,
                              temp_lr-temp_l,temp_l-temp_ll)/size.d_view(m).dx2;
      }
      if (three_d) {
        if (use_e) {
          temp_ll = w0(m,IEN,k-1,j,i-1)/w0(m,IDN,k-1,j,i-1)*gm1;
          temp_lr = w0(m,IEN,k+1,j,i-1)/w0(m,IDN,k+1,j,i-1)*gm1;
          temp_rl = w0(m,IEN,k-1,j,i)/w0(m,IDN,k-1,j,i)*gm1;
          temp_rr = w0(m,IEN,k+1,j,i)/w0(m,IDN,k+1,j,i)*gm1;
        } else {
          temp_ll = w0(m,ITM,k-1,j,i-1);
          temp_lr = w0(m,ITM,k+1,j,i-1);
          temp_rl = w0(m,ITM,k-1,j,i);
          temp_rr = w0(m,ITM,k+1,j,i);
        }
        dtempdx3 = VL4Limiter(temp_rr-temp_r,temp_r-temp_rl,
                              temp_lr-temp_l,temp_l-temp_ll)/size.d_view(m).dx3;
      }
      Real tempgrad = sqrt(SQR(dtempdx1)+SQR(dtempdx2)+SQR(dtempdx3));
      Real pres_cs = 0.5*(pres_l*sqrt(temp_l)+pres_r*sqrt(temp_r));
      Real sat_fac = 1.0/(1.0+kappaf*tempgrad/(1.5*pres_cs));
      hflx *= sat_fac;
    }
    flx1(m,IEN,k,j,i) -= hflx;
  });
  if (pmy_pack->pmesh->one_d) {return;}

  //--------------------------------------------------------------------------------------
  // fluxes in x2-direction

  auto &flx2 = flx.x2f;

  par_for("conduct2",DevExeSpace(), 0, nmb1, ks, ke, js, je+1, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    // Add heat fluxes into fluxes of conserved variables: energy
    Real temp_l = 0.0, temp_r = 0.0, pres_l = 0.0, pres_r = 0.0;
    Real temp_ll = 0.0, temp_lr = 0.0, temp_rl = 0.0, temp_rr = 0.0;
    if (use_e) {
      temp_l = w0(m,IEN,k,j-1,i)/w0(m,IDN,k,j-1,i)*gm1;
      temp_r = w0(m,IEN,k,j,i)/w0(m,IDN,k,j,i)*gm1;
      pres_l = w0(m,IEN,k,j-1,i)*gm1;
      pres_r = w0(m,IEN,k,j,i)*gm1;
    } else {
      temp_l = w0(m,ITM,k,j-1,i);
      temp_r = w0(m,ITM,k,j,i);
      pres_l = w0(m,ITM,k,j-1,i)*w0(m,IDN,k,j-1,i);
      pres_r = w0(m,ITM,k,j,i)*w0(m,IDN,k,j,i);
    }
    Real kappaf = 0.5*(KappaTemp(temp_unit*temp_l,kappaceil)+
                  KappaTemp(temp_unit*temp_r,kappaceil))/kappa_unit;
    Real dtempdx2 = (temp_r-temp_l)/size.d_view(m).dx2;
    Real hflx = kappaf*dtempdx2;
    // Saturation of thermal conduction
    if (sat_hflux_) {
      Real dtempdx1 = 0.0, dtempdx3 = 0.0;
      if (use_e) {
        temp_ll = w0(m,IEN,k,j-1,i-1)/w0(m,IDN,k,j-1,i-1)*gm1;
        temp_lr = w0(m,IEN,k,j-1,i+1)/w0(m,IDN,k,j-1,i+1)*gm1;
        temp_rl = w0(m,IEN,k,j,i-1)/w0(m,IDN,k,j,i-1)*gm1;
        temp_rr = w0(m,IEN,k,j,i+1)/w0(m,IDN,k,j,i+1)*gm1;
      } else {
        temp_ll = w0(m,ITM,k,j-1,i-1);
        temp_lr = w0(m,ITM,k,j-1,i+1);
        temp_rl = w0(m,ITM,k,j,i-1);
        temp_rr = w0(m,ITM,k,j,i+1);
      }
      dtempdx1 = VL4Limiter(temp_rr-temp_r,temp_r-temp_rl,
                            temp_lr-temp_l,temp_l-temp_ll)/size.d_view(m).dx1;
      if (three_d) {
        if (use_e) {
          temp_ll = w0(m,IEN,k-1,j-1,i)/w0(m,IDN,k-1,j-1,i)*gm1;
          temp_lr = w0(m,IEN,k+1,j-1,i)/w0(m,IDN,k+1,j-1,i)*gm1;
          temp_rl = w0(m,IEN,k-1,j,i)/w0(m,IDN,k-1,j,i)*gm1;
          temp_rr = w0(m,IEN,k+1,j,i)/w0(m,IDN,k+1,j,i)*gm1;
        } else {
          temp_ll = w0(m,ITM,k-1,j-1,i);
          temp_lr = w0(m,ITM,k+1,j-1,i);
          temp_rl = w0(m,ITM,k-1,j,i);
          temp_rr = w0(m,ITM,k+1,j,i);
        }
        dtempdx3 = VL4Limiter(temp_rr-temp_r,temp_r-temp_rl,
                              temp_lr-temp_l,temp_l-temp_ll)/size.d_view(m).dx3;
      }
      Real tempgrad = sqrt(SQR(dtempdx1)+SQR(dtempdx2)+SQR(dtempdx3));
      Real pres_cs = 0.5*(pres_l*sqrt(temp_l)+pres_r*sqrt(temp_r));
      Real sat_fac = 1.0/(1.0+kappaf*tempgrad/(1.5*pres_cs));
      hflx *= sat_fac;
    }
    flx2(m,IEN,k,j,i) -= hflx;
  });
  if (pmy_pack->pmesh->two_d) {return;}

  //--------------------------------------------------------------------------------------
  // fluxes in x3-direction

  auto &flx3 = flx.x3f;

  par_for("conduct3",DevExeSpace(), 0, nmb1, ks, ke+1, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    // Add heat fluxes into fluxes of conserved variables: energy
    Real temp_l = 0.0, temp_r = 0.0, pres_l = 0.0, pres_r = 0.0;
    Real temp_ll = 0.0, temp_lr = 0.0, temp_rl = 0.0, temp_rr = 0.0;
    if (use_e) {
      temp_l = w0(m,IEN,k-1,j,i)/w0(m,IDN,k-1,j,i)*gm1;
      temp_r = w0(m,IEN,k,j,i)/w0(m,IDN,k,j,i)*gm1;
      pres_l = w0(m,IEN,k-1,j,i)*gm1;
      pres_r = w0(m,IEN,k,j,i)*gm1;
    } else {
      temp_l = w0(m,ITM,k-1,j,i);
      temp_r = w0(m,ITM,k,j,i);
      pres_l = w0(m,ITM,k-1,j,i)*w0(m,IDN,k-1,j,i);
      pres_r = w0(m,ITM,k,j,i)*w0(m,IDN,k,j,i);
    }
    Real kappaf = 0.5*(KappaTemp(temp_unit*temp_l,kappaceil)+
                  KappaTemp(temp_unit*temp_r,kappaceil))/kappa_unit;
    Real dtempdx3 = (temp_r-temp_l)/size.d_view(m).dx3;
    Real hflx = kappaf*dtempdx3;
    // Saturation of thermal conduction
    if (sat_hflux_) {
      Real dtempdx1 = 0.0, dtempdx2 = 0.0;
      if (use_e) {
        temp_ll = w0(m,IEN,k-1,j,i-1)/w0(m,IDN,k-1,j,i-1)*gm1;
        temp_lr = w0(m,IEN,k-1,j,i+1)/w0(m,IDN,k-1,j,i+1)*gm1;
        temp_rl = w0(m,IEN,k,j,i-1)/w0(m,IDN,k,j,i-1)*gm1;
        temp_rr = w0(m,IEN,k,j,i+1)/w0(m,IDN,k,j,i+1)*gm1;
      } else {
        temp_ll = w0(m,ITM,k-1,j,i-1);
        temp_lr = w0(m,ITM,k-1,j,i+1);
        temp_rl = w0(m,ITM,k,j,i-1);
        temp_rr = w0(m,ITM,k,j,i+1);
      }
      dtempdx1 = VL4Limiter(temp_rr-temp_r,temp_r-temp_rl,
                            temp_lr-temp_l,temp_l-temp_ll)/size.d_view(m).dx1;
      if (use_e) {
        temp_ll = w0(m,IEN,k-1,j-1,i)/w0(m,IDN,k-1,j-1,i)*gm1;
        temp_lr = w0(m,IEN,k-1,j+1,i)/w0(m,IDN,k-1,j+1,i)*gm1;
        temp_rl = w0(m,IEN,k,j-1,i)/w0(m,IDN,k,j-1,i)*gm1;
        temp_rr = w0(m,IEN,k,j+1,i)/w0(m,IDN,k,j+1,i)*gm1;
      } else {
        temp_ll = w0(m,ITM,k-1,j-1,i);
        temp_lr = w0(m,ITM,k-1,j+1,i);
        temp_rl = w0(m,ITM,k,j-1,i);
        temp_rr = w0(m,ITM,k,j+1,i);
      }
      dtempdx2 = VL4Limiter(temp_rr-temp_r,temp_r-temp_rl,
                            temp_lr-temp_l,temp_l-temp_ll)/size.d_view(m).dx2;
      Real tempgrad = sqrt(SQR(dtempdx1)+SQR(dtempdx2)+SQR(dtempdx3));
      Real pres_cs = 0.5*(pres_l*sqrt(temp_l)+pres_r*sqrt(temp_r));
      Real sat_fac = 1.0/(1.0+kappaf*tempgrad/(1.5*pres_cs));
      hflx *= sat_fac;
    }
    flx3(m,IEN,k,j,i) -= hflx;
  });

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void Conduction::NewTimeStep()
//! \brief Compute new time step for thermal conduction.

void Conduction::NewTimeStep(const DvceArray5D<Real> &w0, const EOS_Data &eos_data) {
  if (sat_hflux == true) {
    dtnew = static_cast<Real>(std::numeric_limits<float>::max());
    return;
  }
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, nx1 = indcs.nx1;
  int js = indcs.js, nx2 = indcs.nx2;
  int ks = indcs.ks, nx3 = indcs.nx3;
  const int nmkji = (pmy_pack->nmb_thispack)*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1;
  const int nji  = nx2*nx1;
  auto &w0_ = w0;
  auto &multi_d = pmy_pack->pmesh->multi_d;
  auto &three_d = pmy_pack->pmesh->three_d;
  auto &size = pmy_pack->pmb->mb_size;
  const bool &use_e = eos_data.use_e;
  Real gm1 = eos_data.gamma-1.0;
  Real kappa0 = kappa;
  bool tdepkappa = tdep_kappa;
  Real kappaceil = kappa_ceiling;
  Real fac;
  if (pmy_pack->pmesh->three_d) {
    fac = 1.0/6.0;
  } else if (pmy_pack->pmesh->two_d) {
    fac = 0.25;
  } else {
    fac = 0.5;
  }

  Real temp_unit = pmy_pack->punit->temperature_cgs();
  Real kappa_unit = pmy_pack->punit->pressure_cgs()*pmy_pack->punit->velocity_cgs()*
                    pmy_pack->punit->length_cgs()/pmy_pack->punit->temperature_cgs();

  dtnew = static_cast<Real>(std::numeric_limits<float>::max());

  // find smallest timestep for thermal conduction in each cell
  Kokkos::parallel_reduce("cond_newdt", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, Real &min_dt) {
    // compute m,k,j,i indices of thread and call function
    int m = (idx)/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks;
    j += js;

    Real kappa_ = kappa0;
    if (tdepkappa) {
      Real temp = 1.0;
      if (use_e) {
        temp = w0(m,IEN,k,j,i)/w0(m,IDN,k,j,i)*gm1;
      } else {
        temp = w0(m,ITM,k,j,i);
      }
      kappa_ = KappaTemp(temp*temp_unit,kappaceil)/kappa_unit;
    }

    min_dt = fmin(min_dt, SQR(size.d_view(m).dx1)/kappa_*w0_(m,IDN,k,j,i)/gm1);
    if (multi_d) {
      min_dt = fmin(min_dt, SQR(size.d_view(m).dx2)/kappa_*w0_(m,IDN,k,j,i)/gm1);
    }
    if (three_d) {
      min_dt = fmin(min_dt, SQR(size.d_view(m).dx3)/kappa_*w0_(m,IDN,k,j,i)/gm1);
    }
  }, Kokkos::Min<Real>(dtnew));

  dtnew *= fac;

  return;
}
