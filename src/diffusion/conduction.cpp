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

// VanLeer Limiter which takes 2 slopes
KOKKOS_INLINE_FUNCTION
Real VLL2State(const Real a, const Real b) {
  if (a*b > 0) {
    return 2.0*a*b/(a+b);
  } else {
    return 0.0;
  }
}

// VanLeer Limiter which takes 4 slopes
KOKKOS_INLINE_FUNCTION
Real VLL4State(const Real a, const Real b, const Real c, const Real d) {
  return VLL2State(VLL2State(a,b), VLL2State(c,d));
}

//----------------------------------------------------------------------------------------
//! \fn Real TempDepKappa()
//! \brief Temperature-dependent conductivity given by Parker (1953) and Spitzer (1962)

KOKKOS_INLINE_FUNCTION
Real TempDepKappa(Real temp, Real limit) {
  if (temp < 6.5e4) {
    return 2.5e3 * pow(temp, 0.5);
  } else {
    return fmin(6.0e-7*pow(temp, 2.5), limit);
  }
}

//----------------------------------------------------------------------------------------
//! \brief Conduction constructor
// Note first argument passes string ("hydro" or "mhd") denoting in wihch class this
// object is being constructed, and therefore which <block> in the input file from which
// the parameters are read.
// Note that the coefficients of thermal conduction, alpha_iso, etc., correspond to
// diffusivities. The conductivity is kappa = (dens)*alpha, and the energy flux
// q = -kappa * (dT/dx) = - alpha * d * *dT/dx)

Conduction::Conduction(std::string block, MeshBlockPack *pp, ParameterInput *pin) :
    pmy_pack(pp) {
  // Read parameters for thermal diffusivity (if any)
  alpha_iso = pin->GetOrAddReal(block,"alpha_iso", 0.0);
  alpha_aniso = pin->GetOrAddReal(block,"alpha_aniso", 0.0);
  alpha_spitzer = pin->GetOrAddBoolean(block,"alpha_spitzer", false);
  // Limit on thermal heat flux (saturated conduction)
  q_limit = pin->GetOrAddReal(block,"q_limit",
                     static_cast<Real>(std::numeric_limits<float>::max()));
}

//----------------------------------------------------------------------------------------
//! \brief Conduction destructor

Conduction::~Conduction() {
}

//----------------------------------------------------------------------------------------
//! \fn void AddHeatFluxes()
//! \brief Wrapper function that adds heat fluxes for different types of thermal
//! conduction to face-centered fluxes of conserved variables

void Conduction::AddHeatFluxes(const DvceArray5D<Real> &w0, const EOS_Data &eos,
    DvceFaceFld5D<Real> &flx) {
  if (alpha_iso != 0) {
    AddHeatFluxIso(w0, eos, flx);
  }
  if (alpha_aniso != 0) {
    AddHeatFluxAniso(w0, eos, flx);
  }
  if (alpha_spitzer) {
    AddHeatFluxSpitzer(w0, eos, flx);
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void AddHeatFluxIso()
//! \brief Adds isotropic heat flux computed using constant conductivity to face-centered
//! fluxes of conserved variables

void Conduction::AddHeatFluxIso(const DvceArray5D<Real> &w0, const EOS_Data &eos,
    DvceFaceFld5D<Real> &flx) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  auto size = pmy_pack->pmb->mb_size;
  Real gm1 = eos.gamma-1.0;
  Real &alpha_ = alpha_iso;

  // fluxes in x1-direction
  auto &flx1 = flx.x1f;
  par_for("conduct1", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie+1,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real tempr = w0(m,IEN,k,j,i  )/w0(m,IDN,k,j,i  );
    Real templ = w0(m,IEN,k,j,i-1)/w0(m,IDN,k,j,i-1);
    Real dtempdx = (tempr - templ) * gm1 / size.d_view(m).dx1;
    Real densf = 0.5*(w0(m,IDN,k,j,i) + w0(m,IDN,k,j,i-1));
    flx1(m,IEN,k,j,i) -= alpha_ * densf * dtempdx;
  });
  if (pmy_pack->pmesh->one_d) {return;}

  // fluxes in x2-direction
  auto &flx2 = flx.x2f;
  par_for("conduct2",DevExeSpace(), 0, nmb1, ks, ke, js, je+1, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real tempr = w0(m,IEN,k,j  ,i)/w0(m,IDN,k,j  ,i);
    Real templ = w0(m,IEN,k,j-1,i)/w0(m,IDN,k,j-1,i);
    Real dtempdx = (tempr - templ) * gm1 / size.d_view(m).dx2;
    Real densf = 0.5*(w0(m,IDN,k,j,i) + w0(m,IDN,k,j-1,i));
    flx2(m,IEN,k,j,i) -= alpha_ * densf * dtempdx;
  });
  if (pmy_pack->pmesh->two_d) {return;}

  // fluxes in x3-direction
  auto &flx3 = flx.x3f;
  par_for("conduct3",DevExeSpace(), 0, nmb1, ks, ke+1, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real tempr = w0(m,IEN,k  ,j,i)/w0(m,IDN,k  ,j,i);
    Real templ = w0(m,IEN,k-1,j,i)/w0(m,IDN,k-1,j,i);
    Real dtempdx = (tempr - templ) * gm1 / size.d_view(m).dx3;
    Real densf = 0.5*(w0(m,IDN,k,j,i) + w0(m,IDN,k-1,j,i));
    flx3(m,IEN,k,j,i) -= alpha_ * densf * dtempdx;
  });
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void AddHeatFluxAniso()
//! \brief Current a no-op function, to be added later

void Conduction::AddHeatFluxAniso(const DvceArray5D<Real> &w0, const EOS_Data &eos,
    DvceFaceFld5D<Real> &flx) {
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void TempDependentHeatFlux()
//! \brief Adds heat flux to face-centered fluxes of conserved variables with
//! temperature-dependent conductivity

void Conduction::AddHeatFluxSpitzer(const DvceArray5D<Real> &w0, const EOS_Data &eos,
   DvceFaceFld5D<Real> &flx) {
/*
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  auto size = pmy_pack->pmb->mb_size;
  const bool &sat_hflux_ = sat_hflux;
  bool &multi_d = pmy_pack->pmesh->multi_d;
  bool &three_d = pmy_pack->pmesh->three_d;
  Real gm1 = eos.gamma-1.0;
  Real kappaceil = kappa_ceiling;
  Real temp_unit = pmy_pack->punit->temperature_cgs();
  Real kappa_unit = pmy_pack->punit->pressure_cgs()*pmy_pack->punit->velocity_cgs()*
                    pmy_pack->punit->length_cgs()/pmy_pack->punit->temperature_cgs();

  // fluxes in x1-direction
  auto &flx1 = flx.x1f;
  par_for("conduct1", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie+1,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    // Add heat fluxes into fluxes of conserved variables: energy
    Real temp_l = w0(m,IEN,k,j,i-1)/w0(m,IDN,k,j,i-1)*gm1;
    Real temp_r = w0(m,IEN,k,j,i)/w0(m,IDN,k,j,i)*gm1;
    Real pres_l = w0(m,IEN,k,j,i-1)*gm1;
    Real pres_r = w0(m,IEN,k,j,i)*gm1;
    Real kappaf = 0.5*(TempDepKappa(temp_unit*temp_l,kappaceil)+
                  TempDepKappa(temp_unit*temp_r,kappaceil))/kappa_unit;
    Real dtempdx1 = (temp_r-temp_l)/size.d_view(m).dx1;
    Real hflx = kappaf*dtempdx1;
    // Saturation of thermal conduction by harmonic mean
    if (sat_hflux_) {
      Real dtempdx2 = 0.0, dtempdx3 = 0.0;
      if (multi_d) {
        temp_ll = w0(m,IEN,k,j-1,i-1)/w0(m,IDN,k,j-1,i-1)*gm1;
        temp_lr = w0(m,IEN,k,j+1,i-1)/w0(m,IDN,k,j+1,i-1)*gm1;
        temp_rl = w0(m,IEN,k,j-1,i)/w0(m,IDN,k,j-1,i)*gm1;
        temp_rr = w0(m,IEN,k,j+1,i)/w0(m,IDN,k,j+1,i)*gm1;
        dtempdx2 = VanLeerLimiter4State(temp_rr-temp_r,temp_r-temp_rl,
                                        temp_lr-temp_l,temp_l-temp_ll)/size.d_view(m).dx2;
      }
      if (three_d) {
        temp_ll = w0(m,IEN,k-1,j,i-1)/w0(m,IDN,k-1,j,i-1)*gm1;
        temp_lr = w0(m,IEN,k+1,j,i-1)/w0(m,IDN,k+1,j,i-1)*gm1;
        temp_rl = w0(m,IEN,k-1,j,i)/w0(m,IDN,k-1,j,i)*gm1;
        temp_rr = w0(m,IEN,k+1,j,i)/w0(m,IDN,k+1,j,i)*gm1;
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

  // fluxes in x2-direction
  auto &flx2 = flx.x2f;
  par_for("conduct2",DevExeSpace(), 0, nmb1, ks, ke, js, je+1, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    // Add heat fluxes into fluxes of conserved variables: energy
    Real temp_l = 0.0, temp_r = 0.0, pres_l = 0.0, pres_r = 0.0;
    Real temp_ll = 0.0, temp_lr = 0.0, temp_rl = 0.0, temp_rr = 0.0;
    temp_l = w0(m,IEN,k,j-1,i)/w0(m,IDN,k,j-1,i)*gm1;
    temp_r = w0(m,IEN,k,j,i)/w0(m,IDN,k,j,i)*gm1;
    pres_l = w0(m,IEN,k,j-1,i)*gm1;
    pres_r = w0(m,IEN,k,j,i)*gm1;
    Real kappaf = 0.5*(TempDepKappa(temp_unit*temp_l,kappaceil)+
                  TempDepKappa(temp_unit*temp_r,kappaceil))/kappa_unit;
    Real dtempdx2 = (temp_r-temp_l)/size.d_view(m).dx2;
    Real hflx = kappaf*dtempdx2;
    // Saturation of thermal conduction
    if (sat_hflux_) {
      Real dtempdx1 = 0.0, dtempdx3 = 0.0;
      temp_ll = w0(m,IEN,k,j-1,i-1)/w0(m,IDN,k,j-1,i-1)*gm1;
      temp_lr = w0(m,IEN,k,j-1,i+1)/w0(m,IDN,k,j-1,i+1)*gm1;
      temp_rl = w0(m,IEN,k,j,i-1)/w0(m,IDN,k,j,i-1)*gm1;
      temp_rr = w0(m,IEN,k,j,i+1)/w0(m,IDN,k,j,i+1)*gm1;
      dtempdx1 = VL4Limiter(temp_rr-temp_r,temp_r-temp_rl,
                            temp_lr-temp_l,temp_l-temp_ll)/size.d_view(m).dx1;
      if (three_d) {
        temp_ll = w0(m,IEN,k-1,j-1,i)/w0(m,IDN,k-1,j-1,i)*gm1;
        temp_lr = w0(m,IEN,k+1,j-1,i)/w0(m,IDN,k+1,j-1,i)*gm1;
        temp_rl = w0(m,IEN,k-1,j,i)/w0(m,IDN,k-1,j,i)*gm1;
        temp_rr = w0(m,IEN,k+1,j,i)/w0(m,IDN,k+1,j,i)*gm1;
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

  // fluxes in x3-direction
  auto &flx3 = flx.x3f;
  par_for("conduct3",DevExeSpace(), 0, nmb1, ks, ke+1, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    // Add heat fluxes into fluxes of conserved variables: energy
    Real temp_l = 0.0, temp_r = 0.0, pres_l = 0.0, pres_r = 0.0;
    Real temp_ll = 0.0, temp_lr = 0.0, temp_rl = 0.0, temp_rr = 0.0;
    temp_l = w0(m,IEN,k-1,j,i)/w0(m,IDN,k-1,j,i)*gm1;
    temp_r = w0(m,IEN,k,j,i)/w0(m,IDN,k,j,i)*gm1;
    pres_l = w0(m,IEN,k-1,j,i)*gm1;
    pres_r = w0(m,IEN,k,j,i)*gm1;
    Real kappaf = 0.5*(TempDepKappa(temp_unit*temp_l,kappaceil)+
                  TempDepKappa(temp_unit*temp_r,kappaceil))/kappa_unit;
    Real dtempdx3 = (temp_r-temp_l)/size.d_view(m).dx3;
    Real hflx = kappaf*dtempdx3;
    // Saturation of thermal conduction
    if (sat_hflux_) {
      Real dtempdx1 = 0.0, dtempdx2 = 0.0;
      temp_ll = w0(m,IEN,k-1,j,i-1)/w0(m,IDN,k-1,j,i-1)*gm1;
      temp_lr = w0(m,IEN,k-1,j,i+1)/w0(m,IDN,k-1,j,i+1)*gm1;
      temp_rl = w0(m,IEN,k,j,i-1)/w0(m,IDN,k,j,i-1)*gm1;
      temp_rr = w0(m,IEN,k,j,i+1)/w0(m,IDN,k,j,i+1)*gm1;
      dtempdx1 = VL4Limiter(temp_rr-temp_r,temp_r-temp_rl,
                            temp_lr-temp_l,temp_l-temp_ll)/size.d_view(m).dx1;
      temp_ll = w0(m,IEN,k-1,j-1,i)/w0(m,IDN,k-1,j-1,i)*gm1;
      temp_lr = w0(m,IEN,k-1,j+1,i)/w0(m,IDN,k-1,j+1,i)*gm1;
      temp_rl = w0(m,IEN,k,j-1,i)/w0(m,IDN,k,j-1,i)*gm1;
      temp_rr = w0(m,IEN,k,j+1,i)/w0(m,IDN,k,j+1,i)*gm1;
      dtempdx2 = VL4Limiter(temp_rr-temp_r,temp_r-temp_rl,
                            temp_lr-temp_l,temp_l-temp_ll)/size.d_view(m).dx2;
      Real tempgrad = sqrt(SQR(dtempdx1)+SQR(dtempdx2)+SQR(dtempdx3));
      Real pres_cs = 0.5*(pres_l*sqrt(temp_l)+pres_r*sqrt(temp_r));
      Real sat_fac = 1.0/(1.0+kappaf*tempgrad/(1.5*pres_cs));
      hflx *= sat_fac;
    }
    flx3(m,IEN,k,j,i) -= hflx;
  });

*/
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void Conduction::NewTimeStep()
//! \brief Compute new time step for thermal conduction.

void Conduction::NewTimeStep(const DvceArray5D<Real> &w0, const EOS_Data &eos_data) {
  dtnew = static_cast<Real>(std::numeric_limits<float>::max());
  Real fac;
  if (pmy_pack->pmesh->three_d) {
    fac = 1.0/6.0;
  } else if (pmy_pack->pmesh->two_d) {
    fac = 0.25;
  } else {
    fac = 0.5;
  }
//  if (sat_hflux == true) {
//    dtnew = static_cast<Real>(std::numeric_limits<float>::max());
//    return;
//  }

  // set flag for Spitzer conductivity
  bool spitzer_ = alpha_spitzer;
  Real temp_unit = pmy_pack->punit->temperature_cgs();
  Real kappa_unit = pmy_pack->punit->pressure_cgs()*pmy_pack->punit->velocity_cgs()*
                      pmy_pack->punit->length_cgs()/pmy_pack->punit->temperature_cgs();

  // capture variables for kernel
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
  Real gm1 = eos_data.gamma-1.0;
  Real alpha0 = alpha_iso;

  // find smallest timestep for thermal conduction in each cell
  // Note loop over all cells needed even for constant conductivity
  Kokkos::parallel_reduce("cond_newdt", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, Real &min_dt) {
    // compute m,k,j,i indices of thread and call function
    int m = (idx)/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks;
    j += js;

    Real alpha_ = alpha0;
//    if (spitzer_) {
//      Real temp = w0(m,IEN,k,j,i)/w0(m,IDN,k,j,i)*gm1;
//      kappa_ = TempDepKappa(temp*temp_unit, limit_)/kappa_unit;
//    }

    min_dt = fmin(min_dt, SQR(size.d_view(m).dx1)/alpha_*w0_(m,IDN,k,j,i)/gm1);
    if (multi_d) {
      min_dt = fmin(min_dt, SQR(size.d_view(m).dx2)/alpha_*w0_(m,IDN,k,j,i)/gm1);
    }
    if (three_d) {
      min_dt = fmin(min_dt, SQR(size.d_view(m).dx3)/alpha_*w0_(m,IDN,k,j,i)/gm1);
    }
  }, Kokkos::Min<Real>(dtnew));
  dtnew *= fac;

  return;
}
