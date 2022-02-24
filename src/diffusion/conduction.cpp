//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file conduction.cpp
//! \brief Implements functions for Conduction class. This includes isotropic thermal
//! conduction, in which heat flux is proportional to negative local temperature gradient.
//! Conduction may be added to Hydro and/or MHD independently.

#include <algorithm>
#include <limits>
#include <string>

// Athena++ headers
#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "eos/eos.hpp"
#include "conduction.hpp"

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

  // Read thermal conductivity of isotropic thermal conduction
  kappa = pin->GetReal(block,"conductivity");
}

//----------------------------------------------------------------------------------------
//! \brief Conduction destructor

Conduction::~Conduction() {
}

//----------------------------------------------------------------------------------------
//! \fn void IsotropicHeatFlux()
//! \brief Adds isotropic heat flux to face-centered fluxes of conserved variables

void Conduction::IsotropicHeatFlux(const DvceArray5D<Real> &w0, const Real kappa,
  const EOS_Data &eos, DvceFaceFld5D<Real> &flx) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int ncells1 = indcs.nx1 + 2*(indcs.ng);
  int nmb1 = pmy_pack->nmb_thispack - 1;
  auto size = pmy_pack->pmb->mb_size;
  const bool &use_e = eos.use_e;
  Real gm1 = eos.gamma-1.0;

  //--------------------------------------------------------------------------------------
  // fluxes in x1-direction

  int scr_level = 0;
  size_t scr_size = (ScrArray1D<Real>::shmem_size(ncells1)) * 3;
  auto flx1 = flx.x1f;

  par_for_outer("conduct1", DevExeSpace(), scr_size, scr_level, 0, nmb1, ks, ke, js, je,
  KOKKOS_LAMBDA(TeamMember_t member, const int m, const int k, const int j) {
    ScrArray1D<Real> hflx1(member.team_scratch(scr_level), ncells1);

    // Add heat fluxes into fluxes of conserved variables: energy
    par_for_inner(member, is, ie+1, [&](const int i) {
      if (use_e) {
        hflx1(i) = gm1 * (w0(m,IEN,k,j,i)/w0(m,IDN,k,j,i) -
                   w0(m,IEN,k,j,i-1)/w0(m,IDN,k,j,i-1)) / size.d_view(m).dx1;
      } else {
        hflx1(i) = (w0(m,ITM,k,j,i) - w0(m,ITM,k,j,i-1)) / size.d_view(m).dx1;
      }
      flx1(m,IEN,k,j,i) -= kappa * hflx1(i);
    });
  });
  if (pmy_pack->pmesh->one_d) {return;}

  //--------------------------------------------------------------------------------------
  // fluxes in x2-direction

  auto flx2 = flx.x2f;

  par_for_outer("conduct2",DevExeSpace(), scr_size, scr_level, 0, nmb1, ks, ke, js, je+1,
  KOKKOS_LAMBDA(TeamMember_t member, const int m, const int k, const int j) {
    ScrArray1D<Real> hflx2(member.team_scratch(scr_level), ncells1);

    // Add heat fluxes into fluxes of conserved variables: energy
    par_for_inner(member, is, ie, [&](const int i) {
      if (use_e) {
        hflx2(i) = gm1 * (w0(m,IEN,k,j,i)/w0(m,IDN,k,j,i) -
                   w0(m,IEN,k,j-1,i)/w0(m,IDN,k,j-1,i)) / size.d_view(m).dx2;
      } else {
        hflx2(i) = (w0(m,ITM,k,j,i) - w0(m,ITM,k,j-1,i)) / size.d_view(m).dx2;
      }
      flx2(m,IEN,k,j,i) -= kappa * hflx2(i);
    });
  });
  if (pmy_pack->pmesh->two_d) {return;}

  //--------------------------------------------------------------------------------------
  // fluxes in x3-direction

  auto flx3 = flx.x3f;

  par_for_outer("conduct3",DevExeSpace(), scr_size, scr_level, 0, nmb1, ks, ke+1, js, je,
  KOKKOS_LAMBDA(TeamMember_t member, const int m, const int k, const int j) {
    ScrArray1D<Real> hflx3(member.team_scratch(scr_level), ncells1);

    // Add heat fluxes into fluxes of conserved variables: energy
    par_for_inner(member, is, ie, [&](const int i) {
      if (use_e) {
        hflx3(i) = gm1 * (w0(m,IEN,k,j,i)/w0(m,IDN,k,j,i) -
                   w0(m,IEN,k-1,j,i)/w0(m,IDN,k-1,j,i)) / size.d_view(m).dx3;
      } else {
        hflx3(i) = (w0(m,ITM,k,j,i) - w0(m,ITM,k-1,j,i)) / size.d_view(m).dx3;
      }
      flx3(m,IEN,k,j,i) -= kappa * hflx3(i);
    });
  });

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void Conduction::NewTimeStep()
//! \brief Compute new time step for thermal conduction.

void Conduction::NewTimeStep(const DvceArray5D<Real> &w0, const EOS_Data &eos_data) {
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
  Real &kappa_ = kappa;
  Real fac;
  if (pmy_pack->pmesh->three_d) {
    fac = 1.0/6.0;
  } else if (pmy_pack->pmesh->two_d) {
    fac = 0.25;
  } else {
    fac = 0.5;
  }

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
