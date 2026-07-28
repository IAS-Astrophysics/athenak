//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file rhine.cpp
//! \brief Implementation of RHINE class constructor and source-term application.

#include <cstdlib>
#include <iostream>
#include <string>

#include "rhine.hpp"
#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock_pack.hpp"
#include "mhd/mhd.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"
#include "driver/driver.hpp"
#include "units/units.hpp"
#include "coordinates/adm.hpp"

namespace rhine {

namespace {
//----------------------------------------------------------------------------------------
//! \fn Real SqrtGammaVcoord(...)
//! \brief Return sqrt(gamma) * v_coord^dir at cell (m,k,j,i), where v_coord^i = alpha v^i
//!        - beta^i is the fluid coordinate 3-velocity.
template<class ADMVars>
KOKKOS_INLINE_FUNCTION
Real SqrtGammaVcoord(const ADMVars &adm, const DvceArray5D<Real> &w0,
                     int m, int k, int j, int i, int dir) {
  const Real g11 = adm.g_dd(m,0,0,k,j,i), g12 = adm.g_dd(m,0,1,k,j,i);
  const Real g13 = adm.g_dd(m,0,2,k,j,i), g22 = adm.g_dd(m,1,1,k,j,i);
  const Real g23 = adm.g_dd(m,1,2,k,j,i), g33 = adm.g_dd(m,2,2,k,j,i);
  const Real sdetg = Kokkos::sqrt(adm::SpatialDet(g11, g12, g13, g22, g23, g33));

  const Real wv1 = w0(m, IVX, k, j, i);
  const Real wv2 = w0(m, IVY, k, j, i);
  const Real wv3 = w0(m, IVZ, k, j, i);
  const Real Wvsq = g11*wv1*wv1 + g22*wv2*wv2 + g33*wv3*wv3
                  + 2.0*(g12*wv1*wv2 + g13*wv1*wv3 + g23*wv2*wv3);
  const Real iW = 1.0 / Kokkos::sqrt(1.0 + Wvsq);

  const Real wv_dir = (dir == 0) ? wv1 : ((dir == 1) ? wv2 : wv3);
  const Real vcoord = adm.alpha(m, k, j, i) * (wv_dir * iW) - adm.beta_u(m, dir, k, j, i);
  return sdetg * vcoord;
}
}  // namespace

//----------------------------------------------------------------------------------------
//! \fn void RHINE::RHINE(MeshBlockPack *ppack, ParameterInput *pin)
//! \brief RHINE constructor: validate the run configuration and load the networks.
RHINE::RHINE(MeshBlockPack *ppack, ParameterInput *pin) :
  pmy_pack(ppack)
{
  // Some fail-fast flags.
  bool is_dynamical_relativistic = (pin->DoesBlockExist("adm") || pin->DoesBlockExist("z4c"))
                         && pin->DoesBlockExist("mhd");
  if (!is_dynamical_relativistic) {
    std::cout << "### FATAL ERROR in "<< __FILE__ <<" at line " << __LINE__ << std::endl
              << "RHINE requires dyn_grmhd!" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  if (!pin->DoesBlockExist("units")) {
    std::cout << "### FATAL ERROR in "<< __FILE__ <<" at line " << __LINE__ << std::endl
              << "Block <units> required with RHINE!" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  std::string eos_string = pin->GetString("mhd", "dyn_eos");
  if (eos_string.compare("compose") != 0) {
    std::cout << "### FATAL ERROR in " <<__FILE__ << " at line " << __LINE__
              << std::endl << "RHINE needs compose EOS!" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  if (pmy_pack->pmhd->nscalars < 6) {
    std::cout << "### FATAL ERROR in " <<__FILE__ << " at line " << __LINE__
              << std::endl << "RHINE needs >= 6 scalars!" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // Parameters and network loading.
  pmode = pin->GetOrAddInteger("rhine", "pmode", 1);
  std::string models_path = pin->GetString("rhine", "models_path");
  nets.InitFromFiles(models_path);
}

//----------------------------------------------------------------------------------------
//! \fn void RHINE::~RHINE()
//! \brief RHINE destructor function.
RHINE::~RHINE()
{

}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus RHINE::AddSources(Driver *pdrive, int stage)
//! \brief Evaluate RHINE per cell and add the resulting source terms to the MHD conserved
//!        variables.
TaskStatus RHINE::AddSources(Driver *pdrive, int stage) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb = pmy_pack->nmb_thispack;

  auto &w0   = pmy_pack->pmhd->w0;
  auto &u0   = pmy_pack->pmhd->u0;
  auto &temp = pmy_pack->pdyngr->temperature;
  auto &size = pmy_pack->pmb->mb_size;
  auto &adm  = pmy_pack->padm->adm;

  // Code -> cgs conversions.
  const Real dens_to_cgs = pmy_pack->punit->density_cgs();
  const Real time_cgs    = pmy_pack->punit->time_cgs();
  const Real mass_cgs    = pmy_pack->punit->mass_cgs();
  const Real energy_cgs  = pmy_pack->punit->energy_cgs();

  const Real bdt    = (pdrive->beta[stage-1]) * (pmy_pack->pmesh->dt);
  const Real dt_cgs = bdt * time_cgs;

  const bool multi_d = pmy_pack->pmesh->multi_d;
  const bool three_d = pmy_pack->pmesh->three_d;

  // Capture by value so device lambdas get the network View handles / scalars.
  RhineNets nets_ = nets;
  const int pmode_ = pmode;

  // Energy released per baryon = -d(mass excess) [MeV/baryon]; a fraction fnu is lost to
  // neutrinos, leaving (1-fnu) as local thermal energy.
  constexpr Real MEV_TO_ERG = 1.602176634e-6;    // erg per MeV
  constexpr Real M_BARYON_G = 1.660538921e-24;   // g (atomic mass unit)

  par_for("rhine_src", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    const Real rho  = w0(m, IDN, k, j, i) * dens_to_cgs;
    const Real tmev = temp(m, 0, k, j, i);
    const Real ye   = w0(m, I_YE, k, j, i);
    const Real yn   = w0(m, I_YN, k, j, i);
    const Real ya   = w0(m, I_YA, k, j, i);
    const Real yh   = w0(m, I_YH, k, j, i);
    const Real ah   = w0(m, I_AH, k, j, i);
    const Real ma   = w0(m, I_MA, k, j, i);

    const Real g11c = adm.g_dd(m,0,0,k,j,i), g12c = adm.g_dd(m,0,1,k,j,i);
    const Real g13c = adm.g_dd(m,0,2,k,j,i), g22c = adm.g_dd(m,1,1,k,j,i);
    const Real g23c = adm.g_dd(m,1,2,k,j,i), g33c = adm.g_dd(m,2,2,k,j,i);
    const Real isdetg_c = 1.0 / Kokkos::sqrt(adm::SpatialDet(g11c,g12c,g13c,g22c,g23c,g33c));

    Real theta = (SqrtGammaVcoord(adm, w0, m, k, j, i+1, 0)
                - SqrtGammaVcoord(adm, w0, m, k, j, i-1, 0)) * 0.5 / size.d_view(m).dx1;
    if (multi_d) {
      theta += (SqrtGammaVcoord(adm, w0, m, k, j+1, i, 1)
              - SqrtGammaVcoord(adm, w0, m, k, j-1, i, 1)) * 0.5 / size.d_view(m).dx2;
    }
    if (three_d) {
      theta += (SqrtGammaVcoord(adm, w0, m, k+1, j, i, 2)
              - SqrtGammaVcoord(adm, w0, m, k-1, j, i, 2)) * 0.5 / size.d_view(m).dx3;
    }
    theta *= isdetg_c;
    const Real drho = -theta / time_cgs;

    // Evaluate the RHINE network.
    Real dye, dyn, dyp, dya, dyh, dah, dma, fnu;
    nets_.run(rho, tmev, ye, yn, ya, yh, ah, drho, dt_cgs,
              ye, yn, ya, yh, ah, ma,
              dye, dyn, dyp, dya, dyh, dah, dma, fnu, pmode_);

    // Apply the source terms.
    const Real dcons = u0(m, IDN, k, j, i);
    u0(m, I_YE, k, j, i) += dcons * (dye * dt_cgs);
    u0(m, I_YN, k, j, i) += dcons * (dyn * dt_cgs);
    u0(m, I_YA, k, j, i) += dcons * (dya * dt_cgs);
    u0(m, I_YH, k, j, i) += dcons * (dyh * dt_cgs);
    u0(m, I_AH, k, j, i) += dcons * (dah * dt_cgs);
    u0(m, I_MA, k, j, i) += dcons * (dma * dt_cgs);

    const Real de_cgs  = -(dma * dt_cgs) * (1.0 - fnu) * MEV_TO_ERG / M_BARYON_G;
    const Real de_code = de_cgs * (mass_cgs / energy_cgs);
    u0(m, IEN, k, j, i) += dcons * de_code;
  });

  return TaskStatus::complete;
}

}  // namespace rhine
