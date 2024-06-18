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

//----------------------------------------------------------------------------------------
// constructor, parses input file and initializes data structures and parameters
// Only source terms specified in input file are initialized.

SourceTerms::SourceTerms(std::string block, MeshBlockPack *pp, ParameterInput *pin) :
  pmy_pack(pp),
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
}

//----------------------------------------------------------------------------------------
// destructor

SourceTerms::~SourceTerms() {
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

  par_for("cooling", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
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
