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
//#include "hydro/hydro.hpp"
#include "ismcooling.hpp"
#include "mesh/mesh.hpp"
//#include "mhd/mhd.hpp"
#include "parameter_input.hpp"
#include "radiation/radiation.hpp"
#include "radiation/radiation_tetrad.hpp"
#include "units/units.hpp"

//----------------------------------------------------------------------------------------
// constructor, parses input file and initializes data structures and parameters
// Only source terms specified in input file are initialized.
// Block name passed to constructor will be one of "hydro_srcterms", "mhd_srcterms",
// or "rad_srcterms"

SourceTerms::SourceTerms(std::string block, MeshBlockPack *pp, ParameterInput *pin) :
    pmy_pack(pp) {
  // Read flags for each source term implemented (default false)
  const_accel = pin->GetOrAddBoolean(block, "const_accel", false);
  ism_cooling = pin->GetOrAddBoolean(block, "ism_cooling", false);
  rel_cooling = pin->GetOrAddBoolean(block, "rel_cooling", false);
  rad_beam = pin->GetOrAddBoolean(block, "rad_beam", false);

  // (1) read data for (constant) gravitational acceleration
  if (const_accel) {
    const_accel_val = pin->GetReal(block, "const_accel_val");
    const_accel_dir = pin->GetInteger(block, "const_accel_dir");
    if (const_accel_dir < 1 || const_accel_dir > 3) {
      std::cout << "### FATAL ERROR in "<< __FILE__ <<" at line " << __LINE__ << std::endl
                << "const_accle_dir must be 1,2, or 3" << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }

  // (2) Optically thin ISM cooling
  if (ism_cooling) {
    hrate = pin->GetReal(block, "hrate");
  }

  // (3) optically thin relativistic cooling
  if (rel_cooling) {
    crate_rel = pin->GetReal(block, "crate_rel");
    cpower_rel = pin->GetOrAddReal(block, "cpower_rel", 1.);
  }

  // (4) radiation beam source (radiation)
  if (rad_beam) {
    dii_dt = pin->GetReal(block, "dii_dt");
    pos1 = pin->GetReal(block, "pos_1");
    pos2 = pin->GetReal(block, "pos_2");
    pos3 = pin->GetReal(block, "pos_3");
    dir1 = pin->GetReal(block, "dir_1");
    dir2 = pin->GetReal(block, "dir_2");
    dir3 = pin->GetReal(block, "dir_3");
    width = pin->GetReal(block, "width");
    spread = pin->GetReal(block, "spread");
  }
}

//----------------------------------------------------------------------------------------
// destructor

SourceTerms::~SourceTerms() {
}

//----------------------------------------------------------------------------------------
//! \fn SourceTerms::ApplySrcTerms
//! \brief Applies selected source terms to input arrays. Two different versions are
//! implemented for fluid and radiation fields, distinguished by their argument lists

void SourceTerms::ApplySrcTerms(const DvceArray5D<Real> &w0, const EOS_Data &eos_data,
                                const Real bdt, DvceArray5D<Real> &u0) {
  // NOTE source terms must be computed using primitive (w0) and NOT conserved (u0) vars
  if (const_accel) ConstantAccel(w0, eos_data,  bdt, u0);
  if (ism_cooling) ISMCooling(w0, eos_data, bdt, u0);
  if (rel_cooling) RelCooling(w0, eos_data, bdt, u0);
  return;
}

void SourceTerms::ApplySrcTerms(DvceArray5D<Real> &i0, const Real bdt) {
  if (rad_beam) BeamSource(i0, bdt);
  return;
}

//----------------------------------------------------------------------------------------
//! \fn SourceTerms::ConstantAccel
//! \brief Add constant acceleration
//! NOTE source terms must be computed using primitive (w0) and NOT conserved (u0) vars

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
//! NOTE source terms must be computed using primitive (w0) and NOT conserved (u0) vars

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
//! NOTE source terms must be computed using primitive (w0) and NOT conserved (u0) vars

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
//! \brief Add beam of radiation at position (pos1,pos2,pos3) moving in direction
//! (dir1,dir2,dir3) with physical width and angular spread (width,spread)

void SourceTerms::BeamSource(DvceArray5D<Real> &i0, const Real bdt) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = (pmy_pack->nmb_thispack-1);
  int nang1 = (pmy_pack->prad->prgeo->nangles-1);

  auto &excise = pmy_pack->pcoord->coord_data.bh_excise;
  auto &rad_mask_ = pmy_pack->pcoord->excision_floor;

  auto &size = pmy_pack->pmb->mb_size;
  auto &flat = pmy_pack->pcoord->coord_data.is_minkowski;
  auto &spin = pmy_pack->pcoord->coord_data.bh_spin;

  Real &p1 = pos1, &p2 = pos2, &p3 = pos3;
  Real &d1 = dir1, &d2 = dir2, &d3 = dir3;
  Real &dii_dt_ = dii_dt;
  Real &width_ = width;
  Real &spread_ = spread;

  auto &tc = pmy_pack->prad->tetcov_c;
  auto &nh_c_ = pmy_pack->prad->nh_c;
  auto &tet_c_ = pmy_pack->prad->tet_c;
  par_for("rad_beam",DevExeSpace(),0,nmb1,ks,ke,js,je,is,ie,
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

    Real glower[4][4], gupper[4][4];
    ComputeMetricAndInverse(x1v,x2v,x3v,flat,spin,glower,gupper);
    Real dgx[4][4], dgy[4][4], dgz[4][4];
    ComputeMetricDerivatives(x1v,x2v,x3v,flat,spin,dgx,dgy,dgz);
    Real e[4][4], e_cov[4][4], omega[4][4][4];
    ComputeTetrad(x1v,x2v,x3v,flat,spin,glower,gupper,dgx,dgy,dgz,e,e_cov,omega);

    // Calculate proper distance to beam origin and minimum angle between directions
    Real dx1 = x1v - p1;
    Real dx2 = x2v - p2;
    Real dx3 = x3v - p3;
    Real dx_sq = glower[1][1]*dx1*dx1 +2.0*glower[1][2]*dx1*dx2 + 2.0*glower[1][3]*dx1*dx3
               + glower[2][2]*dx2*dx2 +2.0*glower[2][3]*dx2*dx3
               + glower[3][3]*dx3*dx3;
    Real mu_min = cos(spread_/2.0*M_PI/180.0);

    // Calculate contravariant time component of direction
    Real temp_a = glower[0][0];
    Real temp_b = 2.0*(glower[0][1]*d1 + glower[0][2]*d2 + glower[0][3]*d3);
    Real temp_c = glower[1][1]*d1*d1 + 2.0*glower[1][2]*d1*d2 + 2.0*glower[1][3]*d1*d3
                + glower[2][2]*d2*d2 + 2.0*glower[2][3]*d2*d3
                + glower[3][3]*d3*d3;
    Real d0 = ((-temp_b - sqrt(SQR(temp_b) - 4.0*temp_a*temp_c))/(2.0*temp_a));

    // lower indices
    Real dc0 = glower[0][0]*d0 + glower[0][1]*d1 + glower[0][2]*d2 + glower[0][3]*d3;
    Real dc1 = glower[0][1]*d0 + glower[1][1]*d1 + glower[1][2]*d2 + glower[1][3]*d3;
    Real dc2 = glower[0][2]*d0 + glower[1][2]*d1 + glower[2][2]*d2 + glower[2][3]*d3;
    Real dc3 = glower[0][3]*d0 + glower[1][3]*d1 + glower[2][3]*d2 + glower[3][3]*d3;

    // Calculate covariant direction in tetrad frame
    Real dtc0 = (tet_c_(m,0,0,k,j,i)*dc0 + tet_c_(m,0,1,k,j,i)*dc1 +
                 tet_c_(m,0,2,k,j,i)*dc2 + tet_c_(m,0,3,k,j,i)*dc3);
    Real dtc1 = (tet_c_(m,1,0,k,j,i)*dc0 + tet_c_(m,1,1,k,j,i)*dc1 +
                 tet_c_(m,1,2,k,j,i)*dc2 + tet_c_(m,1,3,k,j,i)*dc3)/(-dtc0);
    Real dtc2 = (tet_c_(m,2,0,k,j,i)*dc0 + tet_c_(m,2,1,k,j,i)*dc1 +
                 tet_c_(m,2,2,k,j,i)*dc2 + tet_c_(m,2,3,k,j,i)*dc3)/(-dtc0);
    Real dtc3 = (tet_c_(m,3,0,k,j,i)*dc0 + tet_c_(m,3,1,k,j,i)*dc1 +
                 tet_c_(m,3,2,k,j,i)*dc2 + tet_c_(m,3,3,k,j,i)*dc3)/(-dtc0);

    // Go through angles
    for (int n=0; n<=nang1; ++n) {
      Real mu = (nh_c_.d_view(n,1) * dtc1
               + nh_c_.d_view(n,2) * dtc2
               + nh_c_.d_view(n,3) * dtc3);
      if ((dx_sq < SQR(width_/2.0)) && (mu > mu_min)) {
        Real n0 = tet_c_(m,0,0,k,j,i);
        Real n_0 = tc(m,0,0,k,j,i)*nh_c_.d_view(n,0) + tc(m,1,0,k,j,i)*nh_c_.d_view(n,1)
                 + tc(m,2,0,k,j,i)*nh_c_.d_view(n,2) + tc(m,3,0,k,j,i)*nh_c_.d_view(n,3);
        i0(m,n,k,j,i) += n0*n_0*dii_dt_*bdt;
      }
    }
  });

  return;
}
