//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file coordinates.cpp
//! \brief
#include <iostream> // cout
#include <string>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "cartesian_ks.hpp"
#include "coordinates.hpp"
#include "cell_locations.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"

//----------------------------------------------------------------------------------------
// constructor, initializes coordinates data

Coordinates::Coordinates(ParameterInput *pin, MeshBlockPack *ppack) :
    pmy_pack(ppack),
    excision_floor("excision_floor",1,1,1,1),
    excision_flux("excision_flux",1,1,1,1) {
  // Check for relativistic dynamics
  // WGC: idea for handling new EOS
  is_dynamical_relativistic = (pin->DoesBlockExist("adm") || pin->DoesBlockExist("z4c"))
                         && (pin->DoesBlockExist("hydro") || pin->DoesBlockExist("mhd"));
  if(!is_dynamical_relativistic) {
    is_special_relativistic = pin->GetOrAddBoolean("coord","special_rel",false);
    is_general_relativistic = pin->GetOrAddBoolean("coord","general_rel",false);
  } else {
    is_special_relativistic = is_general_relativistic = false;
  }
  if (is_special_relativistic && is_general_relativistic) {
    std::cout << "### FATAL ERROR in "<< __FILE__ <<" at line " << __LINE__ << std::endl
              << "Cannot specify both SR and GR at same time" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // Read properties of metric and excision from input file for GR.
  if (is_general_relativistic || is_dynamical_relativistic) {
    coord_data.is_minkowski = pin->GetOrAddBoolean("coord","minkowski",false);
    if (!(coord_data.is_minkowski)) {
      coord_data.bh_spin = pin->GetReal("coord","a");
      coord_data.bh_excise = pin->GetOrAddBoolean("coord","excise",true);
    } else {
      coord_data.bh_spin = 0.0;
      coord_data.bh_excise = false;
    }

    if (coord_data.bh_excise) {
      // Set the density and pressure to which cells inside the excision radius will
      // be reset to.  Primitive velocities will be set to zero.
      coord_data.dexcise = pin->GetReal("coord","dexcise");
      coord_data.pexcise = pin->GetReal("coord","pexcise");
      coord_data.flux_excise_r = (pin->DoesBlockExist("radiation")) ?
        1.0+sqrt(1.0-SQR(coord_data.bh_spin)) :
        pin->GetOrAddReal("coord","flux_excise_r",1.0);
      coord_data.rexcise =
        (pin->DoesBlockExist("radiation")) ? 1.0+sqrt(1.0-SQR(coord_data.bh_spin)) : 1.0;

      coord_data.excision_scheme = ExcisionScheme::fixed;
      if (is_dynamical_relativistic) {
        std::string emethod = pin->GetOrAddString("coord","excision_scheme","fixed");
        if (emethod.compare("fixed") == 0) {
          coord_data.excision_scheme = ExcisionScheme::fixed;
        } else if (emethod.compare("lapse") == 0) {
          coord_data.excision_scheme = ExcisionScheme::lapse;
          coord_data.excise_lapse = pin->GetOrAddReal("coord","excise_lapse", 0.25);
        } else {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line "
                    << __LINE__ << std::endl
                    << "Unknown excision method: " << emethod << std::endl;
          std::exit(EXIT_FAILURE);
        }
      }

      // boolean masks allocation
      int nmb = ppack->nmb_thispack;
      auto &indcs = pmy_pack->pmesh->mb_indcs;
      int ncells1 = indcs.nx1 + 2*(indcs.ng);
      int ncells2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
      int ncells3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;
      Kokkos::realloc(excision_floor, nmb, ncells3, ncells2, ncells1);
      Kokkos::realloc(excision_flux, nmb, ncells3, ncells2, ncells1);
      if (coord_data.excision_scheme == ExcisionScheme::fixed) {
        SetExcisionMasks(excision_floor, excision_flux);
      }
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn
// Coordinate (geometric) source term function for GR hydrodynamics

void Coordinates::CoordSrcTerms(const DvceArray5D<Real> &prim, const EOS_Data &eos,
                                const Real dt, DvceArray5D<Real> &cons) {
  // capture variables for kernel
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is; int ie = indcs.ie;
  int js = indcs.js; int je = indcs.je;
  int ks = indcs.ks; int ke = indcs.ke;
  auto &size = pmy_pack->pmb->mb_size;
  auto &flat = coord_data.is_minkowski;
  auto &spin = coord_data.bh_spin;

  Real gamma_prime = eos.gamma / (eos.gamma - 1.0);

  int nmb1 = pmy_pack->nmb_thispack - 1;
  par_for("coord_src", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    // Extract components of metric
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
    ComputeMetricAndInverse(x1v, x2v, x3v, flat, spin, glower, gupper);

    // Extract primitives
    const Real &rho  = prim(m,IDN,k,j,i);
    const Real &uu1  = prim(m,IVX,k,j,i);
    const Real &uu2  = prim(m,IVY,k,j,i);
    const Real &uu3  = prim(m,IVZ,k,j,i);
    Real pgas = eos.IdealGasPressure(prim(m,IEN,k,j,i));

    // Calculate 4-velocity (exploiting symmetry of metric)
    Real uu_sq = glower[1][1]*uu1*uu1 +2.0*glower[1][2]*uu1*uu2 +2.0*glower[1][3]*uu1*uu3
               + glower[2][2]*uu2*uu2 +2.0*glower[2][3]*uu2*uu3
               + glower[3][3]*uu3*uu3;
    Real alpha = sqrt(-1.0/gupper[0][0]);
    Real gamma = sqrt(1.0 + uu_sq);
    Real u0 = gamma / alpha;
    Real u1 = uu1 - alpha * gamma * gupper[0][1];
    Real u2 = uu2 - alpha * gamma * gupper[0][2];
    Real u3 = uu3 - alpha * gamma * gupper[0][3];

    // Calculate stress-energy tensor
    Real wtot = rho + gamma_prime * pgas;
    Real ptot = pgas;
    Real tt[4][4];
    tt[0][0] = wtot * u0 * u0 + ptot * gupper[0][0];
    tt[0][1] = wtot * u0 * u1 + ptot * gupper[0][1];
    tt[0][2] = wtot * u0 * u2 + ptot * gupper[0][2];
    tt[0][3] = wtot * u0 * u3 + ptot * gupper[0][3];
    tt[1][1] = wtot * u1 * u1 + ptot * gupper[1][1];
    tt[1][2] = wtot * u1 * u2 + ptot * gupper[1][2];
    tt[1][3] = wtot * u1 * u3 + ptot * gupper[1][3];
    tt[2][2] = wtot * u2 * u2 + ptot * gupper[2][2];
    tt[2][3] = wtot * u2 * u3 + ptot * gupper[2][3];
    tt[3][3] = wtot * u3 * u3 + ptot * gupper[3][3];

    // compute derivates of metric.
    Real dg_dx1[4][4], dg_dx2[4][4], dg_dx3[4][4];
    ComputeMetricDerivatives(x1v, x2v, x3v, flat, spin, dg_dx1, dg_dx2, dg_dx3);

    // Calculate source terms, exploiting symmetries
    Real s_1 = 0.0, s_2 = 0.0, s_3 = 0.0;
    s_1 += 0.5*dg_dx1[0][0] * tt[0][0];
    s_1 +=     dg_dx1[0][1] * tt[0][1];
    s_1 +=     dg_dx1[0][2] * tt[0][2];
    s_1 +=     dg_dx1[0][3] * tt[0][3];
    s_1 += 0.5*dg_dx1[1][1] * tt[1][1];
    s_1 +=     dg_dx1[1][2] * tt[1][2];
    s_1 +=     dg_dx1[1][3] * tt[1][3];
    s_1 += 0.5*dg_dx1[2][2] * tt[2][2];
    s_1 +=     dg_dx1[2][3] * tt[2][3];
    s_1 += 0.5*dg_dx1[3][3] * tt[3][3];

    s_2 += 0.5*dg_dx2[0][0] * tt[0][0];
    s_2 +=     dg_dx2[0][1] * tt[0][1];
    s_2 +=     dg_dx2[0][2] * tt[0][2];
    s_2 +=     dg_dx2[0][3] * tt[0][3];
    s_2 += 0.5*dg_dx2[1][1] * tt[1][1];
    s_2 +=     dg_dx2[1][2] * tt[1][2];
    s_2 +=     dg_dx2[1][3] * tt[1][3];
    s_2 += 0.5*dg_dx2[2][2] * tt[2][2];
    s_2 +=     dg_dx2[2][3] * tt[2][3];
    s_2 += 0.5*dg_dx2[3][3] * tt[3][3];

    s_3 += 0.5*dg_dx3[0][0] * tt[0][0];
    s_3 +=     dg_dx3[0][1] * tt[0][1];
    s_3 +=     dg_dx3[0][2] * tt[0][2];
    s_3 +=     dg_dx3[0][3] * tt[0][3];
    s_3 += 0.5*dg_dx3[1][1] * tt[1][1];
    s_3 +=     dg_dx3[1][2] * tt[1][2];
    s_3 +=     dg_dx3[1][3] * tt[1][3];
    s_3 += 0.5*dg_dx3[2][2] * tt[2][2];
    s_3 +=     dg_dx3[2][3] * tt[2][3];
    s_3 += 0.5*dg_dx3[3][3] * tt[3][3];

    // Add source terms to conserved quantities
    cons(m,IM1,k,j,i) += dt * s_1;
    cons(m,IM2,k,j,i) += dt * s_2;
    cons(m,IM3,k,j,i) += dt * s_3;
  });

  return;
}

//----------------------------------------------------------------------------------------
//! \fn
// Coordinate (geometric) source term function for GR MHD
//
// TODO(@user): Most of this function just copies the Hydro version.  Only difference is
// the inclusion of the magnetic field in computing the stress-energy tensor.  There must
// be a smarter way to generalize these two functions and avoid duplicated code.
// Functions distinguished only by argument list.

void Coordinates::CoordSrcTerms(const DvceArray5D<Real> &prim,
                                const DvceArray5D<Real> &bcc, const EOS_Data &eos,
                                const Real dt, DvceArray5D<Real> &cons) {
  // capture variables for kernel
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is; int ie = indcs.ie;
  int js = indcs.js; int je = indcs.je;
  int ks = indcs.ks; int ke = indcs.ke;
  auto &size = pmy_pack->pmb->mb_size;
  auto &flat = coord_data.is_minkowski;
  auto &spin = coord_data.bh_spin;

  Real gamma_prime = eos.gamma / (eos.gamma - 1.0);

  int nmb1 = pmy_pack->nmb_thispack - 1;
  par_for("coord_src", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    // Extract components of metric
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
    ComputeMetricAndInverse(x1v, x2v, x3v, flat, spin, glower, gupper);

    // Extract primitives
    const Real &rho  = prim(m,IDN,k,j,i);
    const Real &uu1  = prim(m,IVX,k,j,i);
    const Real &uu2  = prim(m,IVY,k,j,i);
    const Real &uu3  = prim(m,IVZ,k,j,i);
    Real pgas = eos.IdealGasPressure(prim(m,IEN,k,j,i));

    // Calculate 4-velocity
    Real uu_sq = glower[1][1]*uu1*uu1 +2.0*glower[1][2]*uu1*uu2 +2.0*glower[1][3]*uu1*uu3
               + glower[2][2]*uu2*uu2 +2.0*glower[2][3]*uu2*uu3
               + glower[3][3]*uu3*uu3;
    Real alpha = sqrt(-1.0/gupper[0][0]);
    Real gamma = sqrt(1.0 + uu_sq);
    Real u0 = gamma / alpha;
    Real u1 = uu1 - alpha * gamma * gupper[0][1];
    Real u2 = uu2 - alpha * gamma * gupper[0][2];
    Real u3 = uu3 - alpha * gamma * gupper[0][3];

    // lower vector indices
    Real u_1 = glower[1][0]*u0 + glower[1][1]*u1 + glower[1][2]*u2 + glower[1][3]*u3;
    Real u_2 = glower[2][0]*u0 + glower[2][1]*u1 + glower[2][2]*u2 + glower[2][3]*u3;
    Real u_3 = glower[3][0]*u0 + glower[3][1]*u1 + glower[3][2]*u2 + glower[3][3]*u3;

    // calculate 4-magnetic field
    const Real &bb1 = bcc(m,IBX,k,j,i);
    const Real &bb2 = bcc(m,IBY,k,j,i);
    const Real &bb3 = bcc(m,IBZ,k,j,i);
    Real b0 = u_1*bb1 + u_2*bb2 + u_3*bb3;
    Real b1 = (bb1 + b0 * u1) / u0;
    Real b2 = (bb2 + b0 * u2) / u0;
    Real b3 = (bb3 + b0 * u3) / u0;

    // lower vector indices
    Real b_0 = glower[0][0]*b0 + glower[0][1]*b1 + glower[0][2]*b2 + glower[0][3]*b3;
    Real b_1 = glower[1][0]*b0 + glower[1][1]*b1 + glower[1][2]*b2 + glower[1][3]*b3;
    Real b_2 = glower[2][0]*b0 + glower[2][1]*b1 + glower[2][2]*b2 + glower[2][3]*b3;
    Real b_3 = glower[3][0]*b0 + glower[3][1]*b1 + glower[3][2]*b2 + glower[3][3]*b3;
    Real b_sq = b_0*b0 + b_1*b1 + b_2*b2 + b_3*b3;

    // Calculate stress-energy tensor
    Real wtot = rho + gamma_prime * pgas + b_sq;
    Real ptot = pgas + 0.5*b_sq;
    Real tt[4][4];
    tt[0][0] = wtot * u0 * u0 + ptot * gupper[0][0] - b0 * b0;
    tt[0][1] = wtot * u0 * u1 + ptot * gupper[0][1] - b0 * b1;
    tt[0][2] = wtot * u0 * u2 + ptot * gupper[0][2] - b0 * b2;
    tt[0][3] = wtot * u0 * u3 + ptot * gupper[0][3] - b0 * b3;
    tt[1][1] = wtot * u1 * u1 + ptot * gupper[1][1] - b1 * b1;
    tt[1][2] = wtot * u1 * u2 + ptot * gupper[1][2] - b1 * b2;
    tt[1][3] = wtot * u1 * u3 + ptot * gupper[1][3] - b1 * b3;
    tt[2][2] = wtot * u2 * u2 + ptot * gupper[2][2] - b2 * b2;
    tt[2][3] = wtot * u2 * u3 + ptot * gupper[2][3] - b2 * b3;
    tt[3][3] = wtot * u3 * u3 + ptot * gupper[3][3] - b3 * b3;

    // compute derivates of metric.
    Real dg_dx1[4][4], dg_dx2[4][4], dg_dx3[4][4];
    ComputeMetricDerivatives(x1v, x2v, x3v, flat, spin, dg_dx1, dg_dx2, dg_dx3);

    // Calculate source terms
    Real s_1 = 0.0, s_2 = 0.0, s_3 = 0.0;
    s_1 += 0.5*dg_dx1[0][0] * tt[0][0];
    s_1 +=     dg_dx1[0][1] * tt[0][1];
    s_1 +=     dg_dx1[0][2] * tt[0][2];
    s_1 +=     dg_dx1[0][3] * tt[0][3];
    s_1 += 0.5*dg_dx1[1][1] * tt[1][1];
    s_1 +=     dg_dx1[1][2] * tt[1][2];
    s_1 +=     dg_dx1[1][3] * tt[1][3];
    s_1 += 0.5*dg_dx1[2][2] * tt[2][2];
    s_1 +=     dg_dx1[2][3] * tt[2][3];
    s_1 += 0.5*dg_dx1[3][3] * tt[3][3];

    s_2 += 0.5*dg_dx2[0][0] * tt[0][0];
    s_2 +=     dg_dx2[0][1] * tt[0][1];
    s_2 +=     dg_dx2[0][2] * tt[0][2];
    s_2 +=     dg_dx2[0][3] * tt[0][3];
    s_2 += 0.5*dg_dx2[1][1] * tt[1][1];
    s_2 +=     dg_dx2[1][2] * tt[1][2];
    s_2 +=     dg_dx2[1][3] * tt[1][3];
    s_2 += 0.5*dg_dx2[2][2] * tt[2][2];
    s_2 +=     dg_dx2[2][3] * tt[2][3];
    s_2 += 0.5*dg_dx2[3][3] * tt[3][3];

    s_3 += 0.5*dg_dx3[0][0] * tt[0][0];
    s_3 +=     dg_dx3[0][1] * tt[0][1];
    s_3 +=     dg_dx3[0][2] * tt[0][2];
    s_3 +=     dg_dx3[0][3] * tt[0][3];
    s_3 += 0.5*dg_dx3[1][1] * tt[1][1];
    s_3 +=     dg_dx3[1][2] * tt[1][2];
    s_3 +=     dg_dx3[1][3] * tt[1][3];
    s_3 += 0.5*dg_dx3[2][2] * tt[2][2];
    s_3 +=     dg_dx3[2][3] * tt[2][3];
    s_3 += 0.5*dg_dx3[3][3] * tt[3][3];

    // Add source terms to conserved quantities
    cons(m,IM1,k,j,i) += dt * s_1;
    cons(m,IM2,k,j,i) += dt * s_2;
    cons(m,IM3,k,j,i) += dt * s_3;
  });

  return;
}
