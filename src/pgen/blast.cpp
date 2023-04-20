//========================================================================================
// Athena++ astrophysical MHD code, Kokkos version
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file blast.cpp
//! \brief Problem generator for spherical blast wave problem.
//!
//! REFERENCE: P. Londrillo & L. Del Zanna, "High-order upwind schemes for
//!   multidimensional MHD", ApJ, 530, 508 (2000), and references therein.

#include <algorithm>
#include <cmath>
#include <sstream>

#include "parameter_input.hpp"
#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "coordinates/cell_locations.hpp"

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem_()
//! \brief Problem Generator for spherical blast problem

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  if (restart) return;

  Real rout = pin->GetReal("problem", "outer_radius");
  Real rin  = rout - pin->GetReal("problem", "inner_radius");
  // values for neutrals (hydro fluid)
  Real pn_amb   = pin->GetOrAddReal("problem", "pn_amb", 1.0);
  Real dn_amb   = pin->GetOrAddReal("problem", "dn_amb", 1.0);
  // values for ions (hydro fluid)
  Real pi_amb   = pin->GetOrAddReal("problem", "pi_amb", 1.0);
  Real di_amb   = pin->GetOrAddReal("problem", "di_amb", 1.0);
  // ratios in blast (same for both ions and neutrals)
  Real prat = pin->GetReal("problem", "prat");
  Real drat = pin->GetOrAddReal("problem", "drat", 1.0);
  Real b_amb = pin->GetOrAddReal("problem", "b_amb", 0.1);

  // capture variables for the kernel
  auto &indcs = pmy_mesh_->mb_indcs;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  auto &size = pmbp->pmb->mb_size;

  // Select either Hydro or MHD
  hydro::Hydro *phydro = pmbp->phydro;
  mhd::MHD *pmhd = pmbp->pmhd;

  par_for("pgen_blast1",DevExeSpace(),0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(int m,int k,int j,int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    int nx1 = indcs.nx1;
    Real x1v = CellCenterX(i-is, nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    int nx2 = indcs.nx2;
    Real x2v = CellCenterX(j-js, nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    int nx3 = indcs.nx3;
    Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);

    Real rad = std::sqrt(SQR(x1v) + SQR(x2v) + SQR(x3v));

    // initialize Hydro, MHD, or both.
    if (phydro != nullptr) {
      Real den = dn_amb;
      Real pres = pn_amb;
      if (rad < rout) {
        if (rad < rin) {
          den *= drat;
          pres *= prat;
        } else {   // add smooth ramp in density
          Real f = (rad-rin) / (rout-rin);
          Real log_den = (1.0-f) * log(drat*dn_amb) + f * log(dn_amb);
          den = exp(log_den);
          Real log_pres = (1.0-f) * log(prat*pn_amb) + f * log(pn_amb);
          pres = exp(log_pres);
        }
      }
      phydro->w0(m,IDN,k,j,i) = den;
      phydro->w0(m,IVX,k,j,i) = 0.0;
      phydro->w0(m,IVY,k,j,i) = 0.0;
      phydro->w0(m,IVZ,k,j,i) = 0.0;
      phydro->w0(m,IEN,k,j,i) = pres/(phydro->peos->eos_data.gamma - 1.0);
    }
    if (pmhd != nullptr) {
      Real den = di_amb;
      Real pres = pi_amb;
      if (rad < rout) {
        if (rad < rin) {
          den *= drat;
          pres *= prat;
        } else {   // add smooth ramp in density
          Real f = (rad-rin) / (rout-rin);
          Real log_den = (1.0-f) * log(drat*di_amb) + f * log(di_amb);
          den = exp(log_den);
          Real log_pres = (1.0-f) * log(prat*pi_amb) + f * log(pi_amb);
          pres = exp(log_pres);
        }
      }
      pmhd->w0(m,IDN,k,j,i) = den;
      pmhd->w0(m,IVX,k,j,i) = 0.0;
      pmhd->w0(m,IVY,k,j,i) = 0.0;
      pmhd->w0(m,IVZ,k,j,i) = 0.0;
      pmhd->w0(m,IEN,k,j,i) = pres/(pmhd->peos->eos_data.gamma - 1.0);
    }
  });

  // initialize magnetic fields ---------------------------------------

  if (pmhd != nullptr) {
    auto &b0 = pmhd->b0;
    par_for("pgen_blast2",DevExeSpace(),0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      b0.x1f(m,k,j,i) = b_amb;
      b0.x2f(m,k,j,i) = 0.0;
      b0.x3f(m,k,j,i) = 0.0;

      // Include extra face-component at edge of block in each direction
      if (i==ie) {b0.x1f(m,k,j,i+1) = b_amb;}
      if (j==je) {b0.x2f(m,k,j+1,i) = 0.0;}
      if (k==ke) {b0.x3f(m,k+1,j,i) = 0.0;}
    });

    // Compute cell-centered fields
    auto &bcc_ = pmhd->bcc0;
    par_for("pgen_blast3",DevExeSpace(),0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      // cell-centered fields are simple linear average of face-centered fields
      Real& w_bx = bcc_(m,IBX,k,j,i);
      Real& w_by = bcc_(m,IBY,k,j,i);
      Real& w_bz = bcc_(m,IBZ,k,j,i);
      w_bx = 0.5*(b0.x1f(m,k,j,i) + b0.x1f(m,k,j,i+1));
      w_by = 0.5*(b0.x2f(m,k,j,i) + b0.x2f(m,k,j+1,i));
      w_bz = 0.5*(b0.x3f(m,k,j,i) + b0.x3f(m,k+1,j,i));
    });
  }

  // Convert primitives to conserved
  if (phydro != nullptr) {
    phydro->peos->PrimToCons(phydro->w0, phydro->u0, is, ie, js, je, ks, ke);
  }
  if (pmhd != nullptr) {
    pmhd->peos->PrimToCons(pmhd->w0, pmhd->bcc0, pmhd->u0, is, ie, js, je, ks, ke);
  }

  return;
}
