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
#include "coordinates/cell_locations.hpp"

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem_()
//! \brief Problem Generator for spherical blast problem

void ProblemGenerator::UserProblem(MeshBlockPack *pmbp, ParameterInput *pin) {
  Real rout = pin->GetReal("problem", "radius");
  Real rin  = rout - pin->GetOrAddReal("problem", "ramp", 0.0);
  Real pa   = pin->GetOrAddReal("problem", "pamb", 1.0);
  Real da   = pin->GetOrAddReal("problem", "damb", 1.0);
  Real prat = pin->GetReal("problem", "prat");
  Real drat = pin->GetOrAddReal("problem", "drat", 1.0);

  // capture variables for the kernel
  auto &indcs = pmbp->pmesh->mb_indcs;
  auto &size = pmbp->pmb->mb_size;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;

  // setup uniform ambient medium with spherical over-pressured region in Hydro
  if (pmbp->phydro != nullptr) {
    auto &eos = pmbp->phydro->peos->eos_data;
    auto &w0 = pmbp->phydro->w0;
    par_for("pgen_cloud1",DevExeSpace(),0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
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

      Real den = da;
      Real pres = pa;
      if (rad < rout) {
        if (rad < rin) {
          den = drat*da;
          pres = prat*pa;
        } else {   // add smooth ramp in density
          Real f = (rad-rin) / (rout-rin);
          Real log_den = (1.0-f) * log(drat*da) + f * log(da);
          den = exp(log_den);
          Real log_pres = (1.0-f) * log(prat*pa) + f * log(pa);
          pres = exp(log_pres);
        }
      }

      // set either internal energy density or temparature as primitive
      Real prim,
      if (eos.use_e) {
        prim = pres/(eos.gamma - 1.0);
      } else {
        prim = pres/((eos.gamma - 1.0)*den);
      }

      w0(m,IDN,k,j,i) = den;
      w0(m,IVX,k,j,i) = 0.0;
      w0(m,IVY,k,j,i) = 0.0;
      w0(m,IVZ,k,j,i) = 0.0;
      w0(m,IEN,k,j,i) = prim;
    });

    // Convert primitives to conserved
    auto &u0 = pmbp->phydro->u0;
    pmbp->phydro->peos->PrimToCons(w0, u0);
  }

  return;
}
