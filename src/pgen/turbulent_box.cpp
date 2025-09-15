//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file mass_removal_test.cpp
//  \brief Problem generator for a turbulent box
#include <iostream> // cout

#include "athena.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "pgen.hpp"

void UserSource(Mesh* pm, const Real bdt);

//----------------------------------------------------------------------------------------
//  \brief Problem Generator for mass removal

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  auto &indcs = pmy_mesh_->mb_indcs;
  auto &size = pmbp->pmb->mb_size;

  int nscalars = pmbp->phydro->nscalars;
  int nhydro = pmbp->phydro->nhydro;

  // Enroll user functions 
  user_srcs_func = UserSource;

  if (restart) return;

  // Capture variables for kernel
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;

  // Initialize Hydro variables -------------------------------
  auto &u0 = pmbp->phydro->u0;
  EOS_Data &eos = pmbp->phydro->peos->eos_data;
  Real gm1 = eos.gamma - 1.0;

  Real den = 1.0;
  Real temp = 1.0;

  // Set initial conditions
  par_for("pgen_turb", DevExeSpace(),0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
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

    Real R2 = x1v*x1v + x2v*x2v + x3v*x3v;

    u0(m,IDN,k,j,i) = den;
    u0(m,IM1,k,j,i) = 0.0;
    u0(m,IM2,k,j,i) = 0.0;
    u0(m,IM3,k,j,i) = 0.0;
    u0(m,IEN,k,j,i) = den * temp / gm1 +
       0.5*(SQR(u0(m,IM1,k,j,i)) + SQR(u0(m,IM2,k,j,i)) +
       SQR(u0(m,IM3,k,j,i)))/u0(m,IDN,k,j,i);
    
    Real dye_conc = 0.0;
    if (R2 < 0.1) { // RHS should be r^2
      dye_conc = 1.0;
    }
    u0(m, nhydro  , k, j, i) = dye_conc * den; // first scalar
    u0(m, nhydro+1, k, j, i) = dye_conc * den; // second scalar
    u0(m, nhydro+2, k, j, i) = dye_conc * den; // third scalar
    u0(m, nhydro+3, k, j, i) = dye_conc * den; // fourth scalar
  });

  return;
}

KOKKOS_INLINE_FUNCTION
// Advance x in [0,1] using a triangular heating profile.
Real heatStep(Real x, Real amplitude, Real dt)
{
  // Clamp input to [0,1] just in case
  x = Kokkos::clamp(x, 0.0, 1.0);

  // Piecewise-linear heating rate
  Real rate = amplitude * -1.0 * sin(x*2*Kokkos::numbers::pi);

  // Update and clamp to 1.0
  Real x_new = x + rate * dt;
  if (x_new > 1.0) x_new = 1.0;

  return x_new;
}

void UserSource(Mesh* pm, const Real bdt) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  auto &indcs = pm->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmbp->nmb_thispack - 1;
  auto &u0 = pmbp->phydro->u0;
  auto &w0 = pmbp->phydro->w0;
  int nscalars = pmbp->phydro->nscalars;
  int nhydro = pmbp->phydro->nhydro;

  par_for("user_source", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real density = w0(m, IDN, k, j, i);
    Real scalar_conc_1 = w0(m, nhydro  , k, j, i);
    Real scalar_conc_2 = w0(m, nhydro+1, k, j, i);
    Real scalar_conc_3 = w0(m, nhydro+2, k, j, i);
    Real scalar_conc_4 = w0(m, nhydro+3, k, j, i);

    // Apply heating to first scalar
    Real new_scalar_conc_1 = heatStep(scalar_conc_1, 0.0, bdt);
    u0(m, nhydro  , k, j, i) += (new_scalar_conc_1 - scalar_conc_1) * density;

    // Apply heating to second scalar
    Real new_scalar_conc_2 = heatStep(scalar_conc_2, 0.1, bdt);
    u0(m, nhydro+1, k, j, i) += (new_scalar_conc_2 - scalar_conc_2) * density;

    // Apply heating to third scalar
    Real new_scalar_conc_3 = heatStep(scalar_conc_3, 1.0, bdt);
    u0(m, nhydro+2, k, j, i) += (new_scalar_conc_3 - scalar_conc_3) * density;

    // Apply heating to second scalar
    Real new_scalar_conc_4 = heatStep(scalar_conc_4, 10.0, bdt);
    u0(m, nhydro+3, k, j, i) += (new_scalar_conc_4 - scalar_conc_4) * density;

    // We should check that scalars are guaranteed to be in [0,1] after all source terms are added.
    u0(m, nhydro  , k, j, i) = Kokkos::clamp(u0(m, nhydro  , k, j, i), 0.0, u0(m,IDN,k,j,i));
    u0(m, nhydro+1, k, j, i) = Kokkos::clamp(u0(m, nhydro+1, k, j, i), 0.0, u0(m,IDN,k,j,i));
    u0(m, nhydro+2, k, j, i) = Kokkos::clamp(u0(m, nhydro+2, k, j, i), 0.0, u0(m,IDN,k,j,i));
    u0(m, nhydro+3, k, j, i) = Kokkos::clamp(u0(m, nhydro+3, k, j, i), 0.0, u0(m,IDN,k,j,i));
  });

  return;
}
