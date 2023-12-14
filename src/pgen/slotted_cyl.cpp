//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file slotted_cyl.cpp
//  \brief Slotted cylinder passive scalar advection problem generator for 2D/3D problems.

// C++ headers
#include <algorithm>  // min, max
#include <iostream>   // endl
#include <sstream>    // stringstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()

// Athena++ headers
#include "athena.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "pgen.hpp"

// Parameters which define initial solution -- made global so that they can be shared
// Anonymous namespace used to prevent name collisions outside of this file
namespace {

constexpr int N_gl = 12;
constexpr Real d0 = 1.0;
constexpr bool use_gl_quadrature = true;
Real radius, omega_x1, omega_x2, omega;
Real s_width, s_height, center_x1, center_x2;

Real SlottedCylinderProfile(Real x1, Real x2) {
  // positions relative to the center of the cylinder
  Real zx = x1 - center_x1;
  Real zy = x2 - center_x2;
  // distance from center of cylinder
  Real r = std::sqrt(SQR(zx) + SQR(zy));
  Real scalar = 0.0;

  // Initial condition is specified in pointwise fashion as follows:
  // cell-center is outside the cylinder
  if (r > radius)
    scalar = 0.0;
  // cell-center is inside the slot
  else if ((std::abs(2*zx) < s_width) && (zy + radius < s_height) && (0 < zy + radius))
    scalar = 0.0;
  // cell-center is inside the cylinder and outside the slot
  else
    scalar = 1.0;

  return scalar;
}

} // namespace

//----------------------------------------------------------------------------------------
//! \fn void MeshBlock::UserProblem(ParameterInput *pin)
//  \brief Sets initial conditions for linear wave tests

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  if (restart) return;

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->phydro == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Slotted cylinder test can only be run in Hydro, but no <hydro> block "
              << "in input file" << std::endl;
    exit(EXIT_FAILURE);
  }

  // cylinder dimensions
  radius = pin->GetOrAddReal("problem", "radius", 0.15);
  center_x1 = pin->GetOrAddReal("problem", "center_x1", 0.50);
  center_x2 = pin->GetOrAddReal("problem", "center_x2", 0.75);
  // rotational speed and axis
  omega = pin->GetOrAddReal("problem", "omega", 1.0);
  omega_x1 = pin->GetOrAddReal("problem", "omega_x1", 0.50);
  omega_x2 = pin->GetOrAddReal("problem", "omega_x2", 0.50);
  // slot dimensions
  s_width = pin->GetOrAddReal("problem", "s_width", 0.05);
  s_height = pin->GetOrAddReal("problem", "s_height", 0.25);

  // capture variables for kernel
  auto &indcs = pmy_mesh_->mb_indcs;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  int &nscalars = pmbp->phydro->nscalars;
  int &nhydro = pmbp->phydro->nhydro;
  auto &size = pmbp->pmb->mb_size;
  auto &u0 = pmbp->phydro->u0;

  // Set initial conditions
  par_for("pgen_slot_cyl", DevExeSpace(),0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

    // background fluid:
    u0(m,IDN,k,j,i) = d0;
    u0(m,IM1,k,j,i) = -d0*2.0*M_PI*omega*(x2v - omega_x2)*u0(m,IDN,k,j,i);
    u0(m,IM2,k,j,i) = d0*2.0*M_PI*omega*(x1v - omega_x1)*u0(m,IDN,k,j,i);
    u0(m,IM3,k,j,i) = 0.0;

    // Use standard midpoint approximation with cell centered coords:
    Real cell_ave = SlottedCylinderProfile(x1v, x2v);

    // uniformly fill all scalars to have equal concentration
    for (int n=nhydro; n<(nhydro+nscalars); ++n) {
      u0(m,n,k,j,i) = cell_ave*d0;
    }
  });

  return;
}

