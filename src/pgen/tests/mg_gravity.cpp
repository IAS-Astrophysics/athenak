//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file mg_gravity.cpp
//  \brief Problem generator for Plummer potential self-gravity test

#include <cmath>
#include <iostream>
#include <sstream>
#include <iomanip>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "pgen/pgen.hpp"

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::SelfGravity()
//  \brief Problem Generator for Plummer potential self-gravity test
//  Plummer potential: phi(r) = -M / sqrt(r^2 + a^2)
//  where M = total mass, a = softening length, r = distance from center
//  Density: rho(r) = (3*M*a^2) / (4*pi*(r^2 + a^2)^(5/2))
//  Domain: periodic or can use Dirichlet BC (phi -> 0 at infinity)

void ProblemGenerator::SelfGravity(ParameterInput *pin, const bool restart) {
  // nothing needs to be done on restarts for this pgen
  if (restart) return;
  // Read input parameters
  Real four_pi_G = pin->GetOrAddReal("gravity", "four_pi_G", 1.0);
  Real gamma = pin->GetOrAddReal("hydro", "gamma", 5.0/3.0);

  // Plummer parameters
  Real M = pin->GetOrAddReal("problem", "plummer_mass", 1.0);           // total mass
  Real a = pin->GetOrAddReal("problem", "plummer_softening", 0.1);      // softening length
  Real center_x = pin->GetOrAddReal("problem", "center_x", 0.0);        // center location
  Real center_y = pin->GetOrAddReal("problem", "center_y", 0.0);
  Real center_z = pin->GetOrAddReal("problem", "center_z", 0.0);

  // background pressure (for hydro stability)
  Real p0 = pin->GetOrAddReal("problem", "pressure_bg", 1.0);
  Real ie_int = p0 / (gamma - 1.0);

  // capture variables for kernel
  auto &indcs = pmy_mesh_->mb_indcs;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  auto &size = pmbp->pmb->mb_size;

  // Initialize Hydro variables -------------------------------
  if (pmbp->phydro != nullptr) {
    int &nhydro = pmbp->phydro->nhydro;
    int &nscalars = pmbp->phydro->nscalars;
    auto &u0 = pmbp->phydro->u0;

    par_for("plummer_hydro_init", DevExeSpace(), 0, (pmbp->nmb_thispack-1), ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      // compute global coordinates
      Real &x1min = size.d_view(m).x1min;
      Real &x1max = size.d_view(m).x1max;
      Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

      Real &x2min = size.d_view(m).x2min;
      Real &x2max = size.d_view(m).x2max;
      Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

      Real &x3min = size.d_view(m).x3min;
      Real &x3max = size.d_view(m).x3max;
      Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);
      
      // Periodic test: Sinusoidal density wave
      Real rho = 2.0 + 0.1*std::sin(2*M_PI*x1v) * std::sin(2*M_PI*x2v) * std::sin(2*M_PI*x3v);

      // set hydro conserved variables: density, momenta, total energy
      u0(m, IDN, k, j, i) = rho;
      u0(m, IM1, k, j, i) = 1.0;
      u0(m, IM2, k, j, i) = 0.0;
      u0(m, IM3, k, j, i) = 0.0;
      u0(m, IEN, k, j, i) = ie_int;  // no kinetic term (static)
    });
  }  // End initialization of Hydro variables

  return;
}