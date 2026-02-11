//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file jeans_wave.cpp
//  \brief Problem generator for Jeans wave test with self-gravity
//
//  Tests the linear dispersion relation for sound+gravity waves in a uniform medium.
//  For wave vector k and sound speed cs, the dispersion relation is:
//    omega^2 = k^2*cs^2 - 4*pi*G*rho0
//  The Jeans wavenumber is k_J = sqrt(4*pi*G*rho0)/cs
//  For k < k_J: exponential growth (gravitational instability)
//  For k > k_J: oscillatory waves (modified sound waves)
//
//  References:
//  - Jeans (1902) Phil. Trans. R. Soc. Lond. A 199, 1-53
//  - Stone et al. (2008) ApJS 178, 137-177 (Athena test suite)

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
#include "gravity/gravity.hpp"
#include "gravity/mg_gravity.hpp"
#include "pgen/pgen.hpp"

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::SelfGravity()
//  \brief Problem Generator for Jeans instability/wave test
//
//  Sets up a sinusoidal density perturbation in a uniform medium:
//    rho = rho0 * (1 + amp * sin(k*x))
//    v = v0 * sin(k*x)  [optionally for traveling wave]
//    P = P0 * (1 + amp * sin(k*x)) [for isentropic perturbation]
//
//  For periodic domain with box size L, wavelength lambda = L/n_waves:
//    k = 2*pi*n_waves/L
//  Compare k to k_J = sqrt(4*pi*G*rho0)/cs to predict behavior

void ProblemGenerator::SelfGravity(ParameterInput *pin, const bool restart) {
  // nothing needs to be done on restarts for this pgen
  if (restart) return;
  
  // Read input parameters
  Real four_pi_G = pin->GetOrAddReal("gravity", "four_pi_G", 1.0);
  // Jeans wave parameters
  Real rho0 = pin->GetOrAddReal("problem", "rho0", 1.0);              // background density
  Real amp = pin->GetOrAddReal("problem", "amp", 1.0e-6);             // perturbation amplitude
  Real n_jeans = pin->GetOrAddReal("problem", "n_jeans", -1.0);        // lambda/lambda_Jeans

  // Determine sound speed from EOS type
  std::string eos_type = pin->GetString("hydro", "eos");
  Real cs;
  Real gamma = 0.0, gm1 = 0.0, p0 = 0.0;
  bool is_isothermal = (eos_type == "isothermal");
  if (is_isothermal) {
    cs = pin->GetReal("hydro", "iso_sound_speed");
  } else {
    gamma = pin->GetOrAddReal("hydro", "gamma", 5.0/3.0);
    gm1 = gamma - 1.0;
    p0 = pin->GetOrAddReal("problem", "p0", 1.0);
    cs = std::sqrt(gamma * p0 / rho0);
  }
  // Get domain size to compute actual wavenumber
  Real Lx1 = pin->GetReal("mesh", "x1max") - pin->GetReal("mesh", "x1min");
  Real Lx2 = pin->GetReal("mesh", "x2max") - pin->GetReal("mesh", "x2min");
  Real Lx3 = pin->GetReal("mesh", "x3max") - pin->GetReal("mesh", "x3min");

  Real ang_3 = std::atan(Lx1/Lx2);
  Real sin_a3 = std::sin(ang_3);
  Real cos_a3 = std::cos(ang_3);
  Real ang_2 = std::atan(0.5*(Lx1*cos_a3 + Lx2*sin_a3)/Lx3);
  Real sin_a2 = std::sin(ang_2);
  Real cos_a2 = std::cos(ang_2);
  
  Real x1 = Lx1*cos_a2*cos_a3;
  Real x2 = Lx2*cos_a2*sin_a3;
  Real x3 = Lx3*sin_a2;

  Real lambda = std::min(x1, std::min(x2, x3));
  Real lambda_jeans = lambda/n_jeans;
  // If n_jeans is specified, set effective four_pi_G to enforce k_J = n_jeans * k
  if (n_jeans > 0.0){
    Real G = M_PI * (cs*cs)/(rho0*lambda_jeans*lambda_jeans);
    four_pi_G = 4 * M_PI * G ;
    // propagate computed four_pi_G to gravity driver (pgen runs after driver init)
    pin->SetReal("gravity", "four_pi_G", four_pi_G);
    if (pmy_mesh_->pmb_pack->pgrav != nullptr) {
      pmy_mesh_->pmb_pack->pgrav->four_pi_G = four_pi_G;
      if (pmy_mesh_->pmb_pack->pgrav->pmgd != nullptr) {
        pmy_mesh_->pmb_pack->pgrav->pmgd->SetFourPiG(four_pi_G);
      }
    }
  }
  Real k_wave = 2.0*M_PI/lambda;
  Real k_jeans = 2.0*M_PI/lambda_jeans;
  Real omega2 = SQR(k_wave)*cs*cs*(1.0 - SQR(n_jeans));
  Real omega = std::sqrt(std::abs(omega2));

  std::cout << "Jeans wave test parameters:" << std::endl;
  std::cout << "  rho0 = " << rho0 << ", cs = " << cs;
  if (!is_isothermal) std::cout << ", p0 = " << p0;
  std::cout << std::endl;
  std::cout << "  k_Jeans = " << k_jeans << ", k_wave = " << k_wave << std::endl;
  std::cout << "  k/k_J = " << k_wave/k_jeans << std::endl;
  if (n_jeans > 0.0) {
    std::cout << "  n_Jeans = " << n_jeans << " (lambda/lambda_Jeans)" << std::endl;
  }
  std::cout << "  four_pi_G = " << four_pi_G << std::endl;
  if (k_wave < k_jeans) {
    std::cout << "  Gravitationally unstable! Growth rate = " << omega << std::endl;
  } else {
    std::cout << "  Stable oscillation. Frequency = " << omega << std::endl;
  }
  
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

    par_for("jeans_wave_hydro_init", DevExeSpace(), 0, (pmbp->nmb_thispack-1), ks, ke, js, je, is, ie,
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


      Real x = cos_a2*(x1v*cos_a3 + x2v*sin_a3) + x3v*sin_a2;
      Real sinkx = std::sin(x*k_wave);
      Real coskx = std::cos(x*k_wave);
      Real M = (n_jeans > 1.0) ? rho0*(omega/k_wave)*amp*coskx:0.0;
      Real mx = M*cos_a3*cos_a2;
      Real my = M*sin_a3*cos_a2;
      Real mz = M*sin_a2;

      // set hydro conserved variables: density, momenta, and (if not isothermal) energy
      u0(m, IDN, k, j, i) = rho0*(1.0+amp*sinkx+amp*amp*std::sin(x1v*k_wave));
      u0(m, IM1, k, j, i) = mx;
      u0(m, IM2, k, j, i) = my;
      u0(m, IM3, k, j, i) = mz;
      if (!is_isothermal) {
        // Isentropic pressure perturbation (matches Athena++ reference)
        u0(m, IEN, k, j, i) = p0/gm1*(1.0 + gamma*amp*sinkx);
        u0(m, IEN, k, j, i) += 0.5*SQR(u0(m, IM1, k, j, i))/u0(m, IDN, k, j, i);
        u0(m, IEN, k, j, i) += 0.5*SQR(u0(m, IM2, k, j, i))/u0(m, IDN, k, j, i);
        u0(m, IEN, k, j, i) += 0.5*SQR(u0(m, IM3, k, j, i))/u0(m, IDN, k, j, i);
      }
    });
  }  // End initialization of Hydro variables

  return;
}