//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file poisson_mg_test.cpp
//! \brief Scalar Poisson multigrid manufactured-solution scaffold.

// C++ headers
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

// AthenaK headers
#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "gravity/gravity.hpp"
#include "gravity/mg_gravity.hpp"
#include "pgen/pgen.hpp"

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::PoissonMGTest()
//! \brief Initialize rho for a periodic manufactured Poisson solve.

void ProblemGenerator::PoissonMGTest(ParameterInput *pin, const bool restart) {
  if (restart) return;

  if (!pin->GetOrAddBoolean("poisson_test", "enabled", false)) {
    if (global_variable::my_rank == 0) {
      std::cout << "### FATAL ERROR in ProblemGenerator::PoissonMGTest" << std::endl
                << "<poisson_test>/enabled must be true for pgen_name=poisson_mg_test."
                << std::endl;
    }
    std::exit(EXIT_FAILURE);
  }

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->phydro == nullptr || pmbp->pgrav == nullptr) {
    if (global_variable::my_rank == 0) {
      std::cout << "### FATAL ERROR in ProblemGenerator::PoissonMGTest" << std::endl
                << "The Poisson MG test currently uses <hydro> density as a scalar "
                << "source and <gravity> as the scalar Multigrid driver." << std::endl;
    }
    std::exit(EXIT_FAILURE);
  }

  std::string manufactured =
      pin->GetOrAddString("poisson_test", "manufactured", "sin");
  if (manufactured != "sin") {
    if (global_variable::my_rank == 0) {
      std::cout << "### FATAL ERROR in ProblemGenerator::PoissonMGTest" << std::endl
                << "Only <poisson_test>/manufactured=sin is implemented." << std::endl;
    }
    std::exit(EXIT_FAILURE);
  }

  const Real amp = pin->GetOrAddReal("poisson_test", "amplitude", 1.0e-3);
  const Real rho0 = pin->GetOrAddReal("poisson_test", "density_offset", 1.0);
  const int waves = pin->GetOrAddInteger("poisson_test", "waves", 1);
  const Real pressure = pin->GetOrAddReal("poisson_test", "pressure", 1.0);

  auto &indcs = pmy_mesh_->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb = pmbp->nmb_thispack;
  auto &size = pmbp->pmb->mb_size;
  auto &u0 = pmbp->phydro->u0;
  auto &phi = pmbp->pgrav->phi;
  bool is_ideal = pmbp->phydro->peos->eos_data.is_ideal;
  Real gamma = pmbp->phydro->peos->eos_data.gamma;
  Real gm1 = gamma - 1.0;

  const Real pi = std::acos(-1.0);
  const Real two_pi_waves = 2.0*pi*static_cast<Real>(waves);

  par_for("poisson_mg_test_init", DevExeSpace(), 0, nmb - 1,
          ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real x1min = size.d_view(m).x1min, x1max = size.d_view(m).x1max;
    Real x2min = size.d_view(m).x2min, x2max = size.d_view(m).x2max;
    Real x3min = size.d_view(m).x3min, x3max = size.d_view(m).x3max;
    Real lx = x1max - x1min;
    Real ly = x2max - x2min;
    Real lz = x3max - x3min;
    Real x = CellCenterX(i - is, indcs.nx1, x1min, x1max);
    Real y = CellCenterX(j - js, indcs.nx2, x2min, x2max);
    Real z = CellCenterX(k - ks, indcs.nx3, x3min, x3max);

    Real sx = std::sin(two_pi_waves*(x - x1min)/lx);
    Real sy = std::sin(two_pi_waves*(y - x2min)/ly);
    Real sz = std::sin(two_pi_waves*(z - x3min)/lz);
    Real exact = amp*sx*sy*sz;
    Real ksq = SQR(two_pi_waves/lx) + SQR(two_pi_waves/ly) + SQR(two_pi_waves/lz);
    Real lap_exact = -ksq*exact;

    // The gravity MG solve subtracts the periodic mean source, leaving lap_exact.
    u0(m, IDN, k, j, i) = rho0 + lap_exact;
    u0(m, IM1, k, j, i) = 0.0;
    u0(m, IM2, k, j, i) = 0.0;
    u0(m, IM3, k, j, i) = 0.0;
    if (is_ideal) {
      u0(m, IEN, k, j, i) = pressure/gm1;
    }
    phi(m, 0, k, j, i) = 0.0;
  });

  if (global_variable::my_rank == 0) {
    std::cout << "Poisson MG test initialized: manufactured=sin amplitude="
              << amp << " waves=" << waves
              << " density_offset=" << rho0
              << " periodic_mean_source_subtracted=1" << std::endl;
  }
}
