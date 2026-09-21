//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file nr_radiation.cpp
//! \brief Constructor and hook enrollment for the SC (short-characteristics) radiation
//! module.

#include <algorithm>
#include <iostream>
#include <limits>
#include <string>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/coordinates.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "nr_radiation/nr_radiation.hpp"

namespace nr_radiation {

//----------------------------------------------------------------------------------------
// constructor, initializes data structures and parameters

SC::SC(MeshBlockPack *ppack, ParameterInput *pin) :
    ir("sc_ir",1,1,1,1,1),
    coarse_ir("sc_coarse_ir",1,1,1,1,1),
    srad("sc_srad",1,1,1,1,1),
    coarse_srad("sc_coarse_srad",1,1,1,1,1),
    chi("sc_chi",1,1,1,1),
    brad("sc_brad",1,1,1,1),
    moments("sc_moments",1,1,1,1,1),
    j_prev("sc_j_prev",1,1,1,1,1),
    qrad("sc_qrad",1,1,1,1),
    sigma_a("sc_sigma_a",1,1,1,1,1),
    pmy_pack(ppack) {
  // straight-line rays require flat, Cartesian spacetime
  if (pmy_pack->pcoord->is_general_relativistic) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
      << std::endl << "<nr_radiation> requires flat, Cartesian coordinates; the "
      << "short-characteristics straight-ray formal solution is not valid in GR"
      << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // Check for hydro/mhd coupling (mirrors radiation::Radiation)
  is_hydro_enabled = pin->DoesBlockExist("hydro");
  is_mhd_enabled = pin->DoesBlockExist("mhd");
  if (is_hydro_enabled && is_mhd_enabled) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
      << std::endl << "<nr_radiation> does not support two fluid calculations, yet "
      << "both <hydro> and <mhd> blocks exist in input file" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  // radiation -> gas coupling (AddQrad + the radiation dt limit). The gas always emits
  // when present, independent of this flag. crat and prat are required only when true.
  // Without a fluid there is nothing to act on, so the flag is false regardless of the
  // deck (same rule as the GR module's rad_source).
  if (is_hydro_enabled || is_mhd_enabled) {
    affect_fluid = pin->GetOrAddBoolean("nr_radiation", "affect_fluid", true);
  } else {
    affect_fluid = false;
  }

  // Kernel scheme ---------------------------------------------------------------------
  sweep_kernel_name = pin->GetOrAddString("nr_radiation", "sweep_kernel", "wavefront");
  if (sweep_kernel_name == "wavefront") {
    sweep_kernel = SweepKernel::wavefront;
  } else if (sweep_kernel_name == "tiled") {
    sweep_kernel = SweepKernel::tiled;
  } else if (sweep_kernel_name == "plane") {
    sweep_kernel = SweepKernel::plane;
  } else {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
      << std::endl << "<nr_radiation>/sweep_kernel = '" << sweep_kernel_name
      << "' is not recognised; valid values are 'wavefront', 'tiled', 'plane'"
      << std::endl;
    std::exit(EXIT_FAILURE);
  }
  tile_size = pin->GetOrAddInteger("nr_radiation", "tile_size", 0);
  team_size = pin->GetOrAddInteger("nr_radiation", "team_size", 0);
  if (tile_size < 0 || team_size < 0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
      << std::endl << "<nr_radiation>/tile_size and team_size must be >= 0 (0 = whole "
      << "meshblock / Kokkos::AUTO); got " << tile_size << ", " << team_size << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (sweep_kernel != SweepKernel::tiled && (tile_size != 0 || team_size != 0)) {
    if (global_variable::my_rank == 0) {
      std::cout << "### WARNING in " << __FILE__
                << ": <nr_radiation>/tile_size and team_size "
        << "apply only to sweep_kernel = tiled; ignored under sweep_kernel = "
        << sweep_kernel_name << std::endl;
    }
    tile_size = 0;
    team_size = 0;
  }
  if (team_size % 32 != 0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
      << std::endl << "<nr_radiation>/team_size must be a multiple of 32 (a warp); got "
      << team_size << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // Iteration control -----------------------------------------------------------------
  iter_max = pin->GetOrAddInteger("nr_radiation", "iter_max", 100);
  iter_min = pin->GetOrAddInteger("nr_radiation", "iter_min", 1);
  iter_tol = pin->GetOrAddReal("nr_radiation", "iter_tol", 1.0e-6);
  if (iter_max < 1 || iter_min < 1 || iter_min > iter_max || iter_tol < 0.0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
      << std::endl << "<nr_radiation> iteration controls must satisfy 1 <= iter_min <= "
      << "iter_max and iter_tol >= 0; got iter_min = " << iter_min << ", iter_max = "
      << iter_max << ", iter_tol = " << iter_tol << std::endl;
    std::exit(EXIT_FAILURE);
  }
  niter_last = 0;
  resid_last = 0.0;
  converged = false;

  // Opacity / coupling parameters -----------------------------------------------------
  // kappa_a is used only by the default opacity sigma_a = kappa_a * rho, so it needs a
  // fluid; with a fluid it is required unless the pgen enrolls an opacity hook
  // (enrollment happens after this constructor, so that check is deferred to pgen.cpp).
  // crat and prat are required only when the radiation acts back on the gas.
  kappa_a_specified = pin->DoesParameterExist("nr_radiation", "kappa_a");
  kappa_a = pin->GetOrAddReal("nr_radiation", "kappa_a", 0.0);
  if (kappa_a < 0.0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
      << std::endl << "<nr_radiation>/kappa_a must be non-negative" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (kappa_a_specified && !is_hydro_enabled && !is_mhd_enabled) {
    if (global_variable::my_rank == 0) {
      std::cout << "### WARNING in " << __FILE__ << ": <nr_radiation>/kappa_a is ignored "
        << "without a fluid (no density to "
        << "multiply); the opacity comes from the enrolled "
        << "opacity function" << std::endl;
    }
  }
  if (affect_fluid) {
    if (!pin->DoesParameterExist("nr_radiation", "crat") ||
        !pin->DoesParameterExist("nr_radiation", "prat")) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
        << std::endl
        << "<nr_radiation>/crat and prat are required when affect_fluid = true"
        << std::endl;
      std::exit(EXIT_FAILURE);
    }
    crat = pin->GetReal("nr_radiation", "crat");
    prat = pin->GetReal("nr_radiation", "prat");
  } else {
    crat = pin->GetOrAddReal("nr_radiation", "crat", 1.0);
    prat = pin->GetOrAddReal("nr_radiation", "prat", 1.0);
  }

  // Angular quadrature ----------------------------------------------------------------
  int nmu = pin->GetInteger("nr_radiation", "nmu");
  Mesh *pm = pmy_pack->pmesh;
  int ndim = (pm->three_d) ? 3 : ((pm->two_d) ? 2 : 1);
  pang = new SCAngularGrid(ndim, nmu);
  nang_tot = pang->noct * pang->nang;

  // Reject boundary flags the intensity has no treatment for, rather than silently
  // leaving those ghosts at their initial zeros -- which reads as a vacuum edge and is
  // a wrong answer with no symptom. Handled: periodic, outflow, inflow, diode and vacuum
  // in bvals/physics/radiation_bcs.cpp, reflect in SC::ApplyReflectBCs, and user, where
  // the pgen writes the ghosts itself.
  // faces are ordered inner_x1, outer_x1, inner_x2, ...; Mesh fills mesh_bcs only for
  // the dimensions that exist, so entries above 2*ndim are indeterminate and unreadable
  for (int f = 0; f < 2*ndim; ++f) {
    BoundaryFlag fl = pm->mesh_bcs[f];
    if (fl != BoundaryFlag::periodic && fl != BoundaryFlag::outflow &&
        fl != BoundaryFlag::inflow   && fl != BoundaryFlag::user &&
        fl != BoundaryFlag::reflect  && fl != BoundaryFlag::diode &&
        fl != BoundaryFlag::vacuum   && fl != BoundaryFlag::block) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
        << std::endl << "<mesh> boundary flag '" << pm->GetBoundaryString(fl)
        << "' on face " << f << " has no treatment for the radiation intensity. Use "
        << "periodic, outflow, inflow, diode, vacuum, reflect, or user." << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }

  // tiled: the tile must divide the meshblock interior along every active dimension
  if (sweep_kernel == SweepKernel::tiled && tile_size > 0) {
    auto &mbi = pmy_pack->pmesh->mb_indcs;
    int bad = 0;
    if (mbi.nx1 % tile_size != 0) bad = 1;
    if (ndim >= 2 && mbi.nx2 % tile_size != 0) bad = 2;
    if (ndim == 3 && mbi.nx3 % tile_size != 0) bad = 3;
    if (bad) {
      int n = (bad == 1) ? mbi.nx1 : ((bad == 2) ? mbi.nx2 : mbi.nx3);
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
        << std::endl << "<nr_radiation>/tile_size = " << tile_size
        << " does not divide the "
        << "meshblock x" << bad << " interior size " << n << ". Valid tile sizes for x"
        << bad
        << " are:";
      for (int d = 1; d <= n; ++d) {
      if (n % d == 0) std::cout << " " << d;
    }
      std::cout << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }

  // Array allocation ------------------------------------------------------------------
  int nmb = std::max((ppack->nmb_thispack), (ppack->pmesh->nmb_maxperrank));
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int ncells1 = indcs.nx1 + 2*(indcs.ng);
  int ncells2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*(indcs.ng)) : 1;
  int ncells3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*(indcs.ng)) : 1;

  Kokkos::realloc(ir, nmb, nang_tot, ncells3, ncells2, ncells1);
  Kokkos::realloc(srad, nmb, 1, ncells3, ncells2, ncells1);
  Kokkos::realloc(chi, nmb, ncells3, ncells2, ncells1);
  Kokkos::realloc(brad, nmb, ncells3, ncells2, ncells1);
  // full width once, here: a restart rebuilds slot 0 from the restored ir before the
  // outputs are built, and Kokkos::realloc does not preserve contents
  Kokkos::realloc(moments, nmb, 10, ncells3, ncells2, ncells1);
  Kokkos::realloc(j_prev, nmb, 1, ncells3, ncells2, ncells1);
  Kokkos::realloc(qrad, nmb, ncells3, ncells2, ncells1);
  Kokkos::realloc(sigma_a, nmb, 1, ncells3, ncells2, ncells1);
  Kokkos::deep_copy(ir, 0.0);
  Kokkos::deep_copy(srad, 0.0);
  Kokkos::deep_copy(moments, 0.0);   // zero reference: the first solve takes two sweeps
  Kokkos::deep_copy(qrad, 0.0);
  Kokkos::deep_copy(sigma_a, 0.0);    // both recomputed before first use
  Kokkos::deep_copy(brad, 0.0);   // (UpdateOpacity / UpdateEmission)

  // coarse_ir / coarse_srad for SMR/AMR (CC Restrict/Prolong); only a multilevel mesh
  // reads them, so, as in Hydro and Radiation, a uniform mesh does not allocate them
  if (ppack->pmesh->multilevel) {
    int nccells1 = indcs.cnx1 + 2*(indcs.ng);
    int nccells2 = (indcs.cnx2 > 1) ? (indcs.cnx2 + 2*(indcs.ng)) : 1;
    int nccells3 = (indcs.cnx3 > 1) ? (indcs.cnx3 + 2*(indcs.ng)) : 1;
    Kokkos::realloc(coarse_ir, nmb, nang_tot, nccells3, nccells2, nccells1);
    Kokkos::realloc(coarse_srad, nmb, 1, nccells3, nccells2, nccells1);
    Kokkos::deep_copy(coarse_ir, 0.0);
    Kokkos::deep_copy(coarse_srad, 0.0);
  }

  // boundary communication. InitializeBuffers also allocates (zeroed) the inflow table
  // pbval_ir->i_in(nang_tot, 6 faces) that RadiationBCs reads on faces flagged `inflow`;
  // a pgen that wants a lit face writes it, as the GR module's pgens do (default: vacuum)
  pbval_ir = new MeshBoundaryValuesCC(ppack, pin, false);
  pbval_ir->InitializeBuffers(nang_tot);
  pbval_srad = new MeshBoundaryValuesCC(ppack, pin, false);
  pbval_srad->InitializeBuffers(1);

  // hyperplane orderings for the sweep kernels (static in the meshblock/tile dims)
  BuildIndices();

  dtnew = std::numeric_limits<Real>::max();
}

//----------------------------------------------------------------------------------------
// destructor

SC::~SC() {
  delete pbval_srad;
  delete pbval_ir;
  delete pang;
}

//----------------------------------------------------------------------------------------
//! \fn void SC::EnrollOpacityFunction / SC::EnrollEmissionFunction
//! \brief Enroll user opacity / emission hooks (contract in sc/sc_hooks.hpp).
//! Called from the pgen's UserProblem, BEFORE any restart early-return.

void SC::EnrollOpacityFunction(SCOpacityFnPtr myfunc) {
  if (myfunc == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
      << std::endl << "EnrollOpacityFunction called with null function" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  user_opacity_func = myfunc;
}

void SC::EnrollEmissionFunction(SCEmissionFnPtr myfunc) {
  if (myfunc == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
      << std::endl << "EnrollEmissionFunction called with null function" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  user_emission_func = myfunc;
}

}  // namespace nr_radiation
