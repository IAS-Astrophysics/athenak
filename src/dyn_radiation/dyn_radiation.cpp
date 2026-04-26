//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dyn_radiation.cpp
//! \brief implementation of Radiation class constructor and assorted other functions

#include <float.h>

#include <iostream>
#include <algorithm> // max
#include <string>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "srcterms/srcterms.hpp"
#include "bvals/bvals.hpp"
#include "coordinates/adm.hpp"
#include "coordinates/coordinates.hpp"
#include "geodesic-grid/geodesic_grid.hpp"
#include "units/units.hpp"
#include "dyn_radiation/dyn_radiation.hpp"

namespace dyn_radiation {
//----------------------------------------------------------------------------------------
// constructor, initializes data structures and parameters

DynRadiation::DynRadiation(MeshBlockPack *ppack, ParameterInput *pin) :
    pmy_pack(ppack),
    i0("i0",1,1,1,1,1),
    i1("i1",1,1,1,1,1),
    iflx("iflx",1,1,1,1,1),
    divfa("divfa",1,1,1,1,1),
    nh_c("nh_c",1,1),
    nh_f("nh_f",1,1,1),
    tet_c("tet_c",1,1,1,1,1,1),
    tetcov_c("tetcov_c",1,1,1,1,1,1),
    tet_d1_x1f("tet_d1_x1f",1,1,1,1,1),
    tet_d2_x2f("tet_d2_x2f",1,1,1,1,1),
    tet_d3_x3f("tet_d3_x3f",1,1,1,1,1),
    sqrt_detg_c("sqrt_detg_c",1,1,1,1),
    sqrt_detg_x1f("sqrt_detg_x1f",1,1,1,1),
    sqrt_detg_x2f("sqrt_detg_x2f",1,1,1,1),
    sqrt_detg_x3f("sqrt_detg_x3f",1,1,1,1),
    adm_alpha_c("adm_alpha_c",1,1,1,1),
    adm_beta_u_c("adm_beta_u_c",1,1,1,1,1),
    adm_g_dd_c("adm_g_dd_c",1,1,1,1,1,1),
    adm_g_uu_c("adm_g_uu_c",1,1,1,1,1,1),
    adm_K_dd_c("adm_K_dd_c",1,1,1,1,1,1),
    adm_cotriad_c("adm_cotriad_c",1,1,1,1,1,1),
    adm_grad_alpha_c("adm_grad_alpha_c",1,1,1,1,1),
    adm_grad_beta_u_c("adm_grad_beta_u_c",1,1,1,1,1),
    adm_grad_g_dd_c("adm_grad_g_dd_c",1,1,1,1,1),
    adm_grad_g_uu_c("adm_grad_g_uu_c",1,1,1,1,1),
    adm_grad_cotriad_c("adm_grad_cotriad_c",1,1,1,1,1),
    adm_dt_cotriad_c("adm_dt_cotriad_c",1,1,1,1,1),
    na("na",1,1,1,1,1,1),
    norm_to_tet("norm_to_tet",1,1,1,1,1,1) {
  // Check for relativistic geometry.  ADM/Z4c runs provide metric fields through
  // ppack->padm even when the coordinate object is not a stationary GR metric.
  if (!(pmy_pack->pcoord->is_general_relativistic ||
        pmy_pack->pcoord->is_dynamical_relativistic ||
        pin->DoesBlockExist("adm") || pin->DoesBlockExist("z4c"))) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
      << std::endl << "Radiation requires general relativity or ADM/Z4c metric fields"
      << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // Check for hydrodynamics, mhd, and units
  is_hydro_enabled = pin->DoesBlockExist("hydro");
  is_mhd_enabled = pin->DoesBlockExist("mhd");
  if (is_hydro_enabled && is_mhd_enabled) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
      << std::endl << "Radiation does not support two fluid calculations, yet "
      << "both <hydro> and <mhd> blocks exist in input file" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  are_units_enabled = pin->DoesBlockExist("units");

  // Enable dyn_radiation source term (dyn_radiation+(M)HD) by default if hydro or mhd enabled
  // Otherwise, disable dyn_radiation source term.  The former can be overriden by
  // specification in the input file.
  if (is_hydro_enabled || is_mhd_enabled) {
    rad_source = pin->GetOrAddBoolean("dyn_radiation","rad_source",true);
  } else {
    rad_source = false;
  }

  // Set dyn_radiation coupling parameters including scattering and absorption opacities,
  // dyn_radiation constant, and source term behavior.
  if (rad_source) {
    kappa_s = pin->GetReal("dyn_radiation","kappa_s");
    power_opacity = pin->GetOrAddBoolean("dyn_radiation","power_opacity",false);
    if (!(power_opacity)) {
      kappa_a = pin->GetReal("dyn_radiation","kappa_a");
      kappa_p = pin->GetReal("dyn_radiation","kappa_p");
    }
    is_compton_enabled = pin->GetOrAddBoolean("dyn_radiation","compton",false);
    if (is_compton_enabled && !(are_units_enabled)) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
        << std::endl << "Compton requires enabling units" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    if (are_units_enabled) {
      arad = (pmy_pack->punit->rad_constant_cgs*
              SQR(SQR(pmy_pack->punit->temperature_cgs()))/
              pmy_pack->punit->pressure_cgs());
    } else {
      arad = pin->GetReal("dyn_radiation","arad");
    }
    affect_fluid = pin->GetOrAddBoolean("dyn_radiation","affect_fluid",true);
  }

  // Check for fluid evolution
  fixed_fluid = pin->GetOrAddBoolean("dyn_radiation","fixed_fluid",false);

  // Source terms (if needed)
  if (pin->DoesBlockExist("rad_srcterms")) {
    psrc = new SourceTerms("rad_srcterms", ppack, pin);
  }

  // Setup angular mesh and dyn_radiation geometry data
  int nlevel = pin->GetInteger("dyn_radiation", "nlevel");
  std::string geometry = pin->GetOrAddString("dyn_radiation", "geometry",
                                             (pmy_pack->padm != nullptr) ? "adm" : "cks");
  if (geometry.compare("adm") == 0) {
    use_adm_geometry = true;
  } else if (geometry.compare("cks") == 0) {
    use_adm_geometry = false;
  } else {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "<dyn_radiation>/geometry = '" << geometry
              << "' not implemented; use 'adm' or 'cks'." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (use_adm_geometry && pmy_pack->padm == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "<dyn_radiation> geometry='adm' requires an <adm> "
              << "or <z4c> block." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  adm_metric_source = pin->GetOrAddBoolean("dyn_radiation", "adm_metric_source",
                                           use_adm_geometry);
  rotate_geo = pin->GetOrAddBoolean("dyn_radiation","rotate_geo",true);
  angular_fluxes = pin->GetOrAddBoolean("dyn_radiation","angular_fluxes",true);
  n_0_floor = pin->GetOrAddReal("dyn_radiation","n_0_floor",0.1);
  prgeo = new GeodesicGrid(nlevel, rotate_geo, angular_fluxes);

  // Total number of MeshBlocks on this rank to be used in array dimensioning
  int nmb = std::max((ppack->nmb_thispack), (ppack->pmesh->nmb_maxperrank));
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  {
  int ncells1 = indcs.nx1 + 2*(indcs.ng);
  int ncells2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
  int ncells3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;
  Kokkos::realloc(nh_c,prgeo->nangles,4);
  Kokkos::realloc(nh_f,prgeo->nangles,6,4);
  Kokkos::realloc(tet_c,nmb,4,4,ncells3,ncells2,ncells1);
  Kokkos::realloc(tetcov_c,nmb,4,4,ncells3,ncells2,ncells1);
  Kokkos::realloc(tet_d1_x1f,nmb,4,ncells3,ncells2,ncells1+1);
  Kokkos::realloc(tet_d2_x2f,nmb,4,ncells3,ncells2+1,ncells1);
  Kokkos::realloc(tet_d3_x3f,nmb,4,ncells3+1,ncells2,ncells1);
  Kokkos::realloc(sqrt_detg_c,nmb,ncells3,ncells2,ncells1);
  Kokkos::realloc(sqrt_detg_x1f,nmb,ncells3,ncells2,ncells1+1);
  Kokkos::realloc(sqrt_detg_x2f,nmb,ncells3,ncells2+1,ncells1);
  Kokkos::realloc(sqrt_detg_x3f,nmb,ncells3+1,ncells2,ncells1);
  Kokkos::realloc(adm_alpha_c,nmb,ncells3,ncells2,ncells1);
  Kokkos::realloc(adm_beta_u_c,nmb,3,ncells3,ncells2,ncells1);
  Kokkos::realloc(adm_g_dd_c,nmb,3,3,ncells3,ncells2,ncells1);
  Kokkos::realloc(adm_g_uu_c,nmb,3,3,ncells3,ncells2,ncells1);
  Kokkos::realloc(adm_K_dd_c,nmb,3,3,ncells3,ncells2,ncells1);
  Kokkos::realloc(adm_cotriad_c,nmb,3,3,ncells3,ncells2,ncells1);
  Kokkos::realloc(adm_grad_alpha_c,nmb,3,ncells3,ncells2,ncells1);
  Kokkos::realloc(adm_grad_beta_u_c,nmb,9,ncells3,ncells2,ncells1);
  Kokkos::realloc(adm_grad_g_dd_c,nmb,18,ncells3,ncells2,ncells1);
  Kokkos::realloc(adm_grad_g_uu_c,nmb,18,ncells3,ncells2,ncells1);
  Kokkos::realloc(adm_grad_cotriad_c,nmb,27,ncells3,ncells2,ncells1);
  Kokkos::realloc(adm_dt_cotriad_c,nmb,9,ncells3,ncells2,ncells1);
  if (angular_fluxes) {Kokkos::realloc(na,nmb,prgeo->nangles,ncells3,ncells2,ncells1,6);}
  if (is_hydro_enabled || is_mhd_enabled) {
    Kokkos::realloc(norm_to_tet,nmb,4,4,ncells3,ncells2,ncells1);
  }
  }
  PrepareADMGeometry();

  // (3) read time-evolution option [already error checked in driver constructor]
  // Then initialize memory and algorithms for reconstruction and Riemann solvers
  std::string evolution_t = pin->GetString("time","evolution");

  // allocate memory for intensities
  {
  int ncells1 = indcs.nx1 + 2*(indcs.ng);
  int ncells2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
  int ncells3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;
  Kokkos::realloc(i0,nmb,prgeo->nangles,ncells3,ncells2,ncells1);
  }

  // allocate memory for conserved variables on coarse mesh
  if (ppack->pmesh->multilevel) {
    auto &indcs = pmy_pack->pmesh->mb_indcs;
    int nccells1 = indcs.cnx1 + 2*(indcs.ng);
    int nccells2 = (indcs.cnx2 > 1)? (indcs.cnx2 + 2*(indcs.ng)) : 1;
    int nccells3 = (indcs.cnx3 > 1)? (indcs.cnx3 + 2*(indcs.ng)) : 1;
    Kokkos::realloc(coarse_i0,nmb,prgeo->nangles,nccells3,nccells2,nccells1);
  }

  // allocate boundary buffers for conserved (cell-centered) variables
  pbval_i = new MeshBoundaryValuesCC(ppack, pin, false);
  pbval_i->InitializeBuffers(prgeo->nangles);

  // for time-evolving problems, continue to construct methods, allocate arrays
  if (evolution_t.compare("stationary") != 0) {
    // select reconstruction method (default PLM)
    {std::string xorder = pin->GetOrAddString("dyn_radiation","reconstruct","plm");
    if (xorder.compare("dc") == 0) {
      recon_method = ReconstructionMethod::dc;
    } else if (xorder.compare("plm") == 0) {
      recon_method = ReconstructionMethod::plm;
    } else if (xorder.compare("ppm4") == 0 ||
               xorder.compare("ppmx") == 0 ||
               xorder.compare("wenoz") == 0 ||
               xorder.compare("wenomz") == 0) {
      // check that nghost > 2
      if (indcs.ng < 3) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
          << std::endl << xorder << " reconstruction requires at least 3 ghost zones, "
          << "but <mesh>/nghost=" << indcs.ng << std::endl;
        std::exit(EXIT_FAILURE);
      }
      if (xorder.compare("ppm4") == 0) {
        recon_method = ReconstructionMethod::ppm4;
      } else if (xorder.compare("ppmx") == 0) {
        recon_method = ReconstructionMethod::ppmx;
      } else if (xorder.compare("wenoz") == 0) {
        recon_method = ReconstructionMethod::wenoz;
      } else if (xorder.compare("wenomz") == 0) {
        recon_method = ReconstructionMethod::wenomz;
      }
    } else {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "<dyn_radiation> recon = '" << xorder << "' not implemented"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
    }

    // allocate second registers, fluxes, masks
    int ncells1 = indcs.nx1 + 2*(indcs.ng);
    int ncells2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
    int ncells3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;
    Kokkos::realloc(i1,      nmb,prgeo->nangles,ncells3,ncells2,ncells1);
    Kokkos::realloc(iflx.x1f,nmb,prgeo->nangles,ncells3,ncells2,ncells1);
    Kokkos::realloc(iflx.x2f,nmb,prgeo->nangles,ncells3,ncells2,ncells1);
    Kokkos::realloc(iflx.x3f,nmb,prgeo->nangles,ncells3,ncells2,ncells1);
    if (angular_fluxes) {
      Kokkos::realloc(divfa,nmb,prgeo->nangles,ncells3,ncells2,ncells1);
    }
  }
}

//----------------------------------------------------------------------------------------
// destructor

DynRadiation::~DynRadiation() {
  delete pbval_i;
  delete prgeo;
  if (psrc != nullptr) {delete psrc;}
}

//----------------------------------------------------------------------------------------
//! \fn void DynRadiation::PrepareADMGeometry()
//! \brief Refresh ADM geometry and cached tetrads used by radiation transport.

void DynRadiation::PrepareADMGeometry() {
  if (use_adm_geometry && pmy_pack->pz4c == nullptr) {
    pmy_pack->padm->SetADMVariables(pmy_pack);
  }
  SetOrthonormalTetrad();
}

} // namespace dyn_radiation
