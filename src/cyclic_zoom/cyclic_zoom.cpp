//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file cyclic_zoom.cpp
//! \brief Implementation of constructor and functions in CyclicZoom class

#include <iostream>
#include <string>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/adm.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"
#include "coordinates/cartesian_ks.hpp"
#include "coordinates/cell_locations.hpp"
#include "pgen/pgen.hpp"
#include "cyclic_zoom/cyclic_zoom.hpp"

//----------------------------------------------------------------------------------------
// constructor, initializes data structures and parameters

CyclicZoom::CyclicZoom(Mesh *pm, ParameterInput *pin) :
    pmesh(pm) {
  // cycle through ParameterInput list and read each <amr_criterion> block
  int ncriteria = 0;
  for (auto it = pin->block.begin(); it != pin->block.end(); ++it) {
    if (it->block_name.compare(0, 13, "amr_criterion") == 0) {
      ++ncriteria;
      std::string method = pin->GetString(it->block_name, "method");
      if (method.compare("cyclic_zoom") == 0) {
        block_name = it->block_name;
      }
    }
  }

  // Check for unimplemented physics or a 1D/2D mesh
  MeshBlockPack *pmbp = pmesh->pmb_pack;
  if (pmbp->pdyngr != nullptr && ncriteria != 1) {
    Kokkos::abort("Valencia cyclic zoom requires exactly one AMR criterion");
  }
  const char *err = nullptr;
  if (!(pmesh->three_d)) {
    err = "Cyclic zoom requires a 3D mesh";
  } else if (pmbp->phydro == nullptr && pmbp->pmhd == nullptr && pmbp->prad == nullptr) {
    err = "Cyclic zoom requires hydro, MHD, and/or radiation";
  } else if (pmbp->pionn != nullptr) {
    err = "Cyclic zoom is not implemented for ion-neutral (two-fluid) MHD";
  } else if (pmbp->ppart != nullptr) {
    err = "Cyclic zoom is not implemented for particles";
  } else if (pin->DoesBlockExist("gravity")) {
    err = "Cyclic zoom is not implemented for self-gravity";
  } else if (pmbp->pturb != nullptr) {
    err = "Cyclic zoom is not implemented for turbulence driving";
  } else if (pmbp->pz4c != nullptr || pmbp->pdynrad != nullptr ||
             (pmbp->padm != nullptr &&
              (pmbp->pdyngr == nullptr || pmbp->padm->is_dynamic))) {
    err = "Cyclic zoom requires a stationary prescribed ADM metric and Valencia MHD; "
          "Z4c and dynamical radiation are unsupported";
  } else if (pin->DoesBlockExist("shearing_box")) {
    err = "Cyclic zoom is not implemented for the shearing box";
  }
  if (err != nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << err << std::endl;
    std::exit(EXIT_FAILURE);
  }

  if (pmesh->pmr->refinement_interval != 1 || pmesh->pmr->ncyc_check_amr != 1) {
    Kokkos::abort("Cyclic zoom requires ncycle_check=1 and refinement_interval=1");
  }
  // Set basic parameters
  verbose = pin->GetOrAddBoolean("cyclic_zoom","verbose",false);
  read_rst = pin->GetOrAddBoolean("cyclic_zoom","read_rst",true);
  write_rst = pin->GetOrAddBoolean("cyclic_zoom","write_rst",true);

  // Set zoom runtime state parameters
  zstate.id = 0;
  zstate.zone = pin->GetOrAddInteger(block_name,"zone",0);
  zstate.last_zone = zstate.zone;
  zstate.direction = pin->GetOrAddInteger(block_name,"direction",1);

  // Set zoom AMR parameters
  zamr.nlevels = pin->GetOrAddInteger(block_name,"nlevels",2);
  if (zamr.nlevels < 2 || zamr.nlevels > pmesh->max_level-pmesh->root_level+1) {
    Kokkos::abort("Invalid cyclic zoom levels: require 2 <= nlevels <= mesh levels");
  }
  zamr.max_level = pmesh->max_level;
  zamr.min_level = zamr.max_level - zamr.nlevels + 1;
  zamr.level = zamr.max_level - zstate.zone;
  zamr.refine_flag = - zstate.direction;
  zamr.zooming_in = false;
  zamr.zooming_out = false;

  // Set zoom region parameters
  zregion.x1c = pin->GetOrAddReal(block_name,"x1c",0.0);
  zregion.x2c = pin->GetOrAddReal(block_name,"x2c",0.0);
  zregion.x3c = pin->GetOrAddReal(block_name,"x3c",0.0);
  zregion.ncenters = pin->GetOrAddInteger(block_name,"ncenters",1);
  zregion.x1c2 = pin->GetOrAddReal(block_name,"x1c2",0.0);
  zregion.x2c2 = pin->GetOrAddReal(block_name,"x2c2",0.0);
  zregion.x3c2 = pin->GetOrAddReal(block_name,"x3c2",0.0);
  zregion.r_0 = pin->GetOrAddReal(block_name,"r_0",1.0);
  zregion.ref.f = pin->GetOrAddReal(block_name,"f_ref",1.0);
  zregion.ref.r_max = pin->GetOrAddReal(block_name,"r_ref_max",FLT_MAX);
  // default 0.4 based on monopole test
  zregion.exc.f = pin->GetOrAddReal(block_name,"f_exc",pmbp->pdyngr ? 0.0 : 0.4);
  if (pmbp->pdyngr && zregion.exc.f != 0.0) {
    Kokkos::abort("Valencia binary zoom uses physical puncture excision; set f_exc=0");
  }
  zregion.exc.r_max = pin->GetOrAddReal(block_name,"r_exc_max",8.0);
  // default 0.0
  zregion.cut.f = pin->GetOrAddReal(block_name,"f_cut",0.0);
  zregion.cut.r_max = pin->GetOrAddReal(block_name,"r_cut_max",0.0);
  // default 0.0
  zregion.flx.f = pin->GetOrAddReal(block_name,"f_flx",0.0);
  zregion.flx.r_max = pin->GetOrAddReal(block_name,"r_flx_max",0.0);

  // Eq. 17 of Guo et al. (2025), in individual-hole units near the horizons.
  // Equal-mass circular binaries share one radius; unequal masses need separate
  // radius ladders and are deliberately rejected by this preset.
  bondi_hydro_schedule = pin->GetOrAddBoolean(block_name,"bondi_hydro_schedule",false);
  residence_scale = pin->GetOrAddReal(block_name,"residence_scale",1.0);
  if (!std::isfinite(residence_scale) || residence_scale <= 0.0) {
    Kokkos::abort("residence_scale must be finite and positive");
  }
  if (bondi_hydro_schedule) {
    if (pmbp->pdyngr == nullptr || pin->GetOrAddReal("problem","b_inf",0.0) != 0.0 ||
        pin->GetOrAddString("problem","initial_condition","") != "bondi_inflow") {
      Kokkos::abort("Bondi hydro schedule requires the unmagnetized dynbbh Bondi setup");
    }
    bool single = pin->GetOrAddBoolean("problem","single_black_hole",false);
    if (!single && (pin->GetReal("problem","q") != 1.0 ||
                    !pin->GetOrAddBoolean("problem","corotating",false))) {
      Kokkos::abort("Bondi hydro binary schedule requires equal masses and corotation");
    }
    zoom_mass = single ? 1.0 : 0.5;
    merge_radius = single ? 0.0 : pin->GetReal("problem","sep");
    if (!std::isfinite(merge_radius) || merge_radius < 0.0 ||
        (!single && merge_radius <= 0.0)) Kokkos::abort("Invalid binary merge radius");
    zregion.r_0 = zoom_mass;
    zregion.ncenters = single ? 1 : 2;
    zregion.x1c = 0.5*merge_radius;
    zregion.x1c2 = -0.5*merge_radius;
    zregion.x2c = zregion.x3c = zregion.x2c2 = zregion.x3c2 = 0.0;
  }
  base_zregion = zregion;

  // Set zoom interval parameters
  zint.trun_fac = pin->GetOrAddReal(block_name,"trun_fac",1.0);
  zint.trun_pow = pin->GetOrAddReal(block_name,"trun_pow",0.0);
  zint.trun_max = pin->GetOrAddReal(block_name,"trun_max",FLT_MAX);
  // Read number of zones from input parameters
  int num_zones = zamr.nlevels;
  // Initialize the dynamic interval structure
  zint.initialize(num_zones);
  // Read the runtime factors from input file
  for (int i = 0; i < num_zones; ++i) {
    std::string param_name = "trun_fac_" + std::to_string(i);
    zint.trun_facs[i] = pin->GetOrAddReal(block_name, param_name.c_str(), zint.trun_fac);
  }

  if (zamr.nlevels < 2 || zamr.min_level < pmesh->root_level ||
      zstate.zone < 0 || zstate.zone >= zamr.nlevels ||
      (zstate.direction != 1 && zstate.direction != -1) ||
      (zstate.zone == 0 && zstate.direction != 1) ||
      (zstate.zone == zamr.nlevels-1 && zstate.direction != -1) ||
      zregion.r_0 <= 0.0 || zregion.ref.f < 1.0 ||
      zregion.ref.r_max < zregion.r_0*std::pow(2.0,zamr.nlevels-1) ||
      (zregion.ncenters != 1 && zregion.ncenters != 2)) {
    Kokkos::abort("Invalid cyclic zoom levels, direction or region geometry");
  }
  for (Real x : {zregion.r_0,zregion.ref.f,zregion.ref.r_max,
                 zregion.x1c,zregion.x2c,zregion.x3c,
                 zregion.x1c2,zregion.x2c2,zregion.x3c2,
                 zregion.exc.f,zregion.exc.r_max,zregion.cut.f,zregion.cut.r_max,
                 zregion.flx.f,zregion.flx.r_max,zint.trun_pow,zint.trun_max}) {
    if (!std::isfinite(x)) Kokkos::abort("Nonfinite cyclic zoom geometry or interval");
  }
  if (zregion.exc.f < 0 || zregion.exc.r_max < 0 || zregion.cut.f < 0 ||
      zregion.cut.r_max < 0 || zregion.flx.f < 0 || zregion.flx.r_max < 0 ||
      zint.trun_max < 0) Kokkos::abort("Negative cyclic zoom scale or interval");
  for (Real t : zint.trun_facs) {
    if (!std::isfinite(t) || t < 0.0) Kokkos::abort("Invalid cyclic zoom interval");
  }
  // Update zoom state
  zstate.next_time = pmesh->time;
  old_zregion = zregion; // initialize old zoom region
  SetRegionAndInterval();
  zstate.next_time += zint.runtime;

  // Set zoom emf parameters if needed
  zemf.emf_fmax = 1.0;
  zemf.emf_zmax = zamr.nlevels;
  zemf.add_emf = pin->GetOrAddBoolean(block_name,"add_emf",true); // default true
  if (zemf.add_emf) {
    zemf.emf_fmax = pin->GetOrAddReal(block_name,"emf_fmax",zemf.emf_fmax);
    zemf.emf_zmax = pin->GetOrAddInteger(block_name,"emf_zmax",zemf.emf_zmax);
  }

  // Initialize zoom data structures
  Initialize(pin);
  // Print diagnostic information
  PrintCyclicZoomDiagnostics();
}

//----------------------------------------------------------------------------------------
//! \fn void CyclicZoom::Initialize()
//! \brief Initialize CyclicZoom variables

CyclicZoom::~CyclicZoom() {
  delete pzdata;
  delete pzmesh;
}

void CyclicZoom::Initialize(ParameterInput *pin) {
  pzmesh = new ZoomMesh(this, pin);
  pzdata = new ZoomData(this, pin);
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void CyclicZoom::UpdateAMRFromRestart()
//! \brief Update CyclicZoom runtime parameters after reading restart file

void CyclicZoom::UpdateAMRFromRestart() {
  zamr.level = pmesh->max_level - zstate.zone;
  zamr.refine_flag = -zstate.direction;
  SetRegionAndInterval();
  AdjustExcisionForZoom();
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void CyclicZoom::PrintCyclicZoomDiagnostics()
//! \brief Print CyclicZoom information

void CyclicZoom::PrintCyclicZoomDiagnostics() {
  if (verbose && global_variable::my_rank == 0) {
    std::cout << "=============== CyclicZoom Information ===============" << std::endl;
    // print basic parameters
    std::cout << "Basic: read_rst = " << read_rst
              << " write_rst = " << write_rst << std::endl;
    // print mesh parameters
    std::cout << "Mesh: nzmb_max_perdvce = " << pzmesh->nzmb_max_perdvce
              << " nzmb_max_perhost = " << pzmesh->nzmb_max_perhost << std::endl;
    // print data parameters
    std::cout << "Data: nvars = " << pzdata->nvars
              << " nangles = " << pzdata->nangles
              << " zmb_data_cnt = " << pzdata->zmb_data_cnt
              << std::endl;
    std::cout << "Initial: d_zoom = " << pzdata->d_zoom
              << " p_zoom = " << pzdata->p_zoom
              << std::endl;
    // print electric field parameters
    std::cout << "Funcs: add_emf = " << zemf.add_emf << std::endl;
    std::cout << "Efield: emf_fmax = " << zemf.emf_fmax << " emf_zmax = " << zemf.emf_zmax
              << std::endl;
    // print region parameters
    std::cout << "Region: x1c = " << zregion.x1c << " x2c = " << zregion.x2c
              << " x3c = " << zregion.x3c << " r_0 = " << zregion.r_0
              << " radius = " << zregion.radius << std::endl
              << " f_ref = " << zregion.ref.f << " r_ref = " << zregion.ref.r
              << " r_ref_max = " << zregion.ref.r_max << std::endl
              << " f_exc = " << zregion.exc.f << " r_exc = " << zregion.exc.r
              << " r_exc_max = " << zregion.exc.r_max << std::endl
              << " f_cut = " << zregion.cut.f << " r_cut = " << zregion.cut.r
              << " r_cut_max = " << zregion.cut.r_max << std::endl
              << " f_flx = " << zregion.flx.f << " r_flx = " << zregion.flx.r
              << " r_flx_max = " << zregion.flx.r_max << std::endl;
    // print interval parameters
    std::cout << "Interval: trun_fac = " << zint.trun_fac
              << " trun_pow = " << zint.trun_pow
              << " trun_max = " << zint.trun_max
              << std::endl;
    // output zone-specific time factors
    for (int i = 0; i < zint.trun_facs.size(); ++i) {
      std::cout << " trun_fac_" << i << " = " << zint.trun_facs[i]
                << std::endl;
    }
    // print AMR level structure
    std::cout << "AMR: level = " << zamr.level << " max_level = " << zamr.max_level
              << " min_level = " << zamr.min_level << std::endl;
    // print zoom state information
    std::cout << "State: id = " << zstate.id << " zone = " << zstate.zone
              << " last_zone = " << zstate.last_zone
              << " direction = " << zstate.direction << std::endl;
    // print runtime information
    std::cout << "Time: runtime = " << zint.runtime << " next time = "
              << zstate.next_time << std::endl;
    std::cout << "======================================================" << std::endl;
  }
  return;
}
