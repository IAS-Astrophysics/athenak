//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file cyclic_zoom.cpp
//  \brief implementation of constructor and functions in CyclicZoom class

#include <iostream>
#include <string>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/cartesian_ks.hpp"
#include "coordinates/cell_locations.hpp"
#include "pgen/pgen.hpp"
#include "cyclic_zoom/cyclic_zoom.hpp"

//----------------------------------------------------------------------------------------
// constructor, initializes data structures and parameters

CyclicZoom::CyclicZoom(Mesh *pm, ParameterInput *pin) :
    pmesh(pm) {
  // cycle through ParameterInput list and read each <amr_criterion> block
  for (auto it = pin->block.begin(); it != pin->block.end(); ++it) {
    if (it->block_name.compare(0, 13, "amr_criterion") == 0) {
      std::string method = pin->GetString(it->block_name, "method");
      if (method.compare("cyclic_zoom") == 0) {
        block_name = it->block_name;
      }
    }
  }
  verbose = pin->GetOrAddBoolean("cyclic_zoom","verbose",false);
  read_rst = pin->GetOrAddBoolean("cyclic_zoom","read_rst",true);
  write_rst = pin->GetOrAddBoolean("cyclic_zoom","write_rst",true);

  // TODO(@mhguo): may set the parameters so that the initial level equals the max level
  // TODO(@mhguo): currently we need to check whether zamr.level is correct by hand
  zstate.id = 0;
  zstate.zone = pin->GetOrAddInteger(block_name,"zone",0);
  zstate.last_zone = zstate.zone;
  zstate.direction = pin->GetOrAddInteger(block_name,"direction",1);

  // Set zoom AMR parameters
  zamr.nlevels = pin->GetOrAddInteger(block_name,"nlevels",4);
  zamr.max_level = pmesh->max_level;
  zamr.min_level = zamr.max_level - zamr.nlevels + 1;
  zamr.level = zamr.max_level - zstate.zone;
  zamr.refine_flag = - zstate.direction;
  zamr.zooming_in = false;
  zamr.zooming_out = false;

  zregion.x1c = pin->GetOrAddReal(block_name,"x1c",0.0);
  zregion.x2c = pin->GetOrAddReal(block_name,"x2c",0.0);
  zregion.x3c = pin->GetOrAddReal(block_name,"x3c",0.0);
  zregion.r_0 = pin->GetOrAddReal(block_name,"r_0",1.0);
  zint.t_run_fac = pin->GetOrAddReal(block_name,"t_run_fac",1.0);
  zint.t_run_pow = pin->GetOrAddReal(block_name,"t_run_pow",0.0);
  zint.t_run_max = pin->GetOrAddReal(block_name,"t_run_max",FLT_MAX);
  // Read number of zones from input parameters
  int num_zones = zamr.nlevels;
  // Initialize the dynamic interval structure
  zint.initialize(num_zones);
  // Read the runtime factors from input file
  for (int i = 0; i < num_zones; ++i) {
    std::string param_name = "t_run_fac_zone_" + std::to_string(i);
    Real zone_factor = pin->GetOrAddReal(block_name, param_name.c_str(), zint.t_run_fac);
    zint.t_run_fac_zones[i] = zone_factor;
  }

  zstate.next_time = pmesh->time;
  old_zregion = zregion; // initialize old zoom region
  SetRegionAndInterval();
  zstate.next_time += zint.runtime;
  zemf.emf_fmax = 1.0;
  // Think whether to read emf parameters from input file
  zemf.add_emf = pin->GetOrAddBoolean(block_name,"add_emf",true); // default true
  if (zemf.add_emf) {
    zemf.emf_fmax = pin->GetOrAddReal(block_name,"emf_fmax",zemf.emf_fmax);
    zemf.emf_zmax = pin->GetOrAddInteger(block_name,"emf_zmax",zamr.nlevels);
  }

  Initialize(pin);
  PrintCyclicZoomDiagnostics();

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void CyclicZoom::Initialize()
//! \brief Initialize CyclicZoom variables

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
    std::cout << "Funcs: add_emf = " << zemf.add_emf << std::endl;
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
    std::cout << "Efield: emf_fmax = " << zemf.emf_fmax << " emf_zmax = " << zemf.emf_zmax
              << std::endl;
    // print interval parameters
    std::cout << "Interval: t_run_fac = " << zint.t_run_fac
              << " t_run_pow = " << zint.t_run_pow
              << " t_run_max = " << zint.t_run_max
              << std::endl;
    // output zone-specific time factors
    for (int i = 0; i < zint.t_run_fac_zones.size(); ++i) {
      std::cout << " t_run_fac_zone_" << i << " = " << zint.t_run_fac_zones[i]
                << std::endl;
    }
    // print level structure
    std::cout << "Level: zone = " << zstate.zone << " direction = " << zstate.direction
              << " level = " << zamr.level << " max_level = " << zamr.max_level
              << " min_level = " << zamr.min_level << std::endl;
    // print region parameters
    std::cout << "Region: x1c = " << zregion.x1c << " x2c = " << zregion.x2c
              << " x3c = " << zregion.x3c << " r_0 = " << zregion.r_0
              << " radius = " << zregion.radius << std::endl;
    // print runtime information
    std::cout << "Time: runtime = " << zint.runtime << " next time = "
              << zstate.next_time << std::endl;
    std::cout << "======================================================" << std::endl;
  }
  return;
}
