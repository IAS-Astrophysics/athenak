//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file cyclic_zoom.cpp
//  \brief implementation of constructor and functions in CyclicZoom class

//TODO(@mhguo): check whether all includes are necessary
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
    pmesh(pm)
  {
  // cycle through ParameterInput list and read each <amr_criterion> block
  for (auto it = pin->block.begin(); it != pin->block.end(); ++it) {
    if (it->block_name.compare(0, 13, "amr_criterion") == 0) {
      std::string method = pin->GetString(it->block_name, "method");
      if (method.compare("cyclic_zoom") == 0) {
        block_name = it->block_name;
      }
    }
  }
  is_set = pin->GetOrAddBoolean(block_name,"is_set",false);
  read_rst = pin->GetOrAddBoolean(block_name,"read_rst",true);
  write_rst = pin->GetOrAddBoolean(block_name,"write_rst",true);
  zoom_bcs = pin->GetOrAddBoolean(block_name,"zoom_bcs",true);
  zoom_ref = pin->GetOrAddBoolean(block_name,"zoom_ref",true);
  zoom_dt = pin->GetOrAddBoolean(block_name,"zoom_dt",false);
  add_emf = pin->GetOrAddBoolean(block_name,"add_emf",true); // default true
  dump_diag  = pin->GetOrAddBoolean(block_name,"dump_diag",false);
  ndiag = pin->GetOrAddInteger(block_name,"ndiag",-1);

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
  zamr.first_emf = false;
  zamr.dump_rst = true;

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
  SetRegionAndInterval();
  old_zregion = zregion; // initialize old zoom region
  zstate.next_time += zint.runtime;
  // TODO(@mhguo): move to a new struct?
  nflux = 3;
  emf_flag = 0;
  emf_f0 = 1.0;
  emf_f1 = 0.0;
  emf_fmax = 1.0;
  re_fac = 0.8; // TODO(@mhguo): probably change to 1.0?
  r0_efld = 0.0;
  // Think whether to read emf parameters from input file
  if (add_emf) {
    emf_flag = pin->GetOrAddInteger(block_name,"emf_flag",emf_flag);
    emf_f0 = pin->GetOrAddReal(block_name,"emf_f0",emf_f0);
    emf_f1 = pin->GetOrAddReal(block_name,"emf_f1",emf_f1);
    emf_fmax = pin->GetOrAddReal(block_name,"emf_fmax",emf_fmax);
    emf_zmax = pin->GetOrAddInteger(block_name,"emf_zmax",zamr.nlevels);
    re_fac = pin->GetOrAddReal(block_name,"re_fac",re_fac);
    r0_efld = pin->GetOrAddReal(block_name,"r0_efld",0.0); // default value
  }
  // size_t free_mem, total_mem;
  // cudaMemGetInfo(&free_mem, &total_mem);
  // if (global_variable::my_rank == 0) {
  //   std::cout << "GPU memory: " << free_mem/1e9 << " GB free of " 
  //             << total_mem/1e9 << " GB total" << std::endl;
  // }

  Initialize(pin);
  PrintCyclicZoomDiagnostics();

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void CyclicZoom::Initialize()
//! \brief Initialize CyclicZoom variables

void CyclicZoom::Initialize(ParameterInput *pin)
{
  pzmesh = new ZoomMesh(this, pin);
  pzdata = new ZoomData(this, pin);
  // pzmr = new ZoomRefinement(this, pin);
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void CyclicZoom::Update(const bool restart)
//! \brief Update CyclicZoom runtime parameters after reading restart file

void CyclicZoom::Update(const bool restart)
{
  if (!restart) {
    return;
  }

  zamr.level = pmesh->max_level - zstate.zone;
  SetRegionAndInterval();
  // TODO(@mhguo): this should be wrong, need to think how to restart properly
  old_zregion = zregion; // initialize old zoom region

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void CyclicZoom::PrintCyclicZoomDiagnostics()
//! \brief Print CyclicZoom information

void CyclicZoom::PrintCyclicZoomDiagnostics()
{
  if (global_variable::my_rank == 0) {
    std::cout << "============== CyclicZoom Information ==============" << std::endl;
    // print basic parameters
    std::cout << "Basic: is_set = " << is_set << " read_rst = " << read_rst
              << " write_rst = " << write_rst << " ndiag = " << ndiag << std::endl;
    std::cout << "Funcs: zoom_bcs = " << zoom_bcs << " zoom_ref = " << zoom_ref 
              << " zoom_dt = " << zoom_dt << " add_emf = " << add_emf
              << " emf_flag = " << emf_flag << std::endl;
    // print model parameters
    std::cout << "Model: mzoom = " << pzmesh->nzmb_max_perdvce
              << " nvars = " << pzdata->nvars
              << " d_zoom = " << pzdata->d_zoom << " p_zoom = " << pzdata->p_zoom
              << std::endl;
    // print electric field parameters
    std::cout << "Efield: emf_f0 = " << emf_f0 << " emf_f1 = " << emf_f1
              << " emf_fmax = " << emf_fmax << " emf_zmax = " << emf_zmax
              << " re_fac = " << re_fac << " r0_efld = " << r0_efld << std::endl;
    // print interval parameters
    std::cout << "Interval: t_run_fac = " << zint.t_run_fac
              << " t_run_pow = " << zint.t_run_pow
              << " t_run_max = " << zint.t_run_max
              << std::endl;
    // output zone-specific time factors
    for (int i = 0; i < zint.t_run_fac_zones.size(); ++i) {
      std::cout << " t_run_fac_zone_" << i << " = " << zint.t_run_fac_zones[i] << std::endl;
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
    // throw warning
    if (zstate.zone != 0) {
      std::cout << "### WARNING! in " << __FILE__ << " at line " << __LINE__ << std::endl
                << "CyclicZoom zone is not zero, this is not expected" << std::endl;
    }
    std::cout << "==============================================" << std::endl;
  }
  return;
}
