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
#include "mesh/cyclic_zoom.hpp"

//----------------------------------------------------------------------------------------
// constructor, initializes data structures and parameters

CyclicZoom::CyclicZoom(Mesh *pm, ParameterInput *pin) :
    pmesh(pm),
    ndiag(-1)
  {
  is_set = pin->GetOrAddBoolean("cyclic_zoom","is_set",false);
  read_rst = pin->GetOrAddBoolean("cyclic_zoom","read_rst",true);
  write_rst = pin->GetOrAddBoolean("cyclic_zoom","write_rst",true);
  zoom_bcs = pin->GetOrAddBoolean("cyclic_zoom","zoom_bcs",true);
  zoom_ref = pin->GetOrAddBoolean("cyclic_zoom","zoom_ref",true);
  zoom_dt = pin->GetOrAddBoolean("cyclic_zoom","zoom_dt",false);
  fix_efield = pin->GetOrAddBoolean("cyclic_zoom","fix_efield",false);
  dump_diag  = pin->GetOrAddBoolean("cyclic_zoom","dump_diag",false);
  ndiag = pin->GetOrAddInteger("cyclic_zoom","ndiag",-1);

  // TODO(@mhguo): may set the parameters so that the initial level equals the max level
  // TODO(@mhguo): currently we need to check whether zamr.level is correct by hand
  zstate.id = 0;
  zstate.zone = pin->GetOrAddInteger("cyclic_zoom","zone",0);
  zstate.last_zone = zstate.zone;
  zstate.direction = pin->GetOrAddInteger("cyclic_zoom","direction",1);

  // Set zoom AMR parameters
  zamr.nlevels = pin->GetOrAddInteger("cyclic_zoom","nlevels",4);
  zamr.max_level = pmesh->max_level;
  zamr.min_level = zamr.max_level - zamr.nlevels + 1;
  zamr.level = zamr.max_level - zstate.zone;
  zamr.refine_flag = - zstate.direction;
  zamr.zooming_in = false;
  zamr.zooming_out = false;
  zamr.first_emf = false;
  zamr.dump_rst = true;

  zregion.x1c = pin->GetOrAddReal("cyclic_zoom","x1c",0.0);
  zregion.x2c = pin->GetOrAddReal("cyclic_zoom","x2c",0.0);
  zregion.x3c = pin->GetOrAddReal("cyclic_zoom","x3c",0.0);
  zregion.r_0 = pin->GetOrAddReal("cyclic_zoom","r_0",1.0);
  zint.t_run_fac = pin->GetOrAddReal("cyclic_zoom","t_run_fac",1.0);
  zint.t_run_pow = pin->GetOrAddReal("cyclic_zoom","t_run_pow",0.0);
  zint.t_run_max = pin->GetOrAddReal("cyclic_zoom","t_run_max",FLT_MAX);
  // Read number of zones from input parameters
  int num_zones = zamr.nlevels;
  // Initialize the dynamic interval structure
  zint.initialize(num_zones);
  // Read the runtime factors from input file
  for (int i = 0; i < num_zones; ++i) {
    std::string param_name = "t_run_fac_zone_" + std::to_string(i);
    Real default_value = 1.0;  // or some appropriate default
    Real zone_factor = pin->GetOrAddReal("cyclic_zoom", param_name.c_str(), default_value);
    zint.t_run_fac_zones[i] = zone_factor;
  }

  zstate.next_time = pmesh->time;
  SetRegionAndInterval();
  zstate.next_time += zint.runtime;

  nleaf = 2;
  if (pmesh->two_d) nleaf = 4;
  if (pmesh->three_d) nleaf = 8;
  mzoom = nleaf*zamr.nlevels;
  nvars = pin->GetOrAddInteger("cyclic_zoom","nvars",5);
  // TODO(@mhguo): move to a new struct?
  d_zoom = pin->GetOrAddReal("cyclic_zoom","d_zoom",(FLT_MIN));
  p_zoom = pin->GetOrAddReal("cyclic_zoom","p_zoom",(FLT_MIN));
  nflux = 3;
  emf_flag = 0;
  emf_f0 = 1.0;
  emf_f1 = 0.0;
  emf_fmax = 1.0;
  re_fac = 0.8; // TODO(@mhguo): probably change to 1.0?
  r0_efld = 0.0;
  if (fix_efield) {
    emf_flag = pin->GetInteger("cyclic_zoom","emf_flag");
    emf_f0 = pin->GetOrAddReal("cyclic_zoom","emf_f0",emf_f0);
    emf_f1 = pin->GetReal("cyclic_zoom","emf_f1");
    emf_fmax = pin->GetReal("cyclic_zoom","emf_fmax");
    emf_zmax = pin->GetOrAddInteger("cyclic_zoom","emf_zmax",zamr.nlevels);
    re_fac = pin->GetOrAddReal("cyclic_zoom","re_fac",re_fac);
    r0_efld = pin->GetOrAddReal("cyclic_zoom","r0_efld",0.0); // default value
  }
  // size_t free_mem, total_mem;
  // cudaMemGetInfo(&free_mem, &total_mem);
  // if (global_variable::my_rank == 0) {
  //   std::cout << "GPU memory: " << free_mem/1e9 << " GB free of " 
  //             << total_mem/1e9 << " GB total" << std::endl;
  // }

  Initialize();
  PrintInfo();

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void CyclicZoom::Initialize()
//! \brief Initialize CyclicZoom variables

void CyclicZoom::Initialize()
{
  pzdata = new ZoomData(this);
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

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void CyclicZoom::PrintInfo()
//! \brief Print CyclicZoom information

void CyclicZoom::PrintInfo()
{
  if (global_variable::my_rank == 0) {
    std::cout << "============== CyclicZoom Information ==============" << std::endl;
    // print basic parameters
    std::cout << "Basic: is_set = " << is_set << " read_rst = " << read_rst
              << " write_rst = " << write_rst << " ndiag = " << ndiag << std::endl;
    std::cout << "Funcs: zoom_bcs = " << zoom_bcs << " zoom_ref = " << zoom_ref 
              << " zoom_dt = " << zoom_dt << " fix_efield = " << fix_efield
              << " emf_flag = " << emf_flag << std::endl;
    // print model parameters
    std::cout << "Model: mzoom = " << mzoom << " nvars = " << nvars
              << " d_zoom = " << d_zoom
              << " p_zoom = " << p_zoom  << std::endl;
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
