// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file basetype_output.cpp
//  \brief implements BaseTypeOutput constructor, and LoadOutputData functions
//

#include <iostream>
#include <sstream>
#include <string>   // std::string, to_string()
#include <cstdio> // snprintf
#include <algorithm> // min_element
#include <utility> // pair<>
#include <vector>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "globals.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"
#include "coordinates/adm.hpp"
#include "z4c/tmunu.hpp"
#include "z4c/z4c.hpp"
#include "srcterms/srcterms.hpp"
#include "srcterms/turb_driver.hpp"
#include "outputs.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

//----------------------------------------------------------------------------------------
// BaseTypeOutput base class constructor
// Creates vector of output variable data

BaseTypeOutput::BaseTypeOutput(ParameterInput *pin, Mesh *pm, OutputParameters opar) :
    derived_var("derived-var",1,1,1,1,1),
    outarray("cc_outvar",1,1,1,1,1),
    outfield("fc_outvar",1,1,1,1),
    out_params(opar) {
  // exit for history, restart, or event log files
  if (out_params.file_type.compare("hst") == 0 ||
      out_params.file_type.compare("rst") == 0 ||
      out_params.file_type.compare("log") == 0 ||
      out_params.file_type.compare("trk") == 0) {return;}

  // initialize vector containing number of output MBs per rank
  noutmbs.assign(global_variable::nranks, 0);

  // check for valid choice of variables
  int ivar = -1;
  for (int i=0; i<(NOUTPUT_CHOICES); ++i) {
    if (out_params.variable.compare(var_choice[i]) == 0) {ivar = i;}
  }
  if (ivar < 0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
       << "Variable '" << out_params.variable << "' in block '" << out_params.block_name
       << "' in input file is not a valid choice" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // check that appropriate physics is defined for requested output variable
  // TODO(@user): Index limits of variable choices below may change if more choices added
  if ((ivar<16) && (pm->pmb_pack->phydro == nullptr)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
       << "Output of Hydro variable requested in <output> block '"
       << out_params.block_name << "' but no Hydro object has been constructed."
       << std::endl << "Input file is likely missing a <hydro> block" << std::endl;
    exit(EXIT_FAILURE);
  }
  if ((ivar>=16) && (ivar<50) && (pm->pmb_pack->pmhd == nullptr)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
       << "Output of MHD variable requested in <output> block '"
       << out_params.block_name << "' but no MHD object has been constructed."
       << std::endl << "Input file is likely missing a <mhd> block" << std::endl;
    exit(EXIT_FAILURE);
  }
  if ((ivar==38) && (pm->pmb_pack->pdyngr == nullptr)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
       << "Output of DynMHD variable requested in <output> block '"
       << out_params.block_name << "' but no DynMHD object has been constructed."
       << std::endl << "Input file is likely missing a <adm> or <z4c>, and/or <mhd> block"
       << std::endl;
    exit(EXIT_FAILURE);
  }
  if ((ivar==50) && (pm->pmb_pack->pturb == nullptr)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
       << "Output of Force variable requested in <output> block '"
       << out_params.block_name << "' but no Force object has been constructed."
       << std::endl << "Input file is likely missing a <forcing> block" << std::endl;
    exit(EXIT_FAILURE);
  }
  if (ivar==51 && (pm->pmb_pack->prad == nullptr)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
       << "Output of Radiation moments requested in <output> block '"
       << out_params.block_name << "' but no Radiation object has been constructed."
       << std::endl << "Input file is likely missing a <radiation> block" << std::endl;
    exit(EXIT_FAILURE);
  }
  if ((ivar==52 || ivar==53) &&
      ((pm->pmb_pack->prad == nullptr) ||
       (pm->pmb_pack->phydro == nullptr && pm->pmb_pack->pmhd == nullptr))) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
       << "Output of Fluid Frame Radiation moments requested in <output> block '"
       << out_params.block_name << "' but either Radiation object has not been "
       << " constructed, or corresponding Hydro or MHD object missing" << std::endl;
    exit(EXIT_FAILURE);
  }
  if ((ivar>=53) && (ivar<68) &&
      (pm->pmb_pack->prad == nullptr || pm->pmb_pack->phydro == nullptr)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
       << "Output of Radiation Hydro variables requested in <output> block '"
       << out_params.block_name << "' but Radiation and/or Hydro object not constructed."
       << std::endl << "Input file is likely missing corresponding block" << std::endl;
    exit(EXIT_FAILURE);
  }
  if ((ivar>=68) && (ivar<88) &&
      (pm->pmb_pack->prad == nullptr || pm->pmb_pack->pmhd == nullptr)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
       << "Output of Radiation MHD variables requested in <output> block '"
       << out_params.block_name << "' but Radiation and/or MHD object not constructed."
       << std::endl << "Input file is likely missing corresponding block" << std::endl;
    exit(EXIT_FAILURE);
  }
  if ((ivar>=88) && (ivar<106) && (pm->pmb_pack->padm == nullptr)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
       << "Output of ADM variable requested in <output> block '"
       << out_params.block_name << "' but ADM object not constructed."
       << std::endl << "Input file is likely missing corresponding block" << std::endl;
    exit(EXIT_FAILURE);
  }
  if ((ivar>=106) && (ivar<129) && (pm->pmb_pack->pz4c == nullptr)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
       << "Output of Z4c variable requested in <output> block '"
       << out_params.block_name << "' but Z4c object not constructed."
       << std::endl << "Input file is likely missing corresponding block" << std::endl;
    exit(EXIT_FAILURE);
  }
  if ((ivar>=129) && (ivar<132) && (pm->pmb_pack->pz4c == nullptr)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
       << "Output of weyl variable requested in <output> block '"
       << out_params.block_name << "' but weyl object not constructed."
       << std::endl << "Input file is likely missing corresponding block" << std::endl;
    exit(EXIT_FAILURE);
  }
  if ((ivar>=132) && (ivar<140) && (pm->pmb_pack->pz4c == nullptr)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
       << "Output of constraint variables request in <output> block '"
       << out_params.block_name << "' but Z4c object not constructed."
       << std::endl << "Input file is likely missing corresponding block" << std::endl;
    exit(EXIT_FAILURE);
  }
  if ((ivar>=140) && (ivar<151) && (pm->pmb_pack->ptmunu == nullptr)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
       << "Output of Tmunu variable requested in <output> block '"
       << out_params.block_name << "' but no Tmunu object has been constructed."
       << std::endl << "Input file is likely missing a <adm> block" << std::endl;
  }
  if ((ivar>=151) && (ivar<153) && (pm->pmb_pack->ppart == nullptr)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
       << "Output of particles requested in <output> block '"
       << out_params.block_name << "' but particle object not constructed."
       << std::endl << "Input file is likely missing corresponding block" << std::endl;
    exit(EXIT_FAILURE);
  }

  // Now load STL vector of output variables
  outvars.clear();

  // make a vector of out_params.variables
  std::vector<std::string> variables;

  variables.push_back(out_params.variable);
  if (out_params.file_type == "pdf") {
    if (out_params.nbin2 > 1) {
      variables.push_back(out_params.variable_2);
    }
  }


  for (const auto& variable : variables) {
    // hydro (lab-frame) density
    if (variable.compare("hydro_u_d") == 0 ||
        variable.compare("hydro_u") == 0 ||
        variable.compare("rad_hydro_u_d") == 0 ||
        variable.compare("rad_hydro_u") == 0) {
      outvars.emplace_back("dens",0,&(pm->pmb_pack->phydro->u0));
    }

    // hydro (rest-frame) density
    if (variable.compare("hydro_w_d") == 0 ||
        variable.compare("hydro_w") == 0 ||
        variable.compare("rad_hydro_w_d") == 0 ||
        variable.compare("rad_hydro_w") == 0) {
      outvars.emplace_back("dens",0,&(pm->pmb_pack->phydro->w0));
    }

    // hydro components of momentum
    if (variable.compare("hydro_u_m1") == 0 ||
        variable.compare("hydro_u") == 0 ||
        variable.compare("rad_hydro_u_m1") == 0 ||
        variable.compare("rad_hydro_u") == 0) {
      outvars.emplace_back("mom1",1,&(pm->pmb_pack->phydro->u0));
    }
    if (variable.compare("hydro_u_m2") == 0 ||
        variable.compare("hydro_u") == 0 ||
        variable.compare("rad_hydro_u_m2") == 0 ||
        variable.compare("rad_hydro_u") == 0) {
      outvars.emplace_back("mom2",2,&(pm->pmb_pack->phydro->u0));
    }
    if (variable.compare("hydro_u_m3") == 0 ||
        variable.compare("hydro_u") == 0 ||
        variable.compare("rad_hydro_u_m3") == 0 ||
        variable.compare("rad_hydro_u") == 0) {
      outvars.emplace_back("mom3",3,&(pm->pmb_pack->phydro->u0));
    }

    // hydro components of velocity
    if (variable.compare("hydro_w_vx") == 0 ||
        variable.compare("hydro_w") == 0 ||
        variable.compare("rad_hydro_w_vx") == 0 ||
        variable.compare("rad_hydro_w") == 0) {
      outvars.emplace_back("velx",1,&(pm->pmb_pack->phydro->w0));
    }
    if (variable.compare("hydro_w_vy") == 0 ||
        variable.compare("hydro_w") == 0 ||
        variable.compare("rad_hydro_w_vy") == 0 ||
        variable.compare("rad_hydro_w") == 0) {
      outvars.emplace_back("vely",2,&(pm->pmb_pack->phydro->w0));
    }
    if (variable.compare("hydro_w_vz") == 0 ||
        variable.compare("hydro_w") == 0 ||
        variable.compare("rad_hydro_w_vz") == 0 ||
        variable.compare("rad_hydro_w") == 0) {
      outvars.emplace_back("velz",3,&(pm->pmb_pack->phydro->w0));
    }

    // hydro total energy
    if (variable.compare("hydro_u_e") == 0 ||
        variable.compare("hydro_u") == 0 ||
        variable.compare("rad_hydro_u_e") == 0 ||
        variable.compare("rad_hydro_u") == 0) {
      if (pm->pmb_pack->phydro->peos->eos_data.is_ideal) {
        outvars.emplace_back("ener",4,&(pm->pmb_pack->phydro->u0));
      }
    }

    // hydro internal energy or temperature
    if (variable.compare("hydro_w_e") == 0 ||
        variable.compare("hydro_w") == 0 ||
        variable.compare("rad_hydro_w_e") == 0 ||
        variable.compare("rad_hydro_w") == 0) {
      if (pm->pmb_pack->phydro->peos->eos_data.is_ideal) {
        if (pm->pmb_pack->pdyngr != nullptr) {
          outvars.emplace_back("press",4,&(pm->pmb_pack->phydro->w0));
        } else {
          outvars.emplace_back("eint",4,&(pm->pmb_pack->phydro->w0));
        }
      }
    }

    // hydro passive scalars mass densities (s*d)
    if (variable.compare("hydro_u_s") == 0 ||
        variable.compare("hydro_u") == 0 ||
        variable.compare("rad_hydro_u_s") == 0 ||
        variable.compare("rad_hydro_u") == 0) {
      int nhyd = pm->pmb_pack->phydro->nhydro;
      int nvars = nhyd + pm->pmb_pack->phydro->nscalars;
      for (int n=nhyd; n<nvars; ++n) {
        char number[3];
        std::snprintf(number,sizeof(number),"%02d",(n - nhyd)%100);
        std::string vname;
        vname.assign("r_");
        vname.append(number);
        outvars.emplace_back(vname,n,&(pm->pmb_pack->phydro->u0));
      }
    }

    // hydro passive scalars (s)
    if (variable.compare("hydro_w_s") == 0 ||
        variable.compare("hydro_w") == 0 ||
        variable.compare("rad_hydro_w_s") == 0 ||
        variable.compare("rad_hydro_w") == 0) {
      int nhyd = pm->pmb_pack->phydro->nhydro;
      int nvars = nhyd + pm->pmb_pack->phydro->nscalars;
      for (int n=nhyd; n<nvars; ++n) {
        char number[3];
        std::snprintf(number,sizeof(number),"%02d",(n - nhyd)%100);
        std::string vname;
        vname.assign("s_");
        vname.append(number);
        outvars.emplace_back(vname,n,&(pm->pmb_pack->phydro->w0));
      }
    }

    // mhd (lab-frame) density
    if (variable.compare("mhd_u_d") == 0 ||
        variable.compare("mhd_u") == 0 ||
        variable.compare("mhd_u_bcc") == 0 ||
        variable.compare("rad_mhd_u_d") == 0 ||
        variable.compare("rad_mhd_u") == 0 ||
        variable.compare("rad_mhd_u_bcc") == 0) {
      outvars.emplace_back("dens",0,&(pm->pmb_pack->pmhd->u0));
    }

    // mhd (rest-frame) density
    if (variable.compare("mhd_w_d") == 0 ||
        variable.compare("mhd_w") == 0 ||
        variable.compare("mhd_w_bcc") == 0 ||
        variable.compare("rad_mhd_w_d") == 0 ||
        variable.compare("rad_mhd_w") == 0 ||
        variable.compare("rad_mhd_w_bcc") == 0) {
      outvars.emplace_back("dens",0,&(pm->pmb_pack->pmhd->w0));
    }

    // mhd components of momentum
    if (variable.compare("mhd_u_m1") == 0 ||
        variable.compare("mhd_u") == 0 ||
        variable.compare("mhd_u_bcc") == 0 ||
        variable.compare("rad_mhd_u_m1") == 0 ||
        variable.compare("rad_mhd_u") == 0 ||
        variable.compare("rad_mhd_u_bcc") == 0) {
      outvars.emplace_back("mom1",1,&(pm->pmb_pack->pmhd->u0));
    }
    if (variable.compare("mhd_u_m2") == 0 ||
        variable.compare("mhd_u") == 0 ||
        variable.compare("mhd_u_bcc") == 0 ||
        variable.compare("rad_mhd_u_m2") == 0 ||
        variable.compare("rad_mhd_u") == 0 ||
        variable.compare("rad_mhd_u_bcc") == 0) {
      outvars.emplace_back("mom2",2,&(pm->pmb_pack->pmhd->u0));
    }
    if (variable.compare("mhd_u_m3") == 0 ||
        variable.compare("mhd_u") == 0 ||
        variable.compare("mhd_u_bcc") == 0 ||
        variable.compare("rad_mhd_u_m3") == 0 ||
        variable.compare("rad_mhd_u") == 0 ||
        variable.compare("rad_mhd_u_bcc") == 0) {
      outvars.emplace_back("mom3",3,&(pm->pmb_pack->pmhd->u0));
    }

    // mhd components of velocity
    if (variable.compare("mhd_w_vx") == 0 ||
        variable.compare("mhd_w") == 0 ||
        variable.compare("mhd_w_bcc") == 0 ||
        variable.compare("rad_mhd_w_vx") == 0 ||
        variable.compare("rad_mhd_w") == 0 ||
        variable.compare("rad_mhd_w_bcc") == 0) {
      outvars.emplace_back("velx",1,&(pm->pmb_pack->pmhd->w0));
    }
    if (variable.compare("mhd_w_vy") == 0 ||
        variable.compare("mhd_w") == 0 ||
        variable.compare("mhd_w_bcc") == 0 ||
        variable.compare("rad_mhd_w_vy") == 0 ||
        variable.compare("rad_mhd_w") == 0 ||
        variable.compare("rad_mhd_w_bcc") == 0) {
      outvars.emplace_back("vely",2,&(pm->pmb_pack->pmhd->w0));
    }
    if (variable.compare("mhd_w_vz") == 0 ||
        variable.compare("mhd_w") == 0 ||
        variable.compare("mhd_w_bcc") == 0 ||
        variable.compare("rad_mhd_w_vz") == 0 ||
        variable.compare("rad_mhd_w") == 0 ||
        variable.compare("rad_mhd_w_bcc") == 0) {
      outvars.emplace_back("velz",3,&(pm->pmb_pack->pmhd->w0));
    }

    // mhd total energy
    if (variable.compare("mhd_u_e") == 0 ||
        variable.compare("mhd_u") == 0 ||
        variable.compare("mhd_u_bcc") == 0 ||
        variable.compare("rad_mhd_u_e") == 0 ||
        variable.compare("rad_mhd_u") == 0 ||
        variable.compare("rad_mhd_u_bcc") == 0) {
      if (pm->pmb_pack->pmhd->peos->eos_data.is_ideal) {
        outvars.emplace_back("ener",4,&(pm->pmb_pack->pmhd->u0));
      }
    }

    // mhd internal energy or temperature
    if (variable.compare("mhd_w_e") == 0 ||
        variable.compare("mhd_w") == 0 ||
        variable.compare("mhd_w_bcc") == 0 ||
        variable.compare("rad_mhd_w_e") == 0 ||
        variable.compare("rad_mhd_w") == 0 ||
        variable.compare("rad_mhd_w_bcc") == 0) {
      if (pm->pmb_pack->pmhd->peos->eos_data.is_ideal) {
        if (pm->pmb_pack->pdyngr != nullptr) {
          outvars.emplace_back("press",4,&(pm->pmb_pack->pmhd->w0));
        } else {
          outvars.emplace_back("eint",4,&(pm->pmb_pack->pmhd->w0));
        }
      }
    }

    // mhd passive scalars mass densities (s*d)
    if (variable.compare("mhd_u_s") == 0 ||
        variable.compare("mhd_u") == 0 ||
        variable.compare("mhd_u_bcc") == 0 ||
        variable.compare("rad_mhd_u_s") == 0 ||
        variable.compare("rad_mhd_u") == 0 ||
        variable.compare("rad_mhd_u_bcc") == 0) {
      int nmhd = pm->pmb_pack->pmhd->nmhd;
      int nvars = nmhd + pm->pmb_pack->pmhd->nscalars;
      for (int n=nmhd; n<nvars; ++n) {
        char number[3];
        std::snprintf(number,sizeof(number),"%02d",(n - nmhd)%100);
        std::string vname;
        vname.assign("r_");
        vname.append(number);
        outvars.emplace_back(vname,n,&(pm->pmb_pack->pmhd->u0));
      }
    }

    // mhd passive scalars (s)
    if (variable.compare("mhd_w_s") == 0 ||
        variable.compare("mhd_w") == 0 ||
        variable.compare("mhd_w_bcc") == 0 ||
        variable.compare("rad_mhd_w_s") == 0 ||
        variable.compare("rad_mhd_w") == 0 ||
        variable.compare("rad_mhd_w_bcc") == 0) {
      int nmhd = pm->pmb_pack->pmhd->nmhd;
      int nvars = nmhd + pm->pmb_pack->pmhd->nscalars;
      for (int n=nmhd; n<nvars; ++n) {
        char number[3];
        std::snprintf(number,sizeof(number),"%02d",(n - nmhd)%100);
        std::string vname;
        vname.assign("s_");
        vname.append(number);
        outvars.emplace_back(vname,n,&(pm->pmb_pack->pmhd->w0));
      }
    }

    // mhd cell-centered magnetic fields
    if (variable.compare("mhd_bcc1") == 0 ||
        variable.compare("mhd_bcc") == 0 ||
        variable.compare("mhd_u_bcc") == 0 ||
        variable.compare("mhd_w_bcc") == 0 ||
        variable.compare("rad_mhd_bcc1") == 0 ||
        variable.compare("rad_mhd_bcc") == 0 ||
        variable.compare("rad_mhd_u_bcc") == 0 ||
        variable.compare("rad_mhd_w_bcc") == 0) {
      outvars.emplace_back("bcc1",0,&(pm->pmb_pack->pmhd->bcc0));
    }
    if (variable.compare("mhd_bcc2") == 0 ||
        variable.compare("mhd_bcc") == 0 ||
        variable.compare("mhd_u_bcc") == 0 ||
        variable.compare("mhd_w_bcc") == 0 ||
        variable.compare("rad_mhd_bcc2") == 0 ||
        variable.compare("rad_mhd_bcc") == 0 ||
        variable.compare("rad_mhd_u_bcc") == 0 ||
        variable.compare("rad_mhd_w_bcc") == 0) {
      outvars.emplace_back("bcc2",1,&(pm->pmb_pack->pmhd->bcc0));
    }
    if (variable.compare("mhd_bcc3") == 0 ||
        variable.compare("mhd_bcc") == 0 ||
        variable.compare("mhd_u_bcc") == 0 ||
        variable.compare("mhd_w_bcc") == 0 ||
        variable.compare("rad_mhd_bcc3") == 0 ||
        variable.compare("rad_mhd_bcc") == 0 ||
        variable.compare("rad_mhd_u_bcc") == 0 ||
        variable.compare("rad_mhd_w_bcc") == 0) {
      outvars.emplace_back("bcc3",2,&(pm->pmb_pack->pmhd->bcc0));
    }

    // MHD temperature
    if (variable.compare("mhd_t") == 0 ||
        ((variable.compare("mhd_w") == 0 ||
          variable.compare("mhd_w_bcc") == 0) && pm->pmb_pack->pdyngr !=nullptr)) {
      outvars.emplace_back("temperature",0,&(pm->pmb_pack->pdyngr->temperature));
    }

    // hydro/mhd z-component of vorticity (useful in 2D)
    if (variable.compare("hydro_wz") == 0 ||
        variable.compare("mhd_wz") == 0) {
      out_params.contains_derived = true;
      out_params.n_derived += 1;
      int i_derived = out_params.n_derived - 1;
      outvars.emplace_back("vorz",i_derived,&(derived_var));
    }

    // hydro/mhd magnitude of vorticity (useful in 3D)
    if (variable.compare("hydro_w2") == 0 ||
        variable.compare("mhd_w2") == 0) {
      out_params.contains_derived = true;
      out_params.n_derived += 1;
      int i_derived = out_params.n_derived - 1;
      outvars.emplace_back("vor2",i_derived,&(derived_var));
    }

    // mhd z-component of current density (useful in 2D)
    if (variable.compare("mhd_jz") == 0) {
      out_params.contains_derived = true;
      out_params.n_derived += 1;
      int i_derived = out_params.n_derived - 1;
      outvars.emplace_back("jz",i_derived,&(derived_var));
    }

    // mhd magnitude of current density (useful in 3D)
    if (variable.compare("mhd_j2") == 0) {
      out_params.contains_derived = true;
      out_params.n_derived += 1;
      int i_derived = out_params.n_derived - 1;
      outvars.emplace_back("j2",i_derived,&(derived_var));
    }

    // Added by DBF --- check & update NOUTPUT_CHOICES
    // mhd magnitude of magnetic curvature
    if (variable.compare("mhd_curv") == 0) {
      out_params.contains_derived = true;
      out_params.n_derived += 1;
      int i_derived = out_params.n_derived - 1;
      outvars.emplace_back("curv",i_derived,&(derived_var));
    }

    // mhd magnitude of magnetic curvature
    if (variable.compare("mhd_k_jxb") == 0) {
      out_params.contains_derived = true;
      out_params.n_derived += 1;
      int i_derived = out_params.n_derived - 1;
      outvars.emplace_back("k_jxb",i_derived,&(derived_var));
    }

    // mhd magnitude of magnetic curvature
    if (variable.compare("mhd_curv_perp") == 0) {
      out_params.contains_derived = true;
      out_params.n_derived += 1;
      int i_derived = out_params.n_derived - 1;
      outvars.emplace_back("curv_perp",i_derived,&(derived_var));
    }

    // mhd magnitude of magnetic curvature
    if (variable.compare("mhd_bmag") == 0) {
      out_params.contains_derived = true;
      out_params.n_derived += 1;
      int i_derived = out_params.n_derived - 1;
      outvars.emplace_back("bmag",i_derived,&(derived_var));
    }

    // mhd divergence of B
    if (variable.compare("mhd_divb") == 0) {
      out_params.contains_derived = true;
      out_params.n_derived += 1;
      int i_derived = out_params.n_derived - 1;
      outvars.emplace_back("divb",i_derived,&(derived_var));
    }

    // added by GNW --- contravariant components of magnetic field
    if (out_params.variable.compare("mhd_jcon") == 0) {
      pm->pmb_pack->pmhd->SetSaveWBcc();
      out_params.contains_derived = true;
      out_params.n_derived += 4;
      outvars.emplace_back("jcon0",0,&(derived_var));
      outvars.emplace_back("jcon1",1,&(derived_var));
      outvars.emplace_back("jcon2",2,&(derived_var));
      outvars.emplace_back("jcon3",3,&(derived_var));
    }

    // Hydro SGS tensor
    if (variable.compare("hydro_sgs") == 0) {
      out_params.contains_derived = true;
      // emplace all 23 components of the SGS tensor
      for (int i=0; i<23; ++i) {
          std::string variable_name;
          variable_name.assign("hydro_sgs_");
          variable_name.append(std::to_string(i+1));
          out_params.n_derived += 1;
          outvars.emplace_back(variable_name,i,&(derived_var));
      }
    }

    // Mhd SGS tensor
    if (variable.compare("mhd_sgs") == 0) {
      out_params.contains_derived = true;
      // emplace all 59 components of the SGS tensor
      for (int i=0; i<59; ++i) {
          std::string variable_name;
          variable_name.assign("mhd_sgs_");
          variable_name.append(std::to_string(i+1));
          out_params.n_derived += 1;
          outvars.emplace_back(variable_name.c_str(),i,&(derived_var));
      }
    }

    // mhd_dynamo_ks
    if (variable.compare("mhd_dynamo_ks") == 0) {
      out_params.contains_derived = true;
      // emplace all 8 components of the SGS tensor
      outvars.emplace_back("mhd_dynamo_B^2",0,&(derived_var));
      outvars.emplace_back("mhd_dynamo_B^4",1,&(derived_var));
      outvars.emplace_back("mhd_dynamo_dB^2",2,&(derived_var));
      outvars.emplace_back("mhd_dynamo_BdB^2",3,&(derived_var));
      outvars.emplace_back("mhd_dynamo_|BxJ|^2",4,&(derived_var));
      outvars.emplace_back("mhd_dynamo_|B.J|^2",5,&(derived_var));
      outvars.emplace_back("mhd_dynamo_U^2",6,&(derived_var));
      outvars.emplace_back("mhd_dynamo_dU",7,&(derived_var));
      out_params.n_derived += 8;
    }

    // turbulent forcing
    if (variable.compare("turb_force") == 0) {
      outvars.emplace_back("force1",0,&(pm->pmb_pack->pturb->force));
      outvars.emplace_back("force2",1,&(pm->pmb_pack->pturb->force));
      outvars.emplace_back("force3",2,&(pm->pmb_pack->pturb->force));
    }

    // ADM variables, excluding gauge
    for (int v = 0; v < adm::ADM::nadm - 4; ++v) {
      if (variable.compare("adm") == 0 ||
          variable.compare(adm::ADM::ADM_names[v]) == 0) {
        outvars.emplace_back(adm::ADM::ADM_names[v], v, &(pm->pmb_pack->padm->u_adm));
      }
    }

    // ADM gauge variables
    if (nullptr == pm->pmb_pack->pz4c) {
      for (int v = adm::ADM::nadm - 4; v < adm::ADM::nadm; ++v) {
        if (variable.compare("adm") == 0 ||
            variable.compare(adm::ADM::ADM_names[v]) == 0) {
          outvars.emplace_back(adm::ADM::ADM_names[v], v, &(pm->pmb_pack->padm->u_adm));
        }
      }
    }

    // mat z4c variables
    for (int v = 0; v < Tmunu::N_Tmunu; ++v) {
      if (variable.compare("tmunu") == 0 ||
          variable.compare(Tmunu::Tmunu_names[v]) == 0) {
        outvars.emplace_back(Tmunu::Tmunu_names[v], v, &(pm->pmb_pack->ptmunu->u_tmunu));
      }
    }
    // con z4c variables
    for (int v = 0; v < z4c::Z4c::ncon; ++v) {
      if (variable.compare("con") == 0 ||
          variable.compare(z4c::Z4c::Constraint_names[v]) == 0) {
        outvars.emplace_back(z4c::Z4c::Constraint_names[v], v,
        &(pm->pmb_pack->pz4c->u_con));
      }
    }

    // z4c variables
    for (int v = 0; v < z4c::Z4c::nz4c; ++v) {
      if (variable.compare("z4c") == 0 ||
          variable.compare(z4c::Z4c::Z4c_names[v]) == 0) {
        outvars.emplace_back(z4c::Z4c::Z4c_names[v], v, &(pm->pmb_pack->pz4c->u0));
      }
    }

    // weyl scalars
    if (variable.compare("weyl") == 0) {
      outvars.emplace_back("weyl_rpsi4",0,&(pm->pmb_pack->pz4c->u_weyl));
      outvars.emplace_back("weyl_ipsi4",1,&(pm->pmb_pack->pz4c->u_weyl));
    }

    // radiation moments in coordinate frame
    if (variable.compare(0, 9, "rad_coord") == 0 ||
        variable.compare(0, 9, "rad_hydro") == 0 ||
        variable.compare(0, 7, "rad_mhd") == 0) {
      out_params.contains_derived = true;
      out_params.n_derived += 10;
      outvars.emplace_back("r00",0,&(derived_var));
      outvars.emplace_back("r01",1,&(derived_var));
      outvars.emplace_back("r02",2,&(derived_var));
      outvars.emplace_back("r03",3,&(derived_var));
      outvars.emplace_back("r11",4,&(derived_var));
      outvars.emplace_back("r12",5,&(derived_var));
      outvars.emplace_back("r13",6,&(derived_var));
      outvars.emplace_back("r22",7,&(derived_var));
      outvars.emplace_back("r23",8,&(derived_var));
      outvars.emplace_back("r33",9,&(derived_var));
    }

    // radiation moments in fluid frame
    if (variable.compare("rad_fluid") == 0 ||
        variable.compare("rad_coord_fluid") == 0 ||
        variable.compare(0, 9, "rad_hydro") == 0 ||
        variable.compare(0, 7, "rad_mhd") == 0) {
      bool needs_fluid_only = (variable.compare("rad_fluid") == 0);
      int moments_offset = !(needs_fluid_only) ? 10 : 0;
      out_params.contains_derived = true;
      out_params.n_derived += 10;
      outvars.emplace_back("r00_ff",moments_offset+0,&(derived_var));
      outvars.emplace_back("r01_ff",moments_offset+1,&(derived_var));
      outvars.emplace_back("r02_ff",moments_offset+2,&(derived_var));
      outvars.emplace_back("r03_ff",moments_offset+3,&(derived_var));
      outvars.emplace_back("r11_ff",moments_offset+4,&(derived_var));
      outvars.emplace_back("r12_ff",moments_offset+5,&(derived_var));
      outvars.emplace_back("r13_ff",moments_offset+6,&(derived_var));
      outvars.emplace_back("r22_ff",moments_offset+7,&(derived_var));
      outvars.emplace_back("r23_ff",moments_offset+8,&(derived_var));
      outvars.emplace_back("r33_ff",moments_offset+9,&(derived_var));
    }
  }

  // particle density binned to mesh
  if (out_params.variable.compare("prtcl_d") == 0) {
    out_params.contains_derived = true;
    out_params.n_derived += 1;
    outvars.emplace_back("pdens",0,&(derived_var));
  }

  // initialize vector containing number of output MBs per rank
  noutmbs.assign(global_variable::nranks, 0);
}

//----------------------------------------------------------------------------------------
// BaseTypeOutput::LoadOutputData()
// create std::vector of HostArray3Ds containing data specified in <output> block for
// this output type

void BaseTypeOutput::LoadOutputData(Mesh *pm) {
  // out_data_ vector (indexed over # of output MBs) stores 4D array of variables
  // so start iteration over number of MeshBlocks
  // TODO(@user): get this working for multiple physics, which may be either defined/undef

  // With AMR, number and location of output MBs can change between output times.
  // So start with clean vector of output MeshBlock info, and re-compute
  outmbs.clear();

  // loop over all MeshBlocks
  // set size & starting indices of output arrays, adjusted accordingly if gz included
  auto &indcs = pm->mb_indcs;
  auto &size  = pm->pmb_pack->pmb->mb_size;
  auto &gids  = pm->pmb_pack->gids;
  for (int m=0; m<(pm->pmb_pack->nmb_thispack); ++m) {
    // skip if MeshBlock ID is specified and not equal to this ID
    if (out_params.gid >= 0 && (m+gids) != out_params.gid) { continue; }

    int ois,oie,ojs,oje,oks,oke;

    if (out_params.include_gzs) {
      int nout1 = indcs.nx1 + 2*(indcs.ng);
      int nout2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
      int nout3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;
      ois = 0; oie = nout1-1;
      ojs = 0; oje = nout2-1;
      oks = 0; oke = nout3-1;
    } else {
      ois = indcs.is; oie = indcs.ie;
      ojs = indcs.js; oje = indcs.je;
      oks = indcs.ks; oke = indcs.ke;
    }

    // check for slicing in each dimension, adjust start/end indices accordingly
    if (out_params.slice1) {
      // skip this MB if slice is out of range
      if (out_params.slice_x1 <  size.h_view(m).x1min ||
          out_params.slice_x1 >= size.h_view(m).x1max) { continue; }
      // set index of slice
      ois = CellCenterIndex(out_params.slice_x1, indcs.nx1,
                            size.h_view(m).x1min, size.h_view(m).x1max);
      ois += indcs.is;
      oie = ois;
    }

    if (out_params.slice2) {
      // skip this MB if slice is out of range
      if (out_params.slice_x2 <  size.h_view(m).x2min ||
          out_params.slice_x2 >= size.h_view(m).x2max) { continue; }
      // set index of slice
      ojs = CellCenterIndex(out_params.slice_x2, indcs.nx2,
                            size.h_view(m).x2min, size.h_view(m).x2max);
      ojs += indcs.js;
      oje = ojs;
    }

    if (out_params.slice3) {
      // skip this MB if slice is out of range
      if (out_params.slice_x3 <  size.h_view(m).x3min ||
          out_params.slice_x3 >= size.h_view(m).x3max) { continue; }
      // set index of slice
      oks = CellCenterIndex(out_params.slice_x3, indcs.nx3,
                            size.h_view(m).x3min, size.h_view(m).x3max);
      oks += indcs.ks;
      oke = oks;
    }

    // set coordinate geometry information for MB
    Real x1min = size.h_view(m).x1min;
    Real x1max = size.h_view(m).x1max;
    Real x2min = size.h_view(m).x2min;
    Real x2max = size.h_view(m).x2max;
    Real x3min = size.h_view(m).x3min;
    Real x3max = size.h_view(m).x3max;

    int id = pm->pmb_pack->pmb->mb_gid.h_view(m);
    outmbs.emplace_back(id,ois,oie,ojs,oje,oks,oke,x1min,x1max,x2min,x2max,x3min,x3max);
  }

  std::fill(noutmbs.begin(), noutmbs.end(), 0);
  noutmbs[global_variable::my_rank] = outmbs.size();
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, noutmbs.data(), global_variable::nranks,
                MPI_INT, MPI_SUM, MPI_COMM_WORLD);
#endif
  noutmbs_min = *std::min_element(noutmbs.begin(), noutmbs.end());
  noutmbs_max = *std::max_element(noutmbs.begin(), noutmbs.end());


  // get number of output vars and MBs, then realloc outarray (HostArray)
  int nout_vars = outvars.size();
  int nout_mbs = outmbs.size();
  // note that while ois,oie,etc. can be different on each MB, the number of cells output
  // on each MeshBlock, i.e. (ois-ois+1), etc. is the same.
  if (nout_mbs > 0) {
    int nout1 = (outmbs[0].oie - outmbs[0].ois + 1);
    int nout2 = (outmbs[0].oje - outmbs[0].ojs + 1);
    int nout3 = (outmbs[0].oke - outmbs[0].oks + 1);
    // NB: outarray stores all output data on Host
    Kokkos::realloc(outarray, nout_vars, nout_mbs, nout3, nout2, nout1);
  }

  // Calculate derived variables, if required
  if (out_params.contains_derived) {
    ComputeDerivedVariable(out_params.variable, pm);
  }

  // Now copy data to host (outarray) over all variables and MeshBlocks
  for (int n=0; n<nout_vars; ++n) {
    for (int m=0; m<nout_mbs; ++m) {
      int mbi = pm->FindMeshBlockIndex(outmbs[m].mb_gid);
      std::pair<int,int> irange = std::make_pair(outmbs[m].ois, outmbs[m].oie+1);
      std::pair<int,int> jrange = std::make_pair(outmbs[m].ojs, outmbs[m].oje+1);
      std::pair<int,int> krange = std::make_pair(outmbs[m].oks, outmbs[m].oke+1);
      int nout1 = (outmbs[0].oie - outmbs[0].ois + 1);
      int nout2 = (outmbs[0].oje - outmbs[0].ojs + 1);
      int nout3 = (outmbs[0].oke - outmbs[0].oks + 1);

      // copy output variable to new device View
      DvceArray3D<Real> d_output_var("d_out_var",nout3,nout2,nout1);
      auto d_slice = Kokkos::subview(*(outvars[n].data_ptr), mbi, outvars[n].data_index,
                                     krange,jrange,irange);
      Kokkos::deep_copy(d_output_var,d_slice);

      // copy new device View to host mirror View
      DvceArray3D<Real>::HostMirror h_output_var = Kokkos::create_mirror(d_output_var);
      Kokkos::deep_copy(h_output_var,d_output_var);

      // copy host mirror to 5D host View containing all output variables
      auto h_slice = Kokkos::subview(outarray,n,m,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
      Kokkos::deep_copy(h_slice,h_output_var);
    }
  }
}
