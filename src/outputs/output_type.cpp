//=======================================================================================r
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file output_type.cpp
//  \brief implements base class OutputType constructor, and LoadOutputData functions
//

#include <cstdio>
#include <cstdlib>
#include <cstring>    // strcmp
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>   // std::string, to_string()

#include "athena.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "srcterms/srcterms.hpp"
#include "srcterms/turb_driver.hpp"
#include "outputs.hpp"

//----------------------------------------------------------------------------------------
// OutputType base class constructor
// Creates vector of output variable data

OutputType::OutputType(OutputParameters opar, Mesh *pm) :
   out_params(opar)
{
  // exit for history files
  if (out_params.file_type.compare("hst") == 0) {return;}

  outvars.clear();
  switch (out_params.variable) {
    // Load hydro conserved variables
    case OutputVariable::hydro_u_d:
      if (pm->pmb_pack->phydro == nullptr) ErrHydroOutput(out_params.block_name);
      outvars.emplace_back("dens",0,&(pm->pmb_pack->phydro->u0));
      break;
    case OutputVariable::hydro_u_m1:
      if (pm->pmb_pack->phydro == nullptr) ErrHydroOutput(out_params.block_name);
      outvars.emplace_back("mom1",1,&(pm->pmb_pack->phydro->u0));
      break;
    case OutputVariable::hydro_u_m2:
      if (pm->pmb_pack->phydro == nullptr) ErrHydroOutput(out_params.block_name);
      outvars.emplace_back("mom2",2,&(pm->pmb_pack->phydro->u0));
      break;
    case OutputVariable::hydro_u_m3:
      if (pm->pmb_pack->phydro == nullptr) ErrHydroOutput(out_params.block_name);
      outvars.emplace_back("mom3",3,&(pm->pmb_pack->phydro->u0));
      break;
    case OutputVariable::hydro_u_e:
      if (pm->pmb_pack->phydro == nullptr) ErrHydroOutput(out_params.block_name);
      outvars.emplace_back("ener",4,&(pm->pmb_pack->phydro->u0));
      break;
    case OutputVariable::hydro_u:
      {
      if (pm->pmb_pack->phydro == nullptr) ErrHydroOutput(out_params.block_name);
      outvars.emplace_back("dens",0,&(pm->pmb_pack->phydro->u0));
      outvars.emplace_back("mom1",1,&(pm->pmb_pack->phydro->u0));
      outvars.emplace_back("mom2",2,&(pm->pmb_pack->phydro->u0));
      outvars.emplace_back("mom3",3,&(pm->pmb_pack->phydro->u0));
      if (pm->pmb_pack->phydro->peos->eos_data.is_ideal) {
        outvars.emplace_back("ener",4,&(pm->pmb_pack->phydro->u0));
      }
      int nhyd = pm->pmb_pack->phydro->nhydro;
      int nvars = nhyd + pm->pmb_pack->phydro->nscalars;
      for (int n=nhyd; n<nvars; ++n) {
        char number[2];
        std::snprintf(number,sizeof(number),"%02d",(n - nhyd));
        std::string vname;
        vname.assign("scal");
        vname.append(number);
        outvars.emplace_back(vname,n,&(pm->pmb_pack->phydro->u0));
      }
      }
      break;

    // Load hydro primitive variables
    case OutputVariable::hydro_w_d:
      if (pm->pmb_pack->phydro == nullptr) ErrHydroOutput(out_params.block_name);
      outvars.emplace_back("dens",0,&(pm->pmb_pack->phydro->w0));
      break;
    case OutputVariable::hydro_w_vx:
      if (pm->pmb_pack->phydro == nullptr) ErrHydroOutput(out_params.block_name);
      outvars.emplace_back("velx",1,&(pm->pmb_pack->phydro->w0));
      break;
    case OutputVariable::hydro_w_vy:
      if (pm->pmb_pack->phydro == nullptr) ErrHydroOutput(out_params.block_name);
      outvars.emplace_back("vely",2,&(pm->pmb_pack->phydro->w0));
      break;
    case OutputVariable::hydro_w_vz:
      if (pm->pmb_pack->phydro == nullptr) ErrHydroOutput(out_params.block_name);
      outvars.emplace_back("velz",3,&(pm->pmb_pack->phydro->w0));
      break;
    case OutputVariable::hydro_w_p:
      if (pm->pmb_pack->phydro == nullptr) ErrHydroOutput(out_params.block_name);
      outvars.emplace_back("eint",4,&(pm->pmb_pack->phydro->w0));
      break;
    case OutputVariable::hydro_w:
      {
      if (pm->pmb_pack->phydro == nullptr) ErrHydroOutput(out_params.block_name);
      outvars.emplace_back("dens",0,&(pm->pmb_pack->phydro->w0));
      outvars.emplace_back("velx",1,&(pm->pmb_pack->phydro->w0));
      outvars.emplace_back("vely",2,&(pm->pmb_pack->phydro->w0));
      outvars.emplace_back("velz",3,&(pm->pmb_pack->phydro->w0));
      if (pm->pmb_pack->phydro->peos->eos_data.is_ideal) {
        outvars.emplace_back("eint",4,&(pm->pmb_pack->phydro->w0));
      }
      int nhyd = pm->pmb_pack->phydro->nhydro;
      int nvars = nhyd + pm->pmb_pack->phydro->nscalars;
      for (int n=nhyd; n<nvars; ++n) {
        char number[2];
        std::snprintf(number,sizeof(number),"%02d",(n - nhyd));
        std::string vname;
        vname.assign("scal");
        vname.append(number);
        outvars.emplace_back(vname,n,&(pm->pmb_pack->phydro->w0));
      }
      }
      break;

    // Load mhd conserved variables
    case OutputVariable::mhd_u_d:
      if (pm->pmb_pack->pmhd == nullptr) ErrMHDOutput(out_params.block_name);
      outvars.emplace_back("dens",0,&(pm->pmb_pack->pmhd->u0));
      break;
    case OutputVariable::mhd_u_m1:
      if (pm->pmb_pack->pmhd == nullptr) ErrMHDOutput(out_params.block_name);
      outvars.emplace_back("mom1",1,&(pm->pmb_pack->pmhd->u0));
      break;
    case OutputVariable::mhd_u_m2:
      if (pm->pmb_pack->pmhd == nullptr) ErrMHDOutput(out_params.block_name);
      outvars.emplace_back("mom2",2,&(pm->pmb_pack->pmhd->u0));
      break;
    case OutputVariable::mhd_u_m3:
      if (pm->pmb_pack->pmhd == nullptr) ErrMHDOutput(out_params.block_name);
      outvars.emplace_back("mom3",3,&(pm->pmb_pack->pmhd->u0));
      break;
    case OutputVariable::mhd_u_e:
      if (pm->pmb_pack->pmhd == nullptr) ErrMHDOutput(out_params.block_name);
      outvars.emplace_back("ener",4,&(pm->pmb_pack->pmhd->u0));
      break;
    case OutputVariable::mhd_u:
      {
      if (pm->pmb_pack->pmhd == nullptr) ErrMHDOutput(out_params.block_name);
      outvars.emplace_back("dens",0,&(pm->pmb_pack->pmhd->u0));
      outvars.emplace_back("mom1",1,&(pm->pmb_pack->pmhd->u0));
      outvars.emplace_back("mom2",2,&(pm->pmb_pack->pmhd->u0));
      outvars.emplace_back("mom3",3,&(pm->pmb_pack->pmhd->u0));
      if (pm->pmb_pack->pmhd->peos->eos_data.is_ideal) {
        outvars.emplace_back("ener",4,&(pm->pmb_pack->pmhd->u0));
      }
      int nmhd_ =  pm->pmb_pack->pmhd->nmhd;
      int nvars = nmhd_ + pm->pmb_pack->pmhd->nscalars;
      for (int n=nmhd_; n<nvars; ++n) {
        char number[2];
        std::snprintf(number,sizeof(number),"%02d",(n - nmhd_));
        std::string vname;
        vname.assign("scal");
        vname.append(number);
        outvars.emplace_back(vname,n,&(pm->pmb_pack->pmhd->u0));
      }
      }
      break;

    // Load mhd primitive variables
    case OutputVariable::mhd_w_d:
      if (pm->pmb_pack->pmhd == nullptr) ErrMHDOutput(out_params.block_name);
      outvars.emplace_back("dens",0,&(pm->pmb_pack->pmhd->w0));
      break;
    case OutputVariable::mhd_w_vx:
      if (pm->pmb_pack->pmhd == nullptr) ErrMHDOutput(out_params.block_name);
      outvars.emplace_back("velx",1,&(pm->pmb_pack->pmhd->w0));
      break;
    case OutputVariable::mhd_w_vy:
      if (pm->pmb_pack->pmhd == nullptr) ErrMHDOutput(out_params.block_name);
      outvars.emplace_back("vely",2,&(pm->pmb_pack->pmhd->w0));
      break;
    case OutputVariable::mhd_w_vz:
      if (pm->pmb_pack->pmhd == nullptr) ErrMHDOutput(out_params.block_name);
      outvars.emplace_back("velz",3,&(pm->pmb_pack->pmhd->w0));
      break;
    case OutputVariable::mhd_w_p:
      if (pm->pmb_pack->pmhd == nullptr) ErrMHDOutput(out_params.block_name);
      outvars.emplace_back("eint",4,&(pm->pmb_pack->pmhd->w0));
      break;
    case OutputVariable::mhd_w:
      {
      if (pm->pmb_pack->pmhd == nullptr) ErrMHDOutput(out_params.block_name);
      outvars.emplace_back("dens",0,&(pm->pmb_pack->pmhd->w0));
      outvars.emplace_back("velx",1,&(pm->pmb_pack->pmhd->w0));
      outvars.emplace_back("vely",2,&(pm->pmb_pack->pmhd->w0));
      outvars.emplace_back("velz",3,&(pm->pmb_pack->pmhd->w0));
      if (pm->pmb_pack->pmhd->peos->eos_data.is_ideal) {
        outvars.emplace_back("eint",4,&(pm->pmb_pack->pmhd->w0));
      }
      int nmhd_ =  pm->pmb_pack->pmhd->nmhd;
      int nvars = nmhd_ + pm->pmb_pack->pmhd->nscalars;
      for (int n=nmhd_; n<nvars; ++n) {
        char number[2];
        std::snprintf(number,sizeof(number),"%02d",(n - nmhd_));
        std::string vname;
        vname.assign("scal");
        vname.append(number);
        outvars.emplace_back(vname,n,&(pm->pmb_pack->pmhd->w0));
      }
      }
      break;

    // Load mhd cell-centered magnetic fields
    case OutputVariable::mhd_bcc1:
      if (pm->pmb_pack->pmhd == nullptr) ErrMHDOutput(out_params.block_name);
      outvars.emplace_back("bcc1",0,&(pm->pmb_pack->pmhd->bcc0));
      break;
    case OutputVariable::mhd_bcc2:
      if (pm->pmb_pack->pmhd == nullptr) ErrMHDOutput(out_params.block_name);
      outvars.emplace_back("bcc2",1,&(pm->pmb_pack->pmhd->bcc0));
      break;
    case OutputVariable::mhd_bcc3:
      if (pm->pmb_pack->pmhd == nullptr) ErrMHDOutput(out_params.block_name);
      outvars.emplace_back("bcc3",2,&(pm->pmb_pack->pmhd->bcc0));
      break;
    case OutputVariable::mhd_bcc:
      if (pm->pmb_pack->pmhd == nullptr) ErrMHDOutput(out_params.block_name);
      outvars.emplace_back("bcc1",0,&(pm->pmb_pack->pmhd->bcc0));
      outvars.emplace_back("bcc2",1,&(pm->pmb_pack->pmhd->bcc0));
      outvars.emplace_back("bcc3",2,&(pm->pmb_pack->pmhd->bcc0));
      break;

    // Load mhd conserved variables and cell-centered magnetic fields
    case OutputVariable::mhd_u_bcc:
      {
      if (pm->pmb_pack->pmhd == nullptr) ErrMHDOutput(out_params.block_name);
      outvars.emplace_back("dens",0,&(pm->pmb_pack->pmhd->u0));
      outvars.emplace_back("mom1",1,&(pm->pmb_pack->pmhd->u0));
      outvars.emplace_back("mom2",2,&(pm->pmb_pack->pmhd->u0));
      outvars.emplace_back("mom3",3,&(pm->pmb_pack->pmhd->u0));
      if (pm->pmb_pack->pmhd->peos->eos_data.is_ideal) {
        outvars.emplace_back("ener",4,&(pm->pmb_pack->pmhd->u0));
      }
      outvars.emplace_back("bcc1",0,&(pm->pmb_pack->pmhd->bcc0));
      outvars.emplace_back("bcc2",1,&(pm->pmb_pack->pmhd->bcc0));
      outvars.emplace_back("bcc3",2,&(pm->pmb_pack->pmhd->bcc0));
      int nmhd_ =  pm->pmb_pack->pmhd->nmhd;
      int nvars = nmhd_ + pm->pmb_pack->pmhd->nscalars;
      for (int n=nmhd_; n<nvars; ++n) {
        char number[2];
        std::snprintf(number,sizeof(number),"%02d",(n - nmhd_));
        std::string vname;
        vname.assign("scal");
        vname.append(number);
        outvars.emplace_back(vname,n,&(pm->pmb_pack->pmhd->w0));
      }
      }
      break;

    // Load mhd primitive variables and cell-centered magnetic fields
    case OutputVariable::mhd_w_bcc:
      {
      if (pm->pmb_pack->pmhd == nullptr) ErrMHDOutput(out_params.block_name);
      outvars.emplace_back("dens",0,&(pm->pmb_pack->pmhd->w0));
      outvars.emplace_back("velx",1,&(pm->pmb_pack->pmhd->w0));
      outvars.emplace_back("vely",2,&(pm->pmb_pack->pmhd->w0));
      outvars.emplace_back("velz",3,&(pm->pmb_pack->pmhd->w0));
      if (pm->pmb_pack->pmhd->peos->eos_data.is_ideal) {
        outvars.emplace_back("eint",4,&(pm->pmb_pack->pmhd->w0));
      }
      int nmhd_ =  pm->pmb_pack->pmhd->nmhd;
      outvars.emplace_back("bcc1",0,&(pm->pmb_pack->pmhd->bcc0));
      outvars.emplace_back("bcc2",1,&(pm->pmb_pack->pmhd->bcc0));
      outvars.emplace_back("bcc3",2,&(pm->pmb_pack->pmhd->bcc0));
      int nvars = nmhd_ + pm->pmb_pack->pmhd->nscalars;
      for (int n=nmhd_; n<nvars; ++n) {
        char number[2];
        std::snprintf(number,sizeof(number),"%02d",(n - nmhd_));
        std::string vname;
        vname.assign("scal");
        vname.append(number);
        outvars.emplace_back(vname,n,&(pm->pmb_pack->pmhd->w0));
      }
      }
      break;

    // Load turbulent forcing
    case OutputVariable::turb_force:
      if (pm->pmb_pack->pturb == nullptr) ErrForceOutput(out_params.block_name);
      outvars.emplace_back("force1",0,&(pm->pmb_pack->pturb->force));
      outvars.emplace_back("force2",1,&(pm->pmb_pack->pturb->force));
      outvars.emplace_back("force3",2,&(pm->pmb_pack->pturb->force));
      break;

    default:
      break;
  }
}

//----------------------------------------------------------------------------------------
// OutputType::LoadOutputData()
// create std::vector of HostArray3Ds containing data specified in <output> block for
// this output type

void OutputType::LoadOutputData(Mesh *pm)
{
  // out_data_ vector (indexed over # of output MBs) stores 4D array of variables
  // so start iteration over number of MeshBlocks
  // TODO: get this working for multiple physics, which may be either defined/undef

  // With AMR, number and location of output MBs can change between output times.
  // So start with clean vector of output MeshBlock info, and re-compute
  outmbs.clear();

  // loop over all MeshBlocks
  // set size & starting indices of output arrays, adjusted accordingly if gz included 
  auto &indcs = pm->pmb_pack->coord.coord_data.mb_indcs;
  auto &size  = pm->pmb_pack->coord.coord_data.mb_size;
  for (int m=0; m<(pm->pmb_pack->nmb_thispack); ++m) {

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
      oie = ois;
    }

    if (out_params.slice2) {
      // skip this MB if slice is out of range
      if (out_params.slice_x2 <  size.h_view(m).x2min ||
          out_params.slice_x2 >= size.h_view(m).x2max) { continue; }
      // set index of slice
      ojs = CellCenterIndex(out_params.slice_x2, indcs.nx2,
                            size.h_view(m).x2min, size.h_view(m).x2max);
      oje = ojs;
    }

    if (out_params.slice3) {
      // skip this MB if slice is out of range
      if (out_params.slice_x3 <  size.h_view(m).x3min ||
          out_params.slice_x3 >= size.h_view(m).x3max) { continue; }
      // set index of slice
      oks = CellCenterIndex(out_params.slice_x3, indcs.nx3,
                            size.h_view(m).x3min, size.h_view(m).x3max);
      oke = oks;
    }
    int id = pm->pmb_pack->pmb->mbgid.h_view(m);
    outmbs.emplace_back(id,ois,oie,ojs,oje,oks,oke);
  }

  // get number of output vars and MBs, then realloc HostArray
  int nout_vars = outvars.size();
  int nout_mbs = outmbs.size();
  // note that while ois,oie,etc. can be different on each MB, the number of cells output
  // on each MeshBlock, i.e. (ois-ois+1), etc. is the same. 
  int nout1 = (outmbs[0].oie - outmbs[0].ois + 1);
  int nout2 = (outmbs[0].oje - outmbs[0].ojs + 1);
  int nout3 = (outmbs[0].oke - outmbs[0].oks + 1);
  Kokkos::realloc(outdata, nout_vars, nout_mbs, nout3, nout2, nout1);

  // Now load data over all variables and MeshBlocks
  for (int n=0; n<nout_vars; ++n) {
    for (int m=0; m<nout_mbs; ++m) {
      int &ois = outmbs[m].ois;
      int &oie = outmbs[m].oie;
      int &ojs = outmbs[m].ojs;
      int &oje = outmbs[m].oje;
      int &oks = outmbs[m].oks;
      int &oke = outmbs[m].oke;
      int mbi = pm->FindMeshBlockIndex(outmbs[m].mb_gid);

      // load an output variable on this output MeshBlock
      DvceArray3D<Real> dev_buff("dev_buff",(oke-oks+1),(oje-ojs+1),(oie-ois+1));
      auto dev_slice = Kokkos::subview(*(outvars[n].data_ptr), mbi, outvars[n].data_index,
        std::make_pair(oks,oke+1),std::make_pair(ojs,oje+1),std::make_pair(ois,oie+1));
      Kokkos::deep_copy(dev_buff,dev_slice);

      // copy to host mirror array, and then to 5D host View containing all variables
      DvceArray3D<Real>::HostMirror hst_buff = Kokkos::create_mirror(dev_buff);
      Kokkos::deep_copy(hst_buff,dev_buff);
      auto hst_slice = Kokkos::subview(outdata,n,m,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
      Kokkos::deep_copy(hst_slice,hst_buff);
    }
  }
}

//----------------------------------------------------------------------------------------
// OutputType::ErrHydroOutput()
// Print error message when output of Hydro variable requested but Hydro object not
// constructed, and then quit

void OutputType::ErrHydroOutput(std::string block)
{
  std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
            << "Output of Hydro variable requested in <output> block '" << block
            << "' but no Hydro object " << std::endl << "has been constructed." 
            << " Input file is likely missing a <hydro> block" << std::endl;
  exit(EXIT_FAILURE);
}

//----------------------------------------------------------------------------------------
// OutputType::ErrMHDOutput()
// Print error message when output of MHD variable requested but MHD object not
// constructed, and then quit

void OutputType::ErrMHDOutput(std::string block)
{
  std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
            << "Output of MHD variable requested in <output> block '" << block
            << "' but no MHD object " << std::endl << "has been constructed."
            << " Input file is likely missing a <mhd> block" << std::endl;
  exit(EXIT_FAILURE);
}

//----------------------------------------------------------------------------------------
// OutputType::ErrForceOutput()
// Print error message when output of Force variable requested but Force object not
// constructed, and then quit

void OutputType::ErrForceOutput(std::string block)
{
  std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
            << "Output of Force variable requested in <output> block '" << block
            << "' but no Force object " << std::endl << "has been constructed."
            << " Input file is likely missing a <forcing> block" << std::endl;
  exit(EXIT_FAILURE);
}
