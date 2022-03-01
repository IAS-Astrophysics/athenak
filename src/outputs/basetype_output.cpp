//========================================================================================
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
// BaseTypeOutput base class constructor
// Creates vector of output variable data

BaseTypeOutput::BaseTypeOutput(OutputParameters opar, Mesh *pm) :
    derived_var("derived-var",1,1,1,1,1),
    outarray("cc_outvar",1,1,1,1,1),
    outfield("fc_outvar",1,1,1,1),
    out_params(opar) {
  // exit for history, restart, or event log files
  if (out_params.file_type.compare("hst") == 0 ||
      out_params.file_type.compare("rst") == 0 ||
      out_params.file_type.compare("log") == 0) {return;}

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
  if ((ivar>=16) && (ivar<40) && (pm->pmb_pack->pmhd == nullptr)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
       << "Output of MHD variable requested in <output> block '"
       << out_params.block_name << "' but no MHD object has been constructed."
       << std::endl << "Input file is likely missing a <mhd> block" << std::endl;
    exit(EXIT_FAILURE);
  }
  if ((ivar==40) && (pm->pmb_pack->pturb == nullptr)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
       << "Output of Force variable requested in <output> block '"
       << out_params.block_name << "' but no Force object has been constructed."
       << std::endl << "Input file is likely missing a <forcing> block" << std::endl;
    exit(EXIT_FAILURE);
  }

  // Now load STL vector of output variables
  outvars.clear();
  int ndvars=0;

  // hydro (lab-frame) density
  if (out_params.variable.compare("hydro_u_d") == 0 ||
      out_params.variable.compare("hydro_u") == 0) {
    outvars.emplace_back("dens",0,&(pm->pmb_pack->phydro->u0));
  }

  // hydro (rest-frame) density
  if (out_params.variable.compare("hydro_w_d") == 0 ||
      out_params.variable.compare("hydro_w") == 0) {
    outvars.emplace_back("dens",0,&(pm->pmb_pack->phydro->w0));
  }

  // hydro components of momentum
  if (out_params.variable.compare("hydro_u_m1") == 0 ||
      out_params.variable.compare("hydro_u") == 0) {
    outvars.emplace_back("mom1",1,&(pm->pmb_pack->phydro->u0));
  }
  if (out_params.variable.compare("hydro_u_m2") == 0 ||
      out_params.variable.compare("hydro_u") == 0) {
    outvars.emplace_back("mom2",2,&(pm->pmb_pack->phydro->u0));
  }
  if (out_params.variable.compare("hydro_u_m3") == 0 ||
      out_params.variable.compare("hydro_u") == 0) {
    outvars.emplace_back("mom3",3,&(pm->pmb_pack->phydro->u0));
  }

  // hydro components of velocity
  if (out_params.variable.compare("hydro_w_vx") == 0 ||
      out_params.variable.compare("hydro_w") == 0) {
    outvars.emplace_back("velx",1,&(pm->pmb_pack->phydro->w0));
  }
  if (out_params.variable.compare("hydro_w_vy") == 0 ||
      out_params.variable.compare("hydro_w") == 0) {
    outvars.emplace_back("vely",2,&(pm->pmb_pack->phydro->w0));
  }
  if (out_params.variable.compare("hydro_w_vz") == 0 ||
      out_params.variable.compare("hydro_w") == 0) {
    outvars.emplace_back("velz",3,&(pm->pmb_pack->phydro->w0));
  }

  // hydro total energy
  if (out_params.variable.compare("hydro_u_e") == 0 ||
      out_params.variable.compare("hydro_u") == 0) {
    outvars.emplace_back("ener",4,&(pm->pmb_pack->phydro->u0));
  }

  // hydro internal energy or temperature
  if (out_params.variable.compare("hydro_w_e") == 0 ||
      out_params.variable.compare("hydro_w") == 0) {
    outvars.emplace_back("eint",4,&(pm->pmb_pack->phydro->w0));
  }

  // hydro passive scalars mass densities (s*d)
  if (out_params.variable.compare("hydro_u_s") == 0 ||
      out_params.variable.compare("hydro_u") == 0) {
    int nhyd = pm->pmb_pack->phydro->nhydro;
    int nvars = nhyd + pm->pmb_pack->phydro->nscalars;
    for (int n=nhyd; n<nvars; ++n) {
      char number[2];
      std::snprintf(number,sizeof(number),"%02d",(n - nhyd));
      std::string vname;
      vname.assign("r_");
      vname.append(number);
      outvars.emplace_back(vname,n,&(pm->pmb_pack->phydro->u0));
    }
  }

  // hydro passive scalars (s)
  if (out_params.variable.compare("hydro_w_s") == 0 ||
      out_params.variable.compare("hydro_w") == 0) {
    int nhyd = pm->pmb_pack->phydro->nhydro;
    int nvars = nhyd + pm->pmb_pack->phydro->nscalars;
    for (int n=nhyd; n<nvars; ++n) {
      char number[2];
      std::snprintf(number,sizeof(number),"%02d",(n - nhyd));
      std::string vname;
      vname.assign("s_");
      vname.append(number);
      outvars.emplace_back(vname,n,&(pm->pmb_pack->phydro->w0));
    }
  }

  // mhd (lab-frame) density
  if (out_params.variable.compare("mhd_u_d") == 0 ||
      out_params.variable.compare("mhd_u") == 0 ||
      out_params.variable.compare("mhd_u_bcc") == 0) {
    outvars.emplace_back("dens",0,&(pm->pmb_pack->pmhd->u0));
  }

  // mhd (rest-frame) density
  if (out_params.variable.compare("mhd_w_d") == 0 ||
      out_params.variable.compare("mhd_w") == 0 ||
      out_params.variable.compare("mhd_w_bcc") == 0) {
    outvars.emplace_back("dens",0,&(pm->pmb_pack->pmhd->w0));
  }

  // mhd components of momentum
  if (out_params.variable.compare("mhd_u_m1") == 0 ||
      out_params.variable.compare("mhd_u") == 0 ||
      out_params.variable.compare("mhd_u_bcc") == 0) {
    outvars.emplace_back("mom1",1,&(pm->pmb_pack->pmhd->u0));
  }
  if (out_params.variable.compare("mhd_u_m2") == 0 ||
      out_params.variable.compare("mhd_u") == 0 ||
      out_params.variable.compare("mhd_u_bcc") == 0) {
    outvars.emplace_back("mom2",2,&(pm->pmb_pack->pmhd->u0));
  }
  if (out_params.variable.compare("mhd_u_m3") == 0 ||
      out_params.variable.compare("mhd_u") == 0 ||
      out_params.variable.compare("mhd_u_bcc") == 0) {
    outvars.emplace_back("mom3",3,&(pm->pmb_pack->pmhd->u0));
  }

  // mhd components of velocity
  if (out_params.variable.compare("mhd_w_vx") == 0 ||
      out_params.variable.compare("mhd_w") == 0 ||
      out_params.variable.compare("mhd_w_bcc") == 0) {
    outvars.emplace_back("velx",1,&(pm->pmb_pack->pmhd->w0));
  }
  if (out_params.variable.compare("mhd_w_vy") == 0 ||
      out_params.variable.compare("mhd_w") == 0 ||
      out_params.variable.compare("mhd_w_bcc") == 0) {
    outvars.emplace_back("vely",2,&(pm->pmb_pack->pmhd->w0));
  }
  if (out_params.variable.compare("mhd_w_vz") == 0 ||
      out_params.variable.compare("mhd_w") == 0 ||
      out_params.variable.compare("mhd_w_bcc") == 0) {
    outvars.emplace_back("velz",3,&(pm->pmb_pack->pmhd->w0));
  }

  // mhd total energy
  if (out_params.variable.compare("mhd_u_e") == 0 ||
      out_params.variable.compare("mhd_u") == 0 ||
      out_params.variable.compare("mhd_u_bcc") == 0) {
    outvars.emplace_back("ener",4,&(pm->pmb_pack->pmhd->u0));
  }

  // mhd internal energy or temperature
  if (out_params.variable.compare("mhd_w_e") == 0 ||
      out_params.variable.compare("mhd_w") == 0 ||
      out_params.variable.compare("mhd_w_bcc") == 0) {
    outvars.emplace_back("eint",4,&(pm->pmb_pack->pmhd->w0));
  }

  // mhd passive scalars mass densities (s*d)
  if (out_params.variable.compare("mhd_u_s") == 0 ||
      out_params.variable.compare("mhd_u") == 0 ||
      out_params.variable.compare("mhd_u_bcc") == 0) {
    int nmhd = pm->pmb_pack->pmhd->nmhd;
    int nvars = nmhd + pm->pmb_pack->pmhd->nscalars;
    for (int n=nmhd; n<nvars; ++n) {
      char number[2];
      std::snprintf(number,sizeof(number),"%02d",(n - nmhd));
      std::string vname;
      vname.assign("r_");
      vname.append(number);
      outvars.emplace_back(vname,n,&(pm->pmb_pack->pmhd->u0));
    }
  }

  // mhd passive scalars (s)
  if (out_params.variable.compare("mhd_w_s") == 0 ||
      out_params.variable.compare("mhd_w") == 0 ||
      out_params.variable.compare("mhd_w_bcc") == 0) {
    int nmhd = pm->pmb_pack->pmhd->nmhd;
    int nvars = nmhd + pm->pmb_pack->pmhd->nscalars;
    for (int n=nmhd; n<nvars; ++n) {
      char number[2];
      std::snprintf(number,sizeof(number),"%02d",(n - nmhd));
      std::string vname;
      vname.assign("s_");
      vname.append(number);
      outvars.emplace_back(vname,n,&(pm->pmb_pack->pmhd->w0));
    }
  }

  // mhd cell-centered magnetic fields
  if (out_params.variable.compare("mhd_bcc1") == 0 ||
      out_params.variable.compare("mhd_bcc") == 0 ||
      out_params.variable.compare("mhd_u_bcc") == 0 ||
      out_params.variable.compare("mhd_w_bcc") == 0) {
    outvars.emplace_back("bcc1",0,&(pm->pmb_pack->pmhd->bcc0));
  }
  if (out_params.variable.compare("mhd_bcc2") == 0 ||
      out_params.variable.compare("mhd_bcc") == 0 ||
      out_params.variable.compare("mhd_u_bcc") == 0 ||
      out_params.variable.compare("mhd_w_bcc") == 0) {
    outvars.emplace_back("bcc2",1,&(pm->pmb_pack->pmhd->bcc0));
  }
  if (out_params.variable.compare("mhd_bcc3") == 0 ||
      out_params.variable.compare("mhd_bcc") == 0 ||
      out_params.variable.compare("mhd_u_bcc") == 0 ||
      out_params.variable.compare("mhd_w_bcc") == 0) {
    outvars.emplace_back("bcc3",2,&(pm->pmb_pack->pmhd->bcc0));
  }

  // hydro/mhd z-component of vorticity (useful in 2D)
  if (out_params.variable.compare("hydro_wz") == 0 ||
      out_params.variable.compare("mhd_wz") == 0) {
    outvars.emplace_back(true,"vorz",0,&(derived_var));
    ndvars++;
  }

  // hydro/mhd magnitude of vorticity (useful in 3D)
  if (out_params.variable.compare("hydro_w2") == 0 ||
      out_params.variable.compare("mhd_w2") == 0) {
    outvars.emplace_back(true,"vor2",0,&(derived_var));
    ndvars++;
  }

  // mhd z-component of current density (useful in 2D)
  if (out_params.variable.compare("mhd_jz") == 0) {
    outvars.emplace_back(true,"jz",0,&(derived_var));
    ndvars++;
  }

  // mhd magnitude of current density (useful in 3D)
  if (out_params.variable.compare("mhd_j2") == 0) {
    outvars.emplace_back(true,"j2",0,&(derived_var));
    ndvars++;
  }

  // turbulent forcing
  if (out_params.variable.compare("turb_force") == 0) {
    outvars.emplace_back("force1",0,&(pm->pmb_pack->pturb->force));
    outvars.emplace_back("force2",1,&(pm->pmb_pack->pturb->force));
    outvars.emplace_back("force3",2,&(pm->pmb_pack->pturb->force));
  }

  if (ndvars > 0) {
    int nmb = pm->pmb_pack->nmb_thispack;
    auto &indcs = pm->mb_indcs;
    int &ng = indcs.ng;
    int n1 = indcs.nx1 + 2*ng;
    int n2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng) : 1;
    int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng) : 1;
    Kokkos::realloc(derived_var, nmb, ndvars, n3, n2, n1);
  }
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
  auto &indcs = pm->pmb_pack->pmesh->mb_indcs;
  auto &size  = pm->pmb_pack->pmb->mb_size;
  for (int m=0; m<(pm->pmb_pack->nmb_thispack); ++m) {
    // skip if MeshBlock ID is specified and not equal to this ID
    if (out_params.gid >= 0 && m != out_params.gid) { continue; }

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

    // set coordinate geometry information for MB
    Real x1i = CellCenterX(ois - indcs.is, indcs.nx1, size.h_view(m).x1min,
                           size.h_view(m).x1max);
    Real x2i = CellCenterX(ojs - indcs.js, indcs.nx2, size.h_view(m).x2min,
                           size.h_view(m).x2max);
    Real x3i = CellCenterX(oks - indcs.ks, indcs.nx3, size.h_view(m).x3min,
                           size.h_view(m).x3max);
    Real dx1 = size.h_view(m).dx1;
    Real dx2 = size.h_view(m).dx2;
    Real dx3 = size.h_view(m).dx3;

    int id = pm->pmb_pack->pmb->mb_gid.h_view(m);
    outmbs.emplace_back(id,ois,oie,ojs,oje,oks,oke,x1i,x2i,x3i,dx1,dx2,dx3);
  }

  noutmbs_min = outmbs.size();
  noutmbs_max = outmbs.size();
#if MPI_PARALLEL_ENABLED
  // get minimum number of output MeshBlocks over all MPI ranks
  MPI_Allreduce(MPI_IN_PLACE, &noutmbs_min, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &noutmbs_max, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
#endif

  // get number of output vars and MBs, then realloc HostArray
  int nout_vars = outvars.size();
  int nout_mbs = outmbs.size();
  // note that while ois,oie,etc. can be different on each MB, the number of cells output
  // on each MeshBlock, i.e. (ois-ois+1), etc. is the same.
  if (nout_mbs > 0) {
    int nout1 = (outmbs[0].oie - outmbs[0].ois + 1);
    int nout2 = (outmbs[0].oje - outmbs[0].ojs + 1);
    int nout3 = (outmbs[0].oke - outmbs[0].oks + 1);
    Kokkos::realloc(outarray, nout_vars, nout_mbs, nout3, nout2, nout1);
  }

  // Now load data over all variables and MeshBlocks
  for (int n=0; n<nout_vars; ++n) {
    // Calculate derived variable, if required
    if (outvars[n].derived) {
      ComputeDerivedVariable(out_params.variable, pm);
    }

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
      auto hst_slice = Kokkos::subview(outarray,n,m,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
      Kokkos::deep_copy(hst_slice,hst_buff);
    }
  }
}
