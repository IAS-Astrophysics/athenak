//========================================================================================
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
#include "mesh/mesh.hpp"
#include "hydro/hydro.hpp"
#include "utils/grid_locations.hpp"
#include "outputs.hpp"

//----------------------------------------------------------------------------------------
// OutputType base class constructor
// Sets parameters like size and indices of output arrays

OutputType::OutputType(OutputParameters opar, Mesh *pm) :
   out_params(opar)
{
  // exit for history files
  if (out_params.file_type.compare("hst") == 0) {return;}

  // set size & starting indices of output arrays, adjusted accordingly if gz included 
  // Since all MeshBlocks the same, only need to compute values from first MB
  auto ncells = pm->pmb_pack->mb_cells;
  if (out_params.include_gzs) {
    int nout1 = ncells.nx1 + 2*(ncells.ng);
    int nout2 = (ncells.nx2 > 1)? (ncells.nx2 + 2*(ncells.ng)) : 1;
    int nout3 = (ncells.nx3 > 1)? (ncells.nx3 + 2*(ncells.ng)) : 1;
    ois = 0; oie = nout1-1;
    ojs = 0; oje = nout2-1;
    oks = 0; oke = nout3-1;
  } else {
    ois = ncells.is; oie = ncells.ie;
    ojs = ncells.js; oje = ncells.je;
    oks = ncells.ks; oke = ncells.ke;
  }

  // parse list of variables for each physics and flag variables to be output
  // TODO: get this working for multiple physics
  nvar = 0; int var_cnt=0;
  if (out_params.variable.compare("D") == 0 ||
      out_params.variable.compare("cons") == 0) {
    out_data_label_.push_back("Dens");
    nvar++; var_cnt++;
  }
  if (out_params.variable.compare("E") == 0 ||
      out_params.variable.compare("cons") == 0) {
    out_data_label_.push_back("Ener");
    nvar++; var_cnt++;
  }
  if (out_params.variable.compare("M1") == 0 ||
      out_params.variable.compare("mom") == 0 ||
      out_params.variable.compare("cons") == 0) {
    out_data_label_.push_back("Mom1");
    nvar++; var_cnt++;
  }
  if (out_params.variable.compare("M2") == 0 ||
      out_params.variable.compare("mom") == 0 ||
      out_params.variable.compare("cons") == 0) {
    out_data_label_.push_back("Mom2");
    nvar++; var_cnt++;
  }
  if (out_params.variable.compare("M3") == 0 ||
      out_params.variable.compare("mom") == 0 ||
      out_params.variable.compare("cons") == 0) {
    out_data_label_.push_back("Mom3");
    nvar++; var_cnt++;
  }
  // TODO: get working for multiple scalars
  if (out_params.variable.compare("S") == 0) {
    out_data_label_.push_back("S0");
    nvar++; var_cnt++;
  }
  if (out_params.variable.compare("d") == 0 ||
      out_params.variable.compare("prim") == 0) {
    out_data_label_.push_back("dens");
    nvar++; var_cnt++;
  }
  if (out_params.variable.compare("p") == 0 ||
      out_params.variable.compare("prim") == 0) {
    out_data_label_.push_back("pres");
    nvar++; var_cnt++;
  }
  if (out_params.variable.compare("vx") == 0 ||
      out_params.variable.compare("vel") == 0 ||
      out_params.variable.compare("prim") == 0) {
    out_data_label_.push_back("velx");
    nvar++; var_cnt++;
  }
  if (out_params.variable.compare("vy") == 0 ||
      out_params.variable.compare("vel") == 0 ||
      out_params.variable.compare("prim") == 0) {
    out_data_label_.push_back("vely");
    nvar++; var_cnt++;
  }
  if (out_params.variable.compare("vz") == 0 ||
      out_params.variable.compare("vel") == 0 ||
      out_params.variable.compare("prim") == 0) {
    out_data_label_.push_back("velz");
    nvar++; var_cnt++;
  }
  // TODO: get working for multiple scalars
  if (out_params.variable.compare("r") == 0) {
    out_data_label_.push_back("r0");
    nvar++; var_cnt++;
  }

  // check for valid output variable in <input> block
  if (var_cnt == 0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
       << "Output variable '" << out_params.variable << "' not implemented" << std::endl
       << "Allowed hydro variables: cons,D,E,mom,M1,M2,M3,S,prim,d,p,vel,vx,vy,vz,r"
       << std::endl;
    exit(EXIT_FAILURE);
  }
}

//----------------------------------------------------------------------------------------
// OutputType::LoadOutputData()
// create std::vector of HostArray3Ds containing data specified in <output> block for
// this output type

void OutputType::LoadOutputData(Mesh *pm)
{
  out_data_.clear();  // start with a clean list

  // out_data_ vector (indexed over # of output MBs) stores 4D array of variables
  // so start iteration over number of MeshBlocks
  // TODO: get this working for multiple physics, which may be either defined/undef

  // loop over all MeshBlocks
  MeshBlockPack* pmbp = pm->pmb_pack;;
  int nmb = pmbp->nmb_thispack;
  for (int m=0; m<nmb; ++m) {

    auto &cells = pmbp->mb_cells;
    auto &size  = pmbp->pmb->mbsize;
    // check for slicing in each dimension
    if (out_params.slice1) {
      // skip if slice is out of range of this MB
      if (out_params.slice_x1 <  size.x1min.h_view(m) ||
          out_params.slice_x1 >= size.x1max.h_view(m)) { continue; }
      // set index of slice
      ois = CellCenterIndex(out_params.slice_x1, cells.nx1,
                            size.x1min.h_view(m), size.x1max.h_view(m));
      oie = ois;
    }

    if (out_params.slice2) {
      // skip if slice is out of range of this MB
      if (out_params.slice_x2 <  size.x2min.h_view(m) ||
          out_params.slice_x2 >= size.x2max.h_view(m)) { continue; }
      // set index of slice
      ojs = CellCenterIndex(out_params.slice_x2, cells.nx2,
                            size.x2min.h_view(m), size.x2max.h_view(m));
      oje = ojs;
    }

    if (out_params.slice3) {
      // skip if slice is out of range of this MB
      if (out_params.slice_x3 <  size.x3min.h_view(m) ||
          out_params.slice_x3 >= size.x3max.h_view(m)) { continue; }
      // set index of slice
      oks = CellCenterIndex(out_params.slice_x3, cells.nx3,
                            size.x3min.h_view(m), size.x3max.h_view(m));
      oke = oks;
    }

    // load all the output variables on this MeshBlock
    HostArray4D<Real> new_data("out",nvar,(oke-oks+1),(oje-ojs+1),(oie-ois+1));
    for (int n=0; n<nvar; ++n) {
      AthenaArray3D<Real> dev_buff("dev_buff",(oke-oks+1),(oje-ojs+1),(oie-ois+1));
      if (out_data_label_[n].compare("Dens")  == 0) {
        // Note capital "D" used to distinguish conserved from primitive mass density
        // (important for relativistic dynamics)
        auto dev_slice = Kokkos::subview(pmbp->phydro->u0, m,static_cast<int>(hydro::IDN),
          std::make_pair(oks,oke+1),std::make_pair(ojs,oje+1),std::make_pair(ois,oie+1));
        Kokkos::deep_copy(dev_buff,dev_slice);
      }
      if (out_data_label_[n].compare("Mom1")  == 0) {
        auto dev_slice = Kokkos::subview(pmbp->phydro->u0, m,static_cast<int>(hydro::IM1),
          std::make_pair(oks,oke+1),std::make_pair(ojs,oje+1),std::make_pair(ois,oie+1));
        Kokkos::deep_copy(dev_buff,dev_slice);
      }
      if (out_data_label_[n].compare("Mom2")  == 0) {
        auto dev_slice = Kokkos::subview(pmbp->phydro->u0, m,static_cast<int>(hydro::IM2),
          std::make_pair(oks,oke+1),std::make_pair(ojs,oje+1),std::make_pair(ois,oie+1));
        Kokkos::deep_copy(dev_buff,dev_slice);
      }
      if (out_data_label_[n].compare("Mom3")  == 0) {
        auto dev_slice = Kokkos::subview(pmbp->phydro->u0, m,static_cast<int>(hydro::IM3),
          std::make_pair(oks,oke+1),std::make_pair(ojs,oje+1),std::make_pair(ois,oie+1));
        Kokkos::deep_copy(dev_buff,dev_slice);
      }
      if (out_data_label_[n].compare("Ener")  == 0) {
        auto dev_slice = Kokkos::subview(pmbp->phydro->u0, m,static_cast<int>(hydro::IEN),
          std::make_pair(oks,oke+1),std::make_pair(ojs,oje+1),std::make_pair(ois,oie+1));
        Kokkos::deep_copy(dev_buff,dev_slice);
      }
      if (out_data_label_[n].compare("dens")  == 0) {
        auto dev_slice = Kokkos::subview(pmbp->phydro->w0, m,static_cast<int>(hydro::IDN),
          std::make_pair(oks,oke+1),std::make_pair(ojs,oje+1),std::make_pair(ois,oie+1));
        Kokkos::deep_copy(dev_buff,dev_slice);
      }
      if (out_data_label_[n].compare("velx")  == 0) {
        auto dev_slice = Kokkos::subview(pmbp->phydro->w0, m,static_cast<int>(hydro::IVX),
          std::make_pair(oks,oke+1),std::make_pair(ojs,oje+1),std::make_pair(ois,oie+1));
        Kokkos::deep_copy(dev_buff,dev_slice);
      }
      if (out_data_label_[n].compare("vely")  == 0) {
        auto dev_slice = Kokkos::subview(pmbp->phydro->w0, m,static_cast<int>(hydro::IVY),
          std::make_pair(oks,oke+1),std::make_pair(ojs,oje+1),std::make_pair(ois,oie+1));
        Kokkos::deep_copy(dev_buff,dev_slice);
      }
      if (out_data_label_[n].compare("velz")  == 0) {
        auto dev_slice = Kokkos::subview(pmbp->phydro->w0, m,static_cast<int>(hydro::IVZ),
          std::make_pair(oks,oke+1),std::make_pair(ojs,oje+1),std::make_pair(ois,oie+1));
        Kokkos::deep_copy(dev_buff,dev_slice);
      }
      if (out_data_label_[n].compare("pres")  == 0) {
        auto dev_slice = Kokkos::subview(pmbp->phydro->w0, m,static_cast<int>(hydro::IPR),
          std::make_pair(oks,oke+1),std::make_pair(ojs,oje+1),std::make_pair(ois,oie+1));
        Kokkos::deep_copy(dev_buff,dev_slice);
      }
      // TODO get this working for multiple scalars
      if (out_data_label_[n].compare("S0")  == 0) {
        int nhyd = pmbp->phydro->nhydro;
        auto dev_slice = Kokkos::subview(pmbp->phydro->u0, m, static_cast<int>(nhyd),
          std::make_pair(oks,oke+1),std::make_pair(ojs,oje+1),std::make_pair(ois,oie+1));
        Kokkos::deep_copy(dev_buff,dev_slice);
      }
      if (out_data_label_[n].compare("r0")  == 0) {
        int nhyd = pmbp->phydro->nhydro;
        auto dev_slice = Kokkos::subview(pmbp->phydro->w0, m, static_cast<int>(nhyd),
          std::make_pair(oks,oke+1),std::make_pair(ojs,oje+1),std::make_pair(ois,oie+1));
        Kokkos::deep_copy(dev_buff,dev_slice);
      }

      // copy to host mirror array, and then to 4D host View containing all variables
      AthenaArray3D<Real>::HostMirror hst_buff = Kokkos::create_mirror(dev_buff);
      Kokkos::deep_copy(hst_buff,dev_buff);
      auto hst_slice = Kokkos::subview(new_data,n,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
      Kokkos::deep_copy(hst_slice,hst_buff);
    }

    // append variables on this MeshBlock to end of out_data_ vector
    out_data_.push_back(new_data);
    out_data_gid_.push_back(pmbp->pmb->mbgid.h_view(m));
  }
}
