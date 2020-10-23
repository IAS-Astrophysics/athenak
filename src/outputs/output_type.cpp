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
  auto it = pm->mblocks.begin();
  if (out_params.include_gzs) {
    int nout1 = it->mb_cells.nx1 + 2*(it->mb_cells.ng);
    int nout2 = (it->mb_cells.nx2 > 1)? (it->mb_cells.nx2 + 2*(it->mb_cells.ng)) : 1;
    int nout3 = (it->mb_cells.nx3 > 1)? (it->mb_cells.nx3 + 2*(it->mb_cells.ng)) : 1;
    ois = 0; oie = nout1-1;
    ojs = 0; oje = nout2-1;
    oks = 0; oke = nout3-1;
  } else {
    ois = it->mb_cells.is; oie = it->mb_cells.ie;
    ojs = it->mb_cells.js; oje = it->mb_cells.je;
    oks = it->mb_cells.ks; oke = it->mb_cells.ke;
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

  // check for valid output variable in <input> block
  if (var_cnt == 0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
       << "Output variable '" << out_params.variable << "' not implemented" << std::endl
       << "Allowed hydro variables: cons,D,E,mom,M1,M2,M3,prim,d,p,vel,vx,vy,vz"
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
  for (auto &mb : pm->mblocks) {

    // check for slicing in each dimension
    if (out_params.slice1) {
      // skip if slice is out of range of this MB
      if (out_params.slice_x1 <  mb.mb_size.x1min ||
          out_params.slice_x1 >= mb.mb_size.x1max) { continue; }
      // set index of slice
      Real &xmin = mb.mb_size.x1min;
      Real &xmax = mb.mb_size.x1max;
      ois = CellCenterIndex(out_params.slice_x1, mb.mb_cells.nx1, xmin, xmax);
      oie = ois;
    }

    if (out_params.slice2) {
      // skip if slice is out of range of this MB
      if (out_params.slice_x2 <  mb.mb_size.x2min ||
          out_params.slice_x2 >= mb.mb_size.x2max) { continue; }
      // set index of slice
      Real &xmin = mb.mb_size.x2min;
      Real &xmax = mb.mb_size.x2max;
      ojs = CellCenterIndex(out_params.slice_x2, mb.mb_cells.nx2, xmin, xmax);
      oje = ojs;
    }

    if (out_params.slice3) {
      // skip if slice is out of range of this MB
      if (out_params.slice_x3 <  mb.mb_size.x3min ||
          out_params.slice_x3 >= mb.mb_size.x3max) { continue; }
      // set index of slice
      Real &xmin = mb.mb_size.x3min;
      Real &xmax = mb.mb_size.x3max;
      oks = CellCenterIndex(out_params.slice_x3, mb.mb_cells.nx3, xmin, xmax);
      oke = oks;
    }

    // load all the output variables on this MeshBlock
    HostArray4D<Real> new_data("out",nvar,(oke-oks+1),(oje-ojs+1),(oie-ois+1));
    for (int n=0; n<nvar; ++n) {
      AthenaArray3D<Real> dev_buff("dev_buff",(oke-oks+1),(oje-ojs+1),(oie-ois+1));
      if (out_data_label_[n].compare("Dens")  == 0) {
        // Note capital "D" used to distinguish conserved from primitive mass density
        // (important for relativistic dynamics)
        auto dev_slice = Kokkos::subview(mb.phydro->u0, static_cast<int>(hydro::IDN),
          std::make_pair(oks,oke+1),std::make_pair(ojs,oje+1),std::make_pair(ois,oie+1));
        Kokkos::deep_copy(dev_buff,dev_slice);
      }
      if (out_data_label_[n].compare("Mom1")  == 0) {
        auto dev_slice = Kokkos::subview(mb.phydro->u0, static_cast<int>(hydro::IM1),
          std::make_pair(oks,oke+1),std::make_pair(ojs,oje+1),std::make_pair(ois,oie+1));
        Kokkos::deep_copy(dev_buff,dev_slice);
      }
      if (out_data_label_[n].compare("Mom2")  == 0) {
        auto dev_slice = Kokkos::subview(mb.phydro->u0, static_cast<int>(hydro::IM2),
          std::make_pair(oks,oke+1),std::make_pair(ojs,oje+1),std::make_pair(ois,oie+1));
        Kokkos::deep_copy(dev_buff,dev_slice);
      }
      if (out_data_label_[n].compare("Mom3")  == 0) {
        auto dev_slice = Kokkos::subview(mb.phydro->u0, static_cast<int>(hydro::IM3),
          std::make_pair(oks,oke+1),std::make_pair(ojs,oje+1),std::make_pair(ois,oie+1));
        Kokkos::deep_copy(dev_buff,dev_slice);
      }
      if (out_data_label_[n].compare("Ener")  == 0) {
        auto dev_slice = Kokkos::subview(mb.phydro->u0, static_cast<int>(hydro::IEN),
          std::make_pair(oks,oke+1),std::make_pair(ojs,oje+1),std::make_pair(ois,oie+1));
        Kokkos::deep_copy(dev_buff,dev_slice);
      }
      if (out_data_label_[n].compare("dens")  == 0) {
        auto dev_slice = Kokkos::subview(mb.phydro->w0, static_cast<int>(hydro::IDN),
          std::make_pair(oks,oke+1),std::make_pair(ojs,oje+1),std::make_pair(ois,oie+1));
        Kokkos::deep_copy(dev_buff,dev_slice);
      }
      if (out_data_label_[n].compare("velx")  == 0) {
        auto dev_slice = Kokkos::subview(mb.phydro->w0, static_cast<int>(hydro::IVX),
          std::make_pair(oks,oke+1),std::make_pair(ojs,oje+1),std::make_pair(ois,oie+1));
        Kokkos::deep_copy(dev_buff,dev_slice);
      }
      if (out_data_label_[n].compare("vely")  == 0) {
        auto dev_slice = Kokkos::subview(mb.phydro->w0, static_cast<int>(hydro::IVY),
          std::make_pair(oks,oke+1),std::make_pair(ojs,oje+1),std::make_pair(ois,oie+1));
        Kokkos::deep_copy(dev_buff,dev_slice);
      }
      if (out_data_label_[n].compare("velz")  == 0) {
        auto dev_slice = Kokkos::subview(mb.phydro->w0, static_cast<int>(hydro::IVZ),
          std::make_pair(oks,oke+1),std::make_pair(ojs,oje+1),std::make_pair(ois,oie+1));
        Kokkos::deep_copy(dev_buff,dev_slice);
      }
      if (out_data_label_[n].compare("pres")  == 0) {
        auto dev_slice = Kokkos::subview(mb.phydro->w0, static_cast<int>(hydro::IPR),
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
    out_data_gid_.push_back(mb.mb_gid);
  }
}
