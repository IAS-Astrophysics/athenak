//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file restart.cpp
//! \brief writes restart files

#include <sys/stat.h>  // mkdir

#include <algorithm>
#include <cstdio>      // fwrite(), fclose(), fopen(), fnprintf(), snprintf()
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <utility> // make_pair
#include <vector>

#include "athena.hpp"
#include "coordinates/cell_locations.hpp"
#include "geodesic-grid/geodesic_grid.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "coordinates/adm.hpp"
#include "z4c/compact_object_tracker.hpp"
#include "z4c/z4c.hpp"
#include "radiation/radiation.hpp"
#include "radiation_m1/radiation_m1.hpp"
#include "srcterms/turb_driver.hpp"
//#include "outputs.hpp"

//----------------------------------------------------------------------------------------
// constructor: also calls BaseTypeOutput base class constructor

RestartOutput::RestartOutput(ParameterInput *pin, Mesh *pm, OutputParameters op) :
  BaseTypeOutput(pin, pm, op) {
  // create directories for outputs. Comments in binary.cpp constructor explain why
  mkdir("rst",0775);
  bool single_file_per_rank = op.single_file_per_rank;
  if (single_file_per_rank) {
    char rank_dir[20];
    std::snprintf(rank_dir, sizeof(rank_dir), "rst/rank_%08d/", global_variable::my_rank);
    mkdir(rank_dir, 0775);
  }
}

//----------------------------------------------------------------------------------------
// RestartOutput::LoadOutputData()
// overload of standard load data function specific to restarts.  Loads dependent
// variables, including ghost zones.

void RestartOutput::LoadOutputData(Mesh *pm) {
  // get spatial dimensions of arrays, including ghost zones
  auto &indcs = pm->pmb_pack->pmesh->mb_indcs;
  int nout1 = indcs.nx1 + 2*(indcs.ng);
  int nout2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
  int nout3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;
  int nmb = pm->pmb_pack->nmb_thispack;

  // calculate total number of CC variables
  hydro::Hydro* phydro = pm->pmb_pack->phydro;
  mhd::MHD* pmhd = pm->pmb_pack->pmhd;
  adm::ADM* padm = pm->pmb_pack->padm;
  z4c::Z4c* pz4c = pm->pmb_pack->pz4c;
  radiation::Radiation* prad = pm->pmb_pack->prad;
  radiationm1::RadiationM1* pradm1 = pm->pmb_pack->pradm1;
  TurbulenceDriver* pturb=pm->pmb_pack->pturb;
  int nhydro=0, nmhd=0, nrad=0, nforce=3, nadm=0, nz4c=0, nradm1=0;
  if (phydro != nullptr) {
    nhydro = phydro->nhydro + phydro->nscalars;
  }
  if (pmhd != nullptr) {
    nmhd = pmhd->nmhd + pmhd->nscalars;
  }
  if (pz4c != nullptr) {
    nz4c = pz4c->nz4c;
  } else if (padm != nullptr) {
    nadm = padm->nadm;
  }
  // if the spacetime is evolved, we do not need to checkpoint/recover the ADM variables
  if (prad != nullptr) {
    nrad = prad->prgeo->nangles;
  }
  if (pradm1 != nullptr) {
    nradm1 = pradm1->nvarstot;
  }
  // Note for restarts, outarrays are dimensioned (m,n,k,j,i)
  if (phydro != nullptr) {
    Kokkos::realloc(outarray_hyd, nmb, nhydro, nout3, nout2, nout1);
    Kokkos::deep_copy(outarray_hyd, Kokkos::subview(phydro->u0, std::make_pair(0,nmb),
                      Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL));
  }
  if (pmhd != nullptr) {
    Kokkos::realloc(outarray_mhd, nmb, nmhd, nout3, nout2, nout1);
    Kokkos::deep_copy(outarray_mhd, Kokkos::subview(pmhd->u0, std::make_pair(0,nmb),
                      Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL));
    Kokkos::realloc(outfield.x1f, nmb, nout3, nout2, nout1+1);
    Kokkos::deep_copy(outfield.x1f, Kokkos::subview(pmhd->b0.x1f, std::make_pair(0,nmb),
                      Kokkos::ALL, Kokkos::ALL, Kokkos::ALL));
    Kokkos::realloc(outfield.x2f, nmb, nout3, nout2+1, nout1);
    Kokkos::deep_copy(outfield.x2f, Kokkos::subview(pmhd->b0.x2f, std::make_pair(0,nmb),
                      Kokkos::ALL, Kokkos::ALL, Kokkos::ALL));
    Kokkos::realloc(outfield.x3f, nmb, nout3+1, nout2, nout1);
    Kokkos::deep_copy(outfield.x3f, Kokkos::subview(pmhd->b0.x3f, std::make_pair(0,nmb),
                      Kokkos::ALL, Kokkos::ALL, Kokkos::ALL));
  }
  if (prad != nullptr) {
    Kokkos::realloc(outarray_rad, nmb, nrad, nout3, nout2, nout1);
    Kokkos::deep_copy(outarray_rad, Kokkos::subview(prad->i0, std::make_pair(0,nmb),
                      Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL));
  }
  if (pradm1 != nullptr) {
    Kokkos::realloc(outarray_radm1, nmb, nradm1, nout3, nout2, nout1);
    Kokkos::deep_copy(outarray_radm1, Kokkos::subview(pradm1->u0, std::make_pair(0,nmb),
                      Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL));
  }
  if (pturb != nullptr) {
    Kokkos::realloc(outarray_force, nmb, nforce, nout3, nout2, nout1);
    Kokkos::deep_copy(outarray_force, Kokkos::subview(pturb->force, std::make_pair(0,nmb),
                      Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL));
  }
  if (pz4c != nullptr) {
    Kokkos::realloc(outarray_z4c, nmb, nz4c, nout3, nout2, nout1);
    Kokkos::deep_copy(outarray_z4c, Kokkos::subview(pz4c->u0, std::make_pair(0,nmb),
                      Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL));
  } else if (padm != nullptr) {
    Kokkos::realloc(outarray_adm, nmb, nadm, nout3, nout2, nout1);
    Kokkos::deep_copy(outarray_adm, Kokkos::subview(padm->u_adm, std::make_pair(0,nmb),
                      Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL));
  }

  // calculate max/min number of MeshBlocks across all ranks
  noutmbs_max = pm->nmb_eachrank[0];
  noutmbs_min = pm->nmb_eachrank[0];
  for (int i=0; i<(global_variable::nranks); ++i) {
    noutmbs_max = std::max(noutmbs_max,pm->nmb_eachrank[i]);
    noutmbs_min = std::min(noutmbs_min,pm->nmb_eachrank[i]);
  }
}

//----------------------------------------------------------------------------------------
//! \fn void RestartOutput:::WriteOutputFile(Mesh *pm)
//  \brief Cycles over all MeshBlocks and writes everything to a single restart file

void RestartOutput::WriteOutputFile(Mesh *pm, ParameterInput *pin) {
  // get spatial dimensions of arrays, including ghost zones
  auto &indcs = pm->pmb_pack->pmesh->mb_indcs;
  int nout1 = indcs.nx1 + 2*(indcs.ng);
  int nout2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
  int nout3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;
  hydro::Hydro* phydro = pm->pmb_pack->phydro;
  mhd::MHD* pmhd = pm->pmb_pack->pmhd;
  radiation::Radiation* prad = pm->pmb_pack->prad;
  radiationm1::RadiationM1* pradm1 = pm->pmb_pack->pradm1;
  TurbulenceDriver* pturb=pm->pmb_pack->pturb;
  z4c::Z4c* pz4c = pm->pmb_pack->pz4c;
  adm::ADM* padm = pm->pmb_pack->padm;
  int nhydro=0, nmhd=0, nrad=0, nforce=3, nz4c=0, nadm=0, nco=0, nradm1=0;
  if (phydro != nullptr) {
    nhydro = phydro->nhydro + phydro->nscalars;
  }
  if (pmhd != nullptr) {
    nmhd = pmhd->nmhd + pmhd->nscalars;
  }
  if (prad != nullptr) {
    nrad = prad->prgeo->nangles;
  }
  if (pradm1 != nullptr) {
    nradm1 = pradm1->nvarstot;
  }
  if (pz4c != nullptr) {
    nz4c = pz4c->nz4c;
    nco = pz4c->ptracker.size();
  } else if (padm != nullptr) {
    nadm = padm->nadm;
  }
  bool single_file_per_rank = out_params.single_file_per_rank;
  std::string fname;
  if (single_file_per_rank) {
    // Generate a directory and filename for each rank
    // create filename: "rst/rank_YYYYYYY/file_basename" + "." + XXXXX + ".rst"
    // where YYYYYYY = 8-digit rank number
    // where XXXXX = 5-digit file_number
    char rank_dir[20];
    char number[7];
    std::snprintf(number, sizeof(number), ".%05d", out_params.file_number);
    std::snprintf(rank_dir, sizeof(rank_dir), "rank_%08d/", global_variable::my_rank);
    fname = std::string("rst/") + std::string(rank_dir) + out_params.file_basename
      + number + ".rst";

    // Debugging output to check directory and filename
    // std::cout << "Rank " << global_variable::my_rank << " generated filename: "
    //           << fname << std::endl;
  } else {
    // Existing behavior: single restart file
    // create filename: "rst/file_basename" + "." + XXXXX + ".rst"
    // where XXXXX = 5-digit file_number
    char number[7];
    std::snprintf(number, sizeof(number), ".%05d", out_params.file_number);
    fname = std::string("rst/") + out_params.file_basename + number + ".rst";
  }
  // increment counters now so values for *next* dump are stored in restart file
  out_params.file_number++;
  if (out_params.last_time < 0.0) {
    out_params.last_time = pm->time;
  } else {
    out_params.last_time += out_params.dt;
  }
  pin->SetInteger(out_params.block_name, "file_number", out_params.file_number);
  pin->SetReal(out_params.block_name, "last_time", out_params.last_time);

  // create string holding input parameters (copy of input file)
  std::stringstream ost;
  pin->ParameterDump(ost);
  std::string sbuf = ost.str();

  //--- STEP 1.  Root process writes header data (input file, critical variables)
  // Input file data is read by ParameterInput on restart, and the remaining header
  // variables are read in Mesh::BuildTreeFromRestart()

  // open file and  write the header; this part is serial
  IOWrapper resfile;
  resfile.Open(fname.c_str(), IOWrapper::FileMode::write, single_file_per_rank);
  if (global_variable::my_rank == 0 || single_file_per_rank) {
    // output the input parameters (input file)
    resfile.Write_any_type(sbuf.c_str(), sbuf.size(), "byte", single_file_per_rank);

    // output Mesh information
    resfile.Write_any_type(&(pm->nmb_total), (sizeof(int)), "byte",
                            single_file_per_rank);
    resfile.Write_any_type(&(pm->root_level), (sizeof(int)), "byte",
                            single_file_per_rank);
    resfile.Write_any_type(&(pm->mesh_size), (sizeof(RegionSize)), "byte",
                            single_file_per_rank);
    resfile.Write_any_type(&(pm->mesh_indcs), (sizeof(RegionIndcs)), "byte",
                            single_file_per_rank);
    resfile.Write_any_type(&(pm->mb_indcs), (sizeof(RegionIndcs)), "byte",
                            single_file_per_rank);
    resfile.Write_any_type(&(pm->time), (sizeof(Real)), "byte",
                            single_file_per_rank);
    resfile.Write_any_type(&(pm->dt), (sizeof(Real)), "byte",
                            single_file_per_rank);
    resfile.Write_any_type(&(pm->ncycle), (sizeof(int)), "byte",
                            single_file_per_rank);
  }
  //--- STEP 2.  Root process writes list of logical locations and cost of MeshBlocks
  // This data read in Mesh::BuildTreeFromRestart()

  if (global_variable::my_rank == 0 || single_file_per_rank) {
    resfile.Write_any_type(&(pm->lloc_eachmb[0]),(pm->nmb_total)*sizeof(LogicalLocation),
                           "byte", single_file_per_rank);
    resfile.Write_any_type(&(pm->cost_eachmb[0]), (pm->nmb_total)*sizeof(float),
                           "byte", single_file_per_rank);
  }

  //--- STEP 3.  Root process writes internal state of objects that require it
  if (global_variable::my_rank == 0 || single_file_per_rank) {
    // store z4c information
    if (pz4c != nullptr) {
      resfile.Write_any_type(&(pz4c->last_output_time), sizeof(Real), "byte",
                             single_file_per_rank);
    }
    // output puncture tracker data
    if (nco > 0) {
      for (auto & pt : pz4c->ptracker) {
        resfile.Write_any_type(pt->GetPos(), 3*sizeof(Real), "byte",
                               single_file_per_rank);
      }
    }
    // turbulence driver internal RNG
    if (pturb != nullptr) {
      resfile.Write_any_type(&(pturb->rstate), sizeof(RNG_State), "byte",
                             single_file_per_rank);
    }
  }

  //--- STEP 4.  All ranks write data over all MeshBlocks (5D arrays) in parallel
  // This data read in ProblemGenerator constructor for restarts

  // total size of all cell-centered variables and face-centered fields to be written by
  // this rank
  IOWrapperSizeT data_size = 0;
  const IOWrapperSizeT cells = static_cast<IOWrapperSizeT>(nout1)*nout2*nout3;
  if (phydro != nullptr) {
    data_size += cells*nhydro*sizeof(Real); // hydro u0
  }
  if (pmhd != nullptr) {
    data_size += cells*nmhd*sizeof(Real);   // mhd u0
    data_size += static_cast<IOWrapperSizeT>(nout1+1)*nout2*nout3*sizeof(Real);
    data_size += static_cast<IOWrapperSizeT>(nout1)*(nout2+1)*nout3*sizeof(Real);
    data_size += static_cast<IOWrapperSizeT>(nout1)*nout2*(nout3+1)*sizeof(Real);
  }
  if (prad != nullptr) {
    data_size += cells*nrad*sizeof(Real);   // radiation i0
  }
  if (pradm1 != nullptr) {
    data_size += cells*nradm1*sizeof(Real);   // radiation m1 u0
  }
  if (pturb != nullptr) {
    data_size += cells*nforce*sizeof(Real); // forcing
  }
  if (pz4c != nullptr) {
    data_size += cells*nz4c*sizeof(Real);   // z4c u0
  } else if (padm != nullptr) {
    data_size += cells*nadm*sizeof(Real);   // adm u_adm
  }
  if (global_variable::my_rank == 0 || single_file_per_rank) {
    resfile.Write_any_type(&(data_size), sizeof(IOWrapperSizeT), "byte",
                            single_file_per_rank);
  }

  // calculate size of data written in Steps 1-2 above
  IOWrapperSizeT step1size = sbuf.size()*sizeof(char) + 3*sizeof(int) + 2*sizeof(Real) +
                             sizeof(RegionSize) + 2*sizeof(RegionIndcs);
  IOWrapperSizeT step2size = (pm->nmb_total)*(sizeof(LogicalLocation) + sizeof(float));

  IOWrapperSizeT step3size = 3*nco*sizeof(Real);
  if (pz4c != nullptr) step3size += sizeof(Real);
  if (pturb != nullptr) step3size += sizeof(RNG_State);

  // write cell-centered variables in parallel
  IOWrapperSizeT offset_myrank = (step1size + step2size + step3size
                                  + sizeof(IOWrapperSizeT));

  if (!single_file_per_rank) {
    offset_myrank += data_size*(pm->gids_eachrank[global_variable::my_rank]);
  }

  // Preserve block-major file order without collective data sieving between fields.
  // Bound staging memory and MPI byte counts independently of the rank's block count.
  const IOWrapperSizeT local_size = data_size * pm->nmb_thisrank;
  const std::size_t capacity = std::min<IOWrapperSizeT>(64*1024*1024, local_size);
  std::vector<char> buffer(capacity);
  std::size_t used = 0;
  IOWrapperSizeT written = 0;
  auto flush = [&]() {
    if (used == 0) {return;}
    if (resfile.Write_any_type_at(buffer.data(), used, offset_myrank + written,
                                  "byte", single_file_per_rank) != used) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "packed data not written correctly to rst file, "
                << "restart file is broken." << std::endl;
      std::exit(EXIT_FAILURE);
    }
    written += used;
    used = 0;
  };
  auto append = [&](const auto &view) {
    const char *data = reinterpret_cast<const char *>(view.data());
    std::size_t remaining = view.size()*sizeof(Real);
    if (!view.span_is_contiguous() || remaining > local_size-written-used) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "unexpected array layout or size in packed restart."
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
    while (remaining > 0) {
      std::size_t count = std::min(remaining, capacity-used);
      std::memcpy(buffer.data()+used, data, count);
      used += count;
      data += count;
      remaining -= count;
      if (used == capacity) {flush();}
    }
  };
  for (int m=0; m<pm->nmb_thisrank; ++m) {
    if (phydro != nullptr) {
      append(Kokkos::subview(outarray_hyd,m,Kokkos::ALL,Kokkos::ALL,
                             Kokkos::ALL,Kokkos::ALL));
    }
    if (pmhd != nullptr) {
      append(Kokkos::subview(outarray_mhd,m,Kokkos::ALL,Kokkos::ALL,
                             Kokkos::ALL,Kokkos::ALL));
      append(Kokkos::subview(outfield.x1f,m,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL));
      append(Kokkos::subview(outfield.x2f,m,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL));
      append(Kokkos::subview(outfield.x3f,m,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL));
    }
    if (prad != nullptr) {
      append(Kokkos::subview(outarray_rad,m,Kokkos::ALL,Kokkos::ALL,
                             Kokkos::ALL,Kokkos::ALL));
    }
    if (pradm1 != nullptr) {
      append(Kokkos::subview(outarray_radm1,m,Kokkos::ALL,Kokkos::ALL,
                             Kokkos::ALL,Kokkos::ALL));
    }
    if (pturb != nullptr) {
      append(Kokkos::subview(outarray_force,m,Kokkos::ALL,Kokkos::ALL,
                             Kokkos::ALL,Kokkos::ALL));
    }
    if (pz4c != nullptr) {
      append(Kokkos::subview(outarray_z4c,m,Kokkos::ALL,Kokkos::ALL,
                             Kokkos::ALL,Kokkos::ALL));
    } else if (padm != nullptr) {
      append(Kokkos::subview(outarray_adm,m,Kokkos::ALL,Kokkos::ALL,
                             Kokkos::ALL,Kokkos::ALL));
    }
  }
  flush();
  if (written != local_size) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "packed restart size does not match the header."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }

  //--- STEP 5.  Write cyclic zoom restart data if enabled
  if (pm->pzoom != nullptr && pm->pzoom->write_rst) {
    // calculate offset of zoom data, i.e. after all previous data
    IOWrapperSizeT offset_zoom = step1size + step2size + step3size
                                 + sizeof(IOWrapperSizeT) + data_size*(pm->nmb_total);
    pm->pzoom->WriteRestartFile(resfile, offset_zoom, single_file_per_rank);
  }

  // close file, clean up
  resfile.Close(single_file_per_rank);

  return;
}
