//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file coarsened_binary.cpp
//! \brief writes output data in binary format, which simply consists of each MeshBlock
//! written contiguously in order of "gid" in binary format.

#include <sys/stat.h>  // mkdir

#include <cstdio>      // fwrite(), fclose(), fopen(), fnprintf(), snprintf()
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>
#include <utility>
#include <algorithm>

#include "athena.hpp"
#include "globals.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "outputs.hpp"

//----------------------------------------------------------------------------------------
// Constructor: also calls BaseTypeOutput base class constructor

CoarsenedBinaryOutput::CoarsenedBinaryOutput(ParameterInput *pin, Mesh *pm,
                                             OutputParameters op) :
  BaseTypeOutput(pin, pm, op) {
  // create directories for outputs
  // useful for mpiio-based outputs because on some supercomputers you may need to
  // set different stripe counts depending on whether mpiio is used in order to
  // achieve the best performance and not to crash the filesystem
  std::string dir_name;
  dir_name.assign("cbin_");
  dir_name.append(out_params.file_id);
  dir_name.append("_");
  dir_name.append(std::to_string(out_params.coarsen_factor));
  mkdir(dir_name.c_str(),0775);
  bool single_file_per_rank = op.single_file_per_rank;
  if (single_file_per_rank) {
    char rank_dir[20];
    std::snprintf(rank_dir, sizeof(rank_dir), "rank_%08d/", global_variable::my_rank);
    dir_name.append("/");
    dir_name.append(rank_dir);
    mkdir(dir_name.c_str(), 0775);
  }
}

//----------------------------------------------------------------------------------------
// BaseTypeOutput::LoadOutputData()
// create std::vector of HostArray3Ds containing data specified in <output> block for
// this output type

void CoarsenedBinaryOutput::LoadOutputData(Mesh *pm) {
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

    // DBF: I have never checked if slicing works with coarsened data
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
  int nout_vars_with_moments;
  if (out_params.compute_moments) {
    nout_vars_with_moments = outvars.size() * 4;
  } else {
    nout_vars_with_moments = outvars.size();
  }
  int nout_vars = outvars.size();
  int nout_mbs = outmbs.size();
  // note that while ois,oie,etc. can be different on each MB, the number of cells output
  // on each MeshBlock, i.e. (ois-ois+1), etc. is the same.
  if (nout_mbs > 0) {
    int nout1 = ((outmbs[0].oie - outmbs[0].ois + 1)/out_params.coarsen_factor);
    int nout2 = ((outmbs[0].oje - outmbs[0].ojs + 1)/out_params.coarsen_factor);
    int nout3 = ((outmbs[0].oke - outmbs[0].oks + 1)/out_params.coarsen_factor);
    // NB: outarray stores all output data on Host
    // DBF: outarray is smaller by a factor of coarsen_factor in each dimension
    Kokkos::realloc(outarray, nout_vars_with_moments, nout_mbs, nout3, nout2, nout1);
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
      std::pair<int,int> moment_range;
      if (out_params.compute_moments) {
        moment_range = std::make_pair(n*4, n*4+4);
      } else {
        moment_range = std::make_pair(n, n+1);
      }
      int nout1 = (outmbs[0].oie - outmbs[0].ois + 1);
      int nout2 = (outmbs[0].oje - outmbs[0].ojs + 1);
      int nout3 = (outmbs[0].oke - outmbs[0].oks + 1);
      int coarsened_nout1 = nout1/out_params.coarsen_factor;
      int coarsened_nout2 = nout2/out_params.coarsen_factor;
      int coarsened_nout3 = nout3/out_params.coarsen_factor;

      // copy output variable to new device View
      DvceArray3D<Real> d_output_var("d_out_var",nout3,nout2,nout1);
      auto d_slice = Kokkos::subview(*(outvars[n].data_ptr), mbi, outvars[n].data_index,
                                     krange,jrange,irange);
      Kokkos::deep_copy(d_output_var,d_slice);
      Kokkos::fence(); // Ensure complete copy


      int number_of_moments = 1;
      if (out_params.compute_moments) {
        number_of_moments = 4;
      }
      DvceArray4D<Real> d_output_var_coarsened("d_output_var_coarsened",
        number_of_moments, coarsened_nout3, coarsened_nout2, coarsened_nout1);

      // Coarsen the d_slice and store the result in d_output_var
      // CoarsenVariable(d_output_var, d_output_var_coarsened, out_params.coarsen_factor);
      int coarsen_factor = out_params.coarsen_factor;
      int coarsen_factor_cubed = coarsen_factor * coarsen_factor * coarsen_factor;

      if (nout1 % coarsen_factor != 0 || nout2 % coarsen_factor != 0
                                      || nout3 % coarsen_factor != 0) {
          std::cout << "Error: Full data dimensions are not divisible by coarsen_factor"
          << std::endl;
          exit(EXIT_FAILURE);
      }

      int total_iterations = coarsened_nout3
        * coarsened_nout2 * coarsened_nout1 * coarsen_factor_cubed;

      bool compute_moments = out_params.compute_moments;
      Kokkos::parallel_for("coarsen_variable",
       Kokkos::RangePolicy<DevExeSpace>(0, total_iterations),
      KOKKOS_LAMBDA(const int idx) {
        // Calculate the 3D indices for the coarsened data
        int total_coarsened_elements = coarsened_nout1*coarsened_nout2*coarsened_nout3;
        int k_c = (idx / (coarsened_nout2 * coarsened_nout1)) % coarsened_nout3;
        int j_c = (idx / coarsened_nout1) % coarsened_nout2;
        int i_c = idx % coarsened_nout1;

        // Calculate the offset within the coarsen_factor_cubed cube
        int offset = idx / total_coarsened_elements;
        int kk = offset / (coarsen_factor * coarsen_factor);
        int jj = (offset / coarsen_factor) % coarsen_factor;
        int ii = offset % coarsen_factor;

        // Calculate the corresponding indices in the full data
        int k = k_c * coarsen_factor + kk;
        int j = j_c * coarsen_factor + jj;
        int i = i_c * coarsen_factor + ii;

        // Perform the coarsening operation
        if(k < nout3 && j < nout2 && i < nout1) {
          Kokkos::atomic_add(&d_output_var_coarsened(0, k_c, j_c, i_c),
            d_output_var(k, j, i));
          if (compute_moments) {
            Kokkos::atomic_add(&d_output_var_coarsened(1, k_c, j_c, i_c),
              d_output_var(k, j, i)*d_output_var(k, j, i));
            Kokkos::atomic_add(&d_output_var_coarsened(2, k_c, j_c, i_c),
              d_output_var(k, j, i)*d_output_var(k, j, i)*d_output_var(k, j, i));
            Kokkos::atomic_add(&d_output_var_coarsened(3, k_c, j_c, i_c),
               d_output_var(k, j, i)*d_output_var(k, j, i)
              *d_output_var(k, j, i)*d_output_var(k, j, i));
          }
        }
      });
      // Normalize the coarsened data
      int normalize_iterations = number_of_moments * coarsened_nout3
                                * coarsened_nout2 * coarsened_nout1;
      Kokkos::parallel_for("normalize_coarsened_variable",
        Kokkos::RangePolicy<DevExeSpace>(0, normalize_iterations),
      KOKKOS_LAMBDA(const int idx) {
        int moment_idx = idx / (coarsened_nout3 * coarsened_nout2 * coarsened_nout1);
        int k = (idx / (coarsened_nout2 * coarsened_nout1)) % coarsened_nout3;
        int j = (idx / coarsened_nout1) % coarsened_nout2;
        int i = idx % coarsened_nout1;

        d_output_var_coarsened(moment_idx, k, j, i) /= coarsen_factor_cubed;
      });


      // Now, create a host mirror for the coarsened data.
      DvceArray4D<Real>::HostMirror h_output_var = Kokkos::create_mirror(
        d_output_var_coarsened
      );

      // Copy the coarsened data to the host mirror.
      Kokkos::deep_copy(h_output_var, d_output_var_coarsened);
      Kokkos::fence(); // Ensure complete copy before using h_output_var on the host

      // copy host mirror to 5D host View containing all output variables
      // if (out_params.compute_moments) {
      auto h_slice = Kokkos::subview(outarray,
        moment_range,m,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL
      );
      Kokkos::deep_copy(h_slice,h_output_var);
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn void CoarsenedBinaryOutput:::WriteOutputFile(Mesh *pm)
//  \brief Cycles over all MeshBlocks and writes OutputData in Coarsenedbinary format
//   All MeshBlocks are written to the same file.

void CoarsenedBinaryOutput::WriteOutputFile(Mesh *pm, ParameterInput *pin) {
  // check if slicing
  bool bin_slice = (out_params.slice1 || out_params.slice2 || out_params.slice3);

  // create filename: "cbin_"+"file_id"+"_"+"coarsening_factor"+"/file_basename"
  // + "." + "file_id" + "." + XXXXX + ".cbin"
  // where XXXXX = 5-digit file_number
  bool single_file_per_rank = out_params.single_file_per_rank;

  std::string fname;
  char number[6];
  std::snprintf(number, sizeof(number), "%05d", out_params.file_number);

  fname.assign("cbin_");
  fname.append(out_params.file_id);
  fname.append("_");
  fname.append(std::to_string(out_params.coarsen_factor));
  if (single_file_per_rank) {
    char rank_dir[20];
    std::snprintf(rank_dir, sizeof(rank_dir), "rank_%08d/", global_variable::my_rank);
    fname.append("/");
    fname.append(rank_dir);
  }
  fname.append(out_params.file_basename);
  fname.append(".");
  fname.append(out_params.file_id);
  fname.append(".");
  fname.append(number);
  fname.append(".cbin");

  IOWrapper cbinfile;
  std::size_t header_offset=0;
  cbinfile.Open(fname.c_str(), IOWrapper::FileMode::write, single_file_per_rank);

  int number_of_moments = 1;
  if (out_params.compute_moments) {
    number_of_moments = 4;
  }

  // Basic parts of the format:
  // 1. Size of the header
  // 2. Current time
  // 3. List of variables in the file
  // 4. Header (input file information)
  {std::stringstream msg;
  msg << "Athena binary output version=1.1" << std::endl
      // preheader size includes "size of preheader" line up to "number of variables"
      << "  size of preheader=7" << std::endl
      << "  time=" << pm->time << std::endl
      << "  cycle=" << pm->ncycle << std::endl
      << "  number of moments=" << number_of_moments << std::endl
      << "  coarsening factor=" << out_params.coarsen_factor << std::endl
      << "  size of location=" << sizeof(Real) << std::endl
      << "  size of variable=" << sizeof(float) << std::endl
      << "  number of variables=" << outvars.size()*number_of_moments << std::endl
      << "  variables:  ";
  if (out_params.compute_moments) {
    // need to write the label for each of the 4 moments
    for (int n=0; n<outvars.size(); n++) {
      msg << outvars[n].label.c_str() << "_1st  ";
      msg << outvars[n].label.c_str() << "_2nd  ";
      msg << outvars[n].label.c_str() << "_3rd  ";
      msg << outvars[n].label.c_str() << "_4th  ";
    }
  } else {
    for (int n=0; n<outvars.size(); n++) {
      msg << outvars[n].label.c_str() << "  ";
    }
  }
  msg << std::endl;
  if (global_variable::my_rank == 0 || single_file_per_rank) {
    cbinfile.Write_any_type(msg.str().c_str(),msg.str().size(), "byte",
                            single_file_per_rank);
  }
  header_offset += msg.str().size();}
  {std::stringstream msg;
  // prepare the input parameters
  std::stringstream ost;
  pin->ParameterDump(ost);
  std::string sbuf=ost.str();
  msg << "  header offset=" << sbuf.size()*sizeof(char)  << std::endl;
  if (global_variable::my_rank == 0 || single_file_per_rank) {
    cbinfile.Write_any_type(msg.str().c_str(),msg.str().size(), "byte",
                            single_file_per_rank);
    cbinfile.Write_any_type(sbuf.c_str(),sbuf.size(), "byte", single_file_per_rank);
  }
  header_offset += sbuf.size()*sizeof(char);
  header_offset += msg.str().size();}

  //  5. Data.  An arbitrary number of scalars and vectors can be written (every node
  //  in the OutputData doubly linked lists), all in binary floats format

  int nout_vars = outvars.size();
  if (out_params.compute_moments) {
    nout_vars *= 4;
  }
  int nout_mbs = outmbs.size();
  int nout1 = ((outmbs[0].oie - outmbs[0].ois + 1)/out_params.coarsen_factor);
  int nout2 = ((outmbs[0].oje - outmbs[0].ojs + 1)/out_params.coarsen_factor);
  int nout3 = ((outmbs[0].oke - outmbs[0].oks + 1)/out_params.coarsen_factor);
  int cells = nout1*nout2*nout3;


  // ois, oie, ojs, oje, oks, oke + il1, il2, il3, level +
  // x1min, x1max, x2min, x2max, x3min, x3max + data
  std::size_t data_size = 10*sizeof(int32_t) + 6*sizeof(Real)
                        + (cells*nout_vars)*sizeof(float);

  int ns_mbs = pm->gids_eachrank[global_variable::my_rank];
  int nb_mbs = pm->nmb_eachrank[global_variable::my_rank];

  // allocate 1D vector of floats used to convert and output data
  char *data = new char[nb_mbs*data_size];
  float *single_data = new float[cells];

  // Loop over MeshBlocks
  for (int m=0; m<nout_mbs; ++m) {
    char *pdata=&(data[m*data_size]);
    LogicalLocation loc = pm->lloc_eachmb[outmbs[m].mb_gid];
    // of the starting indexes maybe I need to subtract of nghost,
    // divide by coarsen factor, and then add nghost back in
    int ois = outmbs[m].ois;
    int oie = outmbs[m].ois+nout1-1;
    int ojs = outmbs[m].ojs;
    int oje = outmbs[m].ojs+nout2-1;
    int oks = outmbs[m].oks;
    int oke = outmbs[m].oks+nout3-1;

    // output indexing for MB
    int32_t nx = (int32_t)(ois);
    memcpy(pdata,&(nx),sizeof(nx));
    pdata+=sizeof(nx);
    nx = (int32_t)(oie);
    memcpy(pdata,&(nx),sizeof(nx));
    pdata+=sizeof(nx);
    nx = (int32_t)(ojs);
    memcpy(pdata,&(nx),sizeof(nx));
    pdata+=sizeof(nx);
    nx = (int32_t)(oje);
    memcpy(pdata,&(nx),sizeof(nx));
    pdata+=sizeof(nx);
    nx = (int32_t)(oks);
    memcpy(pdata,&(nx),sizeof(nx));
    pdata+=sizeof(nx);
    nx = (int32_t)(oke);
    memcpy(pdata,&(nx),sizeof(nx));
    pdata+=sizeof(nx);


    // TODO(@DBF): not sure how to shift these properly for the reduced grid
    // logical location lx1, lx2, lx3
    nx = (int32_t)(loc.lx1);
    memcpy(pdata,&(nx),sizeof(nx));
    pdata+=sizeof(nx);
    nx = (int32_t)(loc.lx2);
    memcpy(pdata,&(nx),sizeof(nx));
    pdata+=sizeof(nx);
    nx = (int32_t)(loc.lx3);
    memcpy(pdata,&(nx),sizeof(nx));
    pdata+=sizeof(nx);

    // TODO(@DBF): This probably won't work for AMR
    // physical refinement level
    nx = (int32_t)(loc.level-pm->root_level);
    memcpy(pdata,&(nx),sizeof(nx));
    pdata+=sizeof(nx);

    // coordinate location
    Real xv = outmbs[m].x1min;
    memcpy(pdata,&(xv),sizeof(xv));
    pdata+=sizeof(xv);
    xv = outmbs[m].x1max;
    memcpy(pdata,&(xv),sizeof(xv));
    pdata+=sizeof(xv);
    xv = outmbs[m].x2min;
    memcpy(pdata,&(xv),sizeof(xv));
    pdata+=sizeof(xv);
    xv = outmbs[m].x2max;
    memcpy(pdata,&(xv),sizeof(xv));
    pdata+=sizeof(xv);
    xv = outmbs[m].x3min;
    memcpy(pdata,&(xv),sizeof(xv));
    pdata+=sizeof(xv);
    xv = outmbs[m].x3max;
    memcpy(pdata,&(xv),sizeof(xv));
    pdata+=sizeof(xv);

    // output variables
    float tmp_data;
    for (int n=0; n<nout_vars; n++) {
      int cnt=0;
      for (int k=oks; k<=oke; k++) {
        for (int j=ojs; j<=oje; j++) {
          for (int i=ois; i<=oie; i++) {
            tmp_data = static_cast<float>(outarray(n,m,k-oks,j-ojs,i-ois));
            single_data[cnt] = tmp_data;
            cnt++;
          }
        }
      }
      memcpy(pdata,single_data,cells*sizeof(float));
      pdata+=cells*sizeof(float);
    }
  }

  // now write Coarsenedbinary data
  // check if elements larger than 2^31
  if (data_size*nb_mbs<=2147483648) {
    // now write Coarsenedbinary data in parallel
    std::size_t myoffset=header_offset;
    if (!single_file_per_rank) {
      myoffset += data_size*ns_mbs;
    }
    cbinfile.Write_any_type_at_all(data,(data_size*nb_mbs),myoffset,"byte",
                                    single_file_per_rank);
  } else {
    // check if elements larger than 2^31
    if (data_size*nb_mbs<=2147483648) {
      // now write binary data in parallel
      std::size_t myoffset=header_offset;
      if (!single_file_per_rank) {
        myoffset += data_size*ns_mbs;
      }
      cbinfile.Write_any_type_at_all(data,(data_size*nb_mbs),myoffset,"byte",
                                      single_file_per_rank);
    } else {
      // write data over each MeshBlock sequentially and in parallel
      // calculate max/min number of MeshBlocks across all ranks
      noutmbs_max = pm->nmb_eachrank[0];
      noutmbs_min = pm->nmb_eachrank[0];
      for (int i=0; i<(global_variable::nranks); ++i) {
        noutmbs_max = std::max(noutmbs_max,pm->nmb_eachrank[i]);
        noutmbs_min = std::min(noutmbs_min,pm->nmb_eachrank[i]);
      }
      for (int m=0;  m<noutmbs_max; ++m) {
        char *pdata=&(data[m*data_size]);
        std::size_t myoffset=header_offset + data_size*m;
        if (!single_file_per_rank) {
          myoffset += data_size*ns_mbs;
        }
        // every rank has a MB to write, so write collectively
        if (m < noutmbs_min) {
          if (cbinfile.Write_any_type_at_all(pdata,(data_size),myoffset,"byte",
                                              single_file_per_rank) != data_size) {
            std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "binary data not written correctly to binary file, "
                << "binary file is broken." << std::endl;
            exit(EXIT_FAILURE);
          }
        // some ranks are finished writing, so use non-collective write
        } else if (m < pm->nmb_thisrank) {
          if (cbinfile.Write_any_type_at(pdata,(data_size),myoffset,"byte",
                                          single_file_per_rank) != data_size) {
            std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                 << std::endl << "binary data not written correctly to binary file, "
                 << "binary file is broken." << std::endl;
            exit(EXIT_FAILURE);
          }
        }
      }
    }
  }

  // close the output file and clean up ptrs to data
  cbinfile.Close(single_file_per_rank);
  delete [] data;
  delete [] single_data;

  // increment counters
  out_params.file_number++;
  if (out_params.last_time < 0.0) {
    out_params.last_time = pm->time;
  } else {
    out_params.last_time += out_params.dt;
  }
  pin->SetInteger(out_params.block_name, "file_number", out_params.file_number);
  pin->SetReal(out_params.block_name, "last_time", out_params.last_time);

  return;
}
