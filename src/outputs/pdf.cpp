//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file pdf.cpp
//  \brief writes N-D PDF output data --- Drummond B Fielding
//
//  PDFs can be 1D, 2D, 3D, or 4D and can be volume, mass, or variable weighted.
//  Each PDF is stored in its own directory with binary data files and an ASCII header.
//  Variables can be any from var_choice[] in outputs.hpp, including coordinate variables.
//
//  Input format (new N-D):
//    variable_1 = <var>   bin1_min = <min>   bin1_max = <max>   nbin1 = <n>
//    scale1 = linear|log|symlog   linthresh1 = <value> (symlog only)
//    variable_2 = <var>   bin2_min = <min>   bin2_max = <max>   nbin2 = <n>
//    scale2 = linear|log|symlog   linthresh2 = <value> (symlog only)
//    ...
//    weight = volume|mass|variable   weight_variable = <var>  (optional)
//
//  Output:
//    - ASCII header file: <basename>.header.pdf (written once)
//    - Binary data files: <basename>.XXXXX.pdf (one per output time)

#include <sys/stat.h>  // mkdir

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#include "athena.hpp"
#include "file_sharding.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "z4c/z4c.hpp"
#include "outputs.hpp"

// ScatterView is not part of Kokkos core interface
#include "Kokkos_ScatterView.hpp"


//----------------------------------------------------------------------------------------
// Constructor: initializes N-D PDF data structures

PDFOutput::PDFOutput(ParameterInput *pin, Mesh *pm, OutputParameters op) :
  BaseTypeOutput(pin, pm, op) {

  // Create directory for outputs
  std::string dir_name = "pdf_" + op.file_id;
  for (int d = 1; d < op.pdf_ndim; ++d) {
    dir_name += "_" + op.pdf_variables[d];
  }
  mkdir(dir_name.c_str(), 0775);
  if (op.file_shard_mode != FileShardMode::shared) {
    std::string shard_subdir = dir_name + "/" + ShardDirectoryName(op.file_shard_mode);
    mkdir(shard_subdir.c_str(), 0775);
  }

  // Initialize PDFData with N-D parameters
  pdf_data.Initialize(op.pdf_ndim, op.pdf_nbin, op.pdf_bin_min,
                      op.pdf_bin_max, op.pdf_scale, op.pdf_linthresh);
  pdf_data.PopulateBinEdges();

  int expected_vars = out_params.pdf_ndim;
  if (out_params.pdf_weight.compare("variable") == 0) {
    expected_vars += 1;
  }
  if (out_params.pdf_ndim > 0 &&
      outvars.size() != static_cast<std::size_t>(expected_vars)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "PDF output block '" << out_params.block_name
              << "' must use scalar variables (one per PDF dimension"
              << (expected_vars > out_params.pdf_ndim ? " and weight variable" : "")
              << ")" << std::endl;
    exit(EXIT_FAILURE);
  }
}


//----------------------------------------------------------------------------------------
//! \fn void PDFOutput::LoadOutputData()
//  \brief Computes N-D PDF histogram over all MeshBlocks

void PDFOutput::LoadOutputData(Mesh *pm) {
  // Compute derived variables for all PDF dimensions
  if (out_params.contains_derived) {
    // Reset derived variable index before computing to ensure proper indexing.
    out_params.i_derived = 0;
    for (int d = 0; d < out_params.pdf_ndim; ++d) {
      ComputeDerivedVariable(out_params.pdf_variables[d], pm);
    }
    if (out_params.pdf_weight.compare("variable") == 0 &&
        !out_params.pdf_weight_variable.empty()) {
      ComputeDerivedVariable(out_params.pdf_weight_variable, pm);
    }
  }

  int weight_mode = 0;  // 0=volume, 1=mass, 2=variable
  if (out_params.pdf_weight.compare("mass") == 0) {
    weight_mode = 1;
  } else if (out_params.pdf_weight.compare("variable") == 0) {
    weight_mode = 2;
  }

  // Get physics pointer for mass weighting
  DvceArray5D<Real> *u0_ptr = nullptr;
  if (weight_mode == 1) {
    if (pm->pmb_pack->phydro != nullptr) {
      u0_ptr = &(pm->pmb_pack->phydro->u0);
    } else if (pm->pmb_pack->pmhd != nullptr) {
      u0_ptr = &(pm->pmb_pack->pmhd->u0);
    } else if (pm->pmb_pack->pz4c != nullptr) {
      u0_ptr = &(pm->pmb_pack->pz4c->u0);
    }

    if (u0_ptr == nullptr) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Mass-weighted PDF requires physics module" << std::endl;
      exit(EXIT_FAILURE);
    }
  }

  DvceArray5D<Real> u0_;
  if (u0_ptr != nullptr) {
    u0_ = *u0_ptr;
  }

  auto &size = pm->pmb_pack->pmb->mb_size;
  auto &indcs = pm->pmb_pack->pmesh->mb_indcs;
  int is = indcs.is; int ie = indcs.ie;
  int js = indcs.js; int je = indcs.je;
  int ks = indcs.ks; int ke = indcs.ke;

  int nmb = pm->pmb_pack->nmb_thispack;
  int nx1 = indcs.nx1 + 2*indcs.ng;
  int nx2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*indcs.ng) : 1;
  int nx3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*indcs.ng) : 1;

  // Copy MeshBlock data from host to device
  // Use explicit ranges for ALL dimensions because:
  // 1. Physics arrays may be allocated with more meshblocks than nmb_thispack
  // 2. Need to ensure source and target shapes match exactly for deep_copy
  DvceArray5D<Real> outvars_device("outvars_device", outvars.size(), nmb, nx3, nx2, nx1);
  for (std::size_t i = 0; i < outvars.size(); ++i) {
      auto d_slice = Kokkos::subview(*(outvars[i].data_ptr),
          Kokkos::make_pair(0, nmb), outvars[i].data_index,
          Kokkos::make_pair(0, nx3), Kokkos::make_pair(0, nx2), Kokkos::make_pair(0, nx1));
      auto d_target_slice = Kokkos::subview(outvars_device, i,
          Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL());
      Kokkos::deep_copy(d_target_slice, d_slice);
  }
  Kokkos::fence();

  // Get PDF parameters
  auto result = pdf_data.result_;
  auto scatter = pdf_data.scatter_result;
  int ndim = pdf_data.ndim;
  int total_bins = pdf_data.total_bins;
  int weight_idx = -1;
  if (weight_mode == 2) {
    weight_idx = out_params.pdf_ndim;
  }

  // Create device-accessible copies of PDF parameters
  Kokkos::View<int[PDFData::MAX_DIM]> d_nbin("d_nbin");
  Kokkos::View<int[PDFData::MAX_DIM]> d_stride("d_stride");
  Kokkos::View<int[PDFData::MAX_DIM]> d_scale("d_scale");
  Kokkos::View<Real[PDFData::MAX_DIM]> d_step_size("d_step_size");
  Kokkos::View<Real[PDFData::MAX_DIM]> d_bin_min("d_bin_min");
  Kokkos::View<Real[PDFData::MAX_DIM]> d_bin_max("d_bin_max");
  Kokkos::View<Real[PDFData::MAX_DIM]> d_transformed_min("d_transformed_min");
  Kokkos::View<Real[PDFData::MAX_DIM]> d_linthresh("d_linthresh");

  auto h_nbin = Kokkos::create_mirror_view(d_nbin);
  auto h_stride = Kokkos::create_mirror_view(d_stride);
  auto h_scale = Kokkos::create_mirror_view(d_scale);
  auto h_step_size = Kokkos::create_mirror_view(d_step_size);
  auto h_bin_min = Kokkos::create_mirror_view(d_bin_min);
  auto h_bin_max = Kokkos::create_mirror_view(d_bin_max);
  auto h_transformed_min = Kokkos::create_mirror_view(d_transformed_min);
  auto h_linthresh = Kokkos::create_mirror_view(d_linthresh);

  for (int d = 0; d < PDFData::MAX_DIM; ++d) {
    h_nbin(d) = pdf_data.nbin[d];
    h_stride(d) = pdf_data.stride[d];
    h_scale(d) = pdf_data.scale[d];
    h_step_size(d) = pdf_data.step_size[d];
    h_bin_min(d) = pdf_data.bin_min[d];
    h_bin_max(d) = pdf_data.bin_max[d];
    h_transformed_min(d) = pdf_data.transformed_min[d];
    h_linthresh(d) = pdf_data.linthresh[d];
  }

  Kokkos::deep_copy(d_nbin, h_nbin);
  Kokkos::deep_copy(d_stride, h_stride);
  Kokkos::deep_copy(d_scale, h_scale);
  Kokkos::deep_copy(d_step_size, h_step_size);
  Kokkos::deep_copy(d_bin_min, h_bin_min);
  Kokkos::deep_copy(d_bin_max, h_bin_max);
  Kokkos::deep_copy(d_transformed_min, h_transformed_min);
  Kokkos::deep_copy(d_linthresh, h_linthresh);
  Kokkos::fence();

  // Reset ScatterView and result array
  scatter.reset();
  Kokkos::deep_copy(result, 0);
  Kokkos::fence();

  // N-D binning kernel
  par_for("pdf_nd", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    // Compute flat index for N-D bin
    int flat_idx = 0;
    for (int d = 0; d < ndim; ++d) {
      Real val = outvars_device(d, m, k, j, i);
      int bin_idx;

      // Determine bin index for this dimension
      if (val < d_bin_min(d)) {
        bin_idx = 0;  // underflow
      } else if (!(val < d_bin_max(d))) {
        bin_idx = d_nbin(d) + 1;  // overflow
      } else {
        Real transformed = PDFTransformValue(val, d_scale(d), d_linthresh(d));
        Real bin_pos = (transformed - d_transformed_min(d)) / d_step_size(d);
        if (bin_pos < 0.0) {
          bin_idx = 0;
        } else if (bin_pos >= static_cast<Real>(d_nbin(d))) {
          bin_idx = d_nbin(d);
        } else {
          bin_idx = static_cast<int>(bin_pos) + 1;
        }
      }

      // Clamp to valid range
      bin_idx = (bin_idx < 0) ? 0 : bin_idx;
      bin_idx = (bin_idx > d_nbin(d) + 1) ? d_nbin(d) + 1 : bin_idx;

      flat_idx += bin_idx * d_stride(d);
    }

    // Compute weight (volume, mass, or variable), always scaled by cell volume
    Real weight = size.d_view(m).dx1 * size.d_view(m).dx2 * size.d_view(m).dx3;
    if (weight_mode == 1) {
      weight *= u0_(m, IDN, k, j, i);
    } else if (weight_mode == 2) {
      weight *= outvars_device(weight_idx, m, k, j, i);
    }

    // Atomic add to histogram
    auto res = scatter.access();
    res(flat_idx) += weight;
  });

  // Reduce ScatterView to result array
  Kokkos::Experimental::contribute(result, scatter);
  Kokkos::fence();

  // Reduce within the shard communicator for shared and per-node modes.
#if MPI_PARALLEL_ENABLED
  if (out_params.file_shard_mode == FileShardMode::shared) {
    if (global_variable::my_rank == 0) {
      MPI_Reduce(MPI_IN_PLACE, result.data(), total_bins,
                 MPI_ATHENA_REAL, MPI_SUM, 0, MPI_COMM_WORLD);
    } else {
      MPI_Reduce(result.data(), result.data(), total_bins,
                 MPI_ATHENA_REAL, MPI_SUM, 0, MPI_COMM_WORLD);
    }
  } else if (out_params.file_shard_mode == FileShardMode::per_node) {
    if (global_variable::rank_in_node == 0) {
      MPI_Reduce(MPI_IN_PLACE, result.data(), total_bins,
                 MPI_ATHENA_REAL, MPI_SUM, 0, global_variable::node_comm);
    } else {
      MPI_Reduce(result.data(), result.data(), total_bins,
                 MPI_ATHENA_REAL, MPI_SUM, 0, global_variable::node_comm);
    }
  }
#endif

}


//----------------------------------------------------------------------------------------
//! \fn void PDFOutput::WriteOutputFile()
//  \brief Writes N-D PDF to binary file with ASCII header

void PDFOutput::WriteOutputFile(Mesh *pm, ParameterInput *pin) {
  FileShardMode shard_mode = out_params.file_shard_mode;

  if (!IsShardWriter(shard_mode)) {
    // Still need to update counters on all ranks
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

  // Construct directory name
  std::string dir_name = "pdf_" + out_params.file_id;
  for (int d = 1; d < out_params.pdf_ndim; ++d) {
    dir_name += "_" + out_params.pdf_variables[d];
  }

  // Build rank-specific prefix if needed
  std::string path_prefix = dir_name + "/";
  if (shard_mode != FileShardMode::shared) {
    path_prefix = dir_name + "/" + ShardDirectoryName(shard_mode);
  }

  // Write ASCII header file (once)
  if (!(pdf_data.bins_written)) {
    std::string header_fname = path_prefix + out_params.file_basename
                             + ".header.pdf";
    FILE *hfile;
    if ((hfile = std::fopen(header_fname.c_str(), "w")) == nullptr) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
          << std::endl << "Header file '" << header_fname << "' could not be opened"
          << std::endl;
      exit(EXIT_FAILURE);
    }

    // Write header metadata
    std::fprintf(hfile, "# AthenaK N-D PDF Output\n");
    std::fprintf(hfile, "format = %s\n",
                 shard_mode == FileShardMode::shared ? "dense" : "sparse_coo");
    std::fprintf(hfile, "distribution = %s\n", ShardDistributionName(shard_mode));
    std::fprintf(hfile, "ndim = %d\n", pdf_data.ndim);
    std::fprintf(hfile, "weight = %s\n", out_params.pdf_weight.c_str());
    if (out_params.pdf_weight.compare("variable") == 0) {
      std::fprintf(hfile, "weight_variable = %s\n",
                   out_params.pdf_weight_variable.c_str());
    }
    std::fprintf(hfile, "total_bins = %d\n", pdf_data.total_bins);
    std::fprintf(hfile, "\n");

    // Write dimension info
    for (int d = 0; d < pdf_data.ndim; ++d) {
      std::fprintf(hfile, "# Dimension %d\n", d + 1);
      std::fprintf(hfile, "variable_%d = %s\n", d + 1, out_params.pdf_variables[d].c_str());
      std::fprintf(hfile, "nbin%d = %d\n", d + 1, pdf_data.nbin[d]);
      std::fprintf(hfile, "bin%d_min = %.15e\n", d + 1, pdf_data.bin_min[d]);
      std::fprintf(hfile, "bin%d_max = %.15e\n", d + 1, pdf_data.bin_max[d]);
      std::fprintf(hfile, "scale%d = %s\n", d + 1, PDFScaleName(pdf_data.scale[d]));
      if (pdf_data.scale[d] == PDF_SCALE_SYMLOG) {
        std::fprintf(hfile, "linthresh%d = %.15e\n", d + 1, pdf_data.linthresh[d]);
      }
      std::fprintf(hfile, "logscale%d = %s\n", d + 1, pdf_data.logscale[d] ? "true" : "false");
      std::fprintf(hfile, "stride%d = %d\n", d + 1, pdf_data.stride[d]);
      std::fprintf(hfile, "\n");
    }

    // Write bin edges for each dimension
    std::fprintf(hfile, "# Bin edges (nbin+1 values per dimension)\n");
    for (int d = 0; d < pdf_data.ndim; ++d) {
      auto bins_host = Kokkos::create_mirror_view(pdf_data.bin_edges[d]);
      Kokkos::deep_copy(bins_host, pdf_data.bin_edges[d]);
      Kokkos::fence();

      std::fprintf(hfile, "bin_edges_%d =", d + 1);
      for (int n = 0; n <= pdf_data.nbin[d]; ++n) {
        std::fprintf(hfile, " %.15e", bins_host(n));
      }
      std::fprintf(hfile, "\n");
    }

    std::fclose(hfile);
    pdf_data.bins_written = true;
  }

  // Write binary data file
  char number[6];
  std::snprintf(number, sizeof(number), "%05d", out_params.file_number);
  std::string data_fname = path_prefix + out_params.file_basename + "."
                         + std::string(number) + ".pdf";

  FILE *pfile;
  if ((pfile = std::fopen(data_fname.c_str(), "wb")) == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
        << std::endl << "Data file '" << data_fname << "' could not be opened"
        << std::endl;
    exit(EXIT_FAILURE);
  }

  // Write time as first 8 bytes (double)
  double time_val = static_cast<double>(pm->time);
  std::fwrite(&time_val, sizeof(double), 1, pfile);

  // Copy result to host and write
  auto result_host = Kokkos::create_mirror_view(pdf_data.result_);
  Kokkos::deep_copy(result_host, pdf_data.result_);
  Kokkos::fence();

  if (shard_mode != FileShardMode::shared) {
    // Sparse COO: compact nonzero bins into (idx, val) pairs.
    // Layout after the leading time: uint32 nnz | uint32 idx[nnz] | double val[nnz]
    std::vector<uint32_t> idx_buf;
    std::vector<double> val_buf;
    for (int n = 0; n < pdf_data.total_bins; ++n) {
      double v = static_cast<double>(result_host(n));
      if (v != 0.0) {
        idx_buf.push_back(static_cast<uint32_t>(n));
        val_buf.push_back(v);
      }
    }
    uint32_t nnz = static_cast<uint32_t>(idx_buf.size());
    std::fwrite(&nnz, sizeof(uint32_t), 1, pfile);
    if (nnz > 0) {
      std::fwrite(idx_buf.data(), sizeof(uint32_t), nnz, pfile);
      std::fwrite(val_buf.data(), sizeof(double), nnz, pfile);
    }
  } else {
    // Dense write (only reached on rank 0 after global reduction).
    std::vector<double> write_buf(pdf_data.total_bins);
    for (int n = 0; n < pdf_data.total_bins; ++n) {
      write_buf[n] = static_cast<double>(result_host(n));
    }
    std::fwrite(write_buf.data(), sizeof(double), pdf_data.total_bins, pfile);
  }

  std::fclose(pfile);
  
  // Update counters
  out_params.file_number++;
  if (out_params.last_time < 0.0) {
    out_params.last_time = pm->time;
  } else {
    out_params.last_time += out_params.dt;
  }
  pin->SetInteger(out_params.block_name, "file_number", out_params.file_number);
  pin->SetReal(out_params.block_name, "last_time", out_params.last_time);
}
