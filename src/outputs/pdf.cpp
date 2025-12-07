//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file pdf.cpp
//  \brief writes N-D PDF output data --- Drummond B Fielding
//
//  PDFs can be 1D, 2D, 3D, or 4D and can be either mass or volume weighted.
//  Each PDF is stored in its own directory with binary data files and an ASCII header.
//  Variables can be any from var_choice[] in outputs.hpp, including coordinate variables.
//
//  Input format (new N-D):
//    variable_1 = <var>   bin1_min = <min>   bin1_max = <max>   nbin1 = <n>   logscale1 = <bool>
//    variable_2 = <var>   bin2_min = <min>   bin2_max = <max>   nbin2 = <n>   logscale2 = <bool>
//    ...
//
//  Output:
//    - ASCII header file: <basename>.header.pdf (written once)
//    - Binary data files: <basename>.XXXXX.pdf (one per output time)

#include <sys/stat.h>  // mkdir

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#include "athena.hpp"
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

  // Initialize PDFData with N-D parameters
  pdf_data.Initialize(op.pdf_ndim, op.pdf_nbin, op.pdf_bin_min,
                      op.pdf_bin_max, op.pdf_logscale);
  pdf_data.mass_weighted = op.mass_weighted;
  pdf_data.PopulateBinEdges();
}


//----------------------------------------------------------------------------------------
//! \fn void PDFOutput::LoadOutputData()
//  \brief Computes N-D PDF histogram over all MeshBlocks

void PDFOutput::LoadOutputData(Mesh *pm) {
  // Compute derived variables for all PDF dimensions
  if (out_params.contains_derived) {
    for (int d = 0; d < out_params.pdf_ndim; ++d) {
      ComputeDerivedVariable(out_params.pdf_variables[d], pm);
    }
  }

  // Get physics pointer for mass weighting
  DvceArray5D<Real> *u0_ptr = nullptr;
  if (pm->pmb_pack->phydro != nullptr) {
    u0_ptr = &(pm->pmb_pack->phydro->u0);
  } else if (pm->pmb_pack->pmhd != nullptr) {
    u0_ptr = &(pm->pmb_pack->pmhd->u0);
  } else if (pm->pmb_pack->pz4c != nullptr) {
    u0_ptr = &(pm->pmb_pack->pz4c->u0);
  }

  if (u0_ptr == nullptr && pdf_data.mass_weighted) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Mass-weighted PDF requires physics module" << std::endl;
    exit(EXIT_FAILURE);
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
  int nx2 = indcs.nx2 + 2*indcs.ng;
  int nx3 = indcs.nx3 + 2*indcs.ng;

  // Copy variable data to device array
  DvceArray5D<Real> outvars_device("outvars_device", outvars.size(), nmb, nx3, nx2, nx1);
  for (std::size_t i = 0; i < outvars.size(); ++i) {
    auto d_slice = Kokkos::subview(*(outvars[i].data_ptr),
        Kokkos::ALL(), outvars[i].data_index, Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL());
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
  bool mass_weighted = pdf_data.mass_weighted;

  // Create device-accessible copies of PDF parameters
  Kokkos::View<int[PDFData::MAX_DIM]> d_nbin("d_nbin");
  Kokkos::View<int[PDFData::MAX_DIM]> d_nbin_with_overflow("d_nbin_with_overflow");
  Kokkos::View<int[PDFData::MAX_DIM]> d_stride("d_stride");
  Kokkos::View<Real[PDFData::MAX_DIM]> d_step_size("d_step_size");
  Kokkos::View<Real[PDFData::MAX_DIM]> d_bin_min("d_bin_min");
  Kokkos::View<bool[PDFData::MAX_DIM]> d_logscale("d_logscale");

  auto h_nbin = Kokkos::create_mirror_view(d_nbin);
  auto h_nbin_with_overflow = Kokkos::create_mirror_view(d_nbin_with_overflow);
  auto h_stride = Kokkos::create_mirror_view(d_stride);
  auto h_step_size = Kokkos::create_mirror_view(d_step_size);
  auto h_bin_min = Kokkos::create_mirror_view(d_bin_min);
  auto h_logscale = Kokkos::create_mirror_view(d_logscale);

  for (int d = 0; d < PDFData::MAX_DIM; ++d) {
    h_nbin(d) = pdf_data.nbin[d];
    h_nbin_with_overflow(d) = pdf_data.nbin_with_overflow[d];
    h_stride(d) = pdf_data.stride[d];
    h_step_size(d) = pdf_data.step_size[d];
    h_bin_min(d) = pdf_data.bin_min[d];
    h_logscale(d) = pdf_data.logscale[d];
  }

  Kokkos::deep_copy(d_nbin, h_nbin);
  Kokkos::deep_copy(d_nbin_with_overflow, h_nbin_with_overflow);
  Kokkos::deep_copy(d_stride, h_stride);
  Kokkos::deep_copy(d_step_size, h_step_size);
  Kokkos::deep_copy(d_bin_min, h_bin_min);
  Kokkos::deep_copy(d_logscale, h_logscale);
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
      if (d_logscale(d)) {
        if (val <= 0.0 || val < d_bin_min(d)) {
          bin_idx = 0;  // underflow
        } else if (val >= d_bin_min(d) * std::pow(10.0, d_nbin(d) * d_step_size(d))) {
          bin_idx = d_nbin(d) + 1;  // overflow
        } else {
          bin_idx = static_cast<int>(std::log10(val / d_bin_min(d)) / d_step_size(d)) + 1;
        }
      } else {
        if (val < d_bin_min(d)) {
          bin_idx = 0;  // underflow
        } else if (val >= d_bin_min(d) + d_nbin(d) * d_step_size(d)) {
          bin_idx = d_nbin(d) + 1;  // overflow
        } else {
          bin_idx = static_cast<int>((val - d_bin_min(d)) / d_step_size(d)) + 1;
        }
      }

      // Clamp to valid range
      bin_idx = (bin_idx < 0) ? 0 : bin_idx;
      bin_idx = (bin_idx > d_nbin(d) + 1) ? d_nbin(d) + 1 : bin_idx;

      flat_idx += bin_idx * d_stride(d);
    }

    // Compute weight (volume or mass)
    Real weight = size.d_view(m).dx1 * size.d_view(m).dx2 * size.d_view(m).dx3;
    if (mass_weighted) {
      weight *= u0_(m, IDN, k, j, i);
    }

    // Atomic add to histogram
    auto res = scatter.access();
    res(flat_idx) += weight;
  });

  // Reduce ScatterView to result array
  Kokkos::Experimental::contribute(result, scatter);
  Kokkos::fence();

  // MPI reduce across ranks
#if MPI_PARALLEL_ENABLED
  if (global_variable::my_rank == 0) {
    MPI_Reduce(MPI_IN_PLACE, result.data(), total_bins,
               MPI_ATHENA_REAL, MPI_SUM, 0, MPI_COMM_WORLD);
  } else {
    MPI_Reduce(result.data(), result.data(), total_bins,
               MPI_ATHENA_REAL, MPI_SUM, 0, MPI_COMM_WORLD);
  }
#endif
}


//----------------------------------------------------------------------------------------
//! \fn void PDFOutput::WriteOutputFile()
//  \brief Writes N-D PDF to binary file with ASCII header

void PDFOutput::WriteOutputFile(Mesh *pm, ParameterInput *pin) {
  // Only master rank writes files
  if (global_variable::my_rank == 0) {
    // Construct directory name
    std::string dir_name = "pdf_" + out_params.file_id;
    for (int d = 1; d < out_params.pdf_ndim; ++d) {
      dir_name += "_" + out_params.pdf_variables[d];
    }

    // Write ASCII header file (once)
    if (!(pdf_data.bins_written)) {
      std::string header_fname = dir_name + "/" + out_params.file_basename + ".header.pdf";

      FILE *hfile;
      if ((hfile = std::fopen(header_fname.c_str(), "w")) == nullptr) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
            << std::endl << "Header file '" << header_fname << "' could not be opened"
            << std::endl;
        exit(EXIT_FAILURE);
      }

      // Write header metadata
      std::fprintf(hfile, "# AthenaK N-D PDF Output\n");
      std::fprintf(hfile, "ndim = %d\n", pdf_data.ndim);
      std::fprintf(hfile, "mass_weighted = %s\n", pdf_data.mass_weighted ? "true" : "false");
      std::fprintf(hfile, "total_bins = %d\n", pdf_data.total_bins);
      std::fprintf(hfile, "\n");

      // Write dimension info
      for (int d = 0; d < pdf_data.ndim; ++d) {
        std::fprintf(hfile, "# Dimension %d\n", d + 1);
        std::fprintf(hfile, "variable_%d = %s\n", d + 1, out_params.pdf_variables[d].c_str());
        std::fprintf(hfile, "nbin%d = %d\n", d + 1, pdf_data.nbin[d]);
        std::fprintf(hfile, "bin%d_min = %.15e\n", d + 1, pdf_data.bin_min[d]);
        std::fprintf(hfile, "bin%d_max = %.15e\n", d + 1, pdf_data.bin_max[d]);
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
    std::string data_fname = dir_name + "/" + out_params.file_basename + "."
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

    // Write histogram data as doubles
    for (int n = 0; n < pdf_data.total_bins; ++n) {
      double val = static_cast<double>(result_host(n));
      std::fwrite(&val, sizeof(double), 1, pfile);
    }

    std::fclose(pfile);
  }

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
