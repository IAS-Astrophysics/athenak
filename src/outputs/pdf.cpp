//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file pdf.cpp
//  \brief writes pdf output data --- Drummond B Fielding
//  PDFs can be 1d or 2d and can be either mass or weightume weighted.
//  the user can specify more than one pdf to be calculated
//  each pdf will be stored in its own directory with a new file for each output
//  the user should be able to specify either from the var_choice listed in outputs.hpp
//  or by specifying a custom variable in the pgen
//  but I will need to figure out how to do multiple user defined variables
//
//  The user must also specify the number of bins and the range of the bins and
//  if they should be log or linearly spaced.
//
//  These bins are written to their own file when the first output is written
//  the pdfs are written to their own file for each output

#include <sys/stat.h>  // mkdir

#include <cstdio>
#include <cstdlib>
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
// Constructor: also calls BaseTypeOutput base class constructor
// this is not right yet
PDFOutput::PDFOutput(ParameterInput *pin, Mesh *pm, OutputParameters op) :
  BaseTypeOutput(pin, pm, op) , pdf_data(op.nbin2 == 0 ? 1 : 2, op.nbin, op.nbin2) {
  // create directories for outputs
  // create a new directory for each pdf
  std::string dir_name;
  dir_name.assign("pdf_");
  dir_name.append(op.file_id);
  if (pdf_data.pdf_dimension == 2) {
    dir_name.append("_");
    dir_name.append(op.variable_2);
  }
  mkdir(dir_name.c_str(),0775);

  pdf_data.mass_weighted = op.mass_weighted;
  pdf_data.logscale = op.logscale;

  // throw an error if the user tries to use logscale
  // with a negative bin_min for both 1D and 2D
  if (op.logscale && op.bin_min <= 0.0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
      << std::endl << "logscale is true but bin_min <= 0.0" << std::endl;
    exit(EXIT_FAILURE);
  }

  if (op.logscale2 && op.bin2_min <= 0.0 && pdf_data.pdf_dimension == 2) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
      << std::endl << "logscale2 is true but bin2_min <= 0.0" << std::endl;
    exit(EXIT_FAILURE);
  }


  // Create bins for the pdf
  // Create mirror view on host
  auto bins_host = Kokkos::create_mirror_view(pdf_data.bins);

  // Populate bins_host
  if (op.logscale) {
    Real logbin_min = std::log10(op.bin_min);
    Real logbin_max = std::log10(op.bin_max);
    for (int i = 0; i <= op.nbin; i++) {
      bins_host(i) = std::pow(10, logbin_min + i * (logbin_max - logbin_min) / op.nbin);
    }
  } else {
    Real bin_step = (op.bin_max - op.bin_min) / op.nbin;
    for (int i = 0; i <= op.nbin; i++) {
      bins_host(i) = op.bin_min + i * bin_step;
    }
  }

  // Copy back to device
  Kokkos::deep_copy(pdf_data.bins, bins_host);
  Kokkos::fence();

  // Update the step size in pdf_data
  pdf_data.step_size = op.logscale ?
                       (std::log10(op.bin_max) - std::log10(op.bin_min)) / op.nbin :
                       (op.bin_max - op.bin_min) / op.nbin;




  // Create second bins for the pdf if 2D
  if (pdf_data.pdf_dimension == 2) {
    pdf_data.logscale2 = op.logscale2;
    auto bins2_host = Kokkos::create_mirror_view(pdf_data.bins2);

    if (op.logscale2) {
      Real logbin_min2 = std::log10(op.bin2_min);
      Real logbin_max2 = std::log10(op.bin2_max);
      for (int i = 0; i <= op.nbin2; i++) {
        bins2_host(i) = std::pow(10, logbin_min2+i*(logbin_max2-logbin_min2)/op.nbin2);
      }
    } else {
      Real step2 = (op.bin2_max - op.bin2_min) / op.nbin2;
      for (int i = 0; i <= op.nbin2; i++) {
        bins2_host(i) = op.bin2_min + i * step2;
      }
    }

    // Copy back to device
    Kokkos::deep_copy(pdf_data.bins2, bins2_host);
    Kokkos::fence();

    if (pdf_data.pdf_dimension == 2 && pdf_data.bins2.extent(0) != op.nbin2 + 1) {
      std::cerr << "Error: pdf_data.bins2 size mismatch. Expected size: "
                << op.nbin2 + 1 << ", Actual size: " << pdf_data.bins2.extent(0)
                << std::endl;
      exit(EXIT_FAILURE);
    }


    // Update the step size for bins2
    pdf_data.step_size2 = op.logscale2 ?
                         (std::log10(op.bin2_max) - std::log10(op.bin2_min)) / op.nbin2 :
                         (op.bin2_max - op.bin2_min) / op.nbin2;
  }


  if (pdf_data.pdf_dimension == 2) {
    pdf_data.result_ = DvceArray2D<Real>("result", op.nbin2+2, op.nbin+2);
  } else if (pdf_data.pdf_dimension == 1) {
    pdf_data.result_ = DvceArray2D<Real>("result", 1, op.nbin+2);
  }
  pdf_data.scatter_result = Kokkos::Experimental::ScatterView<Real **, LayoutWrapper>(
    pdf_data.result_
  );
}



//----------------------------------------------------------------------------------------
//! \fn void PDFOutput::LoadOutputData()
//  \brief Wrapper function that cycles through hist_data vector and calls
//  appropriate LoadXXXData() function for that physics

void PDFOutput::LoadOutputData(Mesh *pm) {
  // Calculate derived variables, if required
  // if out_params.variable or out_params.variable_2 not a derived
  // then ComputeDerivedVariable does nothing, so this should be fine
  // although maybe not optimal -- should probably have a way to
  // know beforehand which needs to be computed
  if (out_params.contains_derived) {
    ComputeDerivedVariable(out_params.variable, pm);
    ComputeDerivedVariable(out_params.variable_2, pm);
  }

  // Pointer for initial determination
  DvceArray5D<Real> *u0_ptr = nullptr;

  if (pm->pmb_pack->phydro != nullptr) {
    u0_ptr = &(pm->pmb_pack->phydro->u0);
  } else if (pm->pmb_pack->pmhd != nullptr) {
    u0_ptr = &(pm->pmb_pack->pmhd->u0);
  } else if (pm->pmb_pack->pz4c != nullptr) {
    u0_ptr = &(pm->pmb_pack->pz4c->u0);
  }

  // Check if a valid module was found
  if (u0_ptr == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "No physics module found" << std::endl;
    exit(EXIT_FAILURE);
  }

  // Now assign the reference
  DvceArray5D<Real> &u0_ = *u0_ptr;

  // capture class variables for kernel
  auto &size = pm->pmb_pack->pmb->mb_size;

  // loop over all MeshBlocks in this pack
  auto &indcs = pm->pmb_pack->pmesh->mb_indcs;
  int is = indcs.is; int ie = indcs.ie;
  int js = indcs.js; int je = indcs.je;
  int ks = indcs.ks; int ke = indcs.ke;

  auto result  = pdf_data.result_;
  auto scatter = pdf_data.scatter_result;

  int nmb = pm->pmb_pack->nmb_thispack;
  int nx1 = indcs.nx1 + 2*indcs.ng;
  int nx2 = indcs.nx2 + 2*indcs.ng;
  int nx3 = indcs.nx3 + 2*indcs.ng;

  // Copy MeshBlock data from host to device
  DvceArray5D<Real> outvars_device("outvars_device", outvars.size(), nmb, nx3, nx2, nx1);
  for (std::size_t i = 0; i < outvars.size(); ++i) {
      auto d_slice = Kokkos::subview(*(outvars[i].data_ptr),
      Kokkos::ALL(), outvars[i].data_index, Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL());
      auto d_target_slice = Kokkos::subview(outvars_device, i,
      Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL());
      Kokkos::deep_copy(d_target_slice, d_slice);
  }
  Kokkos::fence();

  //

  // Reset ScatterView from previous output
  scatter.reset();
  // Also reset the histogram from previous call.
  // Currently still required for consistent results between host and device backends, see
  // https://github.com/kokkos/kokkos/issues/6363
  Kokkos::deep_copy(result, 0);
  Kokkos::fence();

  // Capture the necessary data from pdf_data
  auto bins = pdf_data.bins;
  auto bins2 = pdf_data.bins2;
  auto step_size = pdf_data.step_size;
  auto step_size2 = pdf_data.step_size2;
  auto nbin_ = pdf_data.nbin;
  auto nbin2_ = pdf_data.nbin2;
  int pdf_dimension = pdf_data.pdf_dimension;
  bool logscale = pdf_data.logscale;
  bool logscale2 = pdf_data.logscale2;
  bool mass_weighted = pdf_data.mass_weighted;

  par_for("pdf", DevExeSpace(),0,nmb-1,ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    auto &x_val = outvars_device(0, m, k, j, i);
    int x_bin = -1;
    // First handle edge cases explicitly
    if (x_val < bins(0)) {
      x_bin = 0;
    } else if (x_val >= bins(nbin_)) {
      x_bin = nbin_ + 1;
    } else {
      if (logscale == false) {
        x_bin = static_cast<int>((x_val - bins(0)) / step_size) + 1;
      } else if (logscale == true) {
        x_bin = static_cast<int>(std::log10(x_val / bins(0)) / step_size) + 1;
      }
    }
    // needs to be zero as for the 1D histogram we need 0 as first index of the 2D
    // result array
    int y_bin = 0;
    if (pdf_dimension == 2) {
      auto &y_val = outvars_device(1, m, k, j, i);

      y_bin = -1; // reset to impossible value
      // First handle edge cases explicitly
      if (y_val < bins2(0)) {
        y_bin = 0;
      } else if (y_val >= bins2(nbin2_)) {
        y_bin = nbin2_ + 1;
      } else {
        // for lin and log directly pick index
        if (logscale2 == false) {
          y_bin = static_cast<int>((y_val - bins2(0)) / step_size2) + 1;
        } else if (logscale2 == true) {
          y_bin = static_cast<int>(std::log10(y_val/bins2(0)) / step_size2) + 1;
        }
      }
    }
    auto res = scatter.access();
    Real weight = size.d_view(m).dx1*size.d_view(m).dx2*size.d_view(m).dx3;
    weight *= mass_weighted == false
              ? 1.0
              : u0_(m, IDN, k, j, i);
    res(y_bin, x_bin) += weight;
  });

  // "reduce" results from scatter view to original view.
  // May be a no-op depending on backend.
  Kokkos::Experimental::contribute(result, scatter); //.KokkosView()
  // Kokkos::Experimental::contribute(result.KokkosView(), scatter);
  Kokkos::fence(); // May not be required

  // Now reduce over ranks
#if MPI_PARALLEL_ENABLED
  if (global_variable::my_rank == 0) {
    MPI_Reduce(MPI_IN_PLACE, result.data(), result.size(),
                                   MPI_ATHENA_REAL, MPI_SUM, 0, MPI_COMM_WORLD);
  } else {
    MPI_Reduce(result.data(), result.data(), result.size(),
                                   MPI_ATHENA_REAL, MPI_SUM, 0, MPI_COMM_WORLD);
  }
#endif
}

//----------------------------------------------------------------------------------------
//! \fn void PDFOutput::WriteOutputFile()
//  \brief Cycles through hist_data vector and writes history file for each component

void PDFOutput::WriteOutputFile(Mesh *pm, ParameterInput *pin) {
  // only the master rank writes the file
  if (global_variable::my_rank == 0) {
    // Write header, if it has not been written already
    if (!(pdf_data.bins_written)) {
      // create filename: "pdf_"+"file_id"/file_basename" + ".bins.pdf"
      std::string fname;
      fname.assign("pdf_");
      fname.append(out_params.file_id);
      if (pdf_data.pdf_dimension == 2) {
        fname.append("_");
        fname.append(out_params.variable_2);
      }
      fname.append("/");
      fname.append(out_params.file_basename);
      fname.append(".bins.pdf");

      // open file for output
      FILE *pfile;
      if ((pfile = std::fopen(fname.c_str(),"a")) == nullptr) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
          << std::endl << "Output file '" << fname << "' could not be opened" <<std::endl;
        exit(EXIT_FAILURE);
      }

      std::fprintf(pfile,"# pdf bins \n");
      std::fprintf(pfile,"# [1]= %.20s \n", outvars[0].label.c_str());
      if (pdf_data.pdf_dimension == 2) {
        std::fprintf(pfile,"# [2]= %.20s \n", outvars[1].label.c_str());
      }

      // write bins
      // Create a host mirror of the pdf_data.result_ array
      auto bins_host = Kokkos::create_mirror_view(pdf_data.bins);
      Kokkos::deep_copy(bins_host, pdf_data.bins);
      Kokkos::fence();

      for (int n=0; n<pdf_data.nbin+1; ++n) {
        std::fprintf(pfile, out_params.data_format.c_str(), bins_host[n]);
      }
      std::fprintf(pfile,"\n");                              // terminate line
      if (pdf_data.pdf_dimension == 2) {
        auto bins2_host = Kokkos::create_mirror_view(pdf_data.bins2);
        Kokkos::deep_copy(bins2_host, pdf_data.bins2);
        Kokkos::fence();

        for (int n=0; n<pdf_data.nbin2+1; ++n) {
          std::fprintf(pfile, out_params.data_format.c_str(), bins2_host[n]);
        }
        std::fprintf(pfile,"\n");                            // terminate line
      }
      std::fclose(pfile);
      pdf_data.bins_written = true;
    }

    // create filename: "pdf_"+"file_id"/file_basename" + "." + XXXXX + ".pdf"
    // where XXXXX = 5-digit file_number
    std::string fname;
    char number[6];
    std::snprintf(number, sizeof(number), "%05d", out_params.file_number);
    fname.assign("pdf_");
    fname.append(out_params.file_id);
    if (pdf_data.pdf_dimension == 2) {
      fname.append("_");
      fname.append(out_params.variable_2);
    }
    fname.append("/");
    fname.append(out_params.file_basename);
    fname.append(".");
    fname.append(number);
    fname.append(".pdf");

    // open file for output
    FILE *pfile;
    if ((pfile = std::fopen(fname.c_str(),"a")) == nullptr) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
        << std::endl << "Output file '" << fname << "' could not be opened" <<std::endl;
      exit(EXIT_FAILURE);
    }

    // Create a host mirror of the pdf_data.result_ array
    auto result_host = Kokkos::create_mirror_view(pdf_data.result_);

    // Copy the data from the device to the host
    Kokkos::deep_copy(result_host, pdf_data.result_);

    // write history variables
    std::fprintf(pfile, "# time= ");
    std::fprintf(pfile, out_params.data_format.c_str(), pm->time);
    std::fprintf(pfile, "\n");
    int number_n2_bins = pdf_data.pdf_dimension == 2 ? pdf_data.nbin2+2 : 1;
    for (int n2=0; n2<number_n2_bins; ++n2) {
      for (int n=0; n<pdf_data.nbin+2; ++n) {
        std::fprintf(pfile, out_params.data_format.c_str(), result_host(n2, n));
      }
      std::fprintf(pfile,"\n"); // terminate line
    }
    std::fprintf(pfile,"\n"); // terminate line
    std::fclose(pfile);
  }

  // increment counters
  out_params.file_number++; // By doing this I make a new file for each time.
  // I could alternatively have a single file that is appended to each time.
  if (out_params.last_time < 0.0) {
    out_params.last_time = pm->time;
  } else {
    out_params.last_time += out_params.dt;
  }
  pin->SetInteger(out_params.block_name, "file_number", out_params.file_number);
  pin->SetReal(out_params.block_name, "last_time", out_params.last_time);
  return;
}
