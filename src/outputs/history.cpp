//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file history.cpp
//  \brief writes history output data, volume-averaged quantities that are output
//         frequently in time to trace their evolution.

#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "outputs.hpp"

//----------------------------------------------------------------------------------------
// ctor: also calls OutputType base class constructor

HistoryOutput::HistoryOutput(OutputParameters op, Mesh *pm) : OutputType(op, pm)
{
}

//----------------------------------------------------------------------------------------
//! \struct array_type
// Following code is copied from Kokkos wiki pages on building custom reducers.  It allows
// multiple sum reductions to be computed simultaneously, as required for history outputs

namespace hist_sum {  // namespace helps with name resolution in reduction identity 
  template< class ScalarType, int N >
  struct array_type {
    ScalarType the_array[N];
  
    KOKKOS_INLINE_FUNCTION   // Default constructor - Initialize to 0's
    array_type() { 
      for (int i = 0; i < N; i++ ) { the_array[i] = 0; }
    }
    KOKKOS_INLINE_FUNCTION   // Copy Constructor
    array_type(const array_type & rhs) { 
       for (int i = 0; i < N; i++ ){
          the_array[i] = rhs.the_array[i];
       }
    }
    KOKKOS_INLINE_FUNCTION   // add operator
    array_type& operator += (const array_type& src) {
      for ( int i = 0; i < N; i++ ) {
         the_array[i]+=src.the_array[i];
      }
      return *this;
    } 
    KOKKOS_INLINE_FUNCTION   // volatile add operator 
    void operator += (const volatile array_type& src) volatile {
      for ( int i = 0; i < N; i++ ) {
        the_array[i]+=src.the_array[i];
      }
    }
  };
  typedef array_type<Real,(NHISTORY_VARIABLES)> GlobalSum;  // used to simplify code below
}
namespace Kokkos { //reduction identity must be defined in Kokkos namespace
  template<>
  struct reduction_identity< hist_sum::GlobalSum > {
    KOKKOS_FORCEINLINE_FUNCTION static hist_sum::GlobalSum sum() {
      return hist_sum::GlobalSum();
    }
  };
}

//----------------------------------------------------------------------------------------
//! \fn void HistoryOutput::LoadOutputData()
//  \brief Compute and store history data over all MeshBlocks on this rank
//  Data is stored in a Real array defined in derived class.

void HistoryOutput::LoadOutputData(Mesh *pm)
{ 
  // initialize sums over MeshBlocks on this rank to zero
  for (int n=0; n<NHISTORY_VARIABLES; ++n) {
    history_data[n] = 0.0;
  }

  // loop over all MeshBlocks on this MPI rank
  for (int m=0; m<(pm->nmbthisrank); ++m) {
    MeshBlock *pmb = &(pm->mblocks[m]);
    int is = pmb->mb_cells.is; int nx1 = pmb->mb_cells.nx1;
    int js = pmb->mb_cells.js; int nx2 = pmb->mb_cells.nx2;
    int ks = pmb->mb_cells.ks; int nx3 = pmb->mb_cells.nx3;
    
    auto &u0_ = pmb->phydro->u0;
    const int nkji = nx3*nx2*nx1;
    const int nji  = nx2*nx1;

    hist_sum::GlobalSum sum_this_mb;         
    Kokkos::parallel_reduce("HistSums",Kokkos::RangePolicy<>(pmb->exe_space, 0, nkji),
      KOKKOS_LAMBDA(const int &idx, hist_sum::GlobalSum &mb_sum)
      {
        // compute n,k,j,i indices of thread
        int k = (idx)/nji;
        int j = (idx - k*nji)/nx1;
        int i = (idx - k*nji - j*nx1) + is;
        k += ks;
        j += js;

        // Hydro conserved variables:
        hist_sum::GlobalSum hvars;
        hvars.the_array[0] = u0_(hydro::IDN,k,j,i);
        hvars.the_array[1] = u0_(hydro::IM1,k,j,i);
        hvars.the_array[2] = u0_(hydro::IM2,k,j,i);
        hvars.the_array[3] = u0_(hydro::IM3,k,j,i);
        hvars.the_array[4] = u0_(hydro::IEN,k,j,i);

        // Hydro KE
        hvars.the_array[5] = 0.5*SQR(u0_(hydro::IM1,k,j,i))/u0_(hydro::IDN,k,j,i);
        hvars.the_array[6] = 0.5*SQR(u0_(hydro::IM2,k,j,i))/u0_(hydro::IDN,k,j,i);
        hvars.the_array[7] = 0.5*SQR(u0_(hydro::IM3,k,j,i))/u0_(hydro::IDN,k,j,i);

        // sum into parallel reduce
        mb_sum += hvars;

      }, Kokkos::Sum<hist_sum::GlobalSum>(sum_this_mb)
    );

    // normalize sums by volume of this MeshBlock and sum into output array
    Real volume = ((pmb->mb_size.x1max - pmb->mb_size.x1min)*
      (pmb->mb_size.x2max - pmb->mb_size.x2min)*(pmb->mb_size.x3max - pmb->mb_size.x3min))
      /static_cast<Real>(nx1*nx2*nx3);
    for (int n=0; n<NHISTORY_VARIABLES; ++n) {
      history_data[n] += volume*sum_this_mb.the_array[n];
    }

  }  // end loop over MeshBlocks

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void HistoryOutput::WriteOutputFile()
//  \brief Writes history file

void HistoryOutput::WriteOutputFile(Mesh *pm, ParameterInput *pin)
{
#if MPI_PARALLEL_ENABLED
  // in-place sum over all MPI ranks
  if (global_variable::my_rank == 0) {
    MPI_Reduce(MPI_IN_PLACE, &(history_data[0]), NHISTORY_VARIABLES, MPI_ATHENA_REAL,
       MPI_SUM, 0, MPI_COMM_WORLD);
  } else {
    MPI_Reduce(&(history_data[0]), &(history_data[0]), NHISTORY_VARIABLES,
       MPI_ATHENA_REAL, MPI_SUM, 0, MPI_COMM_WORLD);
  }
#endif

  // only the master rank writes the file
  if (global_variable::my_rank == 0) {

    // create filename: "file_basename" + ".hst".  There is no file number.
    std::string fname;
    fname.assign(out_params.file_basename);
    fname.append(".hst");

    // open file for output
    FILE *pfile;
    if ((pfile = std::fopen(fname.c_str(),"a")) == nullptr) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
        << std::endl << "Output file '" << fname << "' could not be opened" << std::endl;
      exit(EXIT_FAILURE);
    }

    // If this is the first output, write header
    if (out_params.file_number == 0) {
      int iout = 1;
      std::fprintf(pfile,"# Athena++ history data\n");
      std::fprintf(pfile,"# [%d]=time     ", iout++);
      std::fprintf(pfile,"[%d]=dt       ", iout++);
      std::fprintf(pfile,"[%d]=mass     ", iout++);
      std::fprintf(pfile,"[%d]=1-mom    ", iout++);
      std::fprintf(pfile,"[%d]=2-mom    ", iout++);
      std::fprintf(pfile,"[%d]=3-mom    ", iout++);
      if (pm->mblocks.begin()->phydro->peos->eos_data.is_adiabatic) {
        std::fprintf(pfile,"[%d]=tot-E   ", iout++);
      }
      std::fprintf(pfile,"[%d]=1-KE     ", iout++);
      std::fprintf(pfile,"[%d]=2-KE     ", iout++);
      std::fprintf(pfile,"[%d]=3-KE     ", iout++);
      std::fprintf(pfile,"\n");                              // terminate line
      // increment counters so headers are not written again
      out_params.file_number++;
      pin->SetInteger(out_params.block_name, "file_number", out_params.file_number);
    }

    // write history variables
    std::fprintf(pfile, out_params.data_format.c_str(), pm->time);
    std::fprintf(pfile, out_params.data_format.c_str(), pm->dt);
    for (int n=0; n<(NHISTORY_VARIABLES); ++n)
      std::fprintf(pfile, out_params.data_format.c_str(), history_data[n]);
    std::fprintf(pfile,"\n"); // terminate line
    std::fclose(pfile);
  }

  // increment counters, clean up
  if (out_params.last_time < 0.0) {
    out_params.last_time = pm->time;
  } else {
    out_params.last_time += out_params.dt;
  }
  pin->SetReal(out_params.block_name, "last_time", out_params.last_time);
  return;
}
