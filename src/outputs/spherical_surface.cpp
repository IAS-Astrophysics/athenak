//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file spherical_surface.cpp
//! \brief writes data on a SphericalSurface sub-grid in binary VTK format

#include "utils/spherical_surface.hpp"

#include <sys/stat.h>  // mkdir

#include <cstdio>  // snprintf
#include <fstream>
#include <sstream>
#include <string>

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "outputs.hpp"


SphericalSurfaceOutput::SphericalSurfaceOutput(ParameterInput *pin, Mesh *pm,
                                               OutputParameters op)
    : BaseTypeOutput(pin, pm, op) {
  mkdir("sph", 0755);

  Real rad = pin->GetReal(op.block_name, "radius");
  int ntheta = pin->GetOrAddInteger(op.block_name, "ntheta", 32);
  Real xc = pin->GetOrAddReal(op.block_name, "xc", 0.0);
  Real yc = pin->GetOrAddReal(op.block_name, "yc", 0.0);
  Real zc = pin->GetOrAddReal(op.block_name, "zc", 0.0);
  psurf = new SphericalSurface(pm->pmb_pack, ntheta, rad, xc, yc, zc);
}

SphericalSurfaceOutput::~SphericalSurfaceOutput() { delete psurf; }

void SphericalSurfaceOutput::LoadOutputData(Mesh *pm) {
  // If AMR is enabled we need to reset the CartesianGrid
  if (pm->adaptive) {
    psurf->SetInterpolationIndices();
    psurf->SetInterpolationWeights();
  }

  int nout_vars = outvars.size();
  Kokkos::realloc(outarray, nout_vars, 1, 1, 1, psurf->nangles);

  // Calculate derived variables, if required
  if (out_params.contains_derived) {
    ComputeDerivedVariable(out_params.variable, pm);
  }

  for (int n = 0; n < nout_vars; ++n) {
    psurf->InterpolateToSphere(outvars[n].data_index, *(outvars[n].data_ptr));
    auto v_slice = Kokkos::subview(outarray, n, 0, 0, 0, Kokkos::ALL);
    Kokkos::deep_copy(v_slice, psurf->interp_vals.h_view);
  }
#if MPI_PARALLEL_ENABLED
  // Note that InterpolateToSphere will set zero values for points not owned by
  // current rank
  int count = nout_vars * psurf->nangles;
  if (0 == global_variable::my_rank) {
    MPI_Reduce(MPI_IN_PLACE, outarray.data(), count, MPI_ATHENA_REAL, MPI_SUM,
               0, MPI_COMM_WORLD);
  } else {
    MPI_Reduce(outarray.data(), outarray.data(), count, MPI_ATHENA_REAL,
               MPI_SUM, 0, MPI_COMM_WORLD);
  }
#endif
}

void SphericalSurfaceOutput::WriteOutputFile(Mesh *pm, ParameterInput *pin) {
  bool big_end = IsBigEndian();

#if MPI_PARALLEL_ENABLED
  if (0 == global_variable::my_rank) {
#endif
    // Assemble filename
    char fname[BUFSIZ];
    std::snprintf(fname, BUFSIZ, "sph/%s.r=%.2f.%s.%05d.vtk",
                  out_params.file_basename.c_str(), psurf->radius,
                  out_params.file_id.c_str(), out_params.file_number);

    // Open file
    std::ofstream ofile(fname, std::ios::binary);

    ofile << "# vtk DataFile Version 3.0" << std::endl;
    ofile << "# AthenaK data at time=" << pm->time
          << " cycle=" << pm->ncycle << " rad=" << psurf->radius
          << " xc=" << psurf->xc << " yc=" << psurf->yc << " zc=" << psurf->zc
          << std::endl;
    ofile << "BINARY" << std::endl;
    ofile << "DATASET STRUCTURED_GRID" << std::endl;
    ofile << "DIMENSIONS 1 " << psurf->ntheta << " " << 2 * psurf->ntheta
          << std::endl;
    ofile << "POINTS " << psurf->nangles << " float\n";

    for (int i = 0; i < psurf->nangles; ++i) {
      float dline[3] = {static_cast<float>(psurf->radius),
                        static_cast<float>(psurf->polar_pos.h_view(i, 0)),
                        static_cast<float>(psurf->polar_pos.h_view(i, 1))};
      if (!big_end) {
        Swap4Bytes(&dline[0]);
        Swap4Bytes(&dline[1]);
        Swap4Bytes(&dline[2]);
      }

      ofile.write(reinterpret_cast<char *>(&dline[0]), 3 * sizeof(float));
    }

    float t = static_cast<float>(pm->time);
    if (!big_end) {
      Swap4Bytes(&t);
    }
    ofile << "\nFIELD FieldData 2\n";
    ofile << "TIME 1 1 float\n";
    ofile.write(reinterpret_cast<char *>(&t), sizeof(float));

    ofile << "\nCYCLE 1 1 int\n";
    int cycle = pm->ncycle;
    if (!big_end) {
      Swap4Bytes(&cycle);
    }
    ofile.write(reinterpret_cast<char *>(&cycle), sizeof(int));

    ofile << "\nPOINT_DATA " << psurf->nangles << std::endl;
    ofile << "SCALARS weights float 1" << std::endl;
    ofile << "LOOKUP_TABLE default" << std::endl;
    for (int i = 0; i < psurf->nangles; ++i) {
      float d = psurf->radius * psurf->radius * psurf->int_weights.h_view(i);
      if (!big_end) {
        Swap4Bytes(&d);
      }
      ofile.write(reinterpret_cast<char *>(&d), sizeof(float));
    }

    int nout_vars = outvars.size();
    for (int n = 0; n < nout_vars; ++n) {
      ofile << "\nSCALARS " << outvars[n].label << " float 1" << std::endl;
      ofile << "LOOKUP_TABLE default" << std::endl;
      for (int i = 0; i < psurf->nangles; ++i) {
        float d = outarray(n, 0, 0, 0, i);
        if (!big_end) {
          Swap4Bytes(&d);
        }
        ofile.write(reinterpret_cast<char *>(&d), sizeof(float));
      }
    }

#if MPI_PARALLEL_ENABLED
  }
#endif

  // increment counters
  out_params.file_number++;
  if (out_params.last_time < 0.0) {
    out_params.last_time = pm->time;
  } else {
    out_params.last_time += out_params.dt;
  }
  pin->SetInteger(out_params.block_name, "file_number", out_params.file_number);
  pin->SetReal(out_params.block_name, "last_time", out_params.last_time);
}
