//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file geodesic_surface.cpp
//! \brief writes data interpolated onto a geodesic grid at a specified radius
//!        as a VTK POLYDATA point cloud with neighbour connectivity metadata

#include <sys/stat.h>  // mkdir

#include <cstdio>  // snprintf
#include <fstream>
#include <string>

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "outputs.hpp"
#include "geodesic-grid/spherical_grid.hpp"

GeodesicSurfaceOutput::GeodesicSurfaceOutput(ParameterInput *pin, Mesh *pm,
                                             OutputParameters op)
    : BaseTypeOutput(pin, pm, op) {
  mkdir("geosph", 0755);

  Real rad  = pin->GetReal(op.block_name, "radius");
  int  nlev = pin->GetOrAddInteger(op.block_name, "nlev", 4);

  // Read ng_interp: controls Lagrange stencil half-width per axis.
  //   ng_interp < 0 : default — full mesh stencil (original behaviour)
  //   ng_interp = 0 : nearest-cell (fastest, strictly monotone — no Runge overshoot)
  //   ng_interp > 0 : Lagrange stencil half-width (2×ng_interp points per axis)
  int ng_mesh = pm->pmb_pack->pmesh->mb_indcs.ng;
  ng_interp_  = pin->GetOrAddInteger(op.block_name, "ng_interp", -1);
  if (ng_interp_ < 0)       ng_interp_ = ng_mesh;
  if (ng_interp_ > ng_mesh) ng_interp_ = ng_mesh;

  pgrid = new SphericalGrid(pm->pmb_pack, nlev, rad, ng_interp_);
}

GeodesicSurfaceOutput::~GeodesicSurfaceOutput() { delete pgrid; }

void GeodesicSurfaceOutput::LoadOutputData(Mesh *pm) {
  if (pm->adaptive) {
    pgrid->SetInterpolationIndices();
    pgrid->SetInterpolationWeights();
  }

  int nout_vars = outvars.size();
  Kokkos::realloc(outarray, nout_vars, 1, 1, 1, pgrid->nangles);

  if (out_params.contains_derived) {
    ComputeDerivedVariable(out_params.variable, pm);
  }

  for (int n = 0; n < nout_vars; ++n) {
    pgrid->InterpolateToSphere(outvars[n].data_index, outvars[n].data_index,
                               *(outvars[n].data_ptr));
    for (int i = 0; i < pgrid->nangles; ++i) {
      outarray(n, 0, 0, 0, i) = pgrid->interp_vals.h_view(i, 0);
    }
  }

#if MPI_PARALLEL_ENABLED
  int count = nout_vars * pgrid->nangles;
  if (0 == global_variable::my_rank) {
    MPI_Reduce(MPI_IN_PLACE, outarray.data(), count, MPI_ATHENA_REAL, MPI_SUM,
               0, MPI_COMM_WORLD);
  } else {
    MPI_Reduce(outarray.data(), outarray.data(), count, MPI_ATHENA_REAL,
               MPI_SUM, 0, MPI_COMM_WORLD);
  }
#endif
}

void GeodesicSurfaceOutput::WriteOutputFile(Mesh *pm, ParameterInput *pin) {
  bool big_end = IsBigEndian();

#if MPI_PARALLEL_ENABLED
  if (0 == global_variable::my_rank) {
#endif
    char fname[BUFSIZ];
    std::snprintf(fname, BUFSIZ, "geosph/%s.r=%.2f.%s.%05d.vtk",
                  out_params.file_basename.c_str(), pgrid->radius,
                  out_params.file_id.c_str(), out_params.file_number);

    std::ofstream ofile(fname, std::ios::binary);

    int nang = pgrid->nangles;

    // VTK header
    ofile << "# vtk DataFile Version 3.0" << std::endl;
    ofile << "# AthenaK geodesic surface at time=" << pm->time
          << " cycle=" << pm->ncycle << " radius=" << pgrid->radius
          << std::endl;
    ofile << "BINARY" << std::endl;
    ofile << "DATASET POLYDATA" << std::endl;

    // POINTS: Cartesian positions on sphere
    ofile << "POINTS " << nang << " float\n";
    for (int i = 0; i < nang; ++i) {
      float pt[3] = {
        static_cast<float>(pgrid->radius * pgrid->cart_pos.h_view(i, 0)),
        static_cast<float>(pgrid->radius * pgrid->cart_pos.h_view(i, 1)),
        static_cast<float>(pgrid->radius * pgrid->cart_pos.h_view(i, 2))
      };
      if (!big_end) {
        Swap4Bytes(&pt[0]);
        Swap4Bytes(&pt[1]);
        Swap4Bytes(&pt[2]);
      }
      ofile.write(reinterpret_cast<char *>(pt), 3 * sizeof(float));
    }

    // VERTICES: one vertex cell per point (required for ParaView to show POINT_DATA)
    ofile << "\nVERTICES " << nang << " " << 2 * nang << "\n";
    for (int i = 0; i < nang; ++i) {
      int vdata[2] = {1, i};
      if (!big_end) {
        Swap4Bytes(&vdata[0]);
        Swap4Bytes(&vdata[1]);
      }
      ofile.write(reinterpret_cast<char *>(vdata), 2 * sizeof(int));
    }

    // FIELD data: time and cycle metadata
    float t = static_cast<float>(pm->time);
    if (!big_end) { Swap4Bytes(&t); }
    ofile << "\nFIELD FieldData 2\n";
    ofile << "TIME 1 1 float\n";
    ofile.write(reinterpret_cast<char *>(&t), sizeof(float));

    int cycle = pm->ncycle;
    if (!big_end) { Swap4Bytes(&cycle); }
    ofile << "\nCYCLE 1 1 int\n";
    ofile.write(reinterpret_cast<char *>(&cycle), sizeof(int));

    // POINT_DATA
    ofile << "\nPOINT_DATA " << nang << std::endl;

    // theta (polar angle)
    ofile << "SCALARS theta float 1" << std::endl;
    ofile << "LOOKUP_TABLE default" << std::endl;
    for (int i = 0; i < nang; ++i) {
      float d = static_cast<float>(pgrid->polar_pos.h_view(i, 0));
      if (!big_end) { Swap4Bytes(&d); }
      ofile.write(reinterpret_cast<char *>(&d), sizeof(float));
    }

    // phi (azimuthal angle)
    ofile << "\nSCALARS phi float 1" << std::endl;
    ofile << "LOOKUP_TABLE default" << std::endl;
    for (int i = 0; i < nang; ++i) {
      float d = static_cast<float>(pgrid->polar_pos.h_view(i, 1));
      if (!big_end) { Swap4Bytes(&d); }
      ofile.write(reinterpret_cast<char *>(&d), sizeof(float));
    }

    // solid_angle (steradians, for surface integration)
    ofile << "\nSCALARS solid_angle float 1" << std::endl;
    ofile << "LOOKUP_TABLE default" << std::endl;
    for (int i = 0; i < nang; ++i) {
      float d = static_cast<float>(pgrid->solid_angles.h_view(i));
      if (!big_end) { Swap4Bytes(&d); }
      ofile.write(reinterpret_cast<char *>(&d), sizeof(float));
    }

    // num_neighbors (5 or 6)
    ofile << "\nSCALARS num_neighbors int 1" << std::endl;
    ofile << "LOOKUP_TABLE default" << std::endl;
    for (int i = 0; i < nang; ++i) {
      int d = pgrid->num_neighbors.h_view(i);
      if (!big_end) { Swap4Bytes(&d); }
      ofile.write(reinterpret_cast<char *>(&d), sizeof(int));
    }

    // neighbour indices: 6 separate integer scalar fields
    for (int nb = 0; nb < 6; ++nb) {
      ofile << "\nSCALARS neighbor_" << nb << " int 1" << std::endl;
      ofile << "LOOKUP_TABLE default" << std::endl;
      for (int i = 0; i < nang; ++i) {
        int d;
        if (nb < pgrid->num_neighbors.h_view(i)) {
          d = pgrid->ind_neighbors.h_view(i, nb);
        } else {
          d = -1;
        }
        if (!big_end) { Swap4Bytes(&d); }
        ofile.write(reinterpret_cast<char *>(&d), sizeof(int));
      }
    }

    // output variables
    int nout_vars = outvars.size();
    for (int n = 0; n < nout_vars; ++n) {
      ofile << "\nSCALARS " << outvars[n].label << " float 1" << std::endl;
      ofile << "LOOKUP_TABLE default" << std::endl;
      for (int i = 0; i < nang; ++i) {
        float d = static_cast<float>(outarray(n, 0, 0, 0, i));
        if (!big_end) { Swap4Bytes(&d); }
        ofile.write(reinterpret_cast<char *>(&d), sizeof(float));
      }
    }

    ofile.close();

#if MPI_PARALLEL_ENABLED
  }
#endif

  out_params.file_number++;
  if (out_params.last_time < 0.0) {
    out_params.last_time = pm->time;
  } else {
    out_params.last_time += out_params.dt;
  }
  pin->SetInteger(out_params.block_name, "file_number", out_params.file_number);
  pin->SetReal(out_params.block_name, "last_time", out_params.last_time);
}
