//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file spherical_slice.cpp
//! \brief writes a 2D spherical slice (constant radius r, origin-centered) in binary
//! format.  Samples are placed on a uniform (cos(theta), phi) grid -- equal-area cells.
//! All requested output variables are trilinearly interpolated from the local Cartesian
//! grid onto the sphere.  Two output modes:
//!   * default: ranks MPI_Reduce-sum into rank 0 which writes one file with the full
//!     (ntheta, nphi) surface.
//!   * partitioned modes: each rank or node writes only its locally-owned sample points
//!     to bin/rank_XXXXXXXX/... or bin/node_XXXXXXXX/... without a global reduction.

#include <sys/stat.h>  // mkdir

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <queue>
#include <sstream>
#include <string>
#include <vector>

#include "athena.hpp"
#include "file_sharding.hpp"
#include "globals.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "outputs.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

namespace {

bool SphsliceOutputStatsEnabled() {
  const char *env = std::getenv("ATHENAK_OUTPUT_IO_STATS");
  return (env != nullptr && env[0] != '\0' && env[0] != '0');
}

void PrintSphsliceStats(FileShardMode mode, int local_points, int node_points,
                        std::size_t sparse_payload_bytes,
                        std::size_t dense_baseline_bytes) {
  if (!SphsliceOutputStatsEnabled()) {
    return;
  }
  std::cout << "[output-io] type=sphslice"
            << " mode=" << ShardDistributionName(mode)
            << " writer=" << ShardWriterId(mode)
            << " local_points=" << local_points
            << " node_points=" << node_points
            << " sparse_payload_bytes=" << sparse_payload_bytes
            << " dense_baseline_bytes=" << dense_baseline_bytes
            << std::endl;
}

#if MPI_PARALLEL_ENABLED
struct SphsliceMergeCursor {
  int rank = 0;
  int offset = 0;
  int32_t angle = 0;
};

struct SphsliceMergeCursorCompare {
  bool operator()(const SphsliceMergeCursor &lhs,
                  const SphsliceMergeCursor &rhs) const {
    return lhs.angle > rhs.angle;
  }
};
#endif

}  // namespace

//========================================================================================
//! \class SphericalSlice
//! \brief helper that builds (theta,phi) sample points and trilinearly interpolates
//!  values from a Cartesian DvceArray5D onto the sphere.

class SphericalSlice {
 public:
  SphericalSlice(MeshBlockPack *pp, Real r, int ntheta_, int nphi_)
      : pmy_pack(pp), radius(r), ntheta(ntheta_), nphi(nphi_),
        nangles(ntheta_*nphi_),
        interp_indcs("sph_indcs", ntheta_*nphi_, 4),
        interp_wghts("sph_wghts", ntheta_*nphi_, 3),
        interp_vals("sph_vals", ntheta_*nphi_) {
    Rebuild();
  }

  // Recompute owning meshblock + trilinear weights for every sample point.
  // Call this after AMR rebalancing.
  void Rebuild() {
    auto &indcs = pmy_pack->pmesh->mb_indcs;
    auto &size = pmy_pack->pmb->mb_size;
    int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
    int is = indcs.is, js = indcs.js, ks = indcs.ks;
    int nmb = pmy_pack->nmb_thispack;

    auto h_idx = interp_indcs.h_view;
    auto h_wgt = interp_wghts.h_view;

    for (int it = 0; it < ntheta; ++it) {
      // uniform in cos(theta), cell-centered in [-1, 1]
      Real ct = -1.0 + 2.0*(static_cast<Real>(it) + 0.5)/static_cast<Real>(ntheta);
      Real st = std::sqrt(std::max(static_cast<Real>(0.0), 1.0 - ct*ct));
      for (int ip = 0; ip < nphi; ++ip) {
        Real ph = 2.0*M_PI*(static_cast<Real>(ip) + 0.5)/static_cast<Real>(nphi);
        Real x = radius*st*std::cos(ph);
        Real y = radius*st*std::sin(ph);
        Real z = radius*ct;
        int a = it*nphi + ip;

        int mfound = -1;
        for (int m = 0; m < nmb; ++m) {
          auto sz = size.h_view(m);
          if (x >= sz.x1min && x < sz.x1max &&
              y >= sz.x2min && y < sz.x2max &&
              z >= sz.x3min && z < sz.x3max) {
            mfound = m;
            break;
          }
        }
        if (mfound < 0) {
          h_idx(a, 0) = -1;  // sentinel; remaining slots ignored when m<0
          continue;
        }

        auto sz = size.h_view(mfound);
        // fractional cell-center index: i_cc = (x - x1min)/dx - 0.5.  i0 = floor(i_cc)
        // is the left bracketing cell; wx in [0,1) is the trilinear weight to (i0+1).
        // Points at the right edge of an MB (or left of an interior cell) reach into
        // ghost cells, which is fine because boundary exchange runs before output.
        Real fi = (x - sz.x1min)/((sz.x1max - sz.x1min)/static_cast<Real>(nx1)) - 0.5;
        Real fj = (y - sz.x2min)/((sz.x2max - sz.x2min)/static_cast<Real>(nx2)) - 0.5;
        Real fk = (z - sz.x3min)/((sz.x3max - sz.x3min)/static_cast<Real>(nx3)) - 0.5;
        int i0 = static_cast<int>(std::floor(fi));
        int j0 = static_cast<int>(std::floor(fj));
        int k0 = static_cast<int>(std::floor(fk));
        Real wx = fi - static_cast<Real>(i0);
        Real wy = fj - static_cast<Real>(j0);
        Real wz = fk - static_cast<Real>(k0);

        h_idx(a, 0) = mfound;
        h_idx(a, 1) = i0 + is;
        h_idx(a, 2) = j0 + js;
        h_idx(a, 3) = k0 + ks;
        h_wgt(a, 0) = wx;
        h_wgt(a, 1) = wy;
        h_wgt(a, 2) = wz;
      }
    }
    interp_indcs.template modify<HostMemSpace>();
    interp_wghts.template modify<HostMemSpace>();
    interp_indcs.template sync<DevExeSpace>();
    interp_wghts.template sync<DevExeSpace>();

    // Cache list of angles owned by this rank for the per-rank writer.
    owned_angles.clear();
    owned_angles.reserve(nangles);
    for (int a = 0; a < nangles; ++a) {
      if (h_idx(a, 0) >= 0) owned_angles.push_back(static_cast<int32_t>(a));
    }
  }

  // Trilinearly interpolate variable component `var` of `u` onto the sphere.
  // Points whose owning meshblock is not on this rank are set to 0.0, so that an
  // MPI_SUM reduction across ranks gives the correct global result.
  void Interpolate(int var, DvceArray5D<Real> &u) {
    auto d_idx = interp_indcs.d_view;
    auto d_wgt = interp_wghts.d_view;
    auto d_val = interp_vals.d_view;
    int n = nangles;
    par_for("sph_interp", DevExeSpace(), 0, n-1,
    KOKKOS_LAMBDA(int a) {
      int m  = d_idx(a, 0);
      if (m < 0) {
        d_val(a) = 0.0;
        return;
      }
      int i0 = d_idx(a, 1);
      int j0 = d_idx(a, 2);
      int k0 = d_idx(a, 3);
      Real wx = d_wgt(a, 0);
      Real wy = d_wgt(a, 1);
      Real wz = d_wgt(a, 2);
      Real c000 = u(m, var, k0  , j0  , i0  );
      Real c100 = u(m, var, k0  , j0  , i0+1);
      Real c010 = u(m, var, k0  , j0+1, i0  );
      Real c110 = u(m, var, k0  , j0+1, i0+1);
      Real c001 = u(m, var, k0+1, j0  , i0  );
      Real c101 = u(m, var, k0+1, j0  , i0+1);
      Real c011 = u(m, var, k0+1, j0+1, i0  );
      Real c111 = u(m, var, k0+1, j0+1, i0+1);
      Real c00 = c000*(1.0-wx) + c100*wx;
      Real c10 = c010*(1.0-wx) + c110*wx;
      Real c01 = c001*(1.0-wx) + c101*wx;
      Real c11 = c011*(1.0-wx) + c111*wx;
      Real c0  = c00*(1.0-wy) + c10*wy;
      Real c1  = c01*(1.0-wy) + c11*wy;
      d_val(a) = c0*(1.0-wz) + c1*wz;
    });
    interp_vals.template modify<DevExeSpace>();
    interp_vals.template sync<HostMemSpace>();
  }

  MeshBlockPack *pmy_pack;
  Real radius;
  int ntheta, nphi, nangles;
  DualArray2D<int>      interp_indcs;  // (nangles, 4) -> m, i0, j0, k0
  DualArray2D<Real>     interp_wghts;  // (nangles, 3) -> wx, wy, wz
  DualArray1D<Real>     interp_vals;   // (nangles)
  // Angle indices owned by this rank.  int32_t for direct fwrite to wire format.
  std::vector<int32_t>  owned_angles;
};

//========================================================================================
// SphericalSliceOutput constructor

SphericalSliceOutput::SphericalSliceOutput(ParameterInput *pin, Mesh *pm,
                                           OutputParameters op)
    : BaseTypeOutput(pin, pm, op), psph(nullptr) {
  mkdir("bin", 0775);
  if (op.file_shard_mode != FileShardMode::shared) {
    std::string shard_dir = std::string("bin/") + ShardDirectoryName(op.file_shard_mode);
    mkdir(shard_dir.c_str(), 0775);
  }

  Real r = pin->GetReal(op.block_name, "slice_r");
  int nt = pin->GetOrAddInteger(op.block_name, "ntheta", 64);
  int np = pin->GetOrAddInteger(op.block_name, "nphi", 128);

  if (pm->mesh_indcs.nx2 <= 1 || pm->mesh_indcs.nx3 <= 1) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "sphslice output requires 3D mesh" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  Real lx = pm->mesh_size.x1max - pm->mesh_size.x1min;
  Real ly = pm->mesh_size.x2max - pm->mesh_size.x2min;
  Real lz = pm->mesh_size.x3max - pm->mesh_size.x3min;
  Real half = 0.5*std::min({lx, ly, lz});
  if (!(r > 0.0 && r < half)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "sphslice radius=" << r << " in block '"
              << op.block_name << "' must satisfy 0 < r < " << half
              << " (half the smallest box length)" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  if (nt < 2 || np < 2) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "sphslice block '" << op.block_name
              << "' requires ntheta>=2 and nphi>=2" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  psph = new SphericalSlice(pm->pmb_pack, r, nt, np);
}

SphericalSliceOutput::~SphericalSliceOutput() {
  delete psph;
}

//----------------------------------------------------------------------------------------
// SphericalSliceOutput::LoadOutputData()

void SphericalSliceOutput::LoadOutputData(Mesh *pm) {
  if (pm->adaptive) {
    psph->Rebuild();
  }

  int nout_vars = outvars.size();
  int ntheta = psph->ntheta;
  int nphi = psph->nphi;
  int nangles = psph->nangles;
  shard_owned_angles.clear();
  shard_values.clear();

  if (out_params.contains_derived) {
    out_params.i_derived = 0;
    ComputeDerivedVariable(out_params.variable, pm);
  }

  if (out_params.file_shard_mode == FileShardMode::shared) {
    Kokkos::realloc(outarray, nout_vars, 1, 1, ntheta, nphi);
    for (int n = 0; n < nout_vars; ++n) {
      psph->Interpolate(outvars[n].data_index, *(outvars[n].data_ptr));
      auto h_vals = psph->interp_vals.h_view;
      for (int it = 0; it < ntheta; ++it) {
        for (int ip = 0; ip < nphi; ++ip) {
          outarray(n, 0, 0, it, ip) = h_vals(it*nphi + ip);
        }
      }
    }
    shard_owned_angles = psph->owned_angles;

#if MPI_PARALLEL_ENABLED
    int count = nout_vars*ntheta*nphi;
    if (global_variable::my_rank == 0) {
      MPI_Reduce(MPI_IN_PLACE, outarray.data(), count, MPI_ATHENA_REAL,
                 MPI_SUM, 0, MPI_COMM_WORLD);
    } else {
      MPI_Reduce(outarray.data(), outarray.data(), count, MPI_ATHENA_REAL,
                 MPI_SUM, 0, MPI_COMM_WORLD);
    }
#endif
    return;
  }

  shard_owned_angles = psph->owned_angles;
  int local_points = static_cast<int>(shard_owned_angles.size());
  std::vector<float> local_values(static_cast<std::size_t>(nout_vars)*local_points);

  for (int n = 0; n < nout_vars; ++n) {
    psph->Interpolate(outvars[n].data_index, *(outvars[n].data_ptr));
    auto h_vals = psph->interp_vals.h_view;
    for (int q = 0; q < local_points; ++q) {
      local_values[static_cast<std::size_t>(n)*local_points + q] =
          static_cast<float>(h_vals(shard_owned_angles[q]));
    }
  }

#if MPI_PARALLEL_ENABLED
  if (out_params.file_shard_mode == FileShardMode::per_node) {
    int nranks_in_node = global_variable::ranks_per_node;
    std::vector<int> point_counts;
    if (global_variable::rank_in_node == 0) {
      point_counts.resize(nranks_in_node, 0);
    }
    MPI_Gather(&local_points, 1, MPI_INT,
               point_counts.empty() ? nullptr : point_counts.data(), 1, MPI_INT, 0,
               global_variable::node_comm);

    std::vector<int> point_displs;
    std::vector<int> value_counts;
    std::vector<int> value_displs;
    std::vector<int32_t> gathered_angles;
    std::vector<float> gathered_values;
    int node_points = local_points;
    if (global_variable::rank_in_node == 0) {
      point_displs.resize(nranks_in_node, 0);
      value_counts.resize(nranks_in_node, 0);
      value_displs.resize(nranks_in_node, 0);
      node_points = 0;
      int running_points = 0;
      int running_values = 0;
      for (int r = 0; r < nranks_in_node; ++r) {
        point_displs[r] = running_points;
        value_displs[r] = running_values;
        running_points += point_counts[r];
        value_counts[r] = nout_vars*point_counts[r];
        running_values += value_counts[r];
      }
      node_points = running_points;
      gathered_angles.resize(node_points);
      gathered_values.resize(static_cast<std::size_t>(nout_vars)*node_points);
    }

    MPI_Gatherv(shard_owned_angles.data(), local_points, MPI_INT32_T,
                gathered_angles.data(),
                point_counts.empty() ? nullptr : point_counts.data(),
                point_displs.empty() ? nullptr : point_displs.data(),
                MPI_INT32_T, 0, global_variable::node_comm);
    MPI_Gatherv(local_values.data(), nout_vars*local_points, MPI_FLOAT,
                gathered_values.data(),
                value_counts.empty() ? nullptr : value_counts.data(),
                value_displs.empty() ? nullptr : value_displs.data(),
                MPI_FLOAT, 0, global_variable::node_comm);

    if (global_variable::rank_in_node == 0) {
      shard_owned_angles.assign(node_points, 0);
      shard_values.assign(static_cast<std::size_t>(nout_vars)*node_points, 0.0f);

      std::priority_queue<SphsliceMergeCursor, std::vector<SphsliceMergeCursor>,
                          SphsliceMergeCursorCompare> merge_heap;
      for (int r = 0; r < nranks_in_node; ++r) {
        if (point_counts[r] > 0) {
          merge_heap.push({r, 0, gathered_angles[point_displs[r]]});
        }
      }

      int merged_points = 0;
      int32_t prev_angle = -1;
      while (!merge_heap.empty()) {
        SphsliceMergeCursor cursor = merge_heap.top();
        merge_heap.pop();

        int angle_offset = point_displs[cursor.rank] + cursor.offset;
        int32_t angle = gathered_angles[angle_offset];
        if (angle < 0 || angle >= nangles) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                    << std::endl << "sphslice node shard contains out-of-range angle "
                    << angle << std::endl;
          std::exit(EXIT_FAILURE);
        }
        if (merged_points > 0 && angle == prev_angle) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                    << std::endl << "sphslice node shard has duplicate angle index "
                    << angle << std::endl;
          std::exit(EXIT_FAILURE);
        }

        shard_owned_angles[merged_points] = angle;
        int rank_points = point_counts[cursor.rank];
        int value_base = value_displs[cursor.rank];
        for (int n = 0; n < nout_vars; ++n) {
          shard_values[static_cast<std::size_t>(n)*node_points + merged_points] =
              gathered_values[value_base + n*rank_points + cursor.offset];
        }
        prev_angle = angle;
        merged_points++;

        cursor.offset++;
        if (cursor.offset < rank_points) {
          cursor.angle = gathered_angles[point_displs[cursor.rank] + cursor.offset];
          merge_heap.push(cursor);
        }
      }

      if (merged_points != node_points) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "sphslice node merge lost points: merged="
                  << merged_points << " expected=" << node_points << std::endl;
        std::exit(EXIT_FAILURE);
      }

      std::size_t sparse_payload_bytes =
          static_cast<std::size_t>(node_points) *
          (sizeof(int32_t) + static_cast<std::size_t>(nout_vars)*sizeof(float));
      std::size_t dense_baseline_bytes =
          static_cast<std::size_t>(nout_vars)*nangles*sizeof(Real) +
          static_cast<std::size_t>(nangles)*sizeof(int);
      PrintSphsliceStats(out_params.file_shard_mode, local_points, node_points,
                         sparse_payload_bytes, dense_baseline_bytes);
    } else {
      shard_owned_angles.clear();
      shard_values.clear();
    }
    return;
  }
#endif

  shard_values.swap(local_values);
  std::size_t sparse_payload_bytes =
      static_cast<std::size_t>(local_points) *
      (sizeof(int32_t) + static_cast<std::size_t>(nout_vars)*sizeof(float));
  std::size_t dense_baseline_bytes =
      static_cast<std::size_t>(nout_vars)*nangles*sizeof(Real);
  PrintSphsliceStats(out_params.file_shard_mode, local_points, local_points,
                     sparse_payload_bytes, dense_baseline_bytes);
}

//----------------------------------------------------------------------------------------
// SphericalSliceOutput::WriteOutputFile()
//
// Binary layout (ASCII preheader, newline-terminated, then input-file dump, then payload):
//   "Athena spherical slice version=1.0"
//   "  single_file_per_rank=<0|1>"
//   "  rank=<R>"                       (rank that wrote this file; 0 for shared file)
//   "  time=<t>"
//   "  cycle=<n>"
//   "  radius=<r>"
//   "  ntheta=<ntheta>"
//   "  nphi=<nphi>"
//   "  size of variable=4"
//   "  number of variables=<nv>"
//   "  npoints=<P>"                    (P = ntheta*nphi shared, or owned count per-rank)
//   "  variables: v1 v2 ..."
//   "  header offset=<bytes of input dump>"
//   <input parameter dump bytes>
//   shared mode:    <nv * ntheta * nphi  float32>     ordering: (var, itheta, iphi)
//   per-rank mode:  <P int32 angle indices a> <nv * P float32 values>  (var-major)
//                   recover (it, ip) via it = a/nphi, ip = a%nphi

void SphericalSliceOutput::WriteOutputFile(Mesh *pm, ParameterInput *pin) {
  int ntheta = psph->ntheta;
  int nphi = psph->nphi;
  Real radius = psph->radius;
  FileShardMode shard_mode = out_params.file_shard_mode;
  bool partitioned = (shard_mode != FileShardMode::shared);
  int nout_vars = outvars.size();

  bool i_write = IsShardWriter(shard_mode);

  if (i_write) {
    char number[7];
    std::snprintf(number, sizeof(number), ".%05d", out_params.file_number);
    char rstr[32];
    std::snprintf(rstr, sizeof(rstr), "r=%g", static_cast<double>(radius));

    std::string dir = "bin/";
    if (partitioned) {
      dir += ShardDirectoryName(shard_mode);
    }
    std::string fname = dir + out_params.file_basename
                      + "." + out_params.file_id + "." + rstr + number + ".sph.bin";

    std::FILE *pfile = std::fopen(fname.c_str(), "wb");
    if (pfile == nullptr) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "could not open '" << fname << "' for writing"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }

    int npoints = partitioned ? static_cast<int>(shard_owned_angles.size())
                              : ntheta*nphi;

    std::stringstream hdr;
    hdr << "Athena spherical slice version=1.0" << std::endl
        << "  distribution=" << ShardDistributionName(shard_mode) << std::endl
        << "  single_file_per_rank=" << (shard_mode == FileShardMode::per_rank ? 1 : 0)
        << std::endl
        << "  rank=" << global_variable::my_rank << std::endl
        << "  time=" << pm->time << std::endl
        << "  cycle=" << pm->ncycle << std::endl
        << "  radius=" << radius << std::endl
        << "  ntheta=" << ntheta << std::endl
        << "  nphi=" << nphi << std::endl
        << "  size of variable=" << sizeof(float) << std::endl
        << "  number of variables=" << nout_vars << std::endl
        << "  npoints=" << npoints << std::endl
        << "  variables: ";
    for (int n = 0; n < nout_vars; ++n) hdr << outvars[n].label << " ";
    hdr << std::endl;

    std::stringstream ost;
    pin->ParameterDump(ost);
    std::string sbuf = ost.str();
    hdr << "  header offset=" << sbuf.size() << std::endl;
    std::string h = hdr.str();
    std::fwrite(h.c_str(), 1, h.size(), pfile);
    std::fwrite(sbuf.c_str(), 1, sbuf.size(), pfile);

    if (partitioned) {
      const auto &owned = shard_owned_angles;
      std::fwrite(owned.data(), sizeof(int32_t), owned.size(), pfile);
      if (shard_values.size() != static_cast<size_t>(npoints)*nout_vars) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "sphslice sparse payload has incorrect size"
                  << std::endl;
        std::exit(EXIT_FAILURE);
      }
      std::fwrite(shard_values.data(), sizeof(float), shard_values.size(), pfile);
    } else {
      std::vector<float> data(static_cast<size_t>(nout_vars)*ntheta*nphi);
      size_t k = 0;
      for (int n = 0; n < nout_vars; ++n) {
        for (int it = 0; it < ntheta; ++it) {
          for (int ip = 0; ip < nphi; ++ip) {
            data[k++] = static_cast<float>(outarray(n, 0, 0, it, ip));
          }
        }
      }
      std::fwrite(data.data(), sizeof(float), data.size(), pfile);
    }

    std::fclose(pfile);
  }

  out_params.file_number++;
  if (out_params.last_time < 0.0) {
    out_params.last_time = pm->time;
  } else {
    out_params.last_time += out_params.dt;
  }
  pin->SetInteger(out_params.block_name, "file_number", out_params.file_number);
  pin->SetReal(out_params.block_name, "last_time", out_params.last_time);
}
