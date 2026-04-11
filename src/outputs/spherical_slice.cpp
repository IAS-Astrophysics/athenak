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
  Kokkos::realloc(outarray, nout_vars, 1, 1, ntheta, nphi);

  if (out_params.contains_derived) {
    out_params.i_derived = 0;
    ComputeDerivedVariable(out_params.variable, pm);
  }

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
  if (out_params.file_shard_mode == FileShardMode::shared) {
    int count = nout_vars*ntheta*nphi;
    if (global_variable::my_rank == 0) {
      MPI_Reduce(MPI_IN_PLACE, outarray.data(), count, MPI_ATHENA_REAL,
                 MPI_SUM, 0, MPI_COMM_WORLD);
    } else {
      MPI_Reduce(outarray.data(), outarray.data(), count, MPI_ATHENA_REAL,
                 MPI_SUM, 0, MPI_COMM_WORLD);
    }
  } else if (out_params.file_shard_mode == FileShardMode::per_node) {
    int count = nout_vars*ntheta*nphi;
    if (global_variable::rank_in_node == 0) {
      MPI_Reduce(MPI_IN_PLACE, outarray.data(), count, MPI_ATHENA_REAL,
                 MPI_SUM, 0, global_variable::node_comm);
    } else {
      MPI_Reduce(outarray.data(), outarray.data(), count, MPI_ATHENA_REAL,
                 MPI_SUM, 0, global_variable::node_comm);
    }

    std::vector<int> local_mask(psph->nangles, 0);
    for (int32_t angle : psph->owned_angles) {
      local_mask[angle] = 1;
    }
    std::vector<int> node_mask(psph->nangles, 0);
    MPI_Reduce(local_mask.data(), node_mask.data(), psph->nangles, MPI_INT, MPI_MAX, 0,
               global_variable::node_comm);
    if (global_variable::rank_in_node == 0) {
      shard_owned_angles.clear();
      shard_owned_angles.reserve(psph->nangles);
      for (int a = 0; a < psph->nangles; ++a) {
        if (node_mask[a] != 0) {
          shard_owned_angles.push_back(static_cast<int32_t>(a));
        }
      }
    } else {
      shard_owned_angles.clear();
    }
  }
#endif
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

      std::vector<float> data(static_cast<size_t>(npoints)*nout_vars);
      size_t k = 0;
      for (int n = 0; n < nout_vars; ++n) {
        for (int q = 0; q < npoints; ++q) {
          int a = owned[q];
          int it = a/nphi;
          int ip = a%nphi;
          data[k++] = static_cast<float>(outarray(n, 0, 0, it, ip));
        }
      }
      std::fwrite(data.data(), sizeof(float), data.size(), pfile);
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
