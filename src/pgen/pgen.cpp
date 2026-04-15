//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file pgen.cpp
//! \brief Implementation of constructors and functions in class ProblemGenerator.
//! Default constructor calls problem generator function, while  constructor for restarts
//! reads data from restart file, as well as re-initializing problem-specific data.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <utility>
#include <algorithm>
#include <vector>

#include "athena.hpp"
#include "geodesic-grid/geodesic_grid.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "coordinates/adm.hpp"
#include "z4c/compact_object_tracker.hpp"
#include "z4c/z4c.hpp"
#include "radiation/radiation.hpp"
#include "srcterms/turb_driver.hpp"
#include "pgen.hpp"

namespace {

struct RestartBlockRequest {
  int local_index;
  int global_id;
  int src_rank;
};

struct PerNodeRestartRequest {
  int local_index;
  IOWrapperSizeT file_offset;
};

struct PerNodeRestartSpan {
  IOWrapperSizeT file_offset;
  int first_request;
  int request_count;
  IOWrapperSizeT byte_count;
};

struct RestartChunkLayout {
  IOWrapperSizeT chunk_stride = 0;
  IOWrapperSizeT hydro_offset = 0;
  IOWrapperSizeT hydro_bytes = 0;
  IOWrapperSizeT mhd_cc_offset = 0;
  IOWrapperSizeT mhd_cc_bytes = 0;
  IOWrapperSizeT mhd_x1f_offset = 0;
  IOWrapperSizeT mhd_x1f_bytes = 0;
  IOWrapperSizeT mhd_x2f_offset = 0;
  IOWrapperSizeT mhd_x2f_bytes = 0;
  IOWrapperSizeT mhd_x3f_offset = 0;
  IOWrapperSizeT mhd_x3f_bytes = 0;
  IOWrapperSizeT rad_offset = 0;
  IOWrapperSizeT rad_bytes = 0;
  IOWrapperSizeT turb_offset = 0;
  IOWrapperSizeT turb_bytes = 0;
  IOWrapperSizeT grav_offset = 0;
  IOWrapperSizeT grav_bytes = 0;
};

bool RestartIoStatsEnabled() {
  static const bool enabled = []() {
    const char *env = std::getenv("ATHENAK_RESTART_IO_STATS");
    return (env != nullptr && env[0] != '\0' && env[0] != '0');
  }();
  return enabled;
}

RestartChunkLayout BuildRestartChunkLayout(const RestartMetaData &meta,
                                           IOWrapperSizeT data_stride,
                                           int nout1, int nout2, int nout3,
                                           int nhydro, int nmhd, int nrad,
                                           int nforce, int nz4c, int nadm,
                                           const char *context) {
  RestartChunkLayout layout;
  layout.chunk_stride = (meta.payload_stride != 0)
      ? static_cast<IOWrapperSizeT>(meta.payload_stride)
      : data_stride;
  if (meta.file_shard_mode == FileShardMode::per_node && layout.chunk_stride != data_stride) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Per-node restart payload stride does not match local "
              << "physics payload size." << std::endl;
    std::exit(EXIT_FAILURE);
  }

  IOWrapperSizeT offset = 0;
  layout.hydro_offset = offset;
  layout.hydro_bytes = static_cast<IOWrapperSizeT>(nout1)*nout2*nout3*nhydro*sizeof(Real);
  offset += layout.hydro_bytes;
  layout.mhd_cc_offset = offset;
  layout.mhd_cc_bytes = static_cast<IOWrapperSizeT>(nout1)*nout2*nout3*nmhd*sizeof(Real);
  offset += layout.mhd_cc_bytes;
  layout.mhd_x1f_offset = offset;
  layout.mhd_x1f_bytes = static_cast<IOWrapperSizeT>(nout1 + 1)*nout2*nout3*sizeof(Real);
  if (nmhd > 0) offset += layout.mhd_x1f_bytes;
  layout.mhd_x2f_offset = offset;
  layout.mhd_x2f_bytes = static_cast<IOWrapperSizeT>(nout1)*(nout2 + 1)*nout3*sizeof(Real);
  if (nmhd > 0) offset += layout.mhd_x2f_bytes;
  layout.mhd_x3f_offset = offset;
  layout.mhd_x3f_bytes = static_cast<IOWrapperSizeT>(nout1)*nout2*(nout3 + 1)*sizeof(Real);
  if (nmhd > 0) offset += layout.mhd_x3f_bytes;
  layout.rad_offset = offset;
  layout.rad_bytes = static_cast<IOWrapperSizeT>(nout1)*nout2*nout3*nrad*sizeof(Real);
  offset += layout.rad_bytes;
  layout.turb_offset = offset;
  layout.turb_bytes = static_cast<IOWrapperSizeT>(nout1)*nout2*nout3*nforce*sizeof(Real);
  if (nforce > 0) offset += layout.turb_bytes;
  layout.grav_offset = offset;
  layout.grav_bytes = static_cast<IOWrapperSizeT>(nout1)*nout2*nout3*
                      ((nz4c > 0) ? nz4c : nadm)*sizeof(Real);
  if ((nz4c > 0) || (nadm > 0)) offset += layout.grav_bytes;

  if (offset != layout.chunk_stride) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << context << " chunk size mismatch, restart file is broken."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  return layout;
}

std::vector<int> BuildPerNodeRankChunkPrefix(const RestartMetaData &meta) {
  std::vector<int> prefix(meta.original_nranks, 0);
  std::vector<int> node_counts(meta.original_nnodes, 0);
  for (int r=0; r<meta.original_nranks; ++r) {
    int node = meta.rank_to_node[r];
    if (node < 0 || node >= meta.original_nnodes) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Restart metadata contains invalid node assignments."
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
    prefix[r] = node_counts[node];
    node_counts[node] += meta.nmb_eachrank[r];
  }
  return prefix;
}

std::vector<std::vector<PerNodeRestartRequest>> BuildPerNodeRestartRequests(
    Mesh *pm, const RestartMetaData &meta, const std::vector<int> &rank_chunk_prefix,
    IOWrapperSizeT chunk_stride) {
  auto *pack = pm->pmb_pack;
  int nmb = pack->nmb_thispack;
  std::vector<std::vector<PerNodeRestartRequest>> requests(meta.original_nnodes);
  for (int m=0; m<nmb; ++m) {
    int gid = pack->pmb->mb_gid.h_view(m);
    if (gid < 0 || gid >= pm->nmb_total) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Invalid MeshBlock gid encountered during restart."
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
    int src_rank = meta.rank_eachmb[gid];
    if (src_rank < 0 || src_rank >= meta.original_nranks) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Restart metadata contains invalid rank assignments."
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
    int local_chunk_index = gid - meta.gids_eachrank[src_rank];
    if (local_chunk_index < 0 || local_chunk_index >= meta.nmb_eachrank[src_rank]) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Restart metadata inconsistent with MeshBlock ids."
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
    int src_node = meta.rank_to_node[src_rank];
    IOWrapperSizeT chunk_index = static_cast<IOWrapperSizeT>(rank_chunk_prefix[src_rank])
                               + static_cast<IOWrapperSizeT>(local_chunk_index);
    requests[src_node].push_back({m, chunk_stride*chunk_index});
  }
  return requests;
}

std::vector<PerNodeRestartSpan> BuildPerNodeRestartSpans(
    std::vector<PerNodeRestartRequest> &requests, IOWrapperSizeT chunk_stride) {
  std::sort(requests.begin(), requests.end(),
            [](const PerNodeRestartRequest &lhs, const PerNodeRestartRequest &rhs) {
              return lhs.file_offset < rhs.file_offset;
            });

  std::vector<PerNodeRestartSpan> spans;
  if (requests.empty()) {
    return spans;
  }

  int first_request = 0;
  IOWrapperSizeT span_offset = requests[0].file_offset;
  IOWrapperSizeT span_bytes = chunk_stride;
  for (std::size_t i = 1; i < requests.size(); ++i) {
    IOWrapperSizeT expected_offset = requests[i-1].file_offset + chunk_stride;
    if (requests[i].file_offset == expected_offset) {
      span_bytes += chunk_stride;
      continue;
    }
    spans.push_back({span_offset, first_request, static_cast<int>(i) - first_request,
                     span_bytes});
    first_request = static_cast<int>(i);
    span_offset = requests[i].file_offset;
    span_bytes = chunk_stride;
  }
  spans.push_back({span_offset, first_request,
                   static_cast<int>(requests.size()) - first_request, span_bytes});
  return spans;
}

void PrintPerNodeRestartStats(int shard_id, int raw_requests, int merged_spans,
                              int collective_reads) {
  if (!RestartIoStatsEnabled()) {
    return;
  }
#if MPI_PARALLEL_ENABLED
  int node_raw_requests = 0;
  int node_merged_spans = 0;
  MPI_Reduce(&raw_requests, &node_raw_requests, 1, MPI_INT, MPI_SUM, 0,
             global_variable::node_comm);
  MPI_Reduce(&merged_spans, &node_merged_spans, 1, MPI_INT, MPI_SUM, 0,
             global_variable::node_comm);
  if (global_variable::rank_in_node == 0) {
    std::cout << "[restart-io] node=" << global_variable::node_id
              << " shard=" << shard_id
              << " shard_opens=1 shard_closes=1"
              << " raw_requests=" << node_raw_requests
              << " merged_spans=" << node_merged_spans
              << " collective_reads=" << collective_reads
              << std::endl;
  }
#else
  std::cout << "[restart-io] node=0"
            << " shard=" << shard_id
            << " shard_opens=1 shard_closes=1"
            << " raw_requests=" << raw_requests
            << " merged_spans=" << merged_spans
            << " collective_reads=" << collective_reads
            << std::endl;
#endif
}

void UnpackPerNodePayloadChunk(const char *chunk_ptr, int local_index,
                               const RestartChunkLayout &layout,
                               hydro::Hydro *phydro, mhd::MHD *pmhd,
                               radiation::Radiation *prad, TurbulenceDriver *pturb,
                               z4c::Z4c *pz4c, adm::ADM *padm,
                               HostArray5D<Real> &hyd_scratch,
                               HostArray5D<Real> &mhd_cc_scratch,
                               HostFaceFld4D<Real> &mhd_fc_scratch,
                               HostArray5D<Real> &rad_scratch,
                               HostArray5D<Real> &turb_scratch,
                               HostArray5D<Real> &grav_scratch) {
  if (phydro != nullptr && layout.hydro_bytes > 0) {
    auto hyd_ptr = Kokkos::subview(hyd_scratch, 0, Kokkos::ALL, Kokkos::ALL,
                                   Kokkos::ALL, Kokkos::ALL);
    std::memcpy(hyd_ptr.data(), chunk_ptr + layout.hydro_offset, layout.hydro_bytes);
    Kokkos::deep_copy(Kokkos::subview(phydro->u0, local_index, Kokkos::ALL, Kokkos::ALL,
                                      Kokkos::ALL, Kokkos::ALL), hyd_ptr);
  }

  if (pmhd != nullptr && layout.mhd_cc_bytes > 0) {
    auto mhd_ptr = Kokkos::subview(mhd_cc_scratch, 0, Kokkos::ALL, Kokkos::ALL,
                                   Kokkos::ALL, Kokkos::ALL);
    std::memcpy(mhd_ptr.data(), chunk_ptr + layout.mhd_cc_offset, layout.mhd_cc_bytes);
    Kokkos::deep_copy(Kokkos::subview(pmhd->u0, local_index, Kokkos::ALL, Kokkos::ALL,
                                      Kokkos::ALL, Kokkos::ALL), mhd_ptr);

    auto x1f_ptr = Kokkos::subview(mhd_fc_scratch.x1f, 0, Kokkos::ALL, Kokkos::ALL,
                                   Kokkos::ALL);
    std::memcpy(x1f_ptr.data(), chunk_ptr + layout.mhd_x1f_offset, layout.mhd_x1f_bytes);
    Kokkos::deep_copy(Kokkos::subview(pmhd->b0.x1f, local_index, Kokkos::ALL,
                                      Kokkos::ALL, Kokkos::ALL), x1f_ptr);

    auto x2f_ptr = Kokkos::subview(mhd_fc_scratch.x2f, 0, Kokkos::ALL, Kokkos::ALL,
                                   Kokkos::ALL);
    std::memcpy(x2f_ptr.data(), chunk_ptr + layout.mhd_x2f_offset, layout.mhd_x2f_bytes);
    Kokkos::deep_copy(Kokkos::subview(pmhd->b0.x2f, local_index, Kokkos::ALL,
                                      Kokkos::ALL, Kokkos::ALL), x2f_ptr);

    auto x3f_ptr = Kokkos::subview(mhd_fc_scratch.x3f, 0, Kokkos::ALL, Kokkos::ALL,
                                   Kokkos::ALL);
    std::memcpy(x3f_ptr.data(), chunk_ptr + layout.mhd_x3f_offset, layout.mhd_x3f_bytes);
    Kokkos::deep_copy(Kokkos::subview(pmhd->b0.x3f, local_index, Kokkos::ALL,
                                      Kokkos::ALL, Kokkos::ALL), x3f_ptr);
  }

  if (prad != nullptr && layout.rad_bytes > 0) {
    auto rad_ptr = Kokkos::subview(rad_scratch, 0, Kokkos::ALL, Kokkos::ALL,
                                   Kokkos::ALL, Kokkos::ALL);
    std::memcpy(rad_ptr.data(), chunk_ptr + layout.rad_offset, layout.rad_bytes);
    Kokkos::deep_copy(Kokkos::subview(prad->i0, local_index, Kokkos::ALL, Kokkos::ALL,
                                      Kokkos::ALL, Kokkos::ALL), rad_ptr);
  }

  if (pturb != nullptr && layout.turb_bytes > 0) {
    auto turb_ptr = Kokkos::subview(turb_scratch, 0, Kokkos::ALL, Kokkos::ALL,
                                    Kokkos::ALL, Kokkos::ALL);
    std::memcpy(turb_ptr.data(), chunk_ptr + layout.turb_offset, layout.turb_bytes);
    Kokkos::deep_copy(Kokkos::subview(pturb->force, local_index, Kokkos::ALL, Kokkos::ALL,
                                      Kokkos::ALL, Kokkos::ALL), turb_ptr);
  }

  if (pz4c != nullptr && layout.grav_bytes > 0) {
    auto grav_ptr = Kokkos::subview(grav_scratch, 0, Kokkos::ALL, Kokkos::ALL,
                                    Kokkos::ALL, Kokkos::ALL);
    std::memcpy(grav_ptr.data(), chunk_ptr + layout.grav_offset, layout.grav_bytes);
    Kokkos::deep_copy(Kokkos::subview(pz4c->u0, local_index, Kokkos::ALL, Kokkos::ALL,
                                      Kokkos::ALL, Kokkos::ALL), grav_ptr);
  } else if (padm != nullptr && layout.grav_bytes > 0) {
    auto grav_ptr = Kokkos::subview(grav_scratch, 0, Kokkos::ALL, Kokkos::ALL,
                                    Kokkos::ALL, Kokkos::ALL);
    std::memcpy(grav_ptr.data(), chunk_ptr + layout.grav_offset, layout.grav_bytes);
    Kokkos::deep_copy(Kokkos::subview(padm->u_adm, local_index, Kokkos::ALL, Kokkos::ALL,
                                      Kokkos::ALL, Kokkos::ALL), grav_ptr);
  }
}

void LoadPerNodeRestartData(Mesh *pm, IOWrapperSizeT data_stride,
                            int nout1, int nout2, int nout3,
                            int nhydro, int nmhd, int nrad,
                            int nforce, int nz4c, int nadm) {
  MeshBlockPack *pack = pm->pmb_pack;
  hydro::Hydro *phydro = pack->phydro;
  mhd::MHD *pmhd = pack->pmhd;
  adm::ADM *padm = pack->padm;
  z4c::Z4c *pz4c = pack->pz4c;
  radiation::Radiation *prad = pack->prad;
  TurbulenceDriver *pturb = pack->pturb;
  const RestartMetaData &meta = pm->restart_meta;

  RestartChunkLayout layout = BuildRestartChunkLayout(meta, data_stride, nout1, nout2, nout3,
                                                      nhydro, nmhd, nrad,
                                                      (pturb != nullptr) ? nforce : 0,
                                                      nz4c, nadm,
                                                      "Per-node restart data");
  std::vector<int> rank_chunk_prefix = BuildPerNodeRankChunkPrefix(meta);
  auto requests = BuildPerNodeRestartRequests(pm, meta, rank_chunk_prefix,
                                              layout.chunk_stride);

  HostArray5D<Real> hyd_scratch("rst-hyd-scratch", 1, 1, 1, 1, 1);
  HostArray5D<Real> mhd_cc_scratch("rst-mhd-cc-scratch", 1, 1, 1, 1, 1);
  HostFaceFld4D<Real> mhd_fc_scratch("rst-mhd-fc-scratch", 1, 1, 1, 1);
  HostArray5D<Real> rad_scratch("rst-rad-scratch", 1, 1, 1, 1, 1);
  HostArray5D<Real> turb_scratch("rst-turb-scratch", 1, 1, 1, 1, 1);
  HostArray5D<Real> grav_scratch("rst-grav-scratch", 1, 1, 1, 1, 1);

  if (phydro != nullptr && nhydro > 0) {
    Kokkos::realloc(hyd_scratch, 1, nhydro, nout3, nout2, nout1);
  }
  if (pmhd != nullptr && nmhd > 0) {
    Kokkos::realloc(mhd_cc_scratch, 1, nmhd, nout3, nout2, nout1);
    Kokkos::realloc(mhd_fc_scratch.x1f, 1, nout3, nout2, nout1 + 1);
    Kokkos::realloc(mhd_fc_scratch.x2f, 1, nout3, nout2 + 1, nout1);
    Kokkos::realloc(mhd_fc_scratch.x3f, 1, nout3 + 1, nout2, nout1);
  }
  if (prad != nullptr && nrad > 0) {
    Kokkos::realloc(rad_scratch, 1, nrad, nout3, nout2, nout1);
  }
  if (pturb != nullptr && nforce > 0) {
    Kokkos::realloc(turb_scratch, 1, nforce, nout3, nout2, nout1);
  }
  if (pz4c != nullptr && nz4c > 0) {
    Kokkos::realloc(grav_scratch, 1, nz4c, nout3, nout2, nout1);
  } else if (padm != nullptr && nadm > 0) {
    Kokkos::realloc(grav_scratch, 1, nadm, nout3, nout2, nout1);
  }

  std::vector<std::string> shard_paths(meta.original_nnodes);
  for (int shard = 0; shard < meta.original_nnodes; ++shard) {
    char shard_dir[20];
    std::snprintf(shard_dir, sizeof(shard_dir), "node_%08d", shard);
    if (!meta.base_dir.empty()) {
      shard_paths[shard] = meta.base_dir + "/" + shard_dir + "/" + meta.file_name;
    } else {
      shard_paths[shard] = std::string(shard_dir) + "/" + meta.file_name;
    }
  }

  for (int shard = 0; shard < meta.original_nnodes; ++shard) {
    auto &shard_requests = requests[shard];
    int local_has_requests = shard_requests.empty() ? 0 : 1;
#if MPI_PARALLEL_ENABLED
    int node_has_requests = local_has_requests;
    MPI_Allreduce(MPI_IN_PLACE, &node_has_requests, 1, MPI_INT, MPI_MAX,
                  global_variable::node_comm);
    if (node_has_requests == 0) {
      continue;
    }
#else
    if (!local_has_requests) {
      continue;
    }
#endif

    auto spans = BuildPerNodeRestartSpans(shard_requests, layout.chunk_stride);
    int collective_reads = static_cast<int>(spans.size());
#if MPI_PARALLEL_ENABLED
    MPI_Allreduce(MPI_IN_PLACE, &collective_reads, 1, MPI_INT, MPI_MAX,
                  global_variable::node_comm);
#endif

    IOWrapper srcfile;
#if MPI_PARALLEL_ENABLED
    srcfile.SetCommunicator(global_variable::node_comm);
    srcfile.Open(shard_paths[shard].c_str(), IOWrapper::FileMode::read, false);
#else
    srcfile.Open(shard_paths[shard].c_str(), IOWrapper::FileMode::read, true);
#endif

    char dummy = '\0';
    for (int span_idx = 0; span_idx < collective_reads; ++span_idx) {
      const PerNodeRestartSpan *span = (span_idx < static_cast<int>(spans.size()))
          ? &spans[span_idx] : nullptr;
      IOWrapperSizeT local_bytes = (span == nullptr) ? 0 : span->byte_count;
      std::vector<char> span_buffer;
      char *span_data = &dummy;
      if (local_bytes > 0) {
        span_buffer.resize(static_cast<std::size_t>(local_bytes));
        span_data = span_buffer.data();
      }

#if MPI_PARALLEL_ENABLED
      std::size_t bytes_read = srcfile.Read_bytes_at_all(span_data, 1, local_bytes,
                                                         (span == nullptr) ? 0
                                                                           : span->file_offset,
                                                         false);
#else
      std::size_t bytes_read = srcfile.Read_bytes_at(span_data, 1, local_bytes,
                                                     (span == nullptr) ? 0
                                                                       : span->file_offset,
                                                     true);
#endif
      if (bytes_read != local_bytes) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "Per-node restart payload not read correctly from shard "
                  << shard_paths[shard] << ", restart file is broken." << std::endl;
        std::exit(EXIT_FAILURE);
      }

      if (span == nullptr) {
        continue;
      }

      for (int req_idx = 0; req_idx < span->request_count; ++req_idx) {
        const auto &request = shard_requests[span->first_request + req_idx];
        const char *chunk_ptr = span_data
                              + static_cast<IOWrapperSizeT>(req_idx)*layout.chunk_stride;
        UnpackPerNodePayloadChunk(chunk_ptr, request.local_index, layout, phydro, pmhd,
                                  prad, pturb, pz4c, padm, hyd_scratch, mhd_cc_scratch,
                                  mhd_fc_scratch, rad_scratch, turb_scratch, grav_scratch);
      }
    }

#if MPI_PARALLEL_ENABLED
    srcfile.Close(false);
#else
    srcfile.Close(true);
#endif

    PrintPerNodeRestartStats(shard, static_cast<int>(shard_requests.size()),
                             static_cast<int>(spans.size()), collective_reads);
  }

  if (pz4c != nullptr) {
    pz4c->Z4cToADM(pm->pmb_pack);
  }
}

void LoadPartitionedRestartData(Mesh *pm,
                                IOWrapperSizeT headeroffset,
                                IOWrapperSizeT data_stride,
                                int nout1, int nout2, int nout3,
                                int nhydro, int nmhd, int nrad,
                                int nforce, int nz4c, int nadm,
                                HostArray5D<Real> &ccin,
                                HostFaceFld4D<Real> &fcin) {
  MeshBlockPack *pack = pm->pmb_pack;
  int nmb = pack->nmb_thispack;
  hydro::Hydro* phydro = pack->phydro;
  mhd::MHD* pmhd = pack->pmhd;
  adm::ADM* padm = pack->padm;
  z4c::Z4c* pz4c = pack->pz4c;
  radiation::Radiation* prad = pack->prad;
  TurbulenceDriver* pturb = pack->pturb;

  const RestartMetaData &meta = pm->restart_meta;
  if (meta.file_name.empty()) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Restart metadata missing file name for single-file restart."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (meta.rank_eachmb.size() != static_cast<std::size_t>(pm->nmb_total)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Restart metadata inconsistent with MeshBlock count."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (meta.original_nranks <= 0 ||
      meta.gids_eachrank.size() != static_cast<std::size_t>(meta.original_nranks) ||
      meta.nmb_eachrank.size() != static_cast<std::size_t>(meta.original_nranks)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Restart metadata missing original rank layout."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (meta.file_shard_mode == FileShardMode::per_node &&
      (meta.original_nnodes <= 0 ||
       meta.rank_to_node.size() != static_cast<std::size_t>(meta.original_nranks))) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Restart metadata missing original node layout."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }

  if (meta.file_shard_mode == FileShardMode::per_node) {
    LoadPerNodeRestartData(pm, data_stride, nout1, nout2, nout3, nhydro, nmhd, nrad,
                           nforce, nz4c, nadm);
    return;
  }

  int nshards = meta.original_nranks;
  if (meta.file_shard_mode == FileShardMode::per_node) {
    nshards = meta.original_nnodes;
  }
  std::vector<std::vector<RestartBlockRequest>> requests(nshards);
  for (int m=0; m<nmb; ++m) {
    int gid = pack->pmb->mb_gid.h_view(m);
    if (gid < 0 || gid >= pm->nmb_total) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Invalid MeshBlock gid encountered during restart."
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
    int src_rank = meta.rank_eachmb[gid];
    if (src_rank < 0 || src_rank >= meta.original_nranks) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Restart metadata contains invalid rank assignments."
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
    int src_shard = src_rank;
    if (meta.file_shard_mode == FileShardMode::per_node) {
      src_shard = meta.rank_to_node[src_rank];
      if (src_shard < 0 || src_shard >= meta.original_nnodes) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "Restart metadata contains invalid node assignments."
                  << std::endl;
        std::exit(EXIT_FAILURE);
      }
    }
    requests[src_shard].push_back({m, gid, src_rank});
  }

  std::vector<std::string> shard_paths(nshards);
  for (int s=0; s<nshards; ++s) {
    char shard_dir[20];
    std::snprintf(shard_dir, sizeof(shard_dir),
                  meta.file_shard_mode == FileShardMode::per_node ? "node_%08d"
                                                                  : "rank_%08d",
                  s);
    if (!meta.base_dir.empty()) {
      shard_paths[s] = meta.base_dir + "/" + shard_dir + "/" + meta.file_name;
    } else {
      shard_paths[s] = std::string(shard_dir) + "/" + meta.file_name;
    }
  }

  IOWrapperSizeT chunk_stride = data_stride;
  if (meta.payload_stride != 0) {
    chunk_stride = static_cast<IOWrapperSizeT>(meta.payload_stride);
  }
  if (meta.file_shard_mode == FileShardMode::per_node && chunk_stride != data_stride) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Per-node restart payload stride does not match local "
              << "physics payload size." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  const IOWrapperSizeT shard_payload_offset =
      (meta.file_shard_mode == FileShardMode::per_node) ? 0 : headeroffset;
  IOWrapperSizeT chunk_offset = 0;
  const IOWrapperSizeT hydro_offset = chunk_offset;
  chunk_offset += nout1*nout2*nout3*nhydro*sizeof(Real);
  const IOWrapperSizeT mhd_cc_offset = chunk_offset;
  chunk_offset += nout1*nout2*nout3*nmhd*sizeof(Real);
  const IOWrapperSizeT mhd_x1f_offset = chunk_offset;
  if (pmhd != nullptr) chunk_offset += (nout1+1)*nout2*nout3*sizeof(Real);
  const IOWrapperSizeT mhd_x2f_offset = chunk_offset;
  if (pmhd != nullptr) chunk_offset += nout1*(nout2+1)*nout3*sizeof(Real);
  const IOWrapperSizeT mhd_x3f_offset = chunk_offset;
  if (pmhd != nullptr) chunk_offset += nout1*nout2*(nout3+1)*sizeof(Real);
  const IOWrapperSizeT rad_offset = chunk_offset;
  chunk_offset += nout1*nout2*nout3*nrad*sizeof(Real);
  const IOWrapperSizeT turb_offset = chunk_offset;
  if (pturb != nullptr) {
    chunk_offset += nout1*nout2*nout3*nforce*sizeof(Real);
  }
  const IOWrapperSizeT z4c_adm_offset = chunk_offset;
  if (pz4c != nullptr) {
    chunk_offset += nout1*nout2*nout3*nz4c*sizeof(Real);
  } else if (padm != nullptr) {
    chunk_offset += nout1*nout2*nout3*nadm*sizeof(Real);
  }
  if (chunk_offset != chunk_stride) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Restart data chunk size mismatch, restart file is broken."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }

  auto chunk_base = [&](int src_rank, int global_id) -> IOWrapperSizeT {
    int start_gid = meta.gids_eachrank[src_rank];
    int local_index = global_id - start_gid;
    if (local_index < 0 || (meta.nmb_eachrank[src_rank] > 0 &&
                            local_index >= meta.nmb_eachrank[src_rank])) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Restart metadata inconsistent with MeshBlock ids."
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
    if (meta.file_shard_mode == FileShardMode::per_node) {
      int src_node = meta.rank_to_node[src_rank];
      for (int r=0; r<src_rank; ++r) {
        if (meta.rank_to_node[r] == src_node) {
          local_index += meta.nmb_eachrank[r];
        }
      }
    }
    return shard_payload_offset + chunk_stride * static_cast<IOWrapperSizeT>(local_index);
  };

  if (phydro != nullptr && nhydro > 0) {
    Kokkos::realloc(ccin, nmb, nhydro, nout3, nout2, nout1);
    for (int s=0; s<nshards; ++s) {
      auto &reqs = requests[s];
      if (reqs.empty()) continue;
      IOWrapper srcfile;
      srcfile.Open(shard_paths[s].c_str(), IOWrapper::FileMode::read, true);
      for (const auto &req : reqs) {
        auto mbptr = Kokkos::subview(ccin, req.local_index, Kokkos::ALL, Kokkos::ALL,
                                     Kokkos::ALL, Kokkos::ALL);
        int mbcnt = mbptr.size();
        if (mbcnt > 0) {
          IOWrapperSizeT base = chunk_base(req.src_rank, req.global_id);
          if (srcfile.Read_Reals_at(mbptr.data(), mbcnt, base + hydro_offset, true)
              != mbcnt) {
            std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                      << std::endl << "CC hydro data not read correctly from rst file, "
                      << "restart file is broken." << std::endl;
            std::exit(EXIT_FAILURE);
          }
        }
      }
      srcfile.Close(true);
    }
    Kokkos::deep_copy(Kokkos::subview(phydro->u0, std::make_pair(0,nmb), Kokkos::ALL,
                      Kokkos::ALL, Kokkos::ALL, Kokkos::ALL), ccin);
  }

  if (pmhd != nullptr && nmhd > 0) {
    Kokkos::realloc(ccin, nmb, nmhd, nout3, nout2, nout1);
    Kokkos::realloc(fcin.x1f, nmb, nout3, nout2, nout1+1);
    Kokkos::realloc(fcin.x2f, nmb, nout3, nout2+1, nout1);
    Kokkos::realloc(fcin.x3f, nmb, nout3+1, nout2, nout1);
    for (int s=0; s<nshards; ++s) {
      auto &reqs = requests[s];
      if (reqs.empty()) continue;
      IOWrapper srcfile;
      srcfile.Open(shard_paths[s].c_str(), IOWrapper::FileMode::read, true);
      for (const auto &req : reqs) {
        IOWrapperSizeT base = chunk_base(req.src_rank, req.global_id);
        auto mbptr = Kokkos::subview(ccin, req.local_index, Kokkos::ALL, Kokkos::ALL,
                                     Kokkos::ALL, Kokkos::ALL);
        int mbcnt = mbptr.size();
        if (mbcnt > 0) {
          if (srcfile.Read_Reals_at(mbptr.data(), mbcnt, base + mhd_cc_offset, true)
              != mbcnt) {
            std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                      << std::endl << "CC mhd data not read correctly from rst file, "
                      << "restart file is broken." << std::endl;
            std::exit(EXIT_FAILURE);
          }
        }

        auto x1fptr = Kokkos::subview(fcin.x1f, req.local_index, Kokkos::ALL, Kokkos::ALL,
                                       Kokkos::ALL);
        int fldcnt = x1fptr.size();
        if (fldcnt > 0) {
          if (srcfile.Read_Reals_at(x1fptr.data(), fldcnt, base + mhd_x1f_offset, true)
              != fldcnt) {
            std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                      << std::endl << "Input b0.x1f field not read correctly from rst file, "
                      << "restart file is broken." << std::endl;
            std::exit(EXIT_FAILURE);
          }
        }

        auto x2fptr = Kokkos::subview(fcin.x2f, req.local_index, Kokkos::ALL, Kokkos::ALL,
                                       Kokkos::ALL);
        fldcnt = x2fptr.size();
        if (fldcnt > 0) {
          if (srcfile.Read_Reals_at(x2fptr.data(), fldcnt, base + mhd_x2f_offset, true)
              != fldcnt) {
            std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                      << std::endl << "Input b0.x2f field not read correctly from rst file, "
                      << "restart file is broken." << std::endl;
            std::exit(EXIT_FAILURE);
          }
        }

        auto x3fptr = Kokkos::subview(fcin.x3f, req.local_index, Kokkos::ALL, Kokkos::ALL,
                                       Kokkos::ALL);
        fldcnt = x3fptr.size();
        if (fldcnt > 0) {
          if (srcfile.Read_Reals_at(x3fptr.data(), fldcnt, base + mhd_x3f_offset, true)
              != fldcnt) {
            std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                      << std::endl << "Input b0.x3f field not read correctly from rst file, "
                      << "restart file is broken." << std::endl;
            std::exit(EXIT_FAILURE);
          }
        }
      }
      srcfile.Close(true);
    }
    Kokkos::deep_copy(Kokkos::subview(pmhd->u0, std::make_pair(0,nmb), Kokkos::ALL,
                      Kokkos::ALL, Kokkos::ALL, Kokkos::ALL), ccin);
    Kokkos::deep_copy(Kokkos::subview(pmhd->b0.x1f, std::make_pair(0,nmb), Kokkos::ALL,
                      Kokkos::ALL, Kokkos::ALL), fcin.x1f);
    Kokkos::deep_copy(Kokkos::subview(pmhd->b0.x2f, std::make_pair(0,nmb), Kokkos::ALL,
                      Kokkos::ALL, Kokkos::ALL), fcin.x2f);
    Kokkos::deep_copy(Kokkos::subview(pmhd->b0.x3f, std::make_pair(0,nmb), Kokkos::ALL,
                      Kokkos::ALL, Kokkos::ALL), fcin.x3f);
  }

  if (prad != nullptr && nrad > 0) {
    Kokkos::realloc(ccin, nmb, nrad, nout3, nout2, nout1);
    for (int s=0; s<nshards; ++s) {
      auto &reqs = requests[s];
      if (reqs.empty()) continue;
      IOWrapper srcfile;
      srcfile.Open(shard_paths[s].c_str(), IOWrapper::FileMode::read, true);
      for (const auto &req : reqs) {
        auto mbptr = Kokkos::subview(ccin, req.local_index, Kokkos::ALL, Kokkos::ALL,
                                     Kokkos::ALL, Kokkos::ALL);
        int mbcnt = mbptr.size();
        if (mbcnt > 0) {
          IOWrapperSizeT base = chunk_base(req.src_rank, req.global_id);
          if (srcfile.Read_Reals_at(mbptr.data(), mbcnt, base + rad_offset, true)
              != mbcnt) {
            std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                      << std::endl << "CC rad data not read correctly from rst file, "
                      << "restart file is broken." << std::endl;
            std::exit(EXIT_FAILURE);
          }
        }
      }
      srcfile.Close(true);
    }
    Kokkos::deep_copy(Kokkos::subview(prad->i0, std::make_pair(0,nmb), Kokkos::ALL,
                      Kokkos::ALL, Kokkos::ALL, Kokkos::ALL), ccin);
  }

  if (pturb != nullptr && nforce > 0) {
    Kokkos::realloc(ccin, nmb, nforce, nout3, nout2, nout1);
    for (int s=0; s<nshards; ++s) {
      auto &reqs = requests[s];
      if (reqs.empty()) continue;
      IOWrapper srcfile;
      srcfile.Open(shard_paths[s].c_str(), IOWrapper::FileMode::read, true);
      for (const auto &req : reqs) {
        auto mbptr = Kokkos::subview(ccin, req.local_index, Kokkos::ALL, Kokkos::ALL,
                                     Kokkos::ALL, Kokkos::ALL);
        int mbcnt = mbptr.size();
        if (mbcnt > 0) {
          IOWrapperSizeT base = chunk_base(req.src_rank, req.global_id);
          if (srcfile.Read_Reals_at(mbptr.data(), mbcnt, base + turb_offset, true)
              != mbcnt) {
            std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                      << std::endl << "CC turb data not read correctly from rst file, "
                      << "restart file is broken." << std::endl;
            std::exit(EXIT_FAILURE);
          }
        }
      }
      srcfile.Close(true);
    }
    Kokkos::deep_copy(Kokkos::subview(pturb->force, std::make_pair(0,nmb), Kokkos::ALL,
                      Kokkos::ALL, Kokkos::ALL, Kokkos::ALL), ccin);
  }

  if (pz4c != nullptr && nz4c > 0) {
    Kokkos::realloc(ccin, nmb, nz4c, nout3, nout2, nout1);
    for (int s=0; s<nshards; ++s) {
      auto &reqs = requests[s];
      if (reqs.empty()) continue;
      IOWrapper srcfile;
      srcfile.Open(shard_paths[s].c_str(), IOWrapper::FileMode::read, true);
      for (const auto &req : reqs) {
        auto mbptr = Kokkos::subview(ccin, req.local_index, Kokkos::ALL, Kokkos::ALL,
                                     Kokkos::ALL, Kokkos::ALL);
        int mbcnt = mbptr.size();
        if (mbcnt > 0) {
          IOWrapperSizeT base = chunk_base(req.src_rank, req.global_id);
          if (srcfile.Read_Reals_at(mbptr.data(), mbcnt, base + z4c_adm_offset, true)
              != mbcnt) {
            std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                      << std::endl << "CC z4c data not read correctly from rst file, "
                      << "restart file is broken." << std::endl;
            std::exit(EXIT_FAILURE);
          }
        }
      }
      srcfile.Close(true);
    }
    Kokkos::deep_copy(Kokkos::subview(pz4c->u0, std::make_pair(0,nmb), Kokkos::ALL,
                      Kokkos::ALL, Kokkos::ALL, Kokkos::ALL), ccin);
    pz4c->Z4cToADM(pm->pmb_pack);
  } else if (padm != nullptr && nadm > 0) {
    Kokkos::realloc(ccin, nmb, nadm, nout3, nout2, nout1);
    for (int s=0; s<nshards; ++s) {
      auto &reqs = requests[s];
      if (reqs.empty()) continue;
      IOWrapper srcfile;
      srcfile.Open(shard_paths[s].c_str(), IOWrapper::FileMode::read, true);
      for (const auto &req : reqs) {
        auto mbptr = Kokkos::subview(ccin, req.local_index, Kokkos::ALL, Kokkos::ALL,
                                     Kokkos::ALL, Kokkos::ALL);
        int mbcnt = mbptr.size();
        if (mbcnt > 0) {
          IOWrapperSizeT base = chunk_base(req.src_rank, req.global_id);
          if (srcfile.Read_Reals_at(mbptr.data(), mbcnt, base + z4c_adm_offset, true)
              != mbcnt) {
            std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                      << std::endl << "CC adm data not read correctly from rst file, "
                      << "restart file is broken." << std::endl;
            std::exit(EXIT_FAILURE);
          }
        }
      }
      srcfile.Close(true);
    }
    Kokkos::deep_copy(Kokkos::subview(padm->u_adm, std::make_pair(0,nmb), Kokkos::ALL,
                      Kokkos::ALL, Kokkos::ALL, Kokkos::ALL), ccin);
  }
}

}  // namespace

//----------------------------------------------------------------------------------------
// default constructor, calls pgen function.

ProblemGenerator::ProblemGenerator(ParameterInput *pin, Mesh *pm) :
    user_bcs(false),
    user_srcs(false),
    user_hist(false),
    pmy_mesh_(pm) {
  // check for user-defined boundary conditions
  for (int dir=0; dir<6; ++dir) {
    if (pm->mesh_bcs[dir] == BoundaryFlag::user) {
      user_bcs = true;
    }
  }

  user_srcs = pin->GetOrAddBoolean("problem","user_srcs",false);
  user_hist = pin->GetOrAddBoolean("problem","user_hist",false);

#if USER_PROBLEM_ENABLED
  // call user-defined problem generator
  UserProblem(pin, false);
#else
  // else read name of built-in pgen from <problem> block in input file, and call
  std::string pgen_fun_name = pin->GetOrAddString("problem", "pgen_name", "none");

  if (pgen_fun_name.compare("advection") == 0) {
    Advection(pin, false);
  } else if (pgen_fun_name.compare("cpaw") == 0) {
    AlfvenWave(pin, false);
  } else if (pgen_fun_name.compare("gr_bondi") == 0) {
    BondiAccretion(pin, false);
  } else if (pgen_fun_name.compare("tetrad") == 0) {
    CheckOrthonormalTetrad(pin, false);
  } else if (pgen_fun_name.compare("hohlraum") == 0) {
    Hohlraum(pin, false);
  } else if (pgen_fun_name.compare("linear_wave") == 0) {
    LinearWave(pin, false);
  } else if (pgen_fun_name.compare("implode") == 0) {
    LWImplode(pin, false);
  } else if (pgen_fun_name.compare("gr_monopole") == 0) {
    Monopole(pin, false);
  } else if (pgen_fun_name.compare("orszag_tang") == 0) {
    OrszagTang(pin, false);
  } else if (pgen_fun_name.compare("rad_linear_wave") == 0) {
    RadiationLinearWave(pin, false);
  } else if (pgen_fun_name.compare("shock_tube") == 0) {
    ShockTube(pin, false);
  } else if (pgen_fun_name.compare("z4c_linear_wave") == 0) {
    Z4cLinearWave(pin, false);
  } else if (pgen_fun_name.compare("spherical_collapse") == 0) {
    SphericalCollapse(pin, false);
  } else if (pgen_fun_name.compare("diffusion") == 0) {
    Diffusion(pin, false);
  // else, name not set on command line or input file, print warning and quit
  } else {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
        << "Problem generator name could not be found in <problem> block in input file"
        << std::endl
        << "and it was not set by -D PROBLEM option on cmake command line during build"
        << std::endl
        << "Rerun cmake with -D PROBLEM=file to specify custom problem generator file"
        << std::endl;;
    std::exit(EXIT_FAILURE);
  }
#endif

  // Check that user defined BCs were enrolled if needed
  if (user_bcs) {
    if (user_bcs_func == nullptr) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "User BCs specified in <mesh> block, but not enrolled "
                << "by SetProblemData()." << std::endl;
      exit(EXIT_FAILURE);
    }
  }
  // Check that user defined srcterms were enrolled if needed
  if (user_srcs) {
    if (user_srcs_func == nullptr) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "User SRCs specified in <problem> block, but not "
                << "enrolled by UserProblem()." << std::endl;
      exit(EXIT_FAILURE);
    }
  }
  // Check that user defined history outputs were enrolled if needed
  if (user_hist) {
    if (user_hist_func == nullptr) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "User history output specified in <problem> block, but "
                << "not enrolled by UserProblem()." << std::endl;
      exit(EXIT_FAILURE);
    }
  }
}

//----------------------------------------------------------------------------------------
// constructor for restarts
// When called, data needed to rebuild mesh has been read from restart file by
// Mesh::BuildTreeFromRestart() function. This constructor reads from the restart file and
// initializes all the dependent variables (u0,b0,etc) stored in each Physics class. It
// also calls ProblemGenerator::SetProblemData() function to set any user-defined BCs,
// and any data necessary for restart runs to continue correctly.

ProblemGenerator::ProblemGenerator(ParameterInput *pin, Mesh *pm, IOWrapper resfile,
                                   FileShardMode shard_mode) :
    user_bcs(false),
    user_srcs(false),
    user_hist(false),
    restart_file_shard_mode(shard_mode),
    pmy_mesh_(pm) {
  // check for user-defined boundary conditions
  for (int dir=0; dir<6; ++dir) {
    if (pm->mesh_bcs[dir] == BoundaryFlag::user) {
      user_bcs = true;
    }
  }
  user_srcs = pin->GetOrAddBoolean("problem","user_srcs",false);
  user_hist = pin->GetOrAddBoolean("problem","user_hist",false);
  bool use_serial_io = UsesSerialIO(shard_mode);

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
  radiation::Radiation* prad=pm->pmb_pack->prad;
  TurbulenceDriver* pturb=pm->pmb_pack->pturb;
  int nrad = 0, nhydro = 0, nmhd = 0, nforce = 3, nadm = 0, nz4c = 0;
  if (phydro != nullptr) {
    nhydro = phydro->nhydro + phydro->nscalars;
  }
  if (pmhd != nullptr) {
    nmhd = pmhd->nmhd + pmhd->nscalars;
  }
  if (prad != nullptr) {
    nrad = prad->prgeo->nangles;
  }
  if (pz4c != nullptr) {
    nz4c = pz4c->nz4c;
  } else if (padm != nullptr) {
    nadm = padm->nadm;
  }

  // root process reads z4c last_output_time and tracker data
  if (pz4c != nullptr) {
    Real last_output_time;
    if (global_variable::my_rank == 0 || use_serial_io) {
      if (resfile.Read_Reals(&last_output_time, 1, use_serial_io) != 1) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "z4c::last_output_time data size read from restart "
                  << "file is incorrect, restart file is broken." << std::endl;
        exit(EXIT_FAILURE);
      }
    }
#if MPI_PARALLEL_ENABLED
    if (!use_serial_io) {
      io_wrapper::BroadcastBytes(&last_output_time, sizeof(Real), 0, MPI_COMM_WORLD);
    }
#endif
    pz4c->last_output_time = last_output_time;

    for (auto &pt : pz4c->ptracker) {
      Real pos[3];
      if (global_variable::my_rank == 0 || use_serial_io) {
        if (resfile.Read_Reals(&pos[0], 3, use_serial_io) != 3) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                    << std::endl << "compact object tracker data size read from restart "
                    << "file is incorrect, restart file is broken." << std::endl;
          exit(EXIT_FAILURE);
        }
      }
#if MPI_PARALLEL_ENABLED
      if (!use_serial_io) {
        io_wrapper::BroadcastBytes(&pos[0], 3*sizeof(Real), 0, MPI_COMM_WORLD);
      }
#endif
      pt.SetPos(&pos[0]);
    }
  }

  if (pturb != nullptr) {
    // root process reads size the random seed
    char *rng_data = new char[sizeof(RNG_State)];
    // the master process reads the variables data
    if (global_variable::my_rank == 0 || use_serial_io) {
      if (resfile.Read_bytes(rng_data, 1, sizeof(RNG_State), use_serial_io)
          != sizeof(RNG_State)) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "RNG data size read from restart file is incorrect, "
                  << "restart file is broken." << std::endl;
        exit(EXIT_FAILURE);
      }
    }
#if MPI_PARALLEL_ENABLED
    if (!use_serial_io) {
      // then broadcast the RNG information
      io_wrapper::BroadcastBytes(rng_data, sizeof(RNG_State), 0, MPI_COMM_WORLD);
    }
#endif
    std::memcpy(&(pturb->rstate), &(rng_data[0]), sizeof(RNG_State));
  }

  // root process reads size of CC and FC data arrays from restart file
  IOWrapperSizeT variablesize = sizeof(IOWrapperSizeT);
  char *variabledata = new char[variablesize];
  if (global_variable::my_rank == 0 || use_serial_io) {
    if (resfile.Read_bytes(variabledata, 1, variablesize, use_serial_io)
        != variablesize) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Variable data size read from restart file is incorrect, "
                << "restart file is broken." << std::endl;
      exit(EXIT_FAILURE);
    }
  }
#if MPI_PARALLEL_ENABLED
  // then broadcast the datasize information
  if (!use_serial_io) {
    io_wrapper::BroadcastBytes(variabledata, variablesize, 0, MPI_COMM_WORLD);
  }
#endif
  IOWrapperSizeT data_size;
  std::memcpy(&data_size, &(variabledata[0]), sizeof(IOWrapperSizeT));
  pm->restart_meta.payload_stride = static_cast<std::uint64_t>(data_size);

  // calculate total number of CC variables
  IOWrapperSizeT headeroffset;
  // master process gets file offset
  if (global_variable::my_rank == 0 || use_serial_io) {
    headeroffset = resfile.GetPosition(use_serial_io);
  }
#if MPI_PARALLEL_ENABLED
  // then broadcasts it
  if (!use_serial_io) {
    io_wrapper::BroadcastBytes(&headeroffset, sizeof(IOWrapperSizeT), 0, MPI_COMM_WORLD);
  }
#endif

  IOWrapperSizeT data_size_ = 0;
  if (phydro != nullptr) {
    data_size_ += nout1*nout2*nout3*nhydro*sizeof(Real); // hydro u0
  }
  if (pmhd != nullptr) {
    data_size_ += nout1*nout2*nout3*nmhd*sizeof(Real);   // mhd u0
    data_size_ += (nout1+1)*nout2*nout3*sizeof(Real);    // mhd b0.x1f
    data_size_ += nout1*(nout2+1)*nout3*sizeof(Real);    // mhd b0.x2f
    data_size_ += nout1*nout2*(nout3+1)*sizeof(Real);    // mhd b0.x3f
  }
  if (prad != nullptr) {
    data_size_ += nout1*nout2*nout3*nrad*sizeof(Real);   // rad i0
  }
  if (pturb != nullptr) {
    data_size_ += nout1*nout2*nout3*nforce*sizeof(Real); // forcing
  }
  if (pz4c != nullptr) {
    data_size_ += nout1*nout2*nout3*nz4c*sizeof(Real);   // z4c u0
  } else if (padm != nullptr) {
    data_size_ += nout1*nout2*nout3*nadm*sizeof(Real);   // adm u_adm
  }

  if (data_size_ != data_size) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "CC data size read from restart file not equal to size "
              << "of Hydro, MHD, Rad, and/or Z4c arrays, restart file is broken."
              << std::endl;
    exit(EXIT_FAILURE);
  }

  HostArray5D<Real> ccin("rst-cc-in", 1, 1, 1, 1, 1);
  HostFaceFld4D<Real> fcin("rst-fc-in", 1, 1, 1, 1);

  if (shard_mode != FileShardMode::shared) {
    LoadPartitionedRestartData(pm, headeroffset, data_size_, nout1, nout2, nout3,
                               nhydro, nmhd, nrad, nforce, nz4c, nadm,
                               ccin, fcin);
  } else {
    // read CC data into host array
    int mygids = pm->gids_eachrank[global_variable::my_rank];
    IOWrapperSizeT offset_myrank = headeroffset;
    offset_myrank += data_size_ * pm->gids_eachrank[global_variable::my_rank];
    IOWrapperSizeT myoffset = offset_myrank;

    // calculate max/min number of MeshBlocks across all ranks
    int noutmbs_max = pm->nmb_eachrank[0];
    int noutmbs_min = pm->nmb_eachrank[0];
    for (int i=0; i<(global_variable::nranks); ++i) {
      noutmbs_max = std::max(noutmbs_max,pm->nmb_eachrank[i]);
      noutmbs_min = std::min(noutmbs_min,pm->nmb_eachrank[i]);
    }

  if (phydro != nullptr) {
    Kokkos::realloc(ccin, nmb, nhydro, nout3, nout2, nout1);
    for (int m=0;  m<noutmbs_max; ++m) {
      // every rank has a MB to read, so read collectively
      if (m < noutmbs_min) {
        // get ptr to cell-centered MeshBlock data
        auto mbptr = Kokkos::subview(ccin, m, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL,
                                     Kokkos::ALL);
        int mbcnt = mbptr.size();
        if (resfile.Read_Reals_at_all(mbptr.data(), mbcnt, myoffset, use_serial_io)
            != mbcnt) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                    << std::endl << "CC hydro data not read correctly from rst file, "
                    << "restart file is broken." << std::endl;
          exit(EXIT_FAILURE);
        }
        myoffset += data_size;

      // some ranks are finished writing, so use non-collective write
      } else if (m < pm->nmb_thisrank) {
        // get ptr to MeshBlock data
        auto mbptr = Kokkos::subview(ccin, m, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL,
                                     Kokkos::ALL);
        int mbcnt = mbptr.size();
        if (resfile.Read_Reals_at(mbptr.data(), mbcnt, myoffset, use_serial_io)
            != mbcnt) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                    << std::endl << "CC hydro data not read correctly from rst file, "
                    << "restart file is broken." << std::endl;
          exit(EXIT_FAILURE);
        }
        myoffset += data_size;
      }
    }
    Kokkos::deep_copy(Kokkos::subview(phydro->u0, std::make_pair(0,nmb), Kokkos::ALL,
                      Kokkos::ALL, Kokkos::ALL, Kokkos::ALL), ccin);
    offset_myrank += nout1*nout2*nout3*nhydro*sizeof(Real); // hydro u0
    myoffset = offset_myrank;
  }

  if (pmhd != nullptr) {
    Kokkos::realloc(ccin, nmb, nmhd, nout3, nout2, nout1);
    for (int m=0;  m<noutmbs_max; ++m) {
      // every rank has a MB to read, so read collectively
      if (m < noutmbs_min) {
        // get ptr to cell-centered MeshBlock data
        auto mbptr = Kokkos::subview(ccin, m, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL,
                                   Kokkos::ALL);
        int mbcnt = mbptr.size();
        if (resfile.Read_Reals_at_all(mbptr.data(), mbcnt, myoffset, use_serial_io)
            != mbcnt) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                    << std::endl << "CC mhd data not read correctly from rst file, "
                    << "restart file is broken." << std::endl;
          exit(EXIT_FAILURE);
        }
        myoffset += data_size;
      // some ranks are finished writing, so use non-collective write
      } else if (m < pm->nmb_thisrank) {
        // get ptr to MeshBlock data
        auto mbptr = Kokkos::subview(ccin, m, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL,
                                     Kokkos::ALL);
        int mbcnt = mbptr.size();
        if (resfile.Read_Reals_at(mbptr.data(), mbcnt, myoffset, use_serial_io)
            != mbcnt) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                    << std::endl << "CC mhd data not read correctly from rst file, "
                    << "restart file is broken." << std::endl;
          exit(EXIT_FAILURE);
        }
        myoffset += data_size;
      }
    }
    Kokkos::deep_copy(Kokkos::subview(pmhd->u0, std::make_pair(0,nmb), Kokkos::ALL,
                      Kokkos::ALL, Kokkos::ALL, Kokkos::ALL), ccin);
    offset_myrank += nout1*nout2*nout3*nmhd*sizeof(Real);   // mhd u0
    myoffset = offset_myrank;

    Kokkos::realloc(fcin.x1f, nmb, nout3, nout2, nout1+1);
    Kokkos::realloc(fcin.x2f, nmb, nout3, nout2+1, nout1);
    Kokkos::realloc(fcin.x3f, nmb, nout3+1, nout2, nout1);
    // read FC data into host array, again one MeshBlock at a time
    for (int m=0;  m<noutmbs_max; ++m) {
      // every rank has a MB to write, so write collectively
      if (m < noutmbs_min) {
        // get ptr to x1-face field
        auto x1fptr = Kokkos::subview(fcin.x1f, m, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
        int fldcnt = x1fptr.size();

        if (resfile.Read_Reals_at_all(x1fptr.data(), fldcnt, myoffset,
                                      use_serial_io) != fldcnt) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Input b0.x1f field not read correctly from rst file, "
                << "restart file is broken." << std::endl;
          exit(EXIT_FAILURE);
        }
        myoffset += fldcnt*sizeof(Real);

        // get ptr to x2-face field
        auto x2fptr = Kokkos::subview(fcin.x2f, m, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
        fldcnt = x2fptr.size();

        if (resfile.Read_Reals_at_all(x2fptr.data(), fldcnt, myoffset,
                                      use_serial_io) != fldcnt) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Input b0.x2f field not read correctly from rst file, "
                << "restart file is broken." << std::endl;
          exit(EXIT_FAILURE);
        }
        myoffset += fldcnt*sizeof(Real);

        // get ptr to x3-face field
        auto x3fptr = Kokkos::subview(fcin.x3f, m, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
        fldcnt = x3fptr.size();

        if (resfile.Read_Reals_at_all(x3fptr.data(), fldcnt, myoffset,
                                      use_serial_io) != fldcnt) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Input b0.x3f field not read correctly from rst file, "
                << "restart file is broken." << std::endl;
          exit(EXIT_FAILURE);
        }
        myoffset += fldcnt*sizeof(Real);

        myoffset += data_size-(x1fptr.size()+x2fptr.size()+x3fptr.size())*sizeof(Real);
      } else if (m < pm->nmb_thisrank) {
        // get ptr to x1-face field
        auto x1fptr = Kokkos::subview(fcin.x1f, m, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
        int fldcnt = x1fptr.size();

        if (resfile.Read_Reals_at(x1fptr.data(), fldcnt, myoffset,
                                  use_serial_io) != fldcnt) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Input b0.x1f field not read correctly from rst file, "
                << "restart file is broken." << std::endl;
          exit(EXIT_FAILURE);
        }
        myoffset += fldcnt*sizeof(Real);

        // get ptr to x2-face field
        auto x2fptr = Kokkos::subview(fcin.x2f, m, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
        fldcnt = x2fptr.size();

        if (resfile.Read_Reals_at(x2fptr.data(), fldcnt, myoffset,
                                  use_serial_io) != fldcnt) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Input b0.x2f field not read correctly from rst file, "
                << "restart file is broken." << std::endl;
          exit(EXIT_FAILURE);
        }
        myoffset += fldcnt*sizeof(Real);

        // get ptr to x3-face field
        auto x3fptr = Kokkos::subview(fcin.x3f, m, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
        fldcnt = x3fptr.size();

        if (resfile.Read_Reals_at(x3fptr.data(), fldcnt, myoffset,
                                  use_serial_io) != fldcnt) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Input b0.x3f field not read correctly from rst file, "
                << "restart file is broken." << std::endl;
          exit(EXIT_FAILURE);
        }
        myoffset += fldcnt*sizeof(Real);

        myoffset += data_size-(x1fptr.size()+x2fptr.size()+x3fptr.size())*sizeof(Real);
      }
    }
    Kokkos::deep_copy(Kokkos::subview(pmhd->b0.x1f, std::make_pair(0,nmb), Kokkos::ALL,
                      Kokkos::ALL, Kokkos::ALL), fcin.x1f);
    Kokkos::deep_copy(Kokkos::subview(pmhd->b0.x2f, std::make_pair(0,nmb), Kokkos::ALL,
                      Kokkos::ALL, Kokkos::ALL), fcin.x2f);
    Kokkos::deep_copy(Kokkos::subview(pmhd->b0.x3f, std::make_pair(0,nmb), Kokkos::ALL,
                      Kokkos::ALL, Kokkos::ALL), fcin.x3f);
    offset_myrank += (nout1+1)*nout2*nout3*sizeof(Real);    // mhd b0.x1f
    offset_myrank += nout1*(nout2+1)*nout3*sizeof(Real);    // mhd b0.x2f
    offset_myrank += nout1*nout2*(nout3+1)*sizeof(Real);    // mhd b0.x3f
    myoffset = offset_myrank;
  }

  if (prad != nullptr) {
    Kokkos::realloc(ccin, nmb, nrad, nout3, nout2, nout1);
    for (int m=0;  m<noutmbs_max; ++m) {
      // every rank has a MB to read, so read collectively
      if (m < noutmbs_min) {
        // get ptr to cell-centered MeshBlock data
        auto mbptr = Kokkos::subview(ccin, m, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL,
                                     Kokkos::ALL);
        int mbcnt = mbptr.size();
        if (resfile.Read_Reals_at_all(mbptr.data(), mbcnt, myoffset,
                                      use_serial_io) != mbcnt) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                    << std::endl << "CC rad data not read correctly from rst file, "
                    << "restart file is broken." << std::endl;
          exit(EXIT_FAILURE);
        }
        myoffset += data_size;

      // some ranks are finished writing, so use non-collective write
      } else if (m < pm->nmb_thisrank) {
        // get ptr to MeshBlock data
        auto mbptr = Kokkos::subview(ccin, m, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL,
                                     Kokkos::ALL);
        int mbcnt = mbptr.size();
        if (resfile.Read_Reals_at(mbptr.data(), mbcnt, myoffset,
                                  use_serial_io) != mbcnt) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                    << std::endl << "CC rad data not read correctly from rst file, "
                    << "restart file is broken." << std::endl;
          exit(EXIT_FAILURE);
        }
        myoffset += data_size;
      }
    }
    Kokkos::deep_copy(Kokkos::subview(prad->i0, std::make_pair(0,nmb), Kokkos::ALL,
                      Kokkos::ALL, Kokkos::ALL, Kokkos::ALL), ccin);
    offset_myrank += nout1*nout2*nout3*nrad*sizeof(Real);   // radiation i0
    myoffset = offset_myrank;
  }

  if (pturb != nullptr) {
    Kokkos::realloc(ccin, nmb, nforce, nout3, nout2, nout1);
    for (int m=0;  m<noutmbs_max; ++m) {
      // every rank has a MB to read, so read collectively
      if (m < noutmbs_min) {
        // get ptr to cell-centered MeshBlock data
        auto mbptr = Kokkos::subview(ccin, m, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL,
                                     Kokkos::ALL);
        int mbcnt = mbptr.size();
        if (resfile.Read_Reals_at_all(mbptr.data(), mbcnt, myoffset,
                                      use_serial_io) != mbcnt) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                    << std::endl << "CC turb data not read correctly from rst file, "
                    << "restart file is broken." << std::endl;
          exit(EXIT_FAILURE);
        }
        myoffset += data_size;

      // some ranks are finished writing, so use non-collective write
      } else if (m < pm->nmb_thisrank) {
        // get ptr to MeshBlock data
        auto mbptr = Kokkos::subview(ccin, m, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL,
                                     Kokkos::ALL);
        int mbcnt = mbptr.size();
        if (resfile.Read_Reals_at(mbptr.data(), mbcnt, myoffset,
                                  use_serial_io) != mbcnt) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                    << std::endl << "CC turb data not read correctly from rst file, "
                    << "restart file is broken." << std::endl;
          exit(EXIT_FAILURE);
        }
        myoffset += data_size;
      }
    }
    Kokkos::deep_copy(Kokkos::subview(pturb->force, std::make_pair(0,nmb), Kokkos::ALL,
                      Kokkos::ALL, Kokkos::ALL, Kokkos::ALL), ccin);
    offset_myrank += nout1*nout2*nout3*nforce*sizeof(Real); // forcing
    myoffset = offset_myrank;
  }

  if (pz4c != nullptr) {
    Kokkos::realloc(ccin, nmb, nz4c, nout3, nout2, nout1);
    for (int m=0;  m<noutmbs_max; ++m) {
      // every rank has a MB to read, so read collectively
      if (m < noutmbs_min) {
        // get ptr to cell-centered MeshBlock data
        auto mbptr = Kokkos::subview(ccin, m, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL,
                                     Kokkos::ALL);
        int mbcnt = mbptr.size();
        if (resfile.Read_Reals_at_all(mbptr.data(), mbcnt, myoffset,
                                      use_serial_io) != mbcnt) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                    << std::endl << "CC z4c data not read correctly from rst file, "
                    << "restart file is broken." << std::endl;
          exit(EXIT_FAILURE);
        }
        myoffset += data_size;

      // some ranks are finished writing, so use non-collective write
      } else if (m < pm->nmb_thisrank) {
        // get ptr to MeshBlock data
        auto mbptr = Kokkos::subview(ccin, m, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL,
                                     Kokkos::ALL);
        int mbcnt = mbptr.size();
        if (resfile.Read_Reals_at(mbptr.data(), mbcnt, myoffset,
                                  use_serial_io) != mbcnt) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                    << std::endl << "CC z4c data not read correctly from rst file, "
                    << "restart file is broken." << std::endl;
          exit(EXIT_FAILURE);
        }
        myoffset += data_size;
      }
    }
    Kokkos::deep_copy(Kokkos::subview(pz4c->u0, std::make_pair(0,nmb), Kokkos::ALL,
                      Kokkos::ALL, Kokkos::ALL, Kokkos::ALL), ccin);
    offset_myrank += nout1*nout2*nout3*nz4c*sizeof(Real);   // z4c u0
    myoffset = offset_myrank;

    // We also need to reinitialize the ADM data.
    pz4c->Z4cToADM(pmy_mesh_->pmb_pack);
  } else if (padm != nullptr) {
    Kokkos::realloc(ccin, nmb, nadm, nout3, nout2, nout1);
    for (int m=0;  m<noutmbs_max; ++m) {
      // every rank has a MB to read, so read collectively
      if (m < noutmbs_min) {
        // get ptr to cell-centered MeshBlock data
        auto mbptr = Kokkos::subview(ccin, m, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL,
                                     Kokkos::ALL);
        int mbcnt = mbptr.size();
        if (resfile.Read_Reals_at_all(mbptr.data(), mbcnt, myoffset,
                                      use_serial_io) != mbcnt) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                    << std::endl << "CC adm data not read correctly from rst file, "
                    << "restart file is broken." << std::endl;
          exit(EXIT_FAILURE);
        }
        myoffset += data_size;

      // some ranks are finished writing, so use non-collective write
      } else if (m < pm->nmb_thisrank) {
        // get ptr to MeshBlock data
        auto mbptr = Kokkos::subview(ccin, m, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL,
                                     Kokkos::ALL);
        int mbcnt = mbptr.size();
        if (resfile.Read_Reals_at(mbptr.data(), mbcnt, myoffset,
                                  use_serial_io) != mbcnt) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                    << std::endl << "CC adm data not read correctly from rst file, "
                    << "restart file is broken." << std::endl;
          exit(EXIT_FAILURE);
        }
        myoffset += data_size;
      }
    }
    Kokkos::deep_copy(Kokkos::subview(padm->u_adm, std::make_pair(0,nmb), Kokkos::ALL,
                      Kokkos::ALL, Kokkos::ALL, Kokkos::ALL), ccin);
    offset_myrank += nout1*nout2*nout3*nadm*sizeof(Real);   // adm u_adm
    myoffset = offset_myrank;
  }

  }

  // call problem generator again to re-initialize data, fn ptrs, as needed
#if USER_PROBLEM_ENABLED
  UserProblem(pin, true);
#else
  std::string pgen_fun_name = pin->GetOrAddString("problem", "pgen_name", "none");

  if (pgen_fun_name.compare("advection") == 0) {
    Advection(pin, true);
  } else if (pgen_fun_name.compare("cpaw") == 0) {
    AlfvenWave(pin, true);
  } else if (pgen_fun_name.compare("gr_bondi") == 0) {
    BondiAccretion(pin, true);
  } else if (pgen_fun_name.compare("tetrad") == 0) {
    CheckOrthonormalTetrad(pin, true);
  } else if (pgen_fun_name.compare("hohlraum") == 0) {
    Hohlraum(pin, true);
  } else if (pgen_fun_name.compare("linear_wave") == 0) {
    LinearWave(pin, true);
  } else if (pgen_fun_name.compare("implode") == 0) {
    LWImplode(pin, true);
  } else if (pgen_fun_name.compare("gr_monopole") == 0) {
    Monopole(pin, true);
  } else if (pgen_fun_name.compare("orszag_tang") == 0) {
    OrszagTang(pin, true);
  } else if (pgen_fun_name.compare("rad_linear_wave") == 0) {
    RadiationLinearWave(pin, true);
  } else if (pgen_fun_name.compare("shock_tube") == 0) {
    ShockTube(pin, true);
  } else if (pgen_fun_name.compare("z4c_linear_wave") == 0) {
    Z4cLinearWave(pin, true);
  } else if (pgen_fun_name.compare("spherical_collapse") == 0) {
    SphericalCollapse(pin, true);
  } else if (pgen_fun_name.compare("diffusion") == 0) {
    Diffusion(pin, true);
  } else {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
        << "Problem generator name could not be found in <problem> block in input file"
        << std::endl
        << "and it was not set by -D PROBLEM option on cmake command line during build"
        << std::endl
        << "Rerun cmake with -D PROBLEM=file to specify custom problem generator file"
        << std::endl;;
    std::exit(EXIT_FAILURE);
  }
#endif

  // Check that user defined BCs were enrolled if needed
  if (user_bcs) {
    if (user_bcs_func == nullptr) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "User BCs specified in <mesh> block, but not enrolled "
                << "during restart by SetProblemData()." << std::endl;
      exit(EXIT_FAILURE);
    }
  }
  // Check that user defined srcterms were enrolled if needed
  if (user_srcs) {
    if (user_srcs_func == nullptr) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "User SRCs specified in <problem> block, but not "
                << "enrolled by UserProblem()." << std::endl;
      exit(EXIT_FAILURE);
    }
  }
  // Check that user defined history outputs were enrolled if needed
  if (user_hist) {
    if (user_hist_func == nullptr) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "User history output specified in <problem> block, "
                << "but not enrolled by UserProblem()." << std::endl;
      exit(EXIT_FAILURE);
    }
  }
}
