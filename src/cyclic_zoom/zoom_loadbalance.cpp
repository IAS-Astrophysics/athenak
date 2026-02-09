//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file zoom_loadbalance.cpp
//! \brief Functions to handle load balancing of zoom data during cyclic zoom AMR

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "cyclic_zoom/cyclic_zoom.hpp"

// TODO(@mhguo): may need a more flexible packing function
// TODO(@mhguo): also a more flexible function to get the data from host

//----------------------------------------------------------------------------------------
//! \fn ZoomData::PackBuffer()
//! \brief Packs data into AMR communication buffers for all MBs being sent

void ZoomData::PackBuffer() {
  if (pzoom->verbose && global_variable::my_rank == 0) {
    std::cout << "CyclicZoom: Packing data into communication buffer" << std::endl;
  }
  // pack data for all zmbs on this device
  auto &pmbp = pzoom->pmesh->pmb_pack;
  // use size_t for offset to avoid overflow
  size_t offset = 0;
  size_t cc_cnt = u0.extent(1) * u0.extent(2) * u0.extent(3) * u0.extent(4);
  size_t ccc_cnt = coarse_u0.extent(1) * coarse_u0.extent(2) * coarse_u0.extent(3) * coarse_u0.extent(4);
  size_t ec_cnt = 0;
  if (pmbp->pmhd != nullptr) {
    // use efld_pre to get sizes for all edge fields
    auto efld = efld_pre;
    ec_cnt = efld.x1e.extent(1) * efld.x1e.extent(2) * efld.x1e.extent(3);
    ec_cnt += efld.x2e.extent(1) * efld.x2e.extent(2) * efld.x2e.extent(3);
    ec_cnt += efld.x3e.extent(1) * efld.x3e.extent(2) * efld.x3e.extent(3);
  }
  size_t i0_cnt = 0;
  size_t ci0_cnt = 0;
  if (pmbp->prad != nullptr) {
    i0_cnt = i0.extent(1) * i0.extent(2) * i0.extent(3) * i0.extent(4);
    ci0_cnt = coarse_i0.extent(1) * coarse_i0.extent(2) * coarse_i0.extent(3) * coarse_i0.extent(4);
  }
  auto dzbuf = zbuf.d_view;  // Get device view for packing
  for (int zm = 0; zm < pzmesh->nzmb_thisdvce; ++zm) {
    // offset = zm * zmb_data_cnt;
    if (pmbp->phydro != nullptr || pmbp->pmhd != nullptr) {
      // pack conserved variables
      PackBuffersCC(dzbuf, offset, zm, u0);
      offset += cc_cnt;
      // pack primitive variables
      PackBuffersCC(dzbuf, offset, zm, w0);
      offset += cc_cnt;
      // pack coarse conserved variables
      PackBuffersCC(dzbuf, offset, zm, coarse_u0);
      offset += ccc_cnt;
      // pack coarse primitive variables
      PackBuffersCC(dzbuf, offset, zm, coarse_w0);
      offset += ccc_cnt;
    }
    // pack magnetic fields and/or electric fields if MHD
    if (pmbp->pmhd != nullptr) {
      PackBuffersEC(dzbuf, offset, zm, efld_pre);
      offset += ec_cnt;
      PackBuffersEC(dzbuf, offset, zm, efld_aft);
      offset += ec_cnt;
      PackBuffersEC(dzbuf, offset, zm, delta_efld);
      offset += ec_cnt;
    }
    // pack radiation variables if radiation is enabled
    if (pmbp->prad != nullptr) {
      PackBuffersCC(dzbuf, offset, zm, i0);
      offset += i0_cnt;
      PackBuffersCC(dzbuf, offset, zm, coarse_i0);
      offset += ci0_cnt;
    }
  }
  // Single copy: device buffer -> host buffer
  // Only copy the portion that's actually used
  size_t used_size = pzmesh->nzmb_thisdvce * zmb_data_cnt;
  if (offset != used_size) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << ": Packed data size does not match expected size!" << std::endl;
    std::cout << "Packed size: " << offset << ", Expected size: " << used_size << std::endl;
    std::exit(EXIT_FAILURE);
  }
  Kokkos::deep_copy(
    Kokkos::subview(zbuf.h_view, Kokkos::make_pair(size_t(0), used_size)),
    Kokkos::subview(dzbuf, Kokkos::make_pair(size_t(0), used_size))
  );
  return;
}

//----------------------------------------------------------------------------------------
//! \fn ZoomData::PackBuffersCC()
//! \brief Packs data into AMR communication buffers for all MBs being sent

void ZoomData::PackBuffersCC(DvceArray1D<Real> packed_data, size_t offset,
                             const int m, DvceArray5D<Real> a0) {
  // Pack array a0 at MeshBlock m into packed_data starting from offset
  int nv = a0.extent_int(1);
  int nk = a0.extent_int(2);
  int nj = a0.extent_int(3);
  int ni = a0.extent_int(4);
  // Pack using parallel kernel on device
  par_for("pack_cc", DevExeSpace(), 0, nv-1, 0, nk-1, 0, nj-1, 0, ni-1,
  KOKKOS_LAMBDA(const int v, const int k, const int j, const int i) {
    packed_data(offset + (((v*nk + k)*nj + j)*ni + i)) = a0(m,v,k,j,i);
  });
  return;
}

//----------------------------------------------------------------------------------------
//! \fn ZoomData::PackBuffersFC()
//! \brief Packs face-centered data into AMR communication buffers for all MBs being sent

void ZoomData::PackBuffersFC(DvceArray1D<Real> packed_data, size_t offset,
                             const int m, DvceFaceFld4D<Real> fc) {
  // Pack face field fc at MeshBlock m into packed_data starting from offset_fc
  // Pack f1
  int nk = fc.x1f.extent_int(1);
  int nj = fc.x1f.extent_int(2);
  int ni = fc.x1f.extent_int(3);
  par_for("pack_f1", DevExeSpace(), 0, nk-1, 0, nj-1, 0, ni-1,
  KOKKOS_LAMBDA(const int k, const int j, const int i) {
    packed_data(offset + (k*nj + j)*ni + i) = fc.x1f(m,k,j,i);
  });
  offset += nk*nj*ni;
  // Pack f2
  nk = fc.x2f.extent_int(1);
  nj = fc.x2f.extent_int(2);
  ni = fc.x2f.extent_int(3);
  par_for("pack_f2", DevExeSpace(), 0, nk-1, 0, nj-1, 0, ni-1,
  KOKKOS_LAMBDA(const int k, const int j, const int i) {
    packed_data(offset + (k*nj + j)*ni + i) = fc.x2f(m,k,j,i);
  });
  offset += nk*nj*ni;
  // Pack f3
  nk = fc.x3f.extent_int(1);
  nj = fc.x3f.extent_int(2);
  ni = fc.x3f.extent_int(3);
  par_for("pack_f3", DevExeSpace(), 0, nk-1, 0, nj-1, 0, ni-1,
  KOKKOS_LAMBDA(const int k, const int j, const int i) {
    packed_data(offset + (k*nj + j)*ni + i) = fc.x3f(m,k,j,i);
  });
  return;
}

//----------------------------------------------------------------------------------------
//! \fn ZoomData::PackBuffersEC()
//! \brief Packs edge-centered data into AMR communication buffers for all MBs being sent

void ZoomData::PackBuffersEC(DvceArray1D<Real> packed_data, size_t offset,
                             const int m, DvceEdgeFld4D<Real> ec) {
  // Pack edge field ec at MeshBlock m into packed_data starting from offset
  // Pack e1
  int nk = ec.x1e.extent_int(1);
  int nj = ec.x1e.extent_int(2);
  int ni = ec.x1e.extent_int(3);
  par_for("pack_e1", DevExeSpace(), 0, nk-1, 0, nj-1, 0, ni-1,
  KOKKOS_LAMBDA(const int k, const int j, const int i) {
    packed_data(offset + (k*nj + j)*ni + i) = ec.x1e(m,k,j,i);
  });
  offset += nk*nj*ni;
  // Pack e2
  nk = ec.x2e.extent_int(1);
  nj = ec.x2e.extent_int(2);
  ni = ec.x2e.extent_int(3);
  par_for("pack_e2", DevExeSpace(), 0, nk-1, 0, nj-1, 0, ni-1,
  KOKKOS_LAMBDA(const int k, const int j, const int i) {
    packed_data(offset + (k*nj + j)*ni + i) = ec.x2e(m,k,j,i);
  });
  offset += nk*nj*ni;
  // Pack e3
  nk = ec.x3e.extent_int(1);
  nj = ec.x3e.extent_int(2);
  ni = ec.x3e.extent_int(3);
  par_for("pack_e3", DevExeSpace(), 0, nk-1, 0, nj-1, 0, ni-1,
  KOKKOS_LAMBDA(const int k, const int j, const int i) {
    packed_data(offset + (k*nj + j)*ni + i) = ec.x3e(m,k,j,i);
  });
  return;
}

//----------------------------------------------------------------------------------------
//! \fn ZoomData::UnpackBuffer()
//! \brief Unpacks data from AMR communication buffers for all MBs being received

void ZoomData::UnpackBuffer() {
  // Sync only the used portion to device for bandwidth efficiency
  size_t used_size = pzmesh->nzmb_thisdvce * zmb_data_cnt;
  Kokkos::deep_copy(
    Kokkos::subview(zbuf.d_view, Kokkos::make_pair(size_t(0), used_size)),
    Kokkos::subview(zbuf.h_view, Kokkos::make_pair(size_t(0), used_size))
  );
  // Unpack data from device buffer to zoom data arrays
  auto dzbuf = zbuf.d_view;  // Get device view for unpacking
  auto &pmbp = pzoom->pmesh->pmb_pack;
  // use size_t for offset to avoid overflow
  size_t offset = 0;
  size_t cc_cnt = u0.extent(1) * u0.extent(2) * u0.extent(3) * u0.extent(4);
  size_t ccc_cnt = coarse_u0.extent(1) * coarse_u0.extent(2) * coarse_u0.extent(3) * coarse_u0.extent(4);
  size_t ec_cnt = 0;
  if (pmbp->pmhd != nullptr) {
    // use efld_pre to get sizes for all edge fields
    auto efld = efld_pre;
    ec_cnt = efld.x1e.extent(1) * efld.x1e.extent(2) * efld.x1e.extent(3);
    ec_cnt += efld.x2e.extent(1) * efld.x2e.extent(2) * efld.x2e.extent(3);
    ec_cnt += efld.x3e.extent(1) * efld.x3e.extent(2) * efld.x3e.extent(3);
  }
  size_t i0_cnt = 0;
  size_t ci0_cnt = 0;
  if (pmbp->prad != nullptr) {
    i0_cnt = i0.extent(1) * i0.extent(2) * i0.extent(3) * i0.extent(4);
    ci0_cnt = coarse_i0.extent(1) * coarse_i0.extent(2) * coarse_i0.extent(3) * coarse_i0.extent(4);
  }
  for (int zm = 0; zm < pzmesh->nzmb_thisdvce; ++zm) {
    if (pzoom->verbose) {
      std::cout << " Rank " << global_variable::my_rank 
                << " Unpacking buffer for zmb " << zm << std::endl;
    }
    // offset = zm * zmb_data_cnt;
    if (pmbp->phydro != nullptr || pmbp->pmhd != nullptr) {
      // unpack conserved variables
      UnpackBuffersCC(dzbuf, offset, zm, u0);
      offset += cc_cnt;
      // unpack primitive variables
      UnpackBuffersCC(dzbuf, offset, zm, w0);
      offset += cc_cnt;
      // unpack coarse conserved variables
      UnpackBuffersCC(dzbuf, offset, zm, coarse_u0);
      offset += ccc_cnt;
      // unpack coarse primitive variables
      UnpackBuffersCC(dzbuf, offset, zm, coarse_w0);
      offset += ccc_cnt;
    }
    // unpack magnetic fields and/or electric fields if MHD
    if (pmbp->pmhd != nullptr) {
      UnpackBuffersEC(dzbuf, offset, zm, efld_pre);
      offset += ec_cnt;
      UnpackBuffersEC(dzbuf, offset, zm, efld_aft);
      offset += ec_cnt;
      UnpackBuffersEC(dzbuf, offset, zm, delta_efld);
      offset += ec_cnt;
    }
    // unpack radiation variables if radiation is enabled
    if (pmbp->prad != nullptr) {
      UnpackBuffersCC(dzbuf, offset, zm, i0);
      offset += i0_cnt;
      UnpackBuffersCC(dzbuf, offset, zm, coarse_i0);
      offset += ci0_cnt;
    }
  }
  if (offset != used_size) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << ": Unpacked data size does not match expected size!" << std::endl;
    std::cout << "Unpacked size: " << offset << ", Expected size: " << used_size << std::endl;
    std::exit(EXIT_FAILURE);
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn ZoomData::UnpackBuffersCC()
//! \brief Unpacks cell-centered data from AMR communication buffers

void ZoomData::UnpackBuffersCC(DvceArray1D<Real> packed_data, size_t offset,
                               int m, DvceArray5D<Real> a0) {
  // Unpack array a0 at MeshBlock m from packed_data starting from offset_a0
  int nv = a0.extent_int(1);
  int nk = a0.extent_int(2);
  int nj = a0.extent_int(3);
  int ni = a0.extent_int(4);
  // Unpack using parallel kernel on device
  par_for("unpack_cc", DevExeSpace(), 0, nv-1, 0, nk-1, 0, nj-1, 0, ni-1,
  KOKKOS_LAMBDA(const int v, const int k, const int j, const int i) {
    a0(m,v,k,j,i) = packed_data(offset + (((v*nk + k)*nj + j)*ni + i));
  });
  return;
}

//----------------------------------------------------------------------------------------
//! \fn ZoomData::UnpackBuffersFC()
//! \brief Unpacks face-centered data from AMR communication buffers

void ZoomData::UnpackBuffersFC(DvceArray1D<Real> packed_data, size_t offset,
                               int m, DvceFaceFld4D<Real> fc) {
  // Unpack face field fc at MeshBlock m from packed_data starting from offset
  // Unpack f1
  int nk = fc.x1f.extent_int(1);
  int nj = fc.x1f.extent_int(2);
  int ni = fc.x1f.extent_int(3);
  par_for("unpack_f1", DevExeSpace(), 0, nk-1, 0, nj-1, 0, ni-1,
  KOKKOS_LAMBDA(const int k, const int j, const int i) {
    fc.x1f(m,k,j,i) = packed_data(offset + (k*nj + j)*ni + i);
  });
  offset += nk*nj*ni;
  // Unpack f2
  nk = fc.x2f.extent_int(1);
  nj = fc.x2f.extent_int(2);
  ni = fc.x2f.extent_int(3);
  par_for("unpack_f2", DevExeSpace(), 0, nk-1, 0, nj-1, 0, ni-1,
  KOKKOS_LAMBDA(const int k, const int j, const int i) {
    fc.x2f(m,k,j,i) = packed_data(offset + (k*nj + j)*ni + i);
  });
  offset += nk*nj*ni;
  // Unpack f3
  nk = fc.x3f.extent_int(1);
  nj = fc.x3f.extent_int(2);
  ni = fc.x3f.extent_int(3);
  par_for("unpack_f3", DevExeSpace(), 0, nk-1, 0, nj-1, 0, ni-1,
  KOKKOS_LAMBDA(const int k, const int j, const int i) {
    fc.x3f(m,k,j,i) = packed_data(offset + (k*nj + j)*ni + i);
  });
  return;
}

//----------------------------------------------------------------------------------------
//! \fn ZoomData::UnpackBuffersEC()
//! \brief Unpacks edge-centered data from AMR communication buffers

void ZoomData::UnpackBuffersEC(DvceArray1D<Real> packed_data, size_t offset,
                               int m, DvceEdgeFld4D<Real> ec) {
  // Unpack edge field ec at MeshBlock m from packed_data starting from offset
  // Unpack e1
  int nk = ec.x1e.extent_int(1);
  int nj = ec.x1e.extent_int(2);
  int ni = ec.x1e.extent_int(3);
  par_for("unpack_e1", DevExeSpace(), 0, nk-1, 0, nj-1, 0, ni-1,
  KOKKOS_LAMBDA(const int k, const int j, const int i) {
    ec.x1e(m,k,j,i) = packed_data(offset + (k*nj + j)*ni + i);
  });
  offset += nk*nj*ni;
  // Unpack e2
  nk = ec.x2e.extent_int(1);
  nj = ec.x2e.extent_int(2);
  ni = ec.x2e.extent_int(3);
  par_for("unpack_e2", DevExeSpace(), 0, nk-1, 0, nj-1, 0, ni-1,
  KOKKOS_LAMBDA(const int k, const int j, const int i) {
    ec.x2e(m,k,j,i) = packed_data(offset + (k*nj + j)*ni + i);
  });
  offset += nk*nj*ni;
  // Unpack e3
  nk = ec.x3e.extent_int(1);
  nj = ec.x3e.extent_int(2);
  ni = ec.x3e.extent_int(3);
  par_for("unpack_e3", DevExeSpace(), 0, nk-1, 0, nj-1, 0, ni-1,
  KOKKOS_LAMBDA(const int k, const int j, const int i) {
    ec.x3e(m,k,j,i) = packed_data(offset + (k*nj + j)*ni + i);
  });
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void ZoomData::RedistZMBs()
//! \brief Generic MPI redistribution between two buffers with flexible indexing
//!
//! \details This function redistributes Zoom MeshBlocks (ZMBs) between two host buffers
//!          across MPI ranks. It supports two indexing schemes:
//!
//!          1. Dense indexing: Sequential local indices (0, 1, 2, ...) for ZMBs owned by
//!             a rank. Used by zbuf during computation where each rank processes a 
//!             contiguous subset of ZMBs.
//!
//!          2. Logical indexing: Global indices specified by lid_eachzmb array, which may
//!             be scattered. Used by zdata for persistent storage where each ZMB has a
//!             fixed global position regardless of which rank owns it.
//!
//!          The redistribution handles three communication patterns:
//!          - Remote receives: ZMBs where destination rank differs from source rank
//!          - Remote sends: ZMBs sent from this rank to other ranks
//!          - Local copies: ZMBs that stay on the same rank but may need reindexing
//!
//!          Typical usage:
//!          - SaveToStorage: zbuf (dense) → zdata (logical) before AMR refinement
//!          - LoadFromStorage: zdata (logical) → zbuf (dense) after AMR refinement
//!
//! \param[in] nlmb      Number of ZMBs at this level
//! \param[in] lmbs      Global starting index for ZMBs at this level
//! \param[in] src_buf   Source buffer containing data to send
//! \param[out] dst_buf  Destination buffer to receive data
//! \param[in] src_ranks Rank ownership array for source (size: total ZMBs at level)
//! \param[in] dst_ranks Rank ownership array for destination (size: total ZMBs at level)
//! \param[in] src_lids  Logical index array for source (nullptr = dense indexing)
//! \param[in] dst_lids  Logical index array for destination (nullptr = dense indexing)

void ZoomData::RedistZMBs(int nlmb, int lmbs,
                          HostArray1D<Real> src_buf,
                          HostArray1D<Real> dst_buf,
                          const std::vector<int>& src_ranks,
                          const std::vector<int>& dst_ranks,
                          const std::vector<int>* src_lids,
                          const std::vector<int>* dst_lids) {
  // Get ZMB information for this level
  size_t data_per_zmb = zmb_data_cnt;       // Data elements per ZMB

  int ncopy = 0;
  int my_rank = global_variable::my_rank;
#if MPI_PARALLEL_ENABLED
  int nsend = 0, nrecv = 0;
  std::vector<MPI_Request> requests;
#endif

  // Dense indexing counters (only incremented when rank owns the ZMB)
  int src_zm = 0;
  int dst_zm = 0;
  
  // Loop over all ZMBs at this level
  for (int lm = 0; lm < nlmb; ++lm) {
    int gzm = lm + lmbs;  // Global ZMB index
    int src_rank_val = src_ranks[gzm];
    int dst_rank_val = dst_ranks[gzm];
    
    // Compute source offset:
    // - If src_lids is nullptr: use dense indexing (sequential local index)
    // - Otherwise: use logical indexing from src_lids array
    size_t offset_src = (src_lids == nullptr) ? 
                        src_zm * data_per_zmb : 
                        (*src_lids)[gzm] * data_per_zmb;
    
    // Compute destination offset with same logic
    size_t offset_dst = (dst_lids == nullptr) ? 
                        dst_zm * data_per_zmb : 
                        (*dst_lids)[gzm] * data_per_zmb;

    // Local copy (same rank, but may need reindexing between dense/logical)
    if (src_rank_val == my_rank && dst_rank_val == my_rank) {
      Kokkos::deep_copy(
        Kokkos::subview(dst_buf, Kokkos::make_pair(offset_dst, offset_dst + data_per_zmb)),
        Kokkos::subview(src_buf, Kokkos::make_pair(offset_src, offset_src + data_per_zmb))
      );
      ++ncopy;
      // Increment both counters for local copies
      if (src_lids == nullptr) ++src_zm;
      if (dst_lids == nullptr) ++dst_zm;
    }

#if MPI_PARALLEL_ENABLED
    // Post receives first (avoids potential deadlock)
    if (dst_rank_val == my_rank && src_rank_val != my_rank) {
      MPI_Request req;
      MPI_Irecv(dst_buf.data() + offset_dst, data_per_zmb, 
                MPI_ATHENA_REAL, src_rank_val, lm, zoom_comm, &req);
      requests.push_back(req);
      ++nrecv;
      // Only increment dense counter if destination uses dense indexing
      if (dst_lids == nullptr) ++dst_zm;
    }

    // Post sends
    if (src_rank_val == my_rank && dst_rank_val != my_rank) {
      MPI_Request req;
      MPI_Isend(src_buf.data() + offset_src, data_per_zmb,
                MPI_ATHENA_REAL, dst_rank_val, lm, zoom_comm, &req);
      requests.push_back(req);
      ++nsend;
      // Only increment dense counter if source uses dense indexing
      if (src_lids == nullptr) ++src_zm;
    }
#endif
  }

#if MPI_PARALLEL_ENABLED
  // Wait for all asynchronous communications to complete
  if (!requests.empty()) {
    MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
  }
#endif

  if (pzoom->verbose && global_variable::my_rank == 0) {
    std::cout << "RedistZMBs: completed " 
#if MPI_PARALLEL_ENABLED
              << requests.size() << " MPI ops (sends: " << nsend 
              << ", recvs: " << nrecv << ", local: " << ncopy << ")" << std::endl;
#else
              << ncopy << " local copies" << std::endl;
#endif
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void ZoomData::SaveToStorage()
//! \brief Save ZMBs from computation buffer to persistent storage
//!
//! \details Wrapper for RedistZMBs that transfers data from zbuf (dense indexing,
//!          distributed for computation) to zdata (logical indexing, fixed global layout).
//!          Called before AMR refinement operations.
//!
//! \param[in] zone Zone level to save

void ZoomData::SaveToStorage(int zone) {
  auto hzbuf = zbuf.h_view;
  int nlmb = pzmesh->nzmb_eachlevel[zone];  // Number of ZMBs at this level
  int lmbs = pzmesh->gzms_eachlevel[zone];  // Global starting index for this level
  RedistZMBs(nlmb, lmbs,
             hzbuf, zdata,  // src: dense buffer, dst: logical storage
             pzmesh->rank_eachmb, pzmesh->rank_eachzmb,
             nullptr, &pzmesh->lid_eachzmb);
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void ZoomData::LoadFromStorage()
//! \brief Load ZMBs from persistent storage to computation buffer
//!
//! \details Wrapper for RedistZMBs that transfers data from zdata (logical indexing,
//!          fixed global layout) to zbuf (dense indexing, redistributed for computation).
//!          Called after AMR refinement operations.
//!
//! \param[in] zone Zone level to load

void ZoomData::LoadFromStorage(int zone) {
  auto hzbuf = zbuf.h_view;
  int nlmb = pzmesh->nzmb_eachlevel[zone];  // Number of ZMBs at this level
  int lmbs = pzmesh->gzms_eachlevel[zone];  // Global starting index for this level
  RedistZMBs(nlmb, lmbs,
             zdata, hzbuf,  // src: logical storage, dst: dense buffer
             pzmesh->rank_eachzmb, pzmesh->rank_eachmb,
             &pzmesh->lid_eachzmb, nullptr);
  return;
}
