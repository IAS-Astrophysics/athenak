//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file bvals.cpp
//! \brief constructors and initializers for both particle and Mesh variable boundary
//! classes.

#include <cstdlib>
#include <iostream>
#include <utility>
#include <algorithm> // max
#include <map>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/nghbr_index.hpp"
#include "mesh/mesh.hpp"
#include "particles/particles.hpp"
#include "bvals.hpp"

//----------------------------------------------------------------------------------------
// MeshBoundaryValues constructor:

MeshBoundaryValues::MeshBoundaryValues(MeshBlockPack *pp, ParameterInput *pin, bool z4c) :
  u_in("uin",1,1),
  b_in("bin",1,1),
  i_in("iin",1,1)
#if MPI_PARALLEL_ENABLED
  ,
  rank_packed_bvals_nvars_(-1),
  rank_packed_mesh_seq_(-1),
  rank_sendbuf_vars_("rank_sendbuf_vars",1),
  rank_recvbuf_vars_("rank_recvbuf_vars",1),
  rank_sendhdr_vars_("rank_sendhdr_vars",1),
  rank_recvhdr_vars_("rank_recvhdr_vars",1),
  send_var_entries_d_("send_var_entries_d",1),
  recv_var_entries_d_("recv_var_entries_d",1),
  unpack_tasks_d_("unpack_tasks_d",1),
  send_agg_offset_("send_agg_offset",1),
  recv_agg_offset_("recv_agg_offset",1)
#endif
  ,
  pmy_pack(pp),
  is_z4c_(z4c)
{
  // allocate vector of status flags and MPI requests (if needed)
  int nnghbr = pmy_pack->pmb->nnghbr;

#if MPI_PARALLEL_ENABLED
  // Initialize all 56 MPI request pointers to nullptr first
  for (int n=0; n<56; ++n) {
    sendbuf[n].vars_req = nullptr;
    sendbuf[n].flux_req = nullptr;
    recvbuf[n].vars_req = nullptr;
    recvbuf[n].flux_req = nullptr;
  }
#endif

  // sendbuf and recvbuf are fixed-length [56-element] arrays
  // Initialize some of the data in appropriate elements based on dimensionality of
  // problem (indicated by value of nnghbr)
  for (int n=0; n<nnghbr; ++n) {
#if MPI_PARALLEL_ENABLED
    // allocate vector of MPI requests (if needed)
    int nmb = std::max((pmy_pack->nmb_thispack), (pmy_pack->pmesh->nmb_maxperrank));
    sendbuf[n].vars_req = new MPI_Request[nmb];
    sendbuf[n].flux_req = new MPI_Request[nmb];
    recvbuf[n].vars_req = new MPI_Request[nmb];
    recvbuf[n].flux_req = new MPI_Request[nmb];
    for (int m=0; m<nmb; ++m) {
      sendbuf[n].vars_req[m] = MPI_REQUEST_NULL;
      sendbuf[n].flux_req[m] = MPI_REQUEST_NULL;
      recvbuf[n].vars_req[m] = MPI_REQUEST_NULL;
      recvbuf[n].flux_req[m] = MPI_REQUEST_NULL;
    }
#endif
    // initialize data sizes in each send/recv buffer to zero
    sendbuf[n].isame_ndat = 0;
    sendbuf[n].isame_z4c_ndat = 0;
    sendbuf[n].icoar_ndat = 0;
    sendbuf[n].ifine_ndat = 0;
    sendbuf[n].iflxs_ndat = 0;
    sendbuf[n].iflxc_ndat = 0;
    recvbuf[n].isame_ndat = 0;
    recvbuf[n].isame_z4c_ndat = 0;
    recvbuf[n].icoar_ndat = 0;
    recvbuf[n].ifine_ndat = 0;
    recvbuf[n].iflxs_ndat = 0;
    recvbuf[n].iflxc_ndat = 0;
  }

#if MPI_PARALLEL_ENABLED
  // create unique communicators for variables and fluxes in this BoundaryValues object
  MPI_Comm_dup(MPI_COMM_WORLD, &comm_vars);
  MPI_Comm_dup(MPI_COMM_WORLD, &comm_flux);
#endif
}

//----------------------------------------------------------------------------------------
// MeshBoundaryValues destructor

MeshBoundaryValues::~MeshBoundaryValues() {
#if MPI_PARALLEL_ENABLED
  int nnghbr = pmy_pack->pmb->nnghbr;
  for (int n=0; n<nnghbr; ++n) {
    delete [] sendbuf[n].vars_req;
    delete [] sendbuf[n].flux_req;
    delete [] recvbuf[n].vars_req;
    delete [] recvbuf[n].flux_req;
  }
#endif
}

#if MPI_PARALLEL_ENABLED
int MeshBoundaryValues::GetVarDataSize(const MeshBoundaryBuffer &buf, int m, int n,
                                       int nvars) const {
  int data_size = nvars;
  auto &nghbr = pmy_pack->pmb->nghbr;
  auto &mblev = pmy_pack->pmb->mb_lev;
  if (nghbr.h_view(m,n).lev < mblev.h_view(m)) {
    data_size *= buf.icoar_ndat;
  } else if (nghbr.h_view(m,n).lev == mblev.h_view(m)) {
    if (is_z4c_) {
      data_size *= buf.isame_z4c_ndat;
    } else {
      data_size *= buf.isame_ndat;
    }
  } else {
    data_size *= buf.ifine_ndat;
  }
  return data_size;
}

void MeshBoundaryValues::BuildRankPackedVarMetadata(const int nvars) {
  rank_packed_bvals_nvars_ = nvars;
  rank_packed_mesh_seq_ = pmy_pack->pmesh->GetAMRLoadBalanceUpdateSeq();
  send_var_entries_.clear();
  recv_var_entries_.clear();
  send_var_msgs_.clear();
  recv_var_msgs_.clear();
  send_var_reqs_.clear();
  recv_var_reqs_.clear();

  int nmb = pmy_pack->nmb_thispack;
  int nnghbr = pmy_pack->pmb->nnghbr;
  auto &nghbr = pmy_pack->pmb->nghbr;
  int my_rank = global_variable::my_rank;

  std::map<int, std::vector<RankPackedVarEntry>> send_by_rank;
  std::map<int, std::vector<RankPackedVarEntry>> recv_by_rank;

  for (int m=0; m<nmb; ++m) {
    for (int n=0; n<nnghbr; ++n) {
      if (nghbr.h_view(m,n).gid < 0) continue;
      int drank = nghbr.h_view(m,n).rank;
      if (drank == my_rank) continue;

      RankPackedVarEntry send_entry;
      send_entry.m = m;
      send_entry.n = n;
      send_entry.lid = nghbr.h_view(m,n).gid - pmy_pack->pmesh->gids_eachrank[drank];
      send_entry.dn = nghbr.h_view(m,n).dest;
      send_entry.data_size = GetVarDataSize(sendbuf[n], m, n, nvars);
      send_entry.offset = 0;
      send_by_rank[drank].push_back(send_entry);

      RankPackedVarEntry recv_entry;
      recv_entry.m = m;
      recv_entry.n = n;
      recv_entry.lid = m;
      recv_entry.dn = n;
      recv_entry.data_size = GetVarDataSize(recvbuf[n], m, n, nvars);
      recv_entry.offset = 0;
      recv_by_rank[drank].push_back(recv_entry);
    }
  }

  int send_total = 0;
  int send_hdr_total = 0;
  int send_entry_total = 0;
  for (auto &kv : send_by_rank) {
    int msg_offset = send_total;
    int hdr_offset = send_hdr_total;
    int entry_offset = send_entry_total;
    for (auto &entry : kv.second) {
      entry.offset = send_total;
      send_var_entries_.push_back(entry);
      send_total += entry.data_size;
      ++send_entry_total;
    }
    send_hdr_total += 3*static_cast<int>(kv.second.size());
    RankPackedVarMessage msg;
    msg.rank = kv.first;
    msg.nentries = static_cast<int>(kv.second.size());
    msg.entry_offset = entry_offset;
    msg.hdr_offset = hdr_offset;
    msg.offset = msg_offset;
    msg.data_size = send_total - msg_offset;
    send_var_msgs_.push_back(msg);
  }

  int recv_total = 0;
  int recv_hdr_total = 0;
  int recv_entry_total = 0;
  for (auto &kv : recv_by_rank) {
    int msg_offset = recv_total;
    int hdr_offset = recv_hdr_total;
    int entry_offset = recv_entry_total;
    for (auto &entry : kv.second) {
      entry.offset = recv_total;
      recv_var_entries_.push_back(entry);
      recv_total += entry.data_size;
      ++recv_entry_total;
    }
    recv_hdr_total += 3*static_cast<int>(kv.second.size());
    RankPackedVarMessage msg;
    msg.rank = kv.first;
    msg.nentries = static_cast<int>(kv.second.size());
    msg.entry_offset = entry_offset;
    msg.hdr_offset = hdr_offset;
    msg.offset = msg_offset;
    msg.data_size = recv_total - msg_offset;
    recv_var_msgs_.push_back(msg);
  }

  Kokkos::realloc(rank_sendbuf_vars_, std::max(1, send_total));
  Kokkos::realloc(rank_recvbuf_vars_, std::max(1, recv_total));
  Kokkos::realloc(rank_sendhdr_vars_, std::max(1, send_hdr_total));
  Kokkos::realloc(rank_recvhdr_vars_, std::max(1, recv_hdr_total));

  // Mirror entry tables to device for the fused pack/unpack kernels.
  {
    const std::size_t nsend = send_var_entries_.size();
    const std::size_t nrecv = recv_var_entries_.size();
    send_var_entries_d_ =
        DvceArray1D<RankPackedVarEntry>("send_var_entries_d", nsend);
    recv_var_entries_d_ =
        DvceArray1D<RankPackedVarEntry>("recv_var_entries_d", nrecv);
    auto h_send = Kokkos::create_mirror_view(send_var_entries_d_);
    auto h_recv = Kokkos::create_mirror_view(recv_var_entries_d_);
    for (std::size_t i = 0; i < nsend; ++i) h_send(i) = send_var_entries_[i];
    for (std::size_t i = 0; i < nrecv; ++i) h_recv(i) = recv_var_entries_[i];
    Kokkos::deep_copy(send_var_entries_d_, h_send);
    Kokkos::deep_copy(recv_var_entries_d_, h_recv);
  }
  send_var_reqs_.assign(send_var_msgs_.size(), MPI_REQUEST_NULL);
  recv_var_reqs_.assign(recv_var_msgs_.size(), MPI_REQUEST_NULL);

  // Populate the send-side header buffer once. Each (lid,dn,data_size) triple
  // tells the receiver where the i-th entry in our payload should land on its
  // side. Derived from this rank's nghbr/mbgid metadata; valid until the next
  // BuildRankPackedVarMetadata.
  for (const auto &msg : send_var_msgs_) {
    for (int e = 0; e < msg.nentries; ++e) {
      const auto &entry = send_var_entries_[msg.entry_offset + e];
      const int hidx = msg.hdr_offset + 3*e;
      rank_sendhdr_vars_(hidx    ) = entry.lid;
      rank_sendhdr_vars_(hidx + 1) = entry.dn;
      rank_sendhdr_vars_(hidx + 2) = entry.data_size;
    }
  }

  // One-shot peer-to-peer header exchange: learn the order in which each peer
  // packs entries destined for this rank, so the per-step path can drop the
  // header message entirely. tag=2 avoids the payload tag=1.
  {
    const int meta_tag = 2;
    std::vector<MPI_Request> exch_reqs(
        send_var_msgs_.size() + recv_var_msgs_.size(), MPI_REQUEST_NULL);
    std::size_t r = 0;
    for (const auto &msg : recv_var_msgs_) {
      const int hdr_size = 3*msg.nentries;
      MPI_Irecv(rank_recvhdr_vars_.data() + msg.hdr_offset, hdr_size, MPI_INT,
                msg.rank, meta_tag, comm_vars, &exch_reqs[r++]);
    }
    for (const auto &msg : send_var_msgs_) {
      const int hdr_size = 3*msg.nentries;
      MPI_Isend(rank_sendhdr_vars_.data() + msg.hdr_offset, hdr_size, MPI_INT,
                msg.rank, meta_tag, comm_vars, &exch_reqs[r++]);
    }
    if (!exch_reqs.empty()) {
      MPI_Waitall(static_cast<int>(exch_reqs.size()), exch_reqs.data(),
                  MPI_STATUSES_IGNORE);
    }
  }

  // Build the cached unpack-task table + per-(m,n) aggregate-offset maps from
  // the received headers. Lets PackBuff/RecvBuff read/write the aggregate
  // buffer directly (fused), and the per-step path carry no header.
  {
    const int nmb_max =
        std::max(pmy_pack->nmb_thispack, pmy_pack->pmesh->nmb_maxperrank);
    const int n_recv_entries = static_cast<int>(recv_var_entries_.size());
    unpack_tasks_d_ = DvceArray1D<RankPackedVarEntry>(
        "unpack_tasks_d", std::max(1, n_recv_entries));
    auto unpack_tasks_h = Kokkos::create_mirror_view(unpack_tasks_d_);

    const int map_len = nmb*nnghbr;
    recv_agg_offset_ = DvceArray1D<int>("recv_agg_offset", std::max(1, map_len));
    send_agg_offset_ = DvceArray1D<int>("send_agg_offset", std::max(1, map_len));
    auto recv_off_h = Kokkos::create_mirror_view(recv_agg_offset_);
    auto send_off_h = Kokkos::create_mirror_view(send_agg_offset_);
    for (int i = 0; i < map_len; ++i) { recv_off_h(i) = -1; send_off_h(i) = -1; }

    int t = 0;
    for (const auto &msg : recv_var_msgs_) {
      int off = msg.offset;
      for (int e = 0; e < msg.nentries; ++e) {
        const int hidx = msg.hdr_offset + 3*e;
        const int lid = rank_recvhdr_vars_(hidx);
        const int dn = rank_recvhdr_vars_(hidx + 1);
        const int dsize = rank_recvhdr_vars_(hidx + 2);
        if ((lid < 0) || (lid >= nmb_max) || (dn < 0) || (dn >= nnghbr) ||
            (dsize < 0)) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line "
                    << __LINE__ << std::endl
                    << "Invalid rank-packed recv header from peer "
                    << msg.rank << std::endl;
          std::exit(EXIT_FAILURE);
        }
        unpack_tasks_h(t).m = 0;
        unpack_tasks_h(t).n = 0;
        unpack_tasks_h(t).lid = lid;
        unpack_tasks_h(t).dn = dn;
        unpack_tasks_h(t).data_size = dsize;
        unpack_tasks_h(t).offset = off;
        if (lid < nmb) recv_off_h(lid*nnghbr + dn) = off;
        ++t;
        off += dsize;
      }
    }
    Kokkos::deep_copy(unpack_tasks_d_, unpack_tasks_h);

    for (const auto &entry : send_var_entries_) {
      send_off_h(entry.m*nnghbr + entry.n) = entry.offset;
    }
    Kokkos::deep_copy(recv_agg_offset_, recv_off_h);
    Kokkos::deep_copy(send_agg_offset_, send_off_h);
  }
}
#endif

//----------------------------------------------------------------------------------------
//! \fn void MeshBoundaryValues::InitializeBuffers
//! \brief initialize each element of send/recv MeshBoundaryBuffers fixed-length arrays
//!
//! NOTE: order of vector elements is crucial and cannot be changed.  It must match
//! order of boundaries in nghbr vector
//! NOTE2: work here cannot be done in MeshBoundaryValues constructor since it calls pure
//! virtual functions that only get instantiated when the derived classes are constructed

void MeshBoundaryValues::InitializeBuffers(const int nvar) {
  // allocate memory for inflow BCs (but only if domain not strictly periodic)
  if (!(pmy_pack->pmesh->strictly_periodic)) {
    Kokkos::realloc(u_in, nvar, 6);
    Kokkos::realloc(b_in, 3, 6);   // always 3 components of face-fields
    Kokkos::realloc(i_in, nvar, 6);
  }

  // set number of subblocks in x2- and x3-dirs
  int nfx = 1, nfy = 1, nfz = 1;
  if (pmy_pack->pmesh->multilevel) {
    nfx = 2;
    if (pmy_pack->pmesh->multi_d) nfy = 2;
    if (pmy_pack->pmesh->three_d) nfz = 2;
  }

  // initialize buffers used for uniform grid and SMR/AMR calculations

  // x1 faces; NeighborIndex = [0,...,7]
  int nmb = std::max((pmy_pack->nmb_thispack), (pmy_pack->pmesh->nmb_maxperrank));
  for (int n=-1; n<=1; n+=2) {
    for (int fz=0; fz<nfz; fz++) {
      for (int fy = 0; fy<nfy; fy++) {
        int indx = NeighborIndex(n,0,0,fy,fz);
        InitSendIndices(sendbuf[indx],n, 0, 0, fy, fz);
        InitRecvIndices(recvbuf[indx],n, 0, 0, fy, fz);
        sendbuf[indx].AllocateBuffers(nmb, nvar, is_z4c_);
        recvbuf[indx].AllocateBuffers(nmb, nvar, is_z4c_);
        sendbuf[indx].faces.h_view(0) = 1;
        recvbuf[indx].faces.h_view(0) = 1;
        sendbuf[indx].faces.h_view(1) = 0;
        recvbuf[indx].faces.h_view(1) = 0;
        sendbuf[indx].faces.h_view(2) = 0;
        recvbuf[indx].faces.h_view(2) = 0;
        sendbuf[indx].faces.template modify<HostMemSpace>();
        recvbuf[indx].faces.template modify<HostMemSpace>();
        sendbuf[indx].faces.template sync<DevMemSpace>();
        recvbuf[indx].faces.template sync<DevMemSpace>();
      }
    }
  }

  // add more buffers in 2D
  if (pmy_pack->pmesh->multi_d) {
    // x2 faces; NeighborIndex = [8,...,15]
    for (int m=-1; m<=1; m+=2) {
      for (int fz=0; fz<nfz; fz++) {
        for (int fx=0; fx<nfx; fx++) {
          int indx = NeighborIndex(0,m,0,fx,fz);
          InitSendIndices(sendbuf[indx],0, m, 0, fx, fz);
          InitRecvIndices(recvbuf[indx],0, m, 0, fx, fz);
          sendbuf[indx].AllocateBuffers(nmb, nvar, is_z4c_);
          recvbuf[indx].AllocateBuffers(nmb, nvar, is_z4c_);
          sendbuf[indx].faces.h_view(0) = 0;
          recvbuf[indx].faces.h_view(0) = 0;
          sendbuf[indx].faces.h_view(1) = 1;
          recvbuf[indx].faces.h_view(1) = 1;
          sendbuf[indx].faces.h_view(2) = 0;
          recvbuf[indx].faces.h_view(2) = 0;
          sendbuf[indx].faces.template modify<HostMemSpace>();
          recvbuf[indx].faces.template modify<HostMemSpace>();
          sendbuf[indx].faces.template sync<DevMemSpace>();
          recvbuf[indx].faces.template sync<DevMemSpace>();
        }
      }
    }

    // x1x2 edges; NeighborIndex = [16,...,23]
    for (int m=-1; m<=1; m+=2) {
      for (int n=-1; n<=1; n+=2) {
        for (int fz=0; fz<nfz; fz++) {
          int indx = NeighborIndex(n,m,0,fz,0);
          InitSendIndices(sendbuf[indx],n, m, 0, fz, 0);
          InitRecvIndices(recvbuf[indx],n, m, 0, fz, 0);
          sendbuf[indx].AllocateBuffers(nmb, nvar, is_z4c_);
          recvbuf[indx].AllocateBuffers(nmb, nvar, is_z4c_);
          sendbuf[indx].faces.h_view(0) = 1;
          recvbuf[indx].faces.h_view(0) = 1;
          sendbuf[indx].faces.h_view(1) = 1;
          recvbuf[indx].faces.h_view(1) = 1;
          sendbuf[indx].faces.h_view(2) = 0;
          recvbuf[indx].faces.h_view(2) = 0;
          sendbuf[indx].faces.template modify<HostMemSpace>();
          recvbuf[indx].faces.template modify<HostMemSpace>();
          sendbuf[indx].faces.template sync<DevMemSpace>();
          recvbuf[indx].faces.template sync<DevMemSpace>();
        }
      }
    }
  }

  // add more buffers in 3D
  if (pmy_pack->pmesh->three_d) {
    // x3 faces; NeighborIndex = [24,...,31]
    for (int l=-1; l<=1; l+=2) {
      for (int fy=0; fy<nfy; fy++) {
        for (int fx=0; fx<nfx; fx++) {
          int indx = NeighborIndex(0,0,l,fx,fy);
          InitSendIndices(sendbuf[indx],0, 0, l, fx, fy);
          InitRecvIndices(recvbuf[indx],0, 0, l, fx, fy);
          sendbuf[indx].AllocateBuffers(nmb, nvar, is_z4c_);
          recvbuf[indx].AllocateBuffers(nmb, nvar, is_z4c_);
          sendbuf[indx].faces.h_view(0) = 0;
          recvbuf[indx].faces.h_view(0) = 0;
          sendbuf[indx].faces.h_view(1) = 0;
          recvbuf[indx].faces.h_view(1) = 0;
          sendbuf[indx].faces.h_view(2) = 1;
          recvbuf[indx].faces.h_view(2) = 1;
          sendbuf[indx].faces.template modify<HostMemSpace>();
          recvbuf[indx].faces.template modify<HostMemSpace>();
          sendbuf[indx].faces.template sync<DevMemSpace>();
          recvbuf[indx].faces.template sync<DevMemSpace>();
        }
      }
    }

    // x3x1 edges; NeighborIndex = [32,...,39]
    for (int l=-1; l<=1; l+=2) {
      for (int n=-1; n<=1; n+=2) {
        for (int fy=0; fy<nfy; fy++) {
          int indx = NeighborIndex(n,0,l,fy,0);
          InitSendIndices(sendbuf[indx],n, 0, l, fy, 0);
          InitRecvIndices(recvbuf[indx],n, 0, l, fy, 0);
          sendbuf[indx].AllocateBuffers(nmb, nvar, is_z4c_);
          recvbuf[indx].AllocateBuffers(nmb, nvar, is_z4c_);
          sendbuf[indx].faces.h_view(0) = 1;
          recvbuf[indx].faces.h_view(0) = 1;
          sendbuf[indx].faces.h_view(1) = 0;
          recvbuf[indx].faces.h_view(1) = 0;
          sendbuf[indx].faces.h_view(2) = 1;
          recvbuf[indx].faces.h_view(2) = 1;
          sendbuf[indx].faces.template modify<HostMemSpace>();
          recvbuf[indx].faces.template modify<HostMemSpace>();
          sendbuf[indx].faces.template sync<DevMemSpace>();
          recvbuf[indx].faces.template sync<DevMemSpace>();
        }
      }
    }

    // x2x3 edges; NeighborIndex = [40,...,47]
    for (int l=-1; l<=1; l+=2) {
      for (int m=-1; m<=1; m+=2) {
        for (int fx=0; fx<nfx; fx++) {
          int indx = NeighborIndex(0,m,l,fx,0);
          InitSendIndices(sendbuf[indx],0, m, l, fx, 0);
          InitRecvIndices(recvbuf[indx],0, m, l, fx, 0);
          sendbuf[indx].AllocateBuffers(nmb, nvar, is_z4c_);
          recvbuf[indx].AllocateBuffers(nmb, nvar, is_z4c_);
          sendbuf[indx].faces.h_view(0) = 0;
          recvbuf[indx].faces.h_view(0) = 0;
          sendbuf[indx].faces.h_view(1) = 1;
          recvbuf[indx].faces.h_view(1) = 1;
          sendbuf[indx].faces.h_view(2) = 1;
          recvbuf[indx].faces.h_view(2) = 1;
          sendbuf[indx].faces.template modify<HostMemSpace>();
          recvbuf[indx].faces.template modify<HostMemSpace>();
          sendbuf[indx].faces.template sync<DevMemSpace>();
          recvbuf[indx].faces.template sync<DevMemSpace>();
        }
      }
    }

    // corners; NeighborIndex = [48,...,55]
    for (int l=-1; l<=1; l+=2) {
      for (int m=-1; m<=1; m+=2) {
        for (int n=-1; n<=1; n+=2) {
          int indx = NeighborIndex(n,m,l,0,0);
          InitSendIndices(sendbuf[indx],n, m, l, 0, 0);
          InitRecvIndices(recvbuf[indx],n, m, l, 0, 0);
          sendbuf[indx].AllocateBuffers(nmb, nvar, is_z4c_);
          recvbuf[indx].AllocateBuffers(nmb, nvar, is_z4c_);
          sendbuf[indx].faces.h_view(0) = 1;
          recvbuf[indx].faces.h_view(0) = 1;
          sendbuf[indx].faces.h_view(1) = 1;
          recvbuf[indx].faces.h_view(1) = 1;
          sendbuf[indx].faces.h_view(2) = 1;
          recvbuf[indx].faces.h_view(2) = 1;
          sendbuf[indx].faces.template modify<HostMemSpace>();
          recvbuf[indx].faces.template modify<HostMemSpace>();
          sendbuf[indx].faces.template sync<DevMemSpace>();
          recvbuf[indx].faces.template sync<DevMemSpace>();
        }
      }
    }
  }

  return;
}

//----------------------------------------------------------------------------------------
// ParticlesBoundaryValues constructor:

particles::ParticlesBoundaryValues::ParticlesBoundaryValues(
  particles::Particles *pp, ParameterInput *pin) :
    sendlist("sendlist",1),
#if MPI_PARALLEL_ENABLED
    prtcl_rsendbuf("rsend",1),
    prtcl_rrecvbuf("rrecv",1),
    prtcl_isendbuf("isend",1),
    prtcl_irecvbuf("irecv",1),
#endif
    pmy_part(pp) {
#if MPI_PARALLEL_ENABLED
  //resize vectors over number of ranks
  nsends_eachrank.resize(global_variable::nranks);

  // create unique communicator for particles
  MPI_Comm_dup(MPI_COMM_WORLD, &mpi_comm_part);
#endif
}

//----------------------------------------------------------------------------------------
// destructor

particles::ParticlesBoundaryValues::~ParticlesBoundaryValues() {
}
