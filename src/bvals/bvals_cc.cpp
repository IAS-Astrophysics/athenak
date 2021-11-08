//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file bvals_cc.cpp
//  \brief functions to pass boundary values for cell-centered variables as implemented in
//  BValCC class.

#include <cstdlib>
#include <iostream>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "bvals.hpp"
#include "utils/create_mpitag.hpp"

//----------------------------------------------------------------------------------------
// BValCC constructor:

BValCC::BValCC(MeshBlockPack *pp, ParameterInput *pin) : pmy_pack(pp)
{
} 
  
//----------------------------------------------------------------------------------------
// BValCC destructor
  
BValCC::~BValCC()
{
}

//----------------------------------------------------------------------------------------
// \!fn void BValCC::AllocateBuffersCC
// initialize array of send/recv BValBuffers for arbitrary number of cell-centered
// variables, specified by input argument.
//
// NOTE: order of array elements is crucial and cannot be changed.  It must match
// order of boundaries in nghbr vector

// TODO: extend for AMR

void BValCC::AllocateBuffersCC(const int nvar)
{
  auto &indcs = pmy_pack->pcoord->mbdata.indcs;
  int ng = indcs.ng;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;

  auto &cindcs = pmy_pack->pcoord->mbdata.cindcs;
  int cis = cindcs.is, cie = cindcs.ie;
  int cjs = cindcs.js, cje = cindcs.je;
  int cks = cindcs.ks, cke = cindcs.ke;

  int ng1 = ng-1;
  int nmb = pmy_pack->nmb_thispack;
  int nnghbr = pmy_pack->pmb->nnghbr;

  // allocate size of (some) Views
  for (int n=0; n<nnghbr; ++n) {
    // 6 values of index array stores loop bounds for each bbuf on both regular and coarse
    // mesh.  Values [0-5] are: il, iu, jl, ju, kl, ku
    Kokkos::realloc(send_buf[n].index, 6);
    Kokkos::realloc(recv_buf[n].index, 6);
    Kokkos::realloc(send_buf[n].cindex, 6);
    Kokkos::realloc(recv_buf[n].cindex, 6);
    Kokkos::realloc(send_buf[n].bcomm_stat, nmb);
    Kokkos::realloc(recv_buf[n].bcomm_stat, nmb);
#if MPI_PARALLEL_ENABLED
    // cannot create Kokkos::View of type MPI_Request so construct STL vector instead
    for (int m=0; m<nmb; ++m) {
      MPI_Request send_req, recv_req;
      send_buf[n].comm_req.push_back(send_req);
      recv_buf[n].comm_req.push_back(recv_req);
    }
#endif
  }

  // initialize buffers for x1 faces
  send_buf[0].InitIndices(is,     is+ng1, js, je, ks, ke);
  send_buf[1].InitIndices(ie-ng1, ie,     js, je, ks, ke);
  recv_buf[0].InitIndices(is-ng,  is-1,   js, je, ks, ke);
  recv_buf[1].InitIndices(ie+1,   ie+ng,  js, je, ks, ke);

  send_buf[0].InitCoarseIndices(cis,     cis+ng1, cjs, cje, cks, cke);
  send_buf[1].InitCoarseIndices(cie-ng1, cie,     cjs, cje, cks, cke);
  recv_buf[0].InitCoarseIndices(cis-ng,  cis-1,   cjs, cje, cks, cke);
  recv_buf[1].InitCoarseIndices(cie+1,   cie+ng,  cjs, cje, cks, cke);

  for (int n=0; n<2; ++n) {
    send_buf[n].AllocateDataView(nmb, nvar);
    recv_buf[n].AllocateDataView(nmb, nvar);
  }

  // add more buffers in 2D
  if (nnghbr > 2) {
    // initialize buffers for x2 faces
    send_buf[2].InitIndices(is, ie, js,     js+ng1, ks, ke);
    send_buf[3].InitIndices(is, ie, je-ng1, je,     ks, ke);

    recv_buf[2].InitIndices(is, ie, js-ng, js-1,  ks, ke);
    recv_buf[3].InitIndices(is, ie, je+1,  je+ng, ks, ke);

    // initialize buffers for x1x2 edges
    send_buf[4].InitIndices(is,     is+ng1, js,     js+ng1, ks, ke);
    send_buf[5].InitIndices(ie-ng1, ie,     js,     js+ng1, ks, ke);
    send_buf[6].InitIndices(is,     is+ng1, je-ng1, je,     ks, ke);
    send_buf[7].InitIndices(ie-ng1, ie,     je-ng1, je,     ks, ke);

    recv_buf[4].InitIndices(is-ng, is-1,  js-ng, js-1,  ks, ke);
    recv_buf[5].InitIndices(ie+1,  ie+ng, js-ng, js-1,  ks, ke);
    recv_buf[6].InitIndices(is-ng, is-1,  je+1,  je+ng, ks, ke);
    recv_buf[7].InitIndices(ie+1,  ie+ng, je+1,  je+ng, ks, ke);

    // add more buffers in 3D
    if (nnghbr > 8) {

      // initialize buffers for x3 faces
      send_buf[8].InitIndices(is, ie, js, je, ks,     ks+ng1);
      send_buf[9].InitIndices(is, ie, js, je, ke-ng1, ke    );
    
      recv_buf[8].InitIndices(is, ie, js, je, ks-ng, ks-1 );
      recv_buf[9].InitIndices(is, ie, js, je, ke+1,  ke+ng);

      // initialize buffers for x3x1 edges
      send_buf[10].InitIndices(is,     is+ng1, js, je, ks,     ks+ng1);
      send_buf[11].InitIndices(ie-ng1, ie,     js, je, ks,     ks+ng1);
      send_buf[12].InitIndices(is,     is+ng1, js, je, ke-ng1, ke    );
      send_buf[13].InitIndices(ie-ng1, ie,     js, je, ke-ng1, ke    );
    
      recv_buf[10].InitIndices(is-ng, is-1,  js, je, ks-ng, ks-1 );
      recv_buf[11].InitIndices(ie+1,  ie+ng, js, je, ks-ng, ks-1 );
      recv_buf[12].InitIndices(is-ng, is-1,  js, je, ke+1,  ke+ng);
      recv_buf[13].InitIndices(ie+1,  ie+ng, js, je, ke+1,  ke+ng);

      // initialize buffers for x2x3 edges
      send_buf[14].InitIndices(is, ie, js,     js+ng1, ks,     ks+ng1);
      send_buf[15].InitIndices(is, ie, je-ng1, je,     ks,     ks+ng1);
      send_buf[16].InitIndices(is, ie, js,     js+ng1, ke-ng1, ke    );
      send_buf[17].InitIndices(is, ie, je-ng1, je,     ke-ng1, ke    );
  
      recv_buf[14].InitIndices(is, ie, js-ng, js-1,  ks-ng, ks-1 );
      recv_buf[15].InitIndices(is, ie, je+1,  je+ng, ks-ng, ks-1 );
      recv_buf[16].InitIndices(is, ie, js-ng, js-1,  ke+1,  ke+ng);
      recv_buf[17].InitIndices(is, ie, je+1,  je+ng, ke+1,  ke+ng);

      // initialize buffers for corners
      send_buf[18].InitIndices(is,     is+ng1, js,     js+ng1, ks,     ks+ng1);
      send_buf[19].InitIndices(ie-ng1, ie,     js,     js+ng1, ks,     ks+ng1);
      send_buf[20].InitIndices(is,     is+ng1, je-ng1, je,     ks,     ks+ng1);
      send_buf[21].InitIndices(ie-ng1, ie,     je-ng1, je,     ks,     ks+ng1);
      send_buf[22].InitIndices(is,     is+ng1, js,     js+ng1, ke-ng1, ke    );
      send_buf[23].InitIndices(ie-ng1, ie,     js,     js+ng1, ke-ng1, ke    );
      send_buf[24].InitIndices(is,     is+ng1, je-ng1, je,     ke-ng1, ke    );
      send_buf[25].InitIndices(ie-ng1, ie,     je-ng1, je,     ke-ng1, ke    );

      recv_buf[18].InitIndices(is-ng, is-1,  js-ng, js-1,  ks-ng, ks-1 );
      recv_buf[19].InitIndices(ie+1,  ie+ng, js-ng, js-1,  ks-ng, ks-1 );
      recv_buf[20].InitIndices(is-ng, is-1,  je+1,  je+ng, ks-ng, ks-1 );
      recv_buf[21].InitIndices(ie+1,  ie+ng, je+1,  je+ng, ks-ng, ks-1 );
      recv_buf[22].InitIndices(is-ng, is-1,  js-ng, js-1,  ke+1,  ke+ng);
      recv_buf[23].InitIndices(ie+1,  ie+ng, js-ng, js-1,  ke+1,  ke+ng);
      recv_buf[24].InitIndices(is-ng, is-1,  je+1,  je+ng, ke+1,  ke+ng);
      recv_buf[25].InitIndices(ie+1,  ie+ng, je+1,  je+ng, ke+1,  ke+ng);
    }
  }

  // for index DualArray, mark host views as modified, and then sync to device array
  for (int n=0; n<nnghbr; ++n) {
    send_buf[n].index.template modify<HostMemSpace>();
    recv_buf[n].index.template modify<HostMemSpace>();

    send_buf[n].index.template sync<DevExeSpace>();
    recv_buf[n].index.template sync<DevExeSpace>();
  }

  return;
}


//----------------------------------------------------------------------------------------
// \!fn void BValCC::PackAndSendCC()
// \brief Pack cell-centered variables into boundary buffers and send to neighbors.
//
// This routine packs ALL the buffers on ALL the faces, edges, and corners simultaneously,
// for ALL the MeshBlocks.  This reduces the number of kernel launches when there are a
// large number of MeshBlocks per MPI rank.  Buffer data are then sent (via MPI) or copied
// directly for periodic or block boundaries.
//
// Input arrays must be 5D Kokkos View dimensioned (nmb, nvar, nx3, nx2, nx1)
// 5D Kokkos View of coarsened (restricted) array data also required with SMR/AMR 

TaskStatus BValCC::PackAndSendCC(DvceArray5D<Real> &a, DvceArray5D<Real> &ca, int key)
{
  // create local references for variables in kernel
  int nmb = pmy_pack->pmb->nmb;
  // TODO: following only works when all MBs have the same number of neighbors
  int nnghbr = pmy_pack->pmb->nnghbr;
  int nvar = a.extent_int(1);  // TODO: 2nd index from L of input array must be NVAR

  {int &my_rank = global_variable::my_rank;
  auto &nghbr = pmy_pack->pmb->nghbr;
  auto &mbgid = pmy_pack->pmb->mbgid;
  auto &mblev = pmy_pack->pmb->mblev;
  auto &sbuf = send_buf;
  auto &rbuf = recv_buf;

  // load buffers, using 3 levels of hierarchical parallelism
  // Outer loop over (# of MeshBlocks)*(# of buffers)*(# of variables)
  int nmnv = nmb*nnghbr*nvar;
  Kokkos::TeamPolicy<> policy(DevExeSpace(), nmnv, Kokkos::AUTO);
  Kokkos::parallel_for("SendBuff", policy, KOKKOS_LAMBDA(TeamMember_t tmember)
  { 
    const int m = (tmember.league_rank())/(nnghbr*nvar);
    const int n = (tmember.league_rank() - m*(nnghbr*nvar))/nvar;
    const int v = (tmember.league_rank() - m*(nnghbr*nvar) - n*nvar);
    // if neighbor is at same or finer level, use indices for u0
    // indices same for all variables, stored in (0,i) component
    int il, iu, jl, ju, kl, ku;
    if (nghbr.d_view(m,n).lev >= mblev.d_view(m)) {
      il = sbuf[n].index.d_view(0);
      iu = sbuf[n].index.d_view(1);
      jl = sbuf[n].index.d_view(2);
      ju = sbuf[n].index.d_view(3);
      kl = sbuf[n].index.d_view(4);
      ku = sbuf[n].index.d_view(5);
    // else if neighbor is at coarser level, use indices for coarse_u0
    } else {
      il = sbuf[n].cindex.d_view(0);
      iu = sbuf[n].cindex.d_view(1);
      jl = sbuf[n].cindex.d_view(2);
      ju = sbuf[n].cindex.d_view(3);
      kl = sbuf[n].cindex.d_view(4);
      ku = sbuf[n].cindex.d_view(5);
    }
    const int ni = iu - il + 1;
    const int nj = ju - jl + 1;
    const int nk = ku - kl + 1;
    const int nkj  = nk*nj;

    // Middle loop over k,j
    Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nkj), [&](const int idx)
    {
      int k = idx / nj;
      int j = (idx - k * nj) + jl;
      k += kl;
  
      // Inner (vector) loop over i
      // copy directly into recv buffer if MeshBlocks on same rank

      if (nghbr.d_view(m,n).rank == my_rank) {
        // indices of recv'ing MB and buffer: assumes MB IDs are stored sequentially
        // in this MeshBlockPack, so array index equals (target_id - first_id)
        int mm = nghbr.d_view(m,n).gid - mbgid.d_view(0);
        int nn = nghbr.d_view(m,n).destn;
        // if neighbor is at same or finer level, load data from u0
        if (nghbr.d_view(m,n).lev >= mblev.d_view(m)) {
          Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il,iu+1),[&](const int i)
          {
            rbuf[nn].data(mm,v, i-il + ni*(j-jl + nj*(k-kl))) = a(m,v,k,j,i);
          });
        // if neighbor is at coarser level, load data from coarse_u0
        } else {
          Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il,iu+1),[&](const int i)
          {
            rbuf[nn].data(mm,v, i-il + ni*(j-jl + nj*(k-kl))) = ca(m,v,k,j,i);
          });
        }

      // else copy into send buffer for MPI communication below

      } else {
        // if neighbor is at same or finer level, load data from u0
        if (nghbr.d_view(m,n).lev >= mblev.d_view(m)) {
          Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il,iu+1),[&](const int i)
          {
            sbuf[n].data(m, v, i-il + ni*(j-jl + nj*(k-kl))) = a(m,v,k,j,i);
          });
        // if neighbor is at coarser level, load data from coarse_u0
        // Note in this case, sbuf[n].index values refer to coarse_u0
        } else {
          Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il,iu+1),[&](const int i)
          {
            sbuf[n].data(m, v, i-il + ni*(j-jl + nj*(k-kl))) = ca(m,v,k,j,i);
          });
        }
      }
    });

  }); // end par_for_outer
  }

  // Send boundary buffer to neighboring MeshBlocks using MPI

  {int &my_rank = global_variable::my_rank;
  auto &nghbr = pmy_pack->pmb->nghbr;
  auto &rbuf = recv_buf;
#if MPI_PARALLEL_ENABLED
  auto &sbuf = send_buf;
#endif

  for (int m=0; m<nmb; ++m) {
    for (int n=0; n<nnghbr; ++n) {
      if (nghbr.h_view(m,n).gid >= 0) {  // not a physical boundary
        // compute indices of destination MeshBlock and Neighbor 
        int nn = nghbr.h_view(m,n).destn;
        // if MeshBlocks are same rank, data already copied into receive buffer above
        // So simply set communication status tag as received.
        if (nghbr.h_view(m,n).rank == my_rank) {
          int mm = nghbr.h_view(m,n).gid - pmy_pack->gids;
          rbuf[nn].bcomm_stat(mm) = BoundaryCommStatus::received;

#if MPI_PARALLEL_ENABLED
        // Send boundary data using MPI
        } else {
          // create tag using local ID and buffer index of *receiving* MeshBlock
          int lid = nghbr.h_view(m,n).gid -
                    pmy_pack->pmesh->gidslist[nghbr.h_view(m,n).rank];
          int tag = CreateMPITag(lid, nn, key);
          auto send_data = Kokkos::subview(sbuf[n].data, m, Kokkos::ALL, Kokkos::ALL);
          void* send_ptr = send_data.data();
          int ierr = MPI_Isend(send_ptr, send_data.size(), MPI_ATHENA_REAL,
            nghbr.h_view(m,n).rank, tag, MPI_COMM_WORLD, &(sbuf[n].comm_req[m]));
#endif
        }
      }
    }
  }}

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
// \!fn void RecvBuffers()
// \brief Unpack boundary buffers

TaskStatus BValCC::RecvAndUnpackCC(DvceArray5D<Real> &a, DvceArray5D<Real> &ca)
{
  // create local references for variables in kernel
  int nmb = pmy_pack->pmb->nmb;
  // TODO: following only works when all MBs have the same number of neighbors
  int nnghbr = pmy_pack->pmb->nnghbr;

  bool bflag = false;
  {auto &nghbr = pmy_pack->pmb->nghbr;
  auto &rbuf = recv_buf;

#if MPI_PARALLEL_ENABLED
  // probe MPI communications.  This is a bit of black magic that seems to promote
  // communications to top of stack and gets them to complete more quickly
  int test;
  MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &test, MPI_STATUS_IGNORE);
#endif

  //----- STEP 1: check that recv boundary buffer communications have all completed
  for (int m=0; m<nmb; ++m) {
    for (int n=0; n<nnghbr; ++n) {
      if (nghbr.h_view(m,n).gid >= 0) { // ID != -1, so not a physical boundary
        if (nghbr.h_view(m,n).rank == global_variable::my_rank) {
          if (rbuf[n].bcomm_stat(m) == BoundaryCommStatus::waiting) bflag = true;
#if MPI_PARALLEL_ENABLED
        } else {
          MPI_Test(&(rbuf[n].comm_req[m]), &test, MPI_STATUS_IGNORE);
          if (static_cast<bool>(test)) {
            rbuf[n].bcomm_stat(m) = BoundaryCommStatus::received;
          } else {
            bflag = true;
          }
#endif
        }
      }
    }
  }}

  // exit if recv boundary buffer communications have not completed
  if (bflag) {return TaskStatus::incomplete;}

  //----- STEP 2: buffers have all completed, so unpack
  {int nvar = a.extent_int(1);  // TODO: 2nd index from L of input array must be NVAR
  int nmnv = nmb*nnghbr*nvar;
  auto &nghbr = pmy_pack->pmb->nghbr;
  auto &mblev = pmy_pack->pmb->mblev;
  auto &rbuf = recv_buf;

  // Outer loop over (# of MeshBlocks)*(# of buffers)*(# of variables)
  Kokkos::TeamPolicy<> policy(DevExeSpace(), nmnv, Kokkos::AUTO);
  Kokkos::parallel_for("RecvBuff", policy, KOKKOS_LAMBDA(TeamMember_t tmember)
  {
    const int m = (tmember.league_rank())/(nnghbr*nvar);
    const int n = (tmember.league_rank() - m*(nnghbr*nvar))/nvar;
    const int v = (tmember.league_rank() - m*(nnghbr*nvar) - n*nvar);
    // if neighbor is at same or finer level, use indices for u0
    // indices same for all variables, stored in (0,i) component
    int il, iu, jl, ju, kl, ku;
    if (nghbr.d_view(m,n).lev >= mblev.d_view(m)) {
      il = rbuf[n].index.d_view(0);
      iu = rbuf[n].index.d_view(1);
      jl = rbuf[n].index.d_view(2);
      ju = rbuf[n].index.d_view(3);
      kl = rbuf[n].index.d_view(4);
      ku = rbuf[n].index.d_view(5);
    // else if neighbor is at coarser level, use indices for coarse_u0
    } else {
      il = rbuf[n].cindex.d_view(0);
      iu = rbuf[n].cindex.d_view(1);
      jl = rbuf[n].cindex.d_view(2);
      ju = rbuf[n].cindex.d_view(3);
      kl = rbuf[n].cindex.d_view(4);
      ku = rbuf[n].cindex.d_view(5);
    }
    const int ni = iu - il + 1;
    const int nj = ju - jl + 1;
    const int nk = ku - kl + 1;
    const int nkj  = nk*nj;

    // Middle loop over k,j
    Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nkj), [&](const int idx)
    {
      int k = idx / nj;
      int j = (idx - k * nj) + jl;
      k += kl;
         
      // if neighbor is at same or finer level, load data directly into u0
      if (nghbr.d_view(m,n).lev >= mblev.d_view(m)) {
        Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il,iu+1), [&](const int i)
        {
          a(m,v,k,j,i) = rbuf[n].data(m,v,i-il + ni*(j-jl + nj*(k-kl)));
        });

      // if neighbor is at coarser level, load data into coarse_u0 (and prolongate below)
      // Note in this case, rbuf[n].index values refer to coarse_u0
      } else {
        Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il,iu+1), [&](const int i)
        {
          ca(m,v,k,j,i) = rbuf[n].data(m,v,i-il + ni*(j-jl + nj*(k-kl)));
        });
      }

    });
  }); // end par_for_outer
  }

  //----- STEP 3: Prolongate conserved variables when neighbor at coarser level
  // Code here is based on MeshRefinement::RestrictCellCenteredValues() in
  // mesh_refinement.cpp in Athena++
  if (pmy_pack->pmesh->multilevel) {
    int nvar = a.extent_int(1);  // TODO: 2nd index from L of input array must be NVAR
    int nmnv = nmb*nnghbr*nvar;
    auto &nghbr = pmy_pack->pmb->nghbr;
    auto &mblev = pmy_pack->pmb->mblev;
    auto &rbuf = recv_buf;

    // Prolongation for 1D problem
    if (pmy_pack->pmesh->one_d) {
      auto &cis = pmy_pack->pcoord->mbdata.cindcs.is;
      auto &cie = pmy_pack->pcoord->mbdata.cindcs.ie;
      auto &is = pmy_pack->pcoord->mbdata.indcs.is;
      auto &ie = pmy_pack->pcoord->mbdata.indcs.ie;
      auto &js = pmy_pack->pcoord->mbdata.indcs.js;
      auto &ks = pmy_pack->pcoord->mbdata.indcs.ks;
      // Outer loop over (# of MeshBlocks)*(# of buffers)*(# of variables)
      Kokkos::TeamPolicy<> policy(DevExeSpace(), nmnv, Kokkos::AUTO);
      Kokkos::parallel_for("RecvBuff", policy, KOKKOS_LAMBDA(TeamMember_t tmember)
      {
        const int m = (tmember.league_rank())/(nnghbr*nvar);
        const int n = (tmember.league_rank() - m*(nnghbr*nvar))/nvar;
        const int v = (tmember.league_rank() - m*(nnghbr*nvar) - n*nvar);
        // indices in loop below are on normal mesh
        const int il = rbuf[n].index.d_view(0);
        const int iu = rbuf[n].index.d_view(1);

        // if neighbor is at coarser level, prolongate.  Otherwise do nothing
        if (nghbr.d_view(m,n).lev < mblev.d_view(m)) {
          Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il,iu+1),[&](const int i)
          {
            // calculate indices of coarse array elements
            int coari;
            if (il < is) {
              coari = (i-is-1)/2 + cis;
            } else {
              coari = (i-ie+1)/2 + cie;
            }

            // calculate 1D gradient using the min-mod limiter
            Real dl = ca(m,v,ks,js,coari  ) - ca(m,v,ks,js,coari-1);
            Real dr = ca(m,v,ks,js,coari+1) - ca(m,v,ks,js,coari  );
            Real dc = 0.5*(SIGN(dl) + SIGN(dr))*fmin(fabs(dl), fabs(dr));

            // interpolate to the finer grid
              a(m,v,ks,js,i) = ca(m,v,ks,js,coari);
/***
            if (i%2 == 0) {
              a(m,v,ks,js,i) = ca(m,v,ks,js,coari) + 0.25*dc ;
            } else {
              a(m,v,ks,js,i) = ca(m,v,ks,js,coari) - 0.25*dc ;
            }
***/
          });
        }
      }); // end par_for_outer
    }
  }

/******
  std::cout << std::endl << "u0 data" << std::endl;
  for (int m=0; m<nmb; ++m) {
    auto &js = pmy_pack->pcoord->mbdata.indcs.js;
    auto &ks = pmy_pack->pcoord->mbdata.indcs.ks;
    std::cout << "Block = " << m << "  level = " << pmy_pack->pmb->mblev.h_view(m) << std::endl;
    for (int i=pmy_pack->pcoord->mbdata.indcs.is-2; i<=pmy_pack->pcoord->mbdata.indcs.ie+2; ++i) {
      std::cout << "i=" << i << "  d=" << a(m,0,ks,js,i) << std::endl;
    }
  }

  std::cout << std::endl << "coarse u0 data" << std::endl;
  for (int m=0; m<nmb; ++m) {
    auto &js = pmy_pack->pcoord->mbdata.cindcs.js;
    auto &ks = pmy_pack->pcoord->mbdata.cindcs.ks;
    std::cout << "Block = " << m << "  level = " << pmy_pack->pmb->mblev.h_view(m) << std::endl;
    for (int i=pmy_pack->pcoord->mbdata.cindcs.is-2; i<=pmy_pack->pcoord->mbdata.cindcs.ie+2; ++i) {
      std::cout << "i=" << i << "  d=" << ca(m,0,ks,js,i) << std::endl;
    }
  }

*****/

  return TaskStatus::complete;
}
