//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file pbb->cc_sendreceive.cpp
//  \brief implementation of functions in BoundaryValues class

#include <cstdlib>
#include <iostream>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "bvals/bvals.hpp"
#include "mesh/mesh.hpp"

//----------------------------------------------------------------------------------------
// \!fn void BoundaryValues::SendCellCenteredVariables()
// \brief Pack boundary buffers for cell-centered variables, and send to neighbors
// This routine always packs ALL the buffers, but they are only sent (via MPI) or copied
// for periodic or block boundaries

TaskStatus BoundaryValues::SendCellCenteredVariables(AthenaArray4D<Real> &a, int nvar,
                                                     std::string key)
{
  MeshBlock *pmb = pmesh_->FindMeshBlock(my_mbgid_);
  int ng = pmb->mb_cells.ng;
  int is = pmb->mb_cells.is, ie = pmb->mb_cells.ie;
  int js = pmb->mb_cells.js, je = pmb->mb_cells.je;
  int ks = pmb->mb_cells.ks, ke = pmb->mb_cells.ke;
  int nx1 = pmb->mb_cells.nx1;
  int nx2 = pmb->mb_cells.nx2;
  int nx3 = pmb->mb_cells.nx3;

  // Find the physics module containing the send buffer, using bbuf_ptr map and [key]
  BoundaryBuffer *pbb = pmb->pbvals->bbuf_ptr[key];

  // create local references for variables in kernel
  auto &nx3gt1 = pmesh_->nx3gt1;
  auto &nx2gt1 = pmesh_->nx2gt1;
  auto &send_x1face = pbb->send_x1face;
  auto &send_x2face = pbb->send_x2face;
  auto &send_x3face = pbb->send_x3face;
  auto &send_x1x2ed = pbb->send_x1x2ed;
  auto &send_x3x1ed = pbb->send_x3x1ed;
  auto &send_x2x3ed = pbb->send_x2x3ed;
  auto &send_corner = pbb->send_corner;

  // load buffers, NO AMR
  int scr_level = 1;
  par_for_outer("SendCC", pmb->exe_space, 0, scr_level, 0, (nvar-1), ks, ke, js, je,
    KOKKOS_LAMBDA(TeamMember_t member, const int n, const int k, const int j)
    {
      // 2D slice in bottom two cells in k-direction
      if (nx3gt1 && k<(ks+ng)) {
        if (nx2gt1 && j<(js+ng)) {
          par_for_inner(member, is, ie, [&](const int i)
          {
            send_x2x3ed(0,n,k-ks ,j-js ,i-is ) = a(n,k,j,i);
            send_x3face(0,n,k-ks ,j-js ,i-is ) = a(n,k,j,i);
            send_x2face(0,n,k-ks ,j-js ,i-is ) = a(n,k,j,i);
            if (i<(is+ng)) {
              send_corner(0,n,k-ks ,j-js ,i-is ) = a(n,k,j,i);
              send_x3x1ed(0,n,k-ks ,j-js ,i-is ) = a(n,k,j,i);
              send_x1x2ed(0,n,k-ks ,j-js ,i-is ) = a(n,k,j,i);
              send_x1face(0,n,k-ks ,j-js ,i-is ) = a(n,k,j,i);
            }
            if (i>=nx1) {
              send_corner(1,n,k-ks ,j-js ,i-nx1) = a(n,k,j,i);
              send_x3x1ed(1,n,k-ks ,j-js ,i-nx1) = a(n,k,j,i);
              send_x1x2ed(1,n,k-ks ,j-js ,i-nx1) = a(n,k,j,i);
              send_x1face(1,n,k-ks ,j-js ,i-nx1) = a(n,k,j,i);
            }
          });
        } else if (nx2gt1 && j>=nx2) {
          par_for_inner(member, is, ie, [&](const int i)
          {
            send_x3face(0,n,k-ks ,j-js ,i-is ) = a(n,k,j,i);
            send_x2x3ed(1,n,k-ks ,j-nx2,i-is ) = a(n,k,j,i);
            send_x2face(1,n,k-ks ,j-nx2,i-is ) = a(n,k,j,i);
            if (i<(is+ng)) {
              send_x3x1ed(0,n,k-ks ,j-js ,i-is ) = a(n,k,j,i);
              send_corner(2,n,k-ks ,j-nx2,i-is ) = a(n,k,j,i);
              send_x1face(0,n,k-ks ,j-js ,i-is ) = a(n,k,j,i);
              send_x1x2ed(2,n,k-ks ,j-nx2,i-is ) = a(n,k,j,i);
            }
            if (i>=nx1) {
              send_x3x1ed(1,n,k-ks ,j-js ,i-nx1) = a(n,k,j,i);
              send_corner(3,n,k-ks ,j-nx2,i-nx1) = a(n,k,j,i);
              send_x1face(1,n,k-ks ,j-js ,i-nx1) = a(n,k,j,i);
              send_x1x2ed(3,n,k-ks ,j-nx2,i-nx1) = a(n,k,j,i);
            }
          });
        } else {
          par_for_inner(member, is, ie, [&](const int i)
          {
            send_x3face(0,n,k-ks ,j-js ,i-is ) = a(n,k,j,i);
            if (i<(is+ng)) {
              send_x3x1ed(0,n,k-ks ,j-js ,i-is ) = a(n,k,j,i);
              send_x1face(0,n,k-ks ,j-js ,i-is ) = a(n,k,j,i);
            }
            if (i>=nx1) {
              send_x3x1ed(1,n,k-ks ,j-js ,i-nx1) = a(n,k,j,i);
              send_x1face(1,n,k-ks ,j-js ,i-nx1) = a(n,k,j,i);
            }
          });
        }

      // 2D slice in top two cells in k-direction
      } else if (nx3gt1 && k>=nx3) {
        if (nx2gt1 && j<(js+ng)) {
          par_for_inner(member, is, ie, [&](const int i)
          {
            send_x2face(0,n,k-ks ,j-js ,i-is ) = a(n,k,j,i);
            send_x2x3ed(2,n,k-nx3,j-js ,i-is ) = a(n,k,j,i);
            send_x3face(1,n,k-nx3,j-js ,i-is ) = a(n,k,j,i);
            if (i<(is+ng)) {
              send_x1x2ed(0,n,k-ks ,j-js ,i-is ) = a(n,k,j,i);
              send_x1face(0,n,k-ks ,j-js ,i-is ) = a(n,k,j,i);
              send_corner(4,n,k-nx3,j-js ,i-is ) = a(n,k,j,i);
              send_x3x1ed(2,n,k-nx3,j-js ,i-is ) = a(n,k,j,i);
            }
            if (i>=nx1) {
              send_x1x2ed(1,n,k-ks ,j-js ,i-nx1) = a(n,k,j,i);
              send_x1face(1,n,k-ks ,j-js ,i-nx1) = a(n,k,j,i);
              send_corner(5,n,k-nx3,j-js ,i-nx1) = a(n,k,j,i);
              send_x3x1ed(3,n,k-nx3,j-js ,i-nx1) = a(n,k,j,i);
            }
          });
        } else if (nx2gt1 && j>=nx2) {
          par_for_inner(member, is, ie, [&](const int i)
          {
            send_x2face(1,n,k-ks ,j-nx2,i-is ) = a(n,k,j,i);
            send_x3face(1,n,k-nx3,j-js ,i-is ) = a(n,k,j,i);
            send_x2x3ed(3,n,k-nx3,j-nx2,i-is ) = a(n,k,j,i);
            if (i<(is+ng)) {
              send_x1face(0,n,k-ks ,j-js ,i-is ) = a(n,k,j,i);
              send_x1x2ed(2,n,k-ks ,j-nx2,i-is ) = a(n,k,j,i);
              send_x3x1ed(2,n,k-nx3,j-js ,i-is ) = a(n,k,j,i);
              send_corner(6,n,k-nx3,j-nx2,i-is ) = a(n,k,j,i);
            }
            if (i>=nx1) {
              send_x1face(1,n,k-ks ,j-js ,i-nx1) = a(n,k,j,i);
              send_x1x2ed(3,n,k-ks ,j-nx2,i-nx1) = a(n,k,j,i);
              send_x3x1ed(3,n,k-nx3,j-js ,i-nx1) = a(n,k,j,i);
              send_corner(7,n,k-nx3,j-nx2,i-nx1) = a(n,k,j,i);
            }
          });
        } else {
          par_for_inner(member, is, ie, [&](const int i)
          {
            send_x3face(1,n,k-nx3,j-js ,i-is ) = a(n,k,j,i);
            if (i<(is+ng)) {
              send_x1face(0,n,k-ks ,j-js ,i-is ) = a(n,k,j,i);
              send_x3x1ed(2,n,k-nx3,j-js ,i-is ) = a(n,k,j,i);
            }
            if (i>=nx1) {
              send_x1face(1,n,k-ks ,j-js ,i-nx1) = a(n,k,j,i);
              send_x3x1ed(3,n,k-nx3,j-js ,i-nx1) = a(n,k,j,i);
            }
          });
        }

      // 2D slice in middle of grid
      } else {
        if (nx2gt1 && j<(js+ng)) {
          par_for_inner(member, is, ie, [&](const int i)
          {
            send_x2face(0,n,k-ks ,j-js ,i-is ) = a(n,k,j,i);
            if (i<(is+ng)) {
              send_x1x2ed(0,n,k-ks ,j-js ,i-is ) = a(n,k,j,i);
              send_x1face(0,n,k-ks ,j-js ,i-is ) = a(n,k,j,i);
            }
            if (i>=nx1) {
              send_x1x2ed(1,n,k-ks ,j-js ,i-nx1) = a(n,k,j,i);
              send_x1face(1,n,k-ks ,j-js ,i-nx1) = a(n,k,j,i);
            }
          });
        } else if (nx2gt1 && j>=nx2) {
          par_for_inner(member, is, ie, [&](const int i)
          {
            send_x2face(1,n,k-ks ,j-nx2,i-is ) = a(n,k,j,i);
            if (i<(is+ng)) {
              send_x1face(0,n,k-ks ,j-js ,i-is ) = a(n,k,j,i);
              send_x1x2ed(2,n,k-ks ,j-nx2,i-is ) = a(n,k,j,i);
            }
            if (i>=nx1) {
              send_x1face(1,n,k-ks ,j-js ,i-nx1) = a(n,k,j,i);
              send_x1x2ed(3,n,k-ks ,j-nx2,i-nx1) = a(n,k,j,i);
            }
          });
        } else {
          par_for_inner(member, is, ie, [&](const int i)
          {
            if (i<(is+ng)) {
              send_x1face(0,n,k-ks ,j-js,i-is ) = a(n,k,j,i);
            }
            if (i>=nx1) {
              send_x1face(1,n,k-ks ,j-js,i-nx1) = a(n,k,j,i);
            }
          });
        }
      }
    }
  ); // end par_for_outer

  // Now, for block or periodic boundaries, send boundary buffer to neighboring MeshBlocks
  // using MPI, or if neighbor is on same MPI rank, use Kokkos::deep_copy of subviews
  // Note (1) physics module containing the recv buffer is found using bbuf_ptr map and
  // the [key], (2) in general recv_buffer[n] maps to recv_buffer[X-n] (where X is number
  // of buffers of each type), and (3) BoundaryStatus flag must be sent for copies 
  // TODO add MPI sends

  // copy x1 faces
  for (int n=0; n<2; ++n) {
    if (nblocks_x1face[1-n].ngid >= 0) {
//    if (bndry_flag[n]==BoundaryFlag::block || bndry_flag[n]==BoundaryFlag::periodic) {
      MeshBlock *pdest_mb = pmesh_->FindMeshBlock(nblocks_x1face[1-n].ngid);
      auto sendbuf = Kokkos::subview(pbb->send_x1face,
                    (1-n),Kokkos::ALL(),Kokkos::ALL(),Kokkos::ALL(),Kokkos::ALL());
      auto recvbuf = Kokkos::subview(pdest_mb->pbvals->bbuf_ptr[key]->recv_x1face,
                    (n  ),Kokkos::ALL(),Kokkos::ALL(),Kokkos::ALL(),Kokkos::ALL());
      Kokkos::deep_copy(pmb->exe_space, recvbuf, sendbuf);
      pdest_mb->pbvals->bbuf_ptr[key]->bstat_x1face[n] = BoundaryStatus::completed;
    }
  }
  if (!(pmesh_->nx2gt1)) return TaskStatus::complete;

  // copy x2 faces and x1x2 edges
  for (int n=0; n<2; ++n) {
    if (nblocks_x2face[1-n].ngid >= 0) {
//    if (bndry_flag[n+2]==BoundaryFlag::block || bndry_flag[n+2]==BoundaryFlag::periodic) {
      MeshBlock *pdest_mb = pmesh_->FindMeshBlock(nblocks_x2face[1-n].ngid);
      auto sendbuf = Kokkos::subview(pbb->send_x2face,
                    (1-n),Kokkos::ALL(),Kokkos::ALL(),Kokkos::ALL(),Kokkos::ALL());
      auto recvbuf = Kokkos::subview(pdest_mb->pbvals->bbuf_ptr[key]->recv_x2face,
                    (n  ),Kokkos::ALL(),Kokkos::ALL(),Kokkos::ALL(),Kokkos::ALL());
      Kokkos::deep_copy(pmb->exe_space, recvbuf, sendbuf);
      pdest_mb->pbvals->bbuf_ptr[key]->bstat_x2face[n] = BoundaryStatus::completed;
    }
  }
  for (int n=0; n<4; ++n) {
    if (nblocks_x1x2ed[3-n].ngid >= 0) {
//    if (bndry_flag[(n/2)+2]==BoundaryFlag::block ||
//        bndry_flag[(n/2)+2]==BoundaryFlag::periodic) {
      MeshBlock *pdest_mb = pmesh_->FindMeshBlock(nblocks_x1x2ed[3-n].ngid);
      auto sendbuf = Kokkos::subview(pbb->send_x1x2ed,
                    (3-n),Kokkos::ALL(),Kokkos::ALL(),Kokkos::ALL(),Kokkos::ALL());
      auto recvbuf = Kokkos::subview(pdest_mb->pbvals->bbuf_ptr[key]->recv_x1x2ed,
                    (n  ),Kokkos::ALL(),Kokkos::ALL(),Kokkos::ALL(),Kokkos::ALL());
      Kokkos::deep_copy(pmb->exe_space, recvbuf, sendbuf);
      pdest_mb->pbvals->bbuf_ptr[key]->bstat_x1x2ed[n] = BoundaryStatus::completed;
    }
  }
  if (!(pmesh_->nx3gt1)) return TaskStatus::complete;
  
  // copy x3 faces, x3x1 and x2x3 edges, and corners
  for (int n=0; n<2; ++n) {
    if (nblocks_x3face[1-n].ngid >= 0) {
//    if (bndry_flag[n+4]==BoundaryFlag::block || bndry_flag[n+4]==BoundaryFlag::periodic) {
      MeshBlock *pdest_mb = pmesh_->FindMeshBlock(nblocks_x3face[1-n].ngid);
      auto sendbuf = Kokkos::subview(pbb->send_x3face,
                    (1-n),Kokkos::ALL(),Kokkos::ALL(),Kokkos::ALL(),Kokkos::ALL());
      auto recvbuf = Kokkos::subview(pdest_mb->pbvals->bbuf_ptr[key]->recv_x3face,
                    (n  ),Kokkos::ALL(),Kokkos::ALL(),Kokkos::ALL(),Kokkos::ALL());
      Kokkos::deep_copy(pmb->exe_space, recvbuf, sendbuf);
      pdest_mb->pbvals->bbuf_ptr[key]->bstat_x3face[n] = BoundaryStatus::completed;
    }
  }
  for (int n=0; n<4; ++n) {
    if (nblocks_x3x1ed[3-n].ngid >= 0) {
//    if (bndry_flag[(n/2)+4]==BoundaryFlag::block ||
//        bndry_flag[(n/2)+4]==BoundaryFlag::periodic) {
      MeshBlock *pdest_mb = pmesh_->FindMeshBlock(nblocks_x3x1ed[3-n].ngid);
      auto sendbuf = Kokkos::subview(pbb->send_x3x1ed,
                    (3-n),Kokkos::ALL(),Kokkos::ALL(),Kokkos::ALL(),Kokkos::ALL());
      auto recvbuf = Kokkos::subview(pdest_mb->pbvals->bbuf_ptr[key]->recv_x3x1ed,
                    (n  ),Kokkos::ALL(),Kokkos::ALL(),Kokkos::ALL(),Kokkos::ALL());
      Kokkos::deep_copy(pmb->exe_space, recvbuf, sendbuf);
      pdest_mb->pbvals->bbuf_ptr[key]->bstat_x3x1ed[n] = BoundaryStatus::completed;
    }
  }
  for (int n=0; n<4; ++n) {
    if (nblocks_x2x3ed[3-n].ngid >= 0) {
//    if (bndry_flag[(n/2)+4]==BoundaryFlag::block ||
//        bndry_flag[(n/2)+4]==BoundaryFlag::periodic) {
      MeshBlock *pdest_mb = pmesh_->FindMeshBlock(nblocks_x2x3ed[3-n].ngid);
      auto sendbuf = Kokkos::subview(pbb->send_x2x3ed,
                    (3-n),Kokkos::ALL(),Kokkos::ALL(),Kokkos::ALL(),Kokkos::ALL());
      auto recvbuf = Kokkos::subview(pdest_mb->pbvals->bbuf_ptr[key]->recv_x2x3ed,
                    (n  ),Kokkos::ALL(),Kokkos::ALL(),Kokkos::ALL(),Kokkos::ALL());
      Kokkos::deep_copy(pmb->exe_space, recvbuf, sendbuf);
      pdest_mb->pbvals->bbuf_ptr[key]->bstat_x2x3ed[n] = BoundaryStatus::completed;
    }
  }
  for (int n=0; n<8; ++n) {
    if (nblocks_corner[7-n].ngid >= 0) {
//    if (bndry_flag[(n/4)+4]==BoundaryFlag::block ||
//        bndry_flag[(n/4)+4]==BoundaryFlag::periodic) {
      MeshBlock *pdest_mb = pmesh_->FindMeshBlock(nblocks_corner[7-n].ngid);
      auto sendbuf = Kokkos::subview(pbb->send_corner,
                    (7-n),Kokkos::ALL(),Kokkos::ALL(),Kokkos::ALL(),Kokkos::ALL());
      auto recvbuf = Kokkos::subview(pdest_mb->pbvals->bbuf_ptr[key]->recv_corner,
                    (n  ),Kokkos::ALL(),Kokkos::ALL(),Kokkos::ALL(),Kokkos::ALL());
      Kokkos::deep_copy(pmb->exe_space, recvbuf, sendbuf);
      pdest_mb->pbvals->bbuf_ptr[key]->bstat_corner[n] = BoundaryStatus::completed;
    }
  }

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
// \!fn void BoundaryValues::RecvCellCenteredVariables()
// \brief Unpack boundary buffers for cell-centered variables.

TaskStatus BoundaryValues::RecvCellCenteredVariables(AthenaArray4D<Real> &a, int nvar,
                                                     std::string key)
{
  MeshBlock *pmb = pmesh_->FindMeshBlock(my_mbgid_);

  int ng = pmb->mb_cells.ng;
  int is = pmb->mb_cells.is; int ie = pmb->mb_cells.ie;
  int js = pmb->mb_cells.js; int je = pmb->mb_cells.je;
  int ks = pmb->mb_cells.ks; int ke = pmb->mb_cells.ke;
  int ncells1 = pmb->mb_cells.nx1 + 2*ng;
  int ncells2 = (pmb->mb_cells.nx2 > 1)? (pmb->mb_cells.nx2 + 2*ng) : 1;
  int ncells3 = (pmb->mb_cells.nx3 > 1)? (pmb->mb_cells.nx3 + 2*ng) : 1;

  // Find the physics module containing the recv buffer, using bbuf_ptr map and [key]
  BoundaryBuffer *pbb = pmb->pbvals->bbuf_ptr[key];

  // check that recv boundary buffers have all completed, exit if not.
  for (int n=0; n<2; ++n) {
    if (pbb->bstat_x1face[n]==BoundaryStatus::waiting) { return TaskStatus::incomplete;}
  }
  if (pmesh_->nx2gt1) {
    for (int n=0; n<2; ++n) {
      if (pbb->bstat_x2face[n]==BoundaryStatus::waiting) {return TaskStatus::incomplete;}
    }
    for (int n=0; n<4; ++n) {
      if (pbb->bstat_x1x2ed[n]==BoundaryStatus::waiting) {return TaskStatus::incomplete;}
    }
  }
  if (pmesh_->nx3gt1) {
    for (int n=0; n<2; ++n) {
      if (pbb->bstat_x3face[n]==BoundaryStatus::waiting) {return TaskStatus::incomplete;}
    }
    for (int n=0; n<4; ++n) {
      if (pbb->bstat_x3x1ed[n]==BoundaryStatus::waiting) {return TaskStatus::incomplete;}
      if (pbb->bstat_x2x3ed[n]==BoundaryStatus::waiting) {return TaskStatus::incomplete;}
    }
    for (int n=0; n<8; ++n) {
      if (pbb->bstat_corner[n]==BoundaryStatus::waiting) {return TaskStatus::incomplete;}
    }
  }
  
  // buffers have all completed, so unpack (THIS VERSION NO AMR)
  // create local references for variables in kernel
  auto &nx3gt1 = pmesh_->nx3gt1;
  auto &nx2gt1 = pmesh_->nx2gt1;
  auto &recv_x1face = pbb->recv_x1face;
  auto &recv_x2face = pbb->recv_x2face;
  auto &recv_x3face = pbb->recv_x3face;
  auto &recv_x1x2ed = pbb->recv_x1x2ed;
  auto &recv_x3x1ed = pbb->recv_x3x1ed;
  auto &recv_x2x3ed = pbb->recv_x2x3ed;
  auto &recv_corner = pbb->recv_corner;

  int scr_level = 1;
  par_for_outer("RecvCC", pmb->exe_space, 0, scr_level, 0, (nvar-1), 0, (ncells3-1),
                 0, (ncells2-1),
    KOKKOS_LAMBDA(TeamMember_t member, const int n, const int k, const int j)
    {
      // 2D slice in bottom two cells in k-direction
      if (nx3gt1 && k<ks) {
        if (nx2gt1 && j<js) {
          par_for_inner(member, 0, (ncells1-1), [&](const int i)
          {
            if (i<is) {
              a(n,k,j,i) = recv_corner(0,n,k,j,i);
            } else if (i>ie) {
              a(n,k,j,i) = recv_corner(1,n,k,j,i-(ie+1));
            } else {
              a(n,k,j,i) = recv_x2x3ed(0,n,k,j,i-is);
            }
          });
        } else if (nx2gt1 && j>je) {
          par_for_inner(member, 0, (ncells1-1), [&](const int i)
          {
            if (i<is) {
              a(n,k,j,i) = recv_corner(2,n,k,j-je-1,i);
            } else if (i>ie) {
              a(n,k,j,i) = recv_corner(3,n,k,j-je-1,i-(ie+1));
            } else {
              a(n,k,j,i) = recv_x2x3ed(1,n,k,j-je-1,i-is);
            }
          });
        } else {
          par_for_inner(member, 0, (ncells1-1), [&](const int i)
          {
            if (i<is) {
              a(n,k,j,i) = recv_x3x1ed(0,n,k,j-js,i);
            } else if (i>ie) {
              a(n,k,j,i) = recv_x3x1ed(1,n,k,j-js,i-(ie+1));
            } else {
              a(n,k,j,i) = recv_x3face(0,n,k,j-js,i-is);
            }
          });
        }

      // 2D slice in top two cells in k-direction
      } else if (nx3gt1 && k>ke) {
        if (nx2gt1 && j<js) {
          par_for_inner(member, 0, (ncells1-1), [&](const int i)
          {
            if (i<is) {
              a(n,k,j,i) = recv_corner(4,n,k-ke-1,j,i);
            } else if (i>ie) {
              a(n,k,j,i) = recv_corner(5,n,k-ke-1,j,i-(ie+1));
            } else {
              a(n,k,j,i) = recv_x2x3ed(2,n,k-ke-1,j,i-is);
            }
          });
        } else if (nx2gt1 && j>je) {
          par_for_inner(member, 0, (ncells1-1), [&](const int i)
          {
            if (i<is) {
              a(n,k,j,i) = recv_corner(6,n,k-ke-1,j-je-1,i);
            } else if (i>ie) {
              a(n,k,j,i) = recv_corner(7,n,k-ke-1,j-je-1,i-(ie+1));
            } else {
              a(n,k,j,i) = recv_x2x3ed(3,n,k-ke-1,j-je-1,i-is);
            }
          });
        } else {
          par_for_inner(member, 0, (ncells1-1), [&](const int i)
          {
            if (i<is) {
              a(n,k,j,i) = recv_x3x1ed(2,n,k-ke-1,j-js,i);
            } else if (i>ie) {
              a(n,k,j,i) = recv_x3x1ed(3,n,k-ke-1,j-js,i-(ie+1));
            } else {
              a(n,k,j,i) = recv_x3face(1,n,k-ke-1,j-js,i-is);
            }
          });
        }

      // 2D slice in middle of grid
      } else {
        if (nx2gt1 && j<js) {
          par_for_inner(member, 0, (ncells1-1), [&](const int i)
          {
            if (i<is) {
              a(n,k,j,i) = recv_x1x2ed(0,n,k-ks,j,i);
            } else if (i>ie) {
              a(n,k,j,i) = recv_x1x2ed(1,n,k-ks,j,i-(ie+1));
            } else {
              a(n,k,j,i) = recv_x2face(0,n,k-ks,j,i-is);
            }
          });
        } else if (nx2gt1 && j>je) {
          par_for_inner(member, 0, (ncells1-1), [&](const int i)
          {
            if (i<is) {
              a(n,k,j,i) = recv_x1x2ed(2,n,k-ks,j-je-1,i);
            } else if (i>ie) {
              a(n,k,j,i) = recv_x1x2ed(3,n,k-ks,j-je-1,i-(ie+1));
            } else {
              a(n,k,j,i) = recv_x2face(1,n,k-ks,j-je-1,i-is);
            }
          });
        } else {
          par_for_inner(member, 0, (ncells1-1), [&](const int i)
          {
            if (i<is) {
              a(n,k,j,i) = recv_x1face(0,n,k-ks,j-js,i);
            } else if (i>ie) {
              a(n,k,j,i) = recv_x1face(1,n,k-ks,j-js,i-(ie+1));
            }
          });
        }
      }
    }
  );  // end par_for_outer

  return TaskStatus::complete;
}
