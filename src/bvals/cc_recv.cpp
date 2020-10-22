//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file cc_recv.cpp
//  \brief implements receives of cell-centered variables

#include <cstdlib>
#include <iostream>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "bvals/bvals.hpp"
#include "mesh/mesh.hpp"

//----------------------------------------------------------------------------------------
// \!fn void BoundaryValues::RecvCellCenteredVars()
// \brief Unpack boundary buffers for cell-centered variables.

TaskStatus BoundaryValues::RecvCellCenteredVars(AthenaArray4D<Real> &a, int nvar, int key)
{
  // Find the physics module containing the recv buffer, using bbuf_ptr map and [key]
  MeshBlock *pmb = pmesh_->FindMeshBlock(my_mbgid_);
  BBuffer *pbb = pmb->pbvals->bbuf_ptr[key];

#if MPI_PARALLEL_ENABLED
  // probe MPI communications.  This is a bit of black magic that seems to promote
  // communications to top of stack and gets them to complete more quickly
  int test;
  MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &test, MPI_STATUS_IGNORE);
#endif

  bool bflag = false;

  // check that recv boundary buffer communications have all completed
  // x1faces
  for (int n=0; n<2; ++n) {
    if (nghbr_x1face[n].rank == global_variable::my_rank) {
      if (pbb->bstat_x1face[n] == BoundaryRecvStatus::waiting) bflag = true;
#if MPI_PARALLEL_ENABLED
    } else {
      MPI_Test(&(pbb->recv_rq_x1face[n]), &test, MPI_STATUS_IGNORE);
      if (static_cast<bool>(test)) {
        pbb->bstat_x1face[n] = BoundaryRecvStatus::completed;
      } else {
        bflag = true;
      }
#endif
    }
  }

  // x2faces and x1x2 edges
  if (pmesh_->nx2gt1) {
    for (int n=0; n<2; ++n) {
      if (nghbr_x2face[n].rank == global_variable::my_rank) {
        if (pbb->bstat_x2face[n] == BoundaryRecvStatus::waiting) bflag = true;
#if MPI_PARALLEL_ENABLED
      } else {
        MPI_Test(&(pbb->recv_rq_x2face[n]), &test, MPI_STATUS_IGNORE);
        if (static_cast<bool>(test)) {
          pbb->bstat_x2face[n] = BoundaryRecvStatus::completed;
        } else {
          bflag = true;
        }
#endif
      }
    }
    for (int n=0; n<4; ++n) {
      if (nghbr_x1x2ed[n].rank == global_variable::my_rank) {
        if (pbb->bstat_x1x2ed[n] == BoundaryRecvStatus::waiting) bflag = true;
#if MPI_PARALLEL_ENABLED
      } else {
        MPI_Test(&(pbb->recv_rq_x1x2ed[n]), &test, MPI_STATUS_IGNORE);
        if (static_cast<bool>(test)) {
          pbb->bstat_x1x2ed[n] = BoundaryRecvStatus::completed;
        } else {
          bflag = true;
        }
#endif
      }
    }
  }

  // x3faces, x3x1 and x2x3 edges, and corners
  if (pmesh_->nx3gt1) {
    for (int n=0; n<2; ++n) {
      if (nghbr_x3face[n].rank == global_variable::my_rank) {
        if (pbb->bstat_x3face[n] == BoundaryRecvStatus::waiting) bflag = true;
#if MPI_PARALLEL_ENABLED
      } else {
        MPI_Test(&(pbb->recv_rq_x3face[n]), &test, MPI_STATUS_IGNORE);
        if (static_cast<bool>(test)) {
          pbb->bstat_x3face[n] = BoundaryRecvStatus::completed;
        } else {
          bflag = true;
        }
#endif
      }
    }
    for (int n=0; n<4; ++n) {
      if (nghbr_x3x1ed[n].rank == global_variable::my_rank) {
        if (pbb->bstat_x3x1ed[n] == BoundaryRecvStatus::waiting) bflag = true;
#if MPI_PARALLEL_ENABLED
      } else {
        MPI_Test(&(pbb->recv_rq_x3x1ed[n]), &test, MPI_STATUS_IGNORE);
        if (static_cast<bool>(test)) {
          pbb->bstat_x3x1ed[n] = BoundaryRecvStatus::completed;
        } else {
          bflag = true;
        }
#endif
      }
    }
    for (int n=0; n<4; ++n) {
      if (nghbr_x2x3ed[n].rank == global_variable::my_rank) {
        if (pbb->bstat_x2x3ed[n] == BoundaryRecvStatus::waiting) bflag = true;
#if MPI_PARALLEL_ENABLED
      } else {
        MPI_Test(&(pbb->recv_rq_x2x3ed[n]), &test, MPI_STATUS_IGNORE);
        if (static_cast<bool>(test)) {
          pbb->bstat_x2x3ed[n] = BoundaryRecvStatus::completed;
        } else {
          bflag = true;
        }
#endif
      }
    }
    for (int n=0; n<8; ++n) {
      if (nghbr_corner[n].rank == global_variable::my_rank) {
        if (pbb->bstat_corner[n] == BoundaryRecvStatus::waiting) bflag = true;
#if MPI_PARALLEL_ENABLED
      } else {
        MPI_Test(&(pbb->recv_rq_corner[n]), &test, MPI_STATUS_IGNORE);
        if (static_cast<bool>(test)) {
          pbb->bstat_corner[n] = BoundaryRecvStatus::completed;
        } else {
          bflag = true;
        }
#endif
      }
    }
  }

  // exit if recv boundary buffer communications have not completed
  if (bflag) {return TaskStatus::incomplete;}
  
  // buffers have all completed, so unpack (THIS VERSION NO AMR)
  int ng = pmb->mb_cells.ng;
  int is = pmb->mb_cells.is; int ie = pmb->mb_cells.ie;
  int js = pmb->mb_cells.js; int je = pmb->mb_cells.je;
  int ks = pmb->mb_cells.ks; int ke = pmb->mb_cells.ke;
  int ncells1 = pmb->mb_cells.nx1 + 2*ng;
  int ncells2 = (pmb->mb_cells.nx2 > 1)? (pmb->mb_cells.nx2 + 2*ng) : 1;
  int ncells3 = (pmb->mb_cells.nx3 > 1)? (pmb->mb_cells.nx3 + 2*ng) : 1;

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
