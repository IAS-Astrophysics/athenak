#ifndef BVALS_BVALS_HPP_
#define BVALS_BVALS_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file bvals.hpp
//  \brief defines structures and provides prototypes of functions used for boundary
//  communication.  These are not part of any class, but are general utility functions.

// identifiers for all 6 faces of a MeshBlock
enum BoundaryFace {undef=-1, inner_x1, outer_x1, inner_x2, outer_x2, inner_x3, outer_x3};

// identifiers for boundary conditions
enum class BoundaryFlag {undef=-1, block, reflect, outflow, user, periodic};

// identifiers for status of MPI boundary communications
enum class BoundaryCommStatus {undef=-1, waiting, sent, received};

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "tasklist/task_list.hpp"

//----------------------------------------------------------------------------------------
//! \struct NeighborBlock
//  \brief Information about neighboring MeshBlocks

struct NeighborBlock
{
  int gid;
  int level;
  int rank;
  int dn;     // difference between index of this and target neighbor for comms
  // constructor
  NeighborBlock(int id, int lev, int rnk, int deltan) :
    gid(id), level(lev), rank(rnk), dn(deltan) {}
};

//----------------------------------------------------------------------------------------
//! \struct BoundaryBuffer
//  \brief index ranges, storage, and flags for data passed at boundaries

struct BoundaryBuffer
{
  AthenaArray1D<int> index;
  AthenaArray2D<Real> data;
  BoundaryCommStatus bcomm_stat;
#if MPI_PARALLEL_ENABLED
  MPI_Request comm_req;
#endif
  // constructor
  BoundaryBuffer(int nvar, int i0, int i1, int j0, int j1, int k0, int k1) :
    index("bbuff_idx", 6),
    data("bbuff", nvar, ((i1-i0+1)*(j1-j0+1)*(k1-k0+1)))
  {
    index(0) = i0;
    index(1) = i1;
    index(2) = j0;
    index(3) = j1;
    index(4) = k0;
    index(5) = k1;
    bcomm_stat = BoundaryCommStatus::undef;
  }
};

// Forward declarations
class MeshBlock;

//----------------------------------------------------------------------------------------
// boundary function prototypes

void AllocateBuffersCCVars(const int nvar, const RegionCells ncells,
    std::vector<BoundaryBuffer> &send_buf, std::vector<BoundaryBuffer> &recv_buf);
int CreateMPITag(int lid, int buff_id, int phys_id);
TaskStatus SendBuffers(AthenaArray5D<Real> &a, 
  std::vector<std::vector<BoundaryBuffer>> send_buf,
  std::vector<std::vector<BoundaryBuffer>> recv_buf, std::vector<MeshBlock> mblocks);
TaskStatus RecvBuffers(AthenaArray5D<Real> &a,
  std::vector<std::vector<BoundaryBuffer>> send_buf,
  std::vector<std::vector<BoundaryBuffer>> recv_buf, std::vector<MeshBlock> mblocks);

#endif // BVALS_BVALS_HPP_
