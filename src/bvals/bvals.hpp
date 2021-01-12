#ifndef BVALS_BVALS_HPP_
#define BVALS_BVALS_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file bvals.hpp
//  \brief defines BoundaryBase, BoundaryValues classes used for setting BCs on all data

// identifiers for all 6 faces of a MeshBlock
enum BoundaryFace {undef=-1, inner_x1, outer_x1, inner_x2, outer_x2, inner_x3, outer_x3};

// identifiers for boundary conditions
enum class BoundaryFlag {undef=-1, block, reflect, outflow, user, periodic};

// identifiers for status of MPI boundary communications
enum class BoundaryCommStatus {undef=-1, waiting, sent, received};

#include "athena.hpp"
#include "parameter_input.hpp"
#include "tasklist/task_list.hpp"
#include "mesh/mesh.hpp"

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
  int il, iu, jl, ju, kl, ku;
  AthenaArray2D<Real> data;
  BoundaryCommStatus bcomm_stat;
#if MPI_PARALLEL_ENABLED
  MPI_Request comm_req;
#endif
  // constructor
  BoundaryBuffer(int nvar, int i0, int i1, int j0, int j1, int k0, int k1) :
    il(i0), iu(i1), jl(j0), ju(j1), kl(k0), ku(k1),
    data("bbuff", nvar, ((i1-i0+1)*(j1-j0+1)*(k1-k0+1))),
    bcomm_stat(BoundaryCommStatus::undef) {}
};

// Forward declarations
class Mesh;

//----------------------------------------------------------------------------------------
//! \class BoundaryBase
//  \brief

class BoundaryValues {
 public:
  BoundaryValues(Mesh* pm, ParameterInput *pin, int gid, BoundaryFlag *bcs);
  ~BoundaryValues();

  // data
  BoundaryFlag bndry_flag[6]; // enums specifying BCs at all 6 faces of this MeshBlock

  std::vector<NeighborBlock> nghbr;

  // functions
  void AllocateBuffersCC(const RegionCells ncells, const int nvar,
    std::vector<BoundaryBuffer> &send_buf, std::vector<BoundaryBuffer> &recv_buf);
  int CreateMPItag(int lid, int buff_id, int phys_id);
  TaskStatus SendBuffers(AthenaArray4D<Real> &a);
  TaskStatus RecvBuffers(AthenaArray4D<Real> &a);

  TaskStatus ApplyPhysicalBCs(Driver* pd, int stage);
  void ReflectInnerX1();
  void ReflectOuterX1();
  void ReflectInnerX2();
  void ReflectOuterX2();
  void ReflectInnerX3();
  void ReflectOuterX3();
  void OutflowInnerX1();
  void OutflowOuterX1();
  void OutflowInnerX2();
  void OutflowOuterX2();
  void OutflowInnerX3();
  void OutflowOuterX3();

 private:
  Mesh *pmesh_;
  int my_mbgid_;

};

#endif // BVALS_BVALS_HPP_
