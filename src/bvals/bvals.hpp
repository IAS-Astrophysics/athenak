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

// identifiers for status of MPI boundary communications
enum class BoundaryStatus {undef=-1, waiting, completed};

// identifiers for boundary conditions
enum class BoundaryFlag {undef=-1, block, reflect, outflow, user, periodic};

#include <map>
#include "athena.hpp"
#include "parameter_input.hpp"
#include "tasklist/task_list.hpp"

//----------------------------------------------------------------------------------------
//! \struct NeighborBlock
//  \brief Information about neighboring MeshBlocks

struct NeighborBlock
{
  int gid;
  int rank;
  int level;
  NeighborBlock() : gid(-1), rank(-1), level(-1) {}  // set default values
};

//----------------------------------------------------------------------------------------
//! \struct BoundaryBuffer
//  \brief Stores send/receive buffers and BoundaryStatus flags

struct BoundaryBuffer
{
  // face, edge, and corner send buffers
  AthenaArray5D<Real> send_x1face, send_x2face, send_x3face;
  AthenaArray5D<Real> send_x1x2ed, send_x3x1ed, send_x2x3ed;
  AthenaArray5D<Real> send_corner;
  // face, edge, and corner recv buffers
  AthenaArray5D<Real> recv_x1face, recv_x2face, recv_x3face;
  AthenaArray5D<Real> recv_x1x2ed, recv_x3x1ed, recv_x2x3ed;
  AthenaArray5D<Real> recv_corner;
  // face, edge, and corner status flags
  BoundaryStatus bstat_x1face[2];
  BoundaryStatus bstat_x2face[2];
  BoundaryStatus bstat_x3face[2];
  BoundaryStatus bstat_x1x2ed[4];
  BoundaryStatus bstat_x3x1ed[4];
  BoundaryStatus bstat_x2x3ed[4];
  BoundaryStatus bstat_corner[8];
#if MPI_PARALLEL_ENABLED
  MPI_Request send_rq_x1face[2], recv_rq_x1face[2];
  MPI_Request send_rq_x2face[2], recv_rq_x2face[2];
  MPI_Request send_rq_x3face[2], recv_rq_x3face[2];
  MPI_Request send_rq_x1x2ed[4], recv_rq_x1x2ed[4];
  MPI_Request send_rq_x3x1ed[4], recv_rq_x3x1ed[4];
  MPI_Request send_rq_x2x3ed[4], recv_rq_x2x3ed[4];
  MPI_Request send_rq_corner[8], recv_rq_corner[8];
#endif

  // constructor (calls View constructor with appropriate labels)
  BoundaryBuffer() :
    send_x1face("x1face_send_buf",1,1,1,1,1),
    send_x2face("x2face_send_buf",1,1,1,1,1),
    send_x3face("x3face_send_buf",1,1,1,1,1),
    send_x1x2ed("x1x2edge_send_buf",1,1,1,1,1),
    send_x3x1ed("x3x1edge_send_buf",1,1,1,1,1),
    send_x2x3ed("x2x3edge_send_buf",1,1,1,1,1),
    send_corner("corner_send_buf",1,1,1,1,1),
    recv_x1face("x1face_recv_buf",1,1,1,1,1),
    recv_x2face("x2face_recv_buf",1,1,1,1,1),
    recv_x3face("x3face_recv_buf",1,1,1,1,1),
    recv_x1x2ed("x1x2edge_recv_buf",1,1,1,1,1),
    recv_x3x1ed("x3x1edge_recv_buf",1,1,1,1,1),
    recv_x2x3ed("x2x3edge_recv_buf",1,1,1,1,1),
    recv_corner("corner_recv_buf",1,1,1,1,1) {}
};

// Forward delcarations
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

  // map to store pointers to BoundaryBuffers for different physics
  // elements are added in mesh/interface_physics after physics modules cons in MBs
  std::map<std::string,BoundaryBuffer*> bbuf_ptr;

  NeighborBlock nblocks_x1face[2];
  NeighborBlock nblocks_x2face[2]; 
  NeighborBlock nblocks_x3face[2];
  NeighborBlock nblocks_x1x2ed[4];
  NeighborBlock nblocks_x3x1ed[4];
  NeighborBlock nblocks_x2x3ed[4];
  NeighborBlock nblocks_corner[8];

  // functions
  void AllocateBuffers(BoundaryBuffer &bbuf, const int maxv);
  TaskStatus SendCellCenteredVariables(AthenaArray4D<Real> &a, int nvar, std::string key);
  TaskStatus RecvCellCenteredVariables(AthenaArray4D<Real> &a, int nvar, std::string key);

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

