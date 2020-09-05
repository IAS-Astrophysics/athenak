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

#include <map>
#include "athena.hpp"
#include "athena_arrays.hpp"
#include "parameter_input.hpp"
#include "tasklist/task_list.hpp"
#include "boundary_flag.hpp"

//----------------------------------------------------------------------------------------
//! \struct NeighborBlock
//  \brief Information about neighboring MeshBlocks

struct NeighborBlock
{
  int ngid;
  int nrank;
  int nlevel;
  NeighborBlock() : ngid(-1), nrank(-1), nlevel(-1) {}  // set default values
};

//----------------------------------------------------------------------------------------
//! \struct BoundaryBuffer
//  \brief Stores send/receive buffers and BoundaryStatus flags

struct BoundaryBuffer
{
  AthenaArray<Real> cc_send_x1face, cc_send_x2face, cc_send_x3face;
  AthenaArray<Real> cc_send_x1x2ed, cc_send_x3x1ed, cc_send_x2x3ed;
  AthenaArray<Real> cc_send_corner;
  AthenaArray<Real> cc_recv_x1face, cc_recv_x2face, cc_recv_x3face;
  AthenaArray<Real> cc_recv_x1x2ed, cc_recv_x3x1ed, cc_recv_x2x3ed;
  AthenaArray<Real> cc_recv_corner;
  BoundaryStatus bstat_x1face[2];
  BoundaryStatus bstat_x2face[2];
  BoundaryStatus bstat_x3face[2];
  BoundaryStatus bstat_x1x2ed[4];
  BoundaryStatus bstat_x3x1ed[4];
  BoundaryStatus bstat_x2x3ed[4];
  BoundaryStatus bstat_corner[8];
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
  BoundaryFlag bndry_flags[6]; // enums specifying BCs at all 6 faces of this MeshBlock

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
  TaskStatus SendCellCenteredVariables(AthenaArray<Real> &a, int nvar, std::string key);
  TaskStatus ReceiveCellCenteredVariables(AthenaArray<Real> &a,int nvar,std::string key);

  TaskStatus ApplyPhysicalBCs(Driver* pd, int stage);
  void ReflectInnerX1();
  void ReflectOuterX1();
  void ReflectInnerX2();
  void ReflectOuterX2();
  void ReflectInnerX3();
  void ReflectOuterX3();

 private:
  Mesh *pmesh_;
  int my_mbgid_;

};

#endif // BVALS_BVALS_HPP_

