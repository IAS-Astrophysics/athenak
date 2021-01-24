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

// integer constants to specify physics modules (maximum of 16 set by number of bits used
// to encode ID in BoundaryValues::CreateMPItag)
enum PhysicsID {Hydro_ID, MHD_ID};

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "tasklist/task_list.hpp"

//----------------------------------------------------------------------------------------
//! \struct BoundaryBuffer
//  \brief index ranges, storage, and flags for data passed at boundaries

struct BoundaryBuffer
{
  DualArray1D<int> index;
  DvceArray3D<Real> data;
  HostArray1D<BoundaryCommStatus> bcomm_stat;
#if MPI_PARALLEL_ENABLED
  // only accessed from host, so can use STL vector
  std::vector<MPI_Request> comm_req;
#endif
  // constructor
  BoundaryBuffer(){};
  void InitIndices(int nmb, int nvar, int i0, int i1, int j0, int j1, int k0, int k1) {
    index.h_view(0)=i0;
    index.h_view(1)=i1;
    index.h_view(2)=j0;
    index.h_view(3)=j1;
    index.h_view(4)=k0;
    index.h_view(5)=k1;
    Kokkos::realloc(data, nmb, nvar, (i1-i0+1)*(j1-j0+1)*(k1-k0+1));
  }
};

// Forward declarations
class MeshBlockPack;

//----------------------------------------------------------------------------------------
//! \class BoundaryBase
//  \brief

class BoundaryValues {
 public:
  BoundaryValues(MeshBlockPack *ppack, ParameterInput *pin);
  ~BoundaryValues();

  // data
  BoundaryBuffer send_buf[26], recv_buf[26];

  //functions
  void AllocateBuffersCC(const int nvar);
  int CreateMPITag(int lid, int buff_id, int phys_id);
  TaskStatus SendBuffers(DvceArray5D<Real> &a, int key);
  TaskStatus RecvBuffers(DvceArray5D<Real> &a);

 private:
  MeshBlockPack *pmy_pack;
};

#endif // BVALS_BVALS_HPP_
