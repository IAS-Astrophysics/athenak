#ifndef BVALS_BVALS_HPP_
#define BVALS_BVALS_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file bvals_cc.hpp
//  \brief defines classes for handling boundary values for cell-centered, face-centered,
//  variables

// identifiers for all 6 faces of a MeshBlock
enum BoundaryFace {undef=-1, inner_x1, outer_x1, inner_x2, outer_x2, inner_x3, outer_x3};

// identifiers for boundary conditions
enum class BoundaryFlag {undef=-1, block, reflect, outflow, user, periodic};

// identifiers for status of MPI boundary communications
enum class BoundaryCommStatus {undef=-1, waiting, sent, received};

// integer constants to specify variables communicated in MPI calls (maximum of 16 set by
// number of bits used to encode ID in CreateMPItag function in src/utils)
enum VariablesID {FluidCons_ID, BField_ID};

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/coordinates.hpp"
#include "tasklist/task_list.hpp"

// function ptr for user-defined boundary functions enrolled in problem generator 
//using HydroBoundaryFnPtr = void (*)(int m, Mesh* pm, Hydro* phyd, DvceArray5D<Real> &u);
//using MHDBoundaryFnPtr = void (*)(int m, Mesh* pm, MHD* pmhd, DvceArray5D<Real> &u);

//----------------------------------------------------------------------------------------
//! \struct BufferIndcs
//! \brief Cell indices for pack/unpack cells into BvalBuffer

struct BufferIndcs
{
  int bis,bie,bjs,bje,bks,bke;  // start/end buffer ("b") indices in each dir
  int ndat;                     // number of data elements
};

//----------------------------------------------------------------------------------------
//! \struct BValBufferCC
//  \brief index ranges, storage, and flags for data passed at boundaries

struct BValBufferCC
{
  BufferIndcs sindcs; // indices for pack/unpack when dest/src at same level ("s")
  BufferIndcs cindcs; // indices for pack/unpack when dest/src at coarser level ("c")
  BufferIndcs findcs; // indices for pack/unpack when dest/src at finer level ("f")
  BufferIndcs pindcs; // indices for prolongation ("p") (only used for receives)
  DvceArray3D<Real> data;
  HostArray1D<BoundaryCommStatus> bcomm_stat;
#if MPI_PARALLEL_ENABLED
  // only accessed from host, so can use STL vector
  std::vector<MPI_Request> comm_req;
#endif
  // function to allocate memory for buffer data
  void AllocateDataView(int nmb, int nvar) {
    int ndat = sindcs.ndat;  // TODO: this may over-estimate memory needed for some buffs
    Kokkos::realloc(data, nmb, nvar, ndat);
  }
};

//----------------------------------------------------------------------------------------
//! \struct BValBufferFC
//  \brief index ranges, storage, and flags for data passed at boundaries

struct BValBufferFC
{
  DualArray2D<int> index;
  DvceArray3D<Real> data;
  HostArray1D<BoundaryCommStatus> bcomm_stat;
#if MPI_PARALLEL_ENABLED
  // only accessed from host, so can use STL vector
  std::vector<MPI_Request> comm_req;
#endif

  // function to initialize indices/data for FC variables
  void InitIndices(int nmb, int i0, int i1, int j0, int j1, int k0, int k1,
                            int i2, int i3, int j2, int j3, int k2, int k3,
                            int i4, int i5, int j4, int j5, int k4, int k5) {
    index.h_view(0,0)=i0;
    index.h_view(0,1)=i1;
    index.h_view(0,2)=j0;
    index.h_view(0,3)=j1;
    index.h_view(0,4)=k0;
    index.h_view(0,5)=k1;
    int n0 = (i1-i0+1)*(j1-j0+1)*(k1-k0+1);

    index.h_view(1,0)=i2;
    index.h_view(1,1)=i3;
    index.h_view(1,2)=j2;
    index.h_view(1,3)=j3;
    index.h_view(1,4)=k2;
    index.h_view(1,5)=k3;
    int n1 = (i3-i2+1)*(j3-j2+1)*(k3-k2+1);

    index.h_view(2,0)=i4;
    index.h_view(2,1)=i5;
    index.h_view(2,2)=j4;
    index.h_view(2,3)=j5;
    index.h_view(2,4)=k4;
    index.h_view(2,5)=k5;
    int n2 = (i5-i4+1)*(j5-j4+1)*(k5-k4+1);
    int ndata = std::max(std::max(n0,n1),n2);
    Kokkos::realloc(data, nmb, 3, ndata);
  }
};

// Forward declarations
class MeshBlockPack;

//----------------------------------------------------------------------------------------
//! \class BValCC
//  \brief Lightweight boundary values class for cell-centered variables
//  TODO: extend for AMR boundaries

class BValCC {
 public:
  BValCC(MeshBlockPack *ppack, ParameterInput *pin);
  ~BValCC() {};  // only default destructor needed

  // data
  BValBufferCC send_buf[56], recv_buf[56];

  //functions
  void InitSendIndices(BValBufferCC &buf, int ox1, int ox2, int ox3, int f1, int f2);
  void InitRecvIndices(BValBufferCC &buf, int ox1, int ox2, int ox3, int f1, int f2);
  void AllocateBuffersCC(const int nvar);
  TaskStatus PackAndSendCC(DvceArray5D<Real> &a, DvceArray5D<Real> &c, int key);
  TaskStatus RecvAndUnpackCC(DvceArray5D<Real> &a, DvceArray5D<Real> &c);

 private:
  MeshBlockPack *pmy_pack;
};

//----------------------------------------------------------------------------------------
//! \class BValFC
//  \brief Lightweight boundary values class for face-centered vector fields
//  TODO: extend for AMR boundaries

class BValFC {
 public:
  BValFC(MeshBlockPack *ppack, ParameterInput *pin);
  ~BValFC();

  // data
  BValBufferFC send_buf[56], recv_buf[56];

  //functions
  void AllocateBuffersFC();
  TaskStatus PackAndSendFC(DvceFaceFld4D<Real> &b, int key);
  TaskStatus RecvAndUnpackFC(DvceFaceFld4D<Real> &b);

 private:
  MeshBlockPack *pmy_pack;
};

#endif // BVALS_BVALS_HPP_
