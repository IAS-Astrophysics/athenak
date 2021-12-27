#ifndef BVALS_BVALS_HPP_
#define BVALS_BVALS_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file bvals_cc.hpp
//! \brief defines classes for handling boundary values for all types of variables.
//! Currently methods for cell-centered and face-centered fields implemented.

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

//----------------------------------------------------------------------------------------
//! \struct BufferIndcs
//! \brief indices for range of cells packed/unpacked into boundary buffers

struct BufferIndcs
{
  int bis,bie,bjs,bje,bks,bke;  // start/end buffer ("b") indices in each dir
  int ndat;                     // number of data elements
};

//----------------------------------------------------------------------------------------
//! \struct BValBufferCC
//! \brief container for index ranges, storage, and flags for boundary buffers for CC data

struct BValBufferCC
{
  BufferIndcs sindcs; // indices for pack/unpack when dest/src at same level ("s")
  BufferIndcs cindcs; // indices for pack/unpack when dest/src at coarser level ("c")
  BufferIndcs findcs; // indices for pack/unpack when dest/src at finer level ("f")
  BufferIndcs pindcs; // indices for prolongation ("p") (only used for receives)
  DvceArray3D<Real> data;
  HostArray1D<BoundaryCommStatus> bcomm_stat;
#if MPI_PARALLEL_ENABLED
  std::vector<MPI_Request> comm_req;   // only accessed from host, so can use STL vector
#endif
  // function to allocate memory for buffer data
  void AllocateDataView(int nmb, int nvar) {
    int nmax = std::max( std::max(sindcs.ndat,cindcs.ndat), findcs.ndat);
    Kokkos::realloc(data, nmb, nvar, nmax);
  }
};

//----------------------------------------------------------------------------------------
//! \struct BValBufferFC
//! \brief container for index ranges, storage, and flags for boundary buffers for FC data

struct BValBufferFC
{
  // Following fixed-length arrays store indices for each component of vector field (which
  // can be different!);  [0,1,2] --> [x1f, x2f, x3f] 
  BufferIndcs sindcs[3]; // indices for pack/unpack when dest/src at same level ("s")
  BufferIndcs cindcs[3]; // indices for pack/unpack when dest/src at coarser level ("c")
  BufferIndcs findcs[3]; // indices for pack/unpack when dest/src at finer level ("f")
  BufferIndcs pindcs[3]; // indices for prolongation ("p") (only used for receives)

  DvceArray3D<Real> data;
  HostArray1D<BoundaryCommStatus> bcomm_stat;
#if MPI_PARALLEL_ENABLED
  std::vector<MPI_Request> comm_req;   // only accessed from host, so can use STL vector
#endif
  // function to allocate memory for buffer data
  void AllocateDataView(int nmb) {
    int smax = std::max( std::max(sindcs[0].ndat, sindcs[1].ndat), sindcs[2].ndat );
    int cmax = std::max( std::max(cindcs[0].ndat, cindcs[1].ndat), cindcs[2].ndat );
    int fmax = std::max( std::max(findcs[0].ndat, findcs[1].ndat), findcs[2].ndat );
    int nmax = std::max( std::max(smax,cmax), fmax);
    Kokkos::realloc(data, nmb, 3, nmax);
  }
};

// Forward declarations
class MeshBlockPack;

//----------------------------------------------------------------------------------------
//! \class BValCC
//  \brief Lightweight class for boundary values of cell-centered variables

class BValCC
{
public:
  BValCC(MeshBlockPack *ppack, ParameterInput *pin);

  // data for all 56 buffers in most general 3D case. Not all elements used in most cases.
  // However each BValBufferCC requires only 176 bytes, so the convenience of fixed array
  // sizes and index values for array elements outweighs cost of extra memory. 
  BValBufferCC send_buf[56], recv_buf[56];

  //functions
  void InitSendIndices(BValBufferCC &buf, int o1, int o2, int o3, int f1, int f2);
  void InitRecvIndices(BValBufferCC &buf, int o1, int o2, int o3, int f1, int f2);
  void AllocateBuffersCC(const int nvar);
  TaskStatus PackAndSendCC(DvceArray5D<Real> &a, DvceArray5D<Real> &c, int key);
  TaskStatus RecvAndUnpackCC(DvceArray5D<Real> &a, DvceArray5D<Real> &c);
  void ProlongCC(DvceArray5D<Real> &a, DvceArray5D<Real> &c);

private:
  MeshBlockPack* pmy_pack;
};

//----------------------------------------------------------------------------------------
//! \class BValFC
//  \brief Lightweight class for boundary values of face-centered vector fields

class BValFC
{
public:
  BValFC(MeshBlockPack *ppack, ParameterInput *pin);

  // Like BValCC case, not all 56 elements are used in most cases.  Now each BvalBufferFC
  // requires 400 bytes, but still economical to use fixed-length arrays.
  BValBufferFC send_buf[56], recv_buf[56];

  //functions
  void InitSendIndices(BValBufferFC &buf, int o1, int o2, int o3, int f1, int f2);
  void InitRecvIndices(BValBufferFC &buf, int o1, int o2, int o3, int f1, int f2);
  void AllocateBuffersFC();
  TaskStatus PackAndSendFC(DvceFaceFld4D<Real> &b, DvceFaceFld4D<Real> &c, int key);
  TaskStatus RecvAndUnpackFC(DvceFaceFld4D<Real> &b, DvceFaceFld4D<Real> &c);
  void ProlongFC(DvceFaceFld4D<Real> &b, DvceFaceFld4D<Real> &c);

private:
  MeshBlockPack* pmy_pack;
};

#endif // BVALS_BVALS_HPP_
