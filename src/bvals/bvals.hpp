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
enum class BoundaryFlag {undef=-1,block, reflect, inflow, outflow, diode, user, periodic};

#include <algorithm>
#include <vector>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/coordinates.hpp"
#include "tasklist/task_list.hpp"

//----------------------------------------------------------------------------------------
//! \fn int CreateBvals_MPI_Tag(int lid, int bufid)
//! \brief calculate an MPI tag for boundary buffer communications.  Note maximum size of
//! lid that can be encoded is set by (NUM_BITS_LID) macro.
//! The convention in Athena++ is lid and bufid are both for the *receiving* process.
//! The MPI standard requires signed int tag, with MPI_TAG_UB>=2^15-1 = 32,767 (inclusive)
static int CreateBvals_MPI_Tag(int lid, int bufid) {
  return (bufid << (NUM_BITS_LID)) | lid;
}

//----------------------------------------------------------------------------------------
//! \struct BufferIndcs
//! \brief indices for range of cells packed/unpacked into boundary buffers

struct BufferIndcs {
  int bis,bie,bjs,bje,bks,bke;  // start/end buffer ("b") indices in each dir
  BufferIndcs() :
    bis(0), bie(0), bjs(0), bje(0), bks(0), bke(0) {}
};

//----------------------------------------------------------------------------------------
//! \struct BoundaryBuffer
//! \brief container for index ranges, storage, and flags for boundary buffers

struct BoundaryBuffer {
  // fixed-length-3 arrays used to store indices of each buffer for cell-centered vars, or
  // each component of a face-centered vector field ([0,1,2] --> [x1f, x2f, x3f]). For
  // cell-centered variables only first [0] component of index arrays are needed.
  BufferIndcs isame[3];  // indices for pack/unpack when dest/src at same level
  BufferIndcs icoar[3];  // indices for pack/unpack when dest/src at coarser level
  BufferIndcs ifine[3];  // indices for pack/unpack when dest/src at finer level
  BufferIndcs iprol[3];  // indices for prolongation (only used for receives)
  BufferIndcs iflux_same[3];  // indices for pack/unpack for flux correction
  BufferIndcs iflux_coar[3];  // indices for pack/unpack for flux correction

  // Maximum number of data elements (bie-bis+1) across 3 components of above
  int isame_ndat, icoar_ndat, ifine_ndat, iflxs_ndat, iflxc_ndat;

  // 2D Views that store buffer data on device, dimensioned (nmb, ndata)
  DvceArray2D<Real> vars, flux;

#if MPI_PARALLEL_ENABLED
  // Using STL vector causes problems with some GPU compilers, even those these vectors
  // are only ever accessed on host, so just use plain C array
  MPI_Request *vars_req, *flux_req;
#endif

  // function to allocate memory for buffers for variables and their fluxes
  // Must only be called after BufferIndcs above are initialized
  void AllocateBuffers(int nmb, int nvars) {
    int nmax = std::max(isame_ndat, std::max(icoar_ndat, ifine_ndat) );
    Kokkos::realloc(vars, nmb, (nvars*nmax));
    nmax = std::max(iflxs_ndat, iflxc_ndat);
    Kokkos::realloc(flux, nmb, (nvars*nmax));
  }
};

// Forward declarations
class MeshBlockPack;

//----------------------------------------------------------------------------------------
//! \class BoundaryValues
//  \brief Abstract base class for boundary values for different kinds of variables

class BoundaryValues {
 public:
  BoundaryValues(MeshBlockPack *ppack, ParameterInput *pin);
  ~BoundaryValues();

  // data for all 56 buffers in most general 3D case. Not all elements used in most cases.
  // However each BoundaryBuffer is lightweight, so the convenience of fixed array
  // sizes and index values for array elements outweighs cost of extra memory.
  BoundaryBuffer send_buf[56], recv_buf[56];

  // constant inflow states at each face, initialized in problem generator
  DualArray2D<Real> u_in, b_in, i_in;

#if MPI_PARALLEL_ENABLED
  // unique MPI communicators for variables and fluxes
  MPI_Comm vars_comm, flux_comm;
#endif

  //functions
  virtual void InitSendIndices(BoundaryBuffer &buf, int x, int y, int z, int a, int b)=0;
  virtual void InitRecvIndices(BoundaryBuffer &buf, int x, int y, int z, int a, int b)=0;
  void InitializeBuffers(const int nvar);

  TaskStatus InitRecv(const int nvar);
  virtual TaskStatus InitFluxRecv(const int nvar)=0;
  TaskStatus ClearRecv();
  TaskStatus ClearSend();
  TaskStatus ClearFluxRecv();
  TaskStatus ClearFluxSend();

  // BCs associated with various physics modules
  static void HydroBCs(MeshBlockPack *pp, DualArray2D<Real> uin, DvceArray5D<Real> u0);
  static void BFieldBCs(MeshBlockPack *pp, DualArray2D<Real> bin, DvceFaceFld4D<Real> b0);
  static void RadiationBCs(MeshBlockPack *pp,DualArray2D<Real> iin,DvceArray5D<Real> i0);

 protected:
  MeshBlockPack* pmy_pack;
};

//----------------------------------------------------------------------------------------
//! \class BoundaryValuesCC
//  \brief boundary values for cell-centered variables

class BoundaryValuesCC : public BoundaryValues {
 public:
  BoundaryValuesCC(MeshBlockPack *ppack, ParameterInput *pin);

  //functions
  void InitSendIndices(BoundaryBuffer &buf, int o1, int o2,int o3,int f1,int f2) override;
  void InitRecvIndices(BoundaryBuffer &buf, int o1, int o2,int o3,int f1,int f2) override;
  TaskStatus InitFluxRecv(const int nvar) override;

  TaskStatus PackAndSendCC(DvceArray5D<Real> &a, DvceArray5D<Real> &ca);
  TaskStatus RecvAndUnpackCC(DvceArray5D<Real> &a, DvceArray5D<Real> &ca);
  void ProlongateCC(DvceArray5D<Real> &a, DvceArray5D<Real> &ca);

  TaskStatus PackAndSendFluxCC(DvceFaceFld5D<Real> &flx);
  TaskStatus RecvAndUnpackFluxCC(DvceFaceFld5D<Real> &flx);
};

//----------------------------------------------------------------------------------------
//! \class BoundaryValuesFC
//  \brief boundary values for face-centered vector fields

class BoundaryValuesFC : public BoundaryValues {
 public:
  BoundaryValuesFC(MeshBlockPack *ppack, ParameterInput *pin);

  //functions
  void InitSendIndices(BoundaryBuffer &buf, int o1, int o2,int o3,int f1,int f2) override;
  void InitRecvIndices(BoundaryBuffer &buf, int o1, int o2,int o3,int f1,int f2) override;
  TaskStatus InitFluxRecv(const int nvar) override;

  TaskStatus PackAndSendFC(DvceFaceFld4D<Real> &b, DvceFaceFld4D<Real> &cb);
  TaskStatus RecvAndUnpackFC(DvceFaceFld4D<Real> &b, DvceFaceFld4D<Real> &cb);
  void ProlongateFC(DvceFaceFld4D<Real> &b, DvceFaceFld4D<Real> &cb);

  TaskStatus PackAndSendFluxFC(DvceEdgeFld4D<Real> &flx);
  TaskStatus RecvAndUnpackFluxFC(DvceEdgeFld4D<Real> &flx);
  void SumBoundaryFluxes(DvceEdgeFld4D<Real> &flx, const bool same_level,
                         DvceArray2D<int> &nflx);
  void ZeroFluxesAtBoundaryWithFiner(DvceEdgeFld4D<Real> &flx, DvceArray2D<int> &nflx);
  void AverageBoundaryFluxes(DvceEdgeFld4D<Real> &flx, DvceArray2D<int> &nflx);
};

#endif // BVALS_BVALS_HPP_
