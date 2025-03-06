#ifndef SHEARING_BOX_SHEARING_BOX_HPP_
#define SHEARING_BOX_SHEARING_BOX_HPP_
//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file shearing_box.hpp
//! \brief definitions for classes that implement both orbital adfection and shearing box.
//! Both OrbitalAdvection and ShearingBox are abstract base classes that are used to
//! define derived classes for CC and FC variables.

#include "athena.hpp"
#include "parameter_input.hpp"
#include "tasklist/task_list.hpp"
#include "bvals/bvals.hpp"

//----------------------------------------------------------------------------------------
//! \struct ShearingBoxTaskIDs
//  \brief container to hold TaskIDs of all shearing_box tasks

struct ShearingBoxTaskIDs {
  TaskID irecv;
  TaskID copyu;
  TaskID flux;
  TaskID sendf;
  TaskID recvf;
  TaskID csend;
  TaskID crecv;
};

//----------------------------------------------------------------------------------------
//! \struct ShearingBoxBoundaryBuffer
//! \brief container for storing boundary buffers. Used by both the orbital advection and
//! shearing box methods for both CC and FC variables.
//! Basically a much simplified version of the MeshBoundaryBuffer struct.

struct ShearingBoxBoundaryBuffer {
  // Views that store buffer data and fluxes on device
  DvceArray5D<Real> vars, flux;
#if MPI_PARALLEL_ENABLED
  // vectors of length (number of MBs) to hold MPI requests
  // Using STL vector causes problems with some GPU compilers, so just use plain C array
  MPI_Request *vars_req, *flux_req;
#endif
};

//----------------------------------------------------------------------------------------
//! \class OrbitalAdvection
//  \brief Abstract base class for orbital advection of CC and FC variables

class OrbitalAdvection {
 public:
  OrbitalAdvection(MeshBlockPack *ppack, ParameterInput *pin);
  ~OrbitalAdvection();

  // data
  int maxjshift;            // maximum integer shift of any cell in orbital advection

  // data buffers for orbital advection. Only two x2-faces communicate
  ShearingBoxBoundaryBuffer sendbuf[2], recvbuf[2];

#if MPI_PARALLEL_ENABLED
  // unique MPI communicator for orbital advection
  MPI_Comm comm_orb_advect;
#endif

  // functions
  TaskStatus InitRecv();
  TaskStatus ClearRecv();
  TaskStatus ClearSend();

 protected:
  // must use pointer to MBPack and not parent physics module since parent can be one of
  // many types (Hydro, MHD, Radiation, etc.)
  MeshBlockPack *pmy_pack;
};

//----------------------------------------------------------------------------------------
//! \class OrbitalAdvectionCC
//  \brief Derived class implementing orbital advection of CC variables

class OrbitalAdvectionCC : public OrbitalAdvection {
 public:
  OrbitalAdvectionCC(MeshBlockPack *ppack, ParameterInput *pin, int nvar);
  // functions to communicate CC data with orbital advection
  TaskStatus PackAndSendCC(DvceArray5D<Real> &a);
  TaskStatus RecvAndUnpackCC(DvceArray5D<Real> &a, ReconstructionMethod rcon, Real qo);
};

//----------------------------------------------------------------------------------------
//! \class OrbitalAdvectionFC
//  \brief Derived class implementing orbital advection of FC variables

class OrbitalAdvectionFC : public OrbitalAdvection {
 public:
  OrbitalAdvectionFC(MeshBlockPack *ppack, ParameterInput *pin);
  // functions to communicate FC data with orbital advection
  TaskStatus PackAndSendFC(DvceFaceFld4D<Real> &b);
  TaskStatus RecvAndUnpackFC(DvceFaceFld4D<Real> &b0, ReconstructionMethod rcon, Real qo);
};

//----------------------------------------------------------------------------------------
//! \class ShearingBoxBoundary
//  \brief Abstract base class for shearing box boundary conditions for CC and FC vars

class ShearingBoxBoundary {
 public:
  ShearingBoxBoundary(MeshBlockPack *ppack, ParameterInput *pin);
  ~ShearingBoxBoundary();

  // data
  HostArray1D<int> nmb_x1bndry;    // number of MBs that touch x1 boundaries
  DualArray2D<int> x1bndry_mbgid;  // GIDs of MBs at x1 boundaries
  Real yshear;                     // x2-distance x1-boundaries have sheared

  // data buffers for shearing box BCs.  Only two x1-faces get sheared
  // Use seperate variables for ix1/ox1 since number of MBs on each face can be different
  ShearingBoxBoundaryBuffer sendbuf[2], recvbuf[2];

#if MPI_PARALLEL_ENABLED
  // unique MPI communicator for shearing box
  MPI_Comm comm_sbox;
#endif

  // functions
  TaskStatus InitRecv(Real qom, Real time);
  TaskStatus ClearRecv();
  TaskStatus ClearSend();
  // function to find target MB offset by shear.  Returns GID and rank
  void FindTargetMB(const int igid, const int jshift, int &gid, int &rank);
  // function to find index in x1bndry array of MB with input GID
  int TargetIndex(const int n, const int tgid) {
    for (int m=0; m<nmb_x1bndry(n); ++m) {
      if (x1bndry_mbgid.h_view(n,m) == tgid) return m;
    }
    return -1;
  }

 protected:
  // must use pointer to MBPack and not parent physics module since parent can be one of
  // many types (Hydro, MHD, Radiation, etc.)
  MeshBlockPack *pmy_pack;
};

//----------------------------------------------------------------------------------------
//! \class ShearingBoxBoundaryCC
//  \brief Derived class implementing shearing box boundary conditions for CC vars

class ShearingBoxBoundaryCC : public ShearingBoxBoundary {
 public:
  ShearingBoxBoundaryCC(MeshBlockPack *ppack, ParameterInput *pin, int nvar);
  // functions to communicate CC data with shearing box BCs
  TaskStatus PackAndSendCC(DvceArray5D<Real> &a, ReconstructionMethod rcon);
  TaskStatus RecvAndUnpackCC(DvceArray5D<Real> &a);
};

//----------------------------------------------------------------------------------------
//! \class ShearingBoxBoundaryFC
//  \brief Derived class implementing shearing box boundary conditions for FC vars

class ShearingBoxBoundaryFC : public ShearingBoxBoundary {
 public:
  ShearingBoxBoundaryFC(MeshBlockPack *ppack, ParameterInput *pin);
  // functions to communicate CC data with shearing box BCs
  TaskStatus PackAndSendFC(DvceFaceFld4D<Real> &b, ReconstructionMethod rcon);
  TaskStatus RecvAndUnpackFC(DvceFaceFld4D<Real> &b);
};

#endif // SHEARING_BOX_SHEARING_BOX_HPP_
