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
  OrbitalAdvection(MeshBlockPack *ppack, ParameterInput *pin, int nvar);
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
  OrbitalAdvectionFC(MeshBlockPack *ppack, ParameterInput *pin, int nvar);
  // functions to communicate FC data with orbital advection
  TaskStatus PackAndSendFC(DvceFaceFld4D<Real> &b);
  TaskStatus RecvAndUnpackFC(DvceFaceFld4D<Real> &b0, ReconstructionMethod rcon, Real qo);
};

//----------------------------------------------------------------------------------------
//! \class ShearingBoxBoundary
//  \brief Abstract base class for shearing box boundary conditions for CC and FC vars

class ShearingBoxBoundary {
 public:
  ShearingBoxBoundary(MeshBlockPack *ppack, ParameterInput *pin, int nvar);
  ~ShearingBoxBoundary();

  // data
  int nmb_ix1bndry, nmb_ox1bndry;   // number of MBs that touch inner/outer x1 boundaries
  DualArray1D<int> ix1bndry_mbgid;  // GIDs of MBs at inner x1 shear periodic boundaries
  DualArray1D<int> ox1bndry_mbgid;  // GIDs of MBs at outer x1 shear periodic boundaries

  // data buffers for shearing box BCs. One x1-face can communicate with up to 3 nghbrs
  ShearingBoxBoundaryBuffer sendbuf_ix1[3], recvbuf_ix1[3];
  ShearingBoxBoundaryBuffer sendbuf_ox1[3], recvbuf_ox1[3];

#if MPI_PARALLEL_ENABLED
  // unique MPI communicator for shearing box
  MPI_Comm comm_sbox;
#endif

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
  TaskStatus PackAndSendCC(DvceArray5D<Real> &a);
};

#endif // SHEARING_BOX_SHEARING_BOX_HPP_
