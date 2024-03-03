#ifndef SHEARING_BOX_SHEARING_BOX_HPP_
#define SHEARING_BOX_SHEARING_BOX_HPP_
//========================================================================================
// AthenaK astrophysical fluid dynamics code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file shearing_box.hpp
//  \brief definitions for ShearingBox class

#include "athena.hpp"
#include "parameter_input.hpp"
#include "tasklist/task_list.hpp"
#include "bvals/bvals.hpp"

// forward declarations


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
//! \struct ShearingBoxBuffer
//! \brief container for storage for boundary buffers with the shearing box
//! Basically a much simplified version of the BoundaryBuffer struct.

struct ShearingBoxBuffer {
  // 2D Views that store buffer data on device
  DvceArray5D<Real> vars;
#if MPI_PARALLEL_ENABLED
  // vectors of length (number of MBs) to hold MPI requests
  // Using STL vector causes problems with some GPU compilers, so just use plain C array
  MPI_Request *vars_req;
#endif
};

//----------------------------------------------------------------------------------------
//! \class ShearingBox

class ShearingBox {
 public:
  ShearingBox(MeshBlockPack *ppack, ParameterInput *pin, int nvar);
  ~ShearingBox();

  // data
  Real qshear, omega0;     // shearing box parameters
  int maxjshift;              // integer offset for orbital advection

  // container to hold names of TaskIDs
  ShearingBoxTaskIDs id;

  // data buffers for orbital advection. Only two x2-faces communicate
  ShearingBoxBuffer sendbuf_orb[2], recvbuf_orb[2];

  // functions...
  // functions to communicate CC data with orbital advection
  TaskStatus PackAndSendCC_Orb(DvceArray5D<Real> &a);
  TaskStatus RecvAndUnpackCC_Orb(DvceArray5D<Real> &a, ReconstructionMethod rcon);

 private:
  MeshBlockPack *pmy_pack;
};
#endif // SHEARING_BOX_SHEARING_BOX_HPP_
