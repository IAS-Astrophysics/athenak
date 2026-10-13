#ifndef SHEARING_BOX_ORBITAL_ADVECTION_HPP_
#define SHEARING_BOX_ORBITAL_ADVECTION_HPP_
//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file orbital_advection.hpp
//! \brief definitions for classes that implement orbital advection abstract base and
//! derived classes (for CC and FC variables).

#include "athena.hpp"
#include "parameter_input.hpp"
#include "shearing_box/shearing_box.hpp"
#include "tasklist/task_list.hpp"
#include "bvals/bvals.hpp"

//----------------------------------------------------------------------------------------
//! \class OrbitalAdvection
//  \brief Abstract base class for orbital advection of CC and FC variables

class OrbitalAdvection {
 public:
  OrbitalAdvection(MeshBlockPack *ppack, ParameterInput *pin);
  ~OrbitalAdvection();

  // data
  int maxjshift;            // maximum integer shift of any cell in orbital advection
  Real qshear, omega0;      // Copies needed for all OA functions
  bool shearing_box_r_phi;  // NOT YET IMPLEMENTED

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
  TaskStatus RecvAndUnpackCC(DvceArray5D<Real> &a, ReconstructionMethod rcon);
};

//----------------------------------------------------------------------------------------
//! \class OrbitalAdvectionFC
//  \brief Derived class implementing orbital advection of FC variables

class OrbitalAdvectionFC : public OrbitalAdvection {
 public:
  OrbitalAdvectionFC(MeshBlockPack *ppack, ParameterInput *pin);

  // containers for Ex and Ey used to remap magnetic fields
  DvceArray4D<Real> emfx, emfz;

  // functions to communicate FC data with orbital advection
  TaskStatus PackAndSendFC(DvceFaceFld4D<Real> &b);
  TaskStatus RecvAndUnpackFC(DvceFaceFld4D<Real> &b0, ReconstructionMethod rcon);
};

#endif // SHEARING_BOX_ORBITAL_ADVECTION_HPP_
