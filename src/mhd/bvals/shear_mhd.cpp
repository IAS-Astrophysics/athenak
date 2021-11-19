//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file shear_mhd.cpp
//  \brief implementation of shearing periodic BCs for Hydro conserved vars in x1-dir.
//  To use, set ix1/ox1 bcs to 'periodic' so that ghost cells in x1-direction are
//  communicated as periodic. Also set flag speriodic to true (stored as shear_periodic
//  in Mesh class) so that these functions are called to add shear in y-direction.
//  BCs applied to a single MeshBlock specified by input integer index to each function

// Athena++ headers
#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "mhd/mhd.hpp"
#include "eos/eos.hpp"
#include "srcterms/srcterms.hpp"

namespace mhd {

//----------------------------------------------------------------------------------------
//! \fn void MHD::ShearInnerX1(
//  \brief SHEARING (PERIODIC) boundary conditions, inner x1 boundary

void MHD::ShearInnerX1(int m)
{
  auto &indcs = pmy_pack->pcoord->mbdata.indcs;
  int &ng = indcs.ng;
  int n2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng) : 1;
  int &is = indcs.is;
  Real x1size = pmy_pack->pmesh->mesh_size.x1max - pmy_pack->pmesh->mesh_size.x1min;

  // Add shear offset to v3
  Real qomega = (psrc->qshear)*(psrc->omega0);
  auto &eos = peos->eos_data;
  auto &u0_ = u0;
  par_for("shear_ix1", DevExeSpace(),0,(n3-1),0,(n2-1),0,(ng-1),
    KOKKOS_LAMBDA(int k, int j, int i)
    {
      Real deltam = u0_(m,IDN,k,j,is-i-1)*qomega*x1size;
      u0_(m,IM3,k,j,is-i-1) += deltam;
      if (eos.is_ideal) {
        u0_(m,IEN,k,j,is-i-1) += deltam*(2.0*u0_(m,IM3,k,j,is-i-1) - deltam)/
                                         (2.0*u0_(m,IDN,k,j,is-i-1));
      }
    }
  );

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MHD::ShearOuterX1(
//  \brief SHEARING (PERIODIC) boundary conditions, outer x1 boundary

void MHD::ShearOuterX1(int m)
{
  auto &indcs = pmy_pack->pcoord->mbdata.indcs;
  int &ng = indcs.ng;
  int n2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng) : 1;
  int &ie = indcs.ie;
  Real x1size = pmy_pack->pmesh->mesh_size.x1max - pmy_pack->pmesh->mesh_size.x1min;

  // Add shear offset to v3
  Real qomega = (psrc->qshear)*(psrc->omega0);
  auto &eos = peos->eos_data;
  auto &u0_ = u0;
  par_for("shear_ox1", DevExeSpace(),0,(n3-1),0,(n2-1),0,(ng-1),
    KOKKOS_LAMBDA(int k, int j, int i)
    {
      Real deltam = -u0_(m,IDN,k,j,ie+i+1)*qomega*x1size;
      u0_(m,IM3,k,j,ie+i+1) += deltam;
      if (eos.is_ideal) {
        u0_(m,IEN,k,j,ie+i+1) += deltam*(2.0*u0_(m,IM3,k,j,ie+i+1) - deltam)/
                                         (2.0*u0_(m,IDN,k,j,ie+i+1));
      }
    }
  );

  return;
}

} // namespace mhd
