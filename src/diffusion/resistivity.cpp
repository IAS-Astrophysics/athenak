//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file resistivity.cpp
//  \brief Implements functions for Resistivity class.

#include <float.h>
#include <algorithm>
#include <iostream>
#include <limits>
#include <string> // string

// Athena++ headers
#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "resistivity.hpp"

//----------------------------------------------------------------------------------------
// ctor: also calls Resistivity base class constructor

Resistivity::Resistivity(MeshBlockPack *pp, ParameterInput *pin) :
    pmy_pack(pp) {
  // Read parameters for Ohmic resistivity (if any)
  if (pin->DoesParameterExist("mhd","ohmic_resistivity")) {
    iso_resist_type = pin->GetString("mhd","ohmic_resistivity");
    // Check for valid type
    if (iso_resist_type.compare("constant") != 0) {
      std::cout << "### FATAL ERROR in "<< __FILE__ <<" at line " << __LINE__ << std::endl
                << "Invalid choice for Ohmic resistivity type" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    // constant resistivity
    eta_ohm = pin->GetReal("mhd","eta_ohm");
  }
}

//----------------------------------------------------------------------------------------
// Resistivity destructor

Resistivity::~Resistivity() {
}

//----------------------------------------------------------------------------------------
//! \fn void AddResistiveEMFs()
//! \brief Wrapper function that adds electric fields for different types of resistivity
//! Currently only Ohmic resistivity with constant coefficient is implemented.

void Resistivity::AddResistiveEMFs(const DvceEdgeFld4D<Real> &jedge,
    DvceEdgeFld4D<Real> &efld) {
  AddEMFConstantResist(jedge, efld);
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void AddResistiveFluxes()
//! \brief Wrapper function that adds energy (Poynting) fluxes for different types of
//! resistivity.
//! Currently only Ohmic resistivity with constant coefficient is implemented.

void Resistivity::AddResistiveFluxes(const DvceEdgeFld4D<Real> &jedge,
    const DvceFaceFld4D<Real> &b0, DvceFaceFld5D<Real> &flx) {
  AddFluxConstantResist(jedge, b0, flx);
  return;
}

//----------------------------------------------------------------------------------------
//! \fn AddEMFConstantResist()
//  \brief Adds electric field from Ohmic resistivity to corner-centered electric field
//  Using Ohm's Law to compute the electric field:  E + (v x B) = \eta J, then
//    E_{inductive} = - (v x B)  [computed in the MHD Riemann solver]
//    E_{resistive} = \eta J     [computed in this function]

void Resistivity::AddEMFConstantResist(const DvceEdgeFld4D<Real> &jedge,
    DvceEdgeFld4D<Real> &efld) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;

  auto e1 = efld.x1e;
  auto e2 = efld.x2e;
  auto e3 = efld.x3e;
  auto je1 = jedge.x1e;
  auto je2 = jedge.x2e;
  auto je3 = jedge.x3e;
  auto eta_o = eta_ohm;

  //---- 1-D problem:
  if (pmy_pack->pmesh->one_d) {
    par_for("ohm1", DevExeSpace(), 0, nmb1, is, ie+1,
    KOKKOS_LAMBDA(const int m, const int i) {
      e2(m,ks,  js  ,i) += eta_o*je2(m,ks,  js  ,i);
      e2(m,ke+1,js  ,i) += eta_o*je2(m,ke+1,js  ,i);
      e3(m,ks  ,js  ,i) += eta_o*je3(m,ks  ,js  ,i);
      e3(m,ks  ,je+1,i) += eta_o*je3(m,ks  ,je+1,i);
    });
    return;
  }

  //---- 2-D problem:
  if (pmy_pack->pmesh->two_d) {
    par_for("ohm2", DevExeSpace(), 0, nmb1, js, je+1, is, ie+1,
    KOKKOS_LAMBDA(const int m, const int j, const int i) {
      e1(m,ks,  j,i) += eta_o*je1(m,ks,  j,i);
      e1(m,ke+1,j,i) += eta_o*je1(m,ke+1,j,i);
      e2(m,ks,  j,i) += eta_o*je2(m,ks,  j,i);
      e2(m,ke+1,j,i) += eta_o*je2(m,ke+1,j,i);
      e3(m,ks  ,j,i) += eta_o*je3(m,ks  ,j,i);
    });
    return;
  }

  //---- 3-D problem:
  par_for("ohm3", DevExeSpace(), 0, nmb1, ks, ke+1, js, je+1, is, ie+1,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    e1(m,k,j,i) += eta_o*je1(m,k,j,i);
    e2(m,k,j,i) += eta_o*je2(m,k,j,i);
    e3(m,k,j,i) += eta_o*je3(m,k,j,i);
  });

  return;
}

//----------------------------------------------------------------------------------------
//! \fn AddResistiveFluxConstantResist()
//  \brief Adds Poynting flux from Ohmic resistivity to energy flux
//  Total energy equation is dE/dt = - Div(F) where F = (E X B) = \eta (J X B)

void Resistivity::AddFluxConstantResist(const DvceEdgeFld4D<Real> &jedge,
                                        const DvceFaceFld4D<Real> &b,
                                        DvceFaceFld5D<Real> &flx) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  Real qa = 0.25*eta_ohm;

  auto je1 = jedge.x1e;
  auto je2 = jedge.x2e;
  auto je3 = jedge.x3e;

  //------------------------------
  // energy fluxes in x1-direction: S_1 = eta*(J2*B3 - J3*B2)
  auto &flx1 = flx.x1f;
  par_for("ohm_heat1", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie+1,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real j2k   = je2(m,k,j,i);
    Real j2kp1 = je2(m,k+1,j,i);
    Real j3j   = je3(m,k,j,i);
    Real j3jp1 = je3(m,k,j+1,i);

    flx1(m,IEN,k,j,i) += qa*(j2k  *(b.x3f(m,k  ,j  ,i) + b.x3f(m,k  ,j  ,i-1)) +
                             j2kp1*(b.x3f(m,k+1,j  ,i) + b.x3f(m,k+1,j  ,i-1)) -
                             j3j  *(b.x2f(m,k  ,j  ,i) + b.x2f(m,k  ,j  ,i-1)) -
                             j3jp1*(b.x2f(m,k  ,j+1,i) + b.x2f(m,k  ,j+1,i-1)));
  });
  if (pmy_pack->pmesh->one_d) {return;}

  //------------------------------
  // energy fluxes in x2-direction: S_2 = eta*(J3*B1 - J1*B3)
  auto &flx2 = flx.x2f;
  par_for("ohm_heat2", DevExeSpace(), 0, nmb1, ks, ke, js, je+1, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real j1k   = je1(m,k,j,i);
    Real j1kp1 = je1(m,k+1,j,i);
    Real j3i   = je3(m,k,j,i);
    Real j3ip1 = je3(m,k,j,i+1);

    flx2(m,IEN,k,j,i) += qa*(j3i  *(b.x1f(m,k  ,j,i  ) + b.x1f(m,k  ,j-1,i  )) +
                             j3ip1*(b.x1f(m,k  ,j,i+1) + b.x1f(m,k  ,j-1,i+1)) -
                             j1k  *(b.x3f(m,k  ,j,i  ) + b.x3f(m,k  ,j-1,i  )) -
                             j1kp1*(b.x3f(m,k+1,j,i  ) + b.x3f(m,k+1,j-1,i  )));
  });
  if (pmy_pack->pmesh->two_d) {return;}

  //------------------------------
  // energy fluxes in x3-direction: S_3 = eta*(J1*B2 - J2*B1)
  auto &flx3 = flx.x3f;
  par_for("ohm_heat3", DevExeSpace(), 0, nmb1, ks, ke+1, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real j1j   = je1(m,k,j,i);
    Real j1jp1 = je1(m,k,j+1,i);
    Real j2i   = je2(m,k,j,i);
    Real j2ip1 = je2(m,k,j,i+1);

    flx3(m,IEN,k,j,i) += qa*(j1j  *(b.x2f(m,k,j  ,i  ) + b.x2f(m,k-1,j  ,i  )) +
                             j1jp1*(b.x2f(m,k,j+1,i  ) + b.x2f(m,k-1,j+1,i  )) -
                             j2i  *(b.x1f(m,k,j  ,i  ) + b.x1f(m,k-1,j  ,i  )) -
                             j2ip1*(b.x1f(m,k,j  ,i+1) + b.x1f(m,k-1,j  ,i+1)));
  });

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void Resistivity::NewTimeStep()
//! \brief Compute new time step for resistive MHD

void Resistivity::NewTimeStep(const DvceArray5D<Real> &w0, const EOS_Data &eos_data) {
  // resistive timestep on MeshBlock(s) in this pack
  dtnew = std::numeric_limits<float>::max();
  auto size = pmy_pack->pmb->mb_size;
  Real fac;
  if (pmy_pack->pmesh->three_d) {
    fac = 1.0/6.0;
  } else if (pmy_pack->pmesh->two_d) {
    fac = 0.25;
  } else {
    fac = 0.5;
  }
  for (int m=0; m<(pmy_pack->nmb_thispack); ++m) {
    dtnew = std::min(dtnew, fac*SQR(size.h_view(m).dx1)/eta_ohm);
    if (pmy_pack->pmesh->multi_d) {
      dtnew = std::min(dtnew, fac*SQR(size.h_view(m).dx2)/eta_ohm);
    }
    if (pmy_pack->pmesh->three_d) {
      dtnew = std::min(dtnew, fac*SQR(size.h_view(m).dx3)/eta_ohm);
    }
  }
  return;
}
