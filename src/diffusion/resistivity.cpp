//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file resistivity.cpp
//  \brief Implements functions for Resistivity class.

#include <algorithm>
#include <iostream>
#include <limits>

// Athena++ headers
#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "resistivity.hpp"
#include "current_density.hpp"

//----------------------------------------------------------------------------------------
// ctor: also calls Resistivity base class constructor

Resistivity::Resistivity(MeshBlockPack *pp, ParameterInput *pin) :
  pmy_pack(pp) {
  // Read parameters for Ohmic diffusion (if any)
  eta_ohm = pin->GetReal("mhd","ohmic_resistivity");

  // resistive timestep on MeshBlock(s) in this pack
  dtnew = std::numeric_limits<float>::max();
  auto size = pmy_pack->pmb->mb_size;
  Real fac;
  if (pp->pmesh->three_d) {
    fac = 1.0/6.0;
  } else if (pp->pmesh->two_d) {
    fac = 0.25;
  } else {
    fac = 0.5;
  }
  for (int m=0; m<(pp->nmb_thispack); ++m) {
    dtnew = std::min(dtnew, fac*SQR(size.h_view(m).dx1)/eta_ohm);
    if (pp->pmesh->multi_d) {dtnew = std::min(dtnew,fac*SQR(size.h_view(m).dx2)/eta_ohm);}
    if (pp->pmesh->three_d) {dtnew = std::min(dtnew,fac*SQR(size.h_view(m).dx3)/eta_ohm);}
  }
}

//----------------------------------------------------------------------------------------
// Resistivity destructor

Resistivity::~Resistivity() {
}

//----------------------------------------------------------------------------------------
//! \fn OhmicEField()
//  \brief Adds electric field from Ohmic resistivity to corner-centered electric field
//  Using Ohm's Law to compute the electric field:  E + (v x B) = \eta J, then
//    E_{inductive} = - (v x B)  [computed in the MHD Riemann solver]
//    E_{resistive} = \eta J     [computed in this function]

void Resistivity::OhmicEField(const DvceFaceFld4D<Real> &b0, DvceEdgeFld4D<Real> &efld) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int ncells1 = indcs.nx1 + 2*(indcs.ng);
  int nmb1 = pmy_pack->nmb_thispack - 1;

  //---- 1-D problem:
  //  copy face-centered E-fields to edges and return.
  //  Note e2[is:ie+1,js:je,  ks:ke+1]
  //       e3[is:ie+1,js:je+1,ks:ke  ]

  if (pmy_pack->pmesh->one_d) {
    // capture class variables for the kernels
    auto e2 = efld.x2e;
    auto e3 = efld.x3e;
    auto &mbsize = pmy_pack->pmb->mb_size;
    auto eta_o = eta_ohm;

    int scr_level = 0;
    size_t scr_size = ScrArray1D<Real>::shmem_size(ncells1) * 3;

    par_for_outer("ohm1", DevExeSpace(), scr_size, scr_level, 0, nmb1,
    KOKKOS_LAMBDA(TeamMember_t member, const int m) {
      ScrArray1D<Real> j1(member.team_scratch(scr_level), ncells1);
      ScrArray1D<Real> j2(member.team_scratch(scr_level), ncells1);
      ScrArray1D<Real> j3(member.team_scratch(scr_level), ncells1);

      CurrentDensity(member, m, ks, js, is, ie+1, b0, mbsize.d_view(m), j1, j2, j3);

      // Add E_{resistive} = \eta J to corner-centered electric fields
      par_for_inner(member, is, ie+1, [&](const int i) {
        e2(m,ks,  js  ,i) += eta_o*j2(i);
        e2(m,ke+1,js  ,i) += eta_o*j2(i);
        e3(m,ks  ,js  ,i) += eta_o*j3(i);
        e3(m,ks  ,je+1,i) += eta_o*j3(i);
      });
    });
    return;
  }

  //---- 2-D problem:
  if (pmy_pack->pmesh->two_d) {
    // capture class variables for the kernels
    auto e1 = efld.x1e;
    auto e2 = efld.x2e;
    auto e3 = efld.x3e;
    auto &mbsize = pmy_pack->pmb->mb_size;
    auto eta_o = eta_ohm;

    int scr_level = 0;
    size_t scr_size = ScrArray1D<Real>::shmem_size(ncells1) * 3;

    par_for_outer("ohm2", DevExeSpace(), scr_size, scr_level, 0, nmb1, js, je+1,
    KOKKOS_LAMBDA(TeamMember_t member, const int m, const int j) {
      ScrArray1D<Real> j1(member.team_scratch(scr_level), ncells1);
      ScrArray1D<Real> j2(member.team_scratch(scr_level), ncells1);
      ScrArray1D<Real> j3(member.team_scratch(scr_level), ncells1);

      CurrentDensity(member, m, ks, j, is, ie+1, b0, mbsize.d_view(m), j1, j2, j3);

      // Add E_{resistive} = \eta J to corner-centered electric fields
      par_for_inner(member, is, ie+1, [&](const int i) {
        e1(m,ks,  j,i) += eta_o*j1(i);
        e1(m,ke+1,j,i) += eta_o*j1(i);
        e2(m,ks,  j,i) += eta_o*j2(i);
        e2(m,ke+1,j,i) += eta_o*j2(i);
        e3(m,ks  ,j,i) += eta_o*j3(i);
      });
    });
    return;
  }

  //---- 3-D problem:

  // capture class variables for the kernels
  auto e1 = efld.x1e;
  auto e2 = efld.x2e;
  auto e3 = efld.x3e;
  auto &mbsize = pmy_pack->pmb->mb_size;
  auto eta_o = eta_ohm;

  int scr_level = 0;
  size_t scr_size = ScrArray1D<Real>::shmem_size(ncells1) * 3;

  par_for_outer("ohm3", DevExeSpace(), scr_size, scr_level, 0, nmb1, ks, ke+1, js, je+1,
  KOKKOS_LAMBDA(TeamMember_t member, const int m, const int k, const int j) {
    ScrArray1D<Real> j1(member.team_scratch(scr_level), ncells1);
    ScrArray1D<Real> j2(member.team_scratch(scr_level), ncells1);
    ScrArray1D<Real> j3(member.team_scratch(scr_level), ncells1);

    CurrentDensity(member, m, k, j, is, ie+1, b0, mbsize.d_view(m), j1, j2, j3);

    // Add E_{resistive} = \eta J to corner-centered electric fields
    par_for_inner(member, is, ie+1, [&](const int i) {
      e1(m,k,j,i) += eta_o*j1(i);
      e2(m,k,j,i) += eta_o*j2(i);
      e3(m,k,j,i) += eta_o*j3(i);
    });
  });

  return;
}

//----------------------------------------------------------------------------------------
//! \fn OhmicEnergyFlux()
//  \brief Adds Poynting flux from Ohmic resistivity to energy flux
//  Total energy equation is dE/dt = - Div(F) where F = (E X B) = \eta (J X B)


void Resistivity::OhmicEnergyFlux(const DvceFaceFld4D<Real> &b,
                                  DvceFaceFld5D<Real> &flx) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  auto size = pmy_pack->pmb->mb_size;
  bool &multi_d = pmy_pack->pmesh->multi_d;
  bool &three_d = pmy_pack->pmesh->three_d;
  Real qa = 0.25*eta_ohm;

  //------------------------------
  // energy fluxes in x1-direction

  auto &flx1 = flx.x1f;
  par_for("ohm_heat1", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie+1,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real j2k   = -(b.x3f(m,k  ,j,i) - b.x3f(m,k  ,j,i-1))/size.d_view(m).dx1;
    Real j2kp1 = -(b.x3f(m,k+1,j,i) - b.x3f(m,k+1,j,i-1))/size.d_view(m).dx1;

    Real j3j   = (b.x2f(m,k,j  ,i) - b.x2f(m,k,j  ,i-1))/size.d_view(m).dx1;
    Real j3jp1 = (b.x2f(m,k,j+1,i) - b.x2f(m,k,j+1,i-1))/size.d_view(m).dx1;

    if (multi_d) {
      j3j   -= (b.x1f(m,k,j  ,i) - b.x1f(m,k,j-1,i))/size.d_view(m).dx2;
      j3jp1 -= (b.x1f(m,k,j+1,i) - b.x1f(m,k,j  ,i))/size.d_view(m).dx2;
    }
    if (three_d) {
      j2k   += (b.x1f(m,k  ,j,i) - b.x1f(m,k-1,j,i))/size.d_view(m).dx3;
      j2kp1 += (b.x1f(m,k+1,j,i) - b.x1f(m,k  ,j,i))/size.d_view(m).dx3;
    }

    // flx1 = (E X B)_{1} =  ((\eta J) X B)_{1} = \eta (J2*B3 - J3*B2)
    flx1(m,IEN,k,j,i) += qa*(j2k  *(b.x3f(m,k  ,j  ,i) + b.x3f(m,k  ,j  ,i-1)) +
                             j2kp1*(b.x3f(m,k+1,j  ,i) + b.x3f(m,k+1,j  ,i-1)) -
                             j3j  *(b.x2f(m,k  ,j  ,i) + b.x2f(m,k  ,j  ,i-1)) -
                             j3jp1*(b.x2f(m,k  ,j+1,i) + b.x2f(m,k  ,j+1,i-1)));
  });
  if (pmy_pack->pmesh->one_d) {return;}

  //------------------------------
  // energy fluxes in x2-direction

  auto &flx2 = flx.x2f;
  par_for("ohm_heat2", DevExeSpace(), 0, nmb1, ks, ke, js, je+1, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real j1k   = (b.x3f(m,k  ,j,i) - b.x3f(m,k  ,j-1,i))/size.d_view(m).dx2;
    Real j1kp1 = (b.x3f(m,k+1,j,i) - b.x3f(m,k+1,j-1,i))/size.d_view(m).dx2;

    Real j3i   = (b.x2f(m,k,j,i  ) - b.x2f(m,k,j  ,i-1))/size.d_view(m).dx1
               - (b.x1f(m,k,j,i  ) - b.x1f(m,k,j-1,i  ))/size.d_view(m).dx2;
    Real j3ip1 = (b.x2f(m,k,j,i+1) - b.x2f(m,k,j  ,i  ))/size.d_view(m).dx1
               - (b.x1f(m,k,j,i+1) - b.x1f(m,k,j-1,i+1))/size.d_view(m).dx2;

    if (three_d) {
      j1k   -= (b.x2f(m,k  ,j,i) - b.x2f(m,k-1,j,i))/size.d_view(m).dx3;
      j1kp1 -= (b.x2f(m,k+1,j,i) - b.x2f(m,k  ,j,i))/size.d_view(m).dx3;
    }

    // E2 = \eta (J X B)_{2} = \eta (J3*B1 - J1*B3)
    flx2(m,IEN,k,j,i) += qa*(j3i  *(b.x1f(m,k  ,j,i  ) + b.x1f(m,k  ,j-1,i  )) +
                             j3ip1*(b.x1f(m,k  ,j,i+1) + b.x1f(m,k  ,j-1,i+1)) -
                             j1k  *(b.x3f(m,k  ,j,i  ) + b.x3f(m,k  ,j-1,i  )) -
                             j1kp1*(b.x3f(m,k+1,j,i  ) + b.x3f(m,k+1,j-1,i  )));
  });
  if (pmy_pack->pmesh->two_d) {return;}

  //------------------------------
  // energy fluxes in x3-direction

  auto &flx3 = flx.x3f;
  par_for("ohm_heat3", DevExeSpace(), 0, nmb1, ks, ke+1, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real j1j   = (b.x3f(m,k,j  ,i) - b.x3f(m,k  ,j-1,i))/size.d_view(m).dx2
               - (b.x2f(m,k,j  ,i) - b.x2f(m,k-1,j  ,i))/size.d_view(m).dx3;
    Real j1jp1 = (b.x3f(m,k,j+1,i) - b.x3f(m,k  ,j  ,i))/size.d_view(m).dx2
               - (b.x2f(m,k,j+1,i) - b.x2f(m,k-1,j+1,i))/size.d_view(m).dx3;

    Real j2i   = -(b.x3f(m,k,j,i  ) - b.x3f(m,k  ,j,i-1))/size.d_view(m).dx1
                + (b.x1f(m,k,j,i  ) - b.x1f(m,k-1,j,i  ))/size.d_view(m).dx3;
    Real j2ip1 = -(b.x3f(m,k,j,i+1) - b.x3f(m,k  ,j,i  ))/size.d_view(m).dx1
                + (b.x1f(m,k,j,i+1) - b.x1f(m,k-1,j,i+1))/size.d_view(m).dx3;

    // E2 = \eta (J X B)_{2} = \eta (J1*B2 - J2*B1)
    flx3(m,IEN,k,j,i) += qa*(j1j  *(b.x2f(m,k,j  ,i  ) + b.x2f(m,k-1,j  ,i  )) +
                             j1jp1*(b.x2f(m,k,j+1,i  ) + b.x2f(m,k-1,j+1,i  )) -
                             j2i  *(b.x1f(m,k,j  ,i  ) + b.x1f(m,k-1,j  ,i  )) -
                             j2ip1*(b.x1f(m,k,j  ,i+1) + b.x1f(m,k-1,j  ,i+1)));
  });

  return;
}
