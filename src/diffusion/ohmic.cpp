//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file ohmic.cpp
//  \brief Derived class for Ohmic resistivity

#include <iostream>

// Athena++ headers
#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "resistivity.hpp"
#include "current_density.hpp"

//----------------------------------------------------------------------------------------
// ctor: also calls Resistivity base class constructor
    
Ohmic::Ohmic(MeshBlockPack *pp, ParameterInput *pin, Real eta)
  : Resistivity(pp, pin), eta_ohm(eta)
{
  // resistive timestep on MeshBlock(s) in this pack
  auto size = pmy_pack->pmb->mbsize;
  Real fac;
  if (pp->pmesh->nx3gt1) {
    fac = 1.0/6.0;
  } else if (pp->pmesh->nx2gt1) {
    fac = 0.25;
  } else {
    fac = 0.5;
  }
  for (int m=0; m<(pp->nmb_thispack); ++m) {
    dtnew = std::min(dtnew, fac*SQR(size.dx1.h_view(m))/eta_ohm);
    if (pp->pmesh->nx2gt1) {dtnew = std::min(dtnew, fac*SQR(size.dx2.h_view(m))/eta_ohm);}
    if (pp->pmesh->nx3gt1) {dtnew = std::min(dtnew, fac*SQR(size.dx3.h_view(m))/eta_ohm);}
  }
}

//--------------------------------------------------------------------------------------
//! \fn AddResistiveEMF()
//  \brief Adds electric field from Ohmic resistivity to corner-centered electric field

void Ohmic::AddResistiveEMF(const DvceFaceFld4D<Real> &b0, DvceEdgeFld4D<Real> &efld)
{
  int is = pmy_pack->mb_cells.is; int ie = pmy_pack->mb_cells.ie;
  int js = pmy_pack->mb_cells.js; int je = pmy_pack->mb_cells.je;
  int ks = pmy_pack->mb_cells.ks; int ke = pmy_pack->mb_cells.ke;
  int ncells1 = pmy_pack->mb_cells.nx1 + 2*(pmy_pack->mb_cells.ng);
  int nmb1 = pmy_pack->nmb_thispack - 1;

  //---- 1-D problem:
  //  copy face-centered E-fields to edges and return.
  //  Note e2[is:ie+1,js:je,  ks:ke+1]
  //       e3[is:ie+1,js:je+1,ks:ke  ]

  if (!(pmy_pack->pmesh->nx2gt1)) {

    // capture class variables for the kernels
    auto e2 = efld.x2e;
    auto e3 = efld.x3e;
    auto &mbsize = pmy_pack->pmb->mbsize;

    int scr_level = 0;
    size_t scr_size = ScrArray1D<Real>::shmem_size(ncells1) * 3;

    par_for_outer("ohm1", DevExeSpace(), scr_size, scr_level, 0, nmb1,
      KOKKOS_LAMBDA(TeamMember_t member, const int m)
      {
        ScrArray1D<Real> j1(member.team_scratch(scr_level), ncells1);
        ScrArray1D<Real> j2(member.team_scratch(scr_level), ncells1);
        ScrArray1D<Real> j3(member.team_scratch(scr_level), ncells1);

        CurrentDensity(member, m, ks, js, is, ie+1, b0, mbsize, j1, j2, j3);

        par_for_inner(member, is, ie+1, [&](const int i)
        {
          e2(m,ks,  js  ,i) += eta_ohm*j2(i);
          e2(m,ke+1,js  ,i) += eta_ohm*j2(i);
          e3(m,ks  ,js  ,i) += eta_ohm*j3(i);
          e3(m,ks  ,je+1,i) += eta_ohm*j3(i);
        });
      }
    );
    return;
  }

  //---- 2-D problem:
  if (!(pmy_pack->pmesh->nx3gt1)) {

    // capture class variables for the kernels
    auto e1 = efld.x1e;
    auto e2 = efld.x2e;
    auto e3 = efld.x3e;
    auto &mbsize = pmy_pack->pmb->mbsize;

    int scr_level = 0;
    size_t scr_size = ScrArray1D<Real>::shmem_size(ncells1) * 3;

    par_for_outer("ohm2", DevExeSpace(), scr_size, scr_level, 0, nmb1, js, je+1,
      KOKKOS_LAMBDA(TeamMember_t member, const int m, const int j)
      {
        ScrArray1D<Real> j1(member.team_scratch(scr_level), ncells1);
        ScrArray1D<Real> j2(member.team_scratch(scr_level), ncells1);
        ScrArray1D<Real> j3(member.team_scratch(scr_level), ncells1);
        
        CurrentDensity(member, m, ks, j, is, ie+1, b0, mbsize, j1, j2, j3);
    
        par_for_inner(member, is, ie+1, [&](const int i)
        {
          e1(m,ks,  j,i) += eta_ohm*j1(i);
          e1(m,ke+1,j,i) += eta_ohm*j1(i);
          e2(m,ks,  j,i) += eta_ohm*j2(i);
          e2(m,ke+1,j,i) += eta_ohm*j2(i);
          e3(m,ks  ,j,i) += eta_ohm*j3(i);
        });
      } 
    );  
    return;
  }

  //---- 3-D problem:

  // capture class variables for the kernels
  auto e1 = efld.x1e;
  auto e2 = efld.x2e;
  auto e3 = efld.x3e;
  auto &mbsize = pmy_pack->pmb->mbsize;

  int scr_level = 0;
  size_t scr_size = ScrArray1D<Real>::shmem_size(ncells1) * 3;

  par_for_outer("ohm3", DevExeSpace(), scr_size, scr_level, 0, nmb1, ks, ke+1, js, je+1,
    KOKKOS_LAMBDA(TeamMember_t member, const int m, const int k, const int j)
    {
      ScrArray1D<Real> j1(member.team_scratch(scr_level), ncells1);
      ScrArray1D<Real> j2(member.team_scratch(scr_level), ncells1);
      ScrArray1D<Real> j3(member.team_scratch(scr_level), ncells1);

      CurrentDensity(member, m, k, j, is, ie+1, b0, mbsize, j1, j2, j3);

      par_for_inner(member, is, ie+1, [&](const int i)
      {
        e1(m,k,j,i) += eta_ohm*j1(i);
        e2(m,k,j,i) += eta_ohm*j2(i);
        e3(m,k,j,i) += eta_ohm*j3(i);
      });
    }
  );

  return;
}

//--------------------------------------------------------------------------------------
//! \fn AddResistivePoyntingFlux()
//  \brief Adds Poynting flux from non-ideal MHD to energy flux

/*
void Ohmic::AddResistiveEnergyFlux(EdgeField &e, const AthenaArray<Real> &bc) 
{
  auto e1 = efld.x1e;
  auto e2 = efld.x2e;
  auto e3 = efld.x3e;

  // set the loop limits for 1D/2D/3D problems
  int jl,ju,kl,ku;
  if (pmy_pack->pmesh->nx2gt1) {
    if (pmy_pack->pmesh->nx3gt1) { // 3D
      jl = js-1, ju = je+1, kl = ks-1, ku = ke+1;
    } else { // 2D
      jl = js-1, ju = je+1, kl = ks, ku = ke;
    } 
  } else { // 1D
    jl = js, ju = je, kl = ks, ku = ke;
  } 
      
  par_for_outer("ohmic_emf",DevExeSpace(),scr_size,scr_level, 0, nmb1, kl, ku, jl, ju,
    KOKKOS_LAMBDA(TeamMember_t member, const int m, const int k, const int j)
    {   
      ScrArray1D<Real> j1(member.team_scratch(scr_level), ncells1);
      ScrArray1D<Real> j2(member.team_scratch(scr_level), ncells1);
      ScrArray1D<Real> j3(member.team_scratch(scr_level), ncells1);
  
      Current(member, m, k, j, is, ie+1, b0, mbsize, j1, j2, j3)
        
      par_for_inner(member, is, ie+1, [&](const int i)
      {
        f1(k,j,i) = -0.25*(bc(IB2,k,j,i) + bc(IB2,k,j,i-1))
                    *(e3(k,j,i) + e3(k,j+1,i))
                    + 0.25*(bc(IB3,k,j,i) + bc(IB3,k,j,i-1))
                    *(e2(k,j,i) + e2(k+1,j,i));

        f2(k,j,i) = -0.25*(bc(IB3,k,j,i) + bc(IB3,k,j-1,i))
                    *(e1(k,j,i) + e1(k+1,j,i))
                    + 0.25*(bc(IB1,k,j,i) + bc(IB1,k,j-1,i))
                    *(e3(k,j,i) + e3(k,j,i+1));

        f3(k,j,i) = -0.25*(bc(IB1,k,j,i) + bc(IB1,k-1,j,i))
                    *(e2(k,j,i) + e2(k,j,i+1))
                    + 0.25*(bc(IB2,k,j,i) + bc(IB2,k-1,j,i))
                    *(e1(k,j,i) + e1(k,j+1,i));
      });
    } 
  );  
    
  return;             
}       
*/
