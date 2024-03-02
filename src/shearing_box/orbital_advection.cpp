//========================================================================================
// AthenaK astrophysical fluid dynamics code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file orbital_advection.cpp
//! \brief Functions to update cell-centered and face-centered quantities via orbital
//! advection.

#include <iostream>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "orbital_advection.hpp"
#include "coordinates/cell_locations.hpp"

namespace shearing_box {
//----------------------------------------------------------------------------------------
//! \fn void ShearingBox::OrbitalAdvectionCC
//! \brief Remaps cell-centered variables in input array u0 using orbital advection

void ShearingBox::OrbitalAdvectionCC(DvceArray5D<Real> u0, ReconstructionMethod rcon) {
  int nmb = pmy_mesh->pmb_pack->nmb_thispack;
  int nvar = u0.extent_int(1);  // TODO(@user): 2nd index from L of in array must be NVAR

  RegionIndcs &indcs = pmy_mesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int ncells2 = indcs.nx2 + 2*(indcs.ng);

  auto &mb_size = pmy_mesh->pmb_pack->pmb->mb_size;
  auto &mesh_size = pmy_mesh->mesh_size;
  Real &time = pmy_mesh->time;
  Real qom = qshear*omega0;
  Real ly = (mesh_size.x2max - mesh_size.x2min);


  int scr_level=0;
  size_t scr_size = ScrArray1D<Real>::shmem_size(ncells2) * 2;
  par_for_outer("orb_adv",DevExeSpace(),scr_size,scr_level,0,(nmb-1),0,(nvar-1), ks, ke, is, ie,
  KOKKOS_LAMBDA(TeamMember_t member, const int m, const int n, const int k, const int i) {
    ScrArray1D<Real> u0_(member.team_scratch(scr_level), ncells2);
    ScrArray1D<Real> flx(member.team_scratch(scr_level), ncells2);

    Real &x1min = mb_size.d_view(m).x1min;
    Real &x1max = mb_size.d_view(m).x1max;
    int nx1 = indcs.nx1;
    Real x1v = CellCenterX(i-is, nx1, x1min, x1max);

    Real yshear = -qom*x1v*time;
    Real deltay = fmod(yshear, ly);
    int joffset = static_cast<int>(deltay/(mb_size.d_view(m).dx2));

    // Load scratch array.  Index with shift:  jj = j + jshift
    par_for_inner(member, 0, (ncells2-1), [&](const int jj) {
      if (jj < (js + joffset)) {
        // Load scratch arrays from L boundary buffer with offset
        u0_(jj) = recv_buf_orb(n,k,jj,i);
      } else if (jj < (je + joffset)) {
        // Load from array itself with offset
        u0_(jj) = u0(m,n,k,(jj+joffset),i);
      } else {
        // Load scratch arrays from R boundary buffer with offset
        u0_(jj) = recv_buf_orb(n,k,(jj-(je+1)),i);
      }
    });


    // Compute x2-fluxes from fractional offset, including in ghost zones
    Real epsi = fmod(deltay,(mb_size.d_view(m).dx2))/(mb_size.d_view(m).dx2);
    switch (rcon) {
      case ReconstructionMethod::dc:
        DonorCellOrbAdvFlx(member, js, je+1, epsi, u0_, flx);
        break;
      case ReconstructionMethod::plm:
        PiecewiseLinearOrbAdvFlx(member, js, je+1, epsi, u0_, flx);
        break;
//      case ReconstructionMethod::ppm4:
//      case ReconstructionMethod::ppmx:
//          PiecewiseParabolicOrbAdvFlx(member,eos_,extrema,true,m,k,j,il,iu, w0_, wl_jp1, wr);
//        break;
      default:
        break;
    }
    member.team_barrier();

    // Update CC variables (including ghost zones) with orbital advection fluxes
    par_for_inner(member, js, je, [&](const int j) {
      u0(m,n,k,j,i) = u0_(j) + (flx(j+1) - flx(j));
    });
  });

  return;
}

} // namespace shearing_box
