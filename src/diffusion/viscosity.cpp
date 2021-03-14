//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file viscosity.cpp
//  \brief Implements functions for Viscosity class. This includes isotropic viscosity in
//  a Newtonian fluid (in which stress is proportional to shear).

#include <limits>
#include <iostream>

// Athena++ headers
#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "viscosity.hpp"

//----------------------------------------------------------------------------------------
// ctor: also calls Viscosity base class constructor

Viscosity::Viscosity(MeshBlockPack *pp, ParameterInput *pin)
  : pmy_pack(pp)
{
  // Read parameters for Hydro viscosity (if any)
  hydro_nu_iso = pin->GetOrAddReal("viscosity","hydro_nu_iso",0.0);
  if (pp->phydro == nullptr && hydro_nu_iso != 0.0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "<viscosity>/hydro_nu_iso = " << hydro_nu_iso
              << " but no <hydro> block in the input file" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // Read parameters for MHD viscosity (if any)
  mhd_nu_iso = pin->GetOrAddReal("viscosity","mhd_nu_iso",0.0);
  if (pp->pmhd == nullptr && mhd_nu_iso != 0.0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "<viscosity>/mhd_nu_iso = " << mhd_nu_iso 
              << " but no <mhd> block in the input file" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  if (hydro_nu_iso == 0.0 && mhd_nu_iso == 0.0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "<viscosity> block defined in input file, but coefficients of viscosity" 
              << " all zero" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // viscous timestep on MeshBlock(s) in this pack
  dtnew = std::numeric_limits<float>::max();
  auto size = pmy_pack->pmb->mbsize;
  Real fac;
  if (pp->pmesh->nx3gt1) {
    fac = 1.0/6.0;
  } else if (pp->pmesh->nx2gt1) {
    fac = 0.25;
  } else {
    fac = 0.5;
  }
  Real nu_iso = std::max(hydro_nu_iso, mhd_nu_iso);
  for (int m=0; m<(pp->nmb_thispack); ++m) {
    dtnew = std::min(dtnew, fac*SQR(size.dx1.h_view(m))/nu_iso);
    if (pp->pmesh->nx2gt1) {dtnew = std::min(dtnew, fac*SQR(size.dx2.h_view(m))/nu_iso);}
    if (pp->pmesh->nx3gt1) {dtnew = std::min(dtnew, fac*SQR(size.dx3.h_view(m))/nu_iso);}
  }

}

//----------------------------------------------------------------------------------------
// Viscosity destructor

Viscosity::~Viscosity()
{
}

//----------------------------------------------------------------------------------------
//! \fn  void Viscosity::AssembleStageRunTasks
//  \brief inserts Viscosity tasks into stage run TaskList
//  Called by MeshBlockPack::AddPhysicsModules() function directly after Viscosity cons

void Viscosity::AssembleStageRunTasks(TaskList &tl, TaskID start)
{
  if (hydro_nu_iso != 0.0) {
    auto id = tl.InsertTask(&Viscosity::AddViscosityHydro, this, 
                       pmy_pack->phydro->hydro_tasks[HydroTaskName::calc_flux],
                       pmy_pack->phydro->hydro_tasks[HydroTaskName::update]);
    visc_tasks.emplace(ViscosityTaskName::hydro_vflux, id);
  }

  if (mhd_nu_iso != 0.0) {
    auto id = tl.InsertTask(&Viscosity::AddViscosityMHD, this, 
                       pmy_pack->pmhd->mhd_tasks[MHDTaskName::calc_flux],
                       pmy_pack->pmhd->mhd_tasks[MHDTaskName::update]);
    visc_tasks.emplace(ViscosityTaskName::hydro_vflux, id);
  }

/*****/
std::cout << std::endl;
tl.PrintIDs();
std::cout << std::endl;
tl.PrintDependencies();
std::cout << std::endl;
/*****/
  return;
}

//----------------------------------------------------------------------------------------
//! \fn  void Viscosity::AddViscosityHydro
//  \brief

TaskStatus Viscosity::AddViscosityHydro(Driver *pdrive, int stage)
{
  AddIsoViscousFlux(pmy_pack->phydro->u0, pmy_pack->phydro->uflx, hydro_nu_iso);
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  void Viscosity::AddViscosityMHD
//  \brief

TaskStatus Viscosity::AddViscosityMHD(Driver *pdrive, int stage)
{
  AddIsoViscousFlux(pmy_pack->pmhd->u0, pmy_pack->pmhd->uflx, mhd_nu_iso);
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn void AddIsoViscousFlux
//  \brief Adds viscous fluxes to face-centered fluxes of conserved variables

void Viscosity::AddIsoViscousFlux(const DvceArray5D<Real> &w0, DvceFaceFld5D<Real> &flx,
                                  const Real nu_iso)
{
  int is = pmy_pack->mb_cells.is; int ie = pmy_pack->mb_cells.ie;
  int js = pmy_pack->mb_cells.js; int je = pmy_pack->mb_cells.je;
  int ks = pmy_pack->mb_cells.ks; int ke = pmy_pack->mb_cells.ke;
  int ncells1 = pmy_pack->mb_cells.nx1 + 2*(pmy_pack->mb_cells.ng);
  int nmb1 = pmy_pack->nmb_thispack - 1;
  auto size = pmy_pack->pmb->mbsize;
  bool &nx2gt1 = pmy_pack->pmesh->nx2gt1;
  bool &nx3gt1 = pmy_pack->pmesh->nx3gt1;

  //--------------------------------------------------------------------------------------
  // fluxes in x1-direction

  int scr_level = 0;
  size_t scr_size = (ScrArray1D<Real>::shmem_size(ncells1)) * 3;
  auto flx1 = flx.x1f;

  par_for_outer("visc1",DevExeSpace(), scr_size, scr_level, 0, nmb1, ks, ke, js, je,
    KOKKOS_LAMBDA(TeamMember_t member, const int m, const int k, const int j)
    {
      ScrArray1D<Real> fvx(member.team_scratch(scr_level), ncells1);
      ScrArray1D<Real> fvy(member.team_scratch(scr_level), ncells1);
      ScrArray1D<Real> fvz(member.team_scratch(scr_level), ncells1);

      // Add [2(dVx/dx)-(2/3)dVx/dx, dVy/dx, dVz/dx]
      par_for_inner(member, is, ie+1, [&](const int i)
      {
        fvx(i) = 4.0*(w0(m,IVX,k,j,i) - w0(m,IVX,k,j,i-1))/(3.0*size.dx1.d_view(m));
        fvy(i) =     (w0(m,IVY,k,j,i) - w0(m,IVY,k,j,i-1))/size.dx1.d_view(m);
        fvz(i) =     (w0(m,IVZ,k,j,i) - w0(m,IVZ,k,j,i-1))/size.dx1.d_view(m);
      });

      // In 2D/3D Add [(-2/3)dVy/dy, dVx/dy, 0]
      if (nx2gt1) {
        par_for_inner(member, is, ie+1, [&](const int i)
        {
          fvx(i) -= ((w0(m,IVY,k,j+1,i) + w0(m,IVY,k,j+1,i-1)) -
                     (w0(m,IVY,k,j-1,i) + w0(m,IVY,k,j-1,i-1)))/(6.0*size.dx2.d_view(m));
          fvy(i) += ((w0(m,IVX,k,j+1,i) + w0(m,IVX,k,j+1,i-1)) -
                     (w0(m,IVX,k,j-1,i) + w0(m,IVX,k,j-1,i-1)))/(4.0*size.dx2.d_view(m));
        });
      }

      // In 3D Add [(-2/3)dVz/dz, 0,  dVx/dz]
      if (nx3gt1) {
        par_for_inner(member, is, ie+1, [&](const int i)
        {
          fvx(i) -= ((w0(m,IVZ,k+1,j,i) + w0(m,IVZ,k+1,j,i-1)) -
                     (w0(m,IVZ,k-1,j,i) + w0(m,IVZ,k-1,j,i-1)))/(6.0*size.dx3.d_view(m));
          fvz(i) += ((w0(m,IVX,k+1,j,i) + w0(m,IVX,k+1,j,i-1)) -
                     (w0(m,IVX,k-1,j,i) + w0(m,IVX,k-1,j,i-1)))/(4.0*size.dx3.d_view(m));
        });
     }

      // Sum viscous fluxes into fluxes of conserved variables; including energy fluxes
      par_for_inner(member, is, ie+1, [&](const int i)
      {
        Real nud = 0.5*nu_iso*(w0(m,IDN,k,j,i) + w0(m,IDN,k,j,i-1));
        flx1(m,IVX,k,j,i) -= nud*fvx(i);
        flx1(m,IVY,k,j,i) -= nud*fvy(i);
        flx1(m,IVZ,k,j,i) -= nud*fvz(i);
        if (flx1.extent_int(1) == static_cast<int>(IEN)) {   // proxy for eos.is_adiabatic
          flx1(m,IEN,k,j,i) -= 0.5*nud*((w0(m,IVX,k,j,i-1) + w0(m,IVX,k,j,i))*fvx(i) +
                                        (w0(m,IVY,k,j,i-1) + w0(m,IVY,k,j,i))*fvy(i) +
                                        (w0(m,IVZ,k,j,i-1) + w0(m,IVZ,k,j,i))*fvz(i));
        }
      });
    }
  );
  if (!(nx2gt1)) {return;}

  //--------------------------------------------------------------------------------------
  // fluxes in x2-direction

  auto flx2 = flx.x2f;

  par_for_outer("visc2",DevExeSpace(), scr_size, scr_level, 0, nmb1, ks, ke, js, je+1,
    KOKKOS_LAMBDA(TeamMember_t member, const int m, const int k, const int j)
    {
      ScrArray1D<Real> fvx(member.team_scratch(scr_level), ncells1);
      ScrArray1D<Real> fvy(member.team_scratch(scr_level), ncells1);
      ScrArray1D<Real> fvz(member.team_scratch(scr_level), ncells1);

      // Add [(dVx/dy+dVy/dx), 2(dVy/dy)-(2/3)(dVx/dx+dVy/dy), dVz/dy]
      par_for_inner(member, is, ie, [&](const int i)
      {
        fvx(i) = (w0(m,IVX,k,j,i  ) - w0(m,IVX,k,j-1,i  ))/size.dx2.d_view(m) +
                ((w0(m,IVY,k,j,i+1) + w0(m,IVY,k,j-1,i+1)) -
                 (w0(m,IVY,k,j,i-1) + w0(m,IVY,k,j-1,i-1)))/(4.0*size.dx1.d_view(m));
        fvy(i) = (w0(m,IVY,k,j,i) - w0(m,IVY,k,j-1,i))*4.0/(3.0*size.dx2.d_view(m)) -
                ((w0(m,IVX,k,j,i+1) + w0(m,IVX,k,j-1,i+1)) -
                 (w0(m,IVX,k,j,i-1) + w0(m,IVX,k,j-1,i-1)))/(6.0*size.dx1.d_view(m));
        fvz(i) = (w0(m,IVZ,k,j,i  ) - w0(m,IVZ,k,j-1,i  ))/size.dx2.d_view(m);
      });

      // In 3D Add [0, (-2/3)dVz/dz, dVy/dz]
      if (nx3gt1) {
        par_for_inner(member, is, ie, [&](const int i)
        {
          fvy(i) -= ((w0(m,IVZ,k+1,j,i) + w0(m,IVZ,k+1,j-1,i)) -
                     (w0(m,IVZ,k-1,j,i) + w0(m,IVZ,k-1,j-1,i)))/(6.0*size.dx3.d_view(m));
          fvz(i) += ((w0(m,IVY,k+1,j,i) + w0(m,IVY,k+1,j-1,i)) -
                     (w0(m,IVY,k-1,j,i) + w0(m,IVY,k-1,j-1,i)))/(4.0*size.dx3.d_view(m));
        });
     }

      // Sum viscous fluxes into fluxes of conserved variables; including energy fluxes
      par_for_inner(member, is, ie, [&](const int i)
      {
        Real nud = 0.5*nu_iso*(w0(m,IDN,k,j,i) + w0(m,IDN,k,j-1,i));
        flx2(m,IVX,k,j,i) -= nud*fvx(i);
        flx2(m,IVY,k,j,i) -= nud*fvy(i);
        flx2(m,IVZ,k,j,i) -= nud*fvz(i);
        if (flx2.extent_int(1) == static_cast<int>(IEN)) {   // proxy for eos.is_adiabatic
          flx2(m,IEN,k,j,i) -= 0.5*nud*((w0(m,IVX,k,j-1,i) + w0(m,IVX,k,j,i))*fvx(i) +
                                        (w0(m,IVY,k,j-1,i) + w0(m,IVY,k,j,i))*fvy(i) +
                                        (w0(m,IVZ,k,j-1,i) + w0(m,IVZ,k,j,i))*fvz(i));
        }
      });
    }
  );
  if (!(nx3gt1)) {return;}

  //--------------------------------------------------------------------------------------
  // fluxes in x3-direction

  auto flx3 = flx.x3f;

  par_for_outer("visc3",DevExeSpace(), scr_size, scr_level, 0, nmb1, ks, ke+1, js, je,
    KOKKOS_LAMBDA(TeamMember_t member, const int m, const int k, const int j)
    {
      ScrArray1D<Real> fvx(member.team_scratch(scr_level), ncells1);
      ScrArray1D<Real> fvy(member.team_scratch(scr_level), ncells1);
      ScrArray1D<Real> fvz(member.team_scratch(scr_level), ncells1);

      // Add [(dVx/dz+dVz/dx), (dVy/dz+dVz/dy), 2(dVz/dz)-(2/3)(dVx/dx+dVy/dy+dVz/dz)]
      par_for_inner(member, is, ie, [&](const int i)
      {
        fvx(i) = (w0(m,IVX,k,j,i  ) - w0(m,IVX,k-1,j,i  ))/size.dx3.d_view(m) +
                ((w0(m,IVZ,k,j,i+1) + w0(m,IVZ,k-1,j,i+1)) -
                 (w0(m,IVZ,k,j,i-1) + w0(m,IVZ,k-1,j,i-1)))/(4.0*size.dx1.d_view(m));
        fvy(i) = (w0(m,IVY,k,j,i  ) - w0(m,IVY,k-1,j,i  ))/size.dx3.d_view(m) +
                ((w0(m,IVZ,k,j+1,i) + w0(m,IVZ,k-1,j+1,i)) -
                 (w0(m,IVZ,k,j-1,i) + w0(m,IVZ,k-1,j-1,i)))/(4.0*size.dx2.d_view(m));
        fvz(i) = (w0(m,IVZ,k,j,i) - w0(m,IVZ,k-1,j,i))*4.0/(3.0*size.dx3.d_view(m)) -
                ((w0(m,IVX,k,j,i+1) + w0(m,IVX,k-1,j,i+1)) -
                 (w0(m,IVX,k,j,i-1) + w0(m,IVX,k-1,j,i-1)))/(6.0*size.dx1.d_view(m)) -
                ((w0(m,IVY,k,j+1,i) + w0(m,IVY,k-1,j+1,i)) -
                 (w0(m,IVY,k,j-1,i) + w0(m,IVY,k-1,j-1,i)))/(6.0*size.dx2.d_view(m));
      });

      // Sum viscous fluxes into fluxes of conserved variables; including energy fluxes
      par_for_inner(member, is, ie, [&](const int i)
      {
        Real nud = 0.5*nu_iso*(w0(m,IDN,k,j,i) + w0(m,IDN,k-1,j,i));
        flx3(m,IVX,k,j,i) -= nud*fvx(i);
        flx3(m,IVY,k,j,i) -= nud*fvy(i);
        flx3(m,IVZ,k,j,i) -= nud*fvz(i);
        if (flx3.extent_int(1) == static_cast<int>(IEN)) {   // proxy for eos.is_adiabatic
          flx3(m,IEN,k,j,i) -= 0.5*nud*((w0(m,IVX,k-1,j,i) + w0(m,IVX,k,j,i))*fvx(i) +
                                        (w0(m,IVY,k-1,j,i) + w0(m,IVY,k,j,i))*fvy(i) +
                                        (w0(m,IVZ,k-1,j,i) + w0(m,IVZ,k,j,i))*fvz(i));
        }
      });
    }
  );

  return;
}
