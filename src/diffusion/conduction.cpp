//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file conduction.cpp
//! \brief Implements functions for Conduction class. This includes isotropic thermal
//! conduction, in which heat flux is proportional to negative local temperature gradient.
//! Conduction may be added to Hydro and/or MHD independently.

// Athena++ headers
#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "eos/eos.hpp"
#include "conduction.hpp"

//----------------------------------------------------------------------------------------
//! \brief Conduction constructor

Conduction::Conduction(std::string block, MeshBlockPack *pp, ParameterInput *pin)
  : pmy_pack(pp)
{
  // Check that EOS is ideal
  if (pmy_pack->phydro != nullptr) {
    const bool &is_ideal = pmy_pack->phydro->peos->eos_data.is_ideal;
    if (is_ideal == false) {
      std::cout << "### FATAL ERROR in "<< __FILE__ <<" at line " << __LINE__ << std::endl
                << "Thermal conduction only works for ideal gas" << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
  if (pmy_pack->pmhd != nullptr) {
    const bool &is_ideal = pmy_pack->pmhd->peos->eos_data.is_ideal;
    if (is_ideal == false) {
      std::cout << "### FATAL ERROR in "<< __FILE__ <<" at line " << __LINE__ << std::endl
                << "Thermal conduction only works for ideal gas" << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }

  // Read thermal conductivity of isotropic thermal conduction
  kappa = pin->GetReal(block,"conductivity");
  
  // timestep for thermal conduction on MeshBlock(s) in this pack
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
    dtnew = std::min(dtnew, fac*SQR(size.h_view(m).dx1)/kappa);
    if (pmy_pack->pmesh->multi_d) {
      dtnew = std::min(dtnew, fac*SQR(size.h_view(m).dx2)/kappa);
    }
    if (pmy_pack->pmesh->three_d) {
      dtnew = std::min(dtnew, fac*SQR(size.h_view(m).dx3)/kappa);
    }
  }
}

//----------------------------------------------------------------------------------------
//! \brief Conduction destructor

Conduction::~Conduction()
{
}

//----------------------------------------------------------------------------------------
//! \fn void IsotropicHeatFlux()
//! \brief Adds isotropic heat flux to face-centered fluxes of conserved variables

void Conduction::IsotropicHeatFlux(const DvceArray5D<Real> &w0, const Real kappa,
  const EOS_Data &eos, DvceFaceFld5D<Real> &flx)
{
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int ncells1 = indcs.nx1 + 2*(indcs.ng);
  int nmb1 = pmy_pack->nmb_thispack - 1;
  auto size = pmy_pack->pmb->mb_size;
  const bool &use_e = eos.use_e;
  Real gm1 = eos.gamma-1.0;

  //--------------------------------------------------------------------------------------
  // fluxes in x1-direction

  int scr_level = 0;
  size_t scr_size = (ScrArray1D<Real>::shmem_size(ncells1)) * 3;
  auto flx1 = flx.x1f;

  par_for_outer("conduct1", DevExeSpace(), scr_size, scr_level, 0, nmb1, ks, ke, js, je,
    KOKKOS_LAMBDA(TeamMember_t member, const int m, const int k, const int j)
    {
      ScrArray1D<Real> hflx1(member.team_scratch(scr_level), ncells1);
      
      // Add heat fluxes into fluxes of conserved variables: energy
      par_for_inner(member, is, ie+1, [&](const int i)
      {
        if (use_e) {
          hflx1(i) = gm1 * (w0(m,IEN,k,j,i)/w0(m,IDN,k,j,i) - 
                     w0(m,IEN,k,j,i-1)/w0(m,IDN,k,j,i-1)) / size.d_view(m).dx1;
                     
        } else {
          hflx1(i) = (w0(m,ITM,k,j,i) - w0(m,ITM,k,j,i-1)) / size.d_view(m).dx1;
        }
        flx1(m,IEN,k,j,i) -= kappa * hflx1(i);
      });
    }
  );
  if (pmy_pack->pmesh->one_d) {return;}

  //--------------------------------------------------------------------------------------
  // fluxes in x2-direction

  auto flx2 = flx.x2f;

  par_for_outer("conduct2",DevExeSpace(), scr_size, scr_level, 0, nmb1, ks, ke, js, je+1,
    KOKKOS_LAMBDA(TeamMember_t member, const int m, const int k, const int j)
    {
      ScrArray1D<Real> hflx2(member.team_scratch(scr_level), ncells1);
      
      // Add heat fluxes into fluxes of conserved variables: energy
      par_for_inner(member, is, ie, [&](const int i)
      {
        if (use_e) {
          hflx2(i) = gm1 * (w0(m,IEN,k,j,i)/w0(m,IDN,k,j,i) - 
                     w0(m,IEN,k,j-1,i)/w0(m,IDN,k,j-1,i)) / size.d_view(m).dx2;
        } else {
          hflx2(i) = (w0(m,ITM,k,j,i) - w0(m,ITM,k,j-1,i)) / size.d_view(m).dx2;
        }
        flx2(m,IEN,k,j,i) -= kappa * hflx2(i);
      });
    }
  );
  if (pmy_pack->pmesh->two_d) {return;}

  //--------------------------------------------------------------------------------------
  // fluxes in x3-direction

  auto flx3 = flx.x3f;

  par_for_outer("conduct3",DevExeSpace(), scr_size, scr_level, 0, nmb1, ks, ke+1, js, je,
    KOKKOS_LAMBDA(TeamMember_t member, const int m, const int k, const int j)
    {
      ScrArray1D<Real> hflx3(member.team_scratch(scr_level), ncells1);

      // Add heat fluxes into fluxes of conserved variables: energy
      par_for_inner(member, is, ie, [&](const int i)
      {
        if (use_e) {
          hflx3(i) = gm1 * (w0(m,IEN,k,j,i)/w0(m,IDN,k,j,i) - 
                     w0(m,IEN,k-1,j,i)/w0(m,IDN,k-1,j,i)) / size.d_view(m).dx3;
        } else {
          hflx3(i) = (w0(m,ITM,k,j,i) - w0(m,ITM,k-1,j,i)) / size.d_view(m).dx3;
        }
        flx3(m,IEN,k,j,i) -= kappa * hflx3(i);
      });
    }
  );

  return;
}
