//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file shearing_box.cpp
//  \brief source terms in shearing box approximation in 2D (x-z plane) using orbital
//  advection.
//
//  With orbital advection the equations of motion solve for the velocity perturbations
//  from a background shearing flow v_{K} = -q \Omega x. That is, the quantity
//  V' = V - v_{K} is evolved. The shearing box maps cylindrical (r, \phi, z) coordinates
//  to a locally Cartesian (x,y,z) frame. Thus, v_{K} is in the y-dir
//
//  In 2D, the x1-x2 coordinates represent the r-z (poloidal) plane, and the x3-components
//  of vectors represent the azimuthal (\phi, or y) direction.
//
//  REFERENCE: "Implementation of the Shearing Box Approximation in Athena", by J.M.Stone
//  & T.A. Gardiner, ApJS 189, 142 (2010).

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "driver/driver.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "utils/grid_locations.hpp"
#include "srcterms.hpp"

//----------------------------------------------------------------------------------------
//! \fn SourceTerms::AddSBoxMomentumHydro
//! \brief calls function that adds shearing box source terms to the momentum and energy
//! equations for Hydro. This function is inserted into TaskList.

TaskStatus SourceTerms::AddSBoxMomentumHydro(Driver *pdrive, int stage)
{
  Real beta_dt = (pdrive->beta[stage-1])*(pmy_pack->pmesh->dt);
  auto &eos = pmy_pack->phydro->peos->eos_data;
  SBoxMomentumTerms(pmy_pack->phydro->u0, pmy_pack->phydro->w0, eos, beta_dt);
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn SourceTerms::AddSBoxMomentumMHD
//! \brief calls function that adds shearing box source terms to the momentum and energy
//! equations for MHD. This function is inserted into TaskList.

TaskStatus SourceTerms::AddSBoxMomentumMHD(Driver *pdrive, int stage)
{
  Real beta_dt = (pdrive->beta[stage-1])*(pmy_pack->pmesh->dt);
  auto &eos = pmy_pack->pmhd->peos->eos_data;
  auto pmhd = pmy_pack->pmhd;
  SBoxMomentumTerms(pmhd->u0, pmhd->w0, pmhd->b0, eos, beta_dt);
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn SourceTerms::AddSBoxEMF
//! \brief calls function that adds shearing box EMF when orbital advection is used to
//! the corner-centered electric fields. This function is inserted into TaskList.

TaskStatus SourceTerms::AddSBoxEMF(Driver *pdrive, int stage)
{
  SBoxEMF(pmy_pack->pmhd->b0, pmy_pack->pmhd->efld);
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn SourceTerms::SBoxMomentumTerms
//! \brief Shearing box source terms in the momentum and energy equations for Hydro.
//  Terms are implemented with orbital advection, so that v3 represents the perturbation
//  from the Keplerian flow v_{K} = - q \Omega x

void SourceTerms::SBoxMomentumTerms(DvceArray5D<Real> &u, DvceArray5D<Real> &w,
                                    const EOS_Data &eos, Real bdt)
{ 
  int is = pmy_pack->mb_cells.is; int ie = pmy_pack->mb_cells.ie;
  int js = pmy_pack->mb_cells.js; int je = pmy_pack->mb_cells.je;
  int ks = pmy_pack->mb_cells.ks; int ke = pmy_pack->mb_cells.ke;
  
  Real &omega0_ = omega0;
  Real &qshear_ = qshear;
  Real qo  = qshear*omega0;
  
  int nmb1 = pmy_pack->nmb_thispack - 1;
  size_t scr_size = 0;
  int scr_level = 0;
  par_for_outer("acc0", DevExeSpace(), scr_size, scr_level, 0, nmb1, ks, ke, js, je,
    KOKKOS_LAMBDA(TeamMember_t member, const int m, const int k, const int j)
    { 
      par_for_inner(member, is, ie, [&](const int i)
      {  
        Real &den = w(m,IDN,k,j,i);
        Real mom1 = den*w(m,IVX,k,j,i);
        Real mom3 = den*w(m,IVZ,k,j,i);
        u(m,IM1,k,j,i) += 2.0*bdt*(omega0_*mom3);
        u(m,IM3,k,j,i) += (qshear_ - 2.0)*bdt*omega0_*mom1;
        if (eos.is_adiabatic) { u(m,IEN,k,j,i) += qo*bdt*(mom1*mom3/den); }
      });
    }
  );
  
  return;
}

//----------------------------------------------------------------------------------------
//! \fn SourceTerms::SBoxMomentumTerms
//! \brief Shearing box source terms in the momentum and energy equations for MHD.
//  Terms are implemented with orbital advection, so that v3 represents the perturbation
//  from the Keplerian flow v_{K} = - q \Omega x

void SourceTerms::SBoxMomentumTerms(DvceArray5D<Real> &u, DvceArray5D<Real> &w,
                                DvceFaceFld4D<Real> &b, const EOS_Data &eos, Real bdt)
{
  int is = pmy_pack->mb_cells.is; int ie = pmy_pack->mb_cells.ie;
  int js = pmy_pack->mb_cells.js; int je = pmy_pack->mb_cells.je;
  int ks = pmy_pack->mb_cells.ks; int ke = pmy_pack->mb_cells.ke;

  Real &omega0_ = omega0;
  Real &qshear_ = qshear;
  Real qo  = qshear*omega0;

  int nmb1 = pmy_pack->nmb_thispack - 1;
  size_t scr_size = 0; 
  int scr_level = 0;
  auto &b1 = b.x1f;
  auto &b3 = b.x3f;
  par_for_outer("acc0", DevExeSpace(), scr_size, scr_level, 0, nmb1, ks, ke, js, je,
    KOKKOS_LAMBDA(TeamMember_t member, const int m, const int k, const int j)
    {
      par_for_inner(member, is, ie, [&](const int i)
      {  
        Real &den = w(m,IDN,k,j,i);
        Real mom1 = den*w(m,IVX,k,j,i);
        Real mom3 = den*w(m,IVZ,k,j,i);
        u(m,IM1,k,j,i) += 2.0*bdt*(omega0_*mom3);
        u(m,IM3,k,j,i) += (qshear_ - 2.0)*bdt*omega0_*mom1;
        if (eos.is_adiabatic) {
          u(m,IEN,k,j,i) -= qo*bdt*(b1(m,k,j,i)*b3(m,k,j,i) - mom1*mom3/den);
        }
      });
    }
  );

  return;
}

//----------------------------------------------------------------------------------------
//! \fn SourceTerms::SBoxEMF
//  \brief Add electric field in rotating frame E = - (v_{K} x B) where v_{K} is
//  background orbital velocity v_{K} = - q \Omega x in the toriodal (\phi or y) direction
//  See SG eqs. [49-52] (eqs for orbital advection), and [60]
  
void SourceTerms::SBoxEMF(const DvceFaceFld4D<Real> &b0, DvceEdgeFld4D<Real> &efld)
{ 
  int is = pmy_pack->mb_cells.is; int ie = pmy_pack->mb_cells.ie;
  int js = pmy_pack->mb_cells.js; int je = pmy_pack->mb_cells.je;
  int ks = pmy_pack->mb_cells.ks; int ke = pmy_pack->mb_cells.ke;
  int &nx1 = pmy_pack->mb_cells.nx1;
    
  Real qomega  = qshear*omega0;
      
  int nmb1 = pmy_pack->nmb_thispack - 1;
  size_t scr_size = 0; 
  int scr_level = 0;

  //---- 2-D problem:
  // electric field E = - (v_{K} x B), where v_{K} is in the z-direction.  Thus
  // E_{x} = -(v x B)_{x} = -(vy*bz - vz*by) = +v_{K}by --> E1 = -(q\Omega x)b2
  // E_{y} = -(v x B)_{y} =  (vx*bz - vz*bx) = -v_{K}bx --> E2 = +(q\Omega x)b1
  if (!(pmy_pack->pmesh->nx3gt1)) {
    auto &size = pmy_pack->pmb->mbsize;
    auto e1 = efld.x1e;
    auto e2 = efld.x2e;
    auto b1 = b0.x1f;
    auto b2 = b0.x2f;
    par_for_outer("acc0", DevExeSpace(), scr_size, scr_level, 0, nmb1, js, je+1,
      KOKKOS_LAMBDA(TeamMember_t member, const int m, const int j)
      {   
        par_for_inner(member, is, ie+1, [&](const int i)
        {  
          Real x1v = CellCenterX(i-is, nx1, size.x1min.d_view(m), size.x1max.d_view(m));
          e1(m,ks,  j,i) -= qomega*x1v*b2(m,ks,j,i);
          e1(m,ke+1,j,i) -= qomega*x1v*b2(m,ks,j,i);
          Real x1f = LeftEdgeX(i-is, nx1, size.x1min.d_view(m), size.x1max.d_view(m));
          e2(m,ks  ,j,i) += qomega*x1f*b1(m,ks,j,i);
          e2(m,ke+1,j,i) += qomega*x1f*b1(m,ks,j,i);
        });
      }
    );
  }

  return;
}
