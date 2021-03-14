//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file shearing_box.cpp
//  \brief source terms in shearing box approximation in 2D

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "driver/driver.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "utils/grid_locations.hpp"
#include "srcterms.hpp"

//----------------------------------------------------------------------------------------
//! \fn

TaskStatus SourceTerms::HydroShearingBox(Driver *pdrive, int stage)
{
  Real beta_dt = (pdrive->beta[stage-1])*(pmy_pack->pmesh->dt);
  auto &eos = pmy_pack->phydro->peos->eos_data;
  ShearingBox(pmy_pack->phydro->u0, pmy_pack->phydro->w0, eos, beta_dt);
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn

TaskStatus SourceTerms::MHDShearingBox(Driver *pdrive, int stage)
{
  Real beta_dt = (pdrive->beta[stage-1])*(pmy_pack->pmesh->dt);
  auto &eos = pmy_pack->pmhd->peos->eos_data;
  ShearingBox(pmy_pack->pmhd->u0, pmy_pack->pmhd->w0, eos, beta_dt);
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn ConstantAccel

void SourceTerms::ShearingBox(DvceArray5D<Real> &u, DvceArray5D<Real> &w,
                                const EOS_Data &eos, Real bdt)
{
  int is = pmy_pack->mb_cells.is; int ie = pmy_pack->mb_cells.ie;
  int js = pmy_pack->mb_cells.js; int je = pmy_pack->mb_cells.je;
  int ks = pmy_pack->mb_cells.ks; int ke = pmy_pack->mb_cells.ke;
  int &nx1 = pmy_pack->mb_cells.nx1;

  Real &omega0_ = omega0;
  Real qo2  = qshear*SQR(omega0);

  int nmb1 = pmy_pack->nmb_thispack - 1;
  size_t scr_size = 0; 
  int scr_level = 0;
  auto &size = pmy_pack->pmb->mbsize;
  par_for_outer("acc0", DevExeSpace(), scr_size, scr_level, 0, nmb1, ks, ke, js, je,
    KOKKOS_LAMBDA(TeamMember_t member, const int m, const int k, const int j)
    {
      par_for_inner(member, is, ie, [&](const int i)
      {  
        Real x1v = CellCenterX(i-is, nx1, size.x1min.d_view(m), size.x1max.d_view(m));
        Real &den = w(m,IDN,k,j,i);
        Real mom1 = den*w(m,IVX,k,j,i);
        Real mom3 = den*w(m,IVZ,k,j,i);
        u(m,IM1,k,j,i) += 2.0*bdt*(omega0_*mom3 + qo2*den*x1v);
        u(m,IM3,k,j,i) -= 2.0*bdt*omega0_*mom1;
        if (eos.is_adiabatic) {u(m,IEN,k,j,i) += 2.0*bdt*qo2*mom1*x1v;}
      });
    }
  );

  return;
}
