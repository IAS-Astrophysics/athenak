//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file constant_acc.cpp
//  \brief source terms due to constant acceleration (e.g. for RT instability)

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "driver/driver.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "srcterms.hpp"

//----------------------------------------------------------------------------------------
//! \fn

TaskStatus SourceTerms::HydroConstantAccel(Driver *pdrive, int stage)
{
  Real beta_dt = (pdrive->beta[stage-1])*(pmy_pack->pmesh->dt);
  auto &eos = pmy_pack->phydro->peos->eos_data;
  ConstantAccel(pmy_pack->phydro->u0, pmy_pack->phydro->w0, eos, beta_dt);
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn

TaskStatus SourceTerms::MHDConstantAccel(Driver *pdrive, int stage)
{
  Real beta_dt = (pdrive->beta[stage-1])*(pmy_pack->pmesh->dt);
  auto &eos = pmy_pack->pmhd->peos->eos_data;
  ConstantAccel(pmy_pack->pmhd->u0, pmy_pack->pmhd->w0, eos, beta_dt);
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn ConstantAccel

void SourceTerms::ConstantAccel(DvceArray5D<Real> &u, DvceArray5D<Real> &w,
                                const EOS_Data &eos, Real bdt)
{
  int is = pmy_pack->mb_cells.is; int ie = pmy_pack->mb_cells.ie;
  int js = pmy_pack->mb_cells.js; int je = pmy_pack->mb_cells.je;
  int ks = pmy_pack->mb_cells.ks; int ke = pmy_pack->mb_cells.ke;

  Real &g1 = const_acc1;
  Real &g2 = const_acc2;
  Real &g3 = const_acc3;

  int nmb1 = pmy_pack->nmb_thispack - 1;
  size_t scr_size = 0;
  int scr_level = 0;
  par_for_outer("acc0", DevExeSpace(), scr_size, scr_level, 0, nmb1, ks, ke, js, je,
    KOKKOS_LAMBDA(TeamMember_t member, const int m, const int k, const int j)
    {
      // acceleration in 1-direction
      if (g1 != 0.0) {
        par_for_inner(member, is, ie, [&](const int i)
        {
          Real src = bdt*g1*w(m,IDN,k,j,i);
          u(m,IM1,k,j,i) += src;
          if (eos.is_adiabatic) {
            u(m,IEN,k,j,i) += src*w(m,IVX,k,j,i);
          }
        });
      }
      // acceleration in 2-direction
      if (g2 != 0.0) {
        par_for_inner(member, is, ie, [&](const int i)
        {
          Real src = bdt*g2*w(m,IDN,k,j,i);
          u(m,IM2,k,j,i) += src;
          if (eos.is_adiabatic) {
            u(m,IEN,k,j,i) += src*w(m,IVY,k,j,i);
          }
        });
      }
      // acceleration in 3-direction
      if (g3 != 0.0) {
        par_for_inner(member, is, ie, [&](const int i)
        {
          Real src = bdt*g3*w(m,IDN,k,j,i);
          u(m,IM3,k,j,i) += src;
          if (eos.is_adiabatic) {
            u(m,IEN,k,j,i) += src*w(m,IVZ,k,j,i);
          }
        });
      }
    }
  );

  return;
}
