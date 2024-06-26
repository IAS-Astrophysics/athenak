//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file shearing_box_srcterms.cpp
//! \brief Implements shearing box source terms.  All functions are members of the
//! SourceTerm class

#include <iostream>
#include <string>

#include "athena.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "srcterms/srcterms.hpp"

//----------------------------------------------------------------------------------------
//! \fn SourceTerms::ShearingBox
//! \brief Shearing box source terms in the momentum and energy equations for Hydro.
//! Note MHD function has same name but different argument list.
//! Note: srcterms must be computed using primitive (w0) and NOT conserved (u0) vars

void SourceTerms::ShearingBox(const DvceArray5D<Real> &w0, const EOS_Data &eos_data,
                              const Real bdt, DvceArray5D<Real> &u0) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  auto three_d_ = pmy_pack->pmesh->three_d;

  if (shearing_box_r_phi || three_d_) {
    Real coef1 = 2.0*bdt*omega0;
    Real coef2 = (2.0-qshear)*bdt*omega0;
    Real qo = qshear*omega0;
    par_for("sbox", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      Real &den = w0(m,IDN,k,j,i);
      Real mom1 = den*w0(m,IVX,k,j,i);
      Real mom2 = den*w0(m,IVY,k,j,i);
      u0(m,IM1,k,j,i) += coef1*mom2;
      u0(m,IM2,k,j,i) -= coef2*mom1;
      if (eos_data.is_ideal) {
        // For more accuracy, better to use flux values
        u0(m,IEN,k,j,i) += bdt*mom1*mom2/den*qo;
      }
    });
  } else {
    Real coef1 = 2.0*bdt*omega0;
    Real coef3 = (2.0-qshear)*bdt*omega0;
    Real qo = qshear*omega0;
    par_for("sbox", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      Real &den = w0(m,IDN,k,j,i);
      Real mom1 = den*w0(m,IVX,k,j,i);
      Real mom3 = den*w0(m,IVZ,k,j,i);
      u0(m,IM1,k,j,i) += coef1*mom3;
      u0(m,IM3,k,j,i) -= coef3*mom1;
      if (eos_data.is_ideal) {
        // For more accuracy, better to use flux values
        u0(m,IEN,k,j,i) += bdt*mom1*mom3/den*qo;
      }
    });
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn SourceTerms::ShearingBox
//! \brief Shearing box source terms in the momentum and energy equations for MHD.
//! Note Hydro function has same name but different argument list.
//! NOTE: srcterms must be computed using primitive (w0) and NOT conserved (u0) vars

void SourceTerms::ShearingBox(const DvceArray5D<Real> &w0, const DvceArray5D<Real> &bcc0,
                        const EOS_Data &eos_data, const Real bdt, DvceArray5D<Real> &u0) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  auto three_d_ = pmy_pack->pmesh->three_d;

  if (shearing_box_r_phi || three_d_) {
    Real coef1 = 2.0*bdt*omega0;
    Real coef2 = (2.0-qshear)*bdt*omega0;
    Real qo = qshear*omega0;
    par_for("sbox", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      Real &den = w0(m,IDN,k,j,i);
      Real mom1 = den*w0(m,IVX,k,j,i);
      Real mom2 = den*w0(m,IVY,k,j,i);
      u0(m,IM1,k,j,i) += coef1*mom2;
      u0(m,IM2,k,j,i) -= coef2*mom1;
      if (eos_data.is_ideal) {
        // For more accuracy, better to use flux values
        u0(m,IEN,k,j,i) += bdt*(mom1*mom2/den-bcc0(m,IBX,k,j,i)*bcc0(m,IBY,k,j,i))*qo;
      }
    });
  } else {
    Real coef1 = 2.0*bdt*omega0;
    Real coef3 = (2.0-qshear)*bdt*omega0;
    Real qo = qshear*omega0;
    par_for("sbox", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      Real &den = w0(m,IDN,k,j,i);
      Real mom1 = den*w0(m,IVX,k,j,i);
      Real mom3 = den*w0(m,IVZ,k,j,i);
      u0(m,IM1,k,j,i) += coef1*mom3;
      u0(m,IM3,k,j,i) -= coef3*mom1;
      if (eos_data.is_ideal) {
        // For more accuracy, better to use flux values
        u0(m,IEN,k,j,i) += bdt*(mom1*mom3/den-bcc0(m,IBX,k,j,i)*bcc0(m,IBZ,k,j,i))*qo;
      }
    });
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn SourceTerms::AddSBoxEField
//  \brief Add electric field in rotating frame E = - (v_{K} x B) where v_{K} is
//  background orbital velocity v_{K} = - q \Omega x in the toriodal (\phi or y) direction
//  See SG eqs. [49-52] (eqs for orbital advection), and [60]

void SourceTerms::SBoxEField(const DvceFaceFld4D<Real> &b0, DvceEdgeFld4D<Real> &efld) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  auto &size = pmy_pack->pmb->mb_size;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  Real qomega = qshear*omega0;

  int nmb1 = pmy_pack->nmb_thispack - 1;
  size_t scr_size = 0;
  int scr_level = 0;

  //---- 2-D problem:
  // electric field E = - (v_{K} x B), where v_{K} is in the z-direction.  Thus
  // E_{x} = -(v x B)_{x} = -(vy*bz - vz*by) = +v_{K}by --> E1 = -(q\Omega x)b2
  // E_{y} = -(v x B)_{y} =  (vx*bz - vz*bx) = -v_{K}bx --> E2 = +(q\Omega x)b1
  if (pmy_pack->pmesh->two_d) {
    auto e1 = efld.x1e;
    auto e2 = efld.x2e;
    auto b1 = b0.x1f;
    auto b2 = b0.x2f;
    par_for_outer("acc0", DevExeSpace(), scr_size, scr_level, 0, nmb1, js, je+1,
    KOKKOS_LAMBDA(TeamMember_t member, const int m, const int j) {
      par_for_inner(member, is, ie+1, [&](const int i) {
        Real &x1min = size.d_view(m).x1min;
        Real &x1max = size.d_view(m).x1max;
        int nx1 = indcs.nx1;
        Real x1v = CellCenterX(i-is, nx1, x1min, x1max);

        e1(m,ks,  j,i) -= qomega*x1v*b2(m,ks,j,i);
        e1(m,ke+1,j,i) -= qomega*x1v*b2(m,ks,j,i);

        Real x1f = LeftEdgeX(i-is, nx1, x1min, x1max);
        e2(m,ks  ,j,i) += qomega*x1f*b1(m,ks,j,i);
        e2(m,ke+1,j,i) += qomega*x1f*b1(m,ks,j,i);
      });
    });
  }
  // TODO(@user): add 3D shearing box

  return;
}
