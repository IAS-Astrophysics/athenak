//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file ideal_mhd.cpp
//  \brief derived class that implements ideal gas EOS in nonrelativistic mhd

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "mhd/mhd.hpp"
#include "eos.hpp"

//----------------------------------------------------------------------------------------
// ctor: also calls EOS base class constructor

IdealMHD::IdealMHD(MeshBlockPack *pp, ParameterInput *pin)
  : EquationOfState(pp, pin)
{
  eos_data.is_ideal = true;
  eos_data.gamma = pin->GetReal("eos","gamma");
  eos_data.iso_cs = 0.0;
}

//----------------------------------------------------------------------------------------
// \!fn void ConsToPrim()
// \brief Converts conserved into primitive variables.  Operates over entire MeshBlock,
//  including ghost cells.  
// Note that the primitive variables contain the cell-centered magnetic fields, so that
// W contains (nmhd+3+nscalars) elements, while U contains (nmhd+nscalars)

void IdealMHD::ConsToPrim(DvceArray5D<Real> &cons, const DvceFaceFld4D<Real> &b,
                              DvceArray5D<Real> &prim, DvceArray5D<Real> &bcc)
{
  auto &indcs = pmy_pack->coord.coord_data.mb_indcs;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng) : 1;
  int &nmhd  = pmy_pack->pmhd->nmhd;
  int &nscal = pmy_pack->pmhd->nscalars;
  int &nmb = pmy_pack->nmb_thispack;
  Real gm1 = eos_data.gamma - 1.0;

  Real &dfloor_ = eos_data.density_floor;
  Real &pfloor_ = eos_data.pressure_floor;

  par_for("mhd_con2prim", DevExeSpace(), 0, (nmb-1), 0, (n3-1), 0, (n2-1), 0, (n1-1),
    KOKKOS_LAMBDA(int m, int k, int j, int i)
    {
      Real& u_d  = cons(m,IDN,k,j,i);
      Real& u_e  = cons(m,IEN,k,j,i);
      const Real& u_m1 = cons(m,IVX,k,j,i);
      const Real& u_m2 = cons(m,IVY,k,j,i);
      const Real& u_m3 = cons(m,IVZ,k,j,i);

      Real& w_d  = prim(m,IDN,k,j,i);
      Real& w_p  = prim(m,IPR,k,j,i);
      Real& w_vx = prim(m,IVX,k,j,i);
      Real& w_vy = prim(m,IVY,k,j,i);
      Real& w_vz = prim(m,IVZ,k,j,i);

      // apply density floor, without changing momentum or energy
      u_d = (u_d > dfloor_) ?  u_d : dfloor_;
      w_d = u_d;

      Real di = 1.0/u_d;
      w_vx = u_m1*di;
      w_vy = u_m2*di;
      w_vz = u_m3*di;

      // cell-centered fields are simple linear average of face-centered fields
      Real& w_bx = bcc(m,IBX,k,j,i);
      Real& w_by = bcc(m,IBY,k,j,i);
      Real& w_bz = bcc(m,IBZ,k,j,i);
      w_bx = 0.5*(b.x1f(m,k,j,i) + b.x1f(m,k,j,i+1));  
      w_by = 0.5*(b.x2f(m,k,j,i) + b.x2f(m,k,j+1,i));
      w_bz = 0.5*(b.x3f(m,k,j,i) + b.x3f(m,k+1,j,i));

      // pressure computed from linear average of magnetic energy of face-centered fields
      Real pb = 0.25*(( SQR(b.x1f(m,k,j,i)) + SQR(b.x1f(m,k,j,i+1)) )
                   +  ( SQR(b.x2f(m,k,j,i)) + SQR(b.x2f(m,k,j+1,i)) )
                   +  ( SQR(b.x3f(m,k,j,i)) + SQR(b.x3f(m,k+1,j,i)) ));

      Real e_k = 0.5*di*(SQR(u_m1) + SQR(u_m2) + SQR(u_m3));
      w_p = gm1*(u_e - e_k - pb);

      // apply pressure floor, correct total energy
      u_e = (w_p > pfloor_) ?  u_e : ((pfloor_/gm1) + e_k + pb);
      w_p = (w_p > pfloor_) ?  w_p : pfloor_;

      // convert scalars (if any), always stored at end of cons and prim arrays.
      for (int n=nmhd; n<(nmhd+nscal); ++n) {
        prim(m,n,k,j,i) = cons(m,n,k,j,i)*di;
      }
    }
  );

  return;
}

//----------------------------------------------------------------------------------------
// \!fn void PrimToCons()
// \brief Converts conserved into primitive variables.  Operates over only active cells.
//  Does not change cell- or face-centered magnetic fields.

void IdealMHD::PrimToCons(const DvceArray5D<Real> &prim, const DvceArray5D<Real> &bcc,
                              DvceArray5D<Real> &cons)
{
  auto &indcs = pmy_pack->coord.coord_data.mb_indcs;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  int &nmhd  = pmy_pack->pmhd->nmhd;
  int &nscal = pmy_pack->pmhd->nscalars;
  int &nmb = pmy_pack->nmb_thispack;
  Real igm1 = 1.0/(eos_data.gamma - 1.0);
  
  par_for("mhd_prim2con", DevExeSpace(), 0, (nmb-1), ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i)
    { 
      Real& u_d  = cons(m,IDN,k,j,i);
      Real& u_e  = cons(m,IEN,k,j,i);
      Real& u_m1 = cons(m,IVX,k,j,i);
      Real& u_m2 = cons(m,IVY,k,j,i);
      Real& u_m3 = cons(m,IVZ,k,j,i);
      
      const Real& w_d  = prim(m,IDN,k,j,i);
      const Real& w_p  = prim(m,IPR,k,j,i);
      const Real& w_vx = prim(m,IVX,k,j,i);
      const Real& w_vy = prim(m,IVY,k,j,i);
      const Real& w_vz = prim(m,IVZ,k,j,i);

      const Real& bcc1 = bcc(m,IBX,k,j,i);
      const Real& bcc2 = bcc(m,IBY,k,j,i);
      const Real& bcc3 = bcc(m,IBZ,k,j,i);

      u_d = w_d;
      u_m1 = w_vx*w_d;
      u_m2 = w_vy*w_d;
      u_m3 = w_vz*w_d;
      u_e = w_p*igm1 + 0.5*(w_d*(SQR(w_vx) + SQR(w_vy) + SQR(w_vz))
                              + (SQR(bcc1) + SQR(bcc2) + SQR(bcc3)));
      
      // convert scalars (if any), always stored at end of cons and prim arrays.
      for (int n=nmhd; n<(nmhd+nscal); ++n) {
        cons(m,n,k,j,i) = prim(m,n,k,j,i)*w_d;
      }
    } 
  );

  return;
}
