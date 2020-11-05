//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file isothermal_mhd.cpp
//  \brief implements EOS functions in derived class for nonrelativistic isothermal MHD

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "mhd/mhd.hpp"
#include "eos.hpp"

//----------------------------------------------------------------------------------------
// \!fn void ConservedToPrimitive()
// \brief Converts conserved into primitive variables in nonrelativistic isothermal MHD
// Nate that the primitive variables contain the cell-centered magnetic fields, so that
// W contains (nhydro+3+nscalars) elements, while U contains (nhydro+nscalars)

void EquationOfState::ConsToPrimIsoMHD(const AthenaArray4D<Real> &cons,
  AthenaArray4D<Real> &prim, const FaceArray3D<Real> &bfc,  AthenaArray4D<Real> &bcc)
{
  MeshBlock* pmb = pmesh_->FindMeshBlock(my_mbgid_);
  int ng = pmb->mb_cells.ng;
  int ncells1 = pmb->mb_cells.nx1 + 2*ng;
  int ncells2 = (pmb->mb_cells.nx2 > 1)? (pmb->mb_cells.nx2 + 2*ng) : 1;
  int ncells3 = (pmb->mb_cells.nx3 > 1)? (pmb->mb_cells.nx3 + 2*ng) : 1;
  int &nhydro = pmb->pmhd->nhydro;
  int &nscalars = pmb->pmhd->nscalars;
  Real &dfloor_ = eos_data.density_floor;

  par_for("mhd_con2prim", pmb->exe_space, 0, (ncells3-1), 0, (ncells2-1), 0, (ncells1-1),
    KOKKOS_LAMBDA(int k, int j, int i)
    {
      Real& u_d  = cons(hydro::IDN,k,j,i);
      Real& u_m1 = cons(hydro::IVX,k,j,i);
      Real& u_m2 = cons(hydro::IVY,k,j,i);
      Real& u_m3 = cons(hydro::IVZ,k,j,i);

      Real& w_d  = prim(hydro::IDN,k,j,i);
      Real& w_vx = prim(hydro::IVX,k,j,i);
      Real& w_vy = prim(hydro::IVY,k,j,i);
      Real& w_vz = prim(hydro::IVZ,k,j,i);

      // apply density floor, without changing momentum or energy
      u_d = (u_d > dfloor_) ?  u_d : dfloor_;
      w_d = u_d;

      Real di = 1.0/u_d;
      w_vx = u_m1*di;
      w_vy = u_m2*di;
      w_vz = u_m3*di;

      Real& w_bx = bcc(mhd::IBX,k,j,i);
      Real& w_by = bcc(mhd::IBY,k,j,i);
      Real& w_bz = bcc(mhd::IBZ,k,j,i);
      w_bx = 0.5*(bfc.x1f(k,j,i) + bfc.x1f(k,j,i+1));  
      w_by = 0.5*(bfc.x2f(k,j,i) + bfc.x2f(k,j+1,i));
      w_bz = 0.5*(bfc.x3f(k,j,i) + bfc.x3f(k+1,j,i));

      // convert scalars (if any), always stored at end of cons and prim arrays.
      for (int n=nhydro; n<(nhydro+nscalars); ++n) {
        prim(n,k,j,i) = cons(n,k,j,i)/u_d;
      }
    }
  );

  return;
}
