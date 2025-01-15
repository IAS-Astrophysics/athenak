//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file cbd.cpp
//! \brief Problem generator to initialize circumbinary accretion disks 
//!        around binaries, with or without magnetic fields. Written by MSS.


// include headers
#include <stdio.h>
#include <math.h>
#include "pgen.hpp"

// ==================================================================
// ====Define the basic properties of the binary in this namespace===
// ======CONSIDER MAKING THIS A STRUCTURE INSTEAD OF A NAMESPACE!====
// ==================================================================
namespace Binary 
{
  mb = pin->GetReal("binary","mb"); // binary mass (central mass)
  ab = pin->GetReal("binary","ab"); // binary semi-major axis
  eb = pin->GetReal("binary","eb"); // binary eccentricity
  qb = pin->GetReal("binary","qb"); // binary mass ratio
} //namespace Binary
// ==================================================================
// ===================Access these variables:========================
// ======using namespace Binary; (do something with variables)=======
// ==================================================================




//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::UserProblem()
//! \brief This function sets the initial conditions for the CBD problem.
//!        From Athena++ documentation:
//!        For hydrodynamics, the cell-centered conservative variables (phydro->u)
//!        must be set here. The face-centered magnetic fields
//!        (pfield->bx1f, bx2f, and bx3f) also must be set for MHD. 
//!        Primitive variables and cell-centered magnetic fields are 
//!        automatically derived by the code (though with GR the primitives must also be set)
//! Compile athenak with '-D PROBLEM=cbd' to enroll as user-specific problem generator

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
    MeshBlockPack *pmbp = pmy_mesh_->pmb_pack; //not sure what exactly this does yet?

    // capture variables for kernel
    auto &indcs = pmy_mesh_->mb_indcs; // mesh block indices
    int is = indcs.is, js = indcs.js, ks = indcs.ks; 
    int ie = indcs.ie, je = indcs.je, ke = indcs.ke;
    int nmb = pmbp->nmb_thispack; // number of mesh blocks (?)
    auto &coord = pmbp->pcoord->coord_data;
    auto &size = pmbp->pmb->mb_size; //MeshBlock size


    // ==================================================================
    // ------------------BEGIN INITIALIZE (M)HYDRODYNAMICS---------------
    // ==================================================================
    // pmbp->phydro exists if there is a block <hydro> in .athinput file,
    // otherwise, phydro is set to null pointer (nullptr).
    // phydro and pmhd modules created in src/mesh/meshblock_pack.cpp
    // Select either Hydro or MHD
    DvceArray5D<Real> u0_, w0_;
    if (pmbp->phydro != nullptr) {
      u0_ = pmbp->phydro->u0;
      w0_ = pmbp->phydro->w0;
    } else if (pmbp->pmhd != nullptr) {
      u0_ = pmbp->pmhd->u0;
      w0_ = pmbp->pmhd->w0;
    }
    // ==================================================================
    // ------------------END INITIALIZE (M)HYDRODYNAMICS-----------------
    // ==================================================================


    // par_for is a wrapper for Kokkos::parallel_for.
    // parameters: 
    // name: user provided, for profiling and debugging,
    // ExecPolicy: "defines iteration space and other execution properties"
    // const int &il (0): lower index of mesh blocks
    // const int &iu (nmb-1): upper index of mesh blocks
    // ============> const int ni = iu - il + 1; 
    // ============> ni is the total number of mesh blocks over which we loop
    // ks,ke,js,je,is,ie: z, y, x indices of the first and last cell in each mesh block
    // KOKKOS_LAMBDA(int m, int k, int j, int i){...} is the function that is executed
    par_for("pgen_cbd_for1", DevExeSpace(), 0, nmb-1,
            ks,ke,js,je,is,ie,
            KOKKOS_LAMBDA(int m, int k, int j, int i) 
            {
              Real &x1min = size.d_view(m).x1min;
              Real &x1max = size.d_view(m).x1max;
              Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

              Real &x2min = size.d_view(m).x2min;
              Real &x2max = size.d_view(m).x2max;
              Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

              Real &x3min = size.d_view(m).x3min;
              Real &x3max = size.d_view(m).x3max;
              Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);

              using namespace cbd_isothermal;
              // assign 

            });
    // ==================================================================
    // =================END OF par_for loop==============================
    // ==================================================================

    // Conserved variables
    // IDM: index of density
    // IM1, ... : index of x-momentum, ...
    // IEN: index of energy

    // Primitive variables
    // IVX, ... : index of x-velocity, ...


    auto &size = pmbp->pmb->mb_size;
    Kokkos::Random_XorShift64_Pool<> rand_pool64(pmbp->gids);
    Real ptotmax = std::numeric_limits<float>::min();
    const int nmkji = (pmbp->nmb_thispack)*indcs.nx3*indcs.nx2*indcs.nx1;
    const int nkji = indcs.nx3*indcs.nx2*indcs.nx1;
    const int nji  = indcs.nx2*indcs.nx1;

    // ==================================================================
    // =================start of parallel_reduce=========================
    // ==================================================================
    // https://kokkos.org/kokkos-core-wiki/API/core/parallel-dispatch/parallel_reduce.html
    Kokkos::parallel_reduce("pgen_cbd_reduce1", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
    KOKKOS_LAMBDA(const int &idx, Real &max_ptot) {
                  // compute m,k,j,i indices of thread and call function
                  int m = (idx)/nkji;
                  int k = (idx - m*nkji)/nji;
                  int j = (idx - m*nkji - k*nji)/indcs.nx1;
                  int i = (idx - m*nkji - k*nji - j*indcs.nx1) + is;
                  k += ks;
                  j += js;

                  Real &x1min = size.d_view(m).x1min;
                  Real &x1max = size.d_view(m).x1max;
                  Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

                  Real &x2min = size.d_view(m).x2min;
                  Real &x2max = size.d_view(m).x2max;
                  Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

                  Real &x3min = size.d_view(m).x3min;
                  Real &x3max = size.d_view(m).x3max;
                  Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);

                  Real &dx1 = size.d_view(m).dx1;
                  Real &dx2 = size.d_view(m).dx2;
                  Real &dx3 = size.d_view(m).dx3;

                }, 
                result); //see line 469 in gr_torus.cpp
    // ==================================================================
    // ===================end of parallel_reduce=========================
    // ==================================================================
        


};





namespace cbd_isothermal
{

  // \fn rho
  // \brief Function to assign densities to cells
  void rho(MeshBlockPack *pp, ParameterInput *pin){

  }
}



//! From Alex' Athena++ code
void BinarySource(MeshBlock *pmb, const Real time, const Real dt,
              const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
              const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
              AthenaArray<Real> &cons_scalar) {
  Real x1, x2, x3, gm1;
  Real f0, dx, dy, dz, dist, facc;
  Real r1, xp1, yp1, zp1, vxp1, vyp1, vzp1, phi1;
  Real r2, xp2, yp2, zp2, vxp2, vyp2, vzp2, phi2;
  Real vr, vphi, dtarg, m1targ, m2targ, m3targ, fun;
  Real vxa, vya, vza, vxg, vyg, vzg, dvx, dvy, dvz, dxmag, dvmag;
  Real rad, phi, z, ex, ey, rdv, v2, den, kinvisc, dynvisc, press, trqc1, trqc2, trqc;

  Real dtGlob = pmb->pmy_mesh->dt;

  AthenaArray<Real>& Q_hist = pmb->ruser_meshblock_data[0];

  //Binary properties, assumes a circular i=0 orbit. 
  r1 = mu2;
  r2 = mu1;
  phi1 = time;
  phi2 = time+M_PI;

  xp1 = r1*std::cos(phi1);
  yp1 = r1*std::sin(phi1);

  xp2 = r2*std::cos(phi2);
  yp2 = r2*std::sin(phi2);

  vxp1 = -yp1;
  vyp1 = xp1;
  vzp1 = 0.0;

  vxp2 = -yp2;
  vyp2 = xp2;
  vzp2 = 0.0;

  AthenaArray<Real> volume; // 1D array of volumes
  volume.NewAthenaArray(pmb->ncells1);

  gm1 = pmb->peos->GetGamma() - 1.0;
  for (int k=pmb->ks; k<=pmb->ke; ++k) {
    x3 = pmb->pcoord->x3v(k);
    for (int j=pmb->js; j<=pmb->je; ++j) {
      x2 = pmb->pcoord->x2v(j);
      pmb->pcoord->CellVolume(k, j, pmb->is, pmb->ie, volume);
      for (int i=pmb->is; i<=pmb->ie; ++i) {
        x1 = pmb->pcoord->x1v(i);

        den = prim(IDN,k,j,i);
        rad=std::sqrt(SQR(x1) + SQR(x2) + SQR(x3));
        Real cosPhi = x1/rad;
        Real sinPhi = x2/rad;

        vxg = prim(IVX,k,j,i);
        vyg = prim(IVY,k,j,i);
        vzg = prim(IVZ,k,j,i);

        ////Start first particle
        dx = x1 - xp1;
        dy = x2 - yp1;
        dz = x3;

        dxmag = std::sqrt(dx*dx + dy*dy + dz*dz);
        facc = gs1*den*std::exp(-intpow(dxmag/rs1, 4))*dt;
        f0 = -1*dt*den*GetForce(mu1, dxmag, eps1)/(dxmag+float_min);

        if (sink_t  == 1) {
          //standard sink
          vxa = vxg;
          vya = vyg;
          vza = vzg;
        }
        else if (sink_t == 2) {
          //paul/planet sink
          //relative velocity
          vxa = vxp1;
          vya = vyp1;
          vza = vzp1;
        }
        else {
          //torque-free sink - only accrete (relative) radial momentum
          //relative velocity
          dvx = vxg - vxp1;
          dvy = vyg - vyp1;
          dvz = vzg - vzp1;
          //magnitude of parallel component
          dvmag = (dvx*dx + dvy*dy + dvz*dz)/dxmag;
          //torque-free accreting velocity
          vxa = vxp1 + dvmag*dx/dxmag;
          vya = vyp1 + dvmag*dy/dxmag;
          vza = vzp1 + dvmag*dz/dxmag;
        }

        //gravitational acceleration
        cons(IM3,k,j,i) -= dz*f0;
        cons(IM2,k,j,i) -= dy*f0;
        cons(IM1,k,j,i) -= dx*f0;

        //sink terms
        cons(IDN,k,j,i) -= facc;
        cons(IM1,k,j,i) -= facc*vxa;
        cons(IM2,k,j,i) -= facc*vya;
        cons(IM3,k,j,i) -= facc*vza;

        ////Accounting
        if (dtGlob == dt) {
          Q_hist(0) += facc*volume(i);
          Q_hist(2) += facc*(xp1*vya - yp1*vxa)*volume(i);
          Q_hist(4) += f0*(xp1*dy - yp1*dx)*volume(i);
        }

        ////Start second particle
        dx = x1 - xp2;
        dy = x2 - yp2;
        dz = x3;

        dxmag = std::sqrt(dx*dx + dy*dy + dz*dz);
        facc = gs2*den*std::exp(-intpow(dxmag/rs2, 4))*dt;
        f0 = -1*dt*den*GetForce(mu2, dxmag, eps2)/(dxmag+float_min);

        if (sink_t  == 1) {
          //standard sink
          vxa = vxg;
          vya = vyg;
          vza = vzg;
        }
        else if (sink_t == 2) {
          //paul/planet sink
          //relative velocity
          vxa = vxp2;
          vya = vyp2;
          vza = vzp2;
        }
        else {
          //torque-free sink - only accrete (relative) radial momentum
          //relative velocity
          dvx = vxg - vxp2;
          dvy = vyg - vyp2;
          dvz = vzg - vzp2;
          //magnitude of parallel component
          dvmag = (dvx*dx + dvy*dy + dvz*dz)/dxmag;
          //torque-free accreting velocity
          vxa = vxp2 + dvmag*dx/dxmag;
          vya = vyp2 + dvmag*dy/dxmag;
          vza = vzp2 + dvmag*dz/dxmag;
        }

        //gravitational acceleration
        cons(IM3,k,j,i) -= dz*f0;
        cons(IM2,k,j,i) -= dy*f0;
        cons(IM1,k,j,i) -= dx*f0;

        //sink terms
        cons(IDN,k,j,i) -= facc;
        cons(IM1,k,j,i) -= facc*vxa;
        cons(IM2,k,j,i) -= facc*vya;
        cons(IM3,k,j,i) -= facc*vza;

        ////Accounting
        if (dtGlob == dt) {
          Q_hist(1) += facc*volume(i);
          Q_hist(3) += facc*(xp2*vya - yp2*vxa)*volume(i);
          Q_hist(5) += f0*(xp2*dy - yp2*dx)*volume(i);
        }

        ////Damping and locally isothermal 'EOS'
        Real phi=std::atan2(x2, x1);
        Real z=x3;

        dtarg = DenProfileCyl(rad,phi,z);
        VelProfileCyl(vr, vphi, rad,phi,z,time);
        m1targ = dtarg*(vr*x1/rad - vphi*x2/rad);
        m2targ = dtarg*(vr*x2/rad + vphi*x1/rad);
        m3targ = 0.0;

        fun = 0.0;
        if (rad > rdamp) fun = 1.0 - SQR(1.0 - SQR((rad - rdamp)/ldamp));
        if (rad > rdamp + ldamp) fun = 1.0;
        fun *= 10.0*dt*std::fabs(vphi)/rad;

        cons(IDN,k,j,i) -= (den-dtarg)*fun;
        cons(IM1,k,j,i) -= (vxg*den-m1targ)*fun;
        cons(IM2,k,j,i) -= (vyg*den-m2targ)*fun;
        cons(IM3,k,j,i) -= (vzg*den-m3targ)*fun;

        Real p_over_r = PoverR(rad,phi,z, time);
        cons(IEN,k,j,i) = p_over_r*cons(IDN,k,j,i)/(gamma_gas - 1.0);
        cons(IEN,k,j,i) += 0.5*(SQR(cons(IM1,k,j,i))+SQR(cons(IM2,k,j,i))
                                       + SQR(cons(IM3,k,j,i)))/cons(IDN,k,j,i);

      }
    }
  }
  return;
}
