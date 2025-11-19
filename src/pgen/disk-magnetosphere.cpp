//========================================================================================
// AthenaK astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file disk-magnetosphere.cpp
//! \brief Initializes disk interacting with magnetosphere of central star.

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

// C++ headers
#include <algorithm>  // min
#include <cmath>      // sqrt(), pow()
#include <cstdlib>    // srand
#include <cstring>    // strcmp()
#include <cstdio>     // snprintf()
#include <fstream>
#include <iostream>   // endl
#include <limits>
#include <sstream>    // stringstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()

// Athena++ headers
#include "athena.hpp"
#include "bvals/bvals.hpp"
#include "coordinates/coordinates.hpp"
#include "coordinates/cartesian_ks.hpp"
#include "coordinates/cell_locations.hpp"
#include "eos/eos.hpp"
#include "mhd/mhd.hpp"
#include "globals.hpp"
#include "hydro/hydro.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"

// prototypes for functions used internally to this pgen
namespace {

    KOKKOS_INLINE_FUNCTION
    static void GetCylCoord(struct my_params mp, Real &rad,Real &phi,Real &z,Real &x1,Real &x2,Real &x3);

    KOKKOS_INLINE_FUNCTION
    static Real DenDiscCyl(struct my_params mp, const Real rad, const Real phi, const Real z);

    KOKKOS_INLINE_FUNCTION
    static Real DenStarCyl(struct my_params mp, const Real rad, const Real phi, const Real z);

    KOKKOS_INLINE_FUNCTION
    static Real PoverR(struct my_params mp, const Real rad, const Real phi, const Real z);

    KOKKOS_INLINE_FUNCTION
    static void VelProfileCyl(struct my_params mp, const Real rad, const Real phi, const Real z, Real &v1, Real &v2, Real &v3);

    KOKKOS_INLINE_FUNCTION
    static Real A1(struct my_params mp, const Real x1, const Real x2, const Real x3);

    KOKKOS_INLINE_FUNCTION
    static Real A2(struct my_params mp, const Real x1, const Real x2, const Real x3);

    KOKKOS_INLINE_FUNCTION
    static Real A3(struct my_params mp, const Real x1, const Real x2, const Real x3);

    // Initialize global instance of the parameter structure
    // pgen_struct disc_params;

    struct my_params {
    Real gm0, r0, rho0, dslope, p0_over_r0, qslope, tcool, gamma_gas;
    Real dfloor, rho_floor0, rho_floor_slope;
    Real Omega0;
    Real rs, smoothin, smoothtr, rfix;
    Real Rmin, Ri, Ro, Rmax;
    Real thmin, thi, tho, thmax;
    Real rrigid, origid, rmagsph, denstar, ratmagfloor, ratmagfslope;
    Real mm, b0, beta;
    bool is_ideal;
    bool magnetic_fields_enabled;
    static int bc_ix3, bc_ox3;
    };

    my_params mp;

} // End of namespace

KOKKOS_INLINE_FUNCTION
Real rho_floor(struct my_params mp, const Real r);

KOKKOS_INLINE_FUNCTION
Real GravPot_coe(struct my_params mp, Real rc);

// prototypes for user-defined BCs and source functions
void StarSourceTerms(Mesh* pm, const Real bdt);
void CoolingSourceTerms(Mesh* pm, const Real bdt);
void MySourceTerms(Mesh* pm, const Real bdt);

//----------------------------------------------------------------------------------------
//! \fn
//! \brief Problem Generator for warped disc experiments. Sets initial conditions for an equilibrium
//! which is then rotated in spherical shells to introduce a radially dependent tilt and twait profile.
//! Compile with '-D PROBLEM=warp_disc' to enroll as a user-specific problem generator.
//----------------------------------------------------------------------------------------

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {

    // Now enroll user source terms and boundary conditions if specified
    if (user_srcs) {
        user_srcs_func = MySourceTerms;
        }

    // if (user_bcs) {
    //     user_bcs_func = FixedDiscBC;
    // }

    // // If restarting then end initialisation here
    // if (restart) return;

    // Read problem parameters from input file

    MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
    EOS_Data &eos = pmbp->pmhd->peos->eos_data;

    mp.beta = pin->GetReal("problem","beta");
    mp.denstar = pin->GetOrAddReal("problem","denstar",0.0);
    mp.dslope = pin->GetOrAddReal("problem","dslope",0.0);
    mp.gamma_gas = pin->GetReal("hydro","gamma");
    mp.gm0 = 1.0;
    mp.is_ideal = eos.is_ideal;
    if (pmbp->pmhd != nullptr) mp.magnetic_fields_enabled = true;
    else mp.magnetic_fields_enabled = false;
    mp.origid = pin->GetOrAddReal("problem","origid",0.0);
    mp.p0_over_r0 = pin->GetOrAddReal("problem","p0_over_r0",0.0025);
    mp.qslope = pin->GetOrAddReal("problem","qslope",0.0);
    mp.ratmagfloor = pin->GetOrAddReal("problem","ratmagfloor",1.0e6);
    mp.ratmagfslope = pin->GetOrAddReal("problem","ratmagfslope", 5.5);
    mp.rfix = pin->GetOrAddReal("problem", "rfix",0.1);
    mp.rho0 = pin->GetReal("problem","rho0");
    mp.rho_floor0 = pin->GetReal("problem","rho_floor0");
    mp.rho_floor_slope = pin->GetOrAddReal("problem","rho_floor_slope",0.0);
    mp.rrigid = pin->GetOrAddReal("problem","rrigid",0.0);
    mp.rmagsph = pin->GetOrAddReal("problem","rmagsph",0.0);
    mp.rs = pin->GetOrAddReal("problem", "rstar",0.0);
    mp.smoothtr = pin->GetOrAddReal("problem","smoothtr",0.0);
    mp.tcool = pin->GetOrAddReal("problem","tcool",0.0);

    // If magnetic fields are enabled TODO: implement check
    mp.mm = pin->GetOrAddReal("problem","mm",0.0);  // Read in the magnetic moment

    // // Capture variables for kernel - e.g. indices for looping over the meshblocks and the size of the meshblocks.
    // auto &indcs = pmy_mesh_->mb_indcs;
    // int &is = indcs.is; int &ie = indcs.ie;
    // int &js = indcs.js; int &je = indcs.je;
    // int &ks = indcs.ks; int &ke = indcs.ke;
    // MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
    // auto &size = pmbp->pmb->mb_size;

    // // Initialise a pointer to the disc parameter structure   
    // auto disc_params_ = disc_params;

    // // Select either Hydro or MHD and extract the arrays - set on the device specifically since this is where the calculations
    // // are going to be done anyway. 
    // DvceArray5D<Real> u0_, w0_;
    // if (pmbp->phydro != nullptr) {
    //     u0_ = pmbp->phydro->u0;
    //     w0_ = pmbp->phydro->w0;
    // }

    // // Loop over array and assign the quantities.
    // par_for("pgen_UserProblem",DevExeSpace(),0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
    // KOKKOS_LAMBDA(int m,int k,int j,int i) {
    //     Real &x1min = size.d_view(m).x1min;
    //     Real &x1max = size.d_view(m).x1max;
    //     int nx1 = indcs.nx1;
    //     Real xwarp = CellCenterX(i-is, nx1, x1min, x1max);

    //     Real &x2min = size.d_view(m).x2min;
    //     Real &x2max = size.d_view(m).x2max;
    //     int nx2 = indcs.nx2;
    //     Real ywarp = CellCenterX(j-js, nx2, x2min, x2max);

    //     Real &x3min = size.d_view(m).x3min;
    //     Real &x3max = size.d_view(m).x3max;
    //     int nx3 = indcs.nx3;
    //     Real zwarp = CellCenterX(k-ks, nx3, x3min, x3max);

    //     // Declare the primtive variables
    //     Real den(0.0), pgas(0.0), ux(0.0), uy(0.0), uz(0.0);

    //     // Now compute the warped primitive variables at this location
    //     ComputePrimitives(disc_params_,xwarp, ywarp, zwarp, den, pgas, ux, uy, uz);

    //     // Now set the conserved variables using the primitive variables
    //     u0_(m,IDN,k,j,i) = den;
    //     u0_(m,IM1,k,j,i) = den*ux;
    //     u0_(m,IM2,k,j,i) = den*uy;
    //     u0_(m,IM3,k,j,i) = den*uz;
    //     if (disc_params_.eos_flag != disc_params_.eos_isothermal) {
    //         u0_(m,IEN,k,j,i) = pgas/(disc_params_.gamma_gas - 1.0)+0.5*(SQR(ux)+SQR(uy)+ SQR(uz))/den;
    //     }
    // });

    // // Check that no errors were flagged during the initialisation
    // if (disc_params_.error == 1) {
    //     std::cout << "Error: Negative azimuthal velocity detected. Check your input parameters." << std::endl;
    //     exit(1);
    // }

    return; // END OF ProblemGenerator::UserProblem()
}

//----------------------------------------------------------------------------------------
//! Now we define a variety of functions for use in the problem generation within the local 
//! namespace. Compatible with their declaration in the preamble.
//----------------------------------------------------------------------------------------

namespace {

    //----------------------------------------------------------------------------------------
    //! Transform from cartesian to cylindrical coordinates
    KOKKOS_INLINE_FUNCTION
    static void GetCylCoord(struct my_params mp, Real &rad,Real &phi,Real &z,Real &x1,Real &x2,Real &x3) {
        rad=sqrt(x1*x1 + x2*x2);
        phi=atan2(x2,x1);
        z=x3;
        return;
    }

    //----------------------------------------------------------------------------------------
    KOKKOS_INLINE_FUNCTION
    static Real DenDiscCyl(struct my_params mp, const Real rad, const Real phi, const Real z) {
        
        // Compute the density profile in cylindrical coordinates
        // Vertical hydrostatic equilibrium (Nelson et al. 2013) 
        Real den;
        Real r = fmax(rad, mp.rs);
        Real p_over_r = mp.p0_over_r0;
        if (mp.is_ideal) p_over_r = PoverR(mp, r, phi, z);
        Real denmid = mp.rho0*std::pow(r/mp.r0,mp.dslope);
        Real dentem = denmid*std::exp(mp.gm0/p_over_r*(1./std::sqrt(SQR(r)+SQR(z))-1./r));
        den = dentem;

        Real rc = sqrt(rad*rad+z*z);  // spherical radius

        // For region inside magnetosphere, apply exponential tapering
        if (rc<mp.rmagsph) den = den*exp(-SQR((rc-mp.rmagsph)/mp.smoothtr));

        // Apply the density floor
        den = fmax(den,rho_floor(mp,rc));
        return den;
    }

    //----------------------------------------------------------------------------------------
    KOKKOS_INLINE_FUNCTION
    static Real DenStarCyl(struct my_params mp, const Real rad, const Real phi, const Real z) {
        
        // Add the stellar density profile component
        Real den(0.0);
        Real csq0 = mp.p0_over_r0;
        Real rc = sqrt(rad*rad+z*z);  // spherical radius

        if (rc<mp.rmagsph) {

            Real sinsq = rad*rad/rc/rc;
            if (mp.is_ideal) csq0 = PoverR(mp, mp.rs, phi, z);
            
            Real pre0=mp.denstar*csq0; // reference pressure
            Real rint = mp.rs;         // integrate from stellar surface
            Real dr = mp.rs/100.;      // integration step size

            Real pre = pre0*exp(0.5*mp.origid*mp.origid*mp.rs*mp.rs*sinsq/csq0);  // pressure at stellar surface

            if (rc < mp.rs) {
                // analytic solution inside the star
                pre = pre0*exp(0.5*mp.origid*mp.origid*rad*rad/csq0);
            } else {
                // integrate stellar envelope out from the stellar surface towards rc
                while(rint<rc) {
                    pre += - dr*mp.gm0/rint/rint*(rint-mp.rs)*
                                (rint-mp.rs)/((rint-mp.rs)*(rint-mp.rs)+mp.smoothtr*mp.smoothtr)*
                                pre/csq0 + dr*mp.origid*mp.origid*rint*sinsq*pre/csq0;
                    rint = rint + dr;
                }
            }

            den = pre/csq0;

        }
        
        return den;
    }

    //----------------------------------------------------------------------------------------
    KOKKOS_INLINE_FUNCTION
    static Real PoverR(struct my_params mp, const Real rad, const Real phi, const Real z) {
        Real poverr;
        Real r = fmax(rad, mp.rfix);
        poverr = mp.p0_over_r0*std::pow(r/mp.r0, mp.qslope);
        return poverr;
    }

    //----------------------------------------------------------------------------------------
    KOKKOS_INLINE_FUNCTION
    static void VelProfileCyl(struct my_params mp, const Real rad, const Real phi, const Real z, Real &v1, Real &v2, Real &v3) {
        
        Real r = fmax(rad, mp.rs);
        Real p_over_r = PoverR(mp, r, phi, z);
        Real vel = (mp.dslope+mp.qslope)*p_over_r/(mp.gm0/r) + (1.0+mp.qslope) - mp.qslope*r/sqrt(r*r+z*z);
        vel = sqrt(mp.gm0/r)*sqrt(vel);
        Real rc = sqrt(rad*rad+z*z);
        if (rc<mp.rmagsph) vel = vel*exp(-SQR((rc-mp.rmagsph)/mp.smoothtr));
        if (rc<=mp.rrigid) vel += mp.origid*rad;

        v1=-vel*sin(phi);
        v2=+vel*cos(phi);
        v3=0.0;

        return;
    }

    //----------------------------------------------------------------------------------------
    KOKKOS_INLINE_FUNCTION
    static Real A1(struct my_params mp, const Real x1, const Real x2, const Real x3) {
        Real a1=0.0;
        Real rc = fmax(sqrt(x1*x1+x2*x2+x3*x3),mp.rs);
        a1 = mp.mm*mp.b0/rc/rc/rc*(-1.*x2);
        return(a1);
    }

    //----------------------------------------------------------------------------------------
    KOKKOS_INLINE_FUNCTION
    static Real A2(struct my_params mp, const Real x1, const Real x2, const Real x3) {
        Real a2=0.0;
        Real rc = fmax(sqrt(x1*x1+x2*x2+x3*x3),mp.rs);
        a2 = mp.mm*mp.b0/rc/rc/rc*(+1.*x1);
        return(a2);
    }

    //----------------------------------------------------------------------------------------
    KOKKOS_INLINE_FUNCTION
    static Real A3(struct my_params mp, const Real x1, const Real x2, const Real x3) {
        Real a3=0.0;
        return(a3);
    }

} // End of namespace functions

KOKKOS_INLINE_FUNCTION
Real rho_floor(struct my_params mp, Real rc) {
    Real rhofloor = 0.0;
    if (rc > mp.rs) rhofloor = mp.rho_floor0*pow(rc/mp.r0, mp.rho_floor_slope);
    if (mp.mm != 0. && rc > mp.rs) rhofloor += 4.*mp.rho0*mp.mm*mp.mm/mp.beta/mp.ratmagfloor*
                                            pow((mp.r0/rc),mp.ratmagfslope);
    return fmax(rhofloor,mp.dfloor);
}

KOKKOS_INLINE_FUNCTION
Real GravPot_coe(struct my_params mp, Real rc) {
  return(-mp.gm0/rc/rc/rc);
}
        
//----------------------------------------------------------------------------------------
//! Below we will define the custom user defined source terms and boundary conditions.
//----------------------------------------------------------------------------------------
//! Write the user source term function. Note that all source terms should be included in
//! the same function and enrolled together. Currently I have written a cooling function
//! and a velocity damping function.

void MySourceTerms(Mesh* pm, const Real bdt) {

    StarSourceTerms(pm, bdt);
    if(mp.is_ideal && mp.tcool>0.0) CoolingSourceTerms(pm, bdt);
    return;
}

//----------------------------------------------------------------------------------------
void StarSourceTerms(Mesh* pm, const Real bdt) {

    // Capture variables for kernel - e.g. indices for looping over the meshblocks and the size of the meshblocks.
    auto &indcs = pm->mb_indcs;
    int &is = indcs.is; int &ie = indcs.ie;
    int &js = indcs.js; int &je = indcs.je;
    int &ks = indcs.ks; int &ke = indcs.ke;
    MeshBlockPack *pmbp = pm->pmb_pack;
    auto &size = pmbp->pmb->mb_size;

    // Now set a local parameter struct for lambda capturing
    auto mp_ = mp;

    // Select either Hydro or MHD
    DvceArray5D<Real> u0_, w0_, bcc0_;
    if (pm->pmb_pack->phydro != nullptr) {
        u0_ = pm->pmb_pack->phydro->u0;
        w0_ = pm->pmb_pack->phydro->w0;
    } else if (pm->pmb_pack->pmhd != nullptr) {
        u0_ = pm->pmb_pack->pmhd->u0;
        w0_ = pm->pmb_pack->pmhd->w0;
        bcc0_ = pmbp->pmhd->bcc0;
    }

    int nvar = u0_.extent_int(1);
    DvceArray1D<Real> src;
    Kokkos::realloc(src, nvar);

    par_for("pgen_starsource",DevExeSpace(),0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m,int k,int j,int i) {

        // Extract the cell center coordinates
        Real &x1min = size.d_view(m).x1min;
        Real &x1max = size.d_view(m).x1max;
        Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

        Real &x2min = size.d_view(m).x2min;
        Real &x2max = size.d_view(m).x2max;
        Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

        Real &x3min = size.d_view(m).x3min;
        Real &x3max = size.d_view(m).x3max;
        Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);

        Real rc = sqrt(x1v*x1v+x2v*x2v+x3v*x3v);
        Real fcoe=GravPot_coe(mp_,rc); // TODO: Need to define and implement this function.
        
        Real f_x1 = fcoe*x1v;
        Real f_x2 = fcoe*x2v;
        Real f_x3 = fcoe*x3v;

        // Implement the smoothing function
        Real rcv2 = (rc-mp_.rs)*(rc-mp_.rs);
        Real fsmooth = rcv2/(rcv2+mp_.smoothtr*mp_.smoothtr);
        if (rc<mp_.rs) fsmooth=0.;

        // Now set the source terms
        src(IM1) = bdt*w0_(m,IDN,k,j,i)*f_x1*fsmooth;
        src(IM2) = bdt*w0_(m,IDN,k,j,i)*f_x2*fsmooth;
        src(IM3) = bdt*w0_(m,IDN,k,j,i)*f_x3*fsmooth;

        u0_(m,IM1,k,j,i) += src(IM1);
        u0_(m,IM2,k,j,i) += src(IM2);
        u0_(m,IM3,k,j,i) += src(IM3);
        
        if(mp_.is_ideal) {
            u0_(m,IEN,k,j,i) += src(IM1)*w0_(m,IM1,k,j,i) + 
                                src(IM2)*w0_(m,IM2,k,j,i) + 
                                src(IM3)*w0_(m,IM3,k,j,i);
        }

        Real rad(0.0),phi(0.0),z(0.0);
        Real v1(0.0),v2(0.0),v3(0.0);

        GetCylCoord(mp_,rad,phi,z,x1v,x2v,x3v);
        Real rsph=sqrt(x1v*x1v+x2v*x2v+x3v*x3v);
        // TODO: this is the part to be careful with - in the case of a tilted+rotating dipole.
        if (rsph<mp_.rfix) {
            u0_(m,IDN,k,j,i) = DenDiscCyl(mp_,rad,phi,z);
            VelProfileCyl(mp_,rad,phi,z,v1,v2,v3);
            u0_(m,IM1,k,j,i) = v1*u0_(m,IDN,k,j,i);
            u0_(m,IM2,k,j,i) = v2*u0_(m,IDN,k,j,i);
            u0_(m,IM3,k,j,i) = v3*u0_(m,IDN,k,j,i);
            
            if (mp_.is_ideal) {
                u0_(m,IEN,k,j,i) = PoverR(mp_,rad, phi, z)*u0_(m,IDN,k,j,i)/(mp_.gamma_gas - 1.0)+
                                0.5*(SQR(v1)+SQR(v2)+SQR(v3))*u0_(m,IDN,k,j,i);
            
                if (mp_.magnetic_fields_enabled) {
                    u0_(m,IEN,k,j,i) = u0_(m,IEN,k,j,i)+0.5*(SQR(bcc0_(m,IBX,k,j,i))+
                                    SQR(bcc0_(m,IBY,k,j,i))+SQR(bcc0_(m,IBZ,k,j,i)));
                }
            }
        }
    
        u0_(m,IDN,k,j,i) = fmax(u0_(m,IDN,k,j,i),rho_floor(mp_,rc));
        
    }); // end par_for

} // end star source terms 

//----------------------------------------------------------------------------------------
void CoolingSourceTerms(Mesh* pm, const Real bdt) {
    // Implement cooling source terms here if needed

    // Capture variables for kernel - e.g. indices for looping over the meshblocks and the size of the meshblocks.
    auto &indcs = pm->mb_indcs;
    int &is = indcs.is; int &ie = indcs.ie;
    int &js = indcs.js; int &je = indcs.je;
    int &ks = indcs.ks; int &ke = indcs.ke;
    MeshBlockPack *pmbp = pm->pmb_pack;
    auto &size = pmbp->pmb->mb_size;

    // Now set a local parameter struct for lambda capturing
    auto mp_ = mp;

    // Select either Hydro or MHD
    DvceArray5D<Real> u0_, w0_, bcc0_;
    if (pm->pmb_pack->phydro != nullptr) {
        u0_ = pm->pmb_pack->phydro->u0;
        w0_ = pm->pmb_pack->phydro->w0;
    } else if (pm->pmb_pack->pmhd != nullptr) {
        u0_ = pm->pmb_pack->pmhd->u0;
        w0_ = pm->pmb_pack->pmhd->w0;
        bcc0_ = pmbp->pmhd->bcc0;
    }

    par_for("pgen_coolsource",DevExeSpace(),0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m,int k,int j,int i) {

        // Extract the cell center coordinates
        Real &x1min = size.d_view(m).x1min;
        Real &x1max = size.d_view(m).x1max;
        Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

        Real &x2min = size.d_view(m).x2min;
        Real &x2max = size.d_view(m).x2max;
        Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

        Real &x3min = size.d_view(m).x3min;
        Real &x3max = size.d_view(m).x3max;
        Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);

        Real eint = u0_(m,IEN,k,j,i)-0.5*(SQR(u0_(m,IM1,k,j,i))+SQR(u0_(m,IM2,k,j,i))
                                            +SQR(u0_(m,IM3,k,j,i)))/u0_(m,IDN,k,j,i);
        if (mp_.magnetic_fields_enabled) {
            eint = eint-0.5*(SQR(bcc0_(m,IBX,k,j,i))+SQR(bcc0_(m,IBY,k,j,i))+SQR(bcc0_(m,IBZ,k,j,i)));
        }

        Real rad(0.0),phi(0.0),z(0.0);
        GetCylCoord(mp_,rad,phi,z,x1v,x2v,x3v);
        Real p_over_r = PoverR(mp_,rad,phi,z);

        Real dtr = fmax(mp_.tcool*2.*M_PI/sqrt(mp_.gm0/rad/rad/rad),bdt);
        Real dfrac=bdt/dtr;
        Real dE=eint-p_over_r/(mp_.gamma_gas-1.0)*u0_(m,IDN,k,j,i);
        u0_(m,IEN,k,j,i) -= dE*dfrac;
        
    }); // end par_for

}// end cooling source terms 

//----------------------------------------------------------------------------------------
//! \fn FixedDiscBC
//  \brief Sets boundary condition on surfaces of computational domain
// Note quantities at boundaries are held fixed to initial condition values

void FixedDiscBC(Mesh *pm) {

    //   // Start by extracting the mesh block and cell information 
    //   auto &indcs = pm->mb_indcs;
    //   auto &size = pm->pmb_pack->pmb->mb_size;
    //   auto &coord = pm->pmb_pack->pcoord->coord_data;
    //   int &ng = indcs.ng;
    //   int n1 = indcs.nx1 + 2*ng;
    //   int n2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng) : 1;
    //   int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng) : 1;
    //   int &is = indcs.is;  int &ie  = indcs.ie;
    //   int &js = indcs.js;  int &je  = indcs.je;
    //   int &ks = indcs.ks;  int &ke  = indcs.ke;
    //   auto &mb_bcs = pm->pmb_pack->pmb->mb_bcs;

    //   // Initialise a pointer to the disc parameter structure   
    //   auto disc_params_ = disc_params;

    //   int nmb = pm->pmb_pack->nmb_thispack;

    //   // Select either Hydro or MHD arrays
    //   DvceArray5D<Real> u0_, w0_;
    //   if (pm->pmb_pack->phydro != nullptr) {
    //     u0_ = pm->pmb_pack->phydro->u0;
    //     w0_ = pm->pmb_pack->phydro->w0;
    //   }

    //   // X1 BOUNDARY CONDITIONS ---------------> 

    //   // Start off by converting all the conservative variables to primitives so everything is synchronised
    //   pm->pmb_pack->phydro->peos->ConsToPrim(u0_,w0_,false,is-ng,is-1,0,(n2-1),0,(n3-1));
    //   pm->pmb_pack->phydro->peos->ConsToPrim(u0_,w0_,false,ie+1,ie+ng,0,(n2-1),0,(n3-1));
    //   par_for("fixed_x1", DevExeSpace(),0,(nmb-1),0,(n3-1),0,(n2-1),0,(ng-1),
    //   KOKKOS_LAMBDA(int m, int k, int j, int i) {

    //     // Extract coordinates at inner x1 boundary on each meshblock in the pack
    //     Real &x1min = size.d_view(m).x1min;
    //     Real &x1max = size.d_view(m).x1max;
    //     Real xwarp = CellCenterX(i-is, indcs.nx1, x1min, x1max);

    //     Real &x2min = size.d_view(m).x2min;
    //     Real &x2max = size.d_view(m).x2max;
    //     Real ywarp = CellCenterX(j-js, indcs.nx2, x2min, x2max);

    //     Real &x3min = size.d_view(m).x3min;
    //     Real &x3max = size.d_view(m).x3max;
    //     Real zwarp = CellCenterX(k-ks, indcs.nx3, x3min, x3max);

    //     Real den, pgas, ux, uy, uz;
    //     // Inner x1 boundary
    //     if (mb_bcs.d_view(m,BoundaryFace::inner_x1) == BoundaryFlag::user) {
    //         // Compute the warped primitive variables at this location
    //         ComputePrimitives(disc_params_,xwarp, ywarp, zwarp, den, pgas, ux, uy, uz);

    //         // Now set the conserved variables using the primitive variables
    //         w0_(m,IDN,k,j,i) = den;
    //         w0_(m,IVX,k,j,i) = ux;
    //         w0_(m,IVY,k,j,i) = uy;
    //         w0_(m,IVZ,k,j,i) = uz;
    //         if (disc_params_.eos_flag != disc_params_.eos_isothermal) {
    //             w0_(m,IEN,k,j,i) = pgas/(disc_params_.gamma_gas - 1.0);
    //         }
    //     }

    //     // Outer x1 boundary
    //     xwarp = CellCenterX((ie+i+1)-is, indcs.nx1, x1min, x1max);

    //     if (mb_bcs.d_view(m,BoundaryFace::outer_x1) == BoundaryFlag::user) {
    //         // Compute the warped primitive variables at this location
    //         ComputePrimitives(disc_params_,xwarp, ywarp, zwarp, den, pgas, ux, uy, uz);

    //         // Now set the conserved variables using the primitive variables
    //         w0_(m,IDN,k,j,(ie+i+1)) = den;
    //         w0_(m,IVX,k,j,(ie+i+1)) = ux;
    //         w0_(m,IVY,k,j,(ie+i+1)) = uy;
    //         w0_(m,IVZ,k,j,(ie+i+1)) = uz;
    //         if (disc_params_.eos_flag != disc_params_.eos_isothermal) {
    //             w0_(m,IEN,k,j,(ie+i+1)) = pgas/(disc_params_.gamma_gas - 1.0);
    //         }
    //     }
    //   });
    //   // Now synchronise PrimToCons on X1 physical boundary ghost zones
    //   pm->pmb_pack->phydro->peos->PrimToCons(w0_,u0_,is-ng,is-1,0,(n2-1),0,(n3-1));
    //   pm->pmb_pack->phydro->peos->PrimToCons(w0_,u0_,ie+1,ie+ng,0,(n2-1),0,(n3-1));


    //   // X2 BOUNDARY CONDITIONS ---------------> 

    //   pm->pmb_pack->phydro->peos->ConsToPrim(u0_,w0_,false,0,(n1-1),js-ng,js-1,0,(n3-1));
    //   pm->pmb_pack->phydro->peos->ConsToPrim(u0_,w0_,false,0,(n1-1),je+1,je+ng,0,(n3-1));
    //   par_for("fixed_x2", DevExeSpace(),0,(nmb-1),0,(n3-1),0,(ng-1),0,(n1-1),
    //   KOKKOS_LAMBDA(int m, int k, int j, int i) {
    //     // inner x2 boundary
    //     Real &x1min = size.d_view(m).x1min;
    //     Real &x1max = size.d_view(m).x1max;
    //     Real xwarp = CellCenterX(i-is, indcs.nx1, x1min, x1max);

    //     Real &x2min = size.d_view(m).x2min;
    //     Real &x2max = size.d_view(m).x2max;
    //     Real ywarp = CellCenterX(j-js, indcs.nx2, x2min, x2max);

    //     Real &x3min = size.d_view(m).x3min;
    //     Real &x3max = size.d_view(m).x3max;
    //     Real zwarp = CellCenterX(k-ks, indcs.nx3, x3min, x3max);

    //     Real den, pgas, ux, uy, uz;
    //     // Inner x2 boundary
    //     if (mb_bcs.d_view(m,BoundaryFace::inner_x2) == BoundaryFlag::user) {
    //         // Compute the warped primitive variables at this location
    //         ComputePrimitives(disc_params_,xwarp, ywarp, zwarp, den, pgas, ux, uy, uz);

    //         // Now set the conserved variables using the primitive variables
    //         w0_(m,IDN,k,j,i) = den;
    //         w0_(m,IVX,k,j,i) = ux;
    //         w0_(m,IVY,k,j,i) = uy;
    //         w0_(m,IVZ,k,j,i) = uz;
    //         if (disc_params_.eos_flag != disc_params_.eos_isothermal) {
    //             w0_(m,IEN,k,j,i) = pgas/(disc_params_.gamma_gas - 1.0);
    //         }
    //     }

    //     // Outer x2 boundary
    //     xwarp = CellCenterX((je+j+1)-js, indcs.nx2, x2min, x2max);

    //     if (mb_bcs.d_view(m,BoundaryFace::outer_x2) == BoundaryFlag::user) {
    //         // Compute the warped primitive variables at this location
    //         ComputePrimitives(disc_params_,xwarp, ywarp, zwarp, den, pgas, ux, uy, uz);

    //         // Now set the conserved variables using the primitive variables
    //         w0_(m,IDN,k,(je+j+1),i) = den;
    //         w0_(m,IVX,k,(je+j+1),i) = ux;
    //         w0_(m,IVY,k,(je+j+1),i) = uy;
    //         w0_(m,IVZ,k,(je+j+1),i) = uz;
    //         if (disc_params_.eos_flag != disc_params_.eos_isothermal) {
    //             w0_(m,IEN,k,(je+j+1),i) = pgas/(disc_params_.gamma_gas - 1.0);
    //         }
    //     }
    //   });
    //   pm->pmb_pack->phydro->peos->PrimToCons(w0_,u0_,0,(n1-1),js-ng,js-1,0,(n3-1));
    //   pm->pmb_pack->phydro->peos->PrimToCons(w0_,u0_,0,(n1-1),je+1,je+ng,0,(n3-1));

    //   // X3 BOUNDARY CONDITIONS ---------------> 

    //   pm->pmb_pack->phydro->peos->ConsToPrim(u0_,w0_,false,0,(n1-1),0,(n2-1),ks-ng,ks-1);
    //   pm->pmb_pack->phydro->peos->ConsToPrim(u0_,w0_,false,0,(n1-1),0,(n2-1),ke+1,ke+ng);
    //   par_for("fixed_ix3", DevExeSpace(),0,(nmb-1),0,(ng-1),0,(n2-1),0,(n1-1),
    //   KOKKOS_LAMBDA(int m, int k, int j, int i) {
    //     // inner x3 boundary
    //     Real &x1min = size.d_view(m).x1min;
    //     Real &x1max = size.d_view(m).x1max;
    //     Real xwarp = CellCenterX(i-is, indcs.nx1, x1min, x1max);

    //     Real &x2min = size.d_view(m).x2min;
    //     Real &x2max = size.d_view(m).x2max;
    //     Real ywarp = CellCenterX(j-js, indcs.nx2, x2min, x2max);

    //     Real &x3min = size.d_view(m).x3min;
    //     Real &x3max = size.d_view(m).x3max;
    //     Real zwarp = CellCenterX(k-ks, indcs.nx3, x3min, x3max);

    //     // Initialise primitive variables
    //     Real den, pgas, ux, uy, uz;

    //     // Inner x3 boundary
    //     if (mb_bcs.d_view(m,BoundaryFace::inner_x3) == BoundaryFlag::user) {
    //         // Compute the warped primitive variables at this location
    //         ComputePrimitives(disc_params_,xwarp, ywarp, zwarp, den, pgas, ux, uy, uz);

    //         // Now set the conserved variables using the primitive variables
    //         w0_(m,IDN,k,j,i) = den;
    //         w0_(m,IVX,k,j,i) = ux;
    //         w0_(m,IVY,k,j,i) = uy;
    //         w0_(m,IVZ,k,j,i) = uz;
    //         if (disc_params_.eos_flag != disc_params_.eos_isothermal) {
    //             w0_(m,IEN,k,j,i) = pgas/(disc_params_.gamma_gas - 1.0);
    //         }
    //     }

    //     // Outer x3 boundary
    //     xwarp = CellCenterX((ke+k+1)-ks, indcs.nx3, x3min, x3max);

    //     if (mb_bcs.d_view(m,BoundaryFace::outer_x3) == BoundaryFlag::user) {
    //         // Compute the warped primitive variables at this location
    //         ComputePrimitives(disc_params_,xwarp, ywarp, zwarp, den, pgas, ux, uy, uz);

    //         // Now set the conserved variables using the primitive variables
    //         w0_(m,IDN,(ke+k+1),j,i) = den;
    //         w0_(m,IVX,(ke+k+1),j,i) = ux;
    //         w0_(m,IVY,(ke+k+1),j,i) = uy;
    //         w0_(m,IVZ,(ke+k+1),j,i) = uz;
    //         if (disc_params_.eos_flag != disc_params_.eos_isothermal) {
    //             w0_(m,IEN,(ke+k+1),j,i) = pgas/(disc_params_.gamma_gas - 1.0);
    //         }
    //     }
    //   });
    //   pm->pmb_pack->phydro->peos->PrimToCons(w0_,u0_,0,(n1-1),0,(n2-1),ks-ng,ks-1);
    //   pm->pmb_pack->phydro->peos->PrimToCons(w0_,u0_,0,(n1-1),0,(n2-1),ke+1,ke+ng);

  return;
}