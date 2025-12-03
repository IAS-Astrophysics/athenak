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
    static void VelDiscCyl(struct my_params mp, const Real rad, const Real phi, const Real z, Real &v1, Real &v2, Real &v3);

    KOKKOS_INLINE_FUNCTION
    static void VelStarCyl(struct my_params mp, const Real rad, const Real phi, const Real z, Real &v1, Real &v2, Real &v3);

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
    Real dfloor, rho_floor0, rho_floor_slope, rad_in_cutoff, rad_in_smooth, rad_out_cutoff, rad_out_smooth;
    Real Omega0;
    Real rs, smoothin, gravsmooth;
    Real Rmin, Ri, Ro, Rmax;
    Real thmin, thi, tho, thmax;
    Real origid, rmagsph, denstar, ratmagfloor, ratmagfslope;
    Real mm, b0, beta;
    bool is_ideal;
    bool magnetic_fields_enabled;
    static int bc_ix3, bc_ox3;
    };

    my_params mp;

} // End of namespace

KOKKOS_INLINE_FUNCTION
Real rho_floor(struct my_params mp, const Real r);

// prototypes for user-defined BCs and source functions
void StarGravSourceTerm(Mesh* pm, const Real bdt);
void CoolingSourceTerms(Mesh* pm, const Real bdt);
void MySourceTerms(Mesh* pm, const Real bdt);

void StarMask(Mesh* pm, const Real bdt);
void InnerDiskMask(Mesh* pm, const Real bdt);
void FixedBC(Mesh *pm);

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

    if (user_bcs) {
        user_bcs_func = FixedBC;
    }

    // TODO: CALLUM
    // if (user_efldsrcs) {
    //     user_efldsrcs_func = MyEfieldMask;
    // }

    // If restarting then end initialisation here
    if (restart) return;

    // Read problem parameters from input file

    MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;

  if (pmbp->phydro != nullptr) {
    EOS_Data &eos = pmbp->phydro->peos->eos_data;
    mp.is_ideal = eos.is_ideal;
  } 

  if (pmbp->pmhd != nullptr) {
    EOS_Data &eos = pmbp->pmhd->peos->eos_data;
    mp.is_ideal = eos.is_ideal;
  }

    mp.beta = pin->GetReal("problem","beta");
    mp.denstar = pin->GetOrAddReal("problem","denstar",0.0);
    mp.dslope = pin->GetOrAddReal("problem","dslope",0.0);
    mp.gamma_gas = pin->GetReal("hydro","gamma");
    mp.gm0 = pin->GetOrAddReal("problem","gm0",1.0);
    if (pmbp->pmhd != nullptr) mp.magnetic_fields_enabled = true;
    else mp.magnetic_fields_enabled = false;
    mp.mm = pin->GetOrAddReal("problem","mm",0.0);
    mp.origid = pin->GetOrAddReal("problem","origid",0.0);
    if (mp.is_ideal){
        mp.p0_over_r0 = SQR(pin->GetOrAddReal("problem","h_over_r0",0.1));
    } else { mp.p0_over_r0 = SQR(pin->GetReal("hydro","iso_sound_speed")); }
    mp.qslope = pin->GetOrAddReal("problem","qslope",0.0);
    mp.r0 = pin->GetOrAddReal("problem","r0",1.0);
    mp.rad_in_cutoff = pin->GetOrAddReal("problem","rad_in_cutoff",0.0);
    mp.rad_in_smooth = pin->GetOrAddReal("problem","rad_in_smooth",0.1);
    mp.rad_out_cutoff = pin->GetOrAddReal("problem","rad_out_cutoff",0.0);
    mp.rad_out_smooth = pin->GetOrAddReal("problem","rad_out_smooth",0.1);
    mp.ratmagfloor = pin->GetOrAddReal("problem","ratmagfloor",1.0e6);
    mp.ratmagfslope = pin->GetOrAddReal("problem","ratmagfslope", 5.5);
    mp.rho0 = pin->GetReal("problem","rho0");
    mp.rho_floor0 = pin->GetReal("problem","rho_floor0");
    mp.rho_floor_slope = pin->GetOrAddReal("problem","rho_floor_slope",0.0);
    mp.rmagsph = pin->GetOrAddReal("problem","rmagsph",0.0);
    mp.rs = pin->GetOrAddReal("problem", "rstar",0.0);
    mp.gravsmooth = pin->GetOrAddReal("problem","gravsmooth",0.0);
    mp.tcool = pin->GetOrAddReal("problem","tcool",0.0);
    
    // Capture variables for kernel - e.g. indices for looping over the meshblocks and the size of the meshblocks.
    auto &indcs = pmy_mesh_->mb_indcs;
    int &is = indcs.is; int &ie = indcs.ie;
    int &js = indcs.js; int &je = indcs.je;
    int &ks = indcs.ks; int &ke = indcs.ke;
    auto &size = pmbp->pmb->mb_size;

    // Initialise a pointer to the disc parameter structure   
    auto mp_ = mp;

    // std::cout<< mp.is_ideal << std::endl;

    // Select either Hydro or MHD and extract the arrays - set on the device specifically since this is where the calculations
    // are going to be done anyway. 
    DvceArray5D<Real> u0_, w0_;
    if (pmbp->phydro != nullptr) {
        u0_ = pmbp->phydro->u0;
        w0_ = pmbp->phydro->w0;
    } else if (pmbp->pmhd != nullptr) {
        u0_ = pmbp->pmhd->u0;
        w0_ = pmbp->pmhd->w0;
    }

    // initialize conservative variables for new run ---------------------------------------
    par_for("pgen_magnetosphere_pgen",DevExeSpace(),0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m,int k,int j,int i) {
        Real &x1min = size.d_view(m).x1min;
        Real &x1max = size.d_view(m).x1max;
        int nx1 = indcs.nx1;
        Real x1 = CellCenterX(i-is, nx1, x1min, x1max);

        Real &x2min = size.d_view(m).x2min;
        Real &x2max = size.d_view(m).x2max;
        int nx2 = indcs.nx2;
        Real x2 = CellCenterX(j-js, nx2, x2min, x2max);

        Real &x3min = size.d_view(m).x3min;
        Real &x3max = size.d_view(m).x3max;
        int nx3 = indcs.nx3;
        Real x3 = CellCenterX(k-ks, nx3, x3min, x3max);

        Real rad(0.0), phi(0.0), z(0.0);
        Real den(0.0), ux(0.0), uy(0.0), uz(0.0);
        Real v1_disc(0.0), v2_disc(0.0), v3_disc(0.0);
        Real v1_star(0.0), v2_star(0.0), v3_star(0.0);

        // get the cylindrical coodinates corresponding to this cartesian location
        GetCylCoord(mp_,rad, phi, z, x1, x2, x3);

        // compute the disc density component at this location
        if (mp_.rho0 > 0.0){
            den = DenDiscCyl(mp_, rad, phi, z);
        }
        // add the stellar density component at this location
        if (mp_.denstar > 0.0) {
            den += DenStarCyl(mp_, rad, phi, z);
        }

        // compute the disc velocity component at this location
        if (mp_.rho0 > 0.0){
            VelDiscCyl(mp_, rad, phi, z, v1_disc, v2_disc, v3_disc);
        }
        
        // add the stellar velocity component at this location
        if (mp_.denstar > 0.0) {
            VelStarCyl(mp_, rad, phi, z, v1_star, v2_star, v3_star);
        }
        
        // set the total velocity components
        ux = v1_disc + v1_star;
        uy = v2_disc + v2_star;
        uz = v3_disc + v3_star;

        // set the conserved variables
        u0_(m,IDN,k,j,i) = den;
        u0_(m,IM1,k,j,i) = den*ux;
        u0_(m,IM2,k,j,i) = den*uy;
        u0_(m,IM3,k,j,i) = den*uz;
        
        if (mp_.is_ideal) {
            Real p_over_r = PoverR(mp_, rad, phi, z);
            u0_(m,IEN,k,j,i) = p_over_r*den/(mp_.gamma_gas - 1.0)
                               +0.5*(SQR(u0_(m,IM1,k,j,i))
                               +SQR(u0_(m,IM2,k,j,i))
                               +SQR(u0_(m,IM3,k,j,i)))/u0_(m,IDN,k,j,i) ;
        }

    });

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

        Real den(0.0);
        Real r = fmax(rad, mp.rs);
        Real p_over_r = mp.p0_over_r0;
        if (mp.is_ideal) p_over_r = PoverR(mp, r, phi, z);
        Real denmid = mp.rho0*std::pow(r/mp.r0,mp.dslope);

        // Inner magnetosphere exponential cutoff
        if (rad < mp.rad_in_cutoff) {
            Real cutoff = exp(-SQR((rad-mp.rad_in_cutoff)/mp.rad_in_smooth));
            denmid *= cutoff;
        }

        // Outer disc exponential cutoff
        if (rad > mp.rad_out_cutoff) {
            Real cutoff = exp(-SQR((rad - mp.rad_out_cutoff)/mp.rad_out_smooth));
            denmid *= cutoff;
        }

        den = denmid*std::exp(mp.gm0/p_over_r*(1./std::sqrt(SQR(r)+SQR(z))-1./r));

        // Apply the density floor
        Real rc = sqrt(rad*rad+z*z);
        den = fmax(den,rho_floor(mp,rc));
        
        return den;
    }

    //----------------------------------------------------------------------------------------
    KOKKOS_INLINE_FUNCTION
    static Real DenStarCyl(struct my_params mp, const Real rad, const Real phi, const Real z) {
        
        // Add the stellar density profile component
        Real den(0.0);
        Real rc = sqrt(rad*rad+z*z);  // spherical radius

        if (rc<mp.rmagsph) {

            Real sinsq = rad*rad/rc/rc;
            Real csq0 = PoverR(mp, mp.rs, phi, z);
            
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
                    pre += -dr*mp.gm0/rint/rint *
                        (rint-mp.rs) * (rint-mp.rs)/((rint-mp.rs)*(rint-mp.rs)+mp.gravsmooth*mp.gravsmooth) *
                        pre/csq0 + dr*mp.origid*mp.origid*rint*sinsq*pre/csq0;
                    rint = rint + dr;
                }
            }

            den = pre/csq0;

        }

            // Real pre=mp.denstar*csq0; // reference pressure
            // Real rint = 0.0;         // integrate from stellar surface
            // Real dr = mp.rs/100.;      // integration step size

            // while (rint<rc) {
            //     if (rint<mp.rs){
            //         pre = pre + dr*mp.origid*mp.origid*rint*sinsq*pre/csq0;
            //     } else {
            //         pre = pre - dr*mp.gm0/rint/rint*(rint-mp.rs)*
            //                   (rint-mp.rs)/((rint-mp.rs)*(rint-mp.rs)+mp.gravsmooth*mp.gravsmooth)*
            //                    pre/csq0 + dr*mp.origid*mp.origid*rint*sinsq*pre/csq0;
            //     }
            //     rint = rint + dr;
            // }

            // den = pre/csq0;

        // }
        
        return den;
    }

    //----------------------------------------------------------------------------------------
    KOKKOS_INLINE_FUNCTION
    static Real PoverR(struct my_params mp, const Real rad, const Real phi, const Real z) {
        Real poverr;
        Real r = fmax(rad, mp.rs);
        poverr = mp.p0_over_r0*std::pow(r/mp.r0, mp.qslope);
        return poverr;
    }

    //----------------------------------------------------------------------------------------
    KOKKOS_INLINE_FUNCTION
    static void VelDiscCyl(struct my_params mp, const Real rad, const Real phi, const Real z, Real &v1, Real &v2, Real &v3) {
        
        Real r = fmax(rad, mp.rs);
        Real rc = sqrt(r*r+z*z);

        // Old method for power law
        // Real p_over_r = PoverR(mp, r, phi, z);
        // Real vel = (mp.dslope+mp.qslope)*p_over_r/(mp.gm0/r) + (1.0+mp.qslope) - mp.qslope*r/sqrt(r*r+z*z);
        // vel = sqrt(mp.gm0/r)*sqrt(vel);

        // Testing new method for balance with pressure gradients
        Real dR = fmin(mp.rad_in_smooth, mp.rad_out_smooth)/100;
        Real dPdr = (PoverR(mp, r+dR, phi, z) * DenDiscCyl(mp, r+dR, phi, z) - PoverR(mp, r-dR, phi, z) * DenDiscCyl(mp, r - dR, phi, z))/(2 * dR);
        Real vel = sqrt(fmax(mp.gm0*r*r/rc/rc/rc+r/DenDiscCyl(mp, r, phi, z)*dPdr,0.0));

        rc = sqrt(rad*rad+z*z);

        if (rc<mp.rmagsph) {
            vel = vel*exp(-SQR((rc-mp.rmagsph)/mp.rad_in_smooth));
        }

        v1=-vel*sin(phi);
        v2=+vel*cos(phi);
        v3=0.0;

        return;
    }

    //----------------------------------------------------------------------------------------
    KOKKOS_INLINE_FUNCTION
    static void VelStarCyl(struct my_params mp, const Real rad, const Real phi, const Real z, Real &v1, Real &v2, Real &v3) {
        
        Real vel(0.0);
        Real rc = sqrt(rad*rad+z*z);  // spherical radius
        
        vel = mp.origid*rad; // rigid rotation
        if (rc>mp.rmagsph) {
            vel = mp.origid*rad*exp(-SQR((rc-mp.rmagsph)/mp.rad_in_smooth));
        }

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
    Real rhofloor = mp.rho_floor0;
    if (rc > mp.rs) rhofloor = mp.rho_floor0*pow(rc/mp.rs, mp.rho_floor_slope);
    if (mp.mm != 0. && rc > mp.rs) rhofloor += 4.*mp.rho0*mp.mm*mp.mm/mp.beta/mp.ratmagfloor*
                                            pow((mp.r0/rc),mp.ratmagfslope);
    return fmax(rhofloor,mp.dfloor);
}
        
//----------------------------------------------------------------------------------------
//! Below we will define the custom user defined source terms and boundary conditions.
//----------------------------------------------------------------------------------------
//! Write the user source term function. Note that all source terms should be included in
//! the same function and enrolled together. Currently I have written a cooling function
//! and a velocity damping function.

void MySourceTerms(Mesh* pm, const Real bdt) {

    StarGravSourceTerm(pm, bdt);
    if(mp.is_ideal && mp.tcool>0.0) CoolingSourceTerms(pm, bdt);
    if (mp.denstar > 0.0) StarMask(pm, bdt);
    // InnerDiskMask(pm, bdt);
    return;
}

//----------------------------------------------------------------------------------------
void StarGravSourceTerm(Mesh* pm, const Real bdt) {

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
        Real fcoe=-mp_.gm0/rc/rc/rc;
        
        // Implement the smoothing function
        Real rcv2 = (rc-mp_.rs)*(rc-mp_.rs);
        Real fsmooth = rcv2/(rcv2+mp_.gravsmooth*mp_.gravsmooth);
        if (rc<mp_.rs) fcoe=0.;

        Real f_x1 = fcoe*x1v*fsmooth;
        Real f_x2 = fcoe*x2v*fsmooth;
        Real f_x3 = fcoe*x3v*fsmooth;

        // Now set the source terms
        u0_(m,IM1,k,j,i) += bdt*w0_(m,IDN,k,j,i)*f_x1;
        u0_(m,IM2,k,j,i) += bdt*w0_(m,IDN,k,j,i)*f_x2;
        u0_(m,IM3,k,j,i) += bdt*w0_(m,IDN,k,j,i)*f_x3;
        
        if(mp_.is_ideal) {
            u0_(m,IEN,k,j,i) += bdt*w0_(m,IDN,k,j,i)*
                            (w0_(m,IM1,k,j,i)*f_x1 + 
                             w0_(m,IM2,k,j,i)*f_x2 + 
                             w0_(m,IM3,k,j,i)*f_x3);
        }
            
        u0_(m,IDN,k,j,i) = fmax(u0_(m,IDN,k,j,i),rho_floor(mp_,rc));
        
    }); // end par_for

} // end star source terms 

//----------------------------------------------------------------------------------------
void StarMask(Mesh* pm, const Real bdt) {

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

    // Could be more efficient with the masking function...see GRMHD for boolean masking array
    par_for("pgen_starmask",DevExeSpace(),0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
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
        
        Real rad(0.0),phi(0.0),z(0.0);
        Real v1(0.0),v2(0.0),v3(0.0);

        GetCylCoord(mp_,rad,phi,z,x1v,x2v,x3v);
        Real rc=sqrt(x1v*x1v+x2v*x2v+x3v*x3v);
        
        if (rc<mp_.rs) {
            u0_(m,IDN,k,j,i) = DenStarCyl(mp_,rad, phi, z);
            VelStarCyl(mp_,rad,phi,z,v1,v2,v3);
            u0_(m,IM1,k,j,i) = v1*u0_(m,IDN,k,j,i);
            u0_(m,IM2,k,j,i) = v2*u0_(m,IDN,k,j,i);
            u0_(m,IM3,k,j,i) = v3*u0_(m,IDN,k,j,i);
            
            if (mp_.is_ideal) {
                u0_(m,IEN,k,j,i) = PoverR(mp_,rad, phi, z)*u0_(m,IDN,k,j,i)/(mp_.gamma_gas - 1.0)+
                                0.5*(SQR(u0_(m,IM1,k,j,i))+SQR(u0_(m,IM2,k,j,i))+SQR(u0_(m,IM3,k,j,i)))/u0_(m,IDN,k,j,i);
            
                if (mp_.magnetic_fields_enabled) {
                    u0_(m,IEN,k,j,i) = u0_(m,IEN,k,j,i)+0.5*(SQR(bcc0_(m,IBX,k,j,i))+
                                    SQR(bcc0_(m,IBY,k,j,i))+SQR(bcc0_(m,IBZ,k,j,i)));
                }
            }
        }
            
    }); // end par_for

} // end stellar mask  

//----------------------------------------------------------------------------------------
void InnerDiskMask(Mesh* pm, const Real bdt) {

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

    // Could be more efficient with the masking function...see GRMHD for boolean masking array
    par_for("pgen_starmask",DevExeSpace(),0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
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
        
        Real rad(0.0),phi(0.0),z(0.0);
        Real v1(0.0),v2(0.0),v3(0.0);

        GetCylCoord(mp_,rad,phi,z,x1v,x2v,x3v);
        Real rc=sqrt(x1v*x1v+x2v*x2v+x3v*x3v);
        
        if (rc<mp_.rmagsph) {

            u0_(m,IDN,k,j,i) = DenDiscCyl(mp_,rad, phi, z);
            VelDiscCyl(mp_, rad, phi, z, v1,v2,v3);

            u0_(m,IM1,k,j,i) = v1*u0_(m,IDN,k,j,i);
            u0_(m,IM2,k,j,i) = v2*u0_(m,IDN,k,j,i);
            u0_(m,IM3,k,j,i) = v3*u0_(m,IDN,k,j,i);
            
            if (mp_.is_ideal) {
                u0_(m,IEN,k,j,i) = PoverR(mp_,rad, phi, z)*u0_(m,IDN,k,j,i)/(mp_.gamma_gas - 1.0)+
                                0.5*(SQR(u0_(m,IM1,k,j,i))+SQR(u0_(m,IM2,k,j,i))+SQR(u0_(m,IM3,k,j,i)))/u0_(m,IDN,k,j,i);
            
                if (mp_.magnetic_fields_enabled) {
                    u0_(m,IEN,k,j,i) = u0_(m,IEN,k,j,i)+0.5*(SQR(bcc0_(m,IBX,k,j,i))+
                                    SQR(bcc0_(m,IBY,k,j,i))+SQR(bcc0_(m,IBZ,k,j,i)));
                }
            }
        }
            
    }); // end par_for

} // end disk mask  

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
        Real p_over_r(0.0);
        GetCylCoord(mp_,rad,phi,z,x1v,x2v,x3v);
        if (mp_.rho0 <= 0.0){
            p_over_r = PoverR(mp_,rad,phi,z);
        } else {
            p_over_r = PoverR(mp_,rad,phi,z);
        }

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

void FixedBC(Mesh *pm) {

      // Start by extracting the mesh block and cell information 
      auto &indcs = pm->mb_indcs;
      auto &size = pm->pmb_pack->pmb->mb_size;
      int &ng = indcs.ng;
      int n1 = indcs.nx1 + 2*ng;
      int n2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng) : 1;
      int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng) : 1;
      int &is = indcs.is;  int &ie  = indcs.ie;
      int &js = indcs.js;  int &je  = indcs.je;
      int &ks = indcs.ks;  int &ke  = indcs.ke;
      auto &mb_bcs = pm->pmb_pack->pmb->mb_bcs;

      // Initialise a pointer to the disc parameter structure   
      auto mp_ = mp;

      int nmb = pm->pmb_pack->nmb_thispack;

      // Select either Hydro or MHD arrays
      DvceArray5D<Real> u0_, w0_;
      if (pm->pmb_pack->phydro != nullptr) {
        u0_ = pm->pmb_pack->phydro->u0;
        w0_ = pm->pmb_pack->phydro->w0;
      }

      // X1 BOUNDARY CONDITIONS ---------------> 

      // Start off by converting all the conservative variables to primitives so everything is synchronised
      pm->pmb_pack->phydro->peos->ConsToPrim(u0_,w0_,false,is-ng,is-1,0,(n2-1),0,(n3-1));
      pm->pmb_pack->phydro->peos->ConsToPrim(u0_,w0_,false,ie+1,ie+ng,0,(n2-1),0,(n3-1));
      par_for("fixed_x1", DevExeSpace(),0,(nmb-1),0,(n3-1),0,(n2-1),0,(ng-1),
      KOKKOS_LAMBDA(int m, int k, int j, int i) {

        // Extract coordinates at inner x1 boundary on each meshblock in the pack
        Real &x1min = size.d_view(m).x1min;
        Real &x1max = size.d_view(m).x1max;
        Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

        Real &x2min = size.d_view(m).x2min;
        Real &x2max = size.d_view(m).x2max;
        Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

        Real &x3min = size.d_view(m).x3min;
        Real &x3max = size.d_view(m).x3max;
        Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);

        Real den(0.0), pgas, ux(0.0), uy(0.0), uz(0.0);
        Real rad(0.0), phi(0.0), z(0.0);
        
        // Inner x1 boundary
        if (mb_bcs.d_view(m,BoundaryFace::inner_x1) == BoundaryFlag::user) {
            
            // Compute the warped primitive variables at this location
            GetCylCoord(mp_,rad, phi, z, x1v, x2v, x3v);
            if (mp_.rho0 > 0.0){
                den = DenDiscCyl(mp_, rad, phi, z);
                VelDiscCyl(mp_, rad, phi, z, ux, uy, uz);
            } else {
                den = DenStarCyl(mp_, rad, phi, z);
                VelStarCyl(mp_, rad, phi, z, ux, uy, uz);
            } 

            w0_(m,IDN,k,j,i) = den;
            w0_(m,IVX,k,j,i) = ux;
            w0_(m,IVY,k,j,i) = uy;
            w0_(m,IVZ,k,j,i) = uz;

            if (mp_.is_ideal) {
                pgas = PoverR(mp_, rad, phi, z)*den/(mp_.gamma_gas-1.0);
                w0_(m,IEN,k,j,i) = pgas;
            }

        }

        // Outer x1 boundary
        x1v = CellCenterX((ie+i+1)-is, indcs.nx1, x1min, x1max);

        if (mb_bcs.d_view(m,BoundaryFace::outer_x1) == BoundaryFlag::user) {

            GetCylCoord(mp_,rad, phi, z, x1v, x2v, x3v);
            if (mp_.rho0 > 0.0){
                den = DenDiscCyl(mp_, rad, phi, z);
                VelDiscCyl(mp_, rad, phi, z, ux, uy, uz);
            } else {
                den = DenStarCyl(mp_, rad, phi, z);
                VelStarCyl(mp_, rad, phi, z, ux, uy, uz);
            } 
            
            // Now set the conserved variables using the primitive variables
            w0_(m,IDN,k,j,(ie+i+1)) = den;
            w0_(m,IVX,k,j,(ie+i+1)) = ux;
            w0_(m,IVY,k,j,(ie+i+1)) = uy;
            w0_(m,IVZ,k,j,(ie+i+1)) = uz;

            if (mp_.is_ideal) {
                pgas = PoverR(mp_, rad, phi, z)*den/(mp_.gamma_gas-1.0);
                w0_(m,IEN,k,j,(ie+i+1)) = pgas;
            }

        }
      });
      // Now synchronise PrimToCons on X1 physical boundary ghost zones
      pm->pmb_pack->phydro->peos->PrimToCons(w0_,u0_,is-ng,is-1,0,(n2-1),0,(n3-1));
      pm->pmb_pack->phydro->peos->PrimToCons(w0_,u0_,ie+1,ie+ng,0,(n2-1),0,(n3-1));


      // X2 BOUNDARY CONDITIONS ---------------> 

      pm->pmb_pack->phydro->peos->ConsToPrim(u0_,w0_,false,0,(n1-1),js-ng,js-1,0,(n3-1));
      pm->pmb_pack->phydro->peos->ConsToPrim(u0_,w0_,false,0,(n1-1),je+1,je+ng,0,(n3-1));
      par_for("fixed_x2", DevExeSpace(),0,(nmb-1),0,(n3-1),0,(ng-1),0,(n1-1),
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
        // inner x2 boundary
        Real &x1min = size.d_view(m).x1min;
        Real &x1max = size.d_view(m).x1max;
        Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

        Real &x2min = size.d_view(m).x2min;
        Real &x2max = size.d_view(m).x2max;
        Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

        Real &x3min = size.d_view(m).x3min;
        Real &x3max = size.d_view(m).x3max;
        Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);

        Real den, pgas, ux, uy, uz;
        Real rad(0.0), phi(0.0), z(0.0);

        // Inner x2 boundary
        if (mb_bcs.d_view(m,BoundaryFace::inner_x2) == BoundaryFlag::user) {
            
            // Compute the warped primitive variables at this location
            GetCylCoord(mp_,rad, phi, z, x1v, x2v, x3v);
            if (mp_.rho0 > 0.0){
                den = DenDiscCyl(mp_, rad, phi, z);
                VelDiscCyl(mp_, rad, phi, z, ux, uy, uz);
            } else {
                den = DenStarCyl(mp_, rad, phi, z);
                VelStarCyl(mp_, rad, phi, z, ux, uy, uz);
            } 

            // Now set the conserved variables using the primitive variables
            w0_(m,IDN,k,j,i) = den;
            w0_(m,IVX,k,j,i) = ux;
            w0_(m,IVY,k,j,i) = uy;
            w0_(m,IVZ,k,j,i) = uz;

            if (mp_.is_ideal) {
                pgas = PoverR(mp_, rad, phi, z)*den/(mp_.gamma_gas-1.0);
                w0_(m,IEN,k,j,i) = pgas;
            }
        }

        // Outer x2 boundary
        x2v = CellCenterX((je+j+1)-js, indcs.nx2, x2min, x2max);

        if (mb_bcs.d_view(m,BoundaryFace::outer_x2) == BoundaryFlag::user) {
            // Compute the warped primitive variables at this location
            GetCylCoord(mp_,rad, phi, z, x1v, x2v, x3v);
            if (mp_.rho0 > 0.0){
                den = DenDiscCyl(mp_, rad, phi, z);
                VelDiscCyl(mp_, rad, phi, z, ux, uy, uz);
            } else {
                den = DenStarCyl(mp_, rad, phi, z);
                VelStarCyl(mp_, rad, phi, z, ux, uy, uz);
            } 

            // Now set the conserved variables using the primitive variables
            w0_(m,IDN,k,(je+j+1),i) = den;
            w0_(m,IVX,k,(je+j+1),i) = ux;
            w0_(m,IVY,k,(je+j+1),i) = uy;
            w0_(m,IVZ,k,(je+j+1),i) = uz;

            if (mp_.is_ideal){
                pgas = PoverR(mp_, rad, phi, z)*den/(mp_.gamma_gas-1.0);
                w0_(m,IEN,k,(je+j+1),i) = pgas;
            }
        }
      });
      pm->pmb_pack->phydro->peos->PrimToCons(w0_,u0_,0,(n1-1),js-ng,js-1,0,(n3-1));
      pm->pmb_pack->phydro->peos->PrimToCons(w0_,u0_,0,(n1-1),je+1,je+ng,0,(n3-1));

      // X3 BOUNDARY CONDITIONS ---------------> 

      pm->pmb_pack->phydro->peos->ConsToPrim(u0_,w0_,false,0,(n1-1),0,(n2-1),ks-ng,ks-1);
      pm->pmb_pack->phydro->peos->ConsToPrim(u0_,w0_,false,0,(n1-1),0,(n2-1),ke+1,ke+ng);
      par_for("fixed_ix3", DevExeSpace(),0,(nmb-1),0,(ng-1),0,(n2-1),0,(n1-1),
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
        // inner x3 boundary
        Real &x1min = size.d_view(m).x1min;
        Real &x1max = size.d_view(m).x1max;
        Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

        Real &x2min = size.d_view(m).x2min;
        Real &x2max = size.d_view(m).x2max;
        Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

        Real &x3min = size.d_view(m).x3min;
        Real &x3max = size.d_view(m).x3max;
        Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);

        // Initialise primitive variables
        Real den, pgas, ux, uy, uz;
        Real rad(0.0), phi(0.0), z(0.0);

        // Inner x3 boundary
        if (mb_bcs.d_view(m,BoundaryFace::inner_x3) == BoundaryFlag::user) {
            // Compute the warped primitive variables at this location
            GetCylCoord(mp_,rad, phi, z, x1v, x2v, x3v);
            if (mp_.rho0 > 0.0){
                den = DenDiscCyl(mp_, rad, phi, z);
                VelDiscCyl(mp_, rad, phi, z, ux, uy, uz);
            } else {
                den = DenStarCyl(mp_, rad, phi, z);
                VelStarCyl(mp_, rad, phi, z, ux, uy, uz);
            } 

            // Now set the conserved variables using the primitive variables
            w0_(m,IDN,k,j,i) = den;
            w0_(m,IVX,k,j,i) = ux;
            w0_(m,IVY,k,j,i) = uy;
            w0_(m,IVZ,k,j,i) = uz;

            if (mp_.is_ideal) {
                pgas = PoverR(mp_, rad, phi, z)*den/(mp_.gamma_gas-1.0);
                w0_(m,IEN,k,j,i) = pgas;
            }
        }

        // Outer x3 boundary
        x3v = CellCenterX((ke+k+1)-ks, indcs.nx3, x3min, x3max);

        if (mb_bcs.d_view(m,BoundaryFace::outer_x3) == BoundaryFlag::user) {
            // Compute the warped primitive variables at this location
            GetCylCoord(mp_,rad, phi, z, x1v, x2v, x3v);
            if (mp_.rho0 > 0.0){
                den = DenDiscCyl(mp_, rad, phi, z);
                VelDiscCyl(mp_, rad, phi, z, ux, uy, uz);
            } else {
                den = DenStarCyl(mp_, rad, phi, z);
                VelStarCyl(mp_, rad, phi, z, ux, uy, uz);
            } 

            // Now set the conserved variables using the primitive variables
            w0_(m,IDN,(ke+k+1),j,i) = den;
            w0_(m,IVX,(ke+k+1),j,i) = ux;
            w0_(m,IVY,(ke+k+1),j,i) = uy;
            w0_(m,IVZ,(ke+k+1),j,i) = uz;
            if (mp_.is_ideal) {
                pgas = PoverR(mp_, rad, phi, z)*den/(mp_.gamma_gas-1.0);
                w0_(m,IEN,(ke+k+1),j,i) = pgas;
            }
        }
      });
      pm->pmb_pack->phydro->peos->PrimToCons(w0_,u0_,0,(n1-1),0,(n2-1),ks-ng,ks-1);
      pm->pmb_pack->phydro->peos->PrimToCons(w0_,u0_,0,(n1-1),0,(n2-1),ke+1,ke+ng);

  return;
}