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
    Real dfloor, rho_floor1, rho_floor_slope1, rho_floor2, rho_floor_slope2;
    Real rad_in_cutoff, rad_in_smooth, rad_out_cutoff, rad_out_smooth;
    Real Omega0;
    Real rs, smoothin, gravsmooth;
    Real Rmin, Ri, Ro, Rmax;
    Real thmin, thi, tho, thmax;
    Real origid, rmagsph, denstar;
    Real mm;
    bool is_ideal;
    bool magnetic_fields_enabled;
    };

    my_params mp;

} // End of namespace

KOKKOS_INLINE_FUNCTION
Real rho_floor(struct my_params mp, const Real r);

// prototypes for user-defined BCs and source functions
void StarGravSourceTerm(Mesh* pm, const Real bdt);
void CoolingSourceTerms(Mesh* pm, const Real bdt);
void MySourceTerms(Mesh* pm, const Real bdt);
void MyEfieldMask(Mesh* pm);

void StarMask(Mesh* pm, const Real bdt);
void InnerDiskMask(Mesh* pm, const Real bdt);
void FixedHydroBC(Mesh *pm);
void FixedMHDBC(Mesh *pm);

//----------------------------------------------------------------------------------------
//! \fn
//! \brief Problem Generator for warped disc experiments. Sets initial conditions for an equilibrium
//! which is then rotated in spherical shells to introduce a radially dependent tilt and twait profile.
//! Compile with '-D PROBLEM=warp_disc' to enroll as a user-specific problem generator.
//----------------------------------------------------------------------------------------

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {

    MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;

    // Now enroll user source terms and boundary conditions if specified
    if (user_srcs) {
        user_srcs_func = MySourceTerms;
    }

    if (user_bcs) {
        user_bcs_func = FixedMHDBC;
    }

    if (user_esrcs && (pmbp->pmhd != nullptr)) {
        user_esrcs_func = MyEfieldMask;
    }

    if (pmbp->phydro != nullptr) {
        EOS_Data &eos = pmbp->phydro->peos->eos_data;
        mp.is_ideal = eos.is_ideal;
    } 

    if (pmbp->pmhd != nullptr) {
        EOS_Data &eos = pmbp->pmhd->peos->eos_data;
        mp.is_ideal = eos.is_ideal;
    }

    mp.denstar = pin->GetOrAddReal("problem","denstar",0.0);
    mp.dslope = pin->GetOrAddReal("problem","dslope",0.0);
    mp.gamma_gas = pin->GetReal("mhd","gamma");
    mp.gm0 = pin->GetOrAddReal("problem","gm0",1.0);
    if (pmbp->pmhd != nullptr) mp.magnetic_fields_enabled = true;
    else mp.magnetic_fields_enabled = false;
    mp.mm = pin->GetOrAddReal("problem","mm",0.0);
    mp.origid = pin->GetOrAddReal("problem","origid",0.0);
    if (mp.is_ideal){
        mp.p0_over_r0 = SQR(pin->GetOrAddReal("problem","h_over_r0",0.1));
    } else { mp.p0_over_r0 = SQR(pin->GetReal("mhd","iso_sound_speed")); }
    mp.qslope = pin->GetOrAddReal("problem","qslope",0.0);
    mp.r0 = pin->GetOrAddReal("problem","r0",1.0);
    mp.rad_in_cutoff = pin->GetOrAddReal("problem","rad_in_cutoff",0.0);
    mp.rad_in_smooth = pin->GetOrAddReal("problem","rad_in_smooth",0.1);
    mp.rad_out_cutoff = pin->GetOrAddReal("problem","rad_out_cutoff",0.0);
    mp.rad_out_smooth = pin->GetOrAddReal("problem","rad_out_smooth",0.1);
    mp.rho0 = pin->GetReal("problem","rho0");
    mp.rho_floor1 = pin->GetReal("problem","rho_floor1");
    mp.rho_floor_slope1 = pin->GetOrAddReal("problem","rho_floor_slope1",0.0);
    mp.rho_floor2 = pin->GetOrAddReal("problem","rho_floor2",1.0e6);
    mp.rho_floor_slope2 = pin->GetOrAddReal("problem","rho_floor_slope2", 5.5);
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
    int nmb = pmbp->nmb_thispack;

    // Initialise a pointer to the disc parameter structure   
    auto mp_ = mp;

    // If restarting then end initialisation here
    if (restart) return;

    // Select either Hydro or MHD and extract the arrays - set on the device specifically since this is where the calculations
    // are going to be done anyway. 
    DvceArray5D<Real> u0_, w0_;
    if (pmbp->phydro != nullptr){
        u0_ = pmbp->phydro->u0;
        w0_ = pmbp->phydro->w0;
    }   else if (pmbp->pmhd != nullptr){
        u0_ = pmbp->pmhd->u0;
        w0_ = pmbp->pmhd->w0;
    }

    // initialize conservative variables for new run ---------------------------------------
    par_for("magnetosphere_pgen",DevExeSpace(),0,(nmb-1),ks,ke,js,je,is,ie,
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

        // apply the density floor
        Real rc = sqrt(rad*rad+z*z);
        den = fmax(den,rho_floor(mp_,rc));

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

    // initialize magnetic field if required ---------------------------------------
    if (pmbp->pmhd != nullptr) {

        // compute vector potential over all faces
        int ncells1 = indcs.nx1 + 2*(indcs.ng);
        int ncells2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
        int ncells3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;
        DvceArray4D<Real> a1, a2, a3;
        Kokkos::realloc(a1, nmb,ncells3,ncells2,ncells1);
        Kokkos::realloc(a2, nmb,ncells3,ncells2,ncells1);
        Kokkos::realloc(a3, nmb,ncells3,ncells2,ncells1);

        auto &nghbr = pmbp->pmb->nghbr;
        auto &mblev = pmbp->pmb->mb_lev;

        par_for("pgen_vector_potential", DevExeSpace(), 0,nmb-1,ks,ke+1,js,je+1,is,ie+1,
        KOKKOS_LAMBDA(int m, int k, int j, int i) {
            Real &x1min = size.d_view(m).x1min;
            Real &x1max = size.d_view(m).x1max;
            int nx1 = indcs.nx1;
            Real x1v = CellCenterX(i-is, nx1, x1min, x1max);
            Real x1f   = LeftEdgeX(i  -is, nx1, x1min, x1max);

            Real &x2min = size.d_view(m).x2min;
            Real &x2max = size.d_view(m).x2max;
            int nx2 = indcs.nx2;
            Real x2v = CellCenterX(j-js, nx2, x2min, x2max);
            Real x2f   = LeftEdgeX(j  -js, nx2, x2min, x2max);

            Real &x3min = size.d_view(m).x3min;
            Real &x3max = size.d_view(m).x3max;
            int nx3 = indcs.nx3;
            Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);
            Real x3f   = LeftEdgeX(k  -ks, nx3, x3min, x3max);

            Real dx1 = size.d_view(m).dx1;
            Real dx2 = size.d_view(m).dx2;
            Real dx3 = size.d_view(m).dx3;

            a1(m,k,j,i) = A1(mp_, x1v, x2f, x3f);
            a2(m,k,j,i) = A2(mp_, x1f, x2v, x3f);
            a3(m,k,j,i) = A3(mp_, x1f, x2f, x3v);

            // When neighboring MeshBock is at finer level, compute vector potential as sum of
            // values at fine grid resolution.  This guarantees flux on shared fine/coarse
            // faces is identical.

            // Correct A1 at x2-faces, x3-faces, and x2x3-edges
            if ((nghbr.d_view(m,8 ).lev > mblev.d_view(m) && j==js) ||
                (nghbr.d_view(m,9 ).lev > mblev.d_view(m) && j==js) ||
                (nghbr.d_view(m,10).lev > mblev.d_view(m) && j==js) ||
                (nghbr.d_view(m,11).lev > mblev.d_view(m) && j==js) ||
                (nghbr.d_view(m,12).lev > mblev.d_view(m) && j==je+1) ||
                (nghbr.d_view(m,13).lev > mblev.d_view(m) && j==je+1) ||
                (nghbr.d_view(m,14).lev > mblev.d_view(m) && j==je+1) ||
                (nghbr.d_view(m,15).lev > mblev.d_view(m) && j==je+1) ||
                (nghbr.d_view(m,24).lev > mblev.d_view(m) && k==ks) ||
                (nghbr.d_view(m,25).lev > mblev.d_view(m) && k==ks) ||
                (nghbr.d_view(m,26).lev > mblev.d_view(m) && k==ks) ||
                (nghbr.d_view(m,27).lev > mblev.d_view(m) && k==ks) ||
                (nghbr.d_view(m,28).lev > mblev.d_view(m) && k==ke+1) ||
                (nghbr.d_view(m,29).lev > mblev.d_view(m) && k==ke+1) ||
                (nghbr.d_view(m,30).lev > mblev.d_view(m) && k==ke+1) ||
                (nghbr.d_view(m,31).lev > mblev.d_view(m) && k==ke+1) ||
                (nghbr.d_view(m,40).lev > mblev.d_view(m) && j==js && k==ks) ||
                (nghbr.d_view(m,41).lev > mblev.d_view(m) && j==js && k==ks) ||
                (nghbr.d_view(m,42).lev > mblev.d_view(m) && j==je+1 && k==ks) ||
                (nghbr.d_view(m,43).lev > mblev.d_view(m) && j==je+1 && k==ks) ||
                (nghbr.d_view(m,44).lev > mblev.d_view(m) && j==js && k==ke+1) ||
                (nghbr.d_view(m,45).lev > mblev.d_view(m) && j==js && k==ke+1) ||
                (nghbr.d_view(m,46).lev > mblev.d_view(m) && j==je+1 && k==ke+1) ||
                (nghbr.d_view(m,47).lev > mblev.d_view(m) && j==je+1 && k==ke+1)) {
                Real xl = x1v + 0.25*dx1;
                Real xr = x1v - 0.25*dx1;
                a1(m,k,j,i) = 0.5*(A1(mp_, xl,x2f,x3f) + A1(mp_, xr,x2f,x3f));
            }

            // Correct A2 at x1-faces, x3-faces, and x1x3-edges
            if ((nghbr.d_view(m,0 ).lev > mblev.d_view(m) && i==is) ||
                (nghbr.d_view(m,1 ).lev > mblev.d_view(m) && i==is) ||
                (nghbr.d_view(m,2 ).lev > mblev.d_view(m) && i==is) ||
                (nghbr.d_view(m,3 ).lev > mblev.d_view(m) && i==is) ||
                (nghbr.d_view(m,4 ).lev > mblev.d_view(m) && i==ie+1) ||
                (nghbr.d_view(m,5 ).lev > mblev.d_view(m) && i==ie+1) ||
                (nghbr.d_view(m,6 ).lev > mblev.d_view(m) && i==ie+1) ||
                (nghbr.d_view(m,7 ).lev > mblev.d_view(m) && i==ie+1) ||
                (nghbr.d_view(m,24).lev > mblev.d_view(m) && k==ks) ||
                (nghbr.d_view(m,25).lev > mblev.d_view(m) && k==ks) ||
                (nghbr.d_view(m,26).lev > mblev.d_view(m) && k==ks) ||
                (nghbr.d_view(m,27).lev > mblev.d_view(m) && k==ks) ||
                (nghbr.d_view(m,28).lev > mblev.d_view(m) && k==ke+1) ||
                (nghbr.d_view(m,29).lev > mblev.d_view(m) && k==ke+1) ||
                (nghbr.d_view(m,30).lev > mblev.d_view(m) && k==ke+1) ||
                (nghbr.d_view(m,31).lev > mblev.d_view(m) && k==ke+1) ||
                (nghbr.d_view(m,32).lev > mblev.d_view(m) && i==is && k==ks) ||
                (nghbr.d_view(m,33).lev > mblev.d_view(m) && i==is && k==ks) ||
                (nghbr.d_view(m,34).lev > mblev.d_view(m) && i==ie+1 && k==ks) ||
                (nghbr.d_view(m,35).lev > mblev.d_view(m) && i==ie+1 && k==ks) ||
                (nghbr.d_view(m,36).lev > mblev.d_view(m) && i==is && k==ke+1) ||
                (nghbr.d_view(m,37).lev > mblev.d_view(m) && i==is && k==ke+1) ||
                (nghbr.d_view(m,38).lev > mblev.d_view(m) && i==ie+1 && k==ke+1) ||
                (nghbr.d_view(m,39).lev > mblev.d_view(m) && i==ie+1 && k==ke+1)) {
                Real xl = x2v + 0.25*dx2;
                Real xr = x2v - 0.25*dx2;
                a2(m,k,j,i) = 0.5*(A2(mp_, x1f,xl,x3f) + A2(mp_, x1f,xr,x3f));
            }

            // Correct A3 at x1-faces, x2-faces, and x1x2-edges
            if ((nghbr.d_view(m,0 ).lev > mblev.d_view(m) && i==is) ||
                (nghbr.d_view(m,1 ).lev > mblev.d_view(m) && i==is) ||
                (nghbr.d_view(m,2 ).lev > mblev.d_view(m) && i==is) ||
                (nghbr.d_view(m,3 ).lev > mblev.d_view(m) && i==is) ||
                (nghbr.d_view(m,4 ).lev > mblev.d_view(m) && i==ie+1) ||
                (nghbr.d_view(m,5 ).lev > mblev.d_view(m) && i==ie+1) ||
                (nghbr.d_view(m,6 ).lev > mblev.d_view(m) && i==ie+1) ||
                (nghbr.d_view(m,7 ).lev > mblev.d_view(m) && i==ie+1) ||
                (nghbr.d_view(m,8 ).lev > mblev.d_view(m) && j==js) ||
                (nghbr.d_view(m,9 ).lev > mblev.d_view(m) && j==js) ||
                (nghbr.d_view(m,10).lev > mblev.d_view(m) && j==js) ||
                (nghbr.d_view(m,11).lev > mblev.d_view(m) && j==js) ||
                (nghbr.d_view(m,12).lev > mblev.d_view(m) && j==je+1) ||
                (nghbr.d_view(m,13).lev > mblev.d_view(m) && j==je+1) ||
                (nghbr.d_view(m,14).lev > mblev.d_view(m) && j==je+1) ||
                (nghbr.d_view(m,15).lev > mblev.d_view(m) && j==je+1) ||
                (nghbr.d_view(m,16).lev > mblev.d_view(m) && i==is && j==js) ||
                (nghbr.d_view(m,17).lev > mblev.d_view(m) && i==is && j==js) ||
                (nghbr.d_view(m,18).lev > mblev.d_view(m) && i==ie+1 && j==js) ||
                (nghbr.d_view(m,19).lev > mblev.d_view(m) && i==ie+1 && j==js) ||
                (nghbr.d_view(m,20).lev > mblev.d_view(m) && i==is && j==je+1) ||
                (nghbr.d_view(m,21).lev > mblev.d_view(m) && i==is && j==je+1) ||
                (nghbr.d_view(m,22).lev > mblev.d_view(m) && i==ie+1 && j==je+1) ||
                (nghbr.d_view(m,23).lev > mblev.d_view(m) && i==ie+1 && j==je+1)) {
                Real xl = x3v + 0.25*dx3;
                Real xr = x3v - 0.25*dx3;
                a3(m,k,j,i) = 0.5*(A3(mp_, x1f,x2f,xl) + A3(mp_, x1f,x2f,xr));
            }
        
        });

        auto &b0_ = pmbp->pmhd->b0;
        par_for("pgen_b0", DevExeSpace(), 0,nmb-1,ks,ke,js,je,is,ie,
        KOKKOS_LAMBDA(int m, int k, int j, int i) {
            // Compute face-centered fields from curl(A).
            Real dx1 = size.d_view(m).dx1;
            Real dx2 = size.d_view(m).dx2;
            Real dx3 = size.d_view(m).dx3;

            b0_.x1f(m,k,j,i) = ((a3(m,k,j+1,i) - a3(m,k,j,i))/dx2 -
                                (a2(m,k+1,j,i) - a2(m,k,j,i))/dx3);
            b0_.x2f(m,k,j,i) = ((a1(m,k+1,j,i) - a1(m,k,j,i))/dx3 -
                                (a3(m,k,j,i+1) - a3(m,k,j,i))/dx1);
            b0_.x3f(m,k,j,i) = ((a2(m,k,j,i+1) - a2(m,k,j,i))/dx1 -
                                (a1(m,k,j+1,i) - a1(m,k,j,i))/dx2);

            // Include extra face-component at edge of block in each direction
            if (i==ie) {
                b0_.x1f(m,k,j,i+1) = ((a3(m,k,j+1,i+1) - a3(m,k,j,i+1))/dx2 -
                                     (a2(m,k+1,j,i+1) - a2(m,k,j,i+1))/dx3);
            }
            if (j==je) {
                b0_.x2f(m,k,j+1,i) = ((a1(m,k+1,j+1,i) - a1(m,k,j+1,i))/dx3 -
                                     (a3(m,k,j+1,i+1) - a3(m,k,j+1,i))/dx1);
            }
            if (k==ke) {
                b0_.x3f(m,k+1,j,i) = ((a2(m,k+1,j,i+1) - a2(m,k+1,j,i))/dx1 -
                                     (a1(m,k+1,j+1,i) - a1(m,k+1,j,i))/dx2);
            }
        });

        if (mp.is_ideal) {
            par_for("bcc_e", DevExeSpace(), 0,(nmb-1),ks,ke,js,je,is,ie,
            KOKKOS_LAMBDA(int m, int k, int j, int i) {
                u0_(m,IEN,k,j,i) += 0.5*(SQR(0.5*(b0_.x1f(m,k,j,i) + b0_.x1f(m,k,j,i+1))) +
                                    SQR(0.5*(b0_.x2f(m,k,j,i) + b0_.x2f(m,k,j+1,i))) +
                                    SQR(0.5*(b0_.x3f(m,k,j,i) + b0_.x3f(m,k+1,j,i))));
            });
        }

    } // End of magnetic field initialization

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
        a1 = mp.mm/rc/rc/rc*(-1.*x2);
        return(a1);
    }

    //----------------------------------------------------------------------------------------
    KOKKOS_INLINE_FUNCTION
    static Real A2(struct my_params mp, const Real x1, const Real x2, const Real x3) {
        Real a2=0.0;
        Real rc = fmax(sqrt(x1*x1+x2*x2+x3*x3),mp.rs);
        a2 = mp.mm/rc/rc/rc*(+1.*x1);
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
    Real rhofloor = mp.rho_floor1;
    if (rc > mp.rs) rhofloor = mp.rho_floor1*pow(rc/mp.r0, mp.rho_floor_slope1);
    if (mp.mm != 0. && rc > mp.rs) rhofloor += mp.rho_floor2*pow(rc/mp.r0,mp.rho_floor_slope2);
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
void MyEfieldMask(Mesh* pm) {

    // Capture variables for kernel - e.g. indices for looping over the meshblocks and the size of the meshblocks.
    auto &indcs = pm->mb_indcs;
    int &is = indcs.is; int &ie = indcs.ie;
    int &js = indcs.js; int &je = indcs.je;
    int &ks = indcs.ks; int &ke = indcs.ke;
    MeshBlockPack *pmbp = pm->pmb_pack;
    auto &size = pmbp->pmb->mb_size;

    DvceArray4D<Real> e1_,e2_,e3_;
    if (pmbp->pmhd != nullptr) {
        e1_ = pmbp->pmhd->efld.x1e;
        e2_ = pmbp->pmhd->efld.x2e;
        e3_ = pmbp->pmhd->efld.x3e;
    }

    // Now set a local parameter struct for lambda capturing
    auto mp_ = mp;

    // Define E1, E2, E3 on corners
    // Note e1[is:ie,  js:je+1,ks:ke+1]
    //      e2[is:ie+1,js:je,  ks:ke+1]
    //      e3[is:ie+1,js:je+1,ks:ke  ]

    par_for("pgen_e1mask", DevExeSpace(), 0,(pmbp->nmb_thispack-1),ks,ke+1,js,je+1,is,ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {

        // Extract the cell center and left aligned edge coordinates coordinates
        Real &x1min = size.d_view(m).x1min;
        Real &x1max = size.d_view(m).x1max;
        Real x1v    = CellCenterX(i-is, indcs.nx1, x1min, x1max);

        Real &x2min = size.d_view(m).x2min;
        Real &x2max = size.d_view(m).x2max;
        Real x2f    = LeftEdgeX(j-js, indcs.nx2, x2min, x2max);

        Real &x3min = size.d_view(m).x3min;
        Real &x3max = size.d_view(m).x3max;
        Real x3f    = LeftEdgeX(k-ks, indcs.nx3, x3min, x3max);

        Real rc = sqrt(x1v*x1v+x2f*x2f+x3f*x3f);

        if (rc<mp_.rs) {

            Real rad(0.0),phi(0.0),z(0.0);
            Real vx(0.0),vy(0.0),vz(0.0);
            Real By(0.0),Bz(0.0);
            GetCylCoord(mp_,rad,phi,z,x1v,x2f,x3f);

            // Set the stellar velocity at this location
            VelStarCyl(mp_,rad,phi,z,vx,vy,vz);

            // Set the stellar interior magnetic field
            By = 0.0;
            Bz = 2*mp_.mm/pow(mp_.rs,3);

            // E1=-(v X B)=VzBy-VyBz
            e1_(m,k,j,i) = vz*By - vy*Bz;

        }
            
    }); // end par_for

    par_for("pgen_e2mask", DevExeSpace(), 0,(pmbp->nmb_thispack-1),ks,ke+1,js,je,is,ie+1,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {

        Real &x1min = size.d_view(m).x1min;
        Real &x1max = size.d_view(m).x1max;
        Real x1f    = LeftEdgeX(i-is, indcs.nx1, x1min, x1max);

        Real &x2min = size.d_view(m).x2min;
        Real &x2max = size.d_view(m).x2max;
        Real x2v    = CellCenterX(j-js, indcs.nx2, x2min, x2max);

        Real &x3min = size.d_view(m).x3min;
        Real &x3max = size.d_view(m).x3max;
        Real x3f    = LeftEdgeX(k-ks, indcs.nx3, x3min, x3max);

        Real rc = sqrt(x1f*x1f+x2v*x2v+x3f*x3f);

        if (rc<mp_.rs) {

            Real rad(0.0),phi(0.0),z(0.0);
            Real vx(0.0),vy(0.0),vz(0.0);
            Real Bx(0.0),Bz(0.0);
            GetCylCoord(mp_,rad,phi,z,x1f,x2v,x3f);

            // Set the stellar velocity at this location
            VelStarCyl(mp_,rad,phi,z,vx,vy,vz);

            // Set the interior star magnetic field
            Bx = 0.0;
            Bz = 2*mp_.mm/pow(mp_.rs,3);

            // E2=-(v X B)=VxBz-VzBx
            e2_(m,k,j,i) = vx*Bz - vz*Bx;

        }
            
    }); // end par_for

    par_for("pgen_e3mask", DevExeSpace(), 0,(pmbp->nmb_thispack-1),ks,ke,js,je+1,is,ie+1,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {

        Real &x1min = size.d_view(m).x1min;
        Real &x1max = size.d_view(m).x1max;
        Real x1f    = LeftEdgeX(i-is, indcs.nx1, x1min, x1max);

        Real &x2min = size.d_view(m).x2min;
        Real &x2max = size.d_view(m).x2max;
        Real x2f    = LeftEdgeX(j-js, indcs.nx2, x2min, x2max);

        Real &x3min = size.d_view(m).x3min;
        Real &x3max = size.d_view(m).x3max;
        Real x3v    = CellCenterX(k-ks, indcs.nx3, x3min, x3max);

        Real rc = sqrt(x1f*x1f+x2f*x2f+x3v*x3v);

        if (rc<mp_.rs) {

            Real rad(0.0),phi(0.0),z(0.0);
            Real vx(0.0),vy(0.0),vz(0.0);
            Real Bx(0.0),By(0.0);
            GetCylCoord(mp_,rad,phi,z,x1f,x2f,x3v);

            // Set the stellar velocity at this location
            VelStarCyl(mp_,rad,phi,z,vx,vy,vz);

            // Set the interior star magnetic field
            Bx = 0.0;
            By = 0.0;

            // E3=-(v X B)=VyBx-VxBy
            e3_(m,k,j,i) = vy*Bx - vx*By;

        }
            
    }); // end par_for

    return;

} // end E field mask  

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

void FixedHydroBC(Mesh *pm) {

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
            
            // Compute the primitive variables at this location
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
                pgas = PoverR(mp_, rad, phi, z)*den;
                w0_(m,IEN,k,j,i) = pgas/(mp_.gamma_gas - 1.0);
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
                pgas = PoverR(mp_, rad, phi, z)*den;
                w0_(m,IEN,k,j,(ie+i+1)) = pgas/(mp_.gamma_gas - 1.0);
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
            
            // Compute the primitive variables at this location
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
                pgas = PoverR(mp_, rad, phi, z)*den;
                w0_(m,IEN,k,j,i) = pgas/(mp_.gamma_gas - 1.0);
            }
        }

        // Outer x2 boundary
        x2v = CellCenterX((je+j+1)-js, indcs.nx2, x2min, x2max);

        if (mb_bcs.d_view(m,BoundaryFace::outer_x2) == BoundaryFlag::user) {
            // Compute the primitive variables at this location
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
                pgas = PoverR(mp_, rad, phi, z)*den;
                w0_(m,IEN,k,(je+j+1),i) = pgas/(mp_.gamma_gas - 1.0);
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
                pgas = PoverR(mp_, rad, phi, z)*den;
                w0_(m,IEN,k,j,i) = pgas/(mp_.gamma_gas - 1.0);
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
                pgas = PoverR(mp_, rad, phi, z)*den;
                w0_(m,IEN,(ke+k+1),j,i) = pgas/(mp_.gamma_gas - 1.0);
            }
        }
      });
      pm->pmb_pack->phydro->peos->PrimToCons(w0_,u0_,0,(n1-1),0,(n2-1),ks-ng,ks-1);
      pm->pmb_pack->phydro->peos->PrimToCons(w0_,u0_,0,(n1-1),0,(n2-1),ke+1,ke+ng);

  return;
}

//----------------------------------------------------------------------------------------
//! \fn FixedDiscBC
//  \brief Sets boundary condition on surfaces of computational domain
// Note quantities at boundaries are held fixed to initial condition values

void FixedMHDBC(Mesh *pm) {

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

  // Initialise the MHD arrays
  DvceArray5D<Real> u0_, w0_;
  auto &b0_ = pm->pmb_pack->pmhd->b0;
  auto &bcc_ = pm->pmb_pack->pmhd->bcc0;

  if (pm->pmb_pack->pmhd != nullptr) {
    u0_ = pm->pmb_pack->pmhd->u0;
    w0_ = pm->pmb_pack->pmhd->w0;
  }

  // X1 BOUNDARY CONDITIONS ---------------> 

  par_for("mhd_bc_x1", DevExeSpace(),0,(nmb-1),0,(n3-1),0,(n2-1),
  KOKKOS_LAMBDA(int m, int k, int j) {
    if (mb_bcs.d_view(m,BoundaryFace::inner_x1) == BoundaryFlag::user) {
    for (int i=0; i<ng; ++i) {
        b0_.x1f(m,k,j,is-i-1) = b0_.x1f(m,k,j,is);
        b0_.x2f(m,k,j,is-i-1) = b0_.x2f(m,k,j,is);
        if (j == n2-1) {b0_.x2f(m,k,j+1,is-i-1) = b0_.x2f(m,k,j+1,is);}
        b0_.x3f(m,k,j,is-i-1) = b0_.x3f(m,k,j,is);
        if (k == n3-1) {b0_.x3f(m,k+1,j,is-i-1) = b0_.x3f(m,k+1,j,is);}
    }
    }
    if (mb_bcs.d_view(m,BoundaryFace::outer_x1) == BoundaryFlag::user) {
    for (int i=0; i<ng; ++i) {
        b0_.x1f(m,k,j,ie+i+2) = b0_.x1f(m,k,j,ie+1);
        b0_.x2f(m,k,j,ie+i+1) = b0_.x2f(m,k,j,ie);
        if (j == n2-1) {b0_.x2f(m,k,j+1,ie+i+1) = b0_.x2f(m,k,j+1,ie);}
        b0_.x3f(m,k,j,ie+i+1) = b0_.x3f(m,k,j,ie);
        if (k == n3-1) {b0_.x3f(m,k+1,j,ie+i+1) = b0_.x3f(m,k+1,j,ie);}
    }
    }
  });

  // ConsToPrim over all X1 ghost zones *and* at the innermost/outermost X1-active zones
  // of Meshblocks, even if Meshblock face is not at the edge of computational domain
  pm->pmb_pack->pmhd->peos->ConsToPrim(u0_,b0_,w0_,bcc_,false,is-ng,is,0,(n2-1),0,(n3-1));
  pm->pmb_pack->pmhd->peos->ConsToPrim(u0_,b0_,w0_,bcc_,false,ie,ie+ng,0,(n2-1),0,(n3-1));

  par_for("hd_bc_x1", DevExeSpace(),0,(nmb-1),0,(n3-1),0,(n2-1),0,(ng-1),
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

    Real den(0.0), ux(0.0), uy(0.0), uz(0.0);
    Real rad(0.0), phi(0.0), z(0.0);

    // Inner x1 boundary
    if (mb_bcs.d_view(m,BoundaryFace::inner_x1) == BoundaryFlag::user) {
            
        // Compute primitive variables at this location
        GetCylCoord(mp_,rad, phi, z, x1v, x2v, x3v);
        if (mp_.rho0 > 0.0){
            den = DenDiscCyl(mp_, rad, phi, z);
            VelDiscCyl(mp_, rad, phi, z, ux, uy, uz);
        } else {
            den = DenStarCyl(mp_, rad, phi, z);
            VelStarCyl(mp_, rad, phi, z, ux, uy, uz);
        } 

        // Apply density floor
        Real rc = sqrt(x1v*x1v + x2v*x2v + x3v*x3v);
        den = fmax(den,rho_floor(mp_,rc));

        // Now set the conserved variables using the primitive variables
        w0_(m,IDN,k,j,i) = den;
        w0_(m,IVX,k,j,i) = ux;
        w0_(m,IVY,k,j,i) = uy;
        w0_(m,IVZ,k,j,i) = uz;
        if (mp_.is_ideal) {
            w0_(m,IEN,k,j,i) = PoverR(mp_, rad, phi, z)*den/(mp_.gamma_gas - 1.0);
        }
    }

    // Outer x1 boundary
    x1v = CellCenterX((ie+i+1)-is, indcs.nx1, x1min, x1max);

    if (mb_bcs.d_view(m,BoundaryFace::outer_x1) == BoundaryFlag::user) {
        
        // Compute primitive variables at this location
        GetCylCoord(mp_,rad, phi, z, x1v, x2v, x3v);
        if (mp_.rho0 > 0.0){
            den = DenDiscCyl(mp_, rad, phi, z);
            VelDiscCyl(mp_, rad, phi, z, ux, uy, uz);
        } else {
            den = DenStarCyl(mp_, rad, phi, z);
            VelStarCyl(mp_, rad, phi, z, ux, uy, uz);
        } 

        // Apply density floor
        Real rc = sqrt(x1v*x1v + x2v*x2v + x3v*x3v);
        den = fmax(den,rho_floor(mp_,rc));

        // Now set the conserved variables using the primitive variables
        w0_(m,IDN,k,j,(ie+i+1)) = den;
        w0_(m,IVX,k,j,(ie+i+1)) = ux;
        w0_(m,IVY,k,j,(ie+i+1)) = uy;
        w0_(m,IVZ,k,j,(ie+i+1)) = uz;
        if (mp_.is_ideal) {
            w0_(m,IEN,k,j,(ie+i+1)) = PoverR(mp_, rad, phi, z)*den/(mp_.gamma_gas - 1.0);
        }
    }
  });
  // PrimToCons on X1 ghost zones
  pm->pmb_pack->pmhd->peos->PrimToCons(w0_,bcc_,u0_,is-ng,is-1,0,(n2-1),0,(n3-1));
  pm->pmb_pack->pmhd->peos->PrimToCons(w0_,bcc_,u0_,ie+1,ie+ng,0,(n2-1),0,(n3-1));

  // X2 BOUNDARY CONDITIONS ---------------> 

  par_for("mhd_bc_x2", DevExeSpace(),0,(nmb-1),0,(n3-1),0,(n1-1),
    KOKKOS_LAMBDA(int m, int k, int i) {
      if (mb_bcs.d_view(m,BoundaryFace::inner_x2) == BoundaryFlag::user) {
        for (int j=0; j<ng; ++j) {
          b0_.x1f(m,k,js-j-1,i) = b0_.x1f(m,k,js,i);
          if (i == n1-1) {b0_.x1f(m,k,js-j-1,i+1) = b0_.x1f(m,k,js,i+1);}
          b0_.x2f(m,k,js-j-1,i) = b0_.x2f(m,k,js,i);
          b0_.x3f(m,k,js-j-1,i) = b0_.x3f(m,k,js,i);
          if (k == n3-1) {b0_.x3f(m,k+1,js-j-1,i) = b0_.x3f(m,k+1,js,i);}
        }
      }
      if (mb_bcs.d_view(m,BoundaryFace::outer_x2) == BoundaryFlag::user) {
        for (int j=0; j<ng; ++j) {
          b0_.x1f(m,k,je+j+1,i) = b0_.x1f(m,k,je,i);
          if (i == n1-1) {b0_.x1f(m,k,je+j+1,i+1) = b0_.x1f(m,k,je,i+1);}
          b0_.x2f(m,k,je+j+2,i) = b0_.x2f(m,k,je+1,i);
          b0_.x3f(m,k,je+j+1,i) = b0_.x3f(m,k,je,i);
          if (k == n3-1) {b0_.x3f(m,k+1,je+j+1,i) = b0_.x3f(m,k+1,je,i);}
        }
      }
    });

  pm->pmb_pack->pmhd->peos->ConsToPrim(u0_,b0_,w0_,bcc_,false,0,(n1-1),js-ng,js,0,(n3-1));
  pm->pmb_pack->pmhd->peos->ConsToPrim(u0_,b0_,w0_,bcc_,false,0,(n1-1),je,je+ng,0,(n3-1));

  par_for("hd_bc_x2", DevExeSpace(),0,(nmb-1),0,(n3-1),0,(ng-1),0,(n1-1),
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

    Real den(0.0), ux(0.0), uy(0.0), uz(0.0);
    Real rad(0.0), phi(0.0), z(0.0);

    // Inner x2 boundary
    if (mb_bcs.d_view(m,BoundaryFace::inner_x2) == BoundaryFlag::user) {

        // Compute primitive variables at this location
        GetCylCoord(mp_,rad, phi, z, x1v, x2v, x3v);
        if (mp_.rho0 > 0.0){
            den = DenDiscCyl(mp_, rad, phi, z);
            VelDiscCyl(mp_, rad, phi, z, ux, uy, uz);
        } else {
            den = DenStarCyl(mp_, rad, phi, z);
            VelStarCyl(mp_, rad, phi, z, ux, uy, uz);
        } 

        // Apply density floor
        Real rc = sqrt(x1v*x1v + x2v*x2v + x3v*x3v);
        den = fmax(den,rho_floor(mp_,rc));

        // Now set the conserved variables using the primitive variables
        w0_(m,IDN,k,j,i) = den;
        w0_(m,IVX,k,j,i) = ux;
        w0_(m,IVY,k,j,i) = uy;
        w0_(m,IVZ,k,j,i) = uz;
        if (mp_.is_ideal) {
            w0_(m,IEN,k,j,i) = PoverR(mp_, rad, phi, z)*den/(mp_.gamma_gas - 1.0);
        }
    }

    // Outer x2 boundary
    x2v = CellCenterX((je+j+1)-js, indcs.nx2, x2min, x2max);

    if (mb_bcs.d_view(m,BoundaryFace::outer_x2) == BoundaryFlag::user) {

        // Compute primitive variables at this location
        GetCylCoord(mp_,rad, phi, z, x1v, x2v, x3v);
        if (mp_.rho0 > 0.0){
            den = DenDiscCyl(mp_, rad, phi, z);
            VelDiscCyl(mp_, rad, phi, z, ux, uy, uz);
        } else {
            den = DenStarCyl(mp_, rad, phi, z);
            VelStarCyl(mp_, rad, phi, z, ux, uy, uz);
        } 

        // Apply density floor
        Real rc = sqrt(x1v*x1v + x2v*x2v + x3v*x3v);
        den = fmax(den,rho_floor(mp_,rc));

        // Now set the conserved variables using the primitive variables
        w0_(m,IDN,k,(je+j+1),i) = den;
        w0_(m,IVX,k,(je+j+1),i) = ux;
        w0_(m,IVY,k,(je+j+1),i) = uy;
        w0_(m,IVZ,k,(je+j+1),i) = uz;
        if (mp_.is_ideal) {
            w0_(m,IEN,k,(je+j+1),i) = PoverR(mp_, rad, phi, z)*den/(mp_.gamma_gas - 1.0);
        }
    }
  });

  pm->pmb_pack->pmhd->peos->PrimToCons(w0_,bcc_,u0_,0,(n1-1),js-ng,js-1,0,(n3-1));
  pm->pmb_pack->pmhd->peos->PrimToCons(w0_,bcc_,u0_,0,(n1-1),je+1,je+ng,0,(n3-1));

  // X3 BOUNDARY CONDITIONS ---------------> 

  par_for("mhd_bc_x3", DevExeSpace(),0,(nmb-1),0,(n2-1),0,(n1-1),
    KOKKOS_LAMBDA(int m, int j, int i) {
      if (mb_bcs.d_view(m,BoundaryFace::inner_x3) == BoundaryFlag::user) {
        for (int k=0; k<ng; ++k) {
          b0_.x1f(m,ks-k-1,j,i) = b0_.x1f(m,ks,j,i);
          if (i == n1-1) {b0_.x1f(m,ks-k-1,j,i+1) = b0_.x1f(m,ks,j,i+1);}
          b0_.x2f(m,ks-k-1,j,i) = b0_.x2f(m,ks,j,i);
          if (j == n2-1) {b0_.x2f(m,ks-k-1,j+1,i) = b0_.x2f(m,ks,j+1,i);}
          b0_.x3f(m,ks-k-1,j,i) = b0_.x3f(m,ks,j,i);
        }
      }
      if (mb_bcs.d_view(m,BoundaryFace::outer_x3) == BoundaryFlag::user) {
        for (int k=0; k<ng; ++k) {
          b0_.x1f(m,ke+k+1,j,i) = b0_.x1f(m,ke,j,i);
          if (i == n1-1) {b0_.x1f(m,ke+k+1,j,i+1) = b0_.x1f(m,ke,j,i+1);}
          b0_.x2f(m,ke+k+1,j,i) = b0_.x2f(m,ke,j,i);
          if (j == n2-1) {b0_.x2f(m,ke+k+1,j+1,i) = b0_.x2f(m,ke,j+1,i);}
          b0_.x3f(m,ke+k+2,j,i) = b0_.x3f(m,ke+1,j,i);
        }
      }
    });

  pm->pmb_pack->pmhd->peos->ConsToPrim(u0_,b0_,w0_,bcc_,false,0,(n1-1),0,(n2-1),ks-ng,ks);
  pm->pmb_pack->pmhd->peos->ConsToPrim(u0_,b0_,w0_,bcc_,false,0,(n1-1),0,(n2-1),ke,ke+ng);

  par_for("hd_bc_x3", DevExeSpace(),0,(nmb-1),0,(ng-1),0,(n2-1),0,(n1-1),
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

    Real den(0.0), ux(0.0), uy(0.0), uz(0.0);
    Real rad(0.0), phi(0.0), z(0.0);

    // Inner x3 boundary
    if (mb_bcs.d_view(m,BoundaryFace::inner_x3) == BoundaryFlag::user) {
        
        // Compute primitive variables at this location
        GetCylCoord(mp_,rad, phi, z, x1v, x2v, x3v);
        if (mp_.rho0 > 0.0){
            den = DenDiscCyl(mp_, rad, phi, z);
            VelDiscCyl(mp_, rad, phi, z, ux, uy, uz);
        } else {
            den = DenStarCyl(mp_, rad, phi, z);
            VelStarCyl(mp_, rad, phi, z, ux, uy, uz);
        } 

        // Apply density floor
        Real rc = sqrt(x1v*x1v + x2v*x2v + x3v*x3v);
        den = fmax(den,rho_floor(mp_,rc));

        // Now set the conserved variables using the primitive variables
        w0_(m,IDN,k,j,i) = den;
        w0_(m,IVX,k,j,i) = ux;
        w0_(m,IVY,k,j,i) = uy;
        w0_(m,IVZ,k,j,i) = uz;
        if (mp_.is_ideal) {
            w0_(m,IEN,k,j,i) = PoverR(mp_, rad, phi, z)*den/(mp_.gamma_gas - 1.0);
        }
    }

    // Outer x3 boundary
    x3v = CellCenterX((ke+k+1)-ks, indcs.nx3, x3min, x3max);

    if (mb_bcs.d_view(m,BoundaryFace::outer_x3) == BoundaryFlag::user) {

        // Compute primitive variables at this location
        GetCylCoord(mp_,rad, phi, z, x1v, x2v, x3v);
        if (mp_.rho0 > 0.0){
            den = DenDiscCyl(mp_, rad, phi, z);
            VelDiscCyl(mp_, rad, phi, z, ux, uy, uz);
        } else {
            den = DenStarCyl(mp_, rad, phi, z);
            VelStarCyl(mp_, rad, phi, z, ux, uy, uz);
        } 

        // Apply density floor
        Real rc = sqrt(x1v*x1v + x2v*x2v + x3v*x3v);
        den = fmax(den,rho_floor(mp_,rc));

        // Now set the conserved variables using the primitive variables
        w0_(m,IDN,(ke+k+1),j,i) = den;
        w0_(m,IVX,(ke+k+1),j,i) = ux;
        w0_(m,IVY,(ke+k+1),j,i) = uy;
        w0_(m,IVZ,(ke+k+1),j,i) = uz;
        if (mp_.is_ideal) {
            w0_(m,IEN,(ke+k+1),j,i) = PoverR(mp_, rad, phi, z)*den/(mp_.gamma_gas - 1.0);
        }
    }
  });
    
  pm->pmb_pack->pmhd->peos->PrimToCons(w0_,bcc_,u0_,0,(n1-1),0,(n2-1),ks-ng,ks-1);
  pm->pmb_pack->pmhd->peos->PrimToCons(w0_,bcc_,u0_,0,(n1-1),0,(n2-1),ke+1,ke+ng);

  return;
} 