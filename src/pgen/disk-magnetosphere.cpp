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
#include <iomanip>
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

#include "geodesic-grid/geodesic_grid.hpp"  // CF: Add this for integrating over sphere
#include "geodesic-grid/spherical_grid.hpp" // CF: Add this for integrating over sphere

// prototypes for functions used internally to this pgen
namespace {

    KOKKOS_INLINE_FUNCTION
    static void GetCylCoord(struct my_params mp, Real &rad,Real &phi,Real &z, const Real x1, const Real x2, const Real x3);

    KOKKOS_INLINE_FUNCTION
    static void RotateCart(struct my_params mp, Real &x1rot, Real &x2rot, Real &x3rot, const Real x1, const Real x2,const Real x3,const Real rot);

    KOKKOS_INLINE_FUNCTION
    static Real DenDiscCyl(struct my_params mp, const Real rad, const Real phi, const Real z);

    KOKKOS_INLINE_FUNCTION
    static Real DenStarCyl(struct my_params mp, const Real rad, const Real phi, const Real z);

    KOKKOS_INLINE_FUNCTION
    static Real PoverR(struct my_params mp, const Real rad);

    KOKKOS_INLINE_FUNCTION
    static void VelDiscCyl(struct my_params mp, const Real rad, const Real phi, const Real z, Real &v1, Real &v2, Real &v3);

    KOKKOS_INLINE_FUNCTION
    static void VelStarCyl(struct my_params mp, const Real rad, const Real phi, const Real z, Real &v1, Real &v2, Real &v3);
    
    KOKKOS_INLINE_FUNCTION
    static void VelStar(struct my_params mp, const Real x1, const Real x2, const Real x3, Real &v1_star, Real &v2_star, Real &v3_star);

    KOKKOS_INLINE_FUNCTION
    static void DenDiscPlusStar(struct my_params mp, const Real x1, const Real x2, const Real x3, Real &den);

    KOKKOS_INLINE_FUNCTION
    static void VelDiscPlusStar(struct my_params mp, const Real x1, const Real x2, const Real x3, Real &ux, Real &uy, Real &uz);

    KOKKOS_INLINE_FUNCTION
    static Real A1(struct my_params mp, const Real x1, const Real x2, const Real x3);

    KOKKOS_INLINE_FUNCTION
    static Real A2(struct my_params mp, const Real x1, const Real x2, const Real x3);

    KOKKOS_INLINE_FUNCTION
    static Real A3(struct my_params mp, const Real x1, const Real x2, const Real x3);

    KOKKOS_INLINE_FUNCTION
    static void Bfield(struct my_params mp, const Real x1, const Real x2, const Real x3, const Real mmx, const Real mmy, const Real mmz, Real &bx, Real &by,  Real &bz);

    // Initialize global instance of the parameter structure
    // pgen_struct disc_params;

    struct my_params {
    Real thetaw, thetab;
    Real gm0, r0, rho0, dslope, p0_over_r0, qslope, tcool, gamma_gas;
    Real dfloor, rho_floor1, rho_floor_slope1, rho_floor2, rho_floor_slope2;
    Real rfix, rad_in_cutoff, rad_in_smooth, rad_out_cutoff, rad_out_smooth;
    Real sig_star_disc;
    Real Omega0;
    Real rs, smoothin, gravsmooth;
    Real Rmin, Ri, Ro, Rmax;
    Real thmin, thi, tho, thmax;
    Real origid, rmagsph, denstar;
    Real mm, rb, delta;
    Real fofc_scalar_tau;  // timescale for damping FOFC diagnostic scalar (0 = off)
    bool is_ideal;
    bool magnetic_fields_enabled;
    bool avg_grid_bfields;
    int mag_option;
    Real disc_mask_rin;   // inner disc-only mask radius (negative = off)
    Real disc_mask_rout;  // outer disc-only mask radius (negative = off)
    bool cooling_direct_set;      // if true, set temperature to desired profile; else relax toward target
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
void MyHistFunc(HistoryData *pdata, Mesh *pm);

void StarMask(Mesh* pm, const Real bdt);
void DiscOnlyMask(Mesh* pm, const Real bdt);
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

    if (user_constraint) {
        user_constraint_func = CoolingSourceTerms;
    }

    if (user_hist) {

        // Spherical Grid for user-defined history
        auto &grids = spherical_grids;
        Real rslice1 = 1.0;
        Real rslice2 = 2.0;
        grids.push_back(std::make_unique<SphericalGrid>(pmbp, 5, rslice1));
        grids.push_back(std::make_unique<SphericalGrid>(pmbp, 5, rslice2));
        user_hist_func = MyHistFunc;
    }

    if (pmbp->phydro != nullptr) {
        EOS_Data &eos = pmbp->phydro->peos->eos_data;
        mp.is_ideal = eos.is_ideal;
    } 

    if (pmbp->pmhd != nullptr) {
        EOS_Data &eos = pmbp->pmhd->peos->eos_data;
        mp.is_ideal = eos.is_ideal;
    }

    mp.thetaw = pin->GetOrAddReal("problem","thetaw",0.0);
    mp.thetab = pin->GetOrAddReal("problem","thetab",0.0);
    mp.denstar = pin->GetOrAddReal("problem","denstar",0.0);
    mp.dslope = pin->GetOrAddReal("problem","dslope",0.0);
    mp.gamma_gas = pin->GetReal("mhd","gamma");
    mp.gm0 = pin->GetOrAddReal("problem","gm0",1.0);
    if (pmbp->pmhd != nullptr) mp.magnetic_fields_enabled = true;
    else mp.magnetic_fields_enabled = false;
    mp.mm = pin->GetOrAddReal("problem","mm",0.0);
    mp.rb = pin->GetOrAddReal("problem","rb",0.05);
    mp.delta = pin->GetOrAddReal("problem","delta",1.0);
    mp.origid = pin->GetOrAddReal("problem","origid",0.0);
    if (mp.is_ideal){
        mp.p0_over_r0 = SQR(pin->GetOrAddReal("problem","h_over_r0",0.1));
    } else { mp.p0_over_r0 = SQR(pin->GetReal("mhd","iso_sound_speed")); }
    mp.qslope = pin->GetOrAddReal("problem","qslope",0.0);
    mp.r0 = pin->GetOrAddReal("problem","r0",1.0);
    mp.rfix = pin->GetOrAddReal("problem", "rfix",0.1);
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
    mp.rs = pin->GetOrAddReal("problem", "rstar",0.1);
    mp.gravsmooth = pin->GetOrAddReal("problem","gravsmooth",0.1);
    mp.tcool = pin->GetOrAddReal("problem","tcool",0.0);
    mp.fofc_scalar_tau = pin->GetOrAddReal("problem","fofc_scalar_tau",0.0);
    mp.sig_star_disc = pin->GetOrAddReal("problem","sig_star_disc",0.1);
    mp.avg_grid_bfields = pin->GetOrAddBoolean("problem","avg_grid_bfields",false);
    mp.mag_option = pin->GetOrAddInteger("problem","mag_option",1);
    mp.disc_mask_rin  = pin->GetOrAddReal("problem", "disc_mask_rin",  -1.0);
    mp.disc_mask_rout = pin->GetOrAddReal("problem", "disc_mask_rout", -1.0);
    mp.cooling_direct_set = pin->GetOrAddBoolean("problem", "cooling_direct_set", false);

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

        Real den(0.0), ux(0.0), uy(0.0), uz(0.0);

        // Compute density and velocity in the disc+star system
        DenDiscPlusStar(mp_, x1, x2, x3, den);
        VelDiscPlusStar(mp_, x1, x2, x3, ux, uy, uz);

        // set the conserved variables
        u0_(m,IDN,k,j,i) = den;
        u0_(m,IM1,k,j,i) = den*ux;
        u0_(m,IM2,k,j,i) = den*uy;
        u0_(m,IM3,k,j,i) = den*uz;

        if (mp_.is_ideal) {
            // Compute cylindrical radius
            Real rad=sqrt(x1*x1 + x2*x2);
            Real p_over_r = PoverR(mp_, rad);
            u0_(m,IEN,k,j,i) = p_over_r*den/(mp_.gamma_gas - 1.0)
                               +0.5*(SQR(u0_(m,IM1,k,j,i))
                               +SQR(u0_(m,IM2,k,j,i))
                               +SQR(u0_(m,IM3,k,j,i)))/u0_(m,IDN,k,j,i) ;
        }

    });

    // Initialize FOFC diagnostic passive scalar to 0 (when used)
    if (pmbp->pmhd != nullptr && pmbp->pmhd->nscalars >= 1) {
        int nmhd = pmbp->pmhd->nmhd;
        auto u0_scalar_ = pmbp->pmhd->u0;
        par_for("magnetosphere_pgen_fofc_scalar", DevExeSpace(), 0, (nmb - 1), ks, ke, js, je, is, ie,
        KOKKOS_LAMBDA(int m, int k, int j, int i) {
            u0_scalar_(m, nmhd, k, j, i) = 0.0;
        });
    }

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

        par_for("pgen_vector_potential", DevExeSpace(), 0,(nmb-1),ks,ke+1,js,je+1,is,ie+1,
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

        if (mp_.is_ideal) {

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
    KOKKOS_INLINE_FUNCTION //CF:CHECKED
    static void GetCylCoord(struct my_params mp, Real &rad,Real &phi,Real &z, const Real x1, const Real x2, const Real x3) {
        rad=sqrt(x1*x1 + x2*x2);
        phi=atan2(x2,x1);
        z=x3;
        return;
    }

    //----------------------------------------------------------------------------------------
    //! Rotate the cartesian coordinates by some angle to get the tilted coordinates
    KOKKOS_INLINE_FUNCTION //CF:CHECKED
    static void RotateCart(struct my_params mp, Real &x1rot,Real &x2rot,Real &x3rot, const Real x1, const Real x2,const Real x3, const Real theta) {

        // The adopted convetion here rotates the underlying axes counter-clockwise by an angle theta about the y-axis. 
        // Hence one can bring the axes into alignment with the spin axis or initial dipole axis.
        // Note that converting the cartesian vectors in this tilted frame back to the standard frame requires a rotation of components by -theta.
        
        // Rotation matrix R about the y axis
        Real cost=cos(theta);
        Real sint=sin(theta);

        x1rot=cost*x1 + sint*x3;
        x2rot=x2;
        x3rot=-sint*x1 + cost*x3;

        return;
    }

    //----------------------------------------------------------------------------------------
    KOKKOS_INLINE_FUNCTION //CF:CHECKED
    static Real DenDiscCyl(struct my_params mp, const Real rad, const Real phi, const Real z) {
        
        // Compute the density profile in cylindrical coordinates
        // Vertical hydrostatic equilibrium (Nelson et al. 2013) 

        Real den(0.0);
        Real r = fmax(rad, mp.rs);
        Real p_over_r = mp.p0_over_r0;
        if (mp.is_ideal) p_over_r = PoverR(mp, r);
        Real denmid = mp.rho0*std::pow(r/mp.r0,mp.dslope);

        // Cutoff based on the spherical radius
        Real rc = sqrt(rad*rad + z*z);

        // Inner magnetosphere exponential cutoff
        if (rc < mp.rad_in_cutoff) {
            Real cutoff = exp(-SQR((rc-mp.rad_in_cutoff)/mp.rad_in_smooth));
            denmid *= cutoff;
        }

        // Outer disc exponential cutoff
        if (rc > mp.rad_out_cutoff) {
            Real cutoff = exp(-SQR((rc - mp.rad_out_cutoff)/mp.rad_out_smooth));
            denmid *= cutoff;
        }

        den = denmid*std::exp(mp.gm0/p_over_r*(1./std::sqrt(SQR(r)+SQR(z))-1./r));
        
        return den;
    }

    //----------------------------------------------------------------------------------------
    KOKKOS_INLINE_FUNCTION //CF:CHECKED
    static Real DenStarCyl(struct my_params mp, const Real rad, const Real phi, const Real z) {
        
        // Add the stellar density profile component
        Real den(0.0);
        Real rc = sqrt(rad*rad+z*z);  // spherical radius

        if (rc<mp.rmagsph) {

            Real sinsq = rad*rad/rc/rc;
            Real csq0 = PoverR(mp, mp.rs);
            Real pre0=mp.denstar*csq0;   // reference pressure
            Real dr = mp.rs/100.;        // integration step size

            // Real rint = mp.rs;        // integrate from stellar surface
            // Real pre = pre0*exp(0.5*mp.origid*mp.origid*mp.rs*mp.rs*sinsq/csq0);  // pressure at stellar surface
            // if (rc < mp.rs) {
            //     // analytic solution inside the star
            //     pre = pre0*exp(0.5*mp.origid*mp.origid*rad*rad/csq0);
            // } else {
                
            //     // integrate stellar envelope out from the stellar surface towards rc
            //     while(rint<rc) {
            //         pre += -dr*mp.gm0/rint/rint *
            //             (rint-mp.rs) * (rint-mp.rs)/((rint-mp.rs)*(rint-mp.rs)+mp.gravsmooth*mp.gravsmooth) *
            //             pre/csq0 + dr*mp.origid*mp.origid*rint*sinsq*pre/csq0;
            //         rint = rint + dr;
            //     }
            // }

            Real rint = 0.0;             // integrate from stellar interior
            Real pre = pre0;             // pressure at stellar interior
            // integrate stellar envelope out from the stellar interior towards rc
            while (rint<rc) {
                if (rint < mp.rs) {
                    // inside the star
                    pre = pre+dr*mp.origid*mp.origid*rint*sinsq*pre/csq0;
                } else {
                    // outside the star
                    pre = pre-dr*mp.gm0/rint/rint*(rint-mp.rs)*(rint-mp.rs)/((rint-mp.rs)*(rint-mp.rs)+mp.gravsmooth*mp.gravsmooth)*pre/csq0
                          +dr*mp.origid*mp.origid*rint*sinsq*pre/csq0;
                }
                rint += + dr;

            den = pre/csq0;
            }
        }
        
        return den;
    }

    //----------------------------------------------------------------------------------------
    KOKKOS_INLINE_FUNCTION //CF:CHECKED
    static Real PoverR(struct my_params mp, const Real rad) {
        Real poverr;
        Real r = fmax(rad, mp.rfix);
        poverr = mp.p0_over_r0*std::pow(r/mp.r0, mp.qslope);
        return poverr;
    }

    //----------------------------------------------------------------------------------------
    KOKKOS_INLINE_FUNCTION //CF:CHECKED
    static void VelDiscCyl(struct my_params mp, const Real rad, const Real phi, const Real z, Real &v1, Real &v2, Real &v3) {
        
        Real r = fmax(rad, mp.rs);
        Real rc = sqrt(r*r+z*z);

        // Old method for power law
        Real p_over_r = PoverR(mp, r);
        Real vel = (mp.dslope+mp.qslope)*p_over_r/(mp.gm0/r) + (1.0+mp.qslope) - mp.qslope*r/sqrt(r*r+z*z);
        vel = sqrt(mp.gm0/r)*sqrt(vel);

        // Testing new method for balance with pressure gradients
        // Real dR = fmin(mp.rad_in_smooth, mp.rad_out_smooth)/100;
        // Real dPdr = (PoverR(mp, r+dR) * DenDiscCyl(mp, r+dR,phi,z) - PoverR(mp, r-dR) * DenDiscCyl(mp, r - dR,phi,z))/(2 * dR);
        // Real vel = sqrt(fmax(mp.gm0*r*r/rc/rc/rc+r/DenDiscCyl(mp, r, phi, z)*dPdr,0.0));

        v1=-vel*sin(phi);
        v2=+vel*cos(phi);
        v3=0.0;

        return;
    }

    //----------------------------------------------------------------------------------------
    KOKKOS_INLINE_FUNCTION //CF:CHECKED
    static void VelStarCyl(struct my_params mp, const Real rad, const Real phi, const Real z, Real &v1, Real &v2, Real &v3) {
        
        // The rigid body velocity of the star in coordinates aligned with the stellar rotation axis.
        Real vel(0.0);
        vel = mp.origid*rad; // rigid rotation

        v1=-vel*sin(phi);
        v2=+vel*cos(phi);
        v3=0.0;

        return;
    }

    //----------------------------------------------------------------------------------------
    KOKKOS_INLINE_FUNCTION //CF:CHECKED
    static void VelStar(struct my_params mp, const Real x1, const Real x2, const Real x3, Real &v1_star, Real &v2_star, Real &v3_star) {
        // The stellar velocity in the original cartesian frame.
        Real x1w(0.0), x2w(0.0), x3w(0.0);
        Real v1_starw(0.0), v2_starw(0.0), v3_starw(0.0);
        Real radw(0.0), phiw(0.0), zw(0.0);  // coordinates in frame aligned with stellar rotation axis

        // rotate to the frame aligned with the stellar rotation axis
        RotateCart(mp, x1w, x2w, x3w, x1, x2, x3, mp.thetaw);
        GetCylCoord(mp,radw, phiw, zw, x1w, x2w, x3w);
        // add the stellar velocity component at this location
        VelStarCyl(mp, radw, phiw, zw, v1_starw, v2_starw, v3_starw);
        // rotate the velocity components back to the original frame
        RotateCart(mp, v1_star, v2_star, v3_star, v1_starw, v2_starw, v3_starw, -mp.thetaw);

        return;
    }

    //----------------------------------------------------------------------------------------
    KOKKOS_INLINE_FUNCTION
    static void DenDiscPlusStar(struct my_params mp, const Real x1, const Real x2, const Real x3, Real &den) {

        Real x1w(0.0), x2w(0.0), x3w(0.0);
        Real rad(0.0), phi(0.0), z(0.0);
        Real radw(0.0), phiw(0.0), zw(0.0);  // coordinates in frame aligned with stellar rotation axis

        // get the cylindrical coodinates corresponding to this cartesian location
        GetCylCoord(mp,rad, phi, z, x1, x2, x3);
    
        // compute the disc density component at this location
        den = DenDiscCyl(mp, rad, phi, z);
        
        // rotate to the frame aligned with the stellar rotation axis
        RotateCart(mp, x1w, x2w, x3w, x1, x2, x3, mp.thetaw);
        GetCylCoord(mp,radw, phiw, zw, x1w, x2w, x3w);
        // add the stellar density component at this location
        den += DenStarCyl(mp,radw,phiw,zw);

        // apply the density floor
        Real rc = sqrt(rad*rad+z*z);
        den = fmax(den,rho_floor(mp,rc));

        return;
    }

    //----------------------------------------------------------------------------------------
    KOKKOS_INLINE_FUNCTION
    static void VelDiscPlusStar(struct my_params mp, const Real x1, const Real x2, const Real x3, Real &ux, Real &uy, Real &uz) {

        Real rad(0.0), phi(0.0), z(0.0);
        Real v1_disc(0.0), v2_disc(0.0), v3_disc(0.0);
        Real v1_star(0.0), v2_star(0.0), v3_star(0.0);

        // compute the disc velocity component at this location
        GetCylCoord(mp,rad, phi, z, x1, x2, x3);
        VelDiscCyl(mp, rad, phi, z, v1_disc, v2_disc, v3_disc);
    
        // compute the stellar velocity component at this location
        VelStar(mp, x1, x2, x3, v1_star, v2_star, v3_star);

        Real rc = sqrt(rad*rad+z*z);

        // Zhu velocity switch setup ----------------------
        if (rc > mp.rmagsph) {
            // outside the magnetosphere, set the velocity to the disc velocity
            ux = v1_disc;
            uy = v2_disc;
            uz = v3_disc;

        } else {
            // inside the inner cutoff, set the velocity to the stellar velocity
            ux = v1_disc*exp(-SQR((rc-mp.rmagsph)/mp.rs));
            uy = v2_disc*exp(-SQR((rc-mp.rmagsph)/mp.rs));
            uz = v3_disc*exp(-SQR((rc-mp.rmagsph)/mp.rs));

            ux += v1_star;
            uy += v2_star;
            uz += v3_star;
        } 

        // // Smooth velocity siwtch setup ----------------------
        // Real sigma = 1/(1+exp((rc - mp.rmagsph)/mp.sig_star_disc));
        // ux = (1-sigma)*v1_disc + sigma*v1_star;
        // uy = (1-sigma)*v2_disc + sigma*v2_star;
        // uz = (1-sigma)*v3_disc + sigma*v3_star;

        return;
    }

    //----------------------------------------------------------------------------------------
    KOKKOS_INLINE_FUNCTION //CF:CHECKED
    static Real A1(struct my_params mp, const Real x1, const Real x2, const Real x3) {
        
        Real a1=0.0;

        if (mp.mag_option == 1) {
            Real x2b = x2;
            Real rc = fmax(sqrt(x1*x1+x2*x2+x3*x3),mp.rs/2);
            a1 = mp.mm/rc/rc/rc*(-1.*x2b*cos(mp.thetab));
        } else if (mp.mag_option == 2) {
            Real x2b = x2;
            Real rc = sqrt(x1*x1+x2*x2+x3*x3);
            Real f = pow(mp.rb,-3)*pow(pow(rc/mp.rb,3*mp.delta)+1,-1/mp.delta);
            a1 = mp.mm* f * (-1.*x2b*cos(mp.thetab));  
        }      
        
        return(a1);
    }

    //----------------------------------------------------------------------------------------
    KOKKOS_INLINE_FUNCTION //CF:CHECKED
    static Real A2(struct my_params mp, const Real x1, const Real x2, const Real x3) {
        
        Real a2=0.0;
        
        if (mp.mag_option == 1) {
            Real x1b = cos(mp.thetab)*x1 + sin(mp.thetab)*x3;
            Real rc = fmax(sqrt(x1*x1+x2*x2+x3*x3),mp.rs/2);
            a2 = mp.mm/rc/rc/rc*(+1.*x1b);
        } else if (mp.mag_option == 2) {
            Real x1b = cos(mp.thetab)*x1 + sin(mp.thetab)*x3;
            Real rc = sqrt(x1*x1+x2*x2+x3*x3);
            Real f = pow(mp.rb,-3)*pow(pow(rc/mp.rb,3*mp.delta)+1,-1/mp.delta);
            a2 = mp.mm* f * (+1.*x1b);
        }
        
        return(a2);
    }

    //----------------------------------------------------------------------------------------
    KOKKOS_INLINE_FUNCTION //CF:CHECKED
    static Real A3(struct my_params mp, const Real x1, const Real x2, const Real x3) {
        
        Real a3=0.0;
        
        if (mp.mag_option == 1) {
            Real x2b = x2;
            Real rc = fmax(sqrt(x1*x1+x2*x2+x3*x3),mp.rs/2);
            a3 = mp.mm/rc/rc/rc*(-1.*x2b*sin(mp.thetab));
        } else if (mp.mag_option == 2) {
            Real x2b = x2;
            Real rc = sqrt(x1*x1+x2*x2+x3*x3);
            Real f = pow(mp.rb,-3)*pow(pow(rc/mp.rb,3*mp.delta)+1,-1/mp.delta);
            a3 = mp.mm* f * (-1.*x2b*sin(mp.thetab));
        }

        return(a3);
    }

    //----------------------------------------------------------------------------------------
    KOKKOS_INLINE_FUNCTION //CF:CHECKED
    static void Bfield(struct my_params mp, const Real x1, const Real x2, const Real x3, const Real mmx, const Real mmy, const Real mmz, Real &bx, Real &by,  Real &bz) {

        if (mp.mag_option == 1) {
            Real rc = sqrt(x1*x1+x2*x2+x3*x3);
            Real rccubed = rc*rc*rc;
            Real rscubed = (mp.rs/2)*(mp.rs/2)*(mp.rs/2);
            Real mdotr = mmx*x1 + mmy*x2 + mmz*x3;

            if (rc < mp.rs/2) {
                bx = 2.*mmx/rscubed;
                by = 2.*mmy/rscubed;
                bz = 2.*mmz/rscubed;
            } else {
                bx = 3.*x1*mdotr/rccubed/rc/rc - mmx/rccubed;
                by = 3.*x2*mdotr/rccubed/rc/rc - mmy/rccubed;
                bz = 3.*x3*mdotr/rccubed/rc/rc - mmz/rccubed;
            }

        } else if (mp.mag_option == 2) {

            Real mdotr = mmx*x1 + mmy*x2 + mmz*x3;

            Real rc = sqrt(x1*x1+x2*x2+x3*x3);
            Real f = pow(mp.rb,-3)*pow(pow(rc/mp.rb,3*mp.delta)+1,-1/mp.delta);
            Real g = -3*pow(mp.rb,-5)*pow(pow(rc/mp.rb,3*mp.delta)+1,-1/mp.delta-1)*pow(rc/mp.rb,3*mp.delta-2);

            bx = 2*mmx*f + g*(mmx*rc*rc-mdotr*x1);
            by = 2*mmy*f + g*(mmy*rc*rc-mdotr*x2);
            bz = 2*mmz*f + g*(mmz*rc*rc-mdotr*x3);
        }

        return;

    } // end B field 

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

void MySourceTerms(Mesh* pm, const Real bdt) { //CF:CHECKED

    StarGravSourceTerm(pm, bdt);
    if (mp.denstar > 0.0) StarMask(pm, bdt);
    if (mp.disc_mask_rin >= 0.0 || mp.disc_mask_rout >= 0.0) DiscOnlyMask(pm, bdt);
    // if(mp.is_ideal && mp.tcool>0.0) CoolingSourceTerms(pm, bdt);

    // FOFC diagnostic: damp conserved scalar (rho*s) with source -bdt*dens*s/tau
    MeshBlockPack *pmbp = pm->pmb_pack;
    if (pmbp->pmhd != nullptr && pmbp->pmhd->nscalars >= 1 && mp.fofc_scalar_tau > 0.0) {
        auto &indcs = pm->mb_indcs;
        int is = indcs.is, ie = indcs.ie, js = indcs.js, je = indcs.je, ks = indcs.ks, ke = indcs.ke;
        int nmhd = pmbp->pmhd->nmhd;
        Real tau = mp.fofc_scalar_tau;
        auto u0_ = pmbp->pmhd->u0;
        par_for("pgen_fofc_scalar_damp", DevExeSpace(), 0, (pmbp->nmb_thispack - 1), ks, ke, js, je, is, ie,
        KOKKOS_LAMBDA(int m, int k, int j, int i) {
            Real dens = u0_(m, IDN, k, j, i);
            Real rho_s = u0_(m, nmhd, k, j, i);
            Real sink = bdt * rho_s / tau;
            u0_(m, nmhd, k, j, i) = fmax(0.0, rho_s - sink);
        });
    }
    
    return;
}

//----------------------------------------------------------------------------------------
void StarGravSourceTerm(Mesh* pm, const Real bdt) { //CF:CHECKED

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
void StarMask(Mesh* pm, const Real bdt) { //CF:CHECKED

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
        
        Real rad=sqrt(x1v*x1v+x2v*x2v);
        Real rc=sqrt(rad*rad+x3v*x3v);
        
        if (rc<mp_.rfix) {

            Real den(0.0), ux(0.0), uy(0.0), uz(0.0);

            // Get the combined density and velocity at this location
            DenDiscPlusStar(mp_, x1v, x2v, x3v, den);
            VelStar(mp_, x1v, x2v, x3v, ux, uy, uz);

            // Assign to conserved varaibles
            u0_(m,IDN,k,j,i) = den;            
            u0_(m,IM1,k,j,i) = u0_(m,IDN,k,j,i)*ux;
            u0_(m,IM2,k,j,i) = u0_(m,IDN,k,j,i)*uy;
            u0_(m,IM3,k,j,i) = u0_(m,IDN,k,j,i)*uz;
            
            if (mp_.is_ideal) {
                u0_(m,IEN,k,j,i) = PoverR(mp_,rad)*u0_(m,IDN,k,j,i)/(mp_.gamma_gas - 1.0)+
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
void DiscOnlyMask(Mesh* pm, const Real bdt) {

    if (mp.disc_mask_rin < 0.0 && mp.disc_mask_rout < 0.0) return;

    auto &indcs = pm->mb_indcs;
    int &is = indcs.is; int &ie = indcs.ie;
    int &js = indcs.js; int &je = indcs.je;
    int &ks = indcs.ks; int &ke = indcs.ke;
    MeshBlockPack *pmbp = pm->pmb_pack;
    auto &size = pmbp->pmb->mb_size;

    auto mp_ = mp;

    DvceArray5D<Real> u0_, w0_, bcc0_;
    if (pmbp->phydro != nullptr) {
        u0_ = pmbp->phydro->u0;
        w0_ = pmbp->phydro->w0;
    } else if (pmbp->pmhd != nullptr) {
        u0_ = pmbp->pmhd->u0;
        w0_ = pmbp->pmhd->w0;
        bcc0_ = pmbp->pmhd->bcc0;
    }

    par_for("pgen_disconlymask", DevExeSpace(), 0, (pmbp->nmb_thispack - 1), ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {

        Real &x1min = size.d_view(m).x1min;
        Real &x1max = size.d_view(m).x1max;
        Real x1v = CellCenterX(i - is, indcs.nx1, x1min, x1max);

        Real &x2min = size.d_view(m).x2min;
        Real &x2max = size.d_view(m).x2max;
        Real x2v = CellCenterX(j - js, indcs.nx2, x2min, x2max);

        Real &x3min = size.d_view(m).x3min;
        Real &x3max = size.d_view(m).x3max;
        Real x3v = CellCenterX(k - ks, indcs.nx3, x3min, x3max);

        Real rad = sqrt(x1v*x1v + x2v*x2v);
        Real rc = sqrt(rad*rad + x3v*x3v);
        Real phi = atan2(x2v, x1v);
        Real z = x3v;

        bool in_inner = (mp_.disc_mask_rin >= 0.0 && rc < mp_.disc_mask_rin);
        bool in_outer = (mp_.disc_mask_rout >= 0.0 && rc > mp_.disc_mask_rout);
        if (!in_inner && !in_outer) return;

        Real den = DenDiscCyl(mp_, rad, phi, z);
        den = fmax(den, rho_floor(mp_, rc));

        Real v1(0.0), v2(0.0), v3(0.0);
        VelDiscCyl(mp_, rad, phi, z, v1, v2, v3);

        u0_(m, IDN, k, j, i) = den;
        u0_(m, IM1, k, j, i) = den * v1;
        u0_(m, IM2, k, j, i) = den * v2;
        u0_(m, IM3, k, j, i) = den * v3;

        if (mp_.is_ideal) {
            u0_(m, IEN, k, j, i) = PoverR(mp_, rad) * den / (mp_.gamma_gas - 1.0)
                + 0.5 * den * (SQR(v1) + SQR(v2) + SQR(v3));
            if (mp_.magnetic_fields_enabled) {
                u0_(m, IEN, k, j, i) += 0.5 * (SQR(bcc0_(m, IBX, k, j, i)) +
                    SQR(bcc0_(m, IBY, k, j, i)) + SQR(bcc0_(m, IBZ, k, j, i)));
            }
        }
    });
}

//----------------------------------------------------------------------------------------
void MyEfieldMask(Mesh* pm) { //CF:CHECKED

    // Capture variables for kernel - e.g. indices for looping over the meshblocks and the size of the meshblocks.
    auto &indcs = pm->mb_indcs;
    int &is = indcs.is; int &ie = indcs.ie;
    int &js = indcs.js; int &je = indcs.je;
    int &ks = indcs.ks; int &ke = indcs.ke;
    MeshBlockPack *pmbp = pm->pmb_pack;
    auto &size = pmbp->pmb->mb_size;
    Real &time = pm->time;

    DvceArray4D<Real> e1_,e2_,e3_;
    if (pmbp->pmhd != nullptr) {
        e1_ = pmbp->pmhd->efld.x1e;
        e2_ = pmbp->pmhd->efld.x2e;
        e3_ = pmbp->pmhd->efld.x3e;
    }

    // Now set a local parameter struct for lambda capturing
    auto mp_ = mp;

    // The magnetic dipole in the frame aligned with the stellar spin
    Real mmxw = mp.mm*sin(mp.thetaw-mp.thetab)*cos(mp.origid*time);
    Real mmyw = mp.mm*sin(mp.thetaw-mp.thetab)*sin(mp.origid*time);
    Real mmzw = mp.mm*cos(mp.thetaw-mp.thetab);
    Real mmx(0.0),mmy(0.0),mmz(0.0);

    // Rotate from stellar spin frame to standard frame
    RotateCart(mp_,mmx,mmy,mmz,mmxw,mmyw,mmzw,-mp_.thetaw);

    auto &b0_ = pmbp->pmhd->b0;

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

        if (rc<mp_.rfix) {

            Real vx(0.0),vy(0.0),vz(0.0);
            Real Bx(0.0),By(0.0),Bz(0.0);

            VelStar(mp_, x1v, x2f, x3f, vx, vy, vz);
            
            if (mp_.avg_grid_bfields) {
                // Average face-centered B fields to edge location
                // E1 edge at (i,j,k) samples from [k-1,k]x[j-1,j]x[i,i+1]
                // Bx (parallel): 8-point average over [k-1,k]x[j-1,j]x[i,i+1]
                // By (perpendicular): 2-point average in k direction at [k-1,k]x[j]
                // Bz (perpendicular): 2-point average in j direction at [j-1,j]x[k]
                Bx = 0.125*(b0_.x1f(m,k-1,j-1,i) + b0_.x1f(m,k-1,j,i) + 
                            b0_.x1f(m,k,j-1,i) + b0_.x1f(m,k,j,i) +
                            b0_.x1f(m,k-1,j-1,i+1) + b0_.x1f(m,k-1,j,i+1) + 
                            b0_.x1f(m,k,j-1,i+1) + b0_.x1f(m,k,j,i+1));
                By = 0.5*(b0_.x2f(m,k-1,j,i) + b0_.x2f(m,k,j,i));
                Bz = 0.5*(b0_.x3f(m,k,j-1,i) + b0_.x3f(m,k,j,i));
            } else {
                Bfield(mp_, x1v, x2f, x3f, mmx, mmy, mmz, Bx, By, Bz);
            }

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

        if (rc<mp_.rfix) {

            Real vx(0.0),vy(0.0),vz(0.0);
            Real Bx(0.0),By(0.0),Bz(0.0);

            VelStar(mp_, x1f, x2v, x3f, vx, vy, vz);
            
            if (mp_.avg_grid_bfields) {
                // Average face-centered B fields to edge location
                // E2 edge at (i,j,k) samples from [k-1,k]x[j,j+1]x[i-1,i]
                // By (parallel): 8-point average over [k-1,k]x[j,j+1]x[i-1,i]
                // Bx (perpendicular): 2-point average in k direction at [k-1,k]x[i]
                // Bz (perpendicular): 2-point average in i direction at [i-1,i]x[k]
                Bx = 0.5*(b0_.x1f(m,k-1,j,i) + b0_.x1f(m,k,j,i));
                By = 0.125*(b0_.x2f(m,k-1,j,i-1) + b0_.x2f(m,k-1,j+1,i-1) + 
                            b0_.x2f(m,k,j,i-1) + b0_.x2f(m,k,j+1,i-1) +
                            b0_.x2f(m,k-1,j,i) + b0_.x2f(m,k-1,j+1,i) + 
                            b0_.x2f(m,k,j,i) + b0_.x2f(m,k,j+1,i));
                Bz = 0.5*(b0_.x3f(m,k,j,i-1) + b0_.x3f(m,k,j,i));
            } else {
                Bfield(mp_, x1f, x2v, x3f, mmx, mmy, mmz, Bx, By, Bz);
            }

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

        if (rc<mp_.rfix) {

            Real vx(0.0),vy(0.0),vz(0.0);
            Real Bx(0.0),By(0.0),Bz(0.0);

            VelStar(mp_, x1f, x2f, x3v, vx, vy, vz);
            
            if (mp_.avg_grid_bfields) {
                // Average face-centered B fields to edge location
                // E3 edge at (i,j,k) samples from [k,k+1]x[j-1,j]x[i-1,i]
                // Bz (parallel): 8-point average over [k,k+1]x[j-1,j]x[i-1,i]
                // Bx (perpendicular): 2-point average in j direction at [j-1,j]x[i]
                // By (perpendicular): 2-point average in i direction at [i-1,i]x[j]
                Bx = 0.5*(b0_.x1f(m,k,j-1,i) + b0_.x1f(m,k,j,i));
                By = 0.5*(b0_.x2f(m,k,j,i-1) + b0_.x2f(m,k,j,i));
                Bz = 0.125*(b0_.x3f(m,k,j-1,i-1) + b0_.x3f(m,k,j,i-1) + 
                            b0_.x3f(m,k,j-1,i) + b0_.x3f(m,k,j,i) +
                            b0_.x3f(m,k+1,j-1,i-1) + b0_.x3f(m,k+1,j,i-1) + 
                            b0_.x3f(m,k+1,j-1,i) + b0_.x3f(m,k+1,j,i));
            } else {
                Bfield(mp_, x1f, x2f, x3v, mmx, mmy, mmz, Bx, By, Bz);
            }

            // E3=-(v X B)=VyBx-VxBy
            e3_(m,k,j,i) = vy*Bx - vx*By;

        }
            
    }); // end par_for

    return;

} // end E field mask   

//----------------------------------------------------------------------------------------
void CoolingSourceTerms(Mesh* pm, const Real bdt) { //CF:CHECKED
    // Implement cooling source terms here if needed.
    // Loop over interior + ghost cells so no further conserved-var communication
    // is required after this user constraint update.

    auto &indcs = pm->mb_indcs;
    int &is = indcs.is;
    int &ie = indcs.ie;
    int &js = indcs.js;
    int &je = indcs.je;
    int &ks = indcs.ks;
    int &ke = indcs.ke;
    int ng = indcs.ng;
    MeshBlockPack *pmbp = pm->pmb_pack;
    auto &size = pmbp->pmb->mb_size;

    // Bounds including ghost zones (so BCs do not need to exchange U after this)
    int il = is - ng;
    int iu = ie + ng;
    int jl = (indcs.nx2 > 1) ? (js - ng) : js;
    int ju = (indcs.nx2 > 1) ? (je + ng) : je;
    int kl = (indcs.nx3 > 1) ? (ks - ng) : ks;
    int ku = (indcs.nx3 > 1) ? (ke + ng) : ke;

    auto mp_ = mp;

    DvceArray5D<Real> u0_, w0_;
    u0_ = pm->pmb_pack->pmhd->u0;
    w0_ = pm->pmb_pack->pmhd->w0;
    auto &b0_ = pmbp->pmhd->b0;

    par_for("pgen_coolsource", DevExeSpace(), 0, (pmbp->nmb_thispack - 1), kl,
            ku, jl, ju, il, iu,
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

        Real mom2 = 0.5*(SQR(u0_(m,IM1,k,j,i)) + SQR(u0_(m,IM2,k,j,i)) + SQR(u0_(m,IM3,k,j,i)));
        Real e_m = 0.0;
        if (mp_.magnetic_fields_enabled) {
            // Compute cell-centered B from updated face-centered B (same as ConToPrim and setup)
            Real bx_cc = 0.5*(b0_.x1f(m,k,j,i) + b0_.x1f(m,k,j,i+1));
            Real by_cc = 0.5*(b0_.x2f(m,k,j,i) + b0_.x2f(m,k,j+1,i));
            Real bz_cc = 0.5*(b0_.x3f(m,k,j,i) + b0_.x3f(m,k+1,j,i));
            e_m = 0.5*(SQR(bx_cc) + SQR(by_cc) + SQR(bz_cc));
        }

        Real dens_u0 = u0_(m,IDN,k,j,i); // Already floored from previous source term
        Real e_k_u0 = mom2 / dens_u0;

        Real rad(0.0),phi(0.0),z(0.0);
        Real p_over_r(0.0);
        GetCylCoord(mp_,rad,phi,z,x1v,x2v,x3v);
        p_over_r = PoverR(mp_,rad);

        if (mp_.cooling_direct_set) {
            // Set total energy so temperature matches desired profile directly (no relaxation)
            Real eint_set = p_over_r*dens_u0/(mp_.gamma_gas - 1.0);
            u0_(m,IEN,k,j,i) = e_k_u0 + e_m + eint_set;
        } else {
            // Relax toward target over cooling timescale (eint and eint_desired_vol both per unit volume)
            Real dens(0.0), eint(0.0);
            dens = w0_(m,IDN,k,j,i);
            eint = w0_(m,IEN,k,j,i);
            Real eint_target_vol = p_over_r*dens/(mp_.gamma_gas - 1.0);
            Real rad_safe = fmax(rad, mp_.rfix);
            Real dtr = fmax(mp_.tcool*2.*M_PI/sqrt(mp_.gm0/rad_safe/rad_safe/rad_safe),bdt);
            Real dfrac = bdt/dtr;
            Real dE = eint - eint_target_vol;  // per unit volume
            u0_(m,IEN,k,j,i) -= dE*dfrac;
        }
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
                pgas = PoverR(mp_, rad)*den;
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
                pgas = PoverR(mp_, rad)*den;
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
                pgas = PoverR(mp_, rad)*den;
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
                pgas = PoverR(mp_, rad)*den;
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
                pgas = PoverR(mp_, rad)*den;
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
                pgas = PoverR(mp_, rad)*den;
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
            w0_(m,IEN,k,j,i) = PoverR(mp_, rad)*den/(mp_.gamma_gas - 1.0);
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
            w0_(m,IEN,k,j,(ie+i+1)) = PoverR(mp_, rad)*den/(mp_.gamma_gas - 1.0);
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
            w0_(m,IEN,k,j,i) = PoverR(mp_, rad)*den/(mp_.gamma_gas - 1.0);
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
            w0_(m,IEN,k,(je+j+1),i) = PoverR(mp_, rad)*den/(mp_.gamma_gas - 1.0);
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
            w0_(m,IEN,k,j,i) = PoverR(mp_, rad)*den/(mp_.gamma_gas - 1.0);
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
            w0_(m,IEN,(ke+k+1),j,i) = PoverR(mp_, rad)*den/(mp_.gamma_gas - 1.0);
        }
    }
  });
    
  pm->pmb_pack->pmhd->peos->PrimToCons(w0_,bcc_,u0_,0,(n1-1),0,(n2-1),ks-ng,ks-1);
  pm->pmb_pack->pmhd->peos->PrimToCons(w0_,bcc_,u0_,0,(n1-1),0,(n2-1),ke+1,ke+ng);

  return;
} 

//----------------------------------------------------------------------------------------
// Function for computing accretion fluxes through constant spherical KS radius surfaces

void MyHistFunc(HistoryData *pdata, Mesh *pm) {

    MeshBlockPack *pmbp = pm->pmb_pack;

  // set nvars, primitive array w0, and field array bcc0
  int nvars;
  // Load appropriate variables depending on hydro or mhd
  DvceArray5D<Real> w0_, bcc0_;
  if (pmbp->phydro != nullptr) {
    nvars = pmbp->phydro->nhydro + pmbp->phydro->nscalars;
    w0_ = pmbp->phydro->w0;
  } else if (pmbp->pmhd != nullptr) {
    nvars = pmbp->pmhd->nmhd + pmbp->pmhd->nscalars;
    w0_ = pmbp->pmhd->w0;
    bcc0_ = pmbp->pmhd->bcc0;
  }

  // extract grids, number of radii, number of fluxes, and history appending index
  auto &grids = pm->pgen->spherical_grids;
  int nradii = grids.size();
  int nflux = 1;  // increase for different flux diagnostics

  // set number of and names of history variables for hydro or mhd
  // (1) mass accretion rate
  // ... list other fluxes here as needed

  // Number of history variables = no. of radial slices * no. of fluxes
  pdata->nhist = nradii*nflux; 
  if (pdata->nhist > NHISTORY_VARIABLES) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "User history function specified pdata->nhist larger than"
              << " NHISTORY_VARIABLES" << std::endl;
    exit(EXIT_FAILURE);
  }

  // set labels for each history variable
  for (int g=0; g<nradii; ++g) {
    std::stringstream stream;
    stream << std::fixed << std::setprecision(1) << grids[g]->radius;
    std::string rad_str = stream.str();
    pdata->label[nflux*g+0] = "mdot_" + rad_str;
    // Can add other flux labels here
  }

  // go through angles at each radii
  for (int g=0; g<nradii; ++g) {
    // initialize zero fluxes at this radius
    pdata->hdata[nflux*g+0] = 0.0;
    pdata->hdata[nflux*g+1] = 0.0;
    pdata->hdata[nflux*g+2] = 0.0;

    // interpolate the primitive variables onto the sphere for each radial slice
    grids[g]->InterpolateToSphere(nvars, w0_);

    // compute fluxes - loop over angles in each radial slice
    for (int n=0; n<grids[g]->nangles; ++n) {
      // extract coordinate data at this angle
      Real r = grids[g]->radius;
      Real theta = grids[g]->polar_pos.h_view(n,0);
      Real phi = grids[g]->polar_pos.h_view(n,1);
      Real x1 = grids[g]->interp_coord.h_view(n,0);
      Real x2 = grids[g]->interp_coord.h_view(n,1);
      Real x3 = grids[g]->interp_coord.h_view(n,2);

      // extract interpolated primitives
      Real &int_dn = grids[g]->interp_vals.h_view(n,IDN);
      Real &int_vx = grids[g]->interp_vals.h_view(n,IVX);
      Real &int_vy = grids[g]->interp_vals.h_view(n,IVY);
      Real &int_vz = grids[g]->interp_vals.h_view(n,IVZ);
      Real &int_ie = grids[g]->interp_vals.h_view(n,IEN);

      // integration area element - area subtended by solid angle of geodesic element
      Real dA = SQR(r)*grids[g]->solid_angles.h_view(n);

      Real sinth = sin(theta);
      Real costh = cos(theta);
      Real sinph = sin(phi);
      Real cosph = cos(phi);

      // radial component of the velocity at this angle
      Real ur = int_vx*cosph*sinth + int_vy*sinph*sinth + int_vz*costh;

      // compute mass flux -ve sign convention so inwards accretion is +ve
      pdata->hdata[nflux*g+0] += -1.0*int_dn*ur*dA;

    }
  }

  // fill rest of the_array with zeros, if nhist < NHISTORY_VARIABLES
  for (int n=pdata->nhist; n<NHISTORY_VARIABLES; ++n) {
    pdata->hdata[n] = 0.0;
  }

  return;
}