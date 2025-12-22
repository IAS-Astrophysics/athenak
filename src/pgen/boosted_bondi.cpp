//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file gr_bondi.cpp
//! \brief Problem generator for spherically symmetric black hole accretion.

#include <cmath>   // abs(), NAN, pow(), sqrt()
#include <cstring> // strcmp()
#include <iostream>
#include <sstream>
#include <stdio.h>
#include <math.h>
#include <iomanip>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/coordinates.hpp"
#include "coordinates/cartesian_ks.hpp"
#include "coordinates/cell_locations.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "pgen/pgen.hpp"
#include "mhd/mhd.hpp"
#include "radiation/radiation.hpp"
namespace {

KOKKOS_INLINE_FUNCTION
static void ComputePrimitiveSingle(Real x1v, Real x2v, Real x3v, CoordData coord,
                                   struct bondi_pgen pgen,
                                   Real& rho, Real& pgas,
                                   Real& uu1, Real& uu2, Real& uu3);

KOKKOS_INLINE_FUNCTION
static void GetBoyerLindquistCoordinates(struct bondi_pgen pgen,
                                         Real x1, Real x2, Real x3,
                                         Real *pr, Real *ptheta, Real *pphi);

KOKKOS_INLINE_FUNCTION
static void TransformVector(struct bondi_pgen pgen,
                            Real a1_bl, Real a2_bl, Real a3_bl,
                            Real x1, Real x2, Real x3,
                            Real *pa1, Real *pa2, Real *pa3);

KOKKOS_INLINE_FUNCTION
static void CalculatePrimitives(Real x1v, Real x2v, Real x3v, struct bondi_pgen pgen, Real r,
                                Real *prho, Real *ppgas, Real *pur, Real *puth, Real *puph);

KOKKOS_INLINE_FUNCTION
static Real TemperatureMin(struct bondi_pgen pgen, Real r, Real t_min, Real t_max);

KOKKOS_INLINE_FUNCTION
static Real TemperatureBisect(struct bondi_pgen pgen, Real r, Real t_min, Real t_max);

KOKKOS_INLINE_FUNCTION
static Real TemperatureResidual(struct bondi_pgen pgen, Real t, Real r);

KOKKOS_INLINE_FUNCTION
Real A1(struct bondi_pgen pgen, Real x1, Real x2, Real x3);
KOKKOS_INLINE_FUNCTION
Real A2(struct bondi_pgen pgen, Real x1, Real x2, Real x3);
KOKKOS_INLINE_FUNCTION
Real A3(struct bondi_pgen pgen, Real x1, Real x2, Real x3);

struct bondi_pgen {
  Real spin;                // black hole spin
  Real dexcise, pexcise;    // excision parameters
  Real n_adi, k_adi, gm, rho0, v_z, v_x, v_y;    // hydro EOS parameters
  Real r_crit;              // sonic point radius
  Real c1, c2;              // useful constants
  Real temp_min, temp_max;  // bounds for temperature root find
  Real target_beta;
  Real Bmag;                // magnitude of magnetic field in ambient medium
  Real cs;                  // sound speed of ambient medium for BHL problem
  bool reset_ic = false;    // reset initial conditions after run
};

  bondi_pgen bondi;

} // namespace

// prototypes for user-defined BCs and error functions
void FixedBondiInflow(Mesh *pm);
void BondiFluxes(HistoryData *pdata, Mesh *pm);

//void BondiErrors(ParameterInput *pin, Mesh *pm);

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::BondiAccretion_()
//! \brief set initial conditions for Bondi accretion test
//  Compile with '-D PROBLEM=gr_bondi' to enroll as user-specific problem generator
//    reference: Hawley, Smarr, & Wilson 1984, ApJ 277 296 (HSW)

void ProblemGenerator::BondiAccretion(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;

   if (pmbp->phydro != nullptr) {
  if (!(pmbp->phydro->peos->eos_data.use_e)) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                << "gr_bondi test requires hydro/use_e=true" << std::endl;
      exit(EXIT_FAILURE);
  }
} else if (pmbp->pmhd != nullptr) {
  if (!(pmbp->pmhd->peos->eos_data.use_e)) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                << "gr_bondi test requires mhd/use_e=true" << std::endl;
      exit(EXIT_FAILURE);
  }
}
  if (!pmbp->pcoord->is_general_relativistic) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "GR bondi problem can only be run when GR defined in <coord> block"
              << std::endl;
    exit(EXIT_FAILURE);
  }

  // set user-defined BCs and error function pointers
  //pgen_final_func = BondiErrors;
  user_bcs_func = FixedBondiInflow;

  // Spherical Grid for user-defined history
  auto &grids = spherical_grids;
  const Real rflux = 1.0 + sqrt(1.0 - SQR(bondi.spin));
  grids.push_back(std::make_unique<SphericalGrid>(pmbp, 5, rflux));
  // NOTE(@pdmullen): Enroll additional radii for flux analysis by
  // pushing back the grids vector with additional SphericalGrid instances
  grids.push_back(std::make_unique<SphericalGrid>(pmbp, 5, 12.0));
  grids.push_back(std::make_unique<SphericalGrid>(pmbp, 5, 200.0));
  // grids.push_back(std::make_unique<SphericalGrid>(pmbp, 5, 200.0));
  user_hist_func = BondiFluxes;

  if (restart) return;

  DvceArray5D<Real> u0_, w0_;
  if (pmbp->phydro != nullptr) {
    u0_ = pmbp->phydro->u0;
    w0_ = pmbp->phydro->w0;
  } else if (pmbp->pmhd != nullptr) {
    u0_ = pmbp->pmhd->u0;
    w0_ = pmbp->pmhd->w0;
  }
  
  // Read problem-specific parameters from input file
  // global parameters
  bondi.k_adi = pin->GetReal("problem", "k_adi");
  bondi.r_crit = pin->GetReal("problem", "r_crit");
  bondi.v_z = pin->GetReal("problem", "v_z");
  bondi.v_y = pin->GetReal("problem", "v_y");
  bondi.v_x = pin->GetReal("problem", "v_x");
  bondi.cs = pin->GetReal("problem", "cs");
  //bondi.rho0 = pin->GetReal("problem", "rho0");
  bondi.Bmag = pin->GetReal("problem", "Bmag");
  
  // Get ideal gas EOS data
  if (pmbp->phydro != nullptr) {
    bondi.gm = pmbp->phydro->peos->eos_data.gamma;
  } else if (pmbp->pmhd != nullptr) {
    bondi.gm = pmbp->pmhd->peos->eos_data.gamma;
    //bondi.Bmag = pin->GetReal("problem","Bmag");
    bondi.target_beta = pin->GetReal("problem","target_beta");

  }
  Real gm1 = bondi.gm - 1.0;
  Real vw = sqrt(bondi.v_x*bondi.v_x + bondi.v_y*bondi.v_y + bondi.v_z*bondi.v_z);
  Real gw = 1/sqrt(1 - vw*vw); // lorentz factor of wind
  Real gs = 1/sqrt(1-bondi.cs*bondi.cs); // lorentz factor of gas sound speed
  Real Mw = (gw/gs) * (vw/bondi.cs); // relativistic Hydro mach number
  printf("Mach number of wind = %.3e\n", Mw);
  
  // Parameters
  bondi.temp_min = 1.0e-8;  // lesser temperature root must be greater than this
  bondi.temp_max = 1.0e1;   // greater temperature root must be less than this

  // Get spin of black hole
  bondi.spin = pmbp->pcoord->coord_data.bh_spin;

  // Get excision parameters
  bondi.dexcise = pmbp->pcoord->coord_data.dexcise;
  bondi.pexcise = pmbp->pcoord->coord_data.pexcise;

  // Get ratio of specific heats
  bondi.n_adi = 1.0/(bondi.gm - 1.0);
  

  // Prepare various constants for determining primitives
  Real u_crit_sq = 1.0/(2.0*bondi.r_crit);                           // (HSW 71)
  Real u_crit = -sqrt(u_crit_sq);
  Real t_crit = (bondi.n_adi/(bondi.n_adi+1.0)
                 * u_crit_sq/(1.0-(bondi.n_adi+3.0)*u_crit_sq));     // (HSW 74)
  bondi.c1 = pow(t_crit, bondi.n_adi) * u_crit * SQR(bondi.r_crit);  // (HSW 68)
  bondi.c2 = (SQR(1.0 + (bondi.n_adi+1.0) * t_crit)
              * (1.0 - 3.0/(2.0*bondi.r_crit)));                     // (HSW 69)
  // Calculate analytic values for initial temp, rho and pgas
  Real temp_an = bondi.n_adi*SQR(bondi.cs)/((1+bondi.n_adi)*(1-bondi.n_adi*SQR(bondi.cs)));
  //bondi.k_adi = temp/(pow(bondi.rho0, 1/(bondi.n_adi)));
  //printf("k_adi = %.3e", bondi.k_adi);
  Real rho_an = pow(temp_an/bondi.k_adi, bondi.n_adi); //bondi.rho0;//pow(temp/bondi.k_adi, bondi.n_adi);
  Real pgas_an = rho_an*temp_an;
  //printf("Temperature = %.3e\n", temp);
  /*
   if (pmbp->pmhd != nullptr) {
    bondi.Bmag = sqrt(2*(pow(temp,bondi.n_adi+1)/(pow(bondi.k_adi,bondi.n_adi)*bondi.target_beta)));
    printf("Bmag = %.3e\n", bondi.Bmag);
    }
  Real bmag = bondi.Bmag;
  printf("computed beta = %.3e\n", pow(temp/bondi.k_adi,bondi.n_adi)*temp/(0.5*SQR(bmag)));
  */
  // capture variables for the kernel
  auto &indcs = pmy_mesh_->mb_indcs;
  auto &size = pmbp->pmb->mb_size;
  auto &coord = pmbp->pcoord->coord_data;
  auto bondi_ = bondi;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng) : 1;
  int is = indcs.is, js = indcs.js, ks = indcs.ks;
  int ie = indcs.ie, je = indcs.je, ke = indcs.ke;
  int nmb = pmbp->nmb_thispack;
  

  // Initialize primitive values (HYDRO ONLY)
  par_for("pgen_bondi", DevExeSpace(), 0,(nmb-1),0,(n3-1),0,(n2-1),0,(n1-1),
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real rho, pgas, uu1, uu2, uu3;
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);

    ComputePrimitiveSingle(x1v,x2v,x3v,coord,bondi_,rho,pgas,uu1,uu2,uu3);
    w0_(m,IDN,k,j,i) = rho;
    w0_(m,IEN,k,j,i) = pgas/gm1;
    w0_(m,IM1,k,j,i) = uu1;
    w0_(m,IM2,k,j,i) = uu2;
    w0_(m,IM3,k,j,i) = uu3;
    //printf("pgas = %.3e\n", pgas);
    //printf("computed beta = %.3e\n", pgas/(2*SQR(bmag)));
  });

  // Convert primitives to conserved
  //auto &u0_ = pmbp->phydro->u0;
  //auto &u1_ = pmbp->phydro->u1;
  //if (bondi.reset_ic) {
  //  pmbp->phydro->peos->PrimToCons(w0_, u1_, 0, (n1-1), 0, (n2-1), 0, (n3-1));
  //} else {
  //pmbp->phydro->peos->PrimToCons(w0_, u0_, 0, (n1-1), 0, (n2-1), 0, (n3-1));
  //}

  if (pmbp->pmhd != nullptr) {
    // parse some more parameters from input
    //bondi.target_beta = pin->GetOrAddReal("problem", "target_beta", 100.0);
    //bondi.potential_cutoff   = pin->GetOrAddReal("problem", "potential_cutoff", 0.2);

    //torus.is_vertical_field = pin->GetOrAddBoolean("problem", "vertical_field", false);

    // for FM torus:
    // Eqns 33, 34 of arxiv/2202.11721
    //  SANE -> potential_r_pow = 0
    //          potential_falloff = 0  (when 0 -> falloff is disabled)
    //          potential_rho_pow = 1
    //   MAD -> potential_r_pow = 3
    //          potential_falloff = 400
    //          potential_rho_pow = 1
    //torus.potential_falloff  = pin->GetOrAddReal("problem", "potential_falloff", 0.0);
    //torus.potential_r_pow    = pin->GetOrAddReal("problem", "potential_r_pow", 0.0);
    //torus.potential_rho_pow  = pin->GetOrAddReal("problem", "potential_rho_pow", 1.0);

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
    auto bondi_ = bondi;

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

      a1(m,k,j,i) = A1(bondi_, x1v, x2f, x3f);
      a2(m,k,j,i) = A2(bondi_, x1f, x2v, x3f);
      a3(m,k,j,i) = A3(bondi_, x1f, x2f, x3v);

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
        a1(m,k,j,i) = 0.5*(A1(bondi_, xl,x2f,x3f) + A1(bondi_, xr,x2f,x3f));
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
        a2(m,k,j,i) = 0.5*(A2(bondi_, x1f,xl,x3f) + A2(bondi_, x1f,xr,x3f));
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
        a3(m,k,j,i) = 0.5*(A3(bondi_, x1f,x2f,xl) + A3(bondi_, x1f,x2f,xr));
      }
    });

    auto &b0 = pmbp->pmhd->b0;
    par_for("pgen_b0", DevExeSpace(), 0,nmb-1,ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      // Compute face-centered fields from curl(A).
      Real dx1 = size.d_view(m).dx1;
      Real dx2 = size.d_view(m).dx2;
      Real dx3 = size.d_view(m).dx3;

      b0.x1f(m,k,j,i) = ((a3(m,k,j+1,i) - a3(m,k,j,i))/dx2 -
                         (a2(m,k+1,j,i) - a2(m,k,j,i))/dx3);
      b0.x2f(m,k,j,i) = ((a1(m,k+1,j,i) - a1(m,k,j,i))/dx3 -
                         (a3(m,k,j,i+1) - a3(m,k,j,i))/dx1);
      b0.x3f(m,k,j,i) = ((a2(m,k,j,i+1) - a2(m,k,j,i))/dx1 -
                         (a1(m,k,j+1,i) - a1(m,k,j,i))/dx2);

      // Include extra face-component at edge of block in each direction
      if (i==ie) {
        b0.x1f(m,k,j,i+1) = ((a3(m,k,j+1,i+1) - a3(m,k,j,i+1))/dx2 -
                             (a2(m,k+1,j,i+1) - a2(m,k,j,i+1))/dx3);
      }
      if (j==je) {
        b0.x2f(m,k,j+1,i) = ((a1(m,k+1,j+1,i) - a1(m,k,j+1,i))/dx3 -
                             (a3(m,k,j+1,i+1) - a3(m,k,j+1,i))/dx1);
      }
      if (k==ke) {
        b0.x3f(m,k+1,j,i) = ((a2(m,k+1,j,i+1) - a2(m,k+1,j,i))/dx1 -
                             (a1(m,k+1,j+1,i) - a1(m,k+1,j,i))/dx2);
      }
    });

    // Compute cell-centered fields
    auto &bcc_ = pmbp->pmhd->bcc0;
    par_for("pgen_bcc", DevExeSpace(), 0,nmb-1,ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      // cell-centered fields are simple linear average of face-centered fields
      Real& w_bx = bcc_(m,IBX,k,j,i);
      Real& w_by = bcc_(m,IBY,k,j,i);
      Real& w_bz = bcc_(m,IBZ,k,j,i);
      w_bx = 0.5*(b0.x1f(m,k,j,i) + b0.x1f(m,k,j,i+1));
      w_by = 0.5*(b0.x2f(m,k,j,i) + b0.x2f(m,k,j+1,i));
      w_bz = 0.5*(b0.x3f(m,k,j,i) + b0.x3f(m,k+1,j,i));
    });
         //B-FIELD RENORMALIZATION STARTS HERE
    // find maximum bsq
     Real bsqmax = 0; //std::numeric_limits<float>::min();
    //Real bsqmax_intorus = std::numeric_limits<float>::min();
    const int nmkji = (pmbp->nmb_thispack)*indcs.nx3*indcs.nx2*indcs.nx1;
    const int nkji = indcs.nx3*indcs.nx2*indcs.nx1;
    //printf("nkji=%d, nmkji= %d\n", nkji, nmkji);
    
    const int nji  = indcs.nx2*indcs.nx1;
    double bsum = 0;
    //Real ptotmax;
    Kokkos::parallel_reduce("torus_beta", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
    KOKKOS_LAMBDA(const int &idx, Real &max_bsq) {
      // compute m,k,j,i indices of thread and call function
      int m = (idx)/nkji;
      int k = (idx - m*nkji)/nji;
      int j = (idx - m*nkji - k*nji)/indcs.nx1;
      int i = (idx - m*nkji - k*nji - j*indcs.nx1) + is;
      k += ks;
      j += js;

      // Extract metric components
      Real &x1min = size.d_view(m).x1min;
      Real &x1max = size.d_view(m).x1max;
      Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

      Real &x2min = size.d_view(m).x2min;
      Real &x2max = size.d_view(m).x2max;
      Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

      Real &x3min = size.d_view(m).x3min;
      Real &x3max = size.d_view(m).x3max;
      Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);
      Real glower[4][4], gupper[4][4];
      ComputeMetricAndInverse(x1v, x2v, x3v, coord.is_minkowski, coord.bh_spin,
                              glower, gupper);

      // Calculate Boyer-Lindquist coordinates of cell
      Real r, theta, phi;
      GetBoyerLindquistCoordinates(bondi_, x1v, x2v, x3v, &r, &theta, &phi);
      Real sin_theta = sin(theta);
      Real cos_theta = cos(theta);
      Real sin_phi = sin(phi);
      Real cos_phi = cos(phi);

      
              

      // Determine if we are in the torus
      // Real log_h;
      //bool in_torus = false;
      //if (r >= trs.r_edge) {
      //log_h = LogHAux(trs, r, sin_vartheta) - trs.log_h_edge;  // (FM 3.6)
      //if (log_h >= 0.0) {
      //  in_torus = true;
      //}
      //}

      // Extract primitive velocity, magnetic field B^i, and gas pressure
      Real &wvx = w0_(m,IVX,k,j,i);
      Real &wvy = w0_(m,IVY,k,j,i);
      Real &wvz = w0_(m,IVZ,k,j,i);
      Real &wbx = bcc_(m,IBX,k,j,i);
      Real &wby = bcc_(m,IBY,k,j,i);
      Real &wbz = bcc_(m,IBZ,k,j,i);

      // Calculate 4-velocity (exploiting symmetry of metric)
      Real q = glower[1][1]*wvx*wvx +2.0*glower[1][2]*wvx*wvy +2.0*glower[1][3]*wvx*wvz
             + glower[2][2]*wvy*wvy +2.0*glower[2][3]*wvy*wvz
             + glower[3][3]*wvz*wvz;
      Real alpha = sqrt(-1.0/gupper[0][0]);
      Real lor = sqrt(1.0 + q);
      Real u0 = lor / alpha;
      Real u1 = wvx - alpha * lor * gupper[0][1];
      Real u2 = wvy - alpha * lor * gupper[0][2];
      Real u3 = wvz - alpha * lor * gupper[0][3];

      // lower vector indices
      Real u_1 = glower[1][0]*u0 + glower[1][1]*u1 + glower[1][2]*u2 + glower[1][3]*u3;
      Real u_2 = glower[2][0]*u0 + glower[2][1]*u1 + glower[2][2]*u2 + glower[2][3]*u3;
      Real u_3 = glower[3][0]*u0 + glower[3][1]*u1 + glower[3][2]*u2 + glower[3][3]*u3;

      // Calculate 4-magnetic field
      Real b0 = u_1*wbx + u_2*wby + u_3*wbz;
      Real b1 = (wbx + b0 * u1) / u0;
      Real b2 = (wby + b0 * u2) / u0;
      Real b3 = (wbz + b0 * u3) / u0;

      // lower vector indices and compute bsq
      Real b_0 = glower[0][0]*b0 + glower[0][1]*b1 + glower[0][2]*b2 + glower[0][3]*b3;
      Real b_1 = glower[1][0]*b0 + glower[1][1]*b1 + glower[1][2]*b2 + glower[1][3]*b3;
      Real b_2 = glower[2][0]*b0 + glower[2][1]*b1 + glower[2][2]*b2 + glower[2][3]*b3;
      Real b_3 = glower[3][0]*b0 + glower[3][1]*b1 + glower[3][2]*b2 + glower[3][3]*b3;
      Real bsq = b0*b_0 + b1*b_1 + b2*b_2 + b3*b_3;
      //printf("computed bsq = %.3e\n", bsq);
      max_bsq += bsq/nmkji;
      //max_bsq = bsq;
      //printf("computed bsq[%d,%d,%d,%d] = %.3e, bsqsum[%d,%d,%d,%d] = %.3e \n", m,k,j,i,bsq,m,k,j,i,max_bsq);
			    }, Kokkos::Sum<Real>(bsqmax));//Kokkos::Sum<Real>(bsqmax)); 
 
    //    printf("max bsq = %.3e\n",bsqmax);
    printf("avg bsq = %.3e\n",bsqmax);
    

    #if MPI_PARALLEL_ENABLED
    // get maximum value of gas pressure and bsq over all MPI ranks
    //MPI_Allreduce(MPI_IN_PLACE, &ptotmax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    //MPI_Allreduce(MPI_IN_PLACE, &ptotmax, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    //MPI_Allreduce(MPI_IN_PLACE, &bsqmax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &bsqmax, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    //MPI_Allreduce(MPI_IN_PLACE, &bsqmax_intorus, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    #endif

    printf("post MPI avg bsq = %.3e\n",bsqmax);
    printf("post MPI sum ptot = %.3e\n",pgas_an);
    printf("beta_inv = %.3e\n", 0.5*bsqmax/pgas_an);
    // Apply renormalization of magnetic field
    
    Real bnorm = sqrt((pgas_an/(0.5*bsqmax))/bondi.target_beta);
    printf("bnorm = %.3e\n", bnorm);
    // Since vertical field extends beyond torus, normalize based on values in torus
    //if (torus.is_vertical_field) {
			    //bnorm = sqrt((ptotmax/(0.5*bsqmax_intorus))/torus.potential_beta_min);
			    //}
    /*
    par_for("pgen_normb0", DevExeSpace(), 0,nmb-1,ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      b0.x1f(m,k,j,i) *= bnorm;
      b0.x2f(m,k,j,i) *= bnorm;
      b0.x3f(m,k,j,i) *= bnorm;
      if (i==ie) { b0.x1f(m,k,j,i+1) *= bnorm; }
      if (j==je) { b0.x2f(m,k,j+1,i) *= bnorm; }
      if (k==ke) { b0.x3f(m,k+1,j,i) *= bnorm; }
    });

    // Recompute cell-centered magnetic field
    par_for("pgen_normbcc", DevExeSpace(), 0,nmb-1,ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      // cell-centered fields are simple linear average of face-centered fields
      Real& w_bx = bcc_(m,IBX,k,j,i);
      Real& w_by = bcc_(m,IBY,k,j,i);
      Real& w_bz = bcc_(m,IBZ,k,j,i);
      w_bx = 0.5*(b0.x1f(m,k,j,i) + b0.x1f(m,k,j,i+1));
      w_by = 0.5*(b0.x2f(m,k,j,i) + b0.x2f(m,k,j+1,i));
      w_bz = 0.5*(b0.x3f(m,k,j,i) + b0.x3f(m,k+1,j,i));
      }); */ // B-field renormalization ends here
  }

  // Convert primitives to conserved
  if (pmbp->phydro != nullptr) {
    pmbp->phydro->peos->PrimToCons(w0_, u0_, is, ie, js, je, ks, ke);
  } else if (pmbp->pmhd != nullptr) {
    auto &bcc0_ = pmbp->pmhd->bcc0;
    pmbp->pmhd->peos->PrimToCons(w0_, bcc0_, u0_, is, ie, js, je, ks, ke);
  }

  return;
}


//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::LinearWaveErrors_()
//  \brief Computes errors in linear wave solution and outputs to file.

namespace {

//----------------------------------------------------------------------------------------
// Function for returning corresponding Boyer-Lindquist coordinates of point
// Inputs:
//   x1,x2,x3: global coordinates to be converted
// Outputs:
//   pr,ptheta,pphi: variables pointed to set to Boyer-Lindquist coordinates

KOKKOS_INLINE_FUNCTION
static void ComputePrimitiveSingle(Real x1v, Real x2v, Real x3v, CoordData coord,
                                   struct bondi_pgen pgen,
                                   Real& rho, Real& pgas,
                                   Real& uu1, Real& uu2, Real& uu3) {
  // Calculate Boyer-Lindquist coordinates of cell
  Real r, theta, phi;
  GetBoyerLindquistCoordinates(pgen, x1v, x2v, x3v, &r, &theta, &phi);

  // Compute primitive in BL coordinates, transform to Cartesian KS
  Real my_rho, my_pgas, my_ur, my_uth, my_uph;
  CalculatePrimitives(x1v, x2v, x3v, pgen, r, &my_rho, &my_pgas, &my_ur, &my_uth, &my_uph);
  Real u0(0.0), u1(0.0), u2(0.0), u3(0.0);
  TransformVector(pgen, my_ur, my_uth, my_uph, x1v, x2v, x3v, &u1, &u2, &u3);
  Real glower[4][4], gupper[4][4];
  ComputeMetricAndInverse(x1v,x2v,x3v, coord.is_minkowski, coord.bh_spin, glower, gupper);
  //adding wind velocities for each cartesian direction
  u3 += pgen.v_z/sqrt(glower[3][3]);
  u2 += pgen.v_y/sqrt(glower[2][2]);
  u1 += pgen.v_x/sqrt(glower[1][1]);
  
  

  Real tmp = glower[1][1]*u1*u1 + 2.0*glower[1][2]*u1*u2 + 2.0*glower[1][3]*u1*u3
           + glower[2][2]*u2*u2 + 2.0*glower[2][3]*u2*u3
           + glower[3][3]*u3*u3;
  Real gammasq = 1.0 + tmp;
  Real b = glower[0][1]*u1 + glower[0][2]*u2 + glower[0][3]*u3;
  u0 = (-b - sqrt(fmax(SQR(b) - glower[0][0]*gammasq, 0.0)))/glower[0][0];

  if (r > 1.0) {
    rho = my_rho;
    pgas = my_pgas;
    uu1 = u1 - gupper[0][1]/gupper[0][0] * u0;
    uu2 = u2 - gupper[0][2]/gupper[0][0] * u0;
    uu3 = u3 - gupper[0][3]/gupper[0][0] * u0;
  } else {
    rho = pgen.dexcise;
    pgas = pgen.pexcise;
    uu1 = 0.0;
    uu2 = 0.0;
    uu3 = 0.0;
  }

  return;
}


//----------------------------------------------------------------------------------------
// Function for returning corresponding Boyer-Lindquist coordinates of point
// Inputs:
//   x1,x2,x3: global coordinates to be converted
// Outputs:
//   pr,ptheta,pphi: variables pointed to set to Boyer-Lindquist coordinates

KOKKOS_INLINE_FUNCTION
static void GetBoyerLindquistCoordinates(struct bondi_pgen pgen,
                                         Real x1, Real x2, Real x3,
                                         Real *pr, Real *ptheta, Real *pphi) {
    Real rad = sqrt(SQR(x1) + SQR(x2) + SQR(x3));
    Real r = fmax((sqrt( SQR(rad) - SQR(pgen.spin) + sqrt(SQR(SQR(rad)-SQR(pgen.spin))
                        + 4.0*SQR(pgen.spin)*SQR(x3)) ) / sqrt(2.0)), 1.0);
    *pr = r;
    *ptheta = acos(x3/r);
    *pphi = atan2(r*x2-pgen.spin*x1, pgen.spin*x2+r*x1) -
            pgen.spin*r/(SQR(r)-2.0*r+SQR(pgen.spin));
  return;
}

//----------------------------------------------------------------------------------------
// Function for transforming 4-vector from Boyer-Lindquist to desired coordinates
// Inputs:
//   a0_bl,a1_bl,a2_bl,a3_bl: upper 4-vector components in Boyer-Lindquist coordinates
//   x1,x2,x3: Cartesian Kerr-Schild coordinates of point
// Outputs:
//   pa0,pa1,pa2,pa3: pointers to upper 4-vector components in desired coordinates
// Notes:
//   Schwarzschild coordinates match Boyer-Lindquist when a = 0

KOKKOS_INLINE_FUNCTION
static void TransformVector(struct bondi_pgen pgen,
                            Real a1_bl, Real a2_bl, Real a3_bl,
                            Real x1, Real x2, Real x3,
                            Real *pa1, Real *pa2, Real *pa3) {
  Real rad = sqrt( SQR(x1) + SQR(x2) + SQR(x3) );
  Real r = fmax((sqrt( SQR(rad) - SQR(pgen.spin) + sqrt(SQR(SQR(rad)-SQR(pgen.spin))
                      + 4.0*SQR(pgen.spin)*SQR(x3)) ) / sqrt(2.0)), 1.0);
  Real delta = SQR(r) - 2.0*r + SQR(pgen.spin);
  *pa1 = a1_bl * ( (r*x1+pgen.spin*x2)/(SQR(r) + SQR(pgen.spin)) - x2*pgen.spin/delta) +
         a2_bl * x1*x3/r * sqrt((SQR(r) + SQR(pgen.spin))/(SQR(x1) + SQR(x2))) -
         a3_bl * x2;
  *pa2 = a1_bl * ( (r*x2-pgen.spin*x1)/(SQR(r) + SQR(pgen.spin)) + x1*pgen.spin/delta) +
         a2_bl * x2*x3/r * sqrt((SQR(r) + SQR(pgen.spin))/(SQR(x1) + SQR(x2))) +
         a3_bl * x1;
  *pa3 = a1_bl * x3/r -
         a2_bl * r * sqrt((SQR(x1) + SQR(x2))/(SQR(r) + SQR(pgen.spin)));
  return;
}


//-------------------------------------------------------------------------------------------
// Function to compute 1-component of vector potential.  First computes phi-componenent
// in spherical KS coordinates, then transforms to Cartesian KS

KOKKOS_INLINE_FUNCTION
Real A1(struct bondi_pgen pgen, Real x1, Real x2, Real x3) {
  // BL coordinates
  //Real r, theta, phi;
  //GetBoyerLindquistCoordinates(pgen, x1, x2, x3, &r, &theta, &phi);

  // calculate vector potential in spherical KS
  //Real atheta, aphi;
  //CalculateVectorPotentialInTiltedTorus(pgen, r, theta, phi, &atheta, &aphi);

  //Real big_r = sqrt( SQR(x1) + SQR(x2) + SQR(x3) );
  //Real sqrt_term =  2.0*SQR(r) - SQR(big_r) + SQR(pgen.spin);
  //Real isin_term = sqrt((SQR(pgen.spin)+SQR(r))/fmax(SQR(x1)+SQR(x2),1.0e-12));

  //return atheta*(x1*x3*isin_term/(r*sqrt_term)) + aphi*(-x2/(SQR(x1)+SQR(x2))+pgen.spin*x1*r/((SQR(pgen.spin)+SQR(r))*sqrt_term));
  Real Bmag = pgen.Bmag;
  return -0.5*Bmag*x2;
}

//----------------------------------------------------------------------------------------
// Function to compute 2-component of vector potential. See comments for A1.

KOKKOS_INLINE_FUNCTION
Real A2(struct bondi_pgen pgen, Real x1, Real x2, Real x3) {

  
  // BL coordinates
  //Real r, theta, phi;
  //GetBoyerLindquistCoordinates(pgen, x1, x2, x3, &r, &theta, &phi);

  // calculate vector potential in spherical KS
  //Real atheta, aphi;
  //CalculateVectorPotentialInTiltedTorus(pgen, r, theta, phi, &atheta, &aphi);

  //Real big_r = sqrt( SQR(x1) + SQR(x2) + SQR(x3) );
  //Real sqrt_term =  2.0*SQR(r) - SQR(big_r) + SQR(pgen.spin);
  //Real isin_term = sqrt((SQR(pgen.spin)+SQR(r))/fmax(SQR(x1)+SQR(x2),1.0e-12));

  //return atheta*(x2*x3*isin_term/(r*sqrt_term)) + aphi*(x1/(SQR(x1)+SQR(x2))+pgen.spin*x2*r/((SQR(pgen.spin)+SQR(r))*sqrt_term));
  Real Bmag = pgen.Bmag;
  return 0.5*Bmag*x1;
}

//----------------------------------------------------------------------------------------
// Function to compute 3-component of vector potential. See comments for A1.

KOKKOS_INLINE_FUNCTION
Real A3(struct bondi_pgen pgen, Real x1, Real x2, Real x3) {
  // BL coordinates
  //Real r, theta, phi;
  //GetBoyerLindquistCoordinates(pgen, x1, x2, x3, &r, &theta, &phi);

  // calculate vector potential in spherical KS
  //Real atheta, aphi;
  //CalculateVectorPotentialInTiltedTorus(pgen, r, theta, phi, &atheta, &aphi);

  //Real big_r = sqrt( SQR(x1) + SQR(x2) + SQR(x3) );
  //Real sqrt_term =  2.0*SQR(r) - SQR(big_r) + SQR(pgen.spin);
  //Real isin_term = sqrt((SQR(pgen.spin)+SQR(r))/fmax(SQR(x1)+SQR(x2),1.0e-12));

  //return atheta*(((1.0+SQR(pgen.spin/r))*SQR(x3)-sqrt_term)*isin_term/(r*sqrt_term)) + aphi*(pgen.spin*x3/(r*sqrt_term));
  //Real Bmag = pgen.Bmag;
  return 0;
}

  
//----------------------------------------------------------------------------------------
// Function for calculating primitives given radius
// Inputs:
//   r: Schwarzschild radius
//   temp_min,temp_max: bounds on temperature
// Outputs:
//   prho: value set to density
//   ppgas: value set to gas pressure
//   put: value set to u^t in Schwarzschild coordinates
//   pur: value set to u^r in Schwarzschild coordinates
// Notes:
//   references Hawley, Smarr, & Wilson 1984, ApJ 277 296 (HSW)

KOKKOS_INLINE_FUNCTION
static void CalculatePrimitives(Real x1v, Real x2v, Real x3v, struct bondi_pgen pgen, Real r,
                                Real *prho, Real *ppgas, Real *pur, Real *puth, Real *puph) {
  // Calculate solution to (HSW 76)
  /*
  Real temp_neg_res = TemperatureMin(pgen, r, pgen.temp_min, pgen.temp_max);
  Real temp;
  if (r <= pgen.r_crit) {  // use lesser of two roots
   temp = TemperatureBisect(pgen, r, pgen.temp_min, temp_neg_res);
  } else {  // user greater of two roots
   temp = TemperatureBisect(pgen, r, temp_neg_res, pgen.temp_max);
   }*/
  //Real gm = pgen.gm;
  //Real n = 1/(gm-1);
  Real cs = pgen.cs; // sound speed of ambient gas
  //printf("cs = %.3e\n", cs);
  //Real temp = SQR(cs)/gm * SQR(gm-1)/(1-SQR(cs)*(gm-1)) ;

  Real rbl, theta, phi;
  GetBoyerLindquistCoordinates(pgen, x1v, x2v, x3v, &rbl, &theta, &phi);
  Real sth = sin(theta);
  Real cth = cos(theta);
  Real sph = sin(phi);
  Real cph = cos(phi);
  // temperature from ideal polytropic gas, in terms of sound speed
  //Real css = cs;
  //if (rbl<150){
  //css = fmax(0.03, cs);
  //}
  //else{ css = cs;}
  Real temp = pgen.n_adi*SQR(cs)/((1+pgen.n_adi)*(1-pgen.n_adi*SQR(cs))); ;
  //Real temp1 = pgen.n_adi*SQR(cs)/((1+pgen.n_adi)*(1-pgen.n_adi*SQR(cs)));
  //if (pgen.cs>=0.03){
  //   temp = temp1;
  //}
  //else{
  //  Real temp2 = pgen.n_adi*SQR(0.03)/((1+pgen.n_adi)*(1-pgen.n_adi*SQR(0.03)));
  //   temp = temp2*exp(-rbl/200) + temp1;
  //}
 
  
  //printf("temp = %.3e\n", temp);

  // Calculate primitives
  Real rho = pow(temp/pgen.k_adi, pgen.n_adi);  //pgen.rho0; //pow(temp/pgen.k_adi, pgen.n_adi);             // not same K as HSW
  //printf("rho = %.3e\n", rho);  
  Real pgas = temp * rho;
  //Real beta = pgen.target_beta;
  //Real Bmag = sqrt(pgas/(2*beta));
  Real ur = pgen.c1 / (SQR(r) * pow(temp, pgen.n_adi));    // (HSW 75)
  //printf("primitive beta = %.3e\n", pgas/(0.5*SQR(pgen.Bmag)));
  // Set primitives
  *prho = rho;
  *ppgas = pgas;
  //*pur = ur;
  if (pgen.v_z == 0 and pgen.v_y == 0 and  pgen.v_x == 0){
    *pur = ur;
    *puth = 0.0;
    *puph = 0.0;
  }
  else{
    *pur = 0.0; //pgen.v_z * cth + pgen.v_x * sth * cph  + pgen.v_y * sth * sph;
    *puth = 0.0; //-pgen.v_z * sth + pgen.v_x * cth * cph + pgen.v_y * cth * sph;
    *puph =  0.0; //pgen.v_y * cph -  pgen.v_x * sph;
  }
  return;
}

//----------------------------------------------------------------------------------------
// Function for finding temperature at which residual is minimized
// Inputs:
//   r: Schwarzschild radius
//   t_min,t_max: bounds between which minimum must occur
// Outputs:
//   returned value: some temperature for which residual of (HSW 76) is negative
// Notes:
//   references Hawley, Smarr, & Wilson 1984, ApJ 277 296 (HSW)
//   performs golden section search (cf. Numerical Recipes, 3rd ed., 10.2)

KOKKOS_INLINE_FUNCTION
static Real TemperatureMin(struct bondi_pgen pgen, Real r, Real t_min, Real t_max) {
  // Parameters
  const Real ratio = 0.3819660112501051;  // (3+\sqrt{5})/2
  const int max_iterations = 100;          // maximum number of iterations

  // Initialize values
  Real t_mid = t_min + ratio * (t_max - t_min);
  Real res_mid = TemperatureResidual(pgen, t_mid, r);

  // Apply golden section method
  bool larger_to_right = true;  // flag indicating larger subinterval is on right
  for (int n = 0; n < max_iterations; ++n) {
    if (res_mid < 0.0) {
      return t_mid;
    }
    Real t_new;
    if (larger_to_right) {
      t_new = t_mid + ratio * (t_max - t_mid);
      Real res_new = TemperatureResidual(pgen, t_new, r);
      if (res_new < res_mid) {
        t_min = t_mid;
        t_mid = t_new;
        res_mid = res_new;
      } else {
        t_max = t_new;
        larger_to_right = false;
      }
    } else {
      t_new = t_mid - ratio * (t_mid - t_min);
      Real res_new = TemperatureResidual(pgen, t_new, r);
      if (res_new < res_mid) {
        t_max = t_mid;
        t_mid = t_new;
        res_mid = res_new;
      } else {
        t_min = t_new;
        larger_to_right = true;
      }
    }
  }
  return NAN;
}

//----------------------------------------------------------------------------------------
// Bisection root finder
// Inputs:
//   r: Schwarzschild radius
//   t_min,t_max: bounds between which root must occur
// Outputs:
//   returned value: temperature that satisfies (HSW 76)
// Notes:
//   references Hawley, Smarr, & Wilson 1984, ApJ 277 296 (HSW)
//   performs bisection search

KOKKOS_INLINE_FUNCTION
static Real TemperatureBisect(struct bondi_pgen pgen, Real r, Real t_min, Real t_max) {
  // Parameters
  const int max_iterations = 40;
  const Real tol_residual = 1.0e-12;
  const Real tol_temperature = 1.0e-12;

  // Find initial residuals
  Real res_min = TemperatureResidual(pgen, t_min, r);
  Real res_max = TemperatureResidual(pgen, t_max, r);
  if (std::abs(res_min) < tol_residual) {
    return t_min;
  }
  if (std::abs(res_max) < tol_residual) {
    return t_max;
  }
  if ((res_min < 0.0 && res_max < 0.0) || (res_min > 0.0 && res_max > 0.0)) {
    return NAN;
  }

  // Iterate to find root
  Real t_mid;
  for (int i = 0; i < max_iterations; ++i) {
    t_mid = (t_min + t_max) / 2.0;
    if (t_max - t_min < tol_temperature) {
      return t_mid;
    }
    Real res_mid = TemperatureResidual(pgen, t_mid, r);
    if (std::abs(res_mid) < tol_residual) {
      return t_mid;
    }
    if ((res_mid < 0.0 && res_min < 0.0) || (res_mid > 0.0 && res_min > 0.0)) {
      t_min = t_mid;
      res_min = res_mid;
    } else {
      t_max = t_mid;
      res_max = res_mid;
    }
  }
  return t_mid;
}

//----------------------------------------------------------------------------------------
// Function whose value vanishes for correct temperature
// Inputs:
//   t: temperature
//   r: Schwarzschild radius
// Outputs:
//   returned value: residual that should vanish for correct temperature
// Notes:
//   implements (76) from Hawley, Smarr, & Wilson 1984, ApJ 277 296

KOKKOS_INLINE_FUNCTION
static Real TemperatureResidual(struct bondi_pgen pgen, Real t, Real r) {
  return SQR(1.0 + (pgen.n_adi+1.0) * t)
      * (1.0 - 2.0/r + SQR(pgen.c1)
         / (SQR(SQR(r)) * pow(t, 2.0*pgen.n_adi))) - pgen.c2;
}

} // namespace

//----------------------------------------------------------------------------------------
//! \fn FixedBondiInflow
//  \brief Sets boundary condition on surfaces of computational domain
// Note quantities at boundaryies are held fixed to initial condition values

void FixedBondiInflow(Mesh *pm) {
 auto &indcs = pm->mb_indcs;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng) : 1;
  int &is = indcs.is;  int &ie  = indcs.ie;
  int &js = indcs.js;  int &je  = indcs.je;
  int &ks = indcs.ks;  int &ke  = indcs.ke;
  auto &mb_bcs = pm->pmb_pack->pmb->mb_bcs;

  // Select either Hydro or MHD
  DvceArray5D<Real> u0_, w0_;
  if (pm->pmb_pack->phydro != nullptr) {
    u0_ = pm->pmb_pack->phydro->u0;
    w0_ = pm->pmb_pack->phydro->w0;
  } else if (pm->pmb_pack->pmhd != nullptr) {
    u0_ = pm->pmb_pack->pmhd->u0;
    w0_ = pm->pmb_pack->pmhd->w0;
  }
  int nmb = pm->pmb_pack->nmb_thispack;
  int nvar = u0_.extent_int(1);

  // Determine if radiation is enabled
  const bool is_radiation_enabled = (pm->pmb_pack->prad != nullptr);
  DvceArray5D<Real> i0_; int nang1;
  if (is_radiation_enabled) {
    i0_ = pm->pmb_pack->prad->i0;
    nang1 = pm->pmb_pack->prad->prgeo->nangles - 1;
  }

  // X1-Boundary
  // Set X1-BCs on b0 if Meshblock face is at the edge of computational domain
  if (pm->pmb_pack->pmhd != nullptr) {
    auto &b0 = pm->pmb_pack->pmhd->b0;
    par_for("noinflow_field_x1", DevExeSpace(),0,(nmb-1),0,(n3-1),0,(n2-1),
    KOKKOS_LAMBDA(int m, int k, int j) {
      if (mb_bcs.d_view(m,BoundaryFace::inner_x1) == BoundaryFlag::user) {
        for (int i=0; i<ng; ++i) {
          b0.x1f(m,k,j,is-i-1) = b0.x1f(m,k,j,is);
          b0.x2f(m,k,j,is-i-1) = b0.x2f(m,k,j,is);
          if (j == n2-1) {b0.x2f(m,k,j+1,is-i-1) = b0.x2f(m,k,j+1,is);}
          b0.x3f(m,k,j,is-i-1) = b0.x3f(m,k,j,is);
          if (k == n3-1) {b0.x3f(m,k+1,j,is-i-1) = b0.x3f(m,k+1,j,is);}
        }
      }
      if (mb_bcs.d_view(m,BoundaryFace::outer_x1) == BoundaryFlag::user) {
        for (int i=0; i<ng; ++i) {
          b0.x1f(m,k,j,ie+i+2) = b0.x1f(m,k,j,ie+1);
          b0.x2f(m,k,j,ie+i+1) = b0.x2f(m,k,j,ie);
          if (j == n2-1) {b0.x2f(m,k,j+1,ie+i+1) = b0.x2f(m,k,j+1,ie);}
          b0.x3f(m,k,j,ie+i+1) = b0.x3f(m,k,j,ie);
          if (k == n3-1) {b0.x3f(m,k+1,j,ie+i+1) = b0.x3f(m,k+1,j,ie);}
        }
      }
    });
  }
  // ConsToPrim over all X1 ghost zones *and* at the innermost/outermost X1-active zones
  // of Meshblocks, even if Meshblock face is not at the edge of computational domain
  if (pm->pmb_pack->phydro != nullptr) {
    pm->pmb_pack->phydro->peos->ConsToPrim(u0_,w0_,false,is-ng,is,0,(n2-1),0,(n3-1));
    pm->pmb_pack->phydro->peos->ConsToPrim(u0_,w0_,false,ie,ie+ng,0,(n2-1),0,(n3-1));
  } else if (pm->pmb_pack->pmhd != nullptr) {
    auto &b0 = pm->pmb_pack->pmhd->b0;
    auto &bcc = pm->pmb_pack->pmhd->bcc0;
    pm->pmb_pack->pmhd->peos->ConsToPrim(u0_,b0,w0_,bcc,false,is-ng,is,0,(n2-1),0,(n3-1));
    pm->pmb_pack->pmhd->peos->ConsToPrim(u0_,b0,w0_,bcc,false,ie,ie+ng,0,(n2-1),0,(n3-1));
  }
  // Set X1-BCs on w0 if Meshblock face is at the edge of computational domain
  par_for("noinflow_hydro_x1", DevExeSpace(),0,(nmb-1),0,(nvar-1),0,(n3-1),0,(n2-1),
  KOKKOS_LAMBDA(int m, int n, int k, int j) {
    if (mb_bcs.d_view(m,BoundaryFace::inner_x1) == BoundaryFlag::user) {
      for (int i=0; i<ng; ++i) {
        if (n==(IVX)) {
          w0_(m,n,k,j,is-i-1) = fmin(0.0,w0_(m,n,k,j,is));
        } else {
          w0_(m,n,k,j,is-i-1) = w0_(m,n,k,j,is);
        }
      }
    }
    if (mb_bcs.d_view(m,BoundaryFace::outer_x1) == BoundaryFlag::user) {
      for (int i=0; i<ng; ++i) {
        if (n==(IVX)) {
          w0_(m,n,k,j,ie+i+1) = fmax(0.0,w0_(m,n,k,j,ie));
        } else {
          w0_(m,n,k,j,ie+i+1) = w0_(m,n,k,j,ie);
        }
      }
    }
  });
  if (is_radiation_enabled) {
    // Set X1-BCs on i0 if Meshblock face is at the edge of computational domain
    par_for("noinflow_rad_x1", DevExeSpace(),0,(nmb-1),0,nang1,0,(n3-1),0,(n2-1),
    KOKKOS_LAMBDA(int m, int n, int k, int j) {
      if (mb_bcs.d_view(m,BoundaryFace::inner_x1) == BoundaryFlag::user) {
        for (int i=0; i<ng; ++i) {
          i0_(m,n,k,j,is-i-1) = i0_(m,n,k,j,is);
        }
      }
      if (mb_bcs.d_view(m,BoundaryFace::outer_x1) == BoundaryFlag::user) {
        for (int i=0; i<ng; ++i) {
          i0_(m,n,k,j,ie+i+1) = i0_(m,n,k,j,ie);
        }
      }
    });
  }
  // PrimToCons on X1 ghost zones
  if (pm->pmb_pack->phydro != nullptr) {
    pm->pmb_pack->phydro->peos->PrimToCons(w0_,u0_,is-ng,is-1,0,(n2-1),0,(n3-1));
    pm->pmb_pack->phydro->peos->PrimToCons(w0_,u0_,ie+1,ie+ng,0,(n2-1),0,(n3-1));
  } else if (pm->pmb_pack->pmhd != nullptr) {
    auto &bcc0_ = pm->pmb_pack->pmhd->bcc0;
    pm->pmb_pack->pmhd->peos->PrimToCons(w0_,bcc0_,u0_,is-ng,is-1,0,(n2-1),0,(n3-1));
    pm->pmb_pack->pmhd->peos->PrimToCons(w0_,bcc0_,u0_,ie+1,ie+ng,0,(n2-1),0,(n3-1));
  }

  // X2-Boundary
  // Set X2-BCs on b0 if Meshblock face is at the edge of computational domain
  if (pm->pmb_pack->pmhd != nullptr) {
    auto &b0 = pm->pmb_pack->pmhd->b0;
    par_for("noinflow_field_x2", DevExeSpace(),0,(nmb-1),0,(n3-1),0,(n1-1),
    KOKKOS_LAMBDA(int m, int k, int i) {
      if (mb_bcs.d_view(m,BoundaryFace::inner_x2) == BoundaryFlag::user) {
        for (int j=0; j<ng; ++j) {
          b0.x1f(m,k,js-j-1,i) = b0.x1f(m,k,js,i);
          if (i == n1-1) {b0.x1f(m,k,js-j-1,i+1) = b0.x1f(m,k,js,i+1);}
          b0.x2f(m,k,js-j-1,i) = b0.x2f(m,k,js,i);
          b0.x3f(m,k,js-j-1,i) = b0.x3f(m,k,js,i);
          if (k == n3-1) {b0.x3f(m,k+1,js-j-1,i) = b0.x3f(m,k+1,js,i);}
        }
      }
      if (mb_bcs.d_view(m,BoundaryFace::outer_x2) == BoundaryFlag::user) {
        for (int j=0; j<ng; ++j) {
          b0.x1f(m,k,je+j+1,i) = b0.x1f(m,k,je,i);
          if (i == n1-1) {b0.x1f(m,k,je+j+1,i+1) = b0.x1f(m,k,je,i+1);}
          b0.x2f(m,k,je+j+2,i) = b0.x2f(m,k,je+1,i);
          b0.x3f(m,k,je+j+1,i) = b0.x3f(m,k,je,i);
          if (k == n3-1) {b0.x3f(m,k+1,je+j+1,i) = b0.x3f(m,k+1,je,i);}
        }
      }
    });
  }
  // ConsToPrim over all X2 ghost zones *and* at the innermost/outermost X2-active zones
  // of Meshblocks, even if Meshblock face is not at the edge of computational domain
  if (pm->pmb_pack->phydro != nullptr) {
    pm->pmb_pack->phydro->peos->ConsToPrim(u0_,w0_,false,0,(n1-1),js-ng,js,0,(n3-1));
    pm->pmb_pack->phydro->peos->ConsToPrim(u0_,w0_,false,0,(n1-1),je,je+ng,0,(n3-1));
  } else if (pm->pmb_pack->pmhd != nullptr) {
    auto &b0 = pm->pmb_pack->pmhd->b0;
    auto &bcc = pm->pmb_pack->pmhd->bcc0;
    pm->pmb_pack->pmhd->peos->ConsToPrim(u0_,b0,w0_,bcc,false,0,(n1-1),js-ng,js,0,(n3-1));
    pm->pmb_pack->pmhd->peos->ConsToPrim(u0_,b0,w0_,bcc,false,0,(n1-1),je,je+ng,0,(n3-1));
  }
  // Set X2-BCs on w0 if Meshblock face is at the edge of computational domain
  par_for("noinflow_hydro_x2", DevExeSpace(),0,(nmb-1),0,(nvar-1),0,(n3-1),0,(n1-1),
  KOKKOS_LAMBDA(int m, int n, int k, int i) {
    if (mb_bcs.d_view(m,BoundaryFace::inner_x2) == BoundaryFlag::user) {
      for (int j=0; j<ng; ++j) {
        if (n==(IVY)) {
          w0_(m,n,k,js-j-1,i) = fmin(0.0,w0_(m,n,k,js,i));
        } else {
          w0_(m,n,k,js-j-1,i) = w0_(m,n,k,js,i);
        }
      }
    }
    if (mb_bcs.d_view(m,BoundaryFace::outer_x2) == BoundaryFlag::user) {
      for (int j=0; j<ng; ++j) {
        if (n==(IVY)) {
          w0_(m,n,k,je+j+1,i) = fmax(0.0,w0_(m,n,k,je,i));
        } else {
          w0_(m,n,k,je+j+1,i) = w0_(m,n,k,je,i);
        }
      }
    }
  });
  if (is_radiation_enabled) {
    // Set X2-BCs on i0 if Meshblock face is at the edge of computational domain
    par_for("noinflow_rad_x2", DevExeSpace(),0,(nmb-1),0,nang1,0,(n3-1),0,(n1-1),
    KOKKOS_LAMBDA(int m, int n, int k, int i) {
      if (mb_bcs.d_view(m,BoundaryFace::inner_x2) == BoundaryFlag::user) {
        for (int j=0; j<ng; ++j) {
          i0_(m,n,k,js-j-1,i) = i0_(m,n,k,js,i);
        }
      }
      if (mb_bcs.d_view(m,BoundaryFace::outer_x2) == BoundaryFlag::user) {
        for (int j=0; j<ng; ++j) {
          i0_(m,n,k,je+j+1,i) = i0_(m,n,k,je,i);
        }
      }
    });
  }
  // PrimToCons on X2 ghost zones
  if (pm->pmb_pack->phydro != nullptr) {
    pm->pmb_pack->phydro->peos->PrimToCons(w0_,u0_,0,(n1-1),js-ng,js-1,0,(n3-1));
    pm->pmb_pack->phydro->peos->PrimToCons(w0_,u0_,0,(n1-1),je+1,je+ng,0,(n3-1));
  } else if (pm->pmb_pack->pmhd != nullptr) {
    auto &bcc0_ = pm->pmb_pack->pmhd->bcc0;
    pm->pmb_pack->pmhd->peos->PrimToCons(w0_,bcc0_,u0_,0,(n1-1),js-ng,js-1,0,(n3-1));
    pm->pmb_pack->pmhd->peos->PrimToCons(w0_,bcc0_,u0_,0,(n1-1),je+1,je+ng,0,(n3-1));
  }

  // X3-Boundary
  // Set X3-BCs on b0 if Meshblock face is at the edge of computational domain
  if (pm->pmb_pack->pmhd != nullptr) {
    auto &b0 = pm->pmb_pack->pmhd->b0;
    par_for("noinflow_field_x3", DevExeSpace(),0,(nmb-1),0,(n2-1),0,(n1-1),
    KOKKOS_LAMBDA(int m, int j, int i) {
      if (mb_bcs.d_view(m,BoundaryFace::inner_x3) == BoundaryFlag::user) {
        for (int k=0; k<ng; ++k) {
          b0.x1f(m,ks-k-1,j,i) = b0.x1f(m,ks,j,i);
          if (i == n1-1) {b0.x1f(m,ks-k-1,j,i+1) = b0.x1f(m,ks,j,i+1);}
          b0.x2f(m,ks-k-1,j,i) = b0.x2f(m,ks,j,i);
          if (j == n2-1) {b0.x2f(m,ks-k-1,j+1,i) = b0.x2f(m,ks,j+1,i);}
          b0.x3f(m,ks-k-1,j,i) = b0.x3f(m,ks,j,i);
        }
      }
      if (mb_bcs.d_view(m,BoundaryFace::outer_x3) == BoundaryFlag::user) {
        for (int k=0; k<ng; ++k) {
          b0.x1f(m,ke+k+1,j,i) = b0.x1f(m,ke,j,i);
          if (i == n1-1) {b0.x1f(m,ke+k+1,j,i+1) = b0.x1f(m,ke,j,i+1);}
          b0.x2f(m,ke+k+1,j,i) = b0.x2f(m,ke,j,i);
          if (j == n2-1) {b0.x2f(m,ke+k+1,j+1,i) = b0.x2f(m,ke,j+1,i);}
          b0.x3f(m,ke+k+2,j,i) = b0.x3f(m,ke+1,j,i);
        }
      }
    });
  }
  // ConsToPrim over all X3 ghost zones *and* at the innermost/outermost X3-active zones
  // of Meshblocks, even if Meshblock face is not at the edge of computational domain
  if (pm->pmb_pack->phydro != nullptr) {
    pm->pmb_pack->phydro->peos->ConsToPrim(u0_,w0_,false,0,(n1-1),0,(n2-1),ks-ng,ks);
    pm->pmb_pack->phydro->peos->ConsToPrim(u0_,w0_,false,0,(n1-1),0,(n2-1),ke,ke+ng);
  } else if (pm->pmb_pack->pmhd != nullptr) {
    auto &b0 = pm->pmb_pack->pmhd->b0;
    auto &bcc = pm->pmb_pack->pmhd->bcc0;
    pm->pmb_pack->pmhd->peos->ConsToPrim(u0_,b0,w0_,bcc,false,0,(n1-1),0,(n2-1),ks-ng,ks);
    pm->pmb_pack->pmhd->peos->ConsToPrim(u0_,b0,w0_,bcc,false,0,(n1-1),0,(n2-1),ke,ke+ng);
  }
  // Set X3-BCs on w0 if Meshblock face is at the edge of computational domain
  par_for("noinflow_hydro_x3", DevExeSpace(),0,(nmb-1),0,(nvar-1),0,(n2-1),0,(n1-1),
  KOKKOS_LAMBDA(int m, int n, int j, int i) {
    if (mb_bcs.d_view(m,BoundaryFace::inner_x3) == BoundaryFlag::user) {
      for (int k=0; k<ng; ++k) {
        if (n==(IVZ)) {
          w0_(m,n,ks-k-1,j,i) = fmin(0.0,w0_(m,n,ks,j,i));
        } else {
          w0_(m,n,ks-k-1,j,i) = w0_(m,n,ks,j,i);
        }
      }
    }
    if (mb_bcs.d_view(m,BoundaryFace::outer_x3) == BoundaryFlag::user) {
      for (int k=0; k<ng; ++k) {
        if (n==(IVZ)) {
          w0_(m,n,ke+k+1,j,i) = fmax(0.0,w0_(m,n,ke,j,i));
        } else {
          w0_(m,n,ke+k+1,j,i) = w0_(m,n,ke,j,i);
        }
      }
    }
  });
  if (is_radiation_enabled) {
    // Set X3-BCs on i0 if Meshblock face is at the edge of computational domain
    par_for("noinflow_rad_x3", DevExeSpace(),0,(nmb-1),0,nang1,0,(n2-1),0,(n1-1),
    KOKKOS_LAMBDA(int m, int n, int j, int i) {
      if (mb_bcs.d_view(m,BoundaryFace::inner_x3) == BoundaryFlag::user) {
        for (int k=0; k<ng; ++k) {
          i0_(m,n,ks-k-1,j,i) = i0_(m,n,ks,j,i);
        }
      }
      if (mb_bcs.d_view(m,BoundaryFace::outer_x3) == BoundaryFlag::user) {
        for (int k=0; k<ng; ++k) {
          i0_(m,n,ke+k+1,j,i) = i0_(m,n,ke,j,i);
        }
      }
    });
  }
  // PrimToCons on X3 ghost zones
  if (pm->pmb_pack->phydro != nullptr) {
    pm->pmb_pack->phydro->peos->PrimToCons(w0_,u0_,0,(n1-1),0,(n2-1),ks-ng,ks-1);
    pm->pmb_pack->phydro->peos->PrimToCons(w0_,u0_,0,(n1-1),0,(n2-1),ke+1,ke+ng);
  } else if (pm->pmb_pack->pmhd != nullptr) {
    auto &bcc0_ = pm->pmb_pack->pmhd->bcc0;
    pm->pmb_pack->pmhd->peos->PrimToCons(w0_,bcc0_,u0_,0,(n1-1),0,(n2-1),ks-ng,ks-1);
    pm->pmb_pack->pmhd->peos->PrimToCons(w0_,bcc0_,u0_,0,(n1-1),0,(n2-1),ke+1,ke+ng);
  }

 
  return;
}

void BondiFluxes(HistoryData *pdata, Mesh *pm) {
  MeshBlockPack *pmbp = pm->pmb_pack;

  // extract BH parameters
  bool &flat = pmbp->pcoord->coord_data.is_minkowski;
  Real &spin = pmbp->pcoord->coord_data.bh_spin;

  // set nvars, adiabatic index, primitive array w0, and field array bcc0 if is_mhd
  int nvars; Real gamma; bool is_mhd = false;
  DvceArray5D<Real> w0_, bcc0_;
  if (pmbp->phydro != nullptr) {
    nvars = pmbp->phydro->nhydro + pmbp->phydro->nscalars;
    gamma = pmbp->phydro->peos->eos_data.gamma;
    w0_ = pmbp->phydro->w0;
  } else if (pmbp->pmhd != nullptr) {
    is_mhd = true;
    nvars = pmbp->pmhd->nmhd + pmbp->pmhd->nscalars;
    gamma = pmbp->pmhd->peos->eos_data.gamma;
    w0_ = pmbp->pmhd->w0;
    bcc0_ = pmbp->pmhd->bcc0;
  }

  // extract grids, number of radii, number of fluxes, and history appending index
  auto &grids = pm->pgen->spherical_grids;
  int nradii = grids.size();
  int nflux = (is_mhd) ? 4 : 3;

  // set number of and names of history variables for hydro or mhd
  //  (1) mass accretion rate
  //  (2) energy flux
  //  (3) angular momentum flux
  //  (4) magnetic flux (iff MHD)
  pdata->nhist = nradii*nflux;
  if (pdata->nhist > NHISTORY_VARIABLES) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "User history function specified pdata->nhist larger than"
              << " NHISTORY_VARIABLES" << std::endl;
    exit(EXIT_FAILURE);
  }
  for (int g=0; g<nradii; ++g) {
    std::stringstream stream;
    stream << std::fixed << std::setprecision(1) << grids[g]->radius;
    std::string rad_str = stream.str();
    pdata->label[nflux*g+0] = "mdot_" + rad_str;
    pdata->label[nflux*g+1] = "edot_" + rad_str;
    pdata->label[nflux*g+2] = "ldot_" + rad_str;
    if (is_mhd) {
      pdata->label[nflux*g+3] = "phi_" + rad_str;
      //pdata->label[nflux*g+4] = "EMdot_" + rad_str;
    }
  }

  // go through angles at each radii:
  DualArray2D<Real> interpolated_bcc;  // needed for MHD
  for (int g=0; g<nradii; ++g) {
    // zero fluxes at this radius
    pdata->hdata[nflux*g+0] = 0.0;
    pdata->hdata[nflux*g+1] = 0.0;
    pdata->hdata[nflux*g+2] = 0.0;
    if (is_mhd) pdata->hdata[nflux*g+3] = 0.0;

    // interpolate primitives (and cell-centered magnetic fields iff mhd)
    if (is_mhd) {
      grids[g]->InterpolateToSphere(3, bcc0_);
      Kokkos::realloc(interpolated_bcc, grids[g]->nangles, 3);
      Kokkos::deep_copy(interpolated_bcc, grids[g]->interp_vals);
      interpolated_bcc.template modify<DevExeSpace>();
      interpolated_bcc.template sync<HostMemSpace>();
    }
    grids[g]->InterpolateToSphere(nvars, w0_);

    // compute fluxes
    for (int n=0; n<grids[g]->nangles; ++n) {
      // extract coordinate data at this angle
      Real r = grids[g]->radius;
      Real theta = grids[g]->polar_pos.h_view(n,0);
      Real phi = grids[g]->polar_pos.h_view(n,1);
      Real x1 = grids[g]->interp_coord.h_view(n,0);
      Real x2 = grids[g]->interp_coord.h_view(n,1);
      Real x3 = grids[g]->interp_coord.h_view(n,2);
      Real glower[4][4], gupper[4][4];
      ComputeMetricAndInverse(x1,x2,x3,flat,spin,glower,gupper);

      // extract interpolated primitives
      Real &int_dn = grids[g]->interp_vals.h_view(n,IDN);
      Real &int_vx = grids[g]->interp_vals.h_view(n,IVX);
      Real &int_vy = grids[g]->interp_vals.h_view(n,IVY);
      Real &int_vz = grids[g]->interp_vals.h_view(n,IVZ);
      Real &int_ie = grids[g]->interp_vals.h_view(n,IEN);

      // extract interpolated field components (iff is_mhd)
      Real int_bx = 0.0, int_by = 0.0, int_bz = 0.0;
      if (is_mhd) {
        int_bx = interpolated_bcc.h_view(n,IBX);
        int_by = interpolated_bcc.h_view(n,IBY);
        int_bz = interpolated_bcc.h_view(n,IBZ);
      }

      // Compute interpolated u^\mu in CKS
      Real q = glower[1][1]*int_vx*int_vx + 2.0*glower[1][2]*int_vx*int_vy +
               2.0*glower[1][3]*int_vx*int_vz + glower[2][2]*int_vy*int_vy +
               2.0*glower[2][3]*int_vy*int_vz + glower[3][3]*int_vz*int_vz;
      Real alpha = sqrt(-1.0/gupper[0][0]);
      Real lor = sqrt(1.0 + q);
      Real u0 = lor/alpha;
      Real u1 = int_vx - alpha * lor * gupper[0][1];
      Real u2 = int_vy - alpha * lor * gupper[0][2];
      Real u3 = int_vz - alpha * lor * gupper[0][3];

      // Lower vector indices
      Real u_0 = glower[0][0]*u0 + glower[0][1]*u1 + glower[0][2]*u2 + glower[0][3]*u3;
      Real u_1 = glower[1][0]*u0 + glower[1][1]*u1 + glower[1][2]*u2 + glower[1][3]*u3;
      Real u_2 = glower[2][0]*u0 + glower[2][1]*u1 + glower[2][2]*u2 + glower[2][3]*u3;
      Real u_3 = glower[3][0]*u0 + glower[3][1]*u1 + glower[3][2]*u2 + glower[3][3]*u3;

      // Calculate 4-magnetic field (returns zero if not MHD)
      Real b0 = u_1*int_bx + u_2*int_by + u_3*int_bz;
      Real b1 = (int_bx + b0 * u1) / u0;
      Real b2 = (int_by + b0 * u2) / u0;
      Real b3 = (int_bz + b0 * u3) / u0;

      // compute b_\mu in CKS and b_sq (returns zero if not MHD)
      Real b_0 = glower[0][0]*b0 + glower[0][1]*b1 + glower[0][2]*b2 + glower[0][3]*b3;
      Real b_1 = glower[1][0]*b0 + glower[1][1]*b1 + glower[1][2]*b2 + glower[1][3]*b3;
      Real b_2 = glower[2][0]*b0 + glower[2][1]*b1 + glower[2][2]*b2 + glower[2][3]*b3;
      Real b_3 = glower[3][0]*b0 + glower[3][1]*b1 + glower[3][2]*b2 + glower[3][3]*b3;
      Real b_sq = b0*b_0 + b1*b_1 + b2*b_2 + b3*b_3;

      // Transform CKS 4-velocity and 4-magnetic field to spherical KS
      Real a2 = SQR(spin);
      Real rad2 = SQR(x1)+SQR(x2)+SQR(x3);
      Real r2 = SQR(r);
      Real sth = sin(theta);
      Real sph = sin(phi);
      Real cph = cos(phi);
      Real drdx = r*x1/(2.0*r2 - rad2 + a2);
      Real drdy = r*x2/(2.0*r2 - rad2 + a2);
      Real drdz = (r*x3 + a2*x3/r)/(2.0*r2-rad2+a2);
      // contravariant r component of 4-velocity
      Real ur  = drdx *u1 + drdy *u2 + drdz *u3;
      // contravariant r component of 4-magnetic field (returns zero if not MHD)
      Real br  = drdx *b1 + drdy *b2 + drdz *b3;
      // covariant phi component of 4-velocity
      Real u_ph = (-r*sph-spin*cph)*sth*u_1 + (r*cph-spin*sph)*sth*u_2;
      // covariant phi component of 4-magnetic field (returns zero if not MHD)
      Real b_ph = (-r*sph-spin*cph)*sth*b_1 + (r*cph-spin*sph)*sth*b_2;


      // compute flux density terms
      Real fTH = (gamma*int_ie)*ur*u_0; // thermal flux
      Real fKE = (int_dn)*ur*u_0; // Kinetic flux
      Real fEM = (b_sq)*ur*u_0 - br*b_0; // electromagnetic flux
      Real fRM = int_dn*ur; // rest mass flux
      
      // integration params
      Real &domega = grids[g]->solid_angles.h_view(n);
      Real sqrtmdet = (r2+SQR(spin*cos(theta)));

      // compute mass flux
      pdata->hdata[nflux*g+0] += -1.0*int_dn*ur*sqrtmdet*domega;

      // compute energy flux
      Real t1_0 = (int_dn + gamma*int_ie + b_sq)*ur*u_0 - br*b_0;
      pdata->hdata[nflux*g+1] += -1.0*t1_0*sqrtmdet*domega;

      // compute angular momentum flux
      Real t1_3 = (int_dn + gamma*int_ie + b_sq)*ur*u_ph - br*b_ph;
      pdata->hdata[nflux*g+2] += t1_3*sqrtmdet*domega;

      // compute magnetic flux
      if (is_mhd) {
        pdata->hdata[nflux*g+3] += 0.5*fabs(br*u0 - b0*ur)*sqrtmdet*domega;
	//pdata->hdata[nflux*g+4] += -1.0*(b_sq*ur*u_0 - br*b_0)*sqrtmdet*domega;
	Real mu = (fTH + fKE + fEM + fRM)/fRM;
	Real sigma = fEM/fKE;
	
      }
    }
  }

  // fill rest of the_array with zeros, if nhist < NHISTORY_VARIABLES
  for (int n=pdata->nhist; n<NHISTORY_VARIABLES; ++n) {
    pdata->hdata[n] = 0.0;
  }

  return;
}
