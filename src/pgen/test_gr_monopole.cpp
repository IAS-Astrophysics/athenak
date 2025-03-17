//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file gr_torus.cpp
//! \brief Test problem that initializes a monopole as prescribed by Michel 1973 to check particles
//! accelerated as per theoretical prediction in GR+electromagnetic forces
//!
//! References:
//!    Fishbone & Moncrief 1976, ApJ 207 962 (FM)
//!    Fishbone 1977, ApJ 215 323 (F)
//!    Chakrabarti, S. 1985, ApJ 288, 1

#include <stdio.h>
#include <math.h>

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

#include <algorithm>  // max(), max_element(), min(), min_element()
#include <iomanip>
#include <iostream>   // endl
#include <limits>     // numeric_limits::max()
#include <memory>
#include <sstream>    // stringstream
#include <string>     // c_str(), string
#include <vector>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/coordinates.hpp"
#include "coordinates/cartesian_ks.hpp"
#include "coordinates/cell_locations.hpp"
#include "mhd/mhd.hpp"

#include "particles/particles.hpp"

#include <Kokkos_Random.hpp>

// prototypes for functions used internally to this pgen
namespace {

KOKKOS_INLINE_FUNCTION
static Real CalculateCovariantUT(Real spin, Real r, Real sin_theta, Real l);

KOKKOS_INLINE_FUNCTION
static void GetBoyerLindquistCoordinates(Real spin,
                                         Real x1, Real x2, Real x3,
                                         Real *pr, Real *ptheta, Real *pphi);

KOKKOS_INLINE_FUNCTION
static void TransformVector(Real spin,
                            Real a0_bl, Real a1_bl, Real a2_bl, Real a3_bl,
                            Real x1, Real x2, Real x3,
                            Real *pa0, Real *pa1, Real *pa2, Real *pa3);

} // namespace

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::UserProblem()
//! \brief Sets initial conditions for either Fishbone-Moncrief or Chakrabarti torus in GR
//! Compile with '-D PROBLEM=gr_torus' to enroll as user-specific problem generator
//!  assumes x3 is axisymmetric direction

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (!pmbp->pcoord->is_general_relativistic) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "GR torus problem can only be run when GR defined in <coord> block"
              << std::endl;
    exit(EXIT_FAILURE);
  }

  // capture variables for kernel
  auto &indcs = pmy_mesh_->mb_indcs;
  int is = indcs.is, js = indcs.js, ks = indcs.ks;
  int ie = indcs.ie, je = indcs.je, ke = indcs.ke;
  int nmb = pmbp->nmb_thispack;
  auto &coord = pmbp->pcoord->coord_data;

  // Extract BH parameters
  const Real r_excise = coord.rexcise;
  const bool is_radiation_enabled = (pmbp->prad != nullptr);

  const bool has_particles = (pmbp->ppart != nullptr);
  Real B0 = pin->GetOrAddReal("problem", "b0_strength", 1.0);
  // Need to split checks, otherwise a "particles" block will get initialized and written to the restart file
  // Then when trying to restart the mhd part this gives error because it looks for necessary particles parameters
  if (has_particles) {
    const bool inject_particles = pin->GetOrAddBoolean("particles", "inject", false);
    auto &size = pmbp->pmb->mb_size;
    if (inject_particles) {
      auto &indcs = pmbp->pmesh->mb_indcs;
      int &ng = indcs.ng;
      int n1 = indcs.nx1 + 2*ng;
      int n2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng) : 1;
      int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng) : 1;
      const int is = indcs.is;
      const int js = indcs.js;
      const int ks = indcs.ks;
      auto &gids = pmbp->gids;
      auto &gide = pmbp->gide;
      auto &coord = pmbp->pcoord->coord_data;
      int &npart = pmbp->ppart->nprtcl_thispack;
      auto &pr = pmbp->ppart->prtcl_rdata;
      auto &pi = pmbp->ppart->prtcl_idata;
      Real massive = 1.0; //This should be based on ptype, not hard-coded
      Real min_en = pin->GetOrAddReal("problem", "prtcl_energy_min", 1.005);
      Real max_en = pin->GetOrAddReal("problem", "prtcl_energy_max", 1.5);
      std::string prtcl_init_type = pin->GetString("particles","init_type");
      const Real q_over_m = pin->GetOrAddReal("particles", "charge_over_mass", 1);
      // Need these booleans on device, can't use std::string
      // .compare() returns 0 for successful comparison, which is opposite of usual boolean
      const bool prtcl_init_rnd = !(prtcl_init_type.compare("random"));
      const bool prtcl_init_flow = !(prtcl_init_type.compare("flow_align"));
      const bool prtcl_init_blob = !(prtcl_init_type.compare("blob"));
      const bool prtcl_init_rad = !(prtcl_init_type.compare("shell"));
      const bool is_gca = pmbp->ppart->is_gca;
      // Check initialization type has been set

      DvceArray5D<Real> u0_, w0_;
      DvceArray5D<Real> bcc_;
      auto &bface_ = pmbp->pmhd->b0;
      u0_ = pmbp->pmhd->u0;
      w0_ = pmbp->pmhd->w0;
      bcc_ = pmbp->pmhd->bcc0;
      pmbp->pmhd->peos->ConsToPrim(u0_,bface_,w0_,bcc_,false,0,(n1-1),0,(n2-1),0,(n3-1));

      // Define criterium for how to initialize particles
      Real min_rad = pmbp->ppart->min_radius;
      Real crit, crit_min;
      bool is_crit_satisfied;
      int nmb = (pmbp->nmb_thispack);
      // Array stores whether or not the Meshblock is viable for particle injection
      // based on criterium
      DvceArray1D<bool> mb_for_injection;
      Kokkos::realloc(mb_for_injection, nmb);
      par_for("init_mb_for_injection", DevExeSpace(), 0, nmb-1,
          KOKKOS_LAMBDA( const int &im ) {
          mb_for_injection[im] = false;
        });

      // Initialize particles within a specific spherical shell
      if ( ! pin->DoesParameterExist("particles", "r_init_max") ) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
          << "Particle initialization type " << prtcl_init_type <<" missing required parameter: " << "r_init_max" << std::endl;
        std::exit(EXIT_FAILURE);
      }
      crit = pin->GetReal("particles", "r_init_max");
      crit_min = pin->GetOrAddReal("particles", "r_init_min", 0.0); 
      int mbs_in_shell = 0;
      const int try_lim = 15;
      min_rad = fmax( min_rad, crit_min );

      Kokkos::parallel_reduce("pgen_mbp_checkcondition", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmb),
      KOKKOS_LAMBDA(const int &m, int &mb_count ) {
        // Notice the absolute value
        Real x1min = fabs(size.d_view(m).x1min);
        Real x1max = fabs(size.d_view(m).x1max);
        Real x1i = fmin( x1min, x1max );
        Real x1o = fmax( x1min, x1max );
        Real x2min = fabs(size.d_view(m).x2min);
        Real x2max = fabs(size.d_view(m).x2max);
        Real x2i = fmin( x2min, x2max );
        Real x2o = fmax( x2min, x2max );
        Real x3min = fabs(size.d_view(m).x3min);
        Real x3max = fabs(size.d_view(m).x3max);
        Real x3i = fmin( x3min, x3max );
        Real x3o = fmax( x3min, x3max );
        Real r_i, r_o, th, phi;
        GetBoyerLindquistCoordinates(coord.bh_spin, x1i, x2i, x3i, &r_i, &th, &phi);
        GetBoyerLindquistCoordinates(coord.bh_spin, x1o, x2o, x3o, &r_o, &th, &phi);
        //Determine whether the meshblock with index m has cells within the spherical shell
        mb_for_injection[m] = ( mb_for_injection[m] || ( r_o > min_rad && r_i < crit ) );
        if ( mb_for_injection[m] ) { ++mb_count; }
      }, Kokkos::Sum<int>(mbs_in_shell) );
      
      is_crit_satisfied = ( mbs_in_shell > 0 );
      std::cout << "MBs in shell: " << mbs_in_shell << std::endl;

      if ( ! is_crit_satisfied ) {
        npart = 0;
        Kokkos::realloc(pr, pmbp->ppart->nrdata, 0);
        Kokkos::realloc(pi, pmbp->ppart->nidata, 0);
        std::cout << "None of the MBs on this rank satisfy the injection criterium. Deleted particles."<< std::endl;
      } else {

        Kokkos::Random_XorShift64_Pool<> prtcl_rand(gids);

        par_for("part_init", DevExeSpace(),0,(npart-1),
          KOKKOS_LAMBDA(const int p){
            bool found_mb = false;
            auto prtcl_gen = prtcl_rand.get_state();
            while(!found_mb){
              int m = static_cast<int>(prtcl_gen.frand()*(gide-gids+1.0));
              while ( !mb_for_injection[m] ) {
                m = static_cast<int>(prtcl_gen.frand()*(gide-gids+1.0));
              }
              // First check that the meshblock is within the disk, and then outside the horizon
              Real &x1min = size.d_view(m).x1min;
              Real &x1max = size.d_view(m).x1max;
              Real x1v = x1min + prtcl_gen.frand()*(x1max - x1min);
              Real &x2min = size.d_view(m).x2min;
              Real &x2max = size.d_view(m).x2max;
              Real x2v = x2min + prtcl_gen.frand()*(x2max - x2min); 
              Real &x3min = size.d_view(m).x3min;
              Real &x3max = size.d_view(m).x3max;
              Real x3v = x3min + prtcl_gen.frand()*(x3max - x3min);
              Real r, th, phi;
              GetBoyerLindquistCoordinates(coord.bh_spin, x1v, x2v, x3v, &r, &th, &phi);
              if (r >= min_rad){
                int try_this_mb = 0;
                while ( ( r < min_rad || r > crit ) && try_this_mb <= try_lim ) {
                  x1v = x1min + prtcl_gen.frand()*(x1max - x1min);
                  x2v = x2min + prtcl_gen.frand()*(x2max - x2min);
                  x3v = x3min + prtcl_gen.frand()*(x3max - x3min);
                  GetBoyerLindquistCoordinates(coord.bh_spin, x1v, x2v, x3v, &r, &th, &phi);
                  ++try_this_mb;
                }
                if (try_this_mb >= try_lim) {
                  continue;
                }
                found_mb = true;
                if (!is_gca) {
                  Real up0, up1, up2, up3;
                  Real ux0, ux1, ux2, ux3;
                  up3 = max_en;
                  up0 = CalculateCovariantUT(coord.bh_spin, r, sin(th), up3);
                  //up3 *= up0;
                  
                  // These velocities are contravariant
                  TransformVector(coord.bh_spin,
                                up0, 0, 0, up3,
                                x1v, x2v, x3v,
                                &ux0, &ux1, &ux2, &ux3);
                  Real gu[4][4], gl[4][4];
                  ComputeMetricAndInverse(x1v,x2v,x3v,coord.is_minkowski,coord.bh_spin,gl,gu); 
                  Real u0 = gl[1][1]*SQR(ux1) + gl[2][2]*SQR(ux2) + gl[3][3]*SQR(ux3)
                        + 2.0*gl[1][2]*ux1*ux2 + 2.0*gl[1][3]*ux1*ux3
                        + 2.0*gl[3][2]*ux3*ux2;
                  u0 = sqrt(u0 + massive); 
                  pr(IPVX,p) = gl[1][1]*ux1 + gl[1][2]*ux2 + gl[1][3]*ux3;
                  pr(IPVY,p) = gl[2][1]*ux1 + gl[2][2]*ux2 + gl[2][3]*ux3;
                  pr(IPVZ,p) = gl[3][1]*ux1 + gl[3][2]*ux2 + gl[3][3]*ux3;
                } else {
                  // For GCA the prtcl_energy_max actually acts to set the gamma factor/energy
                  pr(IPVX,p) = sqrt(max_en);
                  pr(IPVY,p) = min_en;
                }
                pi(PGID,p) = gids+m;
                pr(IPX,p) = x1v;
                pr(IPY,p) = x2v;
                pr(IPZ,p) = x3v;
              }
            }
            prtcl_rand.free_state(prtcl_gen);
        });
        std::cout << "Injected " << npart << " particles." << std::endl;
      }
    }
    // set timestep (which will remain constant for entire run
    // Assumes uniform mesh (no SMR or AMR)
    // Assumes velocities normalized to one, so dt=min(dx)
    Real &dtnew_ = pmbp->ppart->dtnew;
    dtnew_ = std::min(size.h_view(0).dx1, size.h_view(0).dx2);
    dtnew_ = std::min(dtnew_, size.h_view(0).dx3);
    dtnew_ *= pin->GetOrAddReal("time", "cfl_number", 0.8);
    pmbp->pmesh->dt = dtnew_;
  }

  // return if restart
  if (restart) return;

  // Select either Hydro or MHD
  DvceArray5D<Real> u0_, w0_;
  u0_ = pmbp->pmhd->u0;
  w0_ = pmbp->pmhd->w0;

  // initialize primitive variables for new run ---------------------------------------

  Real r_s = 2.0;
  Real R_LC = 10.0;
  auto &size = pmbp->pmb->mb_size;
  auto &b0 = pmbp->pmhd->b0;
  auto &e0 = pmbp->pmhd->efld;
  Kokkos::Random_XorShift64_Pool<> rand_pool64(pmbp->gids);
  Real ptotmax = std::numeric_limits<float>::min();
  const int nmkji = (pmbp->nmb_thispack)*indcs.nx3*indcs.nx2*indcs.nx1;
  const int nkji = indcs.nx3*indcs.nx2*indcs.nx1;
  const int nji  = indcs.nx2*indcs.nx1;

  par_for("pgen_monopole",DevExeSpace(),0,(nmb-1),ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {

    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);

    // Extract metric and inverse
    Real glower[4][4], gupper[4][4];
    ComputeMetricAndInverse(x1v, x2v, x3v, coord.is_minkowski, coord.bh_spin,
                            glower, gupper);

    // Calculate Boyer-Lindquist coordinates of cell
    Real r, theta, phi;
    GetBoyerLindquistCoordinates(coord.bh_spin, x1v, x2v, x3v, &r, &theta, &phi);
    Real sin_theta = sin(theta);
    Real cos_theta = cos(theta);
    Real sin_phi = sin(phi);
    Real cos_phi = cos(phi);

    // Calculate background primitives
    Real rho_min = pin->GetReal("problem", "rho_min");
    Real rho_pow = pin->GetReal("problem", "rho_pow");
    Real pgas_min = pin->GetReal("problem", "pgas_min");
    Real pgas_pow = pin->GetReal("problem", "pgas_pow");
    Real rho_bg, pgas_bg;
    if (r > 1.0) {
      rho_bg = rho_min * pow(r, rho_pow);
      pgas_bg = pgas_min * pow(r, pgas_pow);
    } else {
      rho_bg = coord.dexcise;
      pgas_bg = coord.pexcise;
    }

    Real rho = rho_bg;
    Real pgas = pgas_bg;
    Real uu1 = 0.0;
    Real uu2 = 0.0;
    Real uu3 = 0.0;

    // Set primitive values, including random perturbations to pressure
    w0_(m,IDN,k,j,i) = fmax(rho, rho_bg);
    w0_(m,IEN,k,j,i) = fmax(pgas, pgas_bg);
    w0_(m,IVX,k,j,i) = uu1;
    w0_(m,IVY,k,j,i) = uu2;
    w0_(m,IVZ,k,j,i) = uu3;

    if (r < r_s) {
      b0.x1f(m,k,j,i) = 0.0;
      b0.x2f(m,k,j,i) = 0.0;
      b0.x3f(m,k,j,i) = 0.0;
      e0.x1e(m,k,j,i) = 0.0;
      e0.x2e(m,k,j,i) = 0.0;
      e0.x3e(m,k,j,i) = 0.0;
    } else {
      Real D_theta = B0*(r_s/R_LC)*(r_s/r)*sin_theta;
      Real B_phi = D_theta;
      Real B_r = B0*SQR(r_s/r);
      Real B_0, B_x, B_y, B_z;
      Real E_0, E_x, E_y, E_z;
    
      TransformVector(coord.bh_spin,
                    0, B_r, 0, B_phi,
                    x1v, x2v, x3v,
                    &B_0, &B_x, &B_y, &B_z);
      b0.x1f(m,k,j,i) = B_x;
      b0.x2f(m,k,j,i) = B_y;
      b0.x3f(m,k,j,i) = B_z;

      TransformVector(coord.bh_spin,
                    0, 0, D_theta, 0,
                    x1v, x2v, x3v,
                    &E_0, &E_x, &E_y, &E_z);
      e0.x1e(m,k,j,i) = E_x;
      e0.x2e(m,k,j,i) = E_y;
      e0.x3e(m,k,j,i) = E_z;

      if (i==ie) {
       b0.x1f(m,k,j,i+1) = B_x;
       e0.x1e(m,k,j,i+1) = E_x;
      }
      if (j==je) {
        b0.x2f(m,k,j+1,i) = B_y;
        e0.x2e(m,k,j+1,i) = E_y;
      }
      if (k==ke) {
        b0.x3f(m,k+1,j,i) = B_z;
        e0.x3e(m,k+1,j,i) = E_z;
      }
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

  // Convert primitives to conserved
  auto &bcc0_ = pmbp->pmhd->bcc0;
  pmbp->pmhd->peos->PrimToCons(w0_, bcc0_, u0_, is, ie, js, je, ks, ke);
  

  return;
}

namespace {
//----------------------------------------------------------------------------------------
// Function for returning corresponding Boyer-Lindquist coordinates of point
// Inputs:
//   x1,x2,x3: global coordinates to be converted
// Outputs:
//   pr,ptheta,pphi: variables pointed to set to Boyer-Lindquist coordinates

KOKKOS_INLINE_FUNCTION
static void GetBoyerLindquistCoordinates(Real spin,
                                         Real x1, Real x2, Real x3,
                                         Real *pr, Real *ptheta, Real *pphi) {
  Real rad = sqrt(SQR(x1) + SQR(x2) + SQR(x3));
  Real r = fmax((sqrt( SQR(rad) - SQR(spin) + sqrt(SQR(SQR(rad)-SQR(spin))
                      + 4.0*SQR(spin)*SQR(x3)) ) / sqrt(2.0)), 1.0);
  *pr = r;
  *ptheta = (fabs(x3/r) < 1.0) ? acos(x3/r) : acos(copysign(1.0, x3));
  *pphi = atan2(r*x2-spin*x1, spin*x2+r*x1) -
          spin*r/(SQR(r)-2.0*r+SQR(spin));
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
static void TransformVector(Real spin,
                            Real a0_bl, Real a1_bl, Real a2_bl, Real a3_bl,
                            Real x1, Real x2, Real x3,
                            Real *pa0, Real *pa1, Real *pa2, Real *pa3) {
  Real rad = sqrt( SQR(x1) + SQR(x2) + SQR(x3) );
  Real r = fmax((sqrt( SQR(rad) - SQR(spin) + sqrt(SQR(SQR(rad)-SQR(spin))
                      + 4.0*SQR(spin)*SQR(x3)) ) / sqrt(2.0)), 1.0);
  Real delta = SQR(r) - 2.0*r + SQR(spin);
  *pa0 = a0_bl + 2.0*r/delta * a1_bl;
  *pa1 = a1_bl * ( (r*x1+spin*x2)/(SQR(r) + SQR(spin)) - x2*spin/delta) +
         a2_bl * x1*x3/r * sqrt((SQR(r) + SQR(spin))/(SQR(x1) + SQR(x2))) -
         a3_bl * x2;
  *pa2 = a1_bl * ( (r*x2-spin*x1)/(SQR(r) + SQR(spin)) + x1*spin/delta) +
         a2_bl * x2*x3/r * sqrt((SQR(r) + SQR(spin))/(SQR(x1) + SQR(x2))) +
         a3_bl * x1;
  *pa3 = a1_bl * x3/r -
         a2_bl * r * sqrt((SQR(x1) + SQR(x2))/(SQR(r) + SQR(spin)));
  return;
}

//----------------------------------------------------------------------------------------
// Function to calculate time component of contravariant four velocity in BL
// Inputs:
//   r: radial Boyer-Lindquist coordinate
//   sin_theta: sine of polar Boyer-Lindquist coordinate
// Outputs:
//   returned value: u_t

KOKKOS_INLINE_FUNCTION
static Real CalculateCovariantUT(Real spin, Real r, Real sin_theta, Real l) {
  // Compute BL metric components
  Real sigma = SQR(r) + SQR(spin)*(1.0-SQR(sin_theta));
  Real g_00 = -1.0 + 2.0*r/sigma;
  Real g_03 = -2.0*spin*r/sigma*SQR(sin_theta);
  Real g_33 = (SQR(r) + SQR(spin) +
               2.0*SQR(spin)*r/sigma*SQR(sin_theta))*SQR(sin_theta);

  // Compute time component of covariant BL 4-velocity
  Real u_t = -sqrt(fmax((SQR(g_03) - g_00*g_33)/(g_33 + 2.0*l*g_03 + SQR(l)*g_00), 0.0));
  return u_t;
}
} // namespace
