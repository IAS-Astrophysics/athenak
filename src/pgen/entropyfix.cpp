//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file entropyfix.cpp
//! \brief Problem generator for entropyfix test.

// C++ headers
#include <cmath>
#include <cstdio> // fopen(), fprintf(), freopen()

// Athena++ headers
#include "athena.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "srcterms/srcterms.hpp"
#include "utils/random.hpp"
#include "pgen.hpp"
#include "geodesic-grid/geodesic_grid.hpp"
#include "mhd/mhd.hpp"
#include "radiation/radiation.hpp"

#include <Kokkos_Random.hpp>

static Real rho_min,     rho_ctr,     rho_max;
static Real tgas_min,    tgas_ctr,    tgas_max;
static Real v_sqr_min,   v_sqr_ctr,   v_sqr_max;
static Real bcc_sqr_min, bcc_sqr_ctr, bcc_sqr_max;

static Real dd_ptb_max, dd_ptb_min;
static Real ee_ptb_max, ee_ptb_min;
static Real mm_ptb_max, mm_ptb_min;
static Real bb_ptb_max, bb_ptb_min;

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::UserProblem()
//  \brief Problem Generator for the Rayleigh-Taylor instability test

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  // return if restart
  if (restart) return;

  // mesh and flags
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  const bool is_two_d = pmbp->pmesh->two_d;
  const bool is_gr_enabled = pmbp->pcoord->is_general_relativistic;
  const bool is_flatspacetime  = pmbp->pcoord->coord_data.is_minkowski;
  const bool is_mhd_enabled = (pmbp->pmhd != nullptr);
  const bool is_rad_enabled = (pmbp->prad != nullptr);
  const bool is_angular_flux_enabled = (is_rad_enabled) ? pmbp->prad->angular_fluxes : false;

  // flag check
  if (!is_two_d) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "problem generator for entropyfix test only works in 2D" << std::endl;
    exit(EXIT_FAILURE);
  }
  if (!is_mhd_enabled) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "problem generator for entropyfix test only works in MHD" << std::endl;
    exit(EXIT_FAILURE);
  }
  if ((!is_gr_enabled)
   || (!is_flatspacetime)
   || (is_angular_flux_enabled)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "problem generator for entropyfix test only works in flat spacetime" << std::endl;
    exit(EXIT_FAILURE);
  }

  // parameters for entropyfix test
  rho_min     = pin->GetOrAddReal("problem", "rho_min",     1.e-8);
  rho_ctr     = pin->GetOrAddReal("problem", "rho_ctr",     1.e-6);
  rho_max     = pin->GetOrAddReal("problem", "rho_max",     1.e-4);
  tgas_min    = pin->GetOrAddReal("problem", "tgas_min",    1.e-4);
  tgas_ctr    = pin->GetOrAddReal("problem", "tgas_ctr",    1.e-2);
  tgas_max    = pin->GetOrAddReal("problem", "tgas_max",    1.e0);
  v_sqr_min   = pin->GetOrAddReal("problem", "v_sqr_min",   1.e-4);
  v_sqr_ctr   = pin->GetOrAddReal("problem", "v_sqr_ctr",   1.e-2);
  v_sqr_max   = pin->GetOrAddReal("problem", "v_sqr_max",   1.-1.e-4);
  bcc_sqr_min = pin->GetOrAddReal("problem", "bcc_sqr_min", 1.e-6);
  bcc_sqr_ctr = pin->GetOrAddReal("problem", "bcc_sqr_ctr", 1.e-4);
  bcc_sqr_max = pin->GetOrAddReal("problem", "bcc_sqr_max", 1.e-2);

  dd_ptb_max = pin->GetOrAddReal("problem", "dd_ptb_max", 0.);
  dd_ptb_min = pin->GetOrAddReal("problem", "dd_ptb_min", 0.);
  ee_ptb_max = pin->GetOrAddReal("problem", "ee_ptb_max", 0.);
  ee_ptb_min = pin->GetOrAddReal("problem", "ee_ptb_min", 0.);
  mm_ptb_max = pin->GetOrAddReal("problem", "mm_ptb_max", 0.);
  mm_ptb_min = pin->GetOrAddReal("problem", "mm_ptb_min", 0.);
  bb_ptb_max = pin->GetOrAddReal("problem", "bb_ptb_max", 0.);
  bb_ptb_min = pin->GetOrAddReal("problem", "bb_ptb_min", 0.);

  // define each region
  Real x1f_min = pin->GetReal("mesh", "x1min");
  Real x1f_max = pin->GetReal("mesh", "x1max");
  Real x2f_min = pin->GetReal("mesh", "x2min");
  Real x2f_max = pin->GetReal("mesh", "x2max");

  Real x2_wdn_min  = (x2f_max-x2f_min)*0./4 + x2f_min;
  Real x2_wdn_max  = (x2f_max-x2f_min)*1./4 + x2f_min;
  Real x2_wen_min  = (x2f_max-x2f_min)*1./4 + x2f_min;
  Real x2_wen_max  = (x2f_max-x2f_min)*2./4 + x2f_min;
  Real x2_wv_min   = (x2f_max-x2f_min)*2./4 + x2f_min;
  Real x2_wv_max   = (x2f_max-x2f_min)*3./4 + x2f_min;
  Real x2_bcc_min  = (x2f_max-x2f_min)*3./4 + x2f_min;
  Real x2_bcc_max  = (x2f_max-x2f_min)*4./4 + x2f_min;

  // capture variables for kernel
  auto &indcs = pmy_mesh_->mb_indcs;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  int ncells1 = indcs.nx1 + 2*(indcs.ng);
  int ncells2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*(indcs.ng)) : 1;
  int ncells3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*(indcs.ng)) : 1;
  int nmb = pmbp->nmb_thispack;
  auto &size = pmbp->pmb->mb_size;
  int nmhd = pmbp->pmhd->nmhd;
  int nscal = pmbp->pmhd->nscalars;
  bool entropy_fix_ = pmbp->pmhd->entropy_fix;
  int entropyIdx = (entropy_fix_) ? nmhd+nscal-1 : -1;

  // set EOS data
  Real gm1 = pmbp->pmhd->peos->eos_data.gamma - 1.0;

  // set primitives
  auto &w0 = pmbp->pmhd->w0;
  auto &u0 = pmbp->pmhd->u0;
  par_for("entropyfix_w0",DevExeSpace(),0,(nmb-1),ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    // coordinates
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

    // set primitives
    Real rho_   = rho_ctr;
    Real tgas_  = tgas_ctr;
    Real v_sqr_ = v_sqr_ctr;

    // varying density
    if ((x2v>=x2_wdn_min) && (x2v<x2_wdn_max)) {
      Real lg_rho = (log10(rho_max)-log10(rho_min))/(x1f_max-x1f_min) * (x1v-x1f_min) + log10(rho_min);
      rho_ = pow(10., lg_rho);
    }

    // varying internal energy
    if ((x2v>=x2_wen_min) && (x2v<x2_wen_max)) {
      Real lg_tgas = (log10(tgas_max)-log10(tgas_min))/(x1f_max-x1f_min) * (x1v-x1f_min) + log10(tgas_min);
      tgas_ = pow(10., lg_tgas);
    }

    // varying velocity
    if ((x2v>=x2_wv_min) && (x2v<x2_wv_max)) {
      Real lg_v_sqr = (log10(v_sqr_max)-log10(v_sqr_min))/(x1f_max-x1f_min) * (x1v-x1f_min) + log10(v_sqr_min);
      v_sqr_ = pow(10., lg_v_sqr);
    }

    Real gamma = 1./sqrt(1-v_sqr_);
    w0(m,IDN,k,j,i) = rho_;
    w0(m,IEN,k,j,i) = rho_ctr*tgas_/gm1;
    w0(m,IVX,k,j,i) = gamma * sqrt(v_sqr_/2);
    w0(m,IVY,k,j,i) = gamma * sqrt(v_sqr_/2);
    w0(m,IVZ,k,j,i) = 0.0;
  }); // end_par_for pbi_w0

  // set magnetic field
  auto &b0 = pmbp->pmhd->b0;
  auto &bcc0 = pmbp->pmhd->bcc0;
  par_for("pgen_b0",DevExeSpace(),0,(nmb-1),ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    // coordinates
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);
    Real x2f = LeftEdgeX(j-js, indcs.nx2, x2min, x2max);

    // magnetic field
    Real bcc_sqr_ = bcc_sqr_ctr;

    // varying magnetic field
    if ((x2v>=x2_bcc_min) && (x2v<x2_bcc_max)) {
      Real lg_bcc_sqr = (log10(bcc_sqr_max)-log10(bcc_sqr_min))/(x1f_max-x1f_min) * (x1v-x1f_min) + log10(bcc_sqr_min);
      bcc_sqr_ = pow(10., lg_bcc_sqr);
    }

    b0.x1f(m,k,j,i) = sqrt(bcc_sqr_/2);
    b0.x2f(m,k,j,i) = sqrt(bcc_sqr_/2);
    b0.x3f(m,k,j,i) = 0.0;
    if (i==ie) b0.x1f(m,k,j,i+1) = sqrt(bcc_sqr_/2);
    if (j==je) b0.x2f(m,k,j+1,i) = sqrt(bcc_sqr_/2);
    if (k==ke) b0.x3f(m,k+1,j,i) = 0.0;
    bcc0(m,IBX,k,j,i) = sqrt(bcc_sqr_/2);
    bcc0(m,IBY,k,j,i) = sqrt(bcc_sqr_/2);
    bcc0(m,IBZ,k,j,i) = 0.0;
  }); // end_par_for pgen_b0

  // set conserved
  if (entropy_fix_) pmbp->pmhd->EntropyReset();
  pmbp->pmhd->peos->PrimToCons(w0, bcc0, u0, is, ie, js, je, ks, ke);

  // varying conserved quantities
  par_for("pgen_u_b0_per",DevExeSpace(),0,(nmb-1),ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    // coordinates
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

    // varying density
    if (  ((x2v >= (x2_wdn_max-x2_wdn_min)*0./4 + x2_wdn_min) && (x2v < (x2_wdn_max-x2_wdn_min)*1./4 + x2_wdn_min))
       || ((x2v >= (x2_wen_max-x2_wen_min)*0./4 + x2_wen_min) && (x2v < (x2_wen_max-x2_wen_min)*1./4 + x2_wen_min))
       || ((x2v >= (x2_wv_max-x2_wv_min)  *0./4 + x2_wv_min)  && (x2v < (x2_wv_max-x2_wv_min)  *1./4 + x2_wv_min))
       || ((x2v >= (x2_bcc_max-x2_bcc_min)*0./4 + x2_bcc_min) && (x2v < (x2_bcc_max-x2_bcc_min)*1./4 + x2_bcc_min))
       ) {
      Real fac = 1.0;
      if ((dd_ptb_max!=0) && (dd_ptb_min!=0)) {
        Real x2loc_min, x2loc_max;
        if ((x2v >= x2_wdn_min) && (x2v < x2_wdn_max)) {
          x2loc_min = (x2_wdn_max-x2_wdn_min)*0./4 + x2_wdn_min;
          x2loc_max = (x2_wdn_max-x2_wdn_min)*1./4 + x2_wdn_min;
        } else if ((x2v >= x2_wen_min) && (x2v < x2_wen_max)) {
          x2loc_min = (x2_wen_max-x2_wen_min)*0./4 + x2_wen_min;
          x2loc_max = (x2_wen_max-x2_wen_min)*1./4 + x2_wen_min;
        } else if ((x2v >= x2_wv_min) && (x2v < x2_wv_max)) {
          x2loc_min = (x2_wv_max-x2_wv_min)  *0./4 + x2_wv_min;
          x2loc_max = (x2_wv_max-x2_wv_min)  *1./4 + x2_wv_min;
        } else {
          x2loc_min = (x2_bcc_max-x2_bcc_min)*0./4 + x2_bcc_min;
          x2loc_max = (x2_bcc_max-x2_bcc_min)*1./4 + x2_bcc_min;
        }
        Real lg_fac = (log10(dd_ptb_max)-log10(dd_ptb_min)) / (0.5*(x2loc_max-x2loc_min)) * (x2v-0.5*(x2loc_max+x2loc_min));
        lg_fac = fabs(lg_fac) + log10(dd_ptb_min);
        fac = pow(10., lg_fac);
        if (x2v <= 0.5*(x2loc_max+x2loc_min)) fac = 1-fac;
        else fac = 1+fac;
      }
      u0(m,IDN,k,j,i) *= fac;
    }

    // varying energy and entropy
    if (  ((x2v >= (x2_wdn_max-x2_wdn_min)*1./4 + x2_wdn_min) && (x2v < (x2_wdn_max-x2_wdn_min)*2./4 + x2_wdn_min))
       || ((x2v >= (x2_wen_max-x2_wen_min)*1./4 + x2_wen_min) && (x2v < (x2_wen_max-x2_wen_min)*2./4 + x2_wen_min))
       || ((x2v >= (x2_wv_max-x2_wv_min)  *1./4 + x2_wv_min)    && (x2v < (x2_wv_max-x2_wv_min)*2./4 + x2_wv_min))
       || ((x2v >= (x2_bcc_max-x2_bcc_min)*1./4 + x2_bcc_min) && (x2v < (x2_bcc_max-x2_bcc_min)*2./4 + x2_bcc_min))
       ) {
      Real fac = 1.0;
      if ((ee_ptb_max!=0) && (ee_ptb_min!=0)) {
        Real x2loc_min, x2loc_max;
        if ((x2v >= x2_wdn_min) && (x2v < x2_wdn_max)) {
          x2loc_min = (x2_wdn_max-x2_wdn_min)*1./4 + x2_wdn_min;
          x2loc_max = (x2_wdn_max-x2_wdn_min)*2./4 + x2_wdn_min;
        } else if ((x2v >= x2_wen_min) && (x2v < x2_wen_max)) {
          x2loc_min = (x2_wen_max-x2_wen_min)*1./4 + x2_wen_min;
          x2loc_max = (x2_wen_max-x2_wen_min)*2./4 + x2_wen_min;
        } else if ((x2v >= x2_wv_min) && (x2v < x2_wv_max)) {
          x2loc_min = (x2_wv_max-x2_wv_min)  *1./4 + x2_wv_min;
          x2loc_max = (x2_wv_max-x2_wv_min)  *2./4 + x2_wv_min;
        } else {
          x2loc_min = (x2_bcc_max-x2_bcc_min)*1./4 + x2_bcc_min;
          x2loc_max = (x2_bcc_max-x2_bcc_min)*2./4 + x2_bcc_min;
        }
        Real lg_fac = (log10(ee_ptb_max)-log10(ee_ptb_min)) / (0.5*(x2loc_max-x2loc_min)) * (x2v-0.5*(x2loc_max+x2loc_min));
        lg_fac = fabs(lg_fac) + log10(ee_ptb_min);
        fac = pow(10., lg_fac);
        if (x2v <= 0.5*(x2loc_max+x2loc_min)) fac = 1-fac;
        else fac = 1+fac;
      }
      u0(m,IEN,k,j,i) *= fac;
      if (entropy_fix_) u0(m,entropyIdx,k,j,i) *= fac;
    }

    // varying momentum
    if (  ((x2v >= (x2_wdn_max-x2_wdn_min)*2./4 + x2_wdn_min) && (x2v < (x2_wdn_max-x2_wdn_min)*3./4 + x2_wdn_min))
       || ((x2v >= (x2_wen_max-x2_wen_min)*2./4 + x2_wen_min) && (x2v < (x2_wen_max-x2_wen_min)*3./4 + x2_wen_min))
       || ((x2v >= (x2_wv_max-x2_wv_min)  *2./4 + x2_wv_min)    && (x2v < (x2_wv_max-x2_wv_min)*3./4 + x2_wv_min))
       || ((x2v >= (x2_bcc_max-x2_bcc_min)*2./4 + x2_bcc_min) && (x2v < (x2_bcc_max-x2_bcc_min)*3./4 + x2_bcc_min))
       ) {
      Real fac = 1.0;
      if ((mm_ptb_max!=0) && (mm_ptb_min!=0)) {
        Real x2loc_min, x2loc_max;
        if ((x2v >= x2_wdn_min) && (x2v < x2_wdn_max)) {
          x2loc_min = (x2_wdn_max-x2_wdn_min)*2./4 + x2_wdn_min;
          x2loc_max = (x2_wdn_max-x2_wdn_min)*3./4 + x2_wdn_min;
        } else if ((x2v >= x2_wen_min) && (x2v < x2_wen_max)) {
          x2loc_min = (x2_wen_max-x2_wen_min)*2./4 + x2_wen_min;
          x2loc_max = (x2_wen_max-x2_wen_min)*3./4 + x2_wen_min;
        } else if ((x2v >= x2_wv_min) && (x2v < x2_wv_max)) {
          x2loc_min = (x2_wv_max-x2_wv_min)  *2./4 + x2_wv_min;
          x2loc_max = (x2_wv_max-x2_wv_min)  *3./4 + x2_wv_min;
        } else {
          x2loc_min = (x2_bcc_max-x2_bcc_min)*2./4 + x2_bcc_min;
          x2loc_max = (x2_bcc_max-x2_bcc_min)*3./4 + x2_bcc_min;
        }
        Real lg_fac = (log10(mm_ptb_max)-log10(mm_ptb_min)) / (0.5*(x2loc_max-x2loc_min)) * (x2v-0.5*(x2loc_max+x2loc_min));
        lg_fac = fabs(lg_fac) + log10(mm_ptb_min);
        fac = pow(10., lg_fac);
        if (x2v <= 0.5*(x2loc_max+x2loc_min)) fac = 1-fac;
        else fac = 1+fac;
      }
      u0(m,IM1,k,j,i) *= fac;
    }

    // varying magnetic field
    if (  ((x2v >= (x2_wdn_max-x2_wdn_min)*3./4 + x2_wdn_min) && (x2v < (x2_wdn_max-x2_wdn_min)*4./4 + x2_wdn_min))
       || ((x2v >= (x2_wen_max-x2_wen_min)*3./4 + x2_wen_min) && (x2v < (x2_wen_max-x2_wen_min)*4./4 + x2_wen_min))
       || ((x2v >= (x2_wv_max-x2_wv_min)  *3./4 + x2_wv_min)    && (x2v < (x2_wv_max-x2_wv_min)*4./4 + x2_wv_min))
       || ((x2v >= (x2_bcc_max-x2_bcc_min)*3./4 + x2_bcc_min) && (x2v < (x2_bcc_max-x2_bcc_min)*4./4 + x2_bcc_min))
       ) {
      Real fac = 1.0;
      if ((bb_ptb_max!=0) && (bb_ptb_min!=0)) {
        Real x2loc_min, x2loc_max;
        if ((x2v >= x2_wdn_min) && (x2v < x2_wdn_max)) {
          x2loc_min = (x2_wdn_max-x2_wdn_min)*3./4 + x2_wdn_min;
          x2loc_max = (x2_wdn_max-x2_wdn_min)*4./4 + x2_wdn_min;
        } else if ((x2v >= x2_wen_min) && (x2v < x2_wen_max)) {
          x2loc_min = (x2_wen_max-x2_wen_min)*3./4 + x2_wen_min;
          x2loc_max = (x2_wen_max-x2_wen_min)*4./4 + x2_wen_min;
        } else if ((x2v >= x2_wv_min) && (x2v < x2_wv_max)) {
          x2loc_min = (x2_wv_max-x2_wv_min)  *3./4 + x2_wv_min;
          x2loc_max = (x2_wv_max-x2_wv_min)  *4./4 + x2_wv_min;
        } else {
          x2loc_min = (x2_bcc_max-x2_bcc_min)*3./4 + x2_bcc_min;
          x2loc_max = (x2_bcc_max-x2_bcc_min)*4./4 + x2_bcc_min;
        }
        Real lg_fac = (log10(bb_ptb_max)-log10(bb_ptb_min)) / (0.5*(x2loc_max-x2loc_min)) * (x2v-0.5*(x2loc_max+x2loc_min));
        lg_fac = fabs(lg_fac) + log10(bb_ptb_min);
        fac = pow(10., lg_fac);
        if (x2v <= 0.5*(x2loc_max+x2loc_min)) fac = 1-fac;
        else fac = 1+fac;
      }
      bcc0(m,IBX,k,j,i) *= fac;
      // b0.x1f(m,k,j,i) *= fac;
      // if (i==ie) b0.x1f(m,k,j,i+1) *= fac;
    }
  }); // end par_for

  // variable inversion
  pmbp->pmhd->peos->ConsToPrim(u0, b0, w0, bcc0, false, is, ie, js, je, ks, ke);
  // printf("rho_ini=%e, egas_ini=%e \n", rho_ctr, rho_ctr*tgas_ctr/gm1);
  // printf("rho_inv=%e, egas_inv=%e \n", w0(0,IDN,0,100,100), w0(0,IEN,0,100,100));

  // set radiation
  if (is_rad_enabled) {
    int nang = pmbp->prad->prgeo->nangles;
    auto &i0 = pmbp->prad->i0;
    auto &nh_c_ = pmbp->prad->nh_c;
    auto &tet_c_ = pmbp->prad->tet_c;
    auto &tetcov_c_ = pmbp->prad->tetcov_c;
    auto &norm_to_tet_ = pmbp->prad->norm_to_tet;

    par_for("pgen_i0",DevExeSpace(),0,(nmb-1),ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      // go through each angle
      for (int n=0; n<nang; ++n) {
        i0(m,n,k,j,i) = 0.0;
      } // endfor n
    }); // end_par_for pgen_i0
  } // endif is_rad_enabled

  return;
}
