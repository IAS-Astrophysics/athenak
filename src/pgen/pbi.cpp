//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file pbi.cpp
//! \brief Problem generator for photon bubble instabilty.

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

static Real b_star, g_star, edd_ratio;
void PBIBoundaryCondition(Mesh *pm);

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::UserProblem()
//  \brief Problem Generator for the Rayleigh-Taylor instability test

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  // return if restart
  if (restart) return;

  // User boundary function
  user_bcs_func = PBIBoundaryCondition;

  // mesh and flags
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  const bool is_one_d = pmy_mesh_->one_d;
  const bool is_two_d = pmbp->pmesh->two_d;
  const bool is_gr_enabled = pmbp->pcoord->is_general_relativistic;
  const bool is_flatspacetime  = pmbp->pcoord->coord_data.is_minkowski;
  const bool is_mhd_enabled = (pmbp->pmhd != nullptr);
  const bool is_rad_enabled = (pmbp->prad != nullptr);
  const bool is_pbi_acc_enabled = (is_mhd_enabled) ? pmbp->pmhd->psrc->pbi_const_accel : false;
  const bool is_angular_flux_enabled = (is_rad_enabled) ? pmbp->prad->angular_fluxes : false;
  const bool output_rad_angles = pin->GetOrAddBoolean("radiation", "output_rad_angles", false);

  // flag check
  if (is_one_d) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "pbi problem generator only works in 2D/3D" << std::endl;
    exit(EXIT_FAILURE);
  }
  if (!is_mhd_enabled) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "pbi problem generator only works in MHD" << std::endl;
    exit(EXIT_FAILURE);
  }
  if ((!is_gr_enabled)
   || (!is_flatspacetime)
   || (!is_pbi_acc_enabled)
   || (is_angular_flux_enabled)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "pbi problem generator only works in flat spacetime with mock-up gravitational source terms" << std::endl;
    exit(EXIT_FAILURE);
  }

  // parameters for pbi problem
  b_star  = pin->GetReal("problem", "b_star");
  g_star  = pin->GetReal("mhd", "pbi_const_accel_val");
  edd_ratio = pin->GetReal("problem", "edd_ratio");
  int num_ini = pin->GetInteger("problem", "num_ini");
  Real amp = pin->GetReal("problem", "amp");

  // read pre-calculated initial condition
  std::string fname_ini;
  if (is_two_d) {
    fname_ini.assign("../ini/ini_condition_2d.txt");
  } else {
    fname_ini.assign("../ini/ini_condition_3d.txt");
  }
  DualArray2D<Real> data_ini; // 0->h, 1->tgas, 2->rho, 3->ircm_neg, 4->ircm_pos
  Kokkos::realloc(data_ini, 5, num_ini);

  FILE *file_ini;
  if ((file_ini = std::fopen(fname_ini.c_str(), "r")) == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Initialization file could not be found" <<std::endl;
    std::exit(EXIT_FAILURE);
  }

  for (int idx=0; idx<num_ini; ++idx) {
    std::fscanf(
      file_ini, "%lf %lf %lf %lf %lf",
      &(data_ini.h_view(0,idx)), &(data_ini.h_view(1,idx)), &(data_ini.h_view(2,idx)),
      &(data_ini.h_view(3,idx)), &(data_ini.h_view(4,idx))
    );
  }
  std::fclose(file_ini);

  data_ini.template modify<HostMemSpace>();
  data_ini.template sync<DevExeSpace>();

  // capture variables for kernel
  auto &indcs = pmy_mesh_->mb_indcs;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  // int ncells1 = indcs.nx1 + 2*(indcs.ng);
  int ncells2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*(indcs.ng)) : 1;
  int ncells3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*(indcs.ng)) : 1;
  int nmb = pmbp->nmb_thispack;
  auto &size = pmbp->pmb->mb_size;

  // set EOS data
  Real gm1 = pmbp->pmhd->peos->eos_data.gamma - 1.0;

  // cast initial condition data into each cell
  int ncell = is_two_d ? ncells2 : ncells3;
  int idx_s = is_two_d ? js : ks;
  int idx_e = is_two_d ? je : ke;
  DvceArray3D<Real> data_ic; // 0->tgas, 1->rho, 2->ircm_neg, 3->ircm_pos
  Kokkos::realloc(data_ic, nmb, 4, ncell);

  par_for("pbi_interpolate_ic",DevExeSpace(),0,(nmb-1),idx_s,idx_e,
  KOKKOS_LAMBDA(int m, int idx) {
    Real &xmin = is_two_d ? size.d_view(m).x2min : size.d_view(m).x3min;
    Real &xmax = is_two_d ? size.d_view(m).x2max : size.d_view(m).x3max;
    int nx  = is_two_d ? indcs.nx2 : indcs.nx3;
    Real xv = CellCenterX(idx-idx_s, nx, xmin, xmax);

    int index1=0; int index2=1;
    while ( (data_ini.d_view(0,index2) < xv) && (index2 < num_ini-1) ) {
      index2 += 1;
    }
    if ( (index2-index1) > 1 ) {
      index1 = index2-1;
    }

    for (int nval=0; nval<4; ++nval) {
      Real x1 = data_ini.d_view(0,index1);
      Real x2 = data_ini.d_view(0,index2);
      Real var1 = data_ini.d_view(nval+1,index1);
      Real var2 = data_ini.d_view(nval+1,index2);
      data_ic(m,nval,idx) = var1 + (xv-x1) * (var2-var1)/(x2-x1);
    }
  }); // end_par_for pbi_interpolate_ic

  // set primitives
  auto &w0 = pmbp->pmhd->w0;
  Kokkos::Random_XorShift64_Pool<> rand_pool64(pmbp->gids);
  par_for("pbi_w0",DevExeSpace(),0,(nmb-1),ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    // data_ic: 0->tgas, 1->rho, 2->ircm_neg, 3->ircm_pos
    Real tgas = is_two_d ? data_ic(m,0,j) : data_ic(m,0,k);
    Real rho  = is_two_d ? data_ic(m,1,j) : data_ic(m,1,k);

    // set primitives
    w0(m,IDN,k,j,i) = rho;
    w0(m,IEN,k,j,i) = rho*tgas/gm1;
    w0(m,IVX,k,j,i) = 0.0;
    w0(m,IVY,k,j,i) = 0.0;
    w0(m,IVZ,k,j,i) = 0.0;

    // add random perturbation
    auto rand_gen = rand_pool64.get_state(); // get random number state this thread
    w0(m,IDN,k,j,i) *= 1.0 + amp * (rand_gen.frand()-0.5)*2;
    w0(m,IEN,k,j,i) *= 1.0 + amp * (rand_gen.frand()-0.5)*2;
    w0(m,IVX,k,j,i) *= 1.0 + amp * (rand_gen.frand()-0.5)*2;
    w0(m,IVY,k,j,i) *= 1.0 + amp * (rand_gen.frand()-0.5)*2;
    w0(m,IVZ,k,j,i) *= 1.0 + amp * (rand_gen.frand()-0.5)*2;
    rand_pool64.free_state(rand_gen); // free state for use by other threads
  }); // end_par_for pbi_w0

  // set magnetic field
  Real b_star_ = b_star;
  auto &b0 = pmbp->pmhd->b0;
  auto &bcc0 = pmbp->pmhd->bcc0;
  par_for("pgen_b0",DevExeSpace(),0,(nmb-1),ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    b0.x1f(m,k,j,i) = 0.0;
    b0.x2f(m,k,j,i) = is_two_d ? b_star_ : 0.0;
    b0.x3f(m,k,j,i) = is_two_d ? 0.0 : b_star_;
    if (i==ie) b0.x1f(m,k,j,i+1) = 0.0;
    if (j==je) b0.x2f(m,k,j+1,i) = is_two_d ? b_star_ : 0.0;
    if (k==ke) b0.x3f(m,k+1,j,i) = is_two_d ? 0.0 : b_star_;
    bcc0(m,IBX,k,j,i) = 0.0;
    bcc0(m,IBY,k,j,i) = is_two_d ? b_star_ : 0.0;
    bcc0(m,IBZ,k,j,i) = is_two_d ? 0.0 : b_star_;
  }); // end_par_for pgen_b0

  // set conserved
  auto &u0 = pmbp->pmhd->u0;
  pmbp->pmhd->peos->PrimToCons(w0, bcc0, u0, is, ie, js, je, ks, ke);

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
      // read initial condition in fluid frame
      // data_ic: 0->tgas, 1->rho, 2->ircm_neg, 3->ircm_pos
      Real ircm_neg = is_two_d ? data_ic(m,2,j) : data_ic(m,2,k);
      Real ircm_pos = is_two_d ? data_ic(m,3,j) : data_ic(m,3,k);

      // convert fluid velocity from normal frame to tetrad frame
      Real uu0_ = 1.0; Real uu1_ = 0.0; Real uu2_ = 0.0; Real uu3_ = 0.0;
      Real u_tet_[4];
      u_tet_[0] = norm_to_tet_(m,0,0,k,j,i)*uu0_ + norm_to_tet_(m,0,1,k,j,i)*uu1_ + norm_to_tet_(m,0,2,k,j,i)*uu2_ + norm_to_tet_(m,0,3,k,j,i)*uu3_;
      u_tet_[1] = norm_to_tet_(m,1,0,k,j,i)*uu0_ + norm_to_tet_(m,1,1,k,j,i)*uu1_ + norm_to_tet_(m,1,2,k,j,i)*uu2_ + norm_to_tet_(m,1,3,k,j,i)*uu3_;
      u_tet_[2] = norm_to_tet_(m,2,0,k,j,i)*uu0_ + norm_to_tet_(m,2,1,k,j,i)*uu1_ + norm_to_tet_(m,2,2,k,j,i)*uu2_ + norm_to_tet_(m,2,3,k,j,i)*uu3_;
      u_tet_[3] = norm_to_tet_(m,3,0,k,j,i)*uu0_ + norm_to_tet_(m,3,1,k,j,i)*uu1_ + norm_to_tet_(m,3,2,k,j,i)*uu2_ + norm_to_tet_(m,3,3,k,j,i)*uu3_;

      // go through each angle
      Real n0_ = tet_c_(m,0,0,k,j,i);
      for (int n=0; n<nang; ++n) {
        // convert radiation direction from tetrad frame to fluid frame
        Real &n0_tet = nh_c_.d_view(n,0);
        Real &n1_tet = nh_c_.d_view(n,1);
        Real &n2_tet = nh_c_.d_view(n,2);
        Real &n3_tet = nh_c_.d_view(n,3);

        Real un_t  = u_tet_[1]*n1_tet + u_tet_[2]*n2_tet + u_tet_[3]*n3_tet;
        Real n0_cm = u_tet_[0]*n0_tet - un_t;
        // Real n1_cm = -u_tet_[1]*n0_tet + n1_tet + u_tet_[1]*un_t/(1.0+u_tet_[0]);
        Real n2_cm = -u_tet_[2]*n0_tet + n2_tet + u_tet_[2]*un_t/(1.0+u_tet_[0]);
        Real n3_cm = -u_tet_[3]*n0_tet + n3_tet + u_tet_[3]*un_t/(1.0+u_tet_[0]);

        // calculate intensity in fluid frame
        Real ii_cm = ircm_pos;
        Real n_cm  = is_two_d ? n2_cm : n3_cm;
        if (n_cm < 0.0) ii_cm = ircm_neg;

        // convert intensity from fluid frame to tetrad frame
        Real n_0_ = 0.0;
        for (int d=0; d<4; ++d) { n_0_ += tetcov_c_(m,d,0,k,j,i)*nh_c_.d_view(n,d); }
        i0(m,n,k,j,i) = n0_*n_0_*ii_cm/SQR(SQR(n0_cm));
      } // endfor n
    }); // end_par_for pgen_i0

    // output coefficients of radiation angles
    if (output_rad_angles) {
      auto &solid_angles_ = pmbp->prad->prgeo->solid_angles;
      std::string fname_rad;
      fname_rad.assign("./rad_angles.txt");
      FILE *file_rad;

      if ((file_rad = std::fopen(fname_rad.c_str(), "r")) != nullptr) {
        // The file exists -- reopen the file in append mode
        if ((file_rad = std::freopen(fname_rad.c_str(), "a", file_rad)) == nullptr) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                    << std::endl << "Radiation angle file could not be opened" <<std::endl;
          std::exit(EXIT_FAILURE);
        }
      } else {
        // The file does not exist -- open the file in write mode and add headers
        if ((file_rad = std::fopen(fname_rad.c_str(), "w")) == nullptr) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                    << std::endl << "Radiation angle file could not be opened" <<std::endl;
          std::exit(EXIT_FAILURE);
        }
        std::fprintf(file_rad, "# solid_angle     nx     ny     nz");
        std::fprintf(file_rad, "\n");
      }

      // convert fluid velocity from normal frame to tetrad frame
      // note: normal frame and tetrad frame are identical in flat spacetime
      Real uu0_ = 1.0; Real uu1_ = 0.0; Real uu2_ = 0.0; Real uu3_ = 0.0;
      Real u_tet_[4];
      u_tet_[0] = 1.0*uu0_ + 0.0*uu1_ + 0.0*uu2_ + 0.0*uu3_;
      u_tet_[1] = 0.0*uu0_ + 1.0*uu1_ + 0.0*uu2_ + 0.0*uu3_;
      u_tet_[2] = 0.0*uu0_ + 0.0*uu1_ + 1.0*uu2_ + 0.0*uu3_;
      u_tet_[3] = 0.0*uu0_ + 0.0*uu1_ + 0.0*uu2_ + 1.0*uu3_;

      // write coefficients of radiation angles
      for (int n=0; n<nang; ++n) {
        Real &n0_tet = nh_c_.h_view(n,0);
        Real &n1_tet = nh_c_.h_view(n,1);
        Real &n2_tet = nh_c_.h_view(n,2);
        Real &n3_tet = nh_c_.h_view(n,3);
        Real un_t  = u_tet_[1]*n1_tet + u_tet_[2]*n2_tet + u_tet_[3]*n3_tet;
        Real n0_cm = u_tet_[0]*n0_tet - un_t;
        Real n1_cm = -u_tet_[1]*n0_tet + n1_tet + u_tet_[1]*un_t/(1.0+u_tet_[0]);
        Real n2_cm = -u_tet_[2]*n0_tet + n2_tet + u_tet_[2]*un_t/(1.0+u_tet_[0]);
        Real n3_cm = -u_tet_[3]*n0_tet + n3_tet + u_tet_[3]*un_t/(1.0+u_tet_[0]);
        Real omega_cm = solid_angles_.h_view(n)/SQR(n0_cm);
        std::fprintf(file_rad, "  %e   %e   %e   %e \n", omega_cm, n1_cm, n2_cm, n3_cm);
      }
      std::fclose(file_rad);
    } // endif output_rad_angles
  } // endif is_rad_enabled

  return;
}


//----------------------------------------------------------------------------------------
//! \fn PBIBoundaryCondition
//  \brief Sets boundary condition for pbi

void PBIBoundaryCondition(Mesh *pm) {
  // parameters and flags
  Real b_star_    = b_star;
  Real g_star_    = g_star;
  Real edd_ratio_ = edd_ratio;
  MeshBlockPack *pmbp = pm->pmb_pack;
  const bool is_two_d = pmbp->pmesh->two_d;
  const bool is_rad_enabled = (pmbp->prad != nullptr);

  // mesh parameters
  auto &indcs = pm->mb_indcs;
  int &ng = indcs.ng;
  // int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  auto &mb_bcs = pmbp->pmb->mb_bcs;
  int ncells1 = indcs.nx1 + 2*ng;
  int ncells2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng) : 1;
  int ncells3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng) : 1;
  int nmb = pmbp->nmb_thispack;

  // mhd parameters
  DvceArray5D<Real> u0_, w0_;
  u0_ = pmbp->pmhd->u0;
  w0_ = pmbp->pmhd->w0;
  int nvar = u0_.extent_int(1);
  auto &b0 = pmbp->pmhd->b0;

  // radiation parameters
  DvceArray5D<Real> i0_; int nang;
  if (is_rad_enabled) {
    i0_ = pmbp->prad->i0;
    nang = pmbp->prad->prgeo->nangles;
  }

  // magnetic bcs
  if (is_two_d) {
    // x2-boundary
    // set x2-bcs on b0 if Meshblock face is at the edge of computational domain
    par_for("pbi_bfield_x2",DevExeSpace(),0,(nmb-1),0,(ncells3-1),0,(ncells1-1),
    KOKKOS_LAMBDA(int m, int k, int i) {
      // bottom magnetic field
      if (mb_bcs.d_view(m,BoundaryFace::inner_x2) == BoundaryFlag::user) {
        for (int j=0; j<ng; ++j) {
          b0.x1f(m,k,js-j-1,i) = 0.0;
          if (i == ncells1-1) {b0.x1f(m,k,js-j-1,i+1) = 0.0;}
          b0.x2f(m,k,js-j-1,i) = b_star_;
          b0.x3f(m,k,js-j-1,i) = 0.0;
          if (k == ncells3-1) {b0.x3f(m,k+1,js-j-1,i) = 0.0;}
        }
      } // endif bottom magnetic field

      // top magnetic field
      if (mb_bcs.d_view(m,BoundaryFace::outer_x2) == BoundaryFlag::user) {
        for (int j=0; j<ng; ++j) {
          b0.x1f(m,k,je+j+1,i) = 0.0;
          if (i == ncells1-1) {b0.x1f(m,k,je+j+1,i+1) = 0.0;}
          b0.x2f(m,k,je+j+2,i) = b_star_;
          b0.x3f(m,k,je+j+1,i) = 0.0;
          if (k == ncells3-1) {b0.x3f(m,k+1,je+j+1,i) = 0.0;}
        }
      } // endif top magnetic field
    }); // end_par_for
  } else { // 3d case
    // x3-boundary
    // set x3-bcs on b0 if Meshblock face is at the edge of computational domain
    par_for("pbi_bfield_x3", DevExeSpace(),0,(nmb-1),0,(ncells2-1),0,(ncells1-1),
    KOKKOS_LAMBDA(int m, int j, int i) {
      // bottom magnetic field
      if (mb_bcs.d_view(m,BoundaryFace::inner_x3) == BoundaryFlag::user) {
        for (int k=0; k<ng; ++k) {
          b0.x1f(m,ks-k-1,j,i) = 0.0;
          if (i == ncells1-1) {b0.x1f(m,ks-k-1,j,i+1) = 0.0;}
          b0.x2f(m,ks-k-1,j,i) = 0.0;
          if (j == ncells2-1) {b0.x2f(m,ks-k-1,j+1,i) = 0.0;}
          b0.x3f(m,ks-k-1,j,i) = b_star_;
        }
      } // endif bottom magnetic field

      // top magnetic field
      if (mb_bcs.d_view(m,BoundaryFace::outer_x3) == BoundaryFlag::user) {
        for (int k=0; k<ng; ++k) {
          b0.x1f(m,ke+k+1,j,i) = 0.0;
          if (i == ncells1-1) {b0.x1f(m,ke+k+1,j,i+1) = 0.0;}
          b0.x2f(m,ke+k+1,j,i) = 0.0;
          if (j == ncells2-1) {b0.x2f(m,ke+k+1,j+1,i) = 0.0;}
          b0.x3f(m,ke+k+2,j,i) = b_star_;
        }
      } // endif top magnetic field
    });
  } // endelse 3d case

  // fluid bcs
  auto &bcc = pmbp->pmhd->bcc0;
  if (is_two_d) {
    // ConsToPrim over all x2 ghost zones *and* at the innermost/outermost x2-active zones
    // of Meshblocks, even if Meshblock face is not at the edge of computational domain
    pmbp->pmhd->peos->ConsToPrim(u0_,b0,w0_,bcc,false,0,(ncells1-1),js-ng,js,0,(ncells3-1));
    pmbp->pmhd->peos->ConsToPrim(u0_,b0,w0_,bcc,false,0,(ncells1-1),je,je+ng,0,(ncells3-1));

    // set x2-bcs on w0 if Meshblock face is at the edge of computational domain
    par_for("pbi_w_x2",DevExeSpace(),0,(nmb-1),0,(nvar-1),0,(ncells3-1),0,(ncells1-1),
    KOKKOS_LAMBDA(int m, int n, int k, int i) {
      // bottom - reflect
      if (mb_bcs.d_view(m,BoundaryFace::inner_x2) == BoundaryFlag::user) {
        for (int j=0; j<ng; ++j) {
          if (n==(IVY)) {
            w0_(m,n,k,js-j-1,i) = -1.0 * w0_(m,n,k,js+j,i);
          } else {
            w0_(m,n,k,js-j-1,i) = w0_(m,n,k,js+j,i);
          }
        }
      } // endif bottom - reflect

      // top - outflow
      if (mb_bcs.d_view(m,BoundaryFace::outer_x2) == BoundaryFlag::user) {
        for (int j=0; j<ng; ++j) {
          if (n==(IVY)) {
            w0_(m,n,k,je+j+1,i) = fmax(0.0, w0_(m,n,k,je,i));
          } else if (n==(IVX)) {
            w0_(m,n,k,je+j+1,i) = 0.0;
          } else if (n==(IVZ)) {
            w0_(m,n,k,je+j+1,i) = 0.0;
          } else {
            w0_(m,n,k,je+j+1,i) = w0_(m,n,k,je,i);
          }
        }
      } // endif top - outflow
    }); // end_par_for

  } else { //3d case
    // ConsToPrim over all x3 ghost zones *and* at the innermost/outermost x3-active zones
    // of Meshblocks, even if Meshblock face is not at the edge of computational domain
    pmbp->pmhd->peos->ConsToPrim(u0_,b0,w0_,bcc,false,0,(ncells1-1),0,(ncells2-1),ks-ng,ks);
    pmbp->pmhd->peos->ConsToPrim(u0_,b0,w0_,bcc,false,0,(ncells1-1),0,(ncells2-1),ke,ke+ng);

    // set x3-bcs on w0 if Meshblock face is at the edge of computational domain
    par_for("pbi_w_x3",DevExeSpace(),0,(nmb-1),0,(nvar-1),0,(ncells2-1),0,(ncells1-1),
    KOKKOS_LAMBDA(int m, int n, int j, int i) {
      // bottom - reflect
      if (mb_bcs.d_view(m,BoundaryFace::inner_x3) == BoundaryFlag::user) {
        for (int k=0; k<ng; ++k) {
          if (n==(IVZ)) {
            w0_(m,n,ks-k-1,j,i) = -1.0 * w0_(m,n,ks+k,j,i);
          } else {
            w0_(m,n,ks-k-1,j,i) = w0_(m,n,ks+k,j,i);
          }
        }
      } // endif bottom - reflect

      // top - outflow
      if (mb_bcs.d_view(m,BoundaryFace::outer_x3) == BoundaryFlag::user) {
        for (int k=0; k<ng; ++k) {
          if (n==(IVZ)) {
            w0_(m,n,ke+k+1,j,i) = fmax(0.0, w0_(m,n,ke,j,i));
          } else if (n==(IVX)) {
            w0_(m,n,ke+k+1,j,i) = 0.0;
          } else if (n==(IVY)) {
            w0_(m,n,ke+k+1,j,i) = 0.0;
          } else {
            w0_(m,n,ke+k+1,j,i) = w0_(m,n,ke,j,i);
          }
        }
      } // endif top - outflow
    }); // end_par_for
  } // endelse 3d case

  // radiation bcs
  if (is_rad_enabled) {
    Real gm1 = pmbp->pmhd->peos->eos_data.gamma - 1.0;
    auto &size = pmbp->pmb->mb_size;
    auto &nh_c_ = pmbp->prad->nh_c;
    auto &tet_c_ = pmbp->prad->tet_c;
    auto &tetcov_c_ = pmbp->prad->tetcov_c;
    auto &norm_to_tet_ = pmbp->prad->norm_to_tet;
    auto &solid_angles_ = pmbp->prad->prgeo->solid_angles;

    if (is_two_d) {
      // compute angle factor for ghost cells
      DvceArray5D<Real> fac_angle;
      Kokkos::realloc(fac_angle, nmb, 2, ncells3, ng, ncells1);
      par_for("pbi_fac_angle",DevExeSpace(),0,(nmb-1),0,(ncells3-1),0,(ng-1),0,(ncells1-1),
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
        // convert fluid velocity from normal frame to tetrad frame
        Real &uu1_ = w0_(m,IVX,k,js-j-1,i);
        Real &uu2_ = w0_(m,IVY,k,js-j-1,i);
        Real &uu3_ = w0_(m,IVZ,k,js-j-1,i);
        Real uu0_  = sqrt(1.0 + SQR(uu1_) + SQR(uu2_) + SQR(uu3_));

        Real u_tet_[4];
        u_tet_[0] = norm_to_tet_(m,0,0,k,js-j-1,i)*uu0_ + norm_to_tet_(m,0,1,k,js-j-1,i)*uu1_ + norm_to_tet_(m,0,2,k,js-j-1,i)*uu2_ + norm_to_tet_(m,0,3,k,js-j-1,i)*uu3_;
        u_tet_[1] = norm_to_tet_(m,1,0,k,js-j-1,i)*uu0_ + norm_to_tet_(m,1,1,k,js-j-1,i)*uu1_ + norm_to_tet_(m,1,2,k,js-j-1,i)*uu2_ + norm_to_tet_(m,1,3,k,js-j-1,i)*uu3_;
        u_tet_[2] = norm_to_tet_(m,2,0,k,js-j-1,i)*uu0_ + norm_to_tet_(m,2,1,k,js-j-1,i)*uu1_ + norm_to_tet_(m,2,2,k,js-j-1,i)*uu2_ + norm_to_tet_(m,2,3,k,js-j-1,i)*uu3_;
        u_tet_[3] = norm_to_tet_(m,3,0,k,js-j-1,i)*uu0_ + norm_to_tet_(m,3,1,k,js-j-1,i)*uu1_ + norm_to_tet_(m,3,2,k,js-j-1,i)*uu2_ + norm_to_tet_(m,3,3,k,js-j-1,i)*uu3_;

        Real a1_pos = 0.0; Real a1_neg = 0.0;
        Real a2_pos = 0.0; Real a2_neg = 0.0;
        for (int n=0; n<nang; n++) {
          // convert radiation direction from tetrad frame to fluid frame
          Real &n0_tet = nh_c_.d_view(n,0);
          Real &n1_tet = nh_c_.d_view(n,1);
          Real &n2_tet = nh_c_.d_view(n,2);
          Real &n3_tet = nh_c_.d_view(n,3);

          Real un_t  = u_tet_[1]*n1_tet + u_tet_[2]*n2_tet + u_tet_[3]*n3_tet;
          Real n0_cm = u_tet_[0]*n0_tet - un_t;
          // Real n1_cm = -u_tet_[1]*n0_tet + n1_tet + u_tet_[1]*un_t/(1.0+u_tet_[0]);
          Real n2_cm = -u_tet_[2]*n0_tet + n2_tet + u_tet_[2]*un_t/(1.0+u_tet_[0]);
          // Real n3_cm = -u_tet_[3]*n0_tet + n3_tet + u_tet_[3]*un_t/(1.0+u_tet_[0]);
          Real omega_cm = solid_angles_.d_view(n)/SQR(n0_cm);

          if (n2_cm >= 0) {
            a1_pos += omega_cm * n2_cm;
            a2_pos += omega_cm * SQR(n2_cm);
          } else {
            a1_neg += omega_cm * n2_cm;
            a2_neg += omega_cm * SQR(n2_cm);
          }
        } // endfor n

        fac_angle(m,0,k,j,i) = 1. / (a2_pos - a1_pos/a1_neg*a2_neg);
        fac_angle(m,1,k,j,i) = 1. / (a2_neg - a1_neg/a1_pos*a2_pos);
      }); // end_par_for

      // set x2-bcs on i0 if Meshblock face is at the edge of computational domain
      par_for("pbi_rad_x2",DevExeSpace(),0,(nmb-1),0,(nang-1),0,(ncells3-1),0,(ncells1-1),
      KOKKOS_LAMBDA(int m, int n, int k, int i) {
        // bottom - hydrostatic equilibrium
        if (mb_bcs.d_view(m,BoundaryFace::inner_x2) == BoundaryFlag::user) {
          Real &x2min = size.d_view(m).x2min;
          Real &x2max = size.d_view(m).x2max;
          int nx2 = indcs.nx2;

          for (int j=0; j<ng; ++j) {
            Real x2v_2 = CellCenterX(js-j, nx2, x2min, x2max);
            Real x2v_1 = CellCenterX(js-j-1, nx2, x2min, x2max);
            Real del_y = x2v_2 - x2v_1;
            Real rho_ = w0_(m,IDN,k,js-j-1,i);

            // convert fluid velocity from normal frame to tetrad frame
            Real &uu1_ = w0_(m,IVX,k,js-j-1,i);
            Real &uu2_ = w0_(m,IVY,k,js-j-1,i);
            Real &uu3_ = w0_(m,IVZ,k,js-j-1,i);
            Real uu0_  = sqrt(1.0 + SQR(uu1_) + SQR(uu2_) + SQR(uu3_));

            Real u_tet_[4];
            u_tet_[0] = norm_to_tet_(m,0,0,k,js-j-1,i)*uu0_ + norm_to_tet_(m,0,1,k,js-j-1,i)*uu1_ + norm_to_tet_(m,0,2,k,js-j-1,i)*uu2_ + norm_to_tet_(m,0,3,k,js-j-1,i)*uu3_;
            u_tet_[1] = norm_to_tet_(m,1,0,k,js-j-1,i)*uu0_ + norm_to_tet_(m,1,1,k,js-j-1,i)*uu1_ + norm_to_tet_(m,1,2,k,js-j-1,i)*uu2_ + norm_to_tet_(m,1,3,k,js-j-1,i)*uu3_;
            u_tet_[2] = norm_to_tet_(m,2,0,k,js-j-1,i)*uu0_ + norm_to_tet_(m,2,1,k,js-j-1,i)*uu1_ + norm_to_tet_(m,2,2,k,js-j-1,i)*uu2_ + norm_to_tet_(m,2,3,k,js-j-1,i)*uu3_;
            u_tet_[3] = norm_to_tet_(m,3,0,k,js-j-1,i)*uu0_ + norm_to_tet_(m,3,1,k,js-j-1,i)*uu1_ + norm_to_tet_(m,3,2,k,js-j-1,i)*uu2_ + norm_to_tet_(m,3,3,k,js-j-1,i)*uu3_;

            // convert radiation direction from tetrad frame to fluid frame
            Real &n0_tet = nh_c_.d_view(n,0);
            Real &n1_tet = nh_c_.d_view(n,1);
            Real &n2_tet = nh_c_.d_view(n,2);
            Real &n3_tet = nh_c_.d_view(n,3);

            Real un_t  = u_tet_[1]*n1_tet + u_tet_[2]*n2_tet + u_tet_[3]*n3_tet;
            Real n0_cm = u_tet_[0]*n0_tet - un_t;
            // Real n1_cm = -u_tet_[1]*n0_tet + n1_tet + u_tet_[1]*un_t/(1.0+u_tet_[0]);
            Real n2_cm = -u_tet_[2]*n0_tet + n2_tet + u_tet_[2]*un_t/(1.0+u_tet_[0]);
            // Real n3_cm = -u_tet_[3]*n0_tet + n3_tet + u_tet_[3]*un_t/(1.0+u_tet_[0]);

            // compute angle factor
            Real fac_pos = fac_angle(m,0,k,j,i);
            Real fac_neg = fac_angle(m,1,k,j,i);
            Real fac_ = (n2_cm >= 0) ? fac_pos : fac_neg;

             // calculate intensity in fluid frame
             Real n0_ = tet_c_(m,0,0,k,js-j-1,i);
             Real n_0_ = 0.0;
             for (int d=0; d<4; ++d) { n_0_ += tetcov_c_(m,d,0,k,js-j-1,i)*nh_c_.d_view(n,d); }
             Real ii_cm_2 = i0_(m,n,k,js-j,i) * SQR(SQR(n0_cm))/(n0_*n_0_) ;
             Real ii_cm_1 = ii_cm_2 + edd_ratio_*fac_*rho_*fabs(g_star_)*del_y;
             i0_(m,n,k,js-j-1,i) = ii_cm_1 * (n0_*n_0_)/SQR(SQR(n0_cm));

          } // endfor j
        } // endif bottom - hydrostatic equilibrium

        // top - vacuum
        if (mb_bcs.d_view(m,BoundaryFace::outer_x2) == BoundaryFlag::user) {
          for (int j=0; j<ng; ++j) {
            if (nh_c_.d_view(n,2) < 0.0) {
              i0_(m,n,k,je+j+1,i) = 0.0;
            } else {
              i0_(m,n,k,je+j+1,i) = i0_(m,n,k,je,i);
            }
          }
        } // endif top - vacuum

      }); // end_par_for

    } else { // 3d case
      // compute angle factor for ghost cells
      DvceArray5D<Real> fac_angle;
      Kokkos::realloc(fac_angle, nmb, 2, ng, ncells2, ncells1);
      par_for("pbi_fac_angle",DevExeSpace(),0,(nmb-1),0,(ng-1),0,(ncells2-1),0,(ncells1-1),
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
        // convert fluid velocity from normal frame to tetrad frame
        Real &uu1_ = w0_(m,IVX,ks-k-1,j,i);
        Real &uu2_ = w0_(m,IVY,ks-k-1,j,i);
        Real &uu3_ = w0_(m,IVZ,ks-k-1,j,i);
        Real uu0_  = sqrt(1.0 + SQR(uu1_) + SQR(uu2_) + SQR(uu3_));

        Real u_tet_[4];
        u_tet_[0] = norm_to_tet_(m,0,0,ks-k-1,j,i)*uu0_ + norm_to_tet_(m,0,1,ks-k-1,j,i)*uu1_ + norm_to_tet_(m,0,2,ks-k-1,j,i)*uu2_ + norm_to_tet_(m,0,3,ks-k-1,j,i)*uu3_;
        u_tet_[1] = norm_to_tet_(m,1,0,ks-k-1,j,i)*uu0_ + norm_to_tet_(m,1,1,ks-k-1,j,i)*uu1_ + norm_to_tet_(m,1,2,ks-k-1,j,i)*uu2_ + norm_to_tet_(m,1,3,ks-k-1,j,i)*uu3_;
        u_tet_[2] = norm_to_tet_(m,2,0,ks-k-1,j,i)*uu0_ + norm_to_tet_(m,2,1,ks-k-1,j,i)*uu1_ + norm_to_tet_(m,2,2,ks-k-1,j,i)*uu2_ + norm_to_tet_(m,2,3,ks-k-1,j,i)*uu3_;
        u_tet_[3] = norm_to_tet_(m,3,0,ks-k-1,j,i)*uu0_ + norm_to_tet_(m,3,1,ks-k-1,j,i)*uu1_ + norm_to_tet_(m,3,2,ks-k-1,j,i)*uu2_ + norm_to_tet_(m,3,3,ks-k-1,j,i)*uu3_;

        Real a1_pos = 0.0; Real a1_neg = 0.0;
        Real a2_pos = 0.0; Real a2_neg = 0.0;
        for (int n=0; n<nang; n++) {
          // convert radiation direction from tetrad frame to fluid frame
          Real &n0_tet = nh_c_.d_view(n,0);
          Real &n1_tet = nh_c_.d_view(n,1);
          Real &n2_tet = nh_c_.d_view(n,2);
          Real &n3_tet = nh_c_.d_view(n,3);

          Real un_t  = u_tet_[1]*n1_tet + u_tet_[2]*n2_tet + u_tet_[3]*n3_tet;
          Real n0_cm = u_tet_[0]*n0_tet - un_t;
          // Real n1_cm = -u_tet_[1]*n0_tet + n1_tet + u_tet_[1]*un_t/(1.0+u_tet_[0]);
          // Real n2_cm = -u_tet_[2]*n0_tet + n2_tet + u_tet_[2]*un_t/(1.0+u_tet_[0]);
          Real n3_cm = -u_tet_[3]*n0_tet + n3_tet + u_tet_[3]*un_t/(1.0+u_tet_[0]);
          Real omega_cm = solid_angles_.d_view(n)/SQR(n0_cm);

          if (n3_cm >= 0) {
            a1_pos += omega_cm * n3_cm;
            a2_pos += omega_cm * SQR(n3_cm);
          } else {
            a1_neg += omega_cm * n3_cm;
            a2_neg += omega_cm * SQR(n3_cm);
          }
        } // endfor n

        fac_angle(m,0,k,j,i) = 1. / (a2_pos - a1_pos/a1_neg*a2_neg);
        fac_angle(m,1,k,j,i) = 1. / (a2_neg - a1_neg/a1_pos*a2_pos);
      }); // end_par_for

      // set x3-bcs on i0 if Meshblock face is at the edge of computational domain
      par_for("pbi_rad_x3",DevExeSpace(),0,(nmb-1),0,(nang-1),0,(ncells2-1),0,(ncells1-1),
      KOKKOS_LAMBDA(int m, int n, int j, int i) {
        // bottom - hydrostatic equilibrium
        if (mb_bcs.d_view(m,BoundaryFace::inner_x3) == BoundaryFlag::user) {
          Real &x3min = size.d_view(m).x3min;
          Real &x3max = size.d_view(m).x3max;
          int nx3 = indcs.nx3;

          for (int k=0; k<ng; ++k) {
            Real x3v_2 = CellCenterX(ks-k, nx3, x3min, x3max);
            Real x3v_1 = CellCenterX(ks-k-1, nx3, x3min, x3max);
            Real del_z = x3v_2 - x3v_1;
            Real rho_ = w0_(m,IDN,ks-k-1,j,i);

            // convert fluid velocity from normal frame to tetrad frame
            Real &uu1_ = w0_(m,IVX,ks-k-1,j,i);
            Real &uu2_ = w0_(m,IVY,ks-k-1,j,i);
            Real &uu3_ = w0_(m,IVZ,ks-k-1,j,i);
            Real uu0_  = sqrt(1.0 + SQR(uu1_) + SQR(uu2_) + SQR(uu3_));

            Real u_tet_[4];
            u_tet_[0] = norm_to_tet_(m,0,0,ks-k-1,j,i)*uu0_ + norm_to_tet_(m,0,1,ks-k-1,j,i)*uu1_ + norm_to_tet_(m,0,2,ks-k-1,j,i)*uu2_ + norm_to_tet_(m,0,3,ks-k-1,j,i)*uu3_;
            u_tet_[1] = norm_to_tet_(m,1,0,ks-k-1,j,i)*uu0_ + norm_to_tet_(m,1,1,ks-k-1,j,i)*uu1_ + norm_to_tet_(m,1,2,ks-k-1,j,i)*uu2_ + norm_to_tet_(m,1,3,ks-k-1,j,i)*uu3_;
            u_tet_[2] = norm_to_tet_(m,2,0,ks-k-1,j,i)*uu0_ + norm_to_tet_(m,2,1,ks-k-1,j,i)*uu1_ + norm_to_tet_(m,2,2,ks-k-1,j,i)*uu2_ + norm_to_tet_(m,2,3,ks-k-1,j,i)*uu3_;
            u_tet_[3] = norm_to_tet_(m,3,0,ks-k-1,j,i)*uu0_ + norm_to_tet_(m,3,1,ks-k-1,j,i)*uu1_ + norm_to_tet_(m,3,2,ks-k-1,j,i)*uu2_ + norm_to_tet_(m,3,3,ks-k-1,j,i)*uu3_;

            // convert radiation direction from tetrad frame to fluid frame
            Real &n0_tet = nh_c_.d_view(n,0);
            Real &n1_tet = nh_c_.d_view(n,1);
            Real &n2_tet = nh_c_.d_view(n,2);
            Real &n3_tet = nh_c_.d_view(n,3);

            Real un_t  = u_tet_[1]*n1_tet + u_tet_[2]*n2_tet + u_tet_[3]*n3_tet;
            Real n0_cm = u_tet_[0]*n0_tet - un_t;
            // Real n1_cm = -u_tet_[1]*n0_tet + n1_tet + u_tet_[1]*un_t/(1.0+u_tet_[0]);
            // Real n2_cm = -u_tet_[2]*n0_tet + n2_tet + u_tet_[2]*un_t/(1.0+u_tet_[0]);
            Real n3_cm = -u_tet_[3]*n0_tet + n3_tet + u_tet_[3]*un_t/(1.0+u_tet_[0]);

            // compute angle factor
            Real fac_pos = fac_angle(m,0,k,j,i);
            Real fac_neg = fac_angle(m,1,k,j,i);
            Real fac_ = (n3_cm >= 0) ? fac_pos : fac_neg;

             // calculate intensity in fluid frame
             Real n0_ = tet_c_(m,0,0,ks-k-1,j,i);
             Real n_0_ = 0.0;
             for (int d=0; d<4; ++d) { n_0_ += tetcov_c_(m,d,0,ks-k-1,j,i)*nh_c_.d_view(n,d); }
             Real ii_cm_2 = i0_(m,n,ks-k,j,i) * SQR(SQR(n0_cm))/(n0_*n_0_) ;
             Real ii_cm_1 = ii_cm_2 + edd_ratio_*fac_*rho_*fabs(g_star_)*del_z;
             i0_(m,n,ks-k-1,j,i) = ii_cm_1 * (n0_*n_0_)/SQR(SQR(n0_cm));

          } // endfor j
        } // endif bottom - hydrostatic equilibrium

        // top - vacuum
        if (mb_bcs.d_view(m,BoundaryFace::outer_x3) == BoundaryFlag::user) {
          for (int k=0; k<ng; ++k) {
            if (nh_c_.d_view(n,3) < 0.0) {
              i0_(m,n,ke+k+1,j,i) = 0.0;
            } else {
              i0_(m,n,ke+k+1,j,i) = i0_(m,n,ke,j,i);
            }
          }
        } // endif top - vacuum

      }); // end_par_for
    } // endelse 3d case

  } // endif is_rad_enabled

  // PrimToCons on ghost zones
  auto &bcc0_ = pm->pmb_pack->pmhd->bcc0;
  if (is_two_d) {
    // PrimToCons on x2 ghost zones
    pm->pmb_pack->pmhd->peos->PrimToCons(w0_,bcc0_,u0_,0,(ncells1-1),js-ng,js-1,0,(ncells3-1));
    pm->pmb_pack->pmhd->peos->PrimToCons(w0_,bcc0_,u0_,0,(ncells1-1),je+1,je+ng,0,(ncells3-1));
  } else { // 3d case
    // PrimToCons on x2 ghost zones
    pm->pmb_pack->pmhd->peos->PrimToCons(w0_,bcc0_,u0_,0,(ncells1-1),0,(ncells2-1),ks-ng,ks-1);
    pm->pmb_pack->pmhd->peos->PrimToCons(w0_,bcc0_,u0_,0,(ncells1-1),0,(ncells2-1),ke+1,ke+ng);
  } // endelse 3d case

  return;
}
