//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file rad_freq_trans.cpp
//  \brief test for frame transformation of multi-frequency radiation

// C++ headers

// Athena++ headers
#include <sys/stat.h>  // mkdir
#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "geodesic-grid/geodesic_grid.hpp"
#include "hydro/hydro.hpp"
#include "driver/driver.hpp"
#include "radiation/radiation.hpp"
#include "radiation/radiation_multi_freq.hpp"

//----------------------------------------------------------------------------------------
//! \fn void MeshBlock::UserProblem(ParameterInput *pin)
//  \brief Sets initial conditions for GR radiation relaxation test

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;

  // return if restart
  if (restart) return;

  // capture variables for kernel
  auto &indcs = pmy_mesh_->mb_indcs;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng) : 1;
  int &is = indcs.is; int &js = indcs.js; int &ks = indcs.ks;
  auto &size = pmbp->pmb->mb_size;
  auto &coord = pmbp->pcoord->coord_data;
  int nmb1  = (pmbp->nmb_thispack-1);

  // get problem parameters
  Real rho = 1.0;
  Real temp  = pin->GetOrAddReal("problem", "temp", 1.0);
  Real v1    = pin->GetOrAddReal("problem", "v1", 0.0);
  Real a_rad = pin->GetOrAddReal("radiation", "arad", 1.0);
  int  order   = pin->GetOrAddInteger("problem", "order", 0);
  int  limiter = pin->GetOrAddInteger("problem", "limiter", 0);
  Real lorz = 1.0 / sqrt(1.0-(SQR(v1)));

  // fluid and radiation variables
  auto &w0 = pmbp->phydro->w0;
  Real gm1 = pmbp->phydro->peos->eos_data.gamma - 1.0;

  int &nang = pmbp->prad->prgeo->nangles;
  int &nfrq = pmbp->prad->nfreq;
  int nfr_ang = nfrq*nang;
  int nfrq1 = nfrq-1; int nang1 = nang-1;

  auto &norm_to_tet_ = pmbp->prad->norm_to_tet;
  auto &nh_c_ = pmbp->prad->nh_c;
  auto &tet_c_ = pmbp->prad->tet_c;
  auto &tetcov_c_ = pmbp->prad->tetcov_c;

  auto &nu_tet = pmbp->prad->freq_grid;
  auto &i0 = pmbp->prad->i0;

  // set primitive variables
  par_for("pgen_rad_freq_trans_w",DevExeSpace(),0,nmb1,0,(n3-1),0,(n2-1),0,(n1-1),
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    w0(m,IDN,k,j,i) = rho;
    w0(m,IVX,k,j,i) = lorz*v1;
    w0(m,IVY,k,j,i) = 0.0;
    w0(m,IVZ,k,j,i) = 0.0;
    w0(m,IEN,k,j,i) = rho*temp/gm1;
  });

  // convert primitives to conserved
  auto &u0 = pmbp->phydro->u0;
  pmbp->phydro->peos->PrimToCons(w0, u0, 0, (n1-1), 0, (n2-1), 0, (n3-1));

  // variables for result printing
  DualArray1D<Real> nu_tet_, n0_cm_;
  DualArray2D<Real> i0_cm, i1_cm;
  Kokkos::realloc(n0_cm_, nang);
  Kokkos::realloc(nu_tet_, nfrq);
  Kokkos::realloc(i0_cm, nfrq, nang);
  Kokkos::realloc(i1_cm, nfrq, nang);

  par_for("pgen_rad_freq_trans_i0",DevExeSpace(),0,nmb1,0,(n3-1),0,(n2-1),0,(n1-1),
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    bool save = ((m==0) && (k==2) && (j==2) && (i==2));

    // compute fluid velocity in tetrad frame
    Real wvx = w0(m,IVX,k,j,i);
    Real wvy = w0(m,IVY,k,j,i);
    Real wvz = w0(m,IVZ,k,j,i);
    Real gamma = sqrt(1.0 + SQR(wvx) + SQR(wvy) + SQR(wvz));

    Real n0 = tet_c_(m,0,0,k,j,i);
    Real u_tet[4];
    u_tet[0] = (norm_to_tet_(m,0,0,k,j,i)*gamma + norm_to_tet_(m,0,1,k,j,i)*wvx +
                norm_to_tet_(m,0,2,k,j,i)*wvy   + norm_to_tet_(m,0,3,k,j,i)*wvz);
    u_tet[1] = (norm_to_tet_(m,1,0,k,j,i)*gamma + norm_to_tet_(m,1,1,k,j,i)*wvx +
                norm_to_tet_(m,1,2,k,j,i)*wvy   + norm_to_tet_(m,1,3,k,j,i)*wvz);
    u_tet[2] = (norm_to_tet_(m,2,0,k,j,i)*gamma + norm_to_tet_(m,2,1,k,j,i)*wvx +
                norm_to_tet_(m,2,2,k,j,i)*wvy   + norm_to_tet_(m,2,3,k,j,i)*wvz);
    u_tet[3] = (norm_to_tet_(m,3,0,k,j,i)*gamma + norm_to_tet_(m,3,1,k,j,i)*wvx +
                norm_to_tet_(m,3,2,k,j,i)*wvy   + norm_to_tet_(m,3,3,k,j,i)*wvz);

    // assign n0_cm and nu_tet for convenience
    if (save) {
      for (int iang=0; iang<=nang1; ++iang) {
        Real n0_cm = (u_tet[0]*nh_c_.d_view(iang,0) - u_tet[1]*nh_c_.d_view(iang,1)
                    - u_tet[2]*nh_c_.d_view(iang,2) - u_tet[3]*nh_c_.d_view(iang,3));
        n0_cm_.d_view(iang) = n0_cm;
      } // endfor iang

      for (int ifr=0; ifr<=nfrq1; ++ifr) {
        nu_tet_.d_view(ifr) = nu_tet(ifr);
      } // endfor ifr
    } // endif save

    // assign fluid-frame intensity in fluid-frame frequency bins
    for (int iang=0; iang<=nang1; ++iang) {
      // calculate direction in coordinate and fluid frames
      Real n_0 = tetcov_c_(m,0,0,k,j,i)*nh_c_.d_view(iang,0) + tetcov_c_(m,1,0,k,j,i)*nh_c_.d_view(iang,1)
               + tetcov_c_(m,2,0,k,j,i)*nh_c_.d_view(iang,2) + tetcov_c_(m,3,0,k,j,i)*nh_c_.d_view(iang,3);

      Real n0_cm = (u_tet[0]*nh_c_.d_view(iang,0) - u_tet[1]*nh_c_.d_view(iang,1)
                  - u_tet[2]*nh_c_.d_view(iang,2) - u_tet[3]*nh_c_.d_view(iang,3));

      // iterate through frequency domain
      for (int ifr=0; ifr<=nfrq1; ++ifr) {
        // assign intensity in fluid frame
        Real eps_f = (ifr < nfrq1) ? BBIntegral(0, n0_cm*nu_tet(ifr+1), temp, a_rad)
                                    : a_rad*SQR(SQR(temp));
        eps_f -= BBIntegral(0, n0_cm*nu_tet(ifr), temp, a_rad);
        eps_f = 1./(4*M_PI) * fmax(0., eps_f);
        Real &i_cm_star_f = eps_f;

        // Convert intensity in tetrad frame
        int n_ = getFreqAngIndex(ifr, iang, nang);
        i0(m,n_,k,j,i) = n0*n_0*i_cm_star_f/SQR(SQR(n0_cm));
        if (save) i0_cm.d_view(ifr,iang) = i_cm_star_f;

      } // endfor ifr
    } // endfor iang

    // map fluid-frame intensity from fluid-frame to tetrad-frame frequency
    bool update_matrix_row = false;
    ScrArray2D<Real> matrix_imap;
    for (int iang=0; iang<=nang1; ++iang) {
      // calculate direction in coordinate and fluid frames
      Real n_0 = tetcov_c_(m,0,0,k,j,i)*nh_c_.d_view(iang,0) + tetcov_c_(m,1,0,k,j,i)*nh_c_.d_view(iang,1)
               + tetcov_c_(m,2,0,k,j,i)*nh_c_.d_view(iang,2) + tetcov_c_(m,3,0,k,j,i)*nh_c_.d_view(iang,3);

      Real n0_cm = (u_tet[0]*nh_c_.d_view(iang,0) - u_tet[1]*nh_c_.d_view(iang,1)
                  - u_tet[2]*nh_c_.d_view(iang,2) - u_tet[3]*nh_c_.d_view(iang,3));

      // iterate through frequency domain
      if (save) {
        for (int ifr=0; ifr<=nfrq1; ++ifr) {
          i1_cm.d_view(ifr,iang) = MapIntensity(ifr, nu_tet, i0, m, k, j, i, iang,
                                                n0_cm, n0, n_0, a_rad, order, limiter,
                                                matrix_imap, update_matrix_row);
        } // endfor ifr
      } // endif save
    } // endfor iang
  });

  // normalize i1_cm
  for (int iang=0; iang<=nang1; ++iang) {
    Real i0_sum=0, i1_sum=0;
    for (int ifr=0; ifr<=nfrq1; ++ifr) {
      i0_sum += i0_cm.d_view(ifr,iang);
      i1_sum += i1_cm.d_view(ifr,iang);
    } // endfor ifr
    Real fac_norm = i0_sum/i1_sum;
    for (int ifr=0; ifr<=nfrq1; ++ifr) {
      i1_cm.d_view(ifr,iang) = fac_norm * i1_cm.d_view(ifr,iang);
    } // endfor ifr
  } // endfor iang


  // sync dual arrays
  n0_cm_.template modify<DevExeSpace>();
  n0_cm_.template sync<HostMemSpace>();
  nu_tet_.template modify<DevExeSpace>();
  nu_tet_.template sync<HostMemSpace>();
  i0_cm.template modify<DevExeSpace>();
  i0_cm.template sync<HostMemSpace>();
  i1_cm.template modify<DevExeSpace>();
  i1_cm.template sync<HostMemSpace>();

  // root process opens output file and writes out results
  if (global_variable::my_rank == 0) {

    for (int iang=0; iang<=nang1; ++iang) {
      Real n0_cm = n0_cm_.h_view(iang);

      // calculate effective temperature for the last frequency bin
      int ne = getFreqAngIndex(nfrq1, iang, nang);
      Real ir_cm_star_e = i0_cm.h_view(nfrq1,iang);
      Real ir_cm_e = i1_cm.h_view(nfrq1,iang);
      Real teff_star = GetEffTemperature(ir_cm_star_e, n0_cm*nu_tet_.h_view(nfrq1), a_rad);
      Real teff = GetEffTemperature(ir_cm_e, nu_tet_.h_view(nfrq1), a_rad);

      // create directory to save outputs if it does not exist
      std::string dir_name;
      // dir_name.assign(pin->GetString("job", "basename") + "_data");
      dir_name.assign("data");
      mkdir(dir_name.c_str(),0775);

      // initialize output file
      std::string fname;
      FILE *pfile;
      fname.assign("./" + dir_name);
      fname.append("/angle" + std::to_string(iang) + ".dat");

      // open the file in write mode and add headers
      if ((pfile = std::fopen(fname.c_str(), "w")) == nullptr) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "Error output file could not be opened" <<std::endl;
        std::exit(EXIT_FAILURE);
      }

      // write outputs
      std::fprintf(pfile, "# angle #%d: nhat^a = (%f, %f, %f, %f) \n", iang,
                   nh_c_.h_view(iang,0), nh_c_.h_view(iang,1),
                   nh_c_.h_view(iang,2), nh_c_.h_view(iang,3));
      std::fprintf(pfile, "# a_rad=%.16f, n0_cm=%.16f \n", a_rad, n0_cm);
      std::fprintf(pfile, "# teff_star=%.16f, teff=%.16f \n", teff_star, teff);
      std::fprintf(pfile, "# order=%d, limiter=%d \n", order, limiter);
      std::fprintf(pfile, "# nu_tet  i_cm_star  ir_cm \n");
      // } // endelse

      // iterate through frequency domain
      for (int ifr=0; ifr<=nfrq1; ++ifr) {
        Real i_cm_star_f = i0_cm.h_view(ifr,iang);
        Real ir_cm_f = i1_cm.h_view(ifr,iang);
        std::fprintf(pfile, "%.16e  %.16e  %.16e \n", nu_tet_.h_view(ifr), i_cm_star_f, ir_cm_f);
      } // endfor ifr

      // close and save output file
      std::fprintf(pfile, "\n");
      std::fclose(pfile);

    } // endfor iang
  } // endif (global_variable::my_rank == 0)

  return;
}
