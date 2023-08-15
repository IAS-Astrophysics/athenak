//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file gcwinds.cpp
//  \brief Problem generator for interacting stellar winds at Galactic Center

#include <iostream>
#include <sstream>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "pgen.hpp"

namespace {
// A structure defining the properties of each of the source 'stars'
typedef struct StarData_s{
  Real mass;      // mass of the star
  Real mdot;      // mass loss rate from solar wind (in M_solar/kyr)
  Real vwind;     // speed of solar wind (in pc/kyr)
  Real x1,x2,x3;  // position in X,Y,Z (in pc)
  Real v1,v2,v3;  // velocity in X,Y,Z
  int i,j,k;      // i,j,k of x,y,z cell in which star is located
  Real alpha;     // euler angles for ZXZ rotation
  Real beta;
  Real gamma;
  Real tau;
  Real mean_angular_motion;
  Real eccentricity;
  Real rotation_matrix[3][3];
  Real period;
  Real radius;   // effective radius of star
  Real volume;   // effective volume of star
//  RegionSize block_size;   // block size of the mesh block in which the star is located
}StarData;

  // global variables used in various functions in this file
  StarData star[30];
  int nstars;
  int ncells_per_radius;

// prototypes for functions used internally to this pgen
void ReadStarData(StarData *star, std::string fname);
void ComputeRotationMatrix(StarData *star, int istar);
void UpdateStar(StarData *star, int istar, const Real t);
void RotateOrbit(StarData *star,const int istar,const Real x1_prime,const Real x2_prime,
                 Real &x1, Real &x2, Real &x3);
void InjectWind(Mesh *pmesh);
}

//----------------------------------------------------------------------------------------
//! \fn
//! \brief Problem Generator for GC stellar winds

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  // read star data
  std::string stardata_fname;
  stardata_fname =  pin->GetString("problem","stardata_fname");
  ReadStarData(star, stardata_fname);

  if (restart) return;
  // read problem parameters from input file

  // capture variables for kernel
  auto &indcs = pmy_mesh_->mb_indcs;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  EOS_Data &eos = pmbp->phydro->peos->eos_data;
  Real gm1 = eos.gamma - 1.0;
  auto &w0 = pmbp->phydro->w0;
  int &nhydro = pmbp->phydro->nhydro;
  int &nscalars = pmbp->phydro->nscalars;

  // Set up a uniform fixed background medium
  Real da = 1.0e-9;
  Real pa = 1.0e-13;
  par_for("pgen_gcw", DevExeSpace(), 0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    w0(m,IDN,k,j,i) = da;
    w0(m,IEN,k,j,i) = pa/gm1;
    w0(m,IVX,k,j,i) = 0.0;
    w0(m,IVY,k,j,i) = 0.0;
    w0(m,IVZ,k,j,i) = 0.0;
    // add passive scalars
    for (int n=nhydro; n<(nhydro+nscalars); ++n) {
      w0(m,n,k,j,i) = 0.0;
    }
  });

  // Convert primitives to conserved
  auto &u0 = pmbp->phydro->u0;
  pmbp->phydro->peos->PrimToCons(w0, u0, is, ie, js, je, ks, ke);

  return;
}

//----------------------------------------------------------------------------------------
//! \fn
//! \brief function to read information about each star from 'fname' specified in
//! argument. The data in this file should be output by the program star_dump.c

void ReadStarData(StarData *star, std::string fname)
{
  FILE *pfile;
  if ((pfile = std::fopen(fname.c_str(),"r")) == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
      << std::endl << "Star data file '" << fname << "' could not be opened" <<std::endl;
    exit(EXIT_FAILURE);
  }

  int nstars;
  float simulation_start_time,gm_tmp;
  std::fscanf(pfile, "%i %g %g \n", &nstars, &simulation_start_time, &gm_tmp);

  // check mass of BH specified in point mass source term in input file is consistent
  // with data in star data file
  if (fabs(gm - gm_tmp)/gm > 1.0e-4){
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
      << "GM in point source term not consistent with star data file" << std::endl
      << "GM_src = " << gm << "GM_star_data_file = " << gm_tmp << std::endl;
    exit(EXIT_FAILURE);
  }

  // read data for each star
  float nx1,nx2,nx3;
  for (int n=0; n<nstars; n++) {
    fscanf(pfile,"%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %f %f %f\n",
      &star[n].mass,
      &star[n].mdot,
      &star[n].vwind,
      &star[n].x1,
      &star[n].x2,
      &star[n].x3,
      &star[n].v1,
      &star[n].v2,
      &star[n].v3,
      &star[n].alpha,
      &star[n].beta,
      &star[n].gamma,
      &star[n].tau,
      &star[n].mean_angular_motion,
      &star[n].eccentricity,
      &nx1,&nx2,&nx3);

    star[n].period = 2.0*(M_PI)/star[n].mean_angular_motion;
    ComputeRotationMatrix(star, n);

    // check for self-consistency
    Real x10 = star[n].x1;
    Real x20 = star[n].x2;
    Real x30 = star[n].x3;
    Real v10 = star[n].v1;
    Real v20 = star[n].v2;
    Real v30 = star[n].v3;
    Real r0 = sqrt( SQR(star[n].x1) + SQR(star[n].x2) + SQR(star[n].x3) );
    Real v0 = sqrt( SQR(star[n].v1) + SQR(star[n].v2) + SQR(star[n].v3) );
    UpdateStar(star, n, -simulation_start_time);
    if ( (fabs(1.0 - star[n].x1/x10) * fabs(x10/r0) > 1.0e-4) ||
         (fabs(1.0 - star[n].x2/x20) * fabs(x20/r0) > 1.0e-4) ||
         (fabs(1.0 - star[n].x3/x30) * fabs(x30/r0) > 1.0e-4) ||

         (fabs(1.0 - star[n].v1/v10) * fabs(v10/v0) > 1.0e-4) ||
         (fabs(1.0 - star[n].v2/v20) * fabs(v20/v0) > 1.0e-4) ||
         (fabs(1.0 - star[n].v3/v30) * fabs(v30/v0) > 1.0e-4) ){
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Error in test solution for Kepler orbit of star " << n
                << std::endl;
      exit(EXIT_FAILURE);
    }
    UpdateStar(star, n, 0.0);
  }

  // close file and return
  std::fclose(pfile);
  return;
}

//----------------------------------------------------------------------------------------
//! \fn
//! \brief

void ComputeRotationMatrix(StarData *star, int istar) {
  Real &alpha = star[istar].alpha;
  Real &beta = star[istar].beta;
  Real &gamma = star[istar].gamma;
  Real x_rot[3][3];
  x_rot[0][0] = 1.0;
  x_rot[0][1] = 0.0;
  x_rot[0][2] = 0.0;
  x_rot[1][0] = 0.0;
  x_rot[1][1] = std::cos(beta);
  x_rot[1][2] = -std::sin(beta);
  x_rot[2][0] = 0.0;
  x_rot[2][1] = std::sin(beta);
  x_rot[2][2] = std::cos(beta);

  Real z_rot[3][3];
  z_rot[0][0] = std::cos(alpha);
  z_rot[0][1] = -std::sin(alpha);
  z_rot[0][2] = 0.0;
  z_rot[1][0] = std::sin(alpha);
  z_rot[1][1] = std::cos(alpha);
  z_rot[1][2] = 0.0;
  z_rot[2][0] = 0.0;
  z_rot[2][1] = 0.0;
  z_rot[2][2] = 1.0;

  Real z_rot2[3][3];
  z_rot2[0][0] = std::cos(gamma);
  z_rot2[0][1] = -std::sin(gamma);
  z_rot2[0][2] = 0.0;
  z_rot2[1][0] = std::sin(gamma);
  z_rot2[1][1] = std::cos(gamma);
  z_rot2[1][2] = 0.0;
  z_rot2[2][0] = 0.0;
  z_rot2[2][1] = 0.0;
  z_rot2[2][2] = 1.0;

  Real tmp[3][3],rot[3][3];
  for (int i=0; i<3; i++) {
    for (int j=0; j<3; j++) {
      rot[i][j] = 0.0;
      tmp[i][j] = 0.0;
    }
  }

  for (int i=0; i<3; i++) {
    for (int j=0; j<3; j++) {
      for (int k=0; k<3; k++) {
        tmp[i][j] += x_rot[i][k] * z_rot[k][j] ;
      }
    }
  }
  for (int i=0; i<3; i++) {
    for (int j=0; j<3; j++) {
      for (int k=0; k<3; k++) {
        rot[i][j] += z_rot2[i][k] * tmp[k][j] ;
      }
    }
  }

  for (int i=0; i<3; i++){
    for (int j=0; j<3; j++) {
      star[istar].rotation_matrix[i][j] = rot[i][j] ;
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn UpdateStar()
//! \brief Solve Kepler's equation for a given star in the plane of the orbit and then
//! rotates solution to the lab frame

void UpdateStar(StarData *star, int istar, const Real t) {
  Real mean_anomaly = star[istar].mean_angular_motion * (t - star[istar].tau);
  Real a = std::pow(gm_/SQR(star[istar].mean_angular_motion),(1.0/3.0));
  Real b;
  if (star[istar].eccentricity < 1.0){
    b = a * sqrt(1.0 - SQR(star[istar].eccentricity));
    mean_anomaly = fmod(mean_anomaly, 2.0*(M_PI));
    if (mean_anomaly >  (M_PI)) mean_anomaly = mean_anomaly - 2.0*(M_PI);
    if (mean_anomaly < -(M_PI)) mean_anomaly = mean_anomaly + 2.0*(M_PI);
  } else {
    b = a * std::sqrt(SQR(star[istar].eccentricity) - 1.0);
  }

  //Construct the initial guess.
  Real E;
  if (star[istar].eccentricity < 1.0){
    Real sgn = 1.0;
    if (std::sin(mean_anomaly) < 0.0) sgn = -1.0;
    E = mean_anomaly + sgn*(0.85)*star[istar].eccentricity;
  } else {
    Real sgn = 1.0;
    if (std::sinh(-mean_anomaly) < 0.0) sgn = -1.0;
    E = mean_anomaly;
  }

  //Solve kepler's equation iteratively to improve the solution E.
  Real error = 1.0;
  Real max_error = 1e-6;
  int i_max = 100;

  if (star[istar].eccentricity < 1.0){
    for (int i = 0; i < i_max; i++){
      Real es = star[istar].eccentricity*std::sin(E);
      Real ec = star[istar].eccentricity*std::cos(E);
      Real f = E - es - mean_anomaly;
      error = fabs(f);
      if (error < max_error) break;
      Real df = 1.0 - ec;
      Real ddf = es;
      Real dddf = ec;
      Real d1 = -f/df;
      Real d2 = -f/(df + d1*ddf/2.0);
      Real d3 = -f/(df + d2*ddf/2.0 + d2*d2*dddf/6.0);
      E = E + d3;
    }
  } else {
    for (int i = 0; i < i_max; i++){
      Real es = star[istar].eccentricity*std::sinh(E);
      Real ec = star[istar].eccentricity*std::cosh(E);
      Real f = E - es + mean_anomaly;
      error = fabs(f);
      if (error < max_error) break;
      Real df = 1.0 - ec;
      Real ddf = -es;
      Real dddf = -ec;
      Real d1 = -f/df;
      Real d2 = -f/(df + d1*ddf/2.0);
      Real d3 = -f/(df + d2*ddf/2.0 + d2*d2*dddf/6.0);
      E = E + d3;
    }
  }

  // Quit if solution did not converge.
  if (error > max_error) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
      << "Error in orbit for star " << istar << " failed to converge" << std::endl;
    exit(EXIT_FAILURE);
  }

  Real x1_prime,x2_prime,v1_prime,v2_prime;
  if (star[istar].eccentricity < 1.0){
    x1_prime= a * (std::cos(E) - star[istar].eccentricity) ;
    x2_prime= b * std::sin(E) ;

    // Time Derivative of E
    Real Edot = star[istar].mean_angular_motion /
                (1.0 - star[istar].eccentricity * std::cos(E));

    v1_prime = - a * std::sin(E) * Edot;
    v2_prime =   b * std::cos(E) * Edot;
  } else {
    x1_prime = a * ( star[istar].eccentricity - std::cosh(E) );
    x2_prime = b * std::sinh(E);

    // Time Derivative of E
    Real Edot = -star[istar].mean_angular_motion /
                 (1.0 - star[istar].eccentricity * std::cosh(E));

    v1_prime = a * (-std::sinh(E)*Edot);
    v2_prime = b * std::cosh(E) * Edot;
  }

  Real x1,x2,x3;
  RotateOrbit(star,istar,x1_prime,x2_prime,x1,x2,x3 );

  star[istar].x1 = x1;
  star[istar].x2 = x2;
  star[istar].x3 = x3;

  Real v1,v2,v3;
  RotateOrbit(star,istar,v1_prime,v2_prime,v1, v2, v3);

  star[istar].v1 = v1;
  star[istar].v2 = v2;
  star[istar].v3 = v3;
}

//----------------------------------------------------------------------------------------
//! \fn RotateOrbit
//! \brief

void RotateOrbit(StarData *star,const int istar,const Real x1_prime,const Real x2_prime,
                 Real &x1, Real &x2, Real &x3) {
  Real x_prime[3];
  x_prime[0] = x1_prime;
  x_prime[1] = x2_prime;
  x_prime[2] = 0.0;

  x1 = 0.0; x2 = 0.0; x3 = 0.0;
  for (int j=0; j<3; j++) {
    // Note this is inverse rotation so rot[j,i] instead of rot[i,j]
    x1 += star[istar].rotation_matrix[j][0]*x_prime[j];
    x2 += star[istar].rotation_matrix[j][1]*x_prime[j];
    x3 += star[istar].rotation_matrix[j][2]*x_prime[j];
  }
}

//----------------------------------------------------------------------------------------
//! \fn InjectWind
//! \brief Adds mass, momentum, and energy over "mask" of star to drive stellar winds

void InjectWind(Mesh *pmesh) {
  // capture variables for kernel
  auto &indcs = pmesh->mb_indcs;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  auto &size = pmesh->pmb_pack->pmb->mb_size;
  auto &u0 = pmesh->pmb_pack->phydro->u0;
  int nmb1 = pmesh->pmb_pack->nmb_thispack - 1;

  par_for("inject_wind", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    // Find L/R edges of cells
    Real x1[2], x2[2], x3[2];
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    x1[0] = LeftEdgeX(i-is, indcs.nx1, x1min, x1max);
    x1[1] = x1[0] + size.d_view(m).dx1;

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    x2[0] = LeftEdgeX(j-js, indcs.nx2, x2min, x2max);
    x2[1] = x2[0] + size.d_view(m).dx2;

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    x3[0] = LeftEdgeX(k-ks, indcs.nx3, x3min, x3max);
    x3[1] = x3[0] + size.d_view(m).dx3;

    // Loop over stars
    for (int istar=0; istar < nstars; istar++) {
      Real r2 = SQR(star[istar].radius);
      Real &x0 = star[istar].x1;
      Real &y0 = star[istar].x2;
      Real &z0 = star[istar].x3;
      Real edge = 0.0;

      // Check whether spherical mask region intersects any x1-x2 edges of cell
      for (int j=0; j<2; ++j) {
        for (int i=0; i<2; ++i) {
          Real z = std::sqrt(r2 - SQR(x1[i] - x0) - SQR(x2[j] - y0));
          if (x3[1] > (z0 - z)) {
            if (x3[0] < (z0 + z)) {
              if (x3[1] > (z0 + z)) {
                edge += z - x3[0];
              } else if (x3[0] < (z0 - z)) {
                edge += x3[1] - z;
              } else {
                edge += size.d_view(m).dx3;
              }
            }
          }
        }
      }
      // Check whether spherical mask region intersects any x1-x3 edges of cell
      for (int k=0; k<2; ++k) {
        for (int i=0; i<2; ++i) {
          Real y = std::sqrt(r2 - SQR(x1[i] - x0) - SQR(x3[k] - z0));
          if (x2[1] > (y0 - y)) {
            if (x2[0] < (y0 + y)) {
              if (x2[1] > (y0 + y)) {
                edge += y - x2[0];
              } else if (x2[0] < (y0 - y)) {
                edge += x2[1] - y;
              } else {
                edge += size.d_view(m).dx2;
              }
            }
          }
        }
      }
      // Check whether spherical mask region intersects any x2-x3 edges of cell
      for (int k=0; k<2; ++k) {
        for (int j=0; j<2; ++j) {
          Real x = std::sqrt(r2 - SQR(x2[j] - y0) - SQR(x3[k] - z0));
          if (x1[1] > (x0 - x)) {
            if (x1[0] < (x0 + x)) {
              if (x1[1] > (x0 + x)) {
                edge += x - x1[0];
              } else if (x1[0] < (x0 - x)) {
                edge += x1[1] - x;
              } else {
                edge += size.d_view(m).dx1;
              }
            }
          }
        }
      }

      if (edge > 0.0) {
        Real volume_frac =
          (edge*(size.d_view(m).dx1)*(size.d_view(m).dx2)*(size.d_view(m).dx3))/
          (4.0*(size.d_view(m).dx1 + size.d_view(m).dx2 + size.d_view(m).dx3));

        Real d0 = volume_frac * (star[istar].mdot/star[istar].volume)*(pmesh->dt);
        Real dx1 = (x1[0] + 0.5*size.d_view(m).dx1) - star[istar].x1;
        Real dx2 = (x2[0] + 0.5*size.d_view(m).dx2) - star[istar].x2;
        Real dx3 = (x3[0] + 0.5*size.d_view(m).dx3) - star[istar].x3;
        Real r0 = sqrt(dx1*dx1 + dx2*dx2 + dx3*dx3);
        Real wind_v1 = ((dx1/r0)*star[istar].vwind) + star[istar].v1;
        Real wind_v2 = ((dx2/r0)*star[istar].vwind) + star[istar].v2;
        Real wind_v3 = ((dx3/r0)*star[istar].vwind) + star[istar].v3;

        Real d_old = u0(m,IDN,k,j,i);
        Real vsq_old = (SQR(u0(m,IM1,k,j,i))+SQR(u0(m,IM2,k,j,i))+SQR(u0(m,IM3,k,j,i)))/
                        u0(m,IDN,k,j,i) ;

        u0(m,IDN,k,j,i) += d0 ;
        u0(m,IM1,k,j,i) += d0 * wind_v1;
        u0(m,IM2,k,j,i) += d0 * wind_v2;
        u0(m,IM3,k,j,i) += d0 * wind_v3;

        Real d_new = u0(m,IDN,k,j,i);
        Real vsq_new = (SQR(u0(m,IM1,k,j,i))+SQR(u0(m,IM2,k,j,i))+SQR(u0(m,IM3,k,j,i)))/
                        u0(m,IDN,k,j,i) ;

        // Add thermal energy to keep T const, and add KE
        u0(m,IEN,k,j,i) = d_new*(u0(m,IEN,k,j,i)/d_old + 0.5*(vsq_new - vsq_old));
      }
    } // end loop over stars
  });

  return;
}
