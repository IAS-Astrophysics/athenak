//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file geodesic_grid.cpp
//  \brief implements constructor and some fns for GeodesicGrid class

// C/C++ headers
#include <float.h>
#include <iostream>

// AthenaK headers
#include "athena.hpp"
#include "geodesic_grid.hpp"

//----------------------------------------------------------------------------------------
// constructor, initializes data structures and parameters

GeodesicGrid::GeodesicGrid(int nlev, bool rotate, bool fluxes) :
    nlevel(nlev),
    rotate_geo(rotate),
    geo_fluxes(fluxes),
    amesh_normals("amesh_normals",1,1,1,1),
    ameshp_normals("ameshp_normals",1,1),
    amesh_indices("amesh_indices",1,1,1),
    ameshp_indices("ameshp_indices",1),
    num_neighbors("num_neighbors",1),
    ind_neighbors("ind_neighbors",1,1),
    ind_neighbors_edges("ind_neighbors_edges",1,1),
    solid_angles("solid_angles",1),
    arc_lengths("arc_lengths",1,1),
    cart_pos("cart_pos",1,1),
    cart_pos_mid("cart_pos_mid",1,1,1),
    polar_pos("polar_pos",1,1),
    polar_pos_mid("polar_pos_mid",1,1,1),
    unit_flux("unit_flux",1,1,1) {
  if (nlevel > 0) {  // construct geodesic mesh
    // number of angles
    nangles = 5*2*SQR(nlevel) + 2;

    // reallocate geodesic mesh arrays
    Kokkos::realloc(amesh_normals,5,2+nlevel,2+2*nlevel,3);
    Kokkos::realloc(ameshp_normals,2,3);
    Kokkos::realloc(amesh_indices,5,2+nlevel,2+2*nlevel);
    Kokkos::realloc(ameshp_indices,2);
    Kokkos::realloc(num_neighbors,nangles);
    Kokkos::realloc(ind_neighbors,nangles,6);
    Kokkos::realloc(ind_neighbors_edges,nangles,6);
    Kokkos::realloc(solid_angles,nangles);
    Kokkos::realloc(arc_lengths,nangles,6);
    Kokkos::realloc(cart_pos,nangles,3);
    Kokkos::realloc(cart_pos_mid,nangles,6,3);
    Kokkos::realloc(polar_pos,nangles,2);
    Kokkos::realloc(polar_pos_mid,nangles,6,2);

    // construction parameters
    Real sin_ang = 2.0/sqrt(5.0);
    Real cos_ang = 1.0/sqrt(5.0);
    Real p1[3] = {0.0, 0.0, 1.0};
    Real p2[3] = {sin_ang, 0.0, cos_ang};
    Real p3[3] = {sin_ang*cos( 0.2*M_PI), sin_ang*sin( 0.2*M_PI), -cos_ang};
    Real p4[3] = {sin_ang*cos(-0.4*M_PI), sin_ang*sin(-0.4*M_PI),  cos_ang};
    Real p5[3] = {sin_ang*cos(-0.2*M_PI), sin_ang*sin(-0.2*M_PI), -cos_ang};
    Real p6[3] = {0.0, 0.0, -1.0};

    // set pole normal components explicitly
    auto &apnorm = ameshp_normals;
    apnorm(0,0) = 0.0;
    apnorm(0,1) = 0.0;
    apnorm(0,2) = 1.0;
    apnorm(1,0) = 0.0;
    apnorm(1,1) = 0.0;
    apnorm(1,2) = -1.0;

    // get normal components of all other angle centers
    // start by filling in one of the five blocks
    auto &anorm = amesh_normals;
    int row_index = 1;
    for (int l=0; l<nlevel; ++l) {
      int col_index = 1;
      for (int m=l; m<nlevel; ++m) {
        Real x = ((m-l+1)*p2[0] + (nlevel-m-1)*p1[0] + l*p4[0])/(Real)(nlevel);
        Real y = ((m-l+1)*p2[1] + (nlevel-m-1)*p1[1] + l*p4[1])/(Real)(nlevel);
        Real z = ((m-l+1)*p2[2] + (nlevel-m-1)*p1[2] + l*p4[2])/(Real)(nlevel);
        Real norm = sqrt(SQR(x) + SQR(y) + SQR(z));
        anorm(0,row_index,col_index,0) = x/norm;
        anorm(0,row_index,col_index,1) = y/norm;
        anorm(0,row_index,col_index,2) = z/norm;
        col_index += 1;
      }
      for (int m=nlevel-l; m<nlevel; ++m) {
        Real x =((nlevel-l)*p2[0]+(m-nlevel+l+1)*p5[0]+(nlevel-m-1)*p4[0])/(Real)(nlevel);
        Real y =((nlevel-l)*p2[1]+(m-nlevel+l+1)*p5[1]+(nlevel-m-1)*p4[1])/(Real)(nlevel);
        Real z =((nlevel-l)*p2[2]+(m-nlevel+l+1)*p5[2]+(nlevel-m-1)*p4[2])/(Real)(nlevel);
        Real norm = sqrt(SQR(x) + SQR(y) + SQR(z));
        anorm(0,row_index,col_index,0) = x/norm;
        anorm(0,row_index,col_index,1) = y/norm;
        anorm(0,row_index,col_index,2) = z/norm;
        col_index += 1;
      }
      for (int m=l; m<nlevel; ++m) {
        Real x = ((m-l+1)*p3[0] + (nlevel-m-1)*p2[0] + l*p5[0])/(Real)(nlevel);
        Real y = ((m-l+1)*p3[1] + (nlevel-m-1)*p2[1] + l*p5[1])/(Real)(nlevel);
        Real z = ((m-l+1)*p3[2] + (nlevel-m-1)*p2[2] + l*p5[2])/(Real)(nlevel);
        Real norm = sqrt(SQR(x) + SQR(y) + SQR(z));
        anorm(0,row_index,col_index,0) = x/norm;
        anorm(0,row_index,col_index,1) = y/norm;
        anorm(0,row_index,col_index,2) = z/norm;
        col_index += 1;
      }
      for (int m=nlevel-l; m<nlevel; ++m) {
        Real x =((nlevel-l)*p3[0]+(m-nlevel+l+1)*p6[0]+(nlevel-m-1)*p5[0])/(Real)(nlevel);
        Real y =((nlevel-l)*p3[1]+(m-nlevel+l+1)*p6[1]+(nlevel-m-1)*p5[1])/(Real)(nlevel);
        Real z =((nlevel-l)*p3[2]+(m-nlevel+l+1)*p6[2]+(nlevel-m-1)*p5[2])/(Real)(nlevel);
        Real norm = sqrt(SQR(x) + SQR(y) + SQR(z));
        anorm(0,row_index,col_index,0) = x/norm;
        anorm(0,row_index,col_index,1) = y/norm;
        anorm(0,row_index,col_index,2) = z/norm;
        col_index += 1;
      }
      row_index += 1;
    }

    // fill the other four patches by rotating the first one
    for (int ptch=1; ptch<5; ++ptch) {
      for (int l=1; l<1+nlevel; ++l) {
        for (int m=1; m<1+2*nlevel; ++m) {
          Real x0 = anorm(0,l,m,0);
          Real y0 = anorm(0,l,m,1);
          Real z0 = anorm(0,l,m,2);
          anorm(ptch,l,m,0) = (x0*cos(ptch*0.4*M_PI)+y0*sin(ptch*0.4*M_PI));
          anorm(ptch,l,m,1) = (y0*cos(ptch*0.4*M_PI)-x0*sin(ptch*0.4*M_PI));
          anorm(ptch,l,m,2) = z0;
        }
      }
    }

    // fill in the ghost cells of all blocks
    for (int i=0; i<3; ++i) {
      for (int bl=0; bl<5; ++bl) {
        for (int k=0; k<nlevel; ++k) {
          anorm(bl,0,k+1,i)          = anorm((bl+4)%5,k+1,1,i);
          anorm(bl,0,k+nlevel+1,i)   = anorm((bl+4)%5,nlevel,k+1,i);
          anorm(bl,k+1,2*nlevel+1,i) = anorm((bl+4)%5,nlevel,k+nlevel+1,i);
          anorm(bl,k+2,0,i)          = anorm((bl+1)%5,1,k+1,i);
          anorm(bl,nlevel+1,k+1,i)   = anorm((bl+1)%5,1,k+nlevel+1,i);
          anorm(bl,nlevel+1,k+nlevel+1,i) = anorm((bl+1)%5,k+2,2*nlevel,i);
        }
        anorm(bl,1,0,i) = apnorm(0,i);
        anorm(bl,nlevel+1,2*nlevel,i) = apnorm(1,i);
        anorm(bl,0,2*nlevel+1,i) = anorm(bl,0,2*nlevel,i);
      }
    }

    // generate 2d to 1d map
    auto &apind = ameshp_indices;
    auto &aind = amesh_indices;
    apind(0) = 5*2*SQR(nlevel);
    apind(1) = 5*2*SQR(nlevel) + 1;
    for (int ptch=0; ptch<5; ++ptch) {
      for (int l=0; l<nlevel; ++l) {
        for (int m=0; m<2*nlevel; ++m) {
          aind(ptch,l+1,m+1) = ptch*2*SQR(nlevel) + l*2*nlevel + m;
        }
      }
    }

    // fill ghost cells
    for (int bl=0; bl<5; ++bl) {
      for (int k=0; k<nlevel; ++k) {
        aind(bl,0,k+1)               = aind((bl+4)%5,k+1,1);
        aind(bl,0,k+nlevel+1)        = aind((bl+4)%5,nlevel,k+1);
        aind(bl,k+1,2*nlevel+1)      = aind((bl+4)%5,nlevel,k+nlevel+1);
        aind(bl,k+2,0)               = aind((bl+1)%5,1,k+1);
        aind(bl,nlevel+1,k+1)        = aind((bl+1)%5,1,k+nlevel+1);
        aind(bl,nlevel+1,k+nlevel+1) = aind((bl+1)%5,k+2,2*nlevel);
      }
      aind(bl,1,0) = apind(0);
      aind(bl,nlevel+1,2*nlevel) = apind(1);
      aind(bl,0,2*nlevel+1) = aind(bl,0,2*nlevel);
    }

    // set up arrays for neighbors/neighbor indexing, solid angles, and arc lengths
    auto &numn = num_neighbors;
    auto &indn = ind_neighbors;
    auto &arcl = arc_lengths;
    for (int n=0; n<nangles; ++n) {
      // find the number of neighbors and indices of neighbors
      int num_nghbr; int neighbors[6];
      Neighbors(n,num_nghbr,neighbors);

      // find the solid angle and arc (edge) lengths
      Real omega; Real arcs[6];
      SolidAngleAndArcLengths(n,omega,arcs);

      // store in corresponding arrays
      numn.h_view(n) = num_nghbr;
      solid_angles.h_view(n) = omega;
      for (int nb=0; nb<6; ++nb) {
        indn.h_view(n,nb) = neighbors[nb];
        arcl.h_view(n,nb) = arcs[nb];
      }
    }

    // set up arrays for neighbor edge indexing
    auto &indne = ind_neighbors_edges;
    for (int n=0; n<nangles; ++n) {
      int nn = numn.h_view(n);
      for (int nb=0; nb<nn; ++nb) {
        for (int nnb=0; nnb<numn.h_view(indn.h_view(n,nb)); ++nnb) {
          if (n==indn.h_view(indn.h_view(n,nb),nnb)) {
            indne.h_view(n,nb) = nnb;
          }
        }
      }
      if (nn==5) {
        indne.h_view(n,5) = (INT_MAX);
      }
    }

    // correct for round-off error level diff in arc lengths among shared edges
    for (int n=0; n<nangles; ++n) {
      for (int nb=0; nb<numn.h_view(n); ++nb) {
        Real arc_avg = 0.5*(arcl.h_view(n,nb) +
                            arcl.h_view(indn.h_view(n,nb),indne.h_view(n,nb)));
        arcl.h_view(n,nb) = arc_avg;
        arcl.h_view(indn.h_view(n,nb),indne.h_view(n,nb)) = arc_avg;
      }
    }

    // rotate geodesic mesh
    if (rotate_geo) {
      Real rotangles[2];
      OptimalAngles(rotangles);
      RotateGrid(rotangles[0],rotangles[1]);
    }

    // set grid positions
    for (int n=0; n<nangles; ++n) {
      Real x, y, z;
      GridCartPosition(n,x,y,z);
      cart_pos.h_view(n,0) = x;
      cart_pos.h_view(n,1) = y;
      cart_pos.h_view(n,2) = z;
      int nn = num_neighbors.h_view(n);
      for (int nb=0; nb<nn; ++nb) {
        Real xm, ym, zm;
        GridCartPositionMid(n,ind_neighbors.h_view(n,nb),xm,ym,zm);
        cart_pos_mid.h_view(n,nb,0) = xm;
        cart_pos_mid.h_view(n,nb,1) = ym;
        cart_pos_mid.h_view(n,nb,2) = zm;
      }
      if (nn==5) {
        cart_pos_mid.h_view(n,5,0) = (FLT_MAX);
        cart_pos_mid.h_view(n,5,1) = (FLT_MAX);
        cart_pos_mid.h_view(n,5,2) = (FLT_MAX);
      }
    }

    // set polar coordinate positions
    for (int n=0; n<nangles; ++n) {
      polar_pos.h_view(n,0) = acos(cart_pos.h_view(n,2));
      polar_pos.h_view(n,1) = atan2(cart_pos.h_view(n,1), cart_pos.h_view(n,0));
      int nn = num_neighbors.h_view(n);
      for (int nb=0; nb<nn; ++nb) {
        polar_pos_mid.h_view(n,nb,0) = acos(cart_pos_mid.h_view(n,nb,2));
        polar_pos_mid.h_view(n,nb,1) = atan2(cart_pos_mid.h_view(n,nb,1),
                                             cart_pos_mid.h_view(n,nb,0));
      }
      if (nn==5) {
        polar_pos_mid.h_view(n,5,0) = (FLT_MAX);
        polar_pos_mid.h_view(n,5,1) = (FLT_MAX);
      }
    }

    // set angular unit vectors along edges of angle faces
    if (geo_fluxes) {
      // reallocate unit flux array
      Kokkos::realloc(unit_flux, nangles, 6, 2);
      // set unit flux
      for (int n=0; n<nangles; ++n) {
        Real x, y, z;
        GridCartPosition(n,x,y,z);
        Real zetav = acos(z);
        Real psiv  = atan2(y,x);
        for (int nb=0; nb<num_neighbors.h_view(n); ++nb) {
          Real xm, ym, zm;
          GridCartPositionMid(n,ind_neighbors.h_view(n,nb),xm,ym,zm);
          Real zetaf = acos(zm);
          Real psif  = atan2(ym,xm);
          Real unit_zeta, unit_psi;
          UnitFluxDir(zetav,psiv,zetaf,psif,unit_zeta,unit_psi);
          unit_flux.h_view(n,nb,0) = unit_zeta;
          unit_flux.h_view(n,nb,1) = unit_psi;
        }
      }
      // correct for round-off error level diff in unit vectors among shared edges
      for (int n=0; n<nangles; ++n) {
        for (int nb=0; nb<numn.h_view(n); ++nb) {
          Real tuzeta = unit_flux.h_view(n,nb,0);
          Real tupsi  = unit_flux.h_view(n,nb,1);
          Real nuzeta = unit_flux.h_view(indn.h_view(n,nb),indne.h_view(n,nb),0);
          Real nupsi  = unit_flux.h_view(indn.h_view(n,nb),indne.h_view(n,nb),1);
          Real uzeta_avg = 0.5*(fabs(tuzeta) + fabs(nuzeta));
          Real upsi_avg  = 0.5*(fabs(tupsi ) + fabs(nupsi ));
          unit_flux.h_view(n,nb,0) = copysign(uzeta_avg, tuzeta);
          unit_flux.h_view(n,nb,1) = copysign(upsi_avg, tupsi);
          unit_flux.h_view(indn.h_view(n,nb),indne.h_view(n,nb),0) = copysign(uzeta_avg,
                                                                              nuzeta);
          unit_flux.h_view(indn.h_view(n,nb),indne.h_view(n,nb),1) = copysign(upsi_avg,
                                                                              nupsi);
        }
      }
    }

    // sync dual arrays
    num_neighbors.template modify<HostMemSpace>();
    num_neighbors.template sync<DevExeSpace>();
    ind_neighbors.template modify<HostMemSpace>();
    ind_neighbors.template sync<DevExeSpace>();
    ind_neighbors_edges.template modify<HostMemSpace>();
    ind_neighbors_edges.template sync<DevExeSpace>();
    arc_lengths.template modify<HostMemSpace>();
    arc_lengths.template sync<DevExeSpace>();
    solid_angles.template modify<HostMemSpace>();
    solid_angles.template sync<DevExeSpace>();
    cart_pos.template modify<HostMemSpace>();
    cart_pos.template sync<DevExeSpace>();
    cart_pos_mid.template modify<HostMemSpace>();
    cart_pos_mid.template sync<DevExeSpace>();
    polar_pos.template modify<HostMemSpace>();
    polar_pos.template sync<DevExeSpace>();
    polar_pos_mid.template modify<HostMemSpace>();
    polar_pos_mid.template sync<DevExeSpace>();
    if (geo_fluxes) {
      unit_flux.template modify<HostMemSpace>();
      unit_flux.template sync<DevExeSpace>();
    }

  } else if (nlevel==0) {  // one angle per octant
    // throw warning---this should only ever be used for testing
    std::cout << "### WARNING! in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "nlevel=0 initialization should only be used for testing" << std::endl;

    // throw error if fluxes enabled or rotating mesh
    if (rotate_geo || geo_fluxes) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
        << std::endl << "nlevel=0 incompatible with fluxes and rotations" << std::endl;
      std::exit(EXIT_FAILURE);
    }

    // number of angles
    nangles = 8;

    // reallocate geodesic mesh arrays
    Kokkos::realloc(solid_angles, nangles);
    Kokkos::realloc(cart_pos, nangles, 3);

    // set solid angles and cartesian positions
    Real zetav[2] = {M_PI/4.0, 3.0*M_PI/4.0};
    Real psiv[4] = {M_PI/4.0, 3.0*M_PI/4.0, 5.0*M_PI/4.0, 7.0*M_PI/4.0};
    for (int z=0, n=0; z<2; ++z) {
      for (int p=0; p<4; ++p, ++n) {
        solid_angles.h_view(n) = 4.0*M_PI/nangles;
        cart_pos.h_view(n,0) = sin(zetav[z])*cos(psiv[p])*sqrt(4.0/3.0);
        cart_pos.h_view(n,1) = sin(zetav[z])*sin(psiv[p])*sqrt(4.0/3.0);
        cart_pos.h_view(n,2) = cos(zetav[z])*sqrt(2.0/3.0);
      }
    }

    // sync dual arrays
    solid_angles.template modify<HostMemSpace>();
    solid_angles.template sync<DevExeSpace>();
    cart_pos.template modify<HostMemSpace>();
    cart_pos.template sync<DevExeSpace>();

  } else {  // invalid nlevel
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
        << std::endl << "nlevel must be >= 0, but nlevel=" << nlevel << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

//----------------------------------------------------------------------------------------
//! \brief GeodesicGrid destructor

GeodesicGrid::~GeodesicGrid() {
}

//----------------------------------------------------------------------------------------
//! \fn void GeodesicGrid::GridCartPosition
//! \brief find position at face center

void GeodesicGrid::GridCartPosition(int n, Real& x, Real& y, Real& z) {
  int ibl0 = (n / (2*nlevel*nlevel));
  int ibl1 = (n % (2*nlevel*nlevel)) / (2*nlevel);
  int ibl2 = (n % (2*nlevel*nlevel)) % (2*nlevel);
  if (ibl0 == 5) {
    x = ameshp_normals(ibl2, 0);
    y = ameshp_normals(ibl2, 1);
    z = ameshp_normals(ibl2, 2);
  } else {
    x = amesh_normals(ibl0,ibl1+1,ibl2+1,0);
    y = amesh_normals(ibl0,ibl1+1,ibl2+1,1);
    z = amesh_normals(ibl0,ibl1+1,ibl2+1,2);
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void GeodesicGrid::GridCartPositionMid
//! \brief find mid position between two face centers

void GeodesicGrid::GridCartPositionMid(int n, int nb, Real& x, Real& y, Real& z) {
  Real x1, y1, z1, x2, y2, z2;
  GridCartPosition(n,x1,y1,z1);
  GridCartPosition(nb,x2,y2,z2);
  Real xm = 0.5*(x1+x2);
  Real ym = 0.5*(y1+y2);
  Real zm = 0.5*(z1+z2);
  Real norm = sqrt(SQR(xm)+SQR(ym)+SQR(zm));
  x = xm/norm;
  y = ym/norm;
  z = zm/norm;
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void GeodesicGrid::Neighbors
//! \brief retrieve number of neighbors and indexing of neighbors

void GeodesicGrid::Neighbors(int n, int& num_nghbr, int neighbors[6]) {
  if (n==5*2*nlevel*nlevel) {  // handle north pole
    for (int bl=0; bl<5; ++bl) {
      neighbors[bl] = amesh_indices(bl,1,1);
    }
    neighbors[5] = (INT_MAX);
    num_nghbr = 5;
  } else if (n==5*2*nlevel*nlevel + 1) {  // handle south pole
    for (int bl=0; bl<5; ++bl) {
      neighbors[bl] = amesh_indices(bl,nlevel,2*nlevel);
    }
    neighbors[5] = (INT_MAX);
    num_nghbr = 5;
  } else {
    int ibl0 = (n / (2*nlevel*nlevel));
    int ibl1 = (n % (2*nlevel*nlevel)) / (2*nlevel);
    int ibl2 = (n % (2*nlevel*nlevel)) % (2*nlevel);
    neighbors[0] = amesh_indices(ibl0,ibl1+1,ibl2+2);
    neighbors[1] = amesh_indices(ibl0,ibl1+2,ibl2+1);
    neighbors[2] = amesh_indices(ibl0,ibl1+2,ibl2  );
    neighbors[3] = amesh_indices(ibl0,ibl1+1,ibl2  );
    neighbors[4] = amesh_indices(ibl0,ibl1  ,ibl2+1);

    if (n % (2*nlevel*nlevel) == nlevel-1 ||
        n % (2*nlevel*nlevel) == 2*nlevel-1) {
      neighbors[5] = (INT_MAX);
      num_nghbr = 5;
    } else {
      neighbors[5] = amesh_indices(ibl0, ibl1, ibl2+2);
      num_nghbr = 6;
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void GeodesicGrid::CircumcenterNormalized
//! \brief find circumcenter of face

void GeodesicGrid::CircumcenterNormalized(Real x1, Real x2, Real x3,
                                          Real y1, Real y2, Real y3,
                                          Real z1, Real z2, Real z3,
                                          Real& x, Real& y, Real& z) {
  Real a = sqrt(SQR(x3-x2)+SQR(y3-y2)+SQR(z3-z2));
  Real b = sqrt(SQR(x1-x3)+SQR(y1-y3)+SQR(z1-z3));
  Real c = sqrt(SQR(x2-x1)+SQR(y2-y1)+SQR(z2-z1));
  Real denom = 1.0/((a+c+b)*(a+c-b)*(a+b-c)*(b+c-a));
  Real x_c = (x1*(SQR(a)*(SQR(b)+SQR(c)-SQR(a)))+
              x2*(SQR(b)*(SQR(c)+SQR(a)-SQR(b))) +
              x3*(SQR(c)*(SQR(a)+SQR(b)-SQR(c))))*denom;
  Real y_c = (y1*(SQR(a)*(SQR(b)+SQR(c)-SQR(a)))+
              y2*(SQR(b)*(SQR(c)+SQR(a)-SQR(b))) +
              y3*(SQR(c)*(SQR(a)+SQR(b)-SQR(c))))*denom;
  Real z_c = (z1*(SQR(a)*(SQR(b)+SQR(c)-SQR(a)))+
              z2*(SQR(b)*(SQR(c)+SQR(a)-SQR(b))) +
              z3*(SQR(c)*(SQR(a)+SQR(b)-SQR(c))))*denom;
  Real norm_c = sqrt(SQR(x_c)+SQR(y_c)+SQR(z_c));
  x = x_c/norm_c;
  y = y_c/norm_c;
  z = z_c/norm_c;
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void GeodesicGrid::SolidAngleAndArcLengths
//! \brief retrieve solid angles and arc lengths

void GeodesicGrid::SolidAngleAndArcLengths(int n, Real& weight, Real length[6]) {
  int nnum; int nvec[6];
  Neighbors(n, nnum, nvec);
  Real x0, y0, z0;
  GridCartPosition(n,x0,y0,z0);
  weight = 0.0;
  for (int nb=0; nb<nnum; ++nb) {
    Real xn1, yn1, zn1;
    Real xn2, yn2, zn2;
    Real xn3, yn3, zn3;
    GridCartPosition(nvec[(nb+nnum-1)%nnum],xn1,yn1,zn1);
    GridCartPosition(nvec[nb],              xn2,yn2,zn2);
    GridCartPosition(nvec[(nb+1)%nnum],     xn3,yn3,zn3);
    Real xc1, yc1, zc1;
    Real xc2, yc2, zc2;
    CircumcenterNormalized(x0,xn1,xn2,y0,yn1,yn2,z0,zn1,zn2,xc1,yc1,zc1);
    CircumcenterNormalized(x0,xn2,xn3,y0,yn2,yn3,z0,zn2,zn3,xc2,yc2,zc2);
    Real scalprod_c1 = x0 *xc1 + y0 *yc1 + z0 *zc1;
    Real scalprod_c2 = x0 *xc2 + y0 *yc2 + z0 *zc2;
    Real scalprod_12 = xc1*xc2 + yc1*yc2 + zc1*zc2;
    Real numerator = fabs(x0*(yc1*zc2-zc1*yc2) +
                          y0*(zc1*xc2-xc1*zc2) +
                          z0*(xc1*yc2-yc1*xc2));
    Real denominator = 1.0+scalprod_c1+scalprod_c2+scalprod_12;
    weight += 2.0*atan(numerator/denominator);
    length[nb] = acos(scalprod_12);
  }
  if (nnum == 5) {
    length[5] = (FLT_MAX);
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void GeodesicGrid::ArcLength
//! \brief find arc length between two face centers

Real GeodesicGrid::ArcLength(int n1, int n2) {
  Real x1, y1, z1, x2, y2, z2;
  GridCartPosition(n1,x1,y1,z1);
  GridCartPosition(n2,x2,y2,z2);
  return acos(x1*x2+y1*y2+z1*z2);
}

//----------------------------------------------------------------------------------------
//! \fn void GeodesicGrid::OptimalAngles
//! \brief find an optimal angle by which to rotate the geodesic mesh

void GeodesicGrid::OptimalAngles(Real ang[2]) {
  int nzeta = 200;  // nzeta val inherited from Viktoriya Giryanskaya
  int npsi  = 200;  // npsi  val inherited from Viktoriya Giryanskaya
  Real maxangle = ArcLength(0,1);
  Real deltazeta = maxangle/nzeta;
  Real deltapsi = M_PI/npsi;
  Real vmax = 0.0;
  for (int l=0; l<nzeta; ++l) {
    Real zeta = (l+1)*deltazeta;
    for (int k=0; k<npsi; ++k) {
      Real psi = (k+1)*deltapsi;
      Real kx = -sin(psi);
      Real ky =  cos(psi);
      Real vmin_curr = 1.0;
      for (int n=0; n<nangles; ++n) {
        Real vx, vy, vz;
        GridCartPosition(n,vx,vy,vz);
        Real vrx = vx*cos(zeta)+ky*vz*sin(zeta)+kx*(kx*vx+ky*vy)*(1.0-cos(zeta));
        Real vry = vy*cos(zeta)-kx*vz*sin(zeta)+ky*(kx*vx+ky*vy)*(1.0-cos(zeta));
        Real vrz = vz*cos(zeta)+(kx*vy-ky*vx)*sin(zeta);
        if (fabs(vrx) < vmin_curr) {vmin_curr = fabs(vrx);}
        if (fabs(vry) < vmin_curr) {vmin_curr = fabs(vry);}
        if (fabs(vrz) < vmin_curr) {vmin_curr = fabs(vrz);}
      }
      if (vmin_curr > vmax) {
        vmax = vmin_curr;
        ang[0] = zeta;
        ang[1] = psi;
      }
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void GeodesicGrid::RotateGrid
//! \brief rotate the geodesic grid such that the north pole angle center is assigned
//  to angular coordinate zetanew, psinew

void GeodesicGrid::RotateGrid(Real zetanew, Real psinew) {
  Real kx = -sin(psinew);
  Real ky =  cos(psinew);
  for (int bl=0; bl<5; ++bl) {
    for (int l=0; l<nlevel; ++l) {
      for (int m=0; m<2*nlevel; ++m) {
        Real vx = amesh_normals(bl,l+1,m+1,0);
        Real vy = amesh_normals(bl,l+1,m+1,1);
        Real vz = amesh_normals(bl,l+1,m+1,2);
        Real vrx = vx*cos(zetanew)+ky*vz*sin(zetanew)+kx*(kx*vx+ky*vy)*(1.0-cos(zetanew));
        Real vry = vy*cos(zetanew)-kx*vz*sin(zetanew)+ky*(kx*vx+ky*vy)*(1.0-cos(zetanew));
        Real vrz = vz*cos(zetanew)+(kx*vy-ky*vx)*sin(zetanew);
        amesh_normals(bl,l+1,m+1,0) = vrx;
        amesh_normals(bl,l+1,m+1,1) = vry;
        amesh_normals(bl,l+1,m+1,2) = vrz;
      }
    }
  }
  for (int pl=0; pl<2; ++pl) {
    Real vx = ameshp_normals(pl,0);
    Real vy = ameshp_normals(pl,1);
    Real vz = ameshp_normals(pl,2);
    Real vrx = vx*cos(zetanew)+ky*vz*sin(zetanew)+kx*(kx*vx+ky*vy)*(1.0-cos(zetanew));
    Real vry = vy*cos(zetanew)-kx*vz*sin(zetanew)+ky*(kx*vx+ky*vy)*(1.0-cos(zetanew));
    Real vrz = vz*cos(zetanew)+(kx*vy-ky*vx)*sin(zetanew);
    ameshp_normals(pl,0) = vrx;
    ameshp_normals(pl,1) = vry;
    ameshp_normals(pl,2) = vrz;
  }
  for (int i=0; i<3; ++i) {
    for (int bl=0; bl<5; ++bl) {
      for (int k=0; k<nlevel; ++k) {
        amesh_normals(bl,0,k+1,i)          = amesh_normals((bl+4)%5,k+1,1,i);
        amesh_normals(bl,0,k+nlevel+1,i)   = amesh_normals((bl+4)%5,nlevel,k+1,i);
        amesh_normals(bl,k+1,2*nlevel+1,i) = amesh_normals((bl+4)%5,nlevel,k+nlevel+1,i);
        amesh_normals(bl,k+2,0,i)          = amesh_normals((bl+1)%5,1,k+1,i);
        amesh_normals(bl,nlevel+1,k+1,i)   = amesh_normals((bl+1)%5,1,k+nlevel+1,i);
        amesh_normals(bl,nlevel+1,k+nlevel+1,i) = amesh_normals((bl+1)%5,k+2,2*nlevel,i);
      }
      amesh_normals(bl,1,0,i) = ameshp_normals(0,i);
      amesh_normals(bl,nlevel+1,2*nlevel,i) = ameshp_normals(1,i);
      amesh_normals(bl,0,2*nlevel+1,i) = amesh_normals(bl,0,2*nlevel,i);
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void GeodesicGrid::RotateGrid
//! \brief find components of unit vectors along edges

void GeodesicGrid::UnitFluxDir(Real zetav, Real psiv, Real zetaf, Real psif,
                               Real& dzeta, Real& dpsi) {
  if (fabs(psif-psiv) < 1.0e-10 ||
      fabs(fabs(cos(zetaf))-1.0) < 1.0e-10 ||
      fabs(fabs(cos(zetav))-1.0) < 1.0e-10) {
    dzeta = copysign(1.0,zetaf-zetav);
    dpsi = 0.0;
  } else {
    Real a_par, p_par;
    GreatCircleParam(zetav,zetaf,psiv,psif,a_par,p_par);
    Real zeta_deriv = (a_par*sin(psif-p_par)
                       / (1.0+SQR(a_par)*cos(psif-p_par)*cos(psif-p_par)));
    Real denom = 1.0/sqrt(SQR(zeta_deriv)+SQR(sin(zetaf)));
    Real signfactor = copysign(1.0,psif-psiv)*copysign(1.0,M_PI-fabs(psif-psiv));
    dzeta = signfactor*zeta_deriv*denom;
    dpsi  = signfactor*denom;
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void GeodesicGrid::GreatCircleParam
//! \brief find parameters describing the great circle connecting two angular coordinates

void GeodesicGrid::GreatCircleParam(Real zeta1, Real zeta2, Real psi1, Real psi2,
                                    Real& apar, Real& psi0) {
  Real atilde = (sin(psi2)/tan(zeta1)-sin(psi1)/tan(zeta2))/sin(psi2-psi1);
  Real btilde = (cos(psi2)/tan(zeta1)-cos(psi1)/tan(zeta2))/sin(psi1-psi2);
  psi0 = atan2(btilde, atilde);
  apar = sqrt(SQR(atilde)+SQR(btilde));
  return;
}
