#ifndef RADIATION_RADIATION_TETRAD_HPP_
#define RADIATION_RADIATION_TETRAD_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_tetrad.hpp
//  \brief computes orthonormal tetrad in Cartesian Kerr-Schild

#include <math.h>

#include "athena.hpp"

// computes covariant and contravariant components of Cartesian tetrad for CKS
KOKKOS_INLINE_FUNCTION
void ComputeTetrad(Real x, Real y, Real z, const bool minkowski, const Real a,
                   Real g[][4], Real gi[][4], Real dgx[][4], Real dgy[][4], Real dgz[][4],
                   Real e[][4], Real ecov[][4], Real omega[][4][4]) {
  // if (fabs(z) < (SMALL_NUMBER)) z = (SMALL_NUMBER);  // see cartesian_ks.hpp comments
  Real rad = sqrt(SQR(x) + SQR(y) + SQR(z));
  Real r = sqrt((SQR(rad)-SQR(a)+sqrt(SQR(SQR(rad)-SQR(a))+4.0*SQR(a)*SQR(z)))/2.0);
  r = fmax(r, 1.0);  // floor r_ks to 0.5*(r_inner + r_outer)

  Real ll1 = (r*x + (a)*y)/( SQR(r) + SQR(a) );
  Real ll2 = (r*y - (a)*x)/( SQR(r) + SQR(a) );
  Real ll3 = z/r;
  Real f = 2.0 * SQR(r)*r / (SQR(SQR(r)) + SQR(a)*SQR(z));
  if (minkowski) {f=0.0;}

  // Set Cartesian tetrad
  Real wa = sqrt(1.0+f);
  Real wb = sqrt(1.0+f*(SQR(ll1)+SQR(ll2)));
  Real wc = sqrt(1.0+f*SQR(ll2));
  Real iwa = 1.0/wa;  Real iwasq = SQR(iwa);
  Real iwb = 1.0/wb;  Real iwbsq = SQR(iwb);
  Real iwc = 1.0/wc;  Real iwcsq = SQR(iwc);
  e[0][0] = wa;
  e[0][1] = -f*iwa*ll1;
  e[0][2] = -f*iwa*ll2;
  e[0][3] = -f*iwa*ll3;
  e[1][0] = 0.0;
  e[1][1] = iwb*wc;
  e[1][2] = -f*iwb*iwc*ll1*ll2;
  e[1][3] = 0.0;
  e[2][0] = 0.0;
  e[2][1] = 0.0;
  e[2][2] = iwc;
  e[2][3] = 0.0;
  e[3][0] = 0.0;
  e[3][1] = -f*iwa*iwb*ll1*ll3;
  e[3][2] = -f*iwa*iwb*ll2*ll3;
  e[3][3] = iwa*wb;

  // Set derivatives of tetrad
  Real de[4][4][4] = {0.0};
  Real qa = 2.0*SQR(r) - SQR(rad) + SQR(a);
  Real qb = SQR(r) + SQR(a);
  Real qc = 3.0*SQR(a * z)-SQR(r)*SQR(r);
  Real df_dx1 = SQR(f)*x/(2.0*pow(r,3)) * ( ( qc ) )/ qa;
  Real df_dx2 = SQR(f)*y/(2.0*pow(r,3)) * ( ( qc ) )/ qa;
  Real df_dx3 = SQR(f)*z/(2.0*pow(r,5)) * ( ( qc * qb ) / qa - 2.0*SQR(a*r));
  if (minkowski) {
    f = 0.0;
    df_dx1 = 0.0;
    df_dx2 = 0.0;
    df_dx3 = 0.0;
  }
  Real dl1_dx1 = x*r*(SQR(a)*x-2.0*a*r*y-SQR(r)*x)/(SQR(qb)*qa)+r/(qb);
  Real dl1_dx2 = y*r*(SQR(a)*x-2.0*a*r*y-SQR(r)*x)/(SQR(qb)*qa)+a/(qb);
  Real dl1_dx3 = z/r*(SQR(a)*x-2.0*a*r*y-SQR(r)*x)/((qb)*qa);
  Real dl2_dx1 = x*r*(SQR(a)*y+2.0*a*r*x-SQR(r)*y)/(SQR(qb)*qa)-a/(qb);
  Real dl2_dx2 = y*r*(SQR(a)*y+2.0*a*r*x-SQR(r)*y)/(SQR(qb)*qa)+r/(qb);
  Real dl2_dx3 = z/r*(SQR(a)*y+2.0*a*r*x-SQR(r)*y)/((qb)*qa);
  Real dl3_dx1 = -x*z/(r*qa);
  Real dl3_dx2 = -y*z/(r*qa);
  Real dl3_dx3 = -SQR(z)/(SQR(r)*r) * ( qb )/( qa ) + 1.0/r;
  Real dwa_dx1 = 0.5*iwa*df_dx1;
  Real dwa_dx2 = 0.5*iwa*df_dx2;
  Real dwa_dx3 = 0.5*iwa*df_dx3;
  Real dwb_dx1 = 0.5*iwb*(2.*f*ll1*dl1_dx1+2.0*f*ll2*dl2_dx1+(SQR(ll1)+SQR(ll2))*df_dx1);
  Real dwb_dx2 = 0.5*iwb*(2.*f*ll1*dl1_dx2+2.0*f*ll2*dl2_dx2+(SQR(ll1)+SQR(ll2))*df_dx2);
  Real dwb_dx3 = 0.5*iwb*(2.*f*ll1*dl1_dx3+2.0*f*ll2*dl2_dx3+(SQR(ll1)+SQR(ll2))*df_dx3);
  Real dwc_dx1 = 0.5*iwc*(2.*f*ll2*dl2_dx1+SQR(ll2)*df_dx1);
  Real dwc_dx2 = 0.5*iwc*(2.*f*ll2*dl2_dx2+SQR(ll2)*df_dx2);
  Real dwc_dx3 = 0.5*iwc*(2.*f*ll2*dl2_dx3+SQR(ll2)*df_dx3);
  de[1][0][0]  = dwa_dx1;
  de[1][0][1]  = -f*iwa*dl1_dx1 - iwa*ll1*df_dx1 + f*iwasq*ll1*dwa_dx1;
  de[1][0][2]  = -f*iwa*dl2_dx1 - iwa*ll2*df_dx1 + f*iwasq*ll2*dwa_dx1;
  de[1][0][3]  = -f*iwa*dl3_dx1 - iwa*ll3*df_dx1 + f*iwasq*ll3*dwa_dx1;
  de[1][1][1]  = iwb*dwc_dx1 - iwbsq*wc*dwb_dx1;
  de[1][1][2]  = -f*iwb*iwc*ll1*dl2_dx1 - f*iwb*iwc*ll2*dl1_dx1;
  de[1][1][2] += (-iwb*iwc*df_dx1 + f*iwbsq*iwc*dwb_dx1 + f*iwb*iwcsq*dwc_dx1)*ll1*ll2;
  de[1][2][2]  = -iwcsq*dwc_dx1;
  de[1][3][1]  = -f*iwa*iwb*ll1*dl3_dx1 - f*iwa*iwb*ll3*dl1_dx1;
  de[1][3][1] += (-iwa*iwb*df_dx1 + f*iwasq*iwb*dwa_dx1 + f*iwa*iwbsq*dwb_dx1)*ll1*ll3;
  de[1][3][2]  = -f*iwa*iwb*ll2*dl3_dx1 - f*iwa*iwb*ll3*dl2_dx1;
  de[1][3][2] += (-iwa*iwb*df_dx1 + f*iwasq*iwb*dwa_dx1 + f*iwa*iwbsq*dwb_dx1)*ll2*ll3;
  de[1][3][3]  = iwa*dwb_dx1 - iwasq*wb*dwa_dx1;
  de[2][0][0]  = dwa_dx2;
  de[2][0][1]  = -f*iwa*dl1_dx2 - iwa*ll1*df_dx2 + f*iwasq*ll1*dwa_dx2;
  de[2][0][2]  = -f*iwa*dl2_dx2 - iwa*ll2*df_dx2 + f*iwasq*ll2*dwa_dx2;
  de[2][0][3]  = -f*iwa*dl3_dx2 - iwa*ll3*df_dx2 + f*iwasq*ll3*dwa_dx2;
  de[2][1][1]  = iwb*dwc_dx2 - iwbsq*wc*dwb_dx2;
  de[2][1][2]  = -f*iwb*iwc*ll1*dl2_dx2 - f*iwb*iwc*ll2*dl1_dx2;
  de[2][1][2] += (-iwb*iwc*df_dx2 + f*iwbsq*iwc*dwb_dx2 + f*iwb*iwcsq*dwc_dx2)*ll1*ll2;
  de[2][2][2]  = -iwcsq*dwc_dx2;
  de[2][3][1]  = -f*iwa*iwb*ll1*dl3_dx2 - f*iwa*iwb*ll3*dl1_dx2;
  de[2][3][1] += (-iwa*iwb*df_dx2 + f*iwasq*iwb*dwa_dx2 + f*iwa*iwbsq*dwb_dx2)*ll1*ll3;
  de[2][3][2]  = -f*iwa*iwb*ll2*dl3_dx2 - f*iwa*iwb*ll3*dl2_dx2;
  de[2][3][2] += (-iwa*iwb*df_dx2 + f*iwasq*iwb*dwa_dx2 + f*iwa*iwbsq*dwb_dx2)*ll2*ll3;
  de[2][3][3]  = iwa*dwb_dx2 - iwasq*wb*dwa_dx2;
  de[3][0][0]  = dwa_dx3;
  de[3][0][1]  = -f*iwa*dl1_dx3 - iwa*ll1*df_dx3 + f*iwasq*ll1*dwa_dx3;
  de[3][0][2]  = -f*iwa*dl2_dx3 - iwa*ll2*df_dx3 + f*iwasq*ll2*dwa_dx3;
  de[3][0][3]  = -f*iwa*dl3_dx3 - iwa*ll3*df_dx3 + f*iwasq*ll3*dwa_dx3;
  de[3][1][1]  = iwb*dwc_dx3 - iwbsq*wc*dwb_dx3;
  de[3][1][2]  = -f*iwb*iwc*ll1*dl2_dx3 - f*iwb*iwc*ll2*dl1_dx3;
  de[3][1][2] += (-iwb*iwc*df_dx3 + f*iwbsq*iwc*dwb_dx3 + f*iwb*iwcsq*dwc_dx3)*ll1*ll2;
  de[3][2][2]  = -iwcsq*dwc_dx3;
  de[3][3][1]  = -f*iwa*iwb*ll1*dl3_dx3 - f*iwa*iwb*ll3*dl1_dx3;
  de[3][3][1] += (-iwa*iwb*df_dx3 + f*iwasq*iwb*dwa_dx3 + f*iwa*iwbsq*dwb_dx3)*ll1*ll3;
  de[3][3][2]  = -f*iwa*iwb*ll2*dl3_dx3 - f*iwa*iwb*ll3*dl2_dx3;
  de[3][3][2] += (-iwa*iwb*df_dx3 + f*iwasq*iwb*dwa_dx3 + f*iwa*iwbsq*dwb_dx3)*ll2*ll3;
  de[3][3][3]  = iwa*dwb_dx3 - iwasq*wb*dwa_dx3;

  // Stow metric derivatives in a temporary array
  Real dg[4][4][4] = {0.0};
  for (int i=0; i<4; ++i) {
    for (int j=0; j<4; ++j) {
      dg[0][i][j] = 0.0;
      dg[1][i][j] = dgx[i][j];
      dg[2][i][j] = dgy[i][j];
      dg[3][i][j] = dgz[i][j];
    }
  }

  // Set Minkowski metric
  Real eta[4][4] = {0.0};
  eta[0][0] = -1.0;
  eta[1][1] = 1.0;
  eta[2][2] = 1.0;
  eta[3][3] = 1.0;

  // Calculate covariant tetrad, inverse tetrad, and connection coefficients
  Real ei[4][4] = {0.0};
  Real gamma[4][4][4] = {0.0};
  for (int i=0; i<4; ++i) {
    for (int j=0; j<4; ++j) {
      ecov[i][j] = 0.0;
      ei[i][j] = 0.0;
      for (int k=0; k<4; ++k) {
        gamma[i][j][k] = 0.0;
        ecov[i][j] += g[j][k]*e[i][k];
        for (int l=0; l<4; ++l) {
          ei[i][j] += eta[i][k]*g[j][l]*e[k][l];
          gamma[i][j][k] += 0.5*gi[i][l]*(dg[j][l][k] + dg[k][l][j] - dg[l][j][k]);
        }
      }
    }
  }

  // Calculate Ricci rotation coefficients
  for (int i=0; i<4; ++i) {
    for (int j=0; j<4; ++j) {
      for (int k=0; k<4; ++k) {
        omega[i][j][k] = 0.0;
        for (int l=0; l<4; ++l) {
          for (int m=0; m<4; ++m) {
            omega[i][j][k] += ei[i][l]*e[k][m]*de[m][j][l];
            for (int n=0; n<4; ++n) {
              omega[i][j][k] += ei[i][l]*e[k][m]*gamma[l][m][n]*e[j][n];
            }
          }
        }
      }
    }
  }

  return;
}

#endif // RADIATION_RADIATION_TETRAD_HPP_
