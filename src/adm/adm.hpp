#ifndef ADM_ADM_HPP_
#define ADM_ADM_HPP_

//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file adm.hpp
//! \brief definitions for ADM class

#include "athena.hpp"
#include "athena_tensor.hpp"
#include "parameter_input.hpp"

// forward declarations
class MeshBlockPack;

namespace adm {
//! \class ADM
class ADM {
 public:
  //! WARNING: The ADM object needs to be allocated after Z4c
  ADM(MeshBlockPack *ppack, ParameterInput *pin);
  ~ADM();

  // Indices of ADM variables
  enum {
    I_ADM_GXX, I_ADM_GXY, I_ADM_GXZ, I_ADM_GYY, I_ADM_GYZ, I_ADM_GZZ,
    I_ADM_KXX, I_ADM_KXY, I_ADM_KXZ, I_ADM_KYY, I_ADM_KYZ, I_ADM_KZZ,
    I_ADM_PSI4,
    I_ADM_ALPHA, I_ADM_BETAX, I_ADM_BETAY, I_ADM_BETAZ,
    nadm
  };
  // Names of ADM variables
  static char const * const ADM_names[nadm];

  struct ADM_vars {
    AthenaTensor<Real, TensorSymm::NONE, 3, 0> alpha;     // lapse
    AthenaTensor<Real, TensorSymm::NONE, 3, 1> beta_u;    // shift vector
    AthenaTensor<Real, TensorSymm::NONE, 3, 0> psi4;      // conformal factor
    AthenaTensor<Real, TensorSymm::SYM2, 3, 2> g_dd;      // spatial metric
    AthenaTensor<Real, TensorSymm::SYM2, 3, 2> vK_dd;      // extrinsic curvature
  };
  ADM_vars adm;

  DvceArray5D<Real> u_adm;                                   // adm variables

  // TODO(Francesco): handle regridding

 private:
  MeshBlockPack* pmy_pack;  // ptr to MeshBlockPack containing this Z4c
};

KOKKOS_INLINE_FUNCTION
Real SpatialDet(Real const gxx, Real const gxy, Real const gxz,
                Real const gyy, Real const gyz, Real const gzz) {
  return - SQR(gxz)*gyy + 2*gxy*gxz*gyz
         - SQR(gyz)*gxx
         - SQR(gxy)*gzz +   gxx*gyy*gzz;
}

KOKKOS_INLINE_FUNCTION
Real Trace(Real const detginv,
           Real const gxx, Real const gxy, Real const gxz,
           Real const gyy, Real const gyz, Real const gzz,
           Real const Axx, Real const Axy, Real const Axz,
           Real const Ayy, Real const Ayz, Real const Azz) {
  return (detginv*(
       - 2.*Ayz*gxx*gyz + Axx*gyy*gzz +  gxx*(Azz*gyy + Ayy*gzz)
       + 2.*(gxz*(Ayz*gxy - Axz*gyy + Axy*gyz) + gxy*(Axz*gyz - Axy*gzz))
       - Azz*SQR(gxy) - Ayy*SQR(gxz) - Axx*SQR(gyz)
       ));
}

KOKKOS_INLINE_FUNCTION
// compute inverse of a 3x3 matrix
void SpatialInv(Real const detginv,
                Real const gxx, Real const gxy, Real const gxz,
                Real const gyy, Real const gyz, Real const gzz,
                Real * uxx, Real * uxy, Real * uxz,
                Real * uyy, Real * uyz, Real * uzz) {
  *uxx = (-SQR(gyz) + gyy*gzz)*detginv;
  *uxy = (gxz*gyz  - gxy*gzz)*detginv;
  *uyy = (-SQR(gxz) + gxx*gzz)*detginv;
  *uxz = (-gxz*gyy + gxy*gyz)*detginv;
  *uyz = (gxy*gxz  - gxx*gyz)*detginv;
  *uzz = (-SQR(gxy) + gxx*gyy)*detginv;
  return;
}

KOKKOS_INLINE_FUNCTION
void SpacetimeMetric(Real const alp,
                     Real const betax, Real const betay, Real const betaz,
                     Real const gxx, Real const gxy, Real const gxz,
                     Real const gyy, Real const gyz, Real const gzz,
                     Real g[16]) {
  g[5]  = gxx;
  g[6]  = gxy;
  g[7]  = gxz;
  g[9]  = gxy;
  g[10] = gyy;
  g[11] = gyz;
  g[13] = gxz;
  g[14] = gyz;
  g[15] = gzz;

  Real betaup[3] = {betax, betay, betaz};
  Real betadw[3] = {
    gxx*betax + gxy*betay + gxz*betaz,
    gxy*betax + gyy*betay + gyz*betaz,
    gxz*betax + gyz*betay + gzz*betaz,
  };

  g[0] = - SQR(alp) + betadw[0]*betaup[0] + betadw[1]*betaup[1] +
          betadw[2]*betaup[2];

  g[1] = betadw[0];
  g[2] = betadw[1];
  g[3] = betadw[2];

  g[4]  = betadw[0];
  g[8]  = betadw[1];
  g[12] = betadw[2];
}

KOKKOS_INLINE_FUNCTION
void SpacetimeUpperMetric(Real const alp,
                          Real const betax, Real const betay, Real const betaz,
                          Real const gxx, Real const gxy, Real const gxz,
                          Real const gyy, Real const gyz, Real const gzz,
                          Real u[16]) {
  u[0] = - 1.0/SQR(alp);

  Real const det = SpatialDet(gxx, gxy, gxz, gyy, gyz, gzz);

  Real uxx, uxy, uxz, uyy, uyz, uzz;
  SpatialInv(1.0/det, gxx, gxy, gxz, gyy, gyz, gzz,
             &uxx, &uxy, &uxz, &uyy, &uyz, &uzz);
  u[5]  = uxx + betax*betax*u[0];
  u[6]  = uxy + betax*betay*u[0];
  u[7]  = uxz + betax*betaz*u[0];
  u[9]  = uxy + betax*betay*u[0];
  u[10] = uyy + betay*betay*u[0];
  u[11] = uyz + betay*betaz*u[0];
  u[13] = uxz + betax*betaz*u[0];
  u[14] = uyz + betay*betaz*u[0];
  u[15] = uzz + betaz*betaz*u[0];

  u[1] = betax*(-u[0]);
  u[2] = betay*(-u[0]);
  u[3] = betaz*(-u[0]);

  u[4]  = u[1];
  u[8]  = u[2];
  u[12] = u[3];
}

} // namespace adm
#endif  // ADM_ADM_HPP_
