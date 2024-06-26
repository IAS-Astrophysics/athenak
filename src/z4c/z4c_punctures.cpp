//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file z4c_punctures.cpp
//  \brief implementation of functions in the Z4c class for initializing punctures
//         evolution

// C++ standard headers
#include <cmath> // pow
#include <iostream>
#include <fstream>

// Athena++ headers
#include "parameter_input.hpp"
#include "athena.hpp"
#include "coordinates/adm.hpp"
#include "mesh/mesh.hpp"
#include "z4c/z4c.hpp"
#include "coordinates/cell_locations.hpp"


namespace z4c {
//----------------------------------------------------------------------------------------
// \!fn void Z4c::ADMOnePuncture(MeshBlockPack *pmbp, ParameterInput *pin)
// \brief Initialize ADM vars to single puncture (no spin)

void Z4c::ADMOnePuncture(MeshBlockPack *pmbp, ParameterInput *pin) {
  // capture variables for the kernel
  auto &indcs = pmbp->pmesh->mb_indcs;
  auto &size = pmbp->pmb->mb_size;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  //For GLOOPS
  int isg = is-indcs.ng; int ieg = ie+indcs.ng;
  int jsg = js-indcs.ng; int jeg = je+indcs.ng;
  int ksg = ks-indcs.ng; int keg = ke+indcs.ng;
  int ncells1 = indcs.nx1 + 2*(indcs.ng);
  int nmb = pmbp->nmb_thispack;
  Real ADM_mass = pin->GetOrAddReal("problem", "punc_ADM_mass", 1.);
  auto &z4c = pmbp->pz4c->z4c;
  ADM::ADM_vars &adm = pmbp->padm->adm;

  int scr_level = 0;
  size_t scr_size = ScrArray1D<Real>::shmem_size(ncells1);
  par_for_outer("pgen one puncture",DevExeSpace(),scr_size,scr_level,
                0,nmb-1,ksg,keg,jsg,jeg,
  KOKKOS_LAMBDA(TeamMember_t member, const int m, const int k, const int j) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    int nx1 = indcs.nx1;

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    int nx2 = indcs.nx2;
    Real x2v = CellCenterX(j-js, nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    int nx3 = indcs.nx3;
    Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);
    AthenaScratchTensor<Real, TensorSymm::NONE, 3, 0> r;
    r.NewAthenaScratchTensor(member, scr_level, nx1);

    par_for_inner(member, isg, ieg, [&](const int i) {
      Real x1v = CellCenterX(i-is, nx1, x1min, x1max);
      r(i) = std::sqrt(std::pow(x3v,2) + std::pow(x2v,2) + std::pow(x1v,2));
    });

    // Minkowski spacetime
    for(int a = 0; a < 3; ++a)
    for(int b = a; b < 3; ++b) {
      par_for_inner(member, isg, ieg, [&](const int i) {
        adm.g_dd(m,a,b,k,j,i) = (a == b ? 1. : 0.);
      });
    }
    // admK_dd is automatically set to 0 when is initialized as Kokkos View

    // ADMOnePuncture
    par_for_inner(member, isg, ieg, [&](const int i) {
      adm.psi4(m,k,j,i) = std::pow(1.0 + 0.5*ADM_mass/r(i),4); // adm.psi4
    });
    for(int a = 0; a < 3; ++a)
    for(int b = a; b < 3; ++b) {
      par_for_inner(member, isg, ieg, [&](const int i) {
        adm.g_dd(m,a,b,k,j,i) *= adm.psi4(m,k,j,i);
      });
    }
  });
}

// \!fn void Z4c::ADMTwoPunctures(MeshBlockPack *pmbp, ini_data *data)
// \brief Interpolate two puncture initial data in cartesian grid
//
// p  = detgbar^(-1/3)
// p0 = psi^(-4)
//
// gtilde_ij = p gbar_ij
// Ktilde_ij = p p0 K_ij
//
// phi = - log(p) / 4
// K   = gtildeinv^ij Ktilde_ij
// Atilde_ij = Ktilde_ij - gtilde_ij K / 3
//
// G^i = - del_j gtildeinv^ji
//
#if TWO_PUNCTURES
void Z4c::ADMTwoPunctures(MeshBlockPack *pmbp, ini_data *data) {
  // capture variables for the kernel
  auto &u_adm = pmbp->padm->u_adm;

  HostArray5D<Real>::HostMirror host_u_adm = create_mirror(u_adm);
  ADM_vars host_adm;
  host_adm.psi4.InitWithShallowSlice(host_u_adm, ADM::I_ADM_psi4);
  host_adm.g_dd.InitWithShallowSlice(host_u_adm, ADM::I_ADM_gxx, ADM::I_ADM_gzz);
  host_adm.K_dd.InitWithShallowSlice(host_u_adm, ADM::I_ADM_Kxx, ADM::I_ADM_Kzz);
  auto &indcs = pmbp->pmesh->mb_indcs;
  auto &size = pmbp->pmb->mb_size;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  //For GLOOPS
  int isg = is-indcs.ng; int ieg = ie+indcs.ng;
  int jsg = js-indcs.ng; int jeg = je+indcs.ng;
  int ksg = ks-indcs.ng; int keg = ke+indcs.ng;

  int ncells1 = indcs.nx1 + 2*(indcs.ng);
  int ncells2 = indcs.nx2 + 2*(indcs.ng);
  int ncells3 = indcs.nx3 + 2*(indcs.ng);
  int nmb = pmbp->nmb_thispack;
  Kokkos::parallel_for("pgen two puncture",
  Kokkos::RangePolicy<>(Kokkos::DefaultHostExecutionSpace(), 0, nmb),
  KOKKOS_LAMBDA(const int m) {
    int imin[3] = {0, 0, 0};

    int n[3] = {ncells1, ncells2, ncells3};

    int sz = n[0] * n[1] * n[2];
    // this could be done instead by accessing and casting the Athena vars but
    // then it is coupled to implementation details etc.
    Real *gxx = new Real[sz], *gyy = new Real[sz], *gzz = new Real[sz];
    Real *gxy = new Real[sz], *gxz = new Real[sz], *gyz = new Real[sz];

    Real *Kxx = new Real[sz], *Kyy = new Real[sz], *Kzz = new Real[sz];
    Real *Kxy = new Real[sz], *Kxz = new Real[sz], *Kyz = new Real[sz];

    Real *psi = new Real[sz];
    Real *alp = new Real[sz];

    Real *x = new Real[n[0]];
    Real *y = new Real[n[1]];
    Real *z = new Real[n[2]];

    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    int nx1 = indcs.nx1;

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    int nx2 = indcs.nx2;

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    int nx3 = indcs.nx3;

    // need to populate coordinates
    for(int ix_I = isg; ix_I < ieg+1; ix_I++) {
      x[ix_I] = CellCenterX(ix_I-is, nx1, x1min, x1max);
    }

    for(int ix_J = jsg; ix_J < jeg+1; ix_J++) {
      y[ix_J] = CellCenterX(ix_J-js, nx2, x2min, x2max);
    }

    for(int ix_K = ksg; ix_K < keg+1; ix_K++) {
      z[ix_K] = CellCenterX(ix_K-ks, nx3, x3min, x3max);
    }
    TwoPunctures_Cartesian_interpolation
      (data, // struct containing the previously calculated solution
       imin, // min, max idxs of Cartesian Grid in the three directions
       n,    // <-imax, but this collapses
       n,    // total number of indices in each direction
       x,    // x,         // Cartesian coordinates
       y,    // y,
       z,    // z,
       alp,  // alp,       // lapse
       psi,  // psi,       // conformal factor and derivatives
       NULL, // psix,
       NULL, // psiy,
       NULL, // psiz,
       NULL, // psixx,
       NULL, // psixy,
       NULL, // psixz,
       NULL, // psiyy,
       NULL, // psiyz,
       NULL, // psizz,
       gxx,  // gxx,       // metric components
       gxy,  // gxy,
       gxz,  // gxz,
       gyy,  // gyy,
       gyz,  // gyz,
       gzz,  // gzz,
       Kxx,  // kxx,       // extrinsic curvature components
       Kxy,  // kxy,
       Kxz,  // kxz,
       Kyy,  // kyy,
       Kyz,  // kyz,
       Kzz   // kzz
       );

    par_for("Two punctures",Kokkos::DefaultHostExecutionSpace(),ksg,keg,jsg,jeg,isg,ieg,
    KOKKOS_LAMBDA(const int k, const int j, const int i) {
      int flat_ix = i + n[0]*(j + n[1]*k);
      host_adm.psi4(m,k,j,i) = std::pow(psi[flat_ix], 4);

      host_adm.g_dd(m,0, 0, k, j, i) = host_adm.psi4(m,k,j,i) * gxx[flat_ix];
      host_adm.g_dd(m,1, 1, k, j, i) = host_adm.psi4(m,k,j,i) * gyy[flat_ix];
      host_adm.g_dd(m,2, 2, k, j, i) = host_adm.psi4(m,k,j,i) * gzz[flat_ix];
      host_adm.g_dd(m,0, 1, k, j, i) = host_adm.psi4(m,k,j,i) * gxy[flat_ix];
      host_adm.g_dd(m,0, 2, k, j, i) = host_adm.psi4(m,k,j,i) * gxz[flat_ix];
      host_adm.g_dd(m,1, 2, k, j, i) = host_adm.psi4(m,k,j,i) * gyz[flat_ix];

      host_adm.K_dd(m,0, 0, k, j, i) = Kxx[flat_ix];
      host_adm.K_dd(m,1, 1, k, j, i) = Kyy[flat_ix];
      host_adm.K_dd(m,2, 2, k, j, i) = Kzz[flat_ix];
      host_adm.K_dd(m,0, 1, k, j, i) = Kxy[flat_ix];
      host_adm.K_dd(m,0, 2, k, j, i) = Kxz[flat_ix];
      host_adm.K_dd(m,1, 2, k, j, i) = Kyz[flat_ix];
    });

    free(gxx); free(gyy); free(gzz);
    free(gxy); free(gxz); free(gyz);

    free(Kxx); free(Kyy); free(Kzz);
    free(Kxy); free(Kxz); free(Kyz);

    free(psi); free(alp);

    free(x); free(y); free(z);
  });

  Kokkos::deep_copy(u_adm, host_u_adm);
  return;
}

#endif

} // end namespace z4c
