//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dyngr_tov.cpp
//  \brief Problem generator for TOV star. Only works when ADM is enabled.

#include <stdio.h>
#include <math.h>     // abs(), cos(), exp(), log(), NAN, pow(), sin(), sqrt()

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

#include <algorithm>  // max(), max_element(), min(), min_element()
#include <iostream>   // endl
#include <limits>     // numeric_limits::max()
#include <sstream>    // stringstream
#include <string>     // c_str(), string
#include <cfloat>
#include <type_traits> // conditional

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/adm.hpp"
#include "z4c/z4c.hpp"
#include "coordinates/coordinates.hpp"
#include "coordinates/cell_locations.hpp"
#include "eos/eos.hpp"
#include "eos/primitive-solver/unit_system.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"
#include "utils/tov/tov.hpp"
#include "utils/tov/tov_polytrope.hpp"
#include "utils/tov/tov_tabulated.hpp"
#include "utils/tov/tov_piecewise_poly.hpp"

KOKKOS_INLINE_FUNCTION
static Real Aphi(Real x1, Real x2, Real P_rad, Real tube_rad, Real sigma);

// Prototypes for user-defined BCs and history
void TurbulenceBC(Mesh *pm);
template<class TOVEOS>
void TurbulenceHistory(HistoryData *pdata, Mesh *pm);

template<class TOVEOS>
struct conv_eos_type {
  using type = TOVEOS;
};

template<>
struct conv_eos_type<tov::PolytropeEOS> {
  using type = Primitive::IdealGas;
};

template<>
struct conv_eos_type<tov::TabulatedEOS> {
  using type = Primitive::EOSCompOSE<Primitive::NormalLogs>;
};

template<>
struct conv_eos_type<tov::PiecewisePolytropeEOS> {
  using type = Primitive::PiecewisePolytrope;
};

namespace tov_turb {

std::vector<Real> flux_loc;

} // namespace tov_turb

template<class TOVEOS>
void SetupTOV(ParameterInput *pin, Mesh* pmy_mesh_) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;

  // Use the TOV solver with the specified EOS.
  TOVEOS eos{pin};
  auto my_tov = tov::TOVStar::ConstructTOV(pin, eos);

  constexpr bool use_ye = tov::UsesYe<TOVEOS>;
  Real ye_atmo = pin->GetOrAddReal("mhd", "s0_atmosphere", 0.5);

  Real A_pert = pin->GetOrAddReal("problem", "A_pert", 0.0);
  int k_pert = pin->GetOrAddReal("problem", "k_pert", 1);

  // Parse some parameters
  Real b_norm = pin->GetOrAddReal("problem", "b_norm", 0.0);
  Real q_twist = pin->GetOrAddReal("problem", "q_twist", 0.0);
  Real sigma = pin->GetOrAddReal("problem", "sigma", 1.0);
  Real tube_radius = pin->GetOrAddReal("problem", "tube_radius", 1.0);
  Real P_rad = pin->GetOrAddReal("problem", "P_rad", 6.0);
  bool equilibrium = pin->GetOrAddBoolean("problem", "equilibrium", true);
  Real tube_at_rho = pin->GetOrAddReal("problem", "tube_at_rho", 0.0);

  auto &w0_ = pmbp->pmhd->w0;
  int& nvars_ = pmbp->pmhd->nmhd;
  int& nscal_ = pmbp->pmhd->nscalars;

  // Calculate the magnetic field strength in code units
  auto cgs = Primitive::MakeCGS();
  auto geo = Primitive::MakeGeometricSolar();

  // Note: this should be sqrt(1/4pi), not sqrt(4pi). This means all fields are actually
  // 4pi larger than what shows up in the parameter file, but the results in the paper
  // are correct.
  const Real mag_cgs_to_geo = Kokkos::sqrt(4.0*M_PI*cgs.MassDensityConversion(geo))*
                         (geo.c/cgs.c);

  b_norm *= mag_cgs_to_geo;

  // Capture variables for kernel
  auto &indcs = pmy_mesh_->mb_indcs;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*ng) : 1;
  int &is = indcs.is;
  int &ie = indcs.ie;
  int &js = indcs.js;
  int &je = indcs.je;
  int &ks = indcs.ks;
  int &ke = indcs.ke;
  int nmb1 = pmbp->nmb_thispack - 1;
  auto &coord = pmbp->pcoord->coord_data;

  auto &size = pmbp->pmb->mb_size;
  auto &adm = pmbp->padm->adm;
  auto &tov_ = my_tov;
  auto &eos_ = eos;

  if (tube_at_rho > 0.0) {
    // If the "tube_at_rho" argument is nonzero, we want to place the tube at a specific
    // density inside the star. We can do a bisection search to find this point.
    int il = 0;
    int iu = tov_.n_r;
    int it = (il + iu)/2;
    while (iu > il+1) {
      Real P = tov_.P.h_view(it);
      Real rho = eos_.template GetRhoFromP<tov::LocationTag::Host>(P);
      if (rho > tube_at_rho) {
        il = it;
      } else {
        iu = it;
      }
      it = (il + iu)/2;
    }
    // Interpolate to get the radius to place the flux tube.
    Real lrho_low = Kokkos::log(eos_.template
        GetRhoFromP<tov::LocationTag::Host>(tov_.P.h_view(il)));
    Real lrho_high = Kokkos::log(eos_.template
        GetRhoFromP<tov::LocationTag::Host>(tov_.P.h_view(iu)));

    Real t = (Kokkos::log(tube_at_rho) - lrho_low)/(lrho_high - lrho_low);

    P_rad = tov_.R.h_view(il) + (tov_.R.h_view(iu) - tov_.R.h_view(il))*t;
  }

  par_for("pgen_tov1", DevExeSpace(), 0, nmb1, 0, (n3-1), 0, (n2-1), 0, (n1-1),
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);

    Real rho, p, mass, alp;
    Real ye = ye_atmo;
    auto &use_ye_ = use_ye;
    tov_.GetPrimitivesAtPoint(eos_, x1v, rho, p, mass, alp);
    if constexpr (use_ye) {
      if (x1v < tov_.R_edge) {
        ye = eos_.template GetYeFromRho<tov::LocationTag::Device>(rho);
      }
    }

    // Calculate the radius for the velocity perturbation
    Real rho_pert = 0.0;
    Real tracer = 0.0;
    Real rhosq = x1v*x1v + P_rad*P_rad - 2.*x1v*P_rad*sin(x2v);
    if (rhosq <= tube_radius*tube_radius) {
      rho_pert = -A_pert*cos(k_pert*x3v);
      tracer = 1.0;
    }


    // Set hydrodynamic quantities
    w0_(m,IDN,k,j,i) = rho*(1.0 + rho_pert);
    w0_(m,IPR,k,j,i) = p;
    w0_(m,IVX,k,j,i) = 0.0;
    w0_(m,IVY,k,j,i) = 0.0;
    w0_(m,IVZ,k,j,i) = 0.0;
    auto &nvars = nvars_;
    auto &nscal = nscal_;
    if constexpr (use_ye) {
      w0_(m,nvars,k,j,i) = ye;
      if (nscal > 1) {
        w0_(m,nvars+1,k,j,i) = tracer;
      }
    } else {
      if (nscal > 0) {
        w0_(m,nvars,k,j,i) = tracer;
      }
    }

    // Set ADM variables
    adm.alpha(m,k,j,i) = alp;
    adm.g_dd(m,0,0,k,j,i) = 1./(1.0 - 2.0*mass/x1v);
    adm.g_dd(m,0,1,k,j,i) = 0.0;
    adm.g_dd(m,0,2,k,j,i) = 0.0;
    adm.g_dd(m,1,1,k,j,i) = x1v*x1v;
    adm.g_dd(m,1,2,k,j,i) = 0.0;
    adm.g_dd(m,2,2,k,j,i) = x1v*x1v*sin(x2v)*sin(x2v);
    Real det = adm::SpatialDet(
            adm.g_dd(m,0,0,k,j,i), adm.g_dd(m,0,1,k,j,i),
            adm.g_dd(m,0,2,k,j,i), adm.g_dd(m,1,1,k,j,i),
            adm.g_dd(m,1,2,k,j,i), adm.g_dd(m,2,2,k,j,i));
    adm.psi4(m,k,j,i) = pow(det, 1./3.);
    adm.beta_u(m,0,k,j,i) = adm.beta_u(m,1,k,j,i) = adm.beta_u(m,2,k,j,i) = 0.0;
    adm.vK_dd(m,0,0,k,j,i) = adm.vK_dd(m,0,1,k,j,i) = adm.vK_dd(m,0,2,k,j,i) = 0.0;
    adm.vK_dd(m,1,1,k,j,i) = adm.vK_dd(m,1,2,k,j,i) = adm.vK_dd(m,2,2,k,j,i) = 0.0;
  });

  // compute vector potential over all faces
  int ncells1 = indcs.nx1 + 2*(indcs.ng);
  int ncells2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
  int ncells3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;
  DvceArray4D<Real> a1, a2, a3;
  //Kokkos::realloc(a1, nmb1+1,ncells3,ncells2,ncells1);
  //Kokkos::realloc(a2, nmb1+1,ncells3,ncells2,ncells1);
  Kokkos::realloc(a3, nmb1+1,ncells3,ncells2,ncells1);

  auto &nghbr = pmbp->pmb->nghbr;
  auto &mblev = pmbp->pmb->mb_lev;

  par_for("pgen_vector_potential", DevExeSpace(), 0,nmb1,ks,ke+1,js,je+1,is,ie+1,
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

    a3(m,k,j,i) = q_twist*Aphi(x1f, x2f, P_rad, tube_radius, sigma);

    // When neighboring MeshBock is at finer level, compute vector potential as sum of
    // values at fine grid resolution.  This guarantees flux on shared fine/coarse
    // faces is identical.

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
      a3(m,k,j,i) = 0.5*(q_twist*Aphi(x1f,x2f,P_rad,tube_radius,sigma) +
                         q_twist*Aphi(x1f,x2f,P_rad,tube_radius,sigma));
    }
  });

  auto &b0 = pmbp->pmhd->b0;
  par_for("pgen_Bfc", DevExeSpace(), 0,nmb1,ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    int nx1 = indcs.nx1;
    Real x1v = CellCenterX(i-is, nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    int nx2 = indcs.nx2;
    Real x2v = CellCenterX(j-js, nx2, x2min, x2max);

    // Compute face-centered fields from curl(A).
    Real dx1 = size.d_view(m).dx1;
    Real dx2 = size.d_view(m).dx2;

    /*b0.x1f(m,k,j,i) = ((a3(m,k,j+1,i) - a3(m,k,j,i))/dx2 -
                       (a2(m,k+1,j,i) - a2(m,k,j,i))/dx3);
    b0.x2f(m,k,j,i) = ((a1(m,k+1,j,i) - a1(m,k,j,i))/dx3 -
                       (a3(m,k,j,i+1) - a3(m,k,j,i))/dx1);
    b0.x3f(m,k,j,i) = ((a2(m,k,j,i+1) - a2(m,k,j,i))/dx1 -
                       (a1(m,k,j+1,i) - a1(m,k,j,i))/dx2);*/
    b0.x1f(m,k,j,i) = (a3(m,k,j+1,i) - a3(m,k,j,i))/dx2;
    b0.x2f(m,k,j,i) = -(a3(m,k,j,i+1) - a3(m,k,j,i))/dx1;

    Real rhosq = x1v*x1v + P_rad*P_rad - 2*x1v*P_rad*sin(x2v);
    b0.x3f(m,k,j,i) = (rhosq < tube_radius*tube_radius) ?
                       exp(-(rhosq)/(sigma*sigma)) : 0.0;
    // Include extra face-component at edge of block in each direction
    if (i==ie) {
      b0.x1f(m,k,j,i+1) = (a3(m,k,j+1,i+1) - a3(m,k,j,i+1))/dx2;
    }
    if (j==je) {
      b0.x2f(m,k,j+1,i) = -(a3(m,k,j+1,i+1) - a3(m,k,j+1,i))/dx1;
    }
    if (k==ke) {
      b0.x3f(m,k+1,j,i) = (rhosq < tube_radius*tube_radius) ? 
                           exp(-(rhosq)/(sigma*sigma)) : 0.0;
    }
  });

  // We need to keep the temperature fixed for the equilibrium calculation, so we have to
  // do a really nasty cast here.
  Primitive::EOS<typename conv_eos_type<TOVEOS>::type, Primitive::ResetFloor>& ps_eos =
    static_cast<
      dyngr::DynGRMHDPS<
        typename conv_eos_type<TOVEOS>::type,
        Primitive::ResetFloor
      >*
    >(pmbp->pdyngr)->eos.ps.GetEOSMutable();

  // Compute cell-centered fields
  auto &bcc_ = pmbp->pmhd->bcc0;
  NumTools::Root root;
  const int nmkji = (nmb1+1)*indcs.nx3*indcs.nx2*indcs.nx1;
  const int nkji = indcs.nx3*indcs.nx2*indcs.nx1;
  const int nji = indcs.nx2*indcs.nx1;
  Real rho_frac_max;
  Real betainv_max;
  Real twist_fraction;

  // TODO(JMF): Calculate maximum cell-centered magnetic field and rescale.
  Real bsq_max;
  Kokkos::parallel_reduce("pgen_Bcc",Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, Real &mb_bsq_max) {
    int m = (idx)/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/indcs.nx1;
    int i = (idx - m*nkji - k*nji - j*indcs.nx1) + is;
    k += ks;
    j += js;

    // Cell-centered fields are simple linear average of face-centered fields
    Real& w_bx = bcc_(m,IBX,k,j,i);
    Real& w_by = bcc_(m,IBY,k,j,i);
    Real& w_bz = bcc_(m,IBZ,k,j,i);
    w_bx = 0.5*(b0.x1f(m,k,j,i) + b0.x1f(m,k,j,i+1));
    w_by = 0.5*(b0.x2f(m,k,j,i) + b0.x2f(m,k,j+1,i));
    w_bz = 0.5*(b0.x3f(m,k,j,i) + b0.x3f(m,k+1,j,i));

    // Calculate bsq
    Real g3d[NSPMETRIC] =
               {adm.g_dd(m,0,0,k,j,i), adm.g_dd(m,0,1,k,j,i), adm.g_dd(m,0,2,k,j,i),
                adm.g_dd(m,1,1,k,j,i), adm.g_dd(m,1,2,k,j,i), adm.g_dd(m,2,2,k,j,i)};

    Real detg = Primitive::GetDeterminant(g3d);

    Real b3u[NMAG] = {w_bx, w_by, w_bz};
    Real Wvu[3] = {w0_(m,IVX,k,j,i), w0_(m,IVY,k,j,i), w0_(m,IVZ,k,j,i)};

    Real Wsq = 1.0 + Primitive::SquareVector(Wvu, g3d);

    Real Bu = 0.0;
    for (int a = 0; a < 3; a++) {
      for (int b = 0; b < 3; b++) {
        Bu += b3u[a]*Wvu[b]*adm.g_dd(m,a,b,k,j,i);
      }
    }
    Real Bsq = Primitive::SquareVector(b3u, g3d);

    mb_bsq_max = fmax(mb_bsq_max, (Bsq + Bu*Bu)/(Wsq*detg));
  }, Kokkos::Max<Real>(bsq_max));

#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &bsq_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
#endif

  Real b_scale = b_norm/Kokkos::sqrt(bsq_max);

  Kokkos::parallel_reduce("pgen_Bcc",Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, Real &mb_max, Real &mb_bsq_max, Real &mb_betainv_max,
  Real &mb_twist_fraction) {
    int m = (idx)/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/indcs.nx1;
    int i = (idx - m*nkji - k*nji - j*indcs.nx1) + is;
    k += ks;
    j += js;

    // Cell-centered fields are simple linear average of face-centered fields
    Real& w_bx = bcc_(m,IBX,k,j,i);
    Real& w_by = bcc_(m,IBY,k,j,i);
    Real& w_bz = bcc_(m,IBZ,k,j,i);
    w_bx *= b_scale;
    w_by *= b_scale;
    w_bz *= b_scale;

    b0.x1f(m,k,j,i) *= b_scale;
    b0.x2f(m,k,j,i) *= b_scale;
    b0.x3f(m,k,j,i) *= b_scale;
    if (i == ie) {
      b0.x1f(m,k,j,i+1) *= b_scale;
    }
    if (j == je) {
      b0.x2f(m,k,j+1,i) *= b_scale;
    }
    if (k == ke) {
      b0.x3f(m,k+1,j,i) *= b_scale;
    }

    if (equilibrium) {
      // Compute the magnetic pressure and subtract it from the fluid pressure to force
      // the solution to be in equilibrium.
      // WARNING: This requires p_mag < p_fluid!
      Real g3d[NSPMETRIC] =
                 {adm.g_dd(m,0,0,k,j,i), adm.g_dd(m,0,1,k,j,i), adm.g_dd(m,0,2,k,j,i),
                  adm.g_dd(m,1,1,k,j,i), adm.g_dd(m,1,2,k,j,i), adm.g_dd(m,2,2,k,j,i)};

      Real detg = Primitive::GetDeterminant(g3d);

      Real b3u[NMAG] = {w_bx, w_by, w_bz};
      Real Wvu[3] = {w0_(m,IVX,k,j,i), w0_(m,IVY,k,j,i), w0_(m,IVZ,k,j,i)};

      Real Wsq = 1.0 + Primitive::SquareVector(Wvu, g3d);

      Real Bu = 0.0;
      for (int a = 0; a < 3; a++) {
        for (int b = 0; b < 3; b++) {
          Bu += b3u[a]*Wvu[b]*adm.g_dd(m,a,b,k,j,i);
        }
      }

      Real Bsq = Primitive::SquareVector(b3u, g3d);

      Real pmag = 0.5*(Bsq + Bu*Bu)/(Wsq*detg);
      //mb_bsq_max = fmax(Primitive::SquareVector(b3u, g3d)/detg, mb_bsq_max);
      mb_bsq_max = fmax(2.0*pmag, mb_bsq_max);
      mb_betainv_max = fmax(pmag/w0_(m,IPR,k,j,i),mb_betainv_max);

      if (b3u[IBZ] > 0) {
        b3u[IBZ] = 0.0;
        Real Bsqtrans = Primitive::SquareVector(b3u, g3d);
        mb_twist_fraction = fmax(mb_twist_fraction, sqrt(Bsqtrans/Bsq));
      }

      // Calculate the temperature at this point
      Real n = w0_(m,IDN,k,j,i)/ps_eos.GetBaryonMass();
      Real Y[MAX_SPECIES] = {0.0};
      for (int s = 0; s < nscal_; s++) {
        Y[s] = w0_(m,nvars_+s,k,j,i);
      }
      Real T = ps_eos.GetTemperatureFromP(n, w0_(m,IPR,k,j,i), Y);

      const Real p_old = w0_(m,IPR,k,j,i);

      if (pmag > p_old) {
        Kokkos::printf("pmag is too large!\n");
      }

      w0_(m,IPR,k,j,i) = fmax(0.0, w0_(m,IPR,k,j,i) - pmag);

      Real& p_new = w0_(m,IPR,k,j,i);

      if (p_new != p_old) {

        // Since we've adjusted the pressure, we'll also modify the density to compensate.
        // Note that we need to do a root solve of sorts because we don't have a way to
        // directly calculate rho from p in a finite-temperature EOS. Lowering the pressure
        // should lower the density, so the current density is an upper bound. We set the
        // lower bound by assuming (perhaps incorrectly) that P ~ \rho^\gamma. Generally
        // speaking, we cannot have \gamma < 1, so that should give the most conservative
        // bound on \rho. To be safe, we divide by an extra fudge factor of 2. We also
        // perturb the upper bound upward to prevent floating-point precision errors from
        // causing the bracketing to fail.
        Real ub = w0_(m,IDN,k,j,i)*(1 + 1e-10);
        Real lb = w0_(m,IDN,k,j,i)*(w0_(m,IPR,k,j,i)/(2.0*p_old));
        Real rho_sol;

        bool result = root.FalsePosition([&](Real &rho) {
          Real n = rho/ps_eos.GetBaryonMass();
          Real p = ps_eos.GetPressure(n, T, Y);
          return p - p_new;
        }, lb, ub, rho_sol, 1e-15, false);

        //w0_(m,IDN,k,j,i) = eos_.template GetRhoFromP<tov::LocationTag::Device>(w0_(m,IPR,k,j,i));
        mb_max = fmax(mb_max, w0_(m,IDN,k,j,i)/rho_sol - 1.0);
        w0_(m,IDN,k,j,i) = rho_sol;
      }
    }
  }, Kokkos::Max<Real>(rho_frac_max), Kokkos::Max<Real>(bsq_max),
  Kokkos::Max<Real>(betainv_max), Kokkos::Max<Real>(twist_fraction));

#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &rho_frac_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &bsq_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &betainv_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
#endif

  // Estimate the time for flux emergence
  Real r = P_rad;
  Real v = 0.0;
  Real t = 0.0;
  Real dt = 1e-3;
  int count = 0;
  while (r < tov_.R_edge) {
    // Do an RK4 step
    int idx = static_cast<int>(r/tov_.dr);
    Real g = tov::Interpolate(r, tov_.R.h_view(idx), tov_.R.h_view(idx+1),
                            tov_.M.h_view(idx), tov_.M.h_view(idx+1))/(r*r);
    Real dr1 = v;
    Real dv1 = g*rho_frac_max;
    
    Real r2 = r + 0.5*dt*dr1;
    Real v2 = v + 0.5*dt*dv1;
    idx = static_cast<int>(r2/tov_.dr);
    g = tov::Interpolate(r2, tov_.R.h_view(idx), tov_.R.h_view(idx+1),
                        tov_.M.h_view(idx), tov_.M.h_view(idx+1))/(r2*r2);
    Real dr2 = v2;
    Real dv2 = g*rho_frac_max;

    Real r3 = r + 0.5*dt*dr2;
    Real v3 = v + 0.5*dt*dv2;
    idx = static_cast<int>(r3/tov_.dr);
    g = tov::Interpolate(r3, tov_.R.h_view(idx), tov_.R.h_view(idx+1),
                        tov_.M.h_view(idx), tov_.M.h_view(idx+1))/(r3*r3);
    Real dr3 = v3;
    Real dv3 = g*rho_frac_max;

    Real r4 = r + dt*dr3;
    Real v4 = v + dt*dv3;
    idx = static_cast<int>(r4/tov_.dr);
    g = tov::Interpolate(r4, tov_.R.h_view(idx), tov_.R.h_view(idx+1),
                        tov_.M.h_view(idx), tov_.M.h_view(idx+1))/(r4*r4);
    Real dr4 = v4;
    Real dv4 = g*rho_frac_max;

    r = r + dt*(dr1 + 2.0*dr2 + 2.0*dr3 + dr4)/6.0;
    v = v + dt*(dv1 + 2.0*dv2 + 2.0*dv3 + dv4)/6.0;
    t = t + dt;
  }

  if (global_variable::my_rank == 0) {
    int idx = static_cast<int>(P_rad/tov_.dr);
    Real mr0 = tov::Interpolate(P_rad, tov_.R.h_view(idx), tov_.R.h_view(idx+1),
                              tov_.M.h_view(idx), tov_.M.h_view(idx+1));
    Real P0 = tov::Interpolate(P_rad, tov_.R.h_view(idx), tov_.R.h_view(idx+1),
                              tov_.P.h_view(idx), tov_.P.h_view(idx+1));
    Real rho0 = eos_.template GetRhoFromP<tov::LocationTag::Host>(P0);
    // B from geometric units to Gauss
    Real mag_conv = 1.0/mag_cgs_to_geo;
    std::printf("FLUX EMERGENCE\n"
                "--------------\n"
                "bmax = %20.17e = %20.17e G\n"
                "betainv_max = %20.17e\n"
                "drho/rho = %20.17e\n"
                "m(r_0) = %20.17e\n"
                "Necessary twist: %20.17e\n"
                "Current twist: %20.17e\n",
                  sqrt(bsq_max),sqrt(bsq_max)*mag_conv,betainv_max,rho_frac_max,mr0,
                  sqrt(mr0*rho0*SQR(betainv_max)*
                    (SQR(tov_.R_edge/P_rad)-1)/(2.0*P0*P_rad)),twist_fraction);
  }
}

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::UserProblem()
//  \brief Sets initial conditions for TOV star in DynGRMHD
//  Compile with '-D PROBLEM=dyngr_tov' to enroll as user-specific problem generator

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (!pmbp->pcoord->is_dynamical_relativistic) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "TOV star problem can only be run when <adm> block is present"
              << std::endl;
    exit(EXIT_FAILURE);
  }

  // User boundary function
  user_bcs_func = TurbulenceBC;

  // Spherical grid for user-defined history
  int nsurfaces = pin->GetOrAddInteger("problem", "nsurfaces", 1);
  for (int i = 0; i < nsurfaces; i++) {
    std::stringstream ss;
    ss << "flux_surface_" << i;
    tov_turb::flux_loc.push_back(pin->GetOrAddReal("problem", ss.str(), 10.0));
  }
  if (pmbp->pdyngr->eos_policy == DynGRMHD_EOS::eos_ideal) {
    user_hist_func = TurbulenceHistory<tov::PolytropeEOS>;
  } else if (pmbp->pdyngr->eos_policy == DynGRMHD_EOS::eos_compose) {
    user_hist_func = TurbulenceHistory<tov::TabulatedEOS>;
  } else if (pmbp->pdyngr->eos_policy == DynGRMHD_EOS::eos_piecewise_poly) {
    user_hist_func = TurbulenceHistory<tov::PiecewisePolytropeEOS>;
  } 

  if (restart) {
    return;
  }

  // Select the right TOV template based on the EOS we need.
  if (pmbp->pdyngr->eos_policy == DynGRMHD_EOS::eos_ideal) {
    SetupTOV<tov::PolytropeEOS>(pin, pmy_mesh_);
  } else if (pmbp->pdyngr->eos_policy == DynGRMHD_EOS::eos_compose) {
    SetupTOV<tov::TabulatedEOS>(pin, pmy_mesh_);
  } else if (pmbp->pdyngr->eos_policy == DynGRMHD_EOS::eos_piecewise_poly) {
    SetupTOV<tov::PiecewisePolytropeEOS>(pin, pmy_mesh_);
  } else {
    std::cout << "### WARNING in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Unknown EOS requested for TOV star problem" << std::endl
              << "Defaulting to fixed polytropic EOS" << std::endl;
    SetupTOV<tov::PolytropeEOS>(pin, pmy_mesh_);
  }

  // Mesh block info for loop limits
  auto &indcs = pmy_mesh_->mb_indcs;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*ng) : 1;

  // Convert primitives to conserved
  pmbp->pdyngr->PrimToConInit(0, (n1-1), 0, (n2-1), 0, (n3-1));

  if (pmbp->pz4c != nullptr) {
    switch (indcs.ng) {
      case 2: pmbp->pz4c->ADMToZ4c<2>(pmbp, pin);
              pmbp->pz4c->ADMConstraints<2>(pmbp);
              break;
      case 3: pmbp->pz4c->ADMToZ4c<3>(pmbp, pin);
              pmbp->pz4c->ADMConstraints<3>(pmbp);
              break;
      case 4: pmbp->pz4c->ADMToZ4c<4>(pmbp, pin);
              pmbp->pz4c->ADMConstraints<4>(pmbp);
              break;
    }
  }

  return;
}

// The magnetic field is an untwisted flux tube which forms a torus around \theta = Pi/2.
// We define P as the radius of the torus, \rho as the inner radius inside the tube,
// \alpha as the azimuthal angle around the torus, and \beta as the angle inside the tube.
// In spherical coordinates, we may define these as
// \rho^2 = r^2 + P^2 - 2Pr sin(\theta),
// \alpha = \phi
// tan(\beta) = r cos(\theta)/(r sin(\theta) - P)
//
// In toroidal coordinates, the vector potential has two non-zero components,
// A_\rho = \beta\sigma tan^-1(\rho/\sigma),
// A_\phi = 1/2 \sigma^2 log(1 + (\rho/\sigma)^2),
// where \sigma determines the rate of falloff.
// Because \alpha = \phi and A_\beta = 0, we only need three components of the Jacobian
// to convert to spherical coordinates:
// d\rho/dr = (r - P sin\theta)/\rho
// d\rho/r\theta = (-rP cos\theta)/\rho
// d\alpha/d\phi = 1
//
// For \rho > \rho_0, we want B^i = 0, so we add a constant to the vector potential to
// make it continuous. While not analytically necessary, it does help when we compute
// finite differences.

/*KOKKOS_INLINE_FUNCTION
static Real Ar(Real x1, Real x2, Real P_rad, Real tube_rad, Real sigma) {
  Real rho = x1*x1 + P_rad*P_rad - 2.0*P_rad*x1*sin(x2);
  Real beta = atan2(x1*cos(x2),x1*sin(x2) - P_rad);
  if (rho <= tube_rad) {
    if (rho > 0) {
      return (x1 - tube_rad*sin(x2))*beta*exp(-(rho*rho)/(sigma*sigma));
    } else {
      return 0.0;
    }
  }
  return 0.0;
}

KOKKOS_INLINE_FUNCTION
static Real Atheta(Real x1, Real x2, Real P_rad, Real tube_rad, Real sigma) {
  Real rho = x1*x1 + P_rad*P_rad - 2.0*P_rad*x1*sin(x2);
  Real beta = atan2(x1*cos(x2),x1*sin(x2) - P_rad);
  if (rho <= tube_rad) {
    if (rho > 0) {
      return - x1*P_rad*cos(x2)*beta*exp(-(rho*rho)/(sigma*sigma));
    } else {
      return 0.0;
    }
  }
  return 0.0;
}*/

KOKKOS_INLINE_FUNCTION
static Real Aphi(Real x1, Real x2, Real P_rad, Real tube_rad, Real sigma) {
  Real rhosq = x1*x1 + P_rad*P_rad - 2.0*P_rad*x1*sin(x2);
  if (rhosq < tube_rad*tube_rad) {
    return -0.5*sigma*sigma*exp(-(rhosq)/(sigma*sigma));
  }
  return -0.5*sigma*sigma*exp(-(tube_rad*tube_rad)/(sigma*sigma));
}

// Boundary function
// This BC only applies to the xmin boundary. The primitive variables are fixed, and the
// magnetic field is allowed to flow freely. The logic behind this is that the star
// itself should be fixed, and the flux tube should only exist inside the region we're
// modeling.
void TurbulenceBC(Mesh *pm) {
  auto &indcs = pm->mb_indcs;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng) : 1;
  int &is = indcs.is;  int &ie  = indcs.ie;
  int &js = indcs.js;  int &je  = indcs.je;
  int &ks = indcs.ks;  int &ke  = indcs.ke;
  auto &mb_bcs = pm->pmb_pack->pmb->mb_bcs;

  DvceArray5D<Real> u0_, w0_;
  u0_ = pm->pmb_pack->pmhd->u0;
  w0_ = pm->pmb_pack->pmhd->w0;
  auto &adm = pm->pmb_pack->padm->adm;
  auto &b0 = pm->pmb_pack->pmhd->b0;
  auto &bcc = pm->pmb_pack->pmhd->bcc0;
  int nmb = pm->pmb_pack->nmb_thispack;

  // This BC should *only* be applied to the X1 boundary. Throw an error if this isn't
  // the case.
  if (pm->mesh_bcs[BoundaryFace::outer_x1] == BoundaryFlag::user ||
      pm->mesh_bcs[BoundaryFace::inner_x2] == BoundaryFlag::user ||
      pm->mesh_bcs[BoundaryFace::outer_x2] == BoundaryFlag::user ||
      pm->mesh_bcs[BoundaryFace::inner_x3] == BoundaryFlag::user ||
      pm->mesh_bcs[BoundaryFace::outer_x3] == BoundaryFlag::user) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Flux emergence user BC should only be applied to inner_x1!"
              << std::endl;
    exit(EXIT_FAILURE);
  }

  // X1-Boundary
  // Perform a C2P operation to populate the primitives in the boundary region. We do this
  // before modifying the B-field to prevent changing the primitives, which should be
  // fixed.
  //pm->pmb_pack->pdyngr->ConToPrimBC(is-ng,is,0,(n2-1),0,(n3-1));
  // Set X1-BCs on b0 if Meshblock face is at the edge of the computational domain.
  par_for("noinflow_x1", DevExeSpace(),0,(nmb-1),0,(n3-1),0,(n2-1),
  KOKKOS_LAMBDA(int m, int k, int j) {
    if (mb_bcs.d_view(m,BoundaryFace::inner_x1) == BoundaryFlag::user) {
      for (int i=0; i<ng; ++i) {
        b0.x1f(m,k,j,is-i-1) = -b0.x1f(m,k,j,is+i+1);
        b0.x2f(m,k,j,is-i-1) = -b0.x2f(m,k,j,is+i);
        if (j == n2-1) {b0.x2f(m,k,j+1,is-i-1) = -b0.x2f(m,k,j+1,is+i);}
        b0.x3f(m,k,j,is-i-1) = -b0.x3f(m,k,j,is+i);
        if (k == n3-1) {b0.x3f(m,k+1,j,is-i-1) = -b0.x3f(m,k+1,j,is+i);}
        /*u0_(m, IDN, k, j, is-i-1) = dfloor;
        u0_(m, IM1, k, j, is-i-1) = 0.0;
        u0_(m, IM2, k, j, is-i-1) = 0.0;
        u0_(m, IM3, k, j, is-i-1) = 0.0;
        u0_(m, IEN, k, j, is-i-1) = 0.0;*/
      }
    }
  });
  // We also need to update the cell-centered fields along x1.
  par_for("noinflow_x1_bcc", DevExeSpace(),0,(nmb-1),0,(n3-1),0,(n2-1),
  KOKKOS_LAMBDA(int m, int k, int j) {
    if (mb_bcs.d_view(m,BoundaryFace::inner_x1) == BoundaryFlag::user) {
      for (int i = 0; i < ng; ++i) {
        bcc(m,IBX,k,j,is-i-1) = 0.5*(b0.x1f(m,k,j,is-i-1) + b0.x1f(m,k,j,is-i));
        bcc(m,IBY,k,j,is-i-1) = 0.5*(b0.x2f(m,k,j,is-i-1) + b0.x2f(m,k,j+1,is-i-1));
        bcc(m,IBZ,k,j,is-i-1) = 0.5*(b0.x3f(m,k,j,is-i-1) + b0.x3f(m,k+1,j,is-i-1));
        w0_(m,IVX,k,j,is-i-1) = -w0_(m,IVX,k,j,is+i)*
                                (adm.alpha(m,k,j,is+i)*u0_(m,IDN,k,j,is+i)/
                                 (adm.alpha(m,k,j,is-i-1)*u0_(m,IDN,k,j,is-i-1)));
        //w0_(m,IVX,k,j,is-i-1) = fmax(0,w0_(m,IVX,k,j,is));
        //w0_(m,IVX,k,j,is-i-1) = fmax(0,-w0_(m,IVX,k,j,is+i));
        w0_(m,IVY,k,j,is-i-1) = -w0_(m,IVY,k,j,is+i);
        w0_(m,IVZ,k,j,is-i-1) = -w0_(m,IVZ,k,j,is+i);
        /*w0_(m,IVY,k,j,is-i-1) = 0.0;
        w0_(m,IVZ,k,j,is-i-1) = 0.0;*/
        /*w0_(m,IVX,k,j,is-i-1) = w0_(m,IVX,k,j,is);
        w0_(m,IVY,k,j,is-i-1) = w0_(m,IVY,k,j,is);
        w0_(m,IVZ,k,j,is-i-1) = w0_(m,IVZ,k,j,is);*/
        /*u0_(m,IM1,k,j,is-i-1) = -u0_(m,IM1,k,j,is+i);
        u0_(m,IM2,k,j,is-i-1) = u0_(m,IM2,k,j,is+i);
        u0_(m,IM3,k,j,is-i-1) = u0_(m,IM3,k,j,is+i);*/
      }
    }
  });
  // Because the magnetic field has changed, the conserved variables need to be updated.
  // Otherwise we'll get junk in the primitives.
  //pm->pmb_pack->pdyngr->PrimToConInit(is-ng, is, 0, (n2-1), 0, (n3-1));
}

// History function
template<class TOVEOS>
void TurbulenceHistory(HistoryData *pdata, Mesh *pm) {
  MeshBlockPack *pmbp = pm->pmb_pack;

  // Select the number of outputs and create labels for them.
  int& nvars_ = pmbp->pmhd->nmhd;
  int& nscal_ = pmbp->pmhd->nscalars;
  pdata->nhist = 6*tov_turb::flux_loc.size();

  // capture class variables for kernel
  auto &w0_ = pmbp->pmhd->w0;
  auto &adm = pmbp->padm->adm;
  auto &bcc0_ = pmbp->pmhd->bcc0;

  auto &size = pmbp->pmb->mb_size;

  // Dirty, nasty hack to get the EOS
  Primitive::EOS<typename conv_eos_type<TOVEOS>::type, Primitive::ResetFloor>& ps_eos =
    static_cast<
      dyngr::DynGRMHDPS<
        typename conv_eos_type<TOVEOS>::type,
        Primitive::ResetFloor
      >*
    >(pmbp->pdyngr)->eos.ps.GetEOSMutable();
  constexpr bool use_ye = tov::UsesYe<TOVEOS>;

  // loop over all MeshBlocks in this pack
  auto &indcs = pm->pmb_pack->pmesh->mb_indcs;
  int is = indcs.is; int nx1 = indcs.nx1;
  int js = indcs.js; int nx2 = indcs.nx2;
  int ks = indcs.ks; int nx3 = indcs.nx3;
  const int nmkji = (pm->pmb_pack->nmb_thispack)*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1;
  const int nji = nx2*nx1;
  for (int i = 0; i < tov_turb::flux_loc.size(); i++) {
    std::stringstream ss;
    ss << "mdot_" << i;
    pdata->label[6*i] = ss.str();

    ss.str("");
    ss << "mdotgeo_" << i;
    pdata->label[6*i + 1] = ss.str();

    ss.str("");
    ss << "mdotber_" << i;
    pdata->label[6*i + 2] = ss.str();

    ss.str("");
    ss << "pntg_" << i;
    pdata->label[6*i + 3] = ss.str();

    ss.str("");
    ss << "pntggeo_" << i;
    pdata->label[6*i + 4] = ss.str();

    ss.str("");
    ss << "pntgber_" << i;
    pdata->label[6*i + 5] = ss.str();

    pdata->hdata[6*i] = 0.0;
    pdata->hdata[6*i + 1] = 0.0;
    pdata->hdata[6*i + 2] = 0.0;
    pdata->hdata[6*i + 3] = 0.0;
    pdata->hdata[6*i + 4] = 0.0;
    pdata->hdata[6*i + 5] = 0.0;
    Real r = tov_turb::flux_loc[i];
    Real mdot = 0;
    Real mdotgeo = 0;
    Real mdotber = 0;
    Real pntg = 0;
    Real pntggeo = 0;
    Real pntgber = 0;
    Kokkos::parallel_reduce("TOVHistSums",Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
    KOKKOS_LAMBDA(const int &idx, Real &mb_mdot, Real &mb_mdotgeo, Real &mb_mdotber,
    Real &mb_pntg, Real &mb_pntggeo, Real &mb_pntgber) {
      // coompute n,k,j,i indices of thread
      int m = (idx)/nkji;
      int k = (idx - m*nkji)/nji;
      int j = (idx - m*nkji - k*nji)/nx1;
      int i = (idx - m*nkji - k*nji - j*nx1) + is;
      k += ks;
      j += js;

      // Compute the coordinates of this cell and the neighboring cell in radius.
      Real &x1min = size.d_view(m).x1min;
      Real &x1max = size.d_view(m).x1max;
      Real rm = CellCenterX(i-is, indcs.nx1, x1min, x1max);
      Real rp = CellCenterX(i+1-is, indcs.nx1, x1min, x1max);

      Real &dtheta = size.d_view(m).dx2;
      Real &dphi   = size.d_view(m).dx3;

      // If the integration surface is inside the radius, interpolate to get the primitives,
      // the metric, and the magnetic field.
      if (rm <= r && rp >= r) {
        Real rho = tov::Interpolate(r, rm, rp, w0_(m,IDN,k,j,i), w0_(m,IDN,k,j,i+1));
        Real ur  = tov::Interpolate(r, rm, rp, w0_(m,IVX,k,j,i), w0_(m,IVX,k,j,i+1));
        Real uth = tov::Interpolate(r, rm, rp, w0_(m,IVY,k,j,i), w0_(m,IVY,k,j,i+1));
        Real uph = tov::Interpolate(r, rm, rp, w0_(m,IVZ,k,j,i), w0_(m,IVZ,k,j,i+1));
        Real p   = tov::Interpolate(r, rm, rp, w0_(m,IPR,k,j,i), w0_(m,IPR,k,j,i+1));
        Real Y[MAX_SPECIES] = {0.0};
        for (int s = 0; s < nscal_; s++) {
          Y[s] = tov::Interpolate(r, rm, rp, w0_(m,nvars_+s,k,j,i),
                                             w0_(m,nvars_+s,k,j,i+1));
        }

        Real Br  = tov::Interpolate(r, rm, rp, bcc0_(m,IBX,k,j,i), bcc0_(m,IBX,k,j,i+1));
        Real Bth = tov::Interpolate(r, rm, rp, bcc0_(m,IBY,k,j,i), bcc0_(m,IBY,k,j,i+1));
        Real Bph = tov::Interpolate(r, rm, rp, bcc0_(m,IBZ,k,j,i), bcc0_(m,IBZ,k,j,i+1));

        Real gxx = tov::Interpolate(r, rm, rp, adm.g_dd(m,0,0,k,j,i),
                                    adm.g_dd(m,0,0,k,j,i+1));
        Real gyy = tov::Interpolate(r, rm, rp, adm.g_dd(m,1,1,k,j,i),
                                    adm.g_dd(m,1,1,k,j,i+1));
        Real gzz = tov::Interpolate(r, rm, rp, adm.g_dd(m,2,2,k,j,i),
                                    adm.g_dd(m,2,2,k,j,i+1));
        Real alp = tov::Interpolate(r, rm, rp, adm.alpha(m,k,j,i), adm.alpha(m,k,j,i+1));

        // Integration params
        Real domega = dtheta*dphi;
        Real sqrtmdet = Kokkos::sqrt(gxx*gyy*gzz);

        // Get u0 and the r, t components of b. Take care to undensitize b first.
        Real u_0 = -Kokkos::sqrt(gxx*ur*ur + gyy*uth*uth + gzz*uph*uph + 1.0)*alp;
        //Real b0 = (Br*ur + Bth*uth + Bph*uph)/(sqrtmdet*alp);
        //Real br = (Br/sqrtmdet + alp*b0*ur)/(alp*u0);

        // Compute the enthalpy
        Real n = rho/ps_eos.GetBaryonMass();
        Real T = ps_eos.GetTemperatureFromP(n, p, Y);
        Real e = ps_eos.GetEnergy(n, T, Y);
        Real h = (e + p)/rho;

        // Compute (outward) mass flux
        mb_mdot += rho*ur*sqrtmdet*alp*domega;

        // Compute Poynting flux
        Real u0 = u_0/(alp*alp);
        Real u_r = ur*gxx;
        Real b0 = (Br*ur*gxx + Bth*uth*gyy + Bph*uph*gzz)/(sqrtmdet*alp);
        Real bsq = (Br*Br*gxx + Bth*Bth*gyy + Bph*Bph*gzz)/(sqrtmdet*sqrtmdet) +
                   alp*alp*b0*b0;
        Real b_r = (Br*gxx/sqrtmdet + alp*b0*u_r)/(alp*u0);
        mb_pntg += (bsq*u0*u_r - b0*b_r)*sqrtmdet*alp*domega;

        // Compute outward mass flux assuming geodesic criterion.
        if (u_0 < -1 && ur > 0) {
          mb_mdotgeo += rho*ur*sqrtmdet*alp*domega;
          mb_pntggeo += (bsq*u0*u_r - b0*b_r)*sqrtmdet*alp*domega;
        }

        // Compute outward mass flux assuming Bernoulli criterion
        if (h*u_0 < -1 && ur > 0) {
          mb_mdotber += rho*ur*sqrtmdet*alp*domega;
          mb_pntgber += (bsq*u0*u_r - b0*b_r)*sqrtmdet*alp*domega;
        }

      }
    }, Kokkos::Sum<Real>(mdot), Kokkos::Sum<Real>(mdotgeo), Kokkos::Sum<Real>(mdotber),
    Kokkos::Sum<Real>(pntg),Kokkos::Sum<Real>(pntggeo), Kokkos::Sum<Real>(pntgber));

    // store data in hdata array
    pdata->hdata[6*i] = mdot;
    pdata->hdata[6*i+1] = mdotgeo;
    pdata->hdata[6*i+2] = mdotber;
    pdata->hdata[6*i+3] = pntg;
    pdata->hdata[6*i+4] = pntggeo;
    pdata->hdata[6*i+5] = pntgber;
  }
}
