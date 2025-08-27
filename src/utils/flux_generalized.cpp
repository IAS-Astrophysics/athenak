//========================================================================================
// flux_generalized.cpp — integrate fluxes on arbitrary r(θ,φ) surfaces (Valencia MHD)
//----------------------------------------------------------------------------------------
#include "flux_generalized.hpp"
#include <iomanip>
#include <sstream>
#include <iostream>

// Add all necessary full definitions
#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "mhd/mhd.hpp"
#include "eos/eos.hpp" // ---- FIX: Added to provide full definition of EquationOfState
#include "coordinates/adm.hpp"
#include "utils/surface_grid.hpp"
#include "outputs/outputs.hpp"

// ---- Helpers for ADM on surface -------------------------------------------------------
struct ADMOnSurface {
  DualArray1D<Real> alpha;        // (p)
  DualArray2D<Real> beta_u;       // (p,3)
  DualArray2D<Real> g_dd;         // (p,6): xx,xy,xz,yy,yz,zz
};

KOKKOS_INLINE_FUNCTION
Real det3(Real gxx, Real gxy, Real gxz, Real gyy, Real gyz, Real gzz) {
  return adm::SpatialDet(gxx,gxy,gxz,gyy,gyz,gzz);
}

KOKKOS_INLINE_FUNCTION
void inv3(Real det,
          Real gxx, Real gxy, Real gxz, Real gyy, Real gyz, Real gzz,
          Real &uxx, Real &uxy, Real &uxz, Real &uyy, Real &uyz, Real &uzz) {
  adm::SpatialInv(1.0/det, gxx,gxy,gxz,gyy,gyz,gzz, &uxx,&uxy,&uxz,&uyy,&uyz,&uzz);
}

// ---- FIX: Moved this static helper function BEFORE TorusFluxes_General ----
static void InterpolateADMToSurface(MeshBlockPack* pmbp, SphericalSurfaceGrid* surf,
                                    ADMOnSurface &out) {
  if (pmbp->padm == nullptr) {
    std::cerr << "### FATAL: ADM object (padm) is null; cannot interpolate metric" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  auto &admv = pmbp->padm->adm;
  auto &indcs = pmbp->pmesh->mb_indcs;
  int &is = indcs.is; int &js = indcs.js; int &ks = indcs.ks; int &ng = indcs.ng;

  int np = surf->Npts();
  Kokkos::realloc(out.alpha, np);
  Kokkos::realloc(out.beta_u, np, 3);
  Kokkos::realloc(out.g_dd,   np, 6);

  auto &iindcs = surf->InterpIndices();
  auto &iwghts = surf->InterpWeights();

  par_for("adm2surf", DevExeSpace(), 0, np-1, KOKKOS_LAMBDA(const int p){
    int ii0 = iindcs.d_view(p,0);
    int ii1 = iindcs.d_view(p,1);
    int ii2 = iindcs.d_view(p,2);
    int ii3 = iindcs.d_view(p,3);

    Real a=0.0, bx=0.0, by=0.0, bz=0.0;
    Real gxx=0.0, gxy=0.0, gxz=0.0, gyy=0.0, gyz=0.0, gzz=0.0;

    if (ii0!=-1) {
      for (int k=0; k<2*ng; ++k) for (int j=0; j<2*ng; ++j) for (int i=0; i<2*ng; ++i) {
        Real w = iwghts.d_view(p,i,0) * iwghts.d_view(p,j,1) * iwghts.d_view(p,k,2);
        int I = ii1 - ng + i + 1 - is;
        int J = ii2 - ng + j + 1 - js;
        int K = ii3 - ng + k + 1 - ks;
        a  += w * admv.alpha(ii0,K,J,I);
        bx += w * admv.beta_u(ii0,0,K,J,I);
        by += w * admv.beta_u(ii0,1,K,J,I);
        bz += w * admv.beta_u(ii0,2,K,J,I);
        gxx+= w * admv.g_dd(ii0,0,0,K,J,I);
        gxy+= w * admv.g_dd(ii0,0,1,K,J,I);
        gxz+= w * admv.g_dd(ii0,0,2,K,J,I);
        gyy+= w * admv.g_dd(ii0,1,1,K,J,I);
        gyz+= w * admv.g_dd(ii0,1,2,K,J,I);
        gzz+= w * admv.g_dd(ii0,2,2,K,J,I);
      }
    }
    out.alpha.d_view(p)      = a;
    out.beta_u.d_view(p,0)   = bx;
    out.beta_u.d_view(p,1)   = by;
    out.beta_u.d_view(p,2)   = bz;
    out.g_dd.d_view(p,0)     = gxx;
    out.g_dd.d_view(p,1)     = gxy;
    out.g_dd.d_view(p,2)     = gxz;
    out.g_dd.d_view(p,3)     = gyy;
    out.g_dd.d_view(p,4)     = gyz;
    out.g_dd.d_view(p,5)     = gzz;
  });

  out.alpha.template modify<DevExeSpace>();
  out.beta_u.template modify<DevExeSpace>();
  out.g_dd.template modify<DevExeSpace>();
}

// ---- FIX: Moved this static helper function BEFORE TorusFluxes_General ----
static void BuildSurfaceGeom(const SphericalSurfaceGrid* surf,
                             const ADMOnSurface& A,
                             DualArray1D<Real>& sqrtg,
                             DualArray2D<Real>& dSigma) {
  int np = surf->Npts();
  Kokkos::realloc(sqrtg, np);

  par_for("build_sqrtg", DevExeSpace(), 0, np-1, KOKKOS_LAMBDA(const int p){
    Real gxx=A.g_dd.d_view(p,0), gxy=A.g_dd.d_view(p,1), gxz=A.g_dd.d_view(p,2),
         gyy=A.g_dd.d_view(p,3), gyz=A.g_dd.d_view(p,4), gzz=A.g_dd.d_view(p,5);
    Real det = det3(gxx,gxy,gxz,gyy,gyz,gzz);
    if (!(det >= 0.0) || !isfinite(det)) { det = 0.0; } // Clamp if bad
    sqrtg.d_view(p) = sqrt(det);
  });
  sqrtg.template modify<DevExeSpace>();

  // dΣ_i = √γ ε_ijk e_θ^j e_φ^k ΔθΔφ
  surf->BuildSurfaceCovectors(sqrtg, dSigma);
}


struct FluxAccumulator {
  Real mdot=0, edot=0, ldot=0, phiB=0;

  KOKKOS_INLINE_FUNCTION
  FluxAccumulator& operator+=(const FluxAccumulator& src) {
    mdot += src.mdot;
    edot += src.edot;
    ldot += src.ldot;
    phiB += src.phiB;
    return *this;
  }
};

namespace Kokkos {
  template<>
  struct reduction_identity<FluxAccumulator> {
    KOKKOS_INLINE_FUNCTION static FluxAccumulator sum() {
      return FluxAccumulator{};
    }
  };
} // namespace Kokkos


// --- Main Function Definition ---
void TorusFluxes_General(HistoryData *pdata,
                         MeshBlockPack *pmbp,
                         const std::vector<SphericalSurfaceGrid*>& surfs,
                         const Real axis_n[3]) {
  if (pmbp == nullptr) {
    std::cerr << "### FATAL: MeshBlockPack is null" << std::endl; std::exit(EXIT_FAILURE);
  }

  // --- MHD access ---
  if (pmbp->pmhd == nullptr) {
    std::cerr << "### FATAL: TorusFluxes_General requires a valid MHD object pointer" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  int nvars_mhd = pmbp->pmhd->nmhd + pmbp->pmhd->nscalars;
  Real gamma_ad = pmbp->pmhd->peos->eos_data.gamma;
  auto w0_ = pmbp->pmhd->w0;
  auto bcc0_ = pmbp->pmhd->bcc0;

  Real to_ien = (pmbp->pdyngr ? 1.0/(gamma_ad-1.0) : 1.0);

  if (surfs.size() == 0) {
    pdata->nhist = 0; return;
  }
  int nsurf = surfs.size();
  const int nflux = 4;
  pdata->nhist = nsurf * nflux;
  if (pdata->nhist > NHISTORY_VARIABLES) {
    std::cerr << "### FATAL: pdata->nhist exceeds NHISTORY_VARIABLES" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  for (int g=0; g<nsurf; ++g) {
    pdata->label[nflux*g+0] = "mdot_" + surfs[g]->Label();
    pdata->label[nflux*g+1] = "edot_" + surfs[g]->Label();
    pdata->label[nflux*g+2] = "ldot_" + surfs[g]->Label();
    pdata->label[nflux*g+3] = "phiB_" + surfs[g]->Label();
  }

  Real n_ax[3] = {0.0,0.0,1.0};
  if (axis_n) { n_ax[0]=axis_n[0]; n_ax[1]=axis_n[1]; n_ax[2]=axis_n[2]; }

  DualArray1D<Real> sqrtg;
  DualArray2D<Real> dSigma;

  for (int g=0; g<nsurf; ++g) {
    auto current_surf = surfs[g];
    int np = current_surf->Npts();

    DualArray2D<Real> B_vals(Kokkos::ViewAllocateWithoutInitializing("B_vals"), np, 3);
    current_surf->InterpolateToSurface(3, bcc0_);
    Kokkos::deep_copy(B_vals.d_view, Kokkos::subview(current_surf->InterpVals().d_view, Kokkos::ALL(), Kokkos::make_pair(0,3)));

    current_surf->InterpolateToSurface(nvars_mhd, w0_);
    auto w_vals = current_surf->InterpVals();

    ADMOnSurface A;
    InterpolateADMToSurface(pmbp, current_surf, A);
    BuildSurfaceGeom(current_surf, A, sqrtg, dSigma);

    FluxAccumulator total_fluxes;
    Kokkos::RangePolicy<DevExeSpace> policy(0, np);
    Kokkos::parallel_reduce("flux_integral", policy,
      KOKKOS_LAMBDA(const int p, FluxAccumulator& update) {
        Real rho = w_vals.d_view(p,IDN);
        Real vx  = w_vals.d_view(p,IVX);
        Real vy  = w_vals.d_view(p,IVY);
        Real vz  = w_vals.d_view(p,IVZ);
        Real P   = w_vals.d_view(p,IPR);
        Real ie  = P * to_ien;

        Real Bx = B_vals.d_view(p,0);
        Real By = B_vals.d_view(p,1);
        Real Bz = B_vals.d_view(p,2);

        Real alp = A.alpha.d_view(p);
        Real betx=A.beta_u.d_view(p,0), bety=A.beta_u.d_view(p,1), betz=A.beta_u.d_view(p,2);
        Real gxx=A.g_dd.d_view(p,0), gxy=A.g_dd.d_view(p,1), gxz=A.g_dd.d_view(p,2),
             gyy=A.g_dd.d_view(p,3), gyz=A.g_dd.d_view(p,4), gzz=A.g_dd.d_view(p,5);

        Real betx_d = gxx*betx + gxy*bety + gxz*betz;
        Real bety_d = gxy*betx + gyy*bety + gyz*betz;
        Real betz_d = gxz*betx + gyz*bety + gzz*betz;

        Real v2 = gxx*vx*vx + 2.0*gxy*vx*vy + 2.0*gxz*vx*vz + gyy*vy*vy + 2.0*gyz*vy*vz + gzz*vz*vz;
        Real one_minus_v2 = 1.0 - v2; if (one_minus_v2 < 1e-16) one_minus_v2 = 1e-16;
        Real W  = 1.0/sqrt(one_minus_v2);
        Real u0 = W/alp;
        Real u1 = W*(vx - betx);
        Real u2 = W*(vy - bety);
        Real u3 = W*(vz - betz);

        Real u_0 = W*(-alp + (betx_d*vx + bety_d*vy + betz_d*vz));
        Real u_1 = W*(gxx*vx + gxy*vy + gxz*vz);
        Real u_2 = W*(gxy*vx + gyy*vy + gyz*vz);
        Real u_3 = W*(gxz*vx + gyz*vy + gzz*vz);

        Real Bdotv = (gxx*Bx + gxy*By + gxz*Bz)*vx + (gxy*Bx + gyy*By + gyz*Bz)*vy + (gxz*Bx + gyz*By + gzz*Bz)*vz;
        Real b0 = W * Bdotv;
        Real b1 = (Bx/W) + b0 * u1;
        Real b2 = (By/W) + b0 * u2;
        Real b3 = (Bz/W) + b0 * u3;

        Real gL[16]; adm::SpacetimeMetric(alp,betx,bety,betz,gxx,gxy,gxz,gyy,gyz,gzz,gL);
        Real b_0 = gL[0]*b0 + gL[1]*b1 + gL[2]*b2 + gL[3]*b3;
        Real b_1 = gL[4]*b0 + gL[5]*b1 + gL[6]*b2 + gL[7]*b3;
        Real b_2 = gL[8]*b0 + gL[9]*b1 + gL[10]*b2 + gL[11]*b3;
        Real b_3 = gL[12]*b0+ gL[13]*b1+ gL[14]*b2+ gL[15]*b3;
        Real b2sq = b0*b_0 + b1*b_1 + b2*b_2 + b3*b_3;

        Real dSx = dSigma.d_view(p,0), dSy = dSigma.d_view(p,1), dSz = dSigma.d_view(p,2);

        update.mdot += -(rho*u1*dSx + rho*u2*dSy + rho*u3*dSz);

        Real rho_h = rho + ie + P;
        Real wtot = rho_h + b2sq;
        Real Ti0_1 = wtot*u1*u_0 - b1*b_0;
        Real Ti0_2 = wtot*u2*u_0 - b2*b_0;
        Real Ti0_3 = wtot*u3*u_0 - b3*b_0;
        update.edot += -(Ti0_1*dSx + Ti0_2*dSy + Ti0_3*dSz);

        Real x = current_surf->Coords().d_view(p,0);
        Real y = current_surf->Coords().d_view(p,1);
        Real z = current_surf->Coords().d_view(p,2);
        Real nx=n_ax[0], ny=n_ax[1], nz=n_ax[2];
        Real phx = ny*z - nz*y;
        Real phy = nz*x - nx*z;
        Real phz = nx*y - ny*x;
        Real p_tot = P + 0.5 * b2sq;
        Real Tij_phj_1 = (wtot*u1*u_1 - b1*b_1)*phx + (wtot*u1*u_2 - b1*b_2)*phy + (wtot*u1*u_3 - b1*b_3)*phz + p_tot*phx;
        Real Tij_phj_2 = (wtot*u2*u_1 - b2*b_1)*phx + (wtot*u2*u_2 - b2*b_2)*phy + (wtot*u2*u_3 - b2*b_3)*phz + p_tot*phy;
        Real Tij_phj_3 = (wtot*u3*u_1 - b3*b_1)*phx + (wtot*u3*u_2 - b3*b_2)*phy + (wtot*u3*u_3 - b3*b_3)*phz + p_tot*phz;
        update.ldot += (Tij_phj_1*dSx + Tij_phj_2*dSy + Tij_phj_3*dSz);

        update.phiB += Bx*dSx + By*dSy + Bz*dSz;
    }, Kokkos::Sum<FluxAccumulator>(total_fluxes));

    pdata->hdata[nflux*g+0] = total_fluxes.mdot;
    pdata->hdata[nflux*g+1] = total_fluxes.edot;
    pdata->hdata[nflux*g+2] = total_fluxes.ldot;
    pdata->hdata[nflux*g+3] = total_fluxes.phiB;
  }

  for (int n=pdata->nhist; n<NHISTORY_VARIABLES; ++n) pdata->hdata[n] = 0.0;
}