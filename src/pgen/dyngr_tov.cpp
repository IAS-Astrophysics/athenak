//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dyngr_tov.cpp
//  \brief Problem generator for TOV star. Only works when ADM is enabled.

#include <math.h>     // abs(), cos(), exp(), log(), NAN, pow(), sin(), sqrt()

#include <algorithm>  // max(), max_element(), min(), min_element()
#include <iostream>   // endl
#include <limits>     // numeric_limits::max()
#include <sstream>    // stringstream
#include <string>     // c_str(), string
#include <cfloat>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/adm.hpp"
#include "z4c/z4c.hpp"
#include "coordinates/coordinates.hpp"
#include "coordinates/cell_locations.hpp"
#include "eos/eos.hpp"
#include "eos/primitive-solver/eos.hpp"
#include "eos/primitive-solver/piecewise_polytrope.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"
#include "utils/tr_table.hpp"

#include <Kokkos_Random.hpp>


// Useful container for physical parameters of star
struct tov_pgen {
  Real rhoc;
  Real gamma;
  Real dfloor;
  Real pfloor;

  Real v_pert; // Amplitude of radial velocity perturbation, v^r = U/2(3x - x^3), x = r/R
  Real p_pert; // Amplitude of random pressure perturbations

  Real b_norm;
  Real pcut;
  int magindex;

  int npoints; // Number of points in arrays
  Real dr; // Radial spacing for integration
  DualArray1D<Real> R; // Array of radial coordinates
  DualArray1D<Real> R_iso; // Array of isotropic radial coordinates
  DualArray1D<Real> M; // Integrated mass, M(r)
  DualArray1D<Real> P; // Pressure, P(r)
  DualArray1D<Real> alp; // Lapse, \alpha(r)

  Real R_edge; // Radius of star
  Real R_edge_iso; // Radius of star in isotropic coordinates
  Real M_edge; // Mass of star
  int n_r; // Point where pressure goes to zero.

  bool isotropic; // Whether or not the TOV uses isotropic coordinates.
};

// Prototypes for functions used internally in this pgen.
template<class TOVEOS>
static void ConstructTOV(tov_pgen& pgen, TOVEOS& eos);
template<class TOVEOS>
static void RHS(Real r, Real P, Real m, Real alp, Real R, TOVEOS& eos,
                tov_pgen& tov, Real& dP, Real& dm, Real& dalp, Real& dR);
KOKKOS_INLINE_FUNCTION
static Real FindSchwarzschildR(const tov_pgen& pgen, Real r_iso, Real mass);
KOKKOS_INLINE_FUNCTION
static int FindIsotropicIndex(const tov_pgen& pgen, Real r_iso);
template<class TOVEOS>
KOKKOS_INLINE_FUNCTION
static void GetPrimitivesAtPoint(const tov_pgen& pgen, const TOVEOS& eos, Real r,
                                 Real &rho, Real &p, Real &m, Real &alp);
template<class TOVEOS>
KOKKOS_INLINE_FUNCTION
static void GetPrimitivesAtIsoPoint(const tov_pgen& pgen, const TOVEOS& eos, Real r_iso,
                                 Real &rho, Real &p, Real &m, Real &alp);
template<class TOVEOS>
KOKKOS_INLINE_FUNCTION
static void GetPandRho(const tov_pgen& pgen, const TOVEOS& eos,
                       Real r, Real &rho, Real &p);
template<class TOVEOS>
KOKKOS_INLINE_FUNCTION
static void GetPandRhoIso(const tov_pgen& pgen, const TOVEOS& eos,
                          Real r, Real &rho, Real &p);
KOKKOS_INLINE_FUNCTION
static Real Interpolate(Real x,
                        const Real x1, const Real x2, const Real y1, const Real y2);
template<class TOVEOS>
KOKKOS_INLINE_FUNCTION
static Real A1(const tov_pgen& pgen, const TOVEOS& eos, Real x1, Real x2, Real x3);
template<class TOVEOS>
KOKKOS_INLINE_FUNCTION
static Real A2(const tov_pgen& pgen, const TOVEOS& eos, Real x1, Real x2, Real x3);

enum class LocationTag {Host, Device};

// EOS policies
class PolytropeEOS {
 private:
  Real kappa;
  Real gamma;

 public:
  explicit PolytropeEOS(ParameterInput* pin) {
    kappa = pin->GetReal("problem", "kappa");
    gamma = pin->GetReal("mhd", "gamma");
  }

  template<LocationTag loc>
  KOKKOS_INLINE_FUNCTION
  Real GetPFromRho(Real rho) const {
    return kappa*Kokkos::pow(rho, gamma);
  }

  template<LocationTag loc>
  KOKKOS_INLINE_FUNCTION
  Real GetRhoFromP(Real P) const {
    return Kokkos::pow(P/kappa, 1.0/gamma);
  }

  template<LocationTag loc>
  KOKKOS_INLINE_FUNCTION
  Real GetEFromRho(Real rho) const {
    return rho + kappa*Kokkos::pow(rho, gamma)/(gamma - 1.0);
  }
};

class PiecewisePolytropeEOS: public Primitive::PiecewisePolytrope {
 public:
  explicit PiecewisePolytropeEOS(ParameterInput *pin) {
    ReadParametersFromInput("mhd", pin);
  }

  template<LocationTag loc>
  KOKKOS_INLINE_FUNCTION
  Real GetPFromRho(Real rho) const {
    Real nb = rho/mb;
    int p = FindPiece(nb);
    return GetColdPressure(nb, p);
  }

  template<LocationTag loc>
  KOKKOS_INLINE_FUNCTION
  Real GetRhoFromP(Real P) const {
    Real rhob = GetDensityFromColdPressure(P);
    return rhob;
  }

  template<LocationTag loc>
  KOKKOS_INLINE_FUNCTION
  Real GetEFromRho(Real rho) const {
    Real nb = rho/mb;
    int p = FindPiece(nb);
    return GetColdEnergy(nb, p);
  }
};

class TabulatedEOS {
 private:
  DualArray1D<Real> m_log_rho;
  DualArray1D<Real> m_log_p;
  DualArray1D<Real> m_log_e;
  DualArray1D<Real> m_ye;

  Real dlrho;
  Real lrho_min;
  Real lrho_max;
  Real lP_min;
  Real lP_max;

  std::string fname;
  size_t m_nn;

  //static const Real fm_to_Msun = 6.771781959609192e-19
  //static const Real MeV_to_Msun = 8.962968324680417e-61
  static constexpr Real ener_to_geo = 2.8863099290608455e-6;

 public:
  explicit TabulatedEOS(ParameterInput* pin) {
    fname = pin->GetString("problem", "table");

    TableReader::Table table;

    auto read_result = table.ReadTable(fname);
    if (read_result.error != TableReader::ReadResult::SUCCESS) {
      std::cout << "TOV EOS table could not be read.\n";
      assert(false);
    }
    // TODO(JMF) Check that table has right fields and dimensions
    auto& table_scalars = table.GetScalars();
    Real mb = table_scalars.at("mn");

    // Get table dimensions
    auto& point_info = table.GetPointInfo();
    m_nn = point_info[0].second;

    // Allocate storage
    Kokkos::realloc(m_log_rho, m_nn);
    Kokkos::realloc(m_log_p, m_nn);
    Kokkos::realloc(m_log_e, m_nn);
    Kokkos::realloc(m_ye, m_nn);

    // Read rho
    Real * table_nb = table["nb"];
    for (size_t in = 0; in < m_nn; in++) {
      m_log_rho.h_view(in) = log(table_nb[in]*mb*ener_to_geo);
    }
    dlrho = m_log_rho.h_view(1)-m_log_rho.h_view(0);
    lrho_min = m_log_rho.h_view(0);
    lrho_max = m_log_rho.h_view(m_nn-1);

    // Read pressure
    Real * table_Q1 = table["Q1"];
    for (size_t in = 0; in < m_nn; in++) {
      m_log_p.h_view(in) = log(table_Q1[in]*table_nb[in]*ener_to_geo);
    }
    lP_min = m_log_p.h_view(0);
    lP_max = m_log_p.h_view(m_nn-1);

    // Read energy
    Real * table_Q7 = table["Q7"];
    for (size_t in = 0; in < m_nn; in++) {
      m_log_e.h_view(in) = log(mb*(table_Q7[in] + 1.)*table_nb[in]*ener_to_geo);
    }

    // Read electron fraction
    Real * table_ye = table["Y[e]"];
    for (size_t in = 0; in < m_nn; in++) {
      m_ye.h_view(in) = table_ye[in];
    }

    std::cout << "Loaded table " << fname << std::endl
              << "  rho = [" << exp(lrho_min) << ", " << exp(lrho_max) << "]" << std::endl
              << "  P = [" << exp(lP_min) << ", " << exp(lP_max) << "]" << std::endl;

    // Sync the views to the GPU
    m_log_rho.template modify<HostMemSpace>();
    m_log_p.template modify<HostMemSpace>();
    m_log_e.template modify<HostMemSpace>();
    m_ye.template modify<HostMemSpace>();

    m_log_rho.template sync<DevExeSpace>();
    m_log_p.template sync<DevExeSpace>();
    m_log_e.template sync<DevExeSpace>();
    m_ye.template sync<DevExeSpace>();
  }

  template<LocationTag loc>
  KOKKOS_INLINE_FUNCTION
  Real GetPFromRho(Real rho) const {
    Real lrho = log(rho);
    if (lrho < lrho_min) {
      return 0.0;
    }
    int lb = static_cast<int>((lrho-lrho_min)/dlrho);
    int ub = lb + 1;
    if constexpr (loc == LocationTag::Host) {
      return exp(Interpolate(lrho, m_log_rho.h_view(lb), m_log_rho.h_view(ub),
                              m_log_p.h_view(lb), m_log_p.h_view(ub)));
    } else {
      return exp(Interpolate(lrho, m_log_rho.d_view(lb), m_log_rho.d_view(ub),
                              m_log_p.d_view(lb), m_log_p.d_view(ub)));
    }
  }

  template<LocationTag loc>
  KOKKOS_INLINE_FUNCTION
  Real GetEFromRho(Real rho) const {
    Real lrho = log(rho);
    if (lrho < lrho_min) {
      return 0.0;
    }
    int lb = static_cast<int>((lrho-lrho_min)/dlrho);
    int ub = lb + 1;
    if constexpr (loc == LocationTag::Host) {
      return exp(Interpolate(lrho, m_log_rho.h_view(lb), m_log_rho.h_view(ub),
                              m_log_e.h_view(lb), m_log_e.h_view(ub)));
    } else {
      return exp(Interpolate(lrho, m_log_rho.d_view(lb), m_log_rho.d_view(ub),
                              m_log_e.d_view(lb), m_log_e.d_view(ub)));
    }
  }

  template<LocationTag loc>
  KOKKOS_INLINE_FUNCTION
  Real GetYeFromRho(Real rho) const {
    Real lrho = log(rho);
    int lb = static_cast<int>((lrho-lrho_min)/dlrho);
    int ub = lb + 1;
    if constexpr (loc == LocationTag::Host) {
      return Interpolate(lrho, m_log_rho.h_view(lb), m_log_rho.h_view(ub),
                          m_ye.h_view(lb), m_ye.h_view(ub));
    } else {
      return Interpolate(lrho, m_log_rho.d_view(lb), m_log_rho.d_view(ub),
                          m_ye.d_view(lb), m_ye.d_view(ub));
    }
  }

  template<LocationTag loc>
  KOKKOS_INLINE_FUNCTION
  Real GetRhoFromP(Real P) const {
    Real lP = log(P);
    int lb = 0;
    int ub = m_nn-1;
    // If the pressure is below the minimum of the table, we return zero density.
    if (lP < lP_min) {
      return 0.0;
    }
    // Do a binary search for the lower and upper indices of the pressure
    if constexpr (loc == LocationTag::Host) {
      while (ub - lb > 1) {
        int idx = (lb + ub)/2;
        if (m_log_p.h_view(idx) > lP) {
          ub = idx;
        } else {
          lb = idx;
        }
      }
      return exp(Interpolate(lP, m_log_p.h_view(lb), m_log_p.h_view(ub),
                              m_log_rho.h_view(lb), m_log_rho.h_view(ub)));
    } else {
      while (ub - lb > 1) {
        int idx = (lb + ub)/2;
        if (m_log_p.d_view(idx) > lP) {
          ub = idx;
        } else {
          lb = idx;
        }
      }
      return exp(Interpolate(lP, m_log_p.d_view(lb), m_log_p.d_view(ub),
                              m_log_rho.d_view(lb), m_log_rho.d_view(ub)));
    }
  }
};

// Prototypes for user-defined BCs and history
void TOVHistory(HistoryData *pdata, Mesh *pm);
void VacuumBC(Mesh *pm);

template<class TOVEOS>
void SetupTOV(ParameterInput* pin, Mesh* pmy_mesh_, tov_pgen& tov) {
  MeshBlockPack* pmbp = pmy_mesh_->pmb_pack;
  TOVEOS eos{pin};

  // Set pfloor to be consistent with the density floor.
  tov.pfloor = eos.template GetPFromRho<LocationTag::Host>(tov.dfloor);

  bool minkowski = pin->GetOrAddBoolean("problem", "minkowski", false);

  // Generate the TOV star
  ConstructTOV(tov, eos);

  constexpr bool use_ye = std::is_same<TOVEOS, TabulatedEOS>::value;

  // Select either Hydro or MHD
  DvceArray5D<Real> u0_, w0_;
  int nvars_;
  if (pmbp->phydro != nullptr) {
    u0_ = pmbp->phydro->u0;
    w0_ = pmbp->phydro->w0;
    nvars_ = pmbp->phydro->nhydro;
  } else if (pmbp->pmhd != nullptr) {
    u0_ = pmbp->pmhd->u0;
    w0_ = pmbp->pmhd->w0;
    nvars_ = pmbp->pmhd->nmhd;
  }

  // Capture variables for kernel
  auto &indcs = pmy_mesh_->mb_indcs;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*ng) : 1;
  int &is = indcs.is;
  int &js = indcs.js;
  int &ks = indcs.ks;
  int &ie = indcs.ie;
  int &je = indcs.je;
  int &ke = indcs.ke;
  int nmb1 = pmbp->nmb_thispack - 1;

  auto &size = pmbp->pmb->mb_size;
  auto &adm = pmbp->padm->adm;
  auto &tov_ = tov;
  auto &eos_ = eos;
  Kokkos::Random_XorShift64_Pool<> rand_pool64(pmbp->gids);
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

    // Calculate the rest-mass density, pressure, and mass for a specific isotropic
    // radial coordinate.
    Real r = sqrt(SQR(x1v) + SQR(x2v) + SQR(x3v));
    Real s = sqrt(SQR(x1v) + SQR(x2v));
    Real rho, p, mass, alp, r_schw;
    Real vr = 0.;
    Real p_pert = 0.;
    Real ye = 0.5;
    auto &use_ye_ = use_ye;
    //printf("Grabbing primitives!\n");
    if (!tov.isotropic) {
      GetPrimitivesAtPoint(tov_, eos_, r, rho, p, mass, alp);
      if (r <= tov.R_edge) {
        Real x = r/tov.R_edge;
        vr = 0.5*tov_.v_pert*(3.0*x - x*x*x);
        auto rand_gen = rand_pool64.get_state();
        p_pert = 2.0*tov_.p_pert*(rand_gen.frand() - 0.5);
        rand_pool64.free_state(rand_gen);
        if constexpr (use_ye) {
          ye = eos_.template GetYeFromRho<LocationTag::Device>(rho);
        }
      }
    } else {
      GetPrimitivesAtIsoPoint(tov_, eos_, r, rho, p, mass, alp);
      r_schw = FindSchwarzschildR(tov, r, mass);
      if (r_schw <= tov.R_edge) {
        Real x = r_schw/tov.R_edge;
        vr = 0.5*tov_.v_pert*(3.0*x - x*x*x);
        auto rand_gen = rand_pool64.get_state();
        p_pert = 2.0*tov_.p_pert*(rand_gen.frand() - 0.5);
        rand_pool64.free_state(rand_gen);
        if constexpr (use_ye) {
          ye = eos_.template GetYeFromRho<LocationTag::Device>(rho);
        }
      }
    }
    //printf("Primitives retrieved!\n");


    // FIXME: assumes ideal gas!
    // Set hydrodynamic quantities
    w0_(m,IDN,k,j,i) = fmax(rho, tov_.dfloor);
    //w0_(m,IEN,k,j,i) = fmax(p, tov_.pfloor)/(tov_.gamma - 1.0);
    w0_(m,IPR,k,j,i) = fmax(p*(1. + p_pert), tov_.pfloor);
    w0_(m,IVX,k,j,i) = vr*x1v/r;
    w0_(m,IVY,k,j,i) = vr*x2v/r;
    w0_(m,IVZ,k,j,i) = vr*x3v/r;
    auto &nvars = nvars_;
    if constexpr (use_ye) {
      w0_(m,nvars,k,j,i) = ye;
    }

    // Set ADM variables
    adm.alpha(m,k,j,i) = alp;
    if (minkowski) {
      adm.g_dd(m,0,0,k,j,i) = adm.g_dd(m,1,1,k,j,i) = adm.g_dd(m,2,2,k,j,i) = 1.0;
      adm.g_dd(m,0,1,k,j,i) = adm.g_dd(m,0,2,k,j,i) = adm.g_dd(m,1,2,k,j,i) = 0.0;
      adm.beta_u(m,0,k,j,i) = adm.beta_u(m,1,k,j,i) = adm.beta_u(m,2,k,j,i) = 0.0;
      adm.alpha(m,k,j,i) = 1.0;
    } else if (!tov.isotropic) {
      // Auxiliary metric quantities
      Real fmet = 0.0;
      if (r > 0) {
         // (g_rr - 1)/r^2
         fmet = (1./(1. - 2*mass/r) - 1.)/(r*r);
      }


      adm.g_dd(m,0,0,k,j,i) = x1v*x1v*fmet + 1.0;
      adm.g_dd(m,0,1,k,j,i) = x1v*x2v*fmet;
      adm.g_dd(m,0,2,k,j,i) = x1v*x3v*fmet;
      adm.g_dd(m,1,1,k,j,i) = x2v*x2v*fmet + 1.0;
      adm.g_dd(m,1,2,k,j,i) = x2v*x3v*fmet;
      adm.g_dd(m,2,2,k,j,i) = x3v*x3v*fmet + 1.0;
      Real det = adm::SpatialDet(
              adm.g_dd(m,0,0,k,j,i), adm.g_dd(m,0,1,k,j,i),
              adm.g_dd(m,0,2,k,j,i), adm.g_dd(m,1,1,k,j,i),
              adm.g_dd(m,1,2,k,j,i), adm.g_dd(m,2,2,k,j,i));
      adm.psi4(m,k,j,i) = pow(det, 1./3.);
    } else {
      Real fmet = 1.;
      if (r > 0) {
        fmet = r_schw/r;
      }
      Real psi4 = fmet*fmet;

      adm.g_dd(m,0,0,k,j,i) = adm.g_dd(m,1,1,k,j,i) = adm.g_dd(m,2,2,k,j,i) = psi4;
      adm.g_dd(m,0,1,k,j,i) = adm.g_dd(m,0,2,k,j,i) = adm.g_dd(m,1,2,k,j,i) = 0.0;
      adm.psi4(m,k,j,i) = psi4;
    }
    adm.beta_u(m,0,k,j,i) = adm.beta_u(m,1,k,j,i) = adm.beta_u(m,2,k,j,i) = 0.0;
    adm.vK_dd(m,0,0,k,j,i) = adm.vK_dd(m,0,1,k,j,i) = adm.vK_dd(m,0,2,k,j,i) = 0.0;
    adm.vK_dd(m,1,1,k,j,i) = adm.vK_dd(m,1,2,k,j,i) = adm.vK_dd(m,2,2,k,j,i) = 0.0;
  });

  if (pmbp->pmhd != nullptr) {
    // parse some parameters
    tov.b_norm = pin->GetOrAddReal("problem", "b_norm", 0.0);
    tov.pcut = pin->GetOrAddReal("problem", "pcut", 1e-6);
    tov.magindex = pin->GetOrAddReal("problem", "magindex", 2);

    // If use_pcut_rel = true, we take pcut to be a percentage of pmax rather than
    // an absolute cutoff.
    if (pin->GetOrAddBoolean("problem", "use_pcut_rel", false)) {
      // FIXME: Assumes ideal gas!
      //Real pmax = tov.kappa*pow(tov.rhoc, tov.gamma);
      Real pmax = eos_.template GetPFromRho<LocationTag::Device>(tov.rhoc);
      tov.pcut = tov.pcut * pmax;
    }

    // compute vector potential over all faces
    int ncells1 = indcs.nx1 + 2*(indcs.ng);
    int ncells2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
    int ncells3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;
    int nmb = pmbp->nmb_thispack;
    DvceArray4D<Real> a1, a2, a3;
    Kokkos::realloc(a1, nmb,ncells3,ncells2,ncells1);
    Kokkos::realloc(a2, nmb,ncells3,ncells2,ncells1);
    Kokkos::realloc(a3, nmb,ncells3,ncells2,ncells1);

    auto &nghbr = pmbp->pmb->nghbr;
    auto &mblev = pmbp->pmb->mb_lev;

    par_for("pgen_potential", DevExeSpace(), 0,nmb-1,ks,ke+1,js,je+1,is,ie+1,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real &x1min = size.d_view(m).x1min;
      Real &x1max = size.d_view(m).x1max;
      int nx1 = indcs.nx1;
      Real x1v = CellCenterX(i-is, nx1, x1min, x1max);
      Real x1f = LeftEdgeX(i-is,nx1,x1min,x1max);

      Real &x2min = size.d_view(m).x2min;
      Real &x2max = size.d_view(m).x2max;
      int nx2 = indcs.nx2;
      Real x2v = CellCenterX(j-js, nx2, x2min, x2max);
      Real x2f = LeftEdgeX(j-js,nx2,x2min,x2max);

      Real &x3min = size.d_view(m).x3min;
      Real &x3max = size.d_view(m).x3max;
      int nx3 = indcs.nx3;
      Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);
      Real x3f = LeftEdgeX(k-ks,nx3,x3min,x3max);

      Real x1fp1 = LeftEdgeX(i+1-is, nx1, x1min, x1max);
      Real x2fp1 = LeftEdgeX(j+1-js, nx2, x2min, x2max);
      Real x3fp1 = LeftEdgeX(k+1-ks, nx3, x3min, x3max);
      Real dx1 = size.d_view(m).dx1;
      Real dx2 = size.d_view(m).dx2;
      Real dx3 = size.d_view(m).dx3;

      a1(m,k,j,i) = A1(tov_, eos_, x1v, x2f, x3f);
      a2(m,k,j,i) = A2(tov_, eos_, x1f, x2v, x3f);
      a3(m,k,j,i) = 0.0;

      // When neighboring MeshBock is at finer level, compute vector potential as sum of
      // values at fine grid resolution.  This guarantees flux on shared fine/coarse
      // faces is identical.

      // Correct A1 at x2-faces, x3-faces, and x2x3-edges
      if ((nghbr.d_view(m,8 ).lev > mblev.d_view(m) && j==js) ||
          (nghbr.d_view(m,9 ).lev > mblev.d_view(m) && j==js) ||
          (nghbr.d_view(m,10).lev > mblev.d_view(m) && j==js) ||
          (nghbr.d_view(m,11).lev > mblev.d_view(m) && j==js) ||
          (nghbr.d_view(m,12).lev > mblev.d_view(m) && j==je+1) ||
          (nghbr.d_view(m,13).lev > mblev.d_view(m) && j==je+1) ||
          (nghbr.d_view(m,14).lev > mblev.d_view(m) && j==je+1) ||
          (nghbr.d_view(m,15).lev > mblev.d_view(m) && j==je+1) ||
          (nghbr.d_view(m,24).lev > mblev.d_view(m) && k==ks) ||
          (nghbr.d_view(m,25).lev > mblev.d_view(m) && k==ks) ||
          (nghbr.d_view(m,26).lev > mblev.d_view(m) && k==ks) ||
          (nghbr.d_view(m,27).lev > mblev.d_view(m) && k==ks) ||
          (nghbr.d_view(m,28).lev > mblev.d_view(m) && k==ke+1) ||
          (nghbr.d_view(m,29).lev > mblev.d_view(m) && k==ke+1) ||
          (nghbr.d_view(m,30).lev > mblev.d_view(m) && k==ke+1) ||
          (nghbr.d_view(m,31).lev > mblev.d_view(m) && k==ke+1) ||
          (nghbr.d_view(m,40).lev > mblev.d_view(m) && j==js && k==ks) ||
          (nghbr.d_view(m,41).lev > mblev.d_view(m) && j==js && k==ks) ||
          (nghbr.d_view(m,42).lev > mblev.d_view(m) && j==je+1 && k==ks) ||
          (nghbr.d_view(m,43).lev > mblev.d_view(m) && j==je+1 && k==ks) ||
          (nghbr.d_view(m,44).lev > mblev.d_view(m) && j==js && k==ke+1) ||
          (nghbr.d_view(m,45).lev > mblev.d_view(m) && j==js && k==ke+1) ||
          (nghbr.d_view(m,46).lev > mblev.d_view(m) && j==je+1 && k==ke+1) ||
          (nghbr.d_view(m,47).lev > mblev.d_view(m) && j==je+1 && k==ke+1)) {
        Real xl = x1v + 0.25*dx1;
        Real xr = x1v - 0.25*dx1;
        a1(m,k,j,i) = 0.5*(A1(tov_, eos_, xl,x2f,x3f) + A1(tov_, eos_, xr,x2f,x3f));
      }

      // Correct A2 at x1-faces, x3-faces, and x1x3-edges
      if ((nghbr.d_view(m,0 ).lev > mblev.d_view(m) && i==is) ||
          (nghbr.d_view(m,1 ).lev > mblev.d_view(m) && i==is) ||
          (nghbr.d_view(m,2 ).lev > mblev.d_view(m) && i==is) ||
          (nghbr.d_view(m,3 ).lev > mblev.d_view(m) && i==is) ||
          (nghbr.d_view(m,4 ).lev > mblev.d_view(m) && i==ie+1) ||
          (nghbr.d_view(m,5 ).lev > mblev.d_view(m) && i==ie+1) ||
          (nghbr.d_view(m,6 ).lev > mblev.d_view(m) && i==ie+1) ||
          (nghbr.d_view(m,7 ).lev > mblev.d_view(m) && i==ie+1) ||
          (nghbr.d_view(m,24).lev > mblev.d_view(m) && k==ks) ||
          (nghbr.d_view(m,25).lev > mblev.d_view(m) && k==ks) ||
          (nghbr.d_view(m,26).lev > mblev.d_view(m) && k==ks) ||
          (nghbr.d_view(m,27).lev > mblev.d_view(m) && k==ks) ||
          (nghbr.d_view(m,28).lev > mblev.d_view(m) && k==ke+1) ||
          (nghbr.d_view(m,29).lev > mblev.d_view(m) && k==ke+1) ||
          (nghbr.d_view(m,30).lev > mblev.d_view(m) && k==ke+1) ||
          (nghbr.d_view(m,31).lev > mblev.d_view(m) && k==ke+1) ||
          (nghbr.d_view(m,32).lev > mblev.d_view(m) && i==is && k==ks) ||
          (nghbr.d_view(m,33).lev > mblev.d_view(m) && i==is && k==ks) ||
          (nghbr.d_view(m,34).lev > mblev.d_view(m) && i==ie+1 && k==ks) ||
          (nghbr.d_view(m,35).lev > mblev.d_view(m) && i==ie+1 && k==ks) ||
          (nghbr.d_view(m,36).lev > mblev.d_view(m) && i==is && k==ke+1) ||
          (nghbr.d_view(m,37).lev > mblev.d_view(m) && i==is && k==ke+1) ||
          (nghbr.d_view(m,38).lev > mblev.d_view(m) && i==ie+1 && k==ke+1) ||
          (nghbr.d_view(m,39).lev > mblev.d_view(m) && i==ie+1 && k==ke+1)) {
        Real xl = x2v + 0.25*dx2;
        Real xr = x2v - 0.25*dx2;
        a2(m,k,j,i) = 0.5*(A2(tov_, eos_, x1f,xl,x3f) + A2(tov_, eos_, x1f,xr,x3f));
      }
    });

    auto &b0 = pmbp->pmhd->b0;
    par_for("pgen_Bfc", DevExeSpace(), 0,nmb-1,ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      // Compute face-centered fields from curl(A).
      Real dx1 = size.d_view(m).dx1;
      Real dx2 = size.d_view(m).dx2;
      Real dx3 = size.d_view(m).dx3;

      b0.x1f(m,k,j,i) = ((a3(m,k,j+1,i) - a3(m,k,j,i))/dx2 -
                         (a2(m,k+1,j,i) - a2(m,k,j,i))/dx3);
      b0.x2f(m,k,j,i) = ((a1(m,k+1,j,i) - a1(m,k,j,i))/dx3 -
                         (a3(m,k,j,i+1) - a3(m,k,j,i))/dx1);
      b0.x3f(m,k,j,i) = ((a2(m,k,j,i+1) - a2(m,k,j,i))/dx1 -
                         (a1(m,k,j+1,i) - a1(m,k,j,i))/dx2);

      // Include extra face-component at edge of block in each direction
      if (i==ie) {
        b0.x1f(m,k,j,i+1) = ((a3(m,k,j+1,i+1) - a3(m,k,j,i+1))/dx2 -
                             (a2(m,k+1,j,i+1) - a2(m,k,j,i+1))/dx3);
      }
      if (j==je) {
        b0.x2f(m,k,j+1,i) = ((a1(m,k+1,j+1,i) - a1(m,k,j+1,i))/dx3 -
                             (a3(m,k,j+1,i+1) - a3(m,k,j+1,i))/dx1);
      }
      if (k==ke) {
        b0.x3f(m,k+1,j,i) = ((a2(m,k+1,j,i+1) - a2(m,k+1,j,i))/dx1 -
                             (a1(m,k+1,j+1,i) - a1(m,k+1,j,i))/dx2);
      }
    });

    // Compute cell-centered fields
    auto &bcc_ = pmbp->pmhd->bcc0;
    par_for("pgen_Bcc", DevExeSpace(), 0,nmb-1,ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      // cell-centered fields are simple linear average of face-centered fields
      Real& w_bx = bcc_(m,IBX,k,j,i);
      Real& w_by = bcc_(m,IBY,k,j,i);
      Real& w_bz = bcc_(m,IBZ,k,j,i);
      w_bx = 0.5*(b0.x1f(m,k,j,i) + b0.x1f(m,k,j,i+1));
      w_by = 0.5*(b0.x2f(m,k,j,i) + b0.x2f(m,k,j+1,i));
      w_bz = 0.5*(b0.x3f(m,k,j,i) + b0.x3f(m,k+1,j,i));
    });
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

  tov_pgen tov;
  // FIXME: Set boundary condition function?
  user_bcs_func = VacuumBC;

  // Read problem-specific parameters from input file
  // global parameters
  tov.rhoc  = pin->GetReal("problem", "rhoc");
  tov.npoints = pin->GetReal("problem", "npoints");
  tov.dr    = pin->GetReal("problem", "dr");
  // Select either Hydro or MHD
  std::string block;
  if (pmbp->phydro != nullptr) {
    block = std::string("hydro");
  } else if (pmbp->pmhd != nullptr) {
    block = std::string("mhd");
  }
  tov.gamma = pin->GetOrAddReal(block, "gamma", 5.0/3.0);
  tov.dfloor = pin->GetOrAddReal(block, "dfloor", (FLT_MIN));
  //tov.pfloor = pin->GetOrAddReal(block, "pfloor", (FLT_MIN));
  tov.v_pert = pin->GetOrAddReal("problem" , "v_pert", 0.0);
  tov.p_pert = pin->GetOrAddReal("problem", "p_pert", 0.0);
  tov.isotropic = pin->GetOrAddBoolean("problem", "isotropic", false);

  // Set the history function for a TOV star
  user_hist_func = &TOVHistory;

  // initialize primitive variables for restart ----------------------------------------
  // FIXME: need to load data on restart?
  if (restart) {
    return;
  }

  // Select the right TOV template based on the EOS we need.
  if (pmbp->pdyngr->eos_policy == DynGRMHD_EOS::eos_ideal) {
    SetupTOV<PolytropeEOS>(pin, pmy_mesh_, tov);
  } else if (pmbp->pdyngr->eos_policy == DynGRMHD_EOS::eos_compose) {
    SetupTOV<TabulatedEOS>(pin, pmy_mesh_, tov);
  } else if (pmbp->pdyngr->eos_policy == DynGRMHD_EOS::eos_piecewise_poly) {
    SetupTOV<PiecewisePolytropeEOS>(pin, pmy_mesh_, tov);
  } else {
    std::cout << "### WARNING in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Unknown EOS requested for TOV star problem" << std::endl
              << "Defaulting to fixed polytropic EOS" << std::endl;
    SetupTOV<PolytropeEOS>(pin, pmy_mesh_, tov);
  }

  // Mesh block info for loop limits
  auto &indcs = pmy_mesh_->mb_indcs;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*ng) : 1;

  // Convert primitives to conserved
  if (pmbp->padm == nullptr) {
    // Complain about something here, because this is a dynamic GR test.
  } else {
    pmbp->pdyngr->PrimToConInit(0, (n1-1), 0, (n2-1), 0, (n3-1));
  }

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

template<class TOVEOS>
static void RHS(Real r, Real P, Real m, Real alp, Real R, TOVEOS& eos,
                tov_pgen& tov, Real& dP, Real& dm, Real& dalp, Real& dR) {
  // In our units, the equations take the form
  // dP/dr = -(e + P)/(1 - 2m/r) (m + 4\pi r^3 P)/r^2
  // dm/dr = 4\pi r^2 e
  // d\alpha/dr = \alpha/(1 - 2m/r) (m + 4\pi r^3 P)/r^2
  // dR/dr = R/r (1 - 2m/r)^(-1/2)
  // FIXME: Assumes ideal gas!
  if (r < 1e-3*tov.dr) {
    dP = 0.0;
    dm = 0.0;
    dalp = 0.0;
    dR = 1.0;
    return;
  }
  // FIXME: Assumes ideal gas!
  /*Real rho = pow(P/tov.kappa, 1.0/tov.gamma);
  Real e   = rho + P/(tov.gamma - 1.0);*/
  Real rho = eos.template GetRhoFromP<LocationTag::Host>(P);
  Real e = eos.template GetEFromRho<LocationTag::Host>(rho);

  Real A = 1.0/(1.0 - 2.0*m/r);
  Real B = (m + 4.0*M_PI*r*r*r*P)/SQR(r);
  dP   = -(e + P)*A * B;
  dm   = 4.0*M_PI*SQR(r)*e;
  dalp = alp*A * B;
  dR   = R/r*sqrt(A);
}

// Construct a TOV star using the shooting method.
template<class TOVEOS>
static void ConstructTOV(tov_pgen& tov, TOVEOS& eos) {
  // First, allocate the data.
  /*tov.R   = new Real[tov.npoints];
  tov.M   = new Real[tov.npoints];
  tov.P   = new Real[tov.npoints];
  tov.alp = new Real[tov.npoints];*/
  Kokkos::realloc(tov.R, tov.npoints);
  Kokkos::realloc(tov.R_iso, tov.npoints);
  Kokkos::realloc(tov.M, tov.npoints);
  Kokkos::realloc(tov.P, tov.npoints);
  Kokkos::realloc(tov.alp, tov.npoints);

  // Set aliases
  auto &R = tov.R.h_view;
  auto &R_iso = tov.R_iso.h_view;
  auto &M = tov.M.h_view;
  auto &P = tov.P.h_view;
  auto &alp = tov.alp.h_view;
  int npoints = tov.npoints;
  Real dr = tov.dr;

  // Set initial data
  // FIXME: Assumes ideal gas for now!
  R(0) = 0.0;
  R_iso(0) = 0.0;
  M(0) = 0.0;
  P(0) = eos.template GetPFromRho<LocationTag::Host>(tov.rhoc);
  // FIXME: Assumes ideal gas!
  //P(0) = tov.kappa*pow(tov.rhoc, tov.gamma);
  alp(0) = 1.0;

  // Integrate outward using RK4
  for (int i = 0; i < npoints-1; i++) {
    Real r, P_pt, alp_pt, m_pt, R_pt;

    // First stage
    Real dP1, dm1, dalp1, dR1;
    r = i*dr;
    P_pt = P(i);
    alp_pt = alp(i);
    m_pt = M(i);
    R_pt = R_iso(i);
    RHS(r, P_pt, m_pt, alp_pt, R_pt, eos, tov, dP1, dm1, dalp1, dR1);

    // Second stage
    Real dP2, dm2, dalp2, dR2;
    r = (i + 0.5)*dr;
    P_pt = fmax(P(i) + 0.5*dr*dP1,0.0);
    m_pt = M(i) + 0.5*dr*dm1;
    alp_pt = alp(i) + 0.5*dr*dalp1;
    R_pt = R_iso(i) + 0.5*dr*dR1;
    RHS(r, P_pt, m_pt, alp_pt, R_pt, eos, tov, dP2, dm2, dalp2, dR2);

    // Third stage
    Real dP3, dm3, dalp3, dR3;
    P_pt = fmax(P(i) + 0.5*dr*dP2,0.0);
    m_pt = M(i) + 0.5*dr*dm2;
    alp_pt = alp(i) + 0.5*dr*dalp2;
    R_pt = R_iso(i) + 0.5*dr*dR2;
    RHS(r, P_pt, m_pt, alp_pt, R_pt, eos, tov, dP3, dm3, dalp3, dR3);

    // Fourth stage
    Real dP4, dm4, dalp4, dR4;
    r = (i + 1)*dr;
    P_pt = fmax(P(i) + dr*dP3,0.0);
    m_pt = M(i) + dr*dm3;
    alp_pt = alp(i) + dr*dalp3;
    R_pt = R_iso(i) + dr*dR3;
    RHS(r, P_pt, m_pt, alp_pt, R_pt, eos, tov, dP4, dm4, dalp4, dR4);

    // Combine all the stages together
    R(i+1) = (i + 1)*dr;
    P(i+1) = P(i) + dr*(dP1 + 2.0*dP2 + 2.0*dP3 + dP4)/6.0;
    M(i+1) = M(i) + dr*(dm1 + 2.0*dm2 + 2.0*dm3 + dm4)/6.0;
    alp(i+1) = alp(i) + dr*(dalp1 + 2.0*dalp2 + 2.0*dalp3 + dalp4)/6.0;
    R_iso(i+1) = R_iso(i) + dr*(dR1 + 2.0*dR2 + 2.0*dR3 + dR4)/6.0;

    // If the pressure falls below zero, we've hit the edge of the star.
    if (P(i+1) <= 0.0 || P(i+1) <= tov.pfloor) {
      tov.n_r = i+1;
      break;
    }
  }

  // Now we can do a linear interpolation to estimate the actual edge of the star.
  int n_r = tov.n_r;
  tov.R_edge = Interpolate(tov.pfloor, P(n_r-1), P(n_r), R(n_r-1), R(n_r));
  tov.M_edge = Interpolate(tov.R_edge, R(n_r-1), R(n_r), M(n_r-1), M(n_r));

  // Replace the edges of the star.
  P(n_r) = tov.pfloor;
  M(n_r) = tov.M_edge;
  alp(n_r) = Interpolate(tov.R_edge, R(n_r-1), R(n_r), alp(n_r-1), alp(n_r));
  R(n_r) = tov.R_edge;
  R_iso(n_r) = Interpolate(tov.R_edge, R(n_r-1), R(n_r), R_iso(n_r-1), R_iso(n_r));

  // Rescale alpha so that it matches the Schwarzschild metric at the boundary.
  // We also need to rescale the isotropic radius to agree at the boundary.
  Real rs = 2.0*tov.M_edge;
  Real bound = sqrt(1.0 - rs/tov.R_edge);
  Real scale = bound/alp(n_r);
  tov.R_edge_iso = 0.5*(R(n_r) - M(n_r) + sqrt(R(n_r)*(R(n_r) - 2.0*M(n_r))));
  Real iso_scale = tov.R_edge_iso/R_iso(n_r);
  for (int i = 0; i <= n_r; i++) {
    alp(i) = alp(i)*scale;
    R_iso(i) = R_iso(i)*iso_scale;
  }

  // Print out details of the calculation
  if (global_variable::my_rank == 0) {
    std::cout << "\nTOV INITIAL DATA\n"
              << "----------------\n";
    std::cout << "Total points in buffer: " << tov.npoints << "\n";
    std::cout << "Radial step: " << tov.dr << "\n";
    std::cout << "Radius (Schwarzschild): " << tov.R_edge << "\n";
    std::cout << "Radius (Isotropic): " << tov.R_edge_iso << "\n";
    std::cout << "Mass: " << tov.M_edge << "\n\n";
  }

  // Sync the views to the GPU
  tov.R.template modify<HostMemSpace>();
  tov.R_iso.template modify<HostMemSpace>();
  tov.M.template modify<HostMemSpace>();
  tov.alp.template modify<HostMemSpace>();
  tov.P.template modify<HostMemSpace>();

  tov.R.template sync<DevExeSpace>();
  tov.R_iso.template sync<DevExeSpace>();
  tov.M.template sync<DevExeSpace>();
  tov.alp.template sync<DevExeSpace>();
  tov.P.template sync<DevExeSpace>();
}

template<class TOVEOS>
KOKKOS_INLINE_FUNCTION
static void GetPrimitivesAtPoint(const tov_pgen& tov, const TOVEOS& eos, Real r,
                                 Real &rho, Real &p, Real &m, Real &alp) {
  // Check if we're past the edge of the star.
  // If so, we just return atmosphere with Schwarzschild.
  if (r >= tov.R_edge) {
    rho = 0.0;
    p = 0.0;
    m = tov.M_edge;
    alp = sqrt(1.0 - 2.0*m/r);
    return;
  }
  // Get the lower index for where our point must be located.
  int idx = static_cast<int>(r/tov.dr);
  const auto &R = tov.R.d_view;
  const auto &Ps = tov.P.d_view;
  const auto &alps = tov.alp.d_view;
  const auto &Ms = tov.M.d_view;
  // Interpolate to get the primitive.
  p = Interpolate(r, R(idx), R(idx+1), Ps(idx), Ps(idx+1));
  m = Interpolate(r, R(idx), R(idx+1), Ms(idx), Ms(idx+1));
  alp = Interpolate(r, R(idx), R(idx+1), alps(idx), alps(idx+1));
  // FIXME: Assumes ideal gas!
  //rho = pow(p/tov.kappa, 1.0/tov.gamma);
  rho = eos.template GetRhoFromP<LocationTag::Device>(p);
}

KOKKOS_INLINE_FUNCTION
static int FindIsotropicIndex(const tov_pgen& tov, Real r_iso) {
  // Perform a bisection search to find the closest index to the requested isotropic
  // point.
  const auto &R_iso = tov.R_iso.d_view;
  int lb = 0;
  int ub = tov.n_r;
  int idx = lb;
  while (R_iso(lb+1) < r_iso) {
    idx = (lb + ub)/2;
    if (R_iso(idx) < r_iso) {
      lb = idx;
    } else {
      ub = idx;
    }
  }
  return lb;
}

KOKKOS_INLINE_FUNCTION
static Real FindSchwarzschildR(const tov_pgen& tov, Real r_iso, Real mass) {
  if (r_iso > tov.R_edge_iso) {
    Real psi = 1.0 + mass/(2.*r_iso);
    return r_iso*psi*psi;
  }

  int idx = FindIsotropicIndex(tov, r_iso);
  const auto &R_iso = tov.R_iso.d_view;
  const auto &R = tov.R.d_view;
  return Interpolate(r_iso, R_iso(idx), R_iso(idx+1), R(idx), R(idx+1));
}

template<class TOVEOS>
KOKKOS_INLINE_FUNCTION
static void GetPrimitivesAtIsoPoint(const tov_pgen& tov, const TOVEOS& eos, Real r_iso,
                                    Real &rho, Real &p, Real &m, Real &alp) {
  // Check if we're past the edge of the star.
  // If so, we just return atmosphere with Schwarzschild.
  if (r_iso >= tov.R_edge_iso) {
    rho = 0.0;
    p = 0.0;
    m = tov.M_edge;
    alp = (1. - m/(2.*r_iso))/(1. + m/(2.*r_iso));
    return;
  }
  // Because the isotropic coordinates are not evenly spaced, we need to search to find
  // the right index. We can set a lower bound because r_iso <= r, and then we choose the
  // edge of the star as an upper bound.
  const auto &R_iso = tov.R_iso.d_view;
  int idx = FindIsotropicIndex(tov, r_iso);
  const auto &Ps = tov.P.d_view;
  const auto &alps = tov.alp.d_view;
  const auto &Ms = tov.M.d_view;
  if (idx >= tov.npoints || idx < 0) {
    Kokkos::printf("There's a problem with the index!\n" // NOLINT
           " idx = %d\n"
           " r_iso = %g\n"
           " dr = %g\n",idx,r_iso,tov.dr);
  }
  // Interpolate to get the primitive.
  p = Interpolate(r_iso, R_iso(idx), R_iso(idx+1), Ps(idx), Ps(idx+1));
  m = Interpolate(r_iso, R_iso(idx), R_iso(idx+1), Ms(idx), Ms(idx+1));
  alp = Interpolate(r_iso, R_iso(idx), R_iso(idx+1), alps(idx), alps(idx+1));
  // FIXME: Assumes ideal gas!
  //rho = pow(p/tov.kappa, 1.0/tov.gamma);
  rho = eos.template GetRhoFromP<LocationTag::Device>(fmax(p, tov.pfloor));
  if (!isfinite(p)) {
    Kokkos::printf("There's a problem with p!\n"); // NOLINT
    assert(false);
  }
}

template<class TOVEOS>
KOKKOS_INLINE_FUNCTION
static void GetPandRho(const tov_pgen& tov, const TOVEOS& eos,
                       Real r, Real &rho, Real &p) {
  if (r >= tov.R_edge) {
    rho = 0.;
    p   = 0.;
    return;
  }
  // Get the lower index for where our point must be located.
  int idx = static_cast<int>(r/tov.dr);
  const auto &R = tov.R.d_view;
  const auto &Ps = tov.P.d_view;
  // Interpolate to get the pressure
  p = Interpolate(r, R(idx), R(idx+1), Ps(idx), Ps(idx+1));
  // FIXME: Assumes ideal gas!
  //rho = pow(p/tov.kappa, 1.0/tov.gamma);
  rho = eos.template GetRhoFromP<LocationTag::Device>(p);
}

template<class TOVEOS>
KOKKOS_INLINE_FUNCTION
static void GetPandRhoIso(const tov_pgen& tov, const TOVEOS& eos,
                          Real r, Real &rho, Real &p) {
  if (r >= tov.R_edge_iso) {
    rho = 0.;
    p   = 0.;
    return;
  }
  // We need to search to find the right index because isotropic coordinates aren't
  // evenly spaced.
  int idx = FindIsotropicIndex(tov, r);
  const auto R_iso = tov.R_iso.d_view;
  const auto &Ps = tov.P.d_view;
  p = Interpolate(r, R_iso(idx), R_iso(idx+1), Ps(idx), Ps(idx+1));
  // FIXME: Assumes ideal gas!
  //rho = pow(p/tov.kappa, 1.0/tov.gamma);
  rho = eos.template GetRhoFromP<LocationTag::Device>(p);
}

template<class TOVEOS>
KOKKOS_INLINE_FUNCTION
static Real A1(const tov_pgen& tov, const TOVEOS& eos, Real x1, Real x2, Real x3) {
  Real r = sqrt(SQR(x1) + SQR(x2) + SQR(x3));
  Real p, rho;
  if (!tov.isotropic) {
    GetPandRho(tov, eos, r, rho, p);
  } else {
    GetPandRhoIso(tov, eos, r, rho, p);
  }
  return -x2*tov.b_norm*fmax(p - tov.pcut, 0.0)*pow(1.0 - rho/tov.rhoc,tov.magindex);
}

template<class TOVEOS>
KOKKOS_INLINE_FUNCTION
static Real A2(const tov_pgen& tov, const TOVEOS& eos, Real x1, Real x2, Real x3) {
  Real r = sqrt(SQR(x1) + SQR(x2) + SQR(x3));
  Real p, rho;
  if (!tov.isotropic) {
    GetPandRho(tov, eos, r, rho, p);
  } else {
    GetPandRhoIso(tov, eos, r, rho, p);
  }
  return x1*tov.b_norm*fmax(p - tov.pcut, 0.0)*pow(1.0 - rho/tov.rhoc,tov.magindex);
}

KOKKOS_INLINE_FUNCTION
static Real Interpolate(Real x, const Real x1, const Real x2,
                        const Real y1, const Real y2) {
  return ((y2 - y1)*x + (y1*x2 - y2*x1))/(x2 - x1);
}

// Boundary function
void VacuumBC(Mesh *pm) {
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
  auto &b0 = pm->pmb_pack->pmhd->b0;
  int nmb = pm->pmb_pack->nmb_thispack;
  int nvar = u0_.extent_int(1);
  DvceArray5D<Real> z4c;
  int nz4c;
  bool use_z4c = pm->pmb_pack->pz4c != nullptr;
  if (use_z4c) {
    z4c = pm->pmb_pack->pz4c->u0;
    nz4c = pm->pmb_pack->pz4c->nz4c;
  }

  Real &dfloor = pm->pmb_pack->pmhd->peos->eos_data.dfloor;

  // X1-Boundary
  // Set X1-BCs on b0 if Meshblock face is at the edge of the computational domain.
  par_for("noinflow_x1", DevExeSpace(),0,(nmb-1),0,(n3-1),0,(n2-1),
  KOKKOS_LAMBDA(int m, int k, int j) {
    if (mb_bcs.d_view(m,BoundaryFace::inner_x1) == BoundaryFlag::user) {
      for (int i=0; i<ng; ++i) {
        b0.x1f(m,k,j,is-i-1) = b0.x1f(m,k,j,is);
        b0.x2f(m,k,j,is-i-1) = b0.x2f(m,k,j,is);
        if (j == n2-1) {b0.x2f(m,k,j+1,is-i-1) = b0.x2f(m,k,j+1,is);}
        b0.x3f(m,k,j,is-i-1) = b0.x3f(m,k,j,is);
        if (k == n3-1) {b0.x3f(m,k+1,j,is-i-1) = b0.x3f(m,k+1,j,is);}
        u0_(m, IDN, k, j, is-i-1) = dfloor;
        u0_(m, IM1, k, j, is-i-1) = 0.0;
        u0_(m, IM2, k, j, is-i-1) = 0.0;
        u0_(m, IM3, k, j, is-i-1) = 0.0;
        u0_(m, IEN, k, j, is-i-1) = 0.0;
      }
      /*if (use_z4c) {
        for (int i = 0; i < ng; ++i) {
          for (int n = 0; n < nz4c; n++) {
            z4c(m,n,k,j,is-i-1) = z4c(m,n,k,j,is);
          }
        }
      }*/
    }
    if (mb_bcs.d_view(m,BoundaryFace::outer_x1) == BoundaryFlag::user) {
      for (int i=0; i<ng; ++i) {
        b0.x1f(m,k,j,ie+i+2) = b0.x1f(m,k,j,ie+1);
        b0.x2f(m,k,j,ie+i+1) = b0.x2f(m,k,j,ie);
        if (j == n2-1) {b0.x2f(m,k,j+1,ie+i+1) = b0.x2f(m,k,j+1,ie);}
        b0.x3f(m,k,j,ie+i+1) = b0.x3f(m,k,j,ie);
        if (k == n3-1) {b0.x3f(m,k+1,j,ie+i+1) = b0.x3f(m,k+1,j,ie);}
        u0_(m, IDN, k, j, ie+i+1) = dfloor;
        u0_(m, IM1, k, j, ie+i+1) = 0.0;
        u0_(m, IM2, k, j, ie+i+1) = 0.0;
        u0_(m, IM3, k, j, ie+i+1) = 0.0;
        u0_(m, IEN, k, j, ie+i+1) = 0.0;
      }
      /*if (use_z4c) {
        for (int i = 0; i < ng; ++i) {
          for (int n = 0; n < nz4c; n++) {
            z4c(m,n,k,j,ie+i+1) = z4c(m,n,k,j,ie);
          }
        }
      }*/
    }
  });

  // X2-Boundary
  // Set X2-BCs on b0 if Meshblock face is at the edge of computational domain
  par_for("noinflow_field_x2", DevExeSpace(),0,(nmb-1),0,(n3-1),0,(n1-1),
  KOKKOS_LAMBDA(int m, int k, int i) {
    if (mb_bcs.d_view(m,BoundaryFace::inner_x2) == BoundaryFlag::user) {
      for (int j=0; j<ng; ++j) {
        b0.x1f(m,k,js-j-1,i) = b0.x1f(m,k,js,i);
        if (i == n1-1) {b0.x1f(m,k,js-j-1,i+1) = b0.x1f(m,k,js,i+1);}
        b0.x2f(m,k,js-j-1,i) = b0.x2f(m,k,js,i);
        b0.x3f(m,k,js-j-1,i) = b0.x3f(m,k,js,i);
        if (k == n3-1) {b0.x3f(m,k+1,js-j-1,i) = b0.x3f(m,k+1,js,i);}
        u0_(m, IDN, k, js-j-1, i) = dfloor;
        u0_(m, IM1, k, js-j-1, i) = 0.0;
        u0_(m, IM2, k, js-j-1, i) = 0.0;
        u0_(m, IM3, k, js-j-1, i) = 0.0;
        u0_(m, IEN, k, js-j-1, i) = 0.0;
      }
      /*if (use_z4c) {
        for (int j = 0; j < ng; ++j) {
          for (int n = 0; n < nz4c; ++n) {
            z4c(m,n,k,js-j-1,i) = z4c(m,n,k,js,i);
          }
        }
      }*/
    }
    if (mb_bcs.d_view(m,BoundaryFace::outer_x2) == BoundaryFlag::user) {
      for (int j=0; j<ng; ++j) {
        b0.x1f(m,k,je+j+1,i) = b0.x1f(m,k,je,i);
        if (i == n1-1) {b0.x1f(m,k,je+j+1,i+1) = b0.x1f(m,k,je,i+1);}
        b0.x2f(m,k,je+j+2,i) = b0.x2f(m,k,je+1,i);
        b0.x3f(m,k,je+j+1,i) = b0.x3f(m,k,je,i);
        if (k == n3-1) {b0.x3f(m,k+1,je+j+1,i) = b0.x3f(m,k+1,je,i);}
        u0_(m, IDN, k, je+j+1, i) = dfloor;
        u0_(m, IM1, k, je+j+1, i) = 0.0;
        u0_(m, IM2, k, je+j+1, i) = 0.0;
        u0_(m, IM3, k, je+j+1, i) = 0.0;
        u0_(m, IEN, k, je+j+1, i) = 0.0;
      }
      /*if (use_z4c) {
        for (int j = 0; j < ng; ++j) {
          for (int n = 0; n < nz4c; ++n) {
            z4c(m,n,k,je+j+1,i) = z4c(m,n,k,je,i);
          }
        }
      }*/
    }
  });

  // X3-Boundary
  // Set X3-BCs on b0 if Meshblock face is at the edge of computational domain
  par_for("noinflow_field_x3", DevExeSpace(),0,(nmb-1),0,(n2-1),0,(n1-1),
  KOKKOS_LAMBDA(int m, int j, int i) {
    if (mb_bcs.d_view(m,BoundaryFace::inner_x3) == BoundaryFlag::user) {
      for (int k=0; k<ng; ++k) {
        b0.x1f(m,ks-k-1,j,i) = b0.x1f(m,ks,j,i);
        if (i == n1-1) {b0.x1f(m,ks-k-1,j,i+1) = b0.x1f(m,ks,j,i+1);}
        b0.x2f(m,ks-k-1,j,i) = b0.x2f(m,ks,j,i);
        if (j == n2-1) {b0.x2f(m,ks-k-1,j+1,i) = b0.x2f(m,ks,j+1,i);}
        b0.x3f(m,ks-k-1,j,i) = b0.x3f(m,ks,j,i);
        u0_(m, IDN, ks-k-1, j, i) = dfloor;
        u0_(m, IM1, ks-k-1, j, i) = 0.0;
        u0_(m, IM2, ks-k-1, j, i) = 0.0;
        u0_(m, IM3, ks-k-1, j, i) = 0.0;
        u0_(m, IEN, ks-k-1, j, i) = 0.0;
      }
      /*if (use_z4c) {
        for (int k = 0; k < ng; ++k) {
          for (int n = 0; n < nz4c; ++n) {
            z4c(m,n,ks-k-1,j,i) = z4c(m,n,ks,j,i);
          }
        }
      }*/
    }
    if (mb_bcs.d_view(m,BoundaryFace::outer_x3) == BoundaryFlag::user) {
      for (int k=0; k<ng; ++k) {
        b0.x1f(m,ke+k+1,j,i) = b0.x1f(m,ke,j,i);
        if (i == n1-1) {b0.x1f(m,ke+k+1,j,i+1) = b0.x1f(m,ke,j,i+1);}
        b0.x2f(m,ke+k+1,j,i) = b0.x2f(m,ke,j,i);
        if (j == n2-1) {b0.x2f(m,ke+k+1,j+1,i) = b0.x2f(m,ke,j+1,i);}
        b0.x3f(m,ke+k+2,j,i) = b0.x3f(m,ke+1,j,i);
        u0_(m, IDN, ke+k+1, j, i) = dfloor;
        u0_(m, IM1, ke+k+1, j, i) = 0.0;
        u0_(m, IM2, ke+k+1, j, i) = 0.0;
        u0_(m, IM3, ke+k+1, j, i) = 0.0;
        u0_(m, IEN, ke+k+1, j, i) = 0.0;
      }
      /*if (use_z4c) {
        for (int k = 0; k < ng; ++k) {
          for (int n = 0; n < nz4c; ++n) {
            z4c(m,n,ke+k+1,j,i) = z4c(m,n,ke,j,i);
          }
        }
      }*/
    }
  });
}

// History function
void TOVHistory(HistoryData *pdata, Mesh *pm) {
  // Select the number of outputs and create labels for them.
  pdata->nhist = 2;
  pdata->label[0] = "rho-max";
  pdata->label[1] = "alpha-min";

  // capture class variables for kernel
  auto &w0_ = pm->pmb_pack->pmhd->w0;
  auto &adm = pm->pmb_pack->padm->adm;

  // loop over all MeshBlocks in this pack
  auto &indcs = pm->pmb_pack->pmesh->mb_indcs;
  int is = indcs.is; int nx1 = indcs.nx1;
  int js = indcs.js; int nx2 = indcs.nx2;
  int ks = indcs.ks; int nx3 = indcs.nx3;
  const int nmkji = (pm->pmb_pack->nmb_thispack)*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1;
  const int nji = nx2*nx1;
  Real rho_max = std::numeric_limits<Real>::max();
  Real alpha_min = -rho_max;
  Kokkos::parallel_reduce("TOVHistSums",Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, Real &mb_max, Real &mb_alp_min) {
    // coompute n,k,j,i indices of thread
    int m = (idx)/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks;
    j += js;

    mb_max = fmax(mb_max, w0_(m,IDN,k,j,i));
    mb_alp_min = fmin(mb_alp_min, adm.alpha(m, k, j, i));
  }, Kokkos::Max<Real>(rho_max), Kokkos::Min<Real>(alpha_min));

  // Currently AthenaK only supports MPI_SUM operations between ranks, but we need MPI_MAX
  // and MPI_MIN operations instead. This is a cheap hack to make it work as intended.
#if MPI_PARALLEL_ENABLED
  if (global_variable::my_rank == 0) {
    MPI_Reduce(MPI_IN_PLACE, &rho_max, 1, MPI_ATHENA_REAL, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE, &alpha_min, 1, MPI_ATHENA_REAL, MPI_MIN, 0, MPI_COMM_WORLD);
  } else {
    MPI_Reduce(&rho_max, &rho_max, 1, MPI_ATHENA_REAL, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&alpha_min, &alpha_min, 1, MPI_ATHENA_REAL, MPI_MIN, 0, MPI_COMM_WORLD);
    rho_max = 0.;
    alpha_min = 0.;
  }
#endif

  // store data in hdata array
  pdata->hdata[0] = rho_max;
  pdata->hdata[1] = alpha_min;
}
