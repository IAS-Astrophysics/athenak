//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file kadath_bhns.cpp
//! \brief Initial data reader for black hole-neutron star data from Kadath (FUKa)
//
//  Inlines the spectral interpolation from Kadath/src/Utilities/Exporters/export_bhns.cpp
//  directly into the fill loop, eliminating the large intermediate output arrays and
//  performing a single pass over all grid points.
//
//  NOTE ON PARALLELISM: Kadath's MemoryMapper and coef_1d scratch pools are
//  thread_local (patched in memory.hpp/.cpp and coef_1d.cpp), so val_point() and
//  Point() are safe to call concurrently.  The per-point loop runs via
//  Kokkos::parallel_for on DefaultHostExecutionSpace (OpenMP threads).  A serial
//  warmup call before the loop initialises the summation_1d static dispatch table
//  on the main thread, preventing a first-call race among OMP threads.
//
//  Required input block:
//    <problem>
//      initial_data_file = path/to/bhns.info   # Kadath config file (.info)
//
//  The Kadath space file must reside next to the .info file (same stem, .dat extension).

#include <cmath>
#include <cstdio>

#include <array>
#include <functional>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "athena.hpp"
#include "coordinates/adm.hpp"
#include "coordinates/cell_locations.hpp"
#include "coordinates/coordinates.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"
#include "eos/eos.hpp"
#include "globals.hpp"
#include "hydro/hydro.hpp"
#include "mesh/mesh.hpp"
#include "mhd/mhd.hpp"
#include "parameter_input.hpp"
#include "z4c/compact_object_tracker.hpp"
#include "z4c/z4c.hpp"
#include "z4c/z4c_amr.hpp"
#include "utils/tov/tov_utils.hpp"
#include "utils/tov/tov_polytrope.hpp"
#include "utils/tov/tov_piecewise_poly.hpp"
#include "utils/tov/tov_tabulated.hpp"

// Kadath FUKa
#include "kadath.hpp"
#include "EOS/EOS.hh"
#include "coord_fields.hpp"
#include "Configurator/config_enums.hpp"
#include "Configurator/config_binary.hpp"
#include "Configurator/configurator_boost.hpp"
#include "exporter_utilities.hpp"

void KadathBHNSHistory(HistoryData *pdata, Mesh *pm);
void KadathBHNSRefinementCondition(MeshBlockPack *pmbp);

// Prototypes for magnetic vector potential
KOKKOS_INLINE_FUNCTION
static Real A1(Real x, Real y, Real z, Real I_0, Real r_0);
KOKKOS_INLINE_FUNCTION
static Real A2(Real x, Real y, Real z, Real I_0, Real r_0);

//----------------------------------------------------------------------------------------
//! \fn void::SetupBHNS(ParameterInput *pin, Mesh *pmy_mesh_)
//! \brief Problem generator for BHNS with Kadath (FUKa)
template<class TOVEOS>
void SetupBHNS(ParameterInput *pin, Mesh* pmy_mesh_) {
  // export_utils: field-index enumerators
  using export_utils::PSI;
  using export_utils::ALP;
  using export_utils::BETX;
  using export_utils::BETY;
  using export_utils::BETZ;
  using export_utils::AXX;
  using export_utils::AXY;
  using export_utils::AXZ;
  using export_utils::AYY;
  using export_utils::AYZ;
  using export_utils::AZZ;
  using export_utils::H;
  using export_utils::UX;
  using export_utils::UY;
  using export_utils::UZ;
  using export_utils::NUM_QUANTS;
  using export_utils::point_spherical;
  using export_utils::lagrange_gen_k;

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  auto &indcs         = pmy_mesh_->mb_indcs;
  auto &size          = pmbp->pmb->mb_size;
  int &is             = indcs.is;
  int &ie             = indcs.ie;
  int &js             = indcs.js;
  int &je             = indcs.je;
  int &ks             = indcs.ks;
  int &ke             = indcs.ke;

  std::string fname = pin->GetString("problem", "initial_data_file");
  const int interp_order = pin->GetOrAddInteger("problem", "interpolation_order", 8);
  const Real interp_offset = pin->GetOrAddReal("problem", "interpolation_offset", 0.0);
  const Real delta_r_rel = pin->GetOrAddReal("problem", "relative_dr_spacing", 0.3);

  // MHD parameters
  Real gauss_cgs_to_geo = 8.3519664583273e+19;
  Real b_max   = pin->GetOrAddReal("problem", "b_max", 1e12) / gauss_cgs_to_geo;
  Real r_0     = pin->GetOrAddReal("problem", "r_0_current", 5.0);
  Real I_0     = 4 * r_0 * b_max / (23.0 * M_PI);

  int ncells1      = indcs.nx1 + 2 * (indcs.ng);
  int ncells2      = indcs.nx2 + 2 * (indcs.ng);
  int ncells3      = indcs.nx3 + 2 * (indcs.ng);
  int nmb          = pmbp->nmb_thispack;
  int width        = nmb * ncells1 * ncells2 * ncells3;
  int ncells_per_mb = ncells3 * ncells2 * ncells1;

  // Set up the 1D EOS
  TOVEOS eos{pin};

  // Enable ye only if the EOS supports it AND nscalars > 0 (IYF slot exists).
  const bool use_ye = tov::UsesYe<TOVEOS>;
  const bool read_ye = pin->GetOrAddInteger("mhd", "nscalars", 0) > 0;

  // =========================================================================
  // Kadath BHNS setup (inlined from export_bhns.cpp / KadathExportBHNS)
  // =========================================================================

  if (global_variable::my_rank == 0) {
    std::cout << "Reading Kadath BHNS config from " << fname << " ..." << std::endl;
  }

  kadath_config_boost<BIN_INFO> bconfig(fname);

  const Real h_cut      = bconfig.eos<Real>(HCUT, BCO1);
  const std::string eos_file = bconfig.eos<std::string>(EOSFILE, BCO1);
  const std::string eos_type = bconfig.eos<std::string>(EOSTYPE, BCO1);

  Real &units    = bconfig(QPIG);
  Real &sep      = bconfig(DIST);
  Real &omega    = bconfig(GOMEGA);
  Real &ome1     = bconfig(OMEGA, BCO1);
  Real &ome2     = bconfig(OMEGA, BCO2);
  Real &axis     = bconfig(COM);
  Real &axisy		 = bconfig(COMY);
  Real &q        = bconfig(Q);

  std::string kadath_filename = bconfig.space_filename();

  FILE *fin = fopen(kadath_filename.c_str(), "r");
  Space_bhns space(fin);
  Scalar conf (space, fin);
  Scalar lapse(space, fin);
  Vector shift(space, fin);
  Scalar logh (space, fin);
  Scalar phi  (space, fin);
  fclose(fin);

  // Build the quants array: references to const Scalar fields.
  std::vector<std::reference_wrapper<const Scalar>> quants;
  quants.reserve(NUM_QUANTS);
  for (int i = 0; i < NUM_QUANTS; ++i)
    quants.push_back(std::cref(conf));  // placeholder, overwritten below

  quants[PSI]  = std::cref(conf);
  quants[ALP]  = std::cref(lapse);
  quants[BETX] = std::cref(shift(1));
  quants[BETY] = std::cref(shift(2));
  quants[BETZ] = std::cref(shift(3));

  Base_tensor basis(shift.get_basis());
  Real rbh; // Referenced below
  int ndom = space.get_nbr_domains();

  Index center_pos (space.get_domain(space.NS)->get_nbr_points());
  Real xm = space.get_domain(space.NS)->get_cart(1)(center_pos);
  Real xp = space.get_domain(space.BH)->get_cart(1)(center_pos);
  Real ym = space.get_domain(space.NS)->get_cart(2)(center_pos);
  Real yp = space.get_domain(space.BH)->get_cart(2)(center_pos);
  Real xo  = space.get_domain(ndom-1)->get_cart(1)(center_pos);

  /* Setup system of equations for Aij and Matter terms */
  Metric_flat fmet(space, basis);

  CoordFields<Space_bhns> cfields(space);
  Vector global_rot = cfields.rot_z();
  Vector star_rot = cfields.rot_z(xm);
  Vector ey = cfields.e_cart(2);
  Vector ex = cfields.e_cart(1);
  Vector esurf = cfields.e_rad();

  System_of_eqs syst(space, 0, ndom - 1);
  fmet.set_system(syst, "f");

  Param p;
  if (eos_type == "Cold_Table") {
    using eos_t = Kadath::Margherita::Cold_Table;
    const int interp_pts = (bconfig.eos<int>(INTERP_PTS, BCO1) == 0)
                           ? 2000 : bconfig.eos<int>(INTERP_PTS, BCO1);
    EOS<eos_t, PRESSURE>::init(eos_file, h_cut, interp_pts);
    syst.add_ope("eps",   &EOS<eos_t, EPSILON>::action,  &p);
    syst.add_ope("press", &EOS<eos_t, PRESSURE>::action, &p);
    syst.add_ope("rho",   &EOS<eos_t, DENSITY>::action,  &p);
  } else if (eos_type == "Cold_PWPoly") {
    using eos_t = Kadath::Margherita::Cold_PWPoly;
    EOS<eos_t, PRESSURE>::init(eos_file, h_cut);
    syst.add_ope("eps",   &EOS<eos_t, EPSILON>::action,  &p);
    syst.add_ope("press", &EOS<eos_t, PRESSURE>::action, &p);
    syst.add_ope("rho",   &EOS<eos_t, DENSITY>::action,  &p);
  }

  syst.add_cst("4piG", units);
  syst.add_cst("PI", M_PI);
  syst.add_cst("omes1", ome1);

  syst.add_cst("Mg", global_rot);
  syst.add_cst("ome", omega);
  syst.add_cst("xaxis", axis);
  syst.add_cst("yaxis", axisy);
  syst.add_cst("ex", ex);
  syst.add_cst("ey", ey);

  syst.add_cst("m1", star_rot);
  syst.add_cst("P", conf);
  syst.add_cst("N", lapse);
  syst.add_cst("bet", shift);
  syst.add_cst("phi", phi);
  syst.add_cst("H", logh);

  syst.add_def("NP = P*N");
  syst.add_def("Ntilde = N / P^6");
  syst.add_def("Morb^i = Mg^i + xaxis * ey^i + yaxis * ex^i");
  syst.add_def("B^i = bet^i + ome * Morb^i");

  for (int d = space.NS; d <= space.ADAPTEDNS; ++d) {
    syst.add_def(d, "s^i  = omes1 * m1^i");
  }

  syst.add_def("A_ij = (D_i bet_j + D_j bet_i - 2. / 3.* D^k bet_k * f_ij) /2. / N");
  syst.add_def("h = exp(H)");

  for (int d = 0; d < ndom; ++d) {
    if (d <= space.ADAPTEDNS)
      syst.add_def(d, "eta_i = D_i phi + P^4 * s_i");
    else
      syst.add_def(d, "eta_i = D_i phi");
  }

  syst.add_def("Wsquare = eta^i * eta_i / h^2 / P^4 + 1.");
  syst.add_def("W = sqrt(Wsquare)");
  syst.add_def("U^i = eta^i / P^4 / h / W");

  // Evaluate derived tensor fields (A and U).
  Tensor A_tens(syst.give_val_def("A"));
  Index  ind(A_tens);
  quants[AXX] = std::cref(A_tens(ind));
  ind.inc();
  quants[AXY] = std::cref(A_tens(ind));
  ind.inc();
  quants[AXZ] = std::cref(A_tens(ind));
  ind.inc();
  ind.inc();
  quants[AYY] = std::cref(A_tens(ind));
  ind.inc();
  quants[AYZ] = std::cref(A_tens(ind));
  ind.inc();
  ind.inc();
  ind.inc();
  quants[AZZ] = std::cref(A_tens(ind));

  quants[H] = std::cref(logh);

  Vector vel_kad(syst.give_val_def("U"));
  quants[UX] = std::cref(vel_kad(1));
  quants[UY] = std::cref(vel_kad(2));
  quants[UZ] = std::cref(vel_kad(3));

  /* setup for filling in BH with junk */
  Index I2(space.get_domain(space.BH+2)->get_radius().get_conf().get_dimensions());
  rbh = space.get_domain(space.BH+2)->get_radius()(I2);
  /* end BH setup */

  // Force spectral-coefficient transform for every field once (serial, one-time).
  for (int kq = 0; kq < NUM_QUANTS; ++kq)
    quants[kq].get().coef();

  if (global_variable::my_rank == 0) {
    std::cout << "Kadath system assembled. Starting per-point interpolation..."
              << std::endl;
  }

  // Hoist EOS type checks out of the hot loop.
  const bool use_cold_table  = (eos_type == "Cold_Table");
  const bool use_cold_pwpoly = (eos_type == "Cold_PWPoly");

  // =========================================================================
  // Host-mirror setup
  // =========================================================================
  auto &u_adm = pmbp->padm->u_adm;
  auto &w0    = pmbp->pmhd->w0;
  auto &u_z4c = pmbp->pz4c->u0;

  HostArray5D<Real>::HostMirror host_u_adm = create_mirror_view(u_adm);
  HostArray5D<Real>::HostMirror host_w0    = create_mirror_view(w0);
  HostArray5D<Real>::HostMirror host_u_z4c = create_mirror_view(u_z4c);

  adm::ADM::ADMhost_vars host_adm;
  host_adm.alpha.InitWithShallowSlice(host_u_z4c, z4c::Z4c::I_Z4C_ALPHA);
  host_adm.beta_u.InitWithShallowSlice(host_u_z4c, z4c::Z4c::I_Z4C_BETAX,
                                        z4c::Z4c::I_Z4C_BETAZ);
  host_adm.g_dd.InitWithShallowSlice(host_u_adm, adm::ADM::I_ADM_GXX,
                                      adm::ADM::I_ADM_GZZ);
  host_adm.vK_dd.InitWithShallowSlice(host_u_adm, adm::ADM::I_ADM_KXX,
                                       adm::ADM::I_ADM_KZZ);

  // Warm up the summation_1d static dispatch table
  // on the main thread before the parallel loop.
  {
    Point pt_warm(3);
    pt_warm.set(1) = xm;
    pt_warm.set(2) = 0.0;
    pt_warm.set(3) = 0.0;
    (void)quants[PSI].get().val_point(pt_warm);
  }

  Kokkos::parallel_for("kadath_fill",
  Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, width),
  [&](const int idx) {
    int m   = idx / ncells_per_mb;
    int rem = idx - m * ncells_per_mb;
    int k   = rem / (ncells2 * ncells1);
    rem    -= k * ncells2 * ncells1;
    int j   = rem / ncells1;
    int i   = rem % ncells1;

    // Cell-centre coordinates.
    Real x = CellCenterX(i - is, indcs.nx1,
                          size.h_view(m).x1min, size.h_view(m).x1max);
    Real y = CellCenterX(j - js, indcs.nx2,
                          size.h_view(m).x2min, size.h_view(m).x2max);
    Real z = CellCenterX(k - ks, indcs.nx3,
                          size.h_view(m).x3min, size.h_view(m).x3max);

    // Kadath point shifted to the centre-of-mass frame.
    Real qv[NUM_QUANTS];
    Real x_com = static_cast<Real>(x) - axis;
    Real y_com = static_cast<Real>(y) - axisy;
    Real z_		 = static_cast<Real>(z);
    Real xxp   = x_com - xp;
    Real r2yz  = SQR(y_com) + SQR(z_);

    // Radius measurement centered on the BH.
    Real r_plus = sqrt(SQR(xxp) + r2yz);

    // Lambda function for interpolation inside BH.
    auto interp_f = [&](auto &ah_r, auto &extrap_r, auto &bh_ori) {
      Real theta = acos(z_ / extrap_r);
      Real phi   = atan2(y_com, (x_com - bh_ori));

      std::vector<Real> r_points(interp_order);
      for (int j = 0; j < interp_order; j++) {
        r_points[j] = (1.0 + interp_offset) * (1.0 + j * delta_r_rel) * ah_r;
      }

      for (int k = 0; k < NUM_QUANTS; ++k) {
        std::vector<Real> vals(interp_order);

        for (int j = 0; j < interp_order; j++) {
          vals[j] = quants[k].get().val_point(
              point_spherical(r_points[j], theta, phi, bh_ori));
        }

        if (k == H) { qv[k] = 0.0; }
        else if (k == UX || k == UY || k == UZ) {
          qv[k] = 0.0;
        } else {
          qv[k] = lagrange_gen_k(interp_order, extrap_r, r_points.data(), vals.data());
        }
      }
    };

    if (r_plus <= (1.0 + interp_offset) * rbh) {
      interp_f(rbh, r_plus, xp);
    } else {
      Point pt(3);
      pt.set(1) = x_com;
      pt.set(2) = y_com;
      pt.set(3) = z;

      // Evaluate all spectral quantities at this point.
      for (int kq = 0; kq < NUM_QUANTS; ++kq) {
        qv[kq] = quants[kq].get().val_point(pt);
      }
    }

    // Conformal factor and derived powers.
    const Real psi  = qv[PSI];
    const Real psi4 = psi * psi * psi * psi;

    // Lapse and shift.
    host_adm.alpha(m, k, j, i)     = qv[ALP];
    host_adm.beta_u(m, 0, k, j, i) = qv[BETX];
    host_adm.beta_u(m, 1, k, j, i) = qv[BETY];
    host_adm.beta_u(m, 2, k, j, i) = qv[BETZ];

    // Spatial metric: g_ij = psi^4 * delta_ij
    Real g3d[NSPMETRIC];
    host_adm.g_dd(m, 0, 0, k, j, i) = g3d[S11] = static_cast<Real>(psi4);
    host_adm.g_dd(m, 0, 1, k, j, i) = g3d[S12] = 0.0;
    host_adm.g_dd(m, 0, 2, k, j, i) = g3d[S13] = 0.0;
    host_adm.g_dd(m, 1, 1, k, j, i) = g3d[S22] = static_cast<Real>(psi4);
    host_adm.g_dd(m, 1, 2, k, j, i) = g3d[S23] = 0.0;
    host_adm.g_dd(m, 2, 2, k, j, i) = g3d[S33] = static_cast<Real>(psi4);

    // Extrinsic curvature: K_ij = psi^4 * A_ij
    host_adm.vK_dd(m, 0, 0, k, j, i) = qv[AXX] * psi4;
    host_adm.vK_dd(m, 0, 1, k, j, i) = qv[AXY] * psi4;
    host_adm.vK_dd(m, 0, 2, k, j, i) = qv[AXZ] * psi4;
    host_adm.vK_dd(m, 1, 1, k, j, i) = qv[AYY] * psi4;
    host_adm.vK_dd(m, 1, 2, k, j, i) = qv[AYZ] * psi4;
    host_adm.vK_dd(m, 2, 2, k, j, i) = qv[AZZ] * psi4;

    // Hydro: qv[H] = log(h), h = specific enthalpy.
    const Real h_enth = Kokkos::exp(qv[H]);
    if (h_enth == 1.) {
      // Vacuum: set to atmosphere values.
      host_w0(m, IDN, k, j, i) = 0.0;
      host_w0(m, IPR, k, j, i) = 0.0;
    } else {
      if (use_cold_table) {
        using eos_t = Kadath::Margherita::Cold_Table;
        host_w0(m, IDN, k, j, i) = EOS<eos_t, DENSITY>::get(h_enth);
        host_w0(m, IPR, k, j, i) = EOS<eos_t, PRESSURE>::get(h_enth);
      } else if (use_cold_pwpoly) {
        using eos_t = Kadath::Margherita::Cold_PWPoly;
        host_w0(m, IDN, k, j, i) = EOS<eos_t, DENSITY>::get(h_enth);
        host_w0(m, IPR, k, j, i) = EOS<eos_t, PRESSURE>::get(h_enth);
      }
    }

    if constexpr (use_ye) {
      if (read_ye) {
        Real& rho = host_w0(m, IDN, k, j, i);
        host_w0(m, IYF, k, j, i) = eos.template
                                   GetYeFromRho<tov::LocationTag::Host>(rho);
      }
    }

    // Velocity gamma^ij U_j
    Real vu[3] = {static_cast<Real>(qv[UX]),
                  static_cast<Real>(qv[UY]),
                  static_cast<Real>(qv[UZ])};

    Real vsq = Primitive::SquareVector(vu, g3d);
    if (1.0 - vsq <= 0.0) {
      Real fac = sqrt((1.0 - 1e-15) / vsq);
      vu[0] *= fac;
      vu[1] *= fac;
      vu[2] *= fac;
      vsq = 1.0 - 1.0e-15;
    }

    Real W = sqrt(1.0 / (1.0 - vsq));
    host_w0(m, IVX, k, j, i) = W * vu[0];
    host_w0(m, IVY, k, j, i) = W * vu[1];
    host_w0(m, IVZ, k, j, i) = W * vu[2];
  });
  Kokkos::fence();

  if (global_variable::my_rank == 0) {
    std::cout << "Per-point interpolation complete. Copying to device..." << std::endl;
  }

  // Copy data from host mirrors to the device.
  Kokkos::deep_copy(u_adm, host_u_adm);
  Kokkos::deep_copy(w0, host_w0);
  Kokkos::deep_copy(u_z4c, host_u_z4c);

  Real COM_corr_NS[3] = {xm + axis, ym + axisy, 0.0};
  Real COM_corr_BH[3] = {xp + axis, yp + axisy, 0.0};
  if (pmbp->pz4c->ptracker[0]->GetType() == 0) { // CompactObjectTracker::BlackHole == 0
    pmbp->pz4c->ptracker[0]->SetPos(COM_corr_BH);
    pmbp->pz4c->ptracker[1]->SetPos(COM_corr_NS);

    if (global_variable::my_rank == 0) {
      std::cout << "Adjusted CompactObjectTracker position by COM." << std::endl;
      std::cout << "BH: cx = " << pmbp->pz4c->ptracker[0]->GetPos(0)
                << ", cy = " << pmbp->pz4c->ptracker[0]->GetPos(1)
                << ", cz = " << pmbp->pz4c->ptracker[0]->GetPos(2) << std::endl;
      std::cout << "NS: cx = " << pmbp->pz4c->ptracker[1]->GetPos(0)
                << ", cy = " << pmbp->pz4c->ptracker[1]->GetPos(1)
                << ", cz = " << pmbp->pz4c->ptracker[1]->GetPos(2) << std::endl;
    }
  } else {
    pmbp->pz4c->ptracker[1]->SetPos(COM_corr_BH);
    pmbp->pz4c->ptracker[0]->SetPos(COM_corr_NS);

    if (global_variable::my_rank == 0) {
      std::cout << "Adjusted CompactObjectTracker position by COM." << std::endl;
      std::cout << "BH: cx = " << pmbp->pz4c->ptracker[1]->GetPos(0)
                << ", cy = " << pmbp->pz4c->ptracker[1]->GetPos(1)
                << ", cz = " << pmbp->pz4c->ptracker[1]->GetPos(2) << std::endl;
      std::cout << "NS: cx = " << pmbp->pz4c->ptracker[0]->GetPos(0)
                << ", cy = " << pmbp->pz4c->ptracker[0]->GetPos(1)
                << ", cz = " << pmbp->pz4c->ptracker[0]->GetPos(2) << std::endl;
    }
  }

  // compute vector potential over all faces
  DvceArray4D<Real> a1, a2, a3;
  Kokkos::realloc(a1, nmb,ncells3,ncells2,ncells1);
  Kokkos::realloc(a2, nmb,ncells3,ncells2,ncells1);
  Kokkos::realloc(a3, nmb,ncells3,ncells2,ncells1);

  auto &nghbr = pmbp->pmb->nghbr;
  auto &mblev = pmbp->pmb->mb_lev;

  par_for("pgen_vector_potential", DevExeSpace(), 0,nmb-1,ks,ke+1,js,je+1,is,ie+1,
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

    if (xm > 0.0) {
      a1(m,k,j,i) = A1(x1v - 0.5 * sep - axis, x2f, x3f, I_0, r_0);
      a2(m,k,j,i) = A2(x1f - 0.5 * sep - axis, x2v, x3f, I_0, r_0);
      a3(m,k,j,i) = 0.0;
    } else {
      a1(m,k,j,i) = A1(x1v + 0.5 * sep - axis, x2f, x3f, I_0, r_0);
      a2(m,k,j,i) = A2(x1f + 0.5 * sep - axis, x2v, x3f, I_0, r_0);
      a3(m,k,j,i) = 0.0;
    }

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

      if (xm > 0.0) {
        a1(m,k,j,i) = 0.5*(A1(xl - 0.5 * sep - axis, x2f, x3f, I_0, r_0) +
                           A1(xr - 0.5 * sep - axis, x2f, x3f, I_0, r_0));
      } else {
        a1(m,k,j,i) = 0.5*(A1(xl + 0.5 * sep - axis, x2f, x3f, I_0, r_0) +
                           A1(xr + 0.5 * sep - axis, x2f, x3f, I_0, r_0));
      }
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

      if (xm > 0.0) {
        a2(m,k,j,i) = 0.5*(A2(x1f - 0.5 * sep - axis, xl, x3f, I_0, r_0) +
                           A2(x1f - 0.5 * sep - axis, xr, x3f, I_0, r_0));
      } else {
        a2(m,k,j,i) = 0.5*(A2(x1f + 0.5 * sep - axis, xl, x3f, I_0, r_0) +
                           A2(x1f + 0.5 * sep - axis, xr, x3f, I_0, r_0));
      }
    }
  });

  auto &b0 = pmbp->pmhd->b0;
  par_for("pgen_Bfc", DevExeSpace(), 0, nmb-1,ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    // Compute face-centered fields from curl(A).
    Real dx1 = size.d_view(m).dx1;
    Real dx2 = size.d_view(m).dx2;
    Real dx3 = size.d_view(m).dx3;

    b0.x1f(m, k, j, i) = ((a3(m,k,j+1,i) - a3(m,k,j,i))/dx2 -
                          (a2(m,k+1,j,i) - a2(m,k,j,i))/dx3);
    b0.x2f(m, k, j, i) = ((a1(m,k+1,j,i) - a1(m,k,j,i))/dx3 -
                          (a3(m,k,j,i+1) - a3(m,k,j,i))/dx1);
    b0.x3f(m, k, j, i) = ((a2(m,k,j,i+1) - a2(m,k,j,i))/dx1 -
                          (a1(m,k,j+1,i) - a1(m,k,j,i))/dx2);


    // Include extra face-component at edge of block in each direction
    if (i == ie) {
      b0.x1f(m, k, j, i+1) = ((a3(m,k,j+1,i+1) - a3(m,k,j,i+1))/dx2 -
                              (a2(m,k+1,j,i+1) - a2(m,k,j,i+1))/dx3);
    }
    if (j == je) {
      b0.x2f(m, k, j+1, i) = ((a1(m,k+1,j+1,i) - a1(m,k,j+1,i))/dx3 -
                              (a3(m,k,j+1,i+1) - a3(m,k,j+1,i))/dx1);
    }
    if (k == ke) {
      b0.x3f(m, k+1, j ,i) = ((a2(m,k+1,j,i+1) - a2(m,k+1,j,i))/dx1 -
                              (a1(m,k+1,j+1,i) - a1(m,k+1,j,i))/dx2);
    }
  });

  if (global_variable::my_rank == 0) {
    std::cout << "Face-centered fields calculated." << std::endl;
  }

  // Compute cell-centered fields
  auto &bcc0 = pmbp->pmhd->bcc0;
  par_for("pgen_bcc", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    bcc0(m, IBX, k, j, i) = 0.5*(b0.x1f(m, k, j, i) + b0.x1f(m, k, j, i+1));
    bcc0(m, IBY, k, j, i) = 0.5*(b0.x2f(m, k, j, i) + b0.x2f(m, k, j+1, i));
    bcc0(m, IBZ, k, j, i) = 0.5*(b0.x3f(m, k, j, i) + b0.x3f(m, k+1, j, i));
  });

  if (global_variable::my_rank == 0) {
    std::cout << "Cell-centered fields calculated." << std::endl;
  }

  // Convert primitive hydro variables to conserved.
  pmbp->pdyngr->PrimToConInit(0, (ncells1 - 1), 0, (ncells2 - 1),
                               0, (ncells3 - 1));

  // Convert ADM variables to Z4c variables.
  switch (indcs.ng) {
    case 2:
      pmbp->pz4c->ADMToZ4c<2>(pmbp, pin);
      break;
    case 3:
      pmbp->pz4c->ADMToZ4c<3>(pmbp, pin);
      break;
    case 4:
      pmbp->pz4c->ADMToZ4c<4>(pmbp, pin);
      break;
  }
}

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem_()
//! \brief Problem Generator for BHNS with Kadath FUKa
void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (!pmbp->pcoord->is_dynamical_relativistic) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Kadath BNS problem must have <adm> block to run"
              << std::endl;
    exit(EXIT_FAILURE);
  }
  user_hist_func = &KadathBHNSHistory;
  user_ref_func = &KadathBHNSRefinementCondition;

  if (restart) return;

  // Select the correct EOS template based on the EOS we need.
  if (pmbp->pdyngr->eos_policy == DynGRMHD_EOS::eos_ideal) {
    SetupBHNS<tov::PolytropeEOS>(pin, pmy_mesh_);
  } else if (pmbp->pdyngr->eos_policy == DynGRMHD_EOS::eos_compose) {
    SetupBHNS<tov::TabulatedEOS>(pin, pmy_mesh_);
  } else if (pmbp->pdyngr->eos_policy == DynGRMHD_EOS::eos_hybrid) {
    SetupBHNS<tov::TabulatedEOS>(pin, pmy_mesh_);
  } else if (pmbp->pdyngr->eos_policy == DynGRMHD_EOS::eos_piecewise_poly) {
    SetupBHNS<tov::PiecewisePolytropeEOS>(pin, pmy_mesh_);
  } else {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Unknown EOS requested for Lorene BNS problem" << std::endl;
    exit(EXIT_FAILURE);
  }

  return;
}


//----------------------------------------------------------------------------------------
//! \fn KadathBHNSHistory()
//! \brief History function: tracks maximum rest-mass density and minimum lapse
void KadathBHNSHistory(HistoryData *pdata, Mesh *pm) {
  pdata->nhist    = 2;
  pdata->label[0] = "rho-max";
  pdata->label[1] = "alpha-min";

  auto &w0_ = pm->pmb_pack->pmhd->w0;
  auto &adm = pm->pmb_pack->padm->adm;

  auto &indcs     = pm->pmb_pack->pmesh->mb_indcs;
  int is          = indcs.is;
  int nx1         = indcs.nx1;
  int js          = indcs.js;
  int nx2         = indcs.nx2;
  int ks          = indcs.ks;
  int nx3         = indcs.nx3;
  const int nmkji = (pm->pmb_pack->nmb_thispack) * nx3 * nx2 * nx1;
  const int nkji  = nx3 * nx2 * nx1;
  const int nji   = nx2 * nx1;

  Real rho_max   = std::numeric_limits<Real>::max();
  Real alpha_min = -rho_max;

  Kokkos::parallel_reduce(
      "KadathBHNSHistSums",
      Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
      KOKKOS_LAMBDA(const int &idx, Real &mb_max, Real &mb_alp_min) {
        int m = (idx) / nkji;
        int k = (idx - m * nkji) / nji;
        int j = (idx - m * nkji - k * nji) / nx1;
        int i = (idx - m * nkji - k * nji - j * nx1) + is;
        k += ks;
        j += js;

        mb_max     = fmax(mb_max, w0_(m, IDN, k, j, i));
        mb_alp_min = fmin(mb_alp_min, adm.alpha(m, k, j, i));
      },
      Kokkos::Max<Real>(rho_max), Kokkos::Min<Real>(alpha_min));

#if MPI_PARALLEL_ENABLED
  if (global_variable::my_rank == 0) {
    MPI_Reduce(MPI_IN_PLACE, &rho_max, 1, MPI_ATHENA_REAL, MPI_MAX, 0,
               MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE, &alpha_min, 1, MPI_ATHENA_REAL, MPI_MIN, 0,
               MPI_COMM_WORLD);
  } else {
    MPI_Reduce(&rho_max, &rho_max, 1, MPI_ATHENA_REAL, MPI_MAX, 0,
               MPI_COMM_WORLD);
    MPI_Reduce(&alpha_min, &alpha_min, 1, MPI_ATHENA_REAL, MPI_MIN, 0,
               MPI_COMM_WORLD);
    rho_max   = 0.;
    alpha_min = 0.;
  }
#endif

  pdata->hdata[0] = rho_max;
  pdata->hdata[1] = alpha_min;
}

//----------------------------------------------------------------------------------------
//! \fn KadathBNSRefinementCondition()
//! \brief AMR refinement condition (delegates to Z4c AMR)
void KadathBHNSRefinementCondition(MeshBlockPack *pmbp) {
  pmbp->pz4c->pamr->Refine(pmbp);
}

KOKKOS_INLINE_FUNCTION
static Real A1(Real x, Real y, Real z, Real I_0, Real r_0) {
  Real w2 = SQR(x) + SQR(y);
  Real r2 = w2 + SQR(z);
  return -y * M_PI * SQR(r_0)*I_0 / pow(SQR(r_0) + r2, 1.5) *
         (1.0 + 15.0/8.0*SQR(r_0)*(SQR(r_0)+w2)/SQR(SQR(r_0)+r2));
}

KOKKOS_INLINE_FUNCTION
static Real A2(Real x, Real y, Real z, Real I_0, Real r_0) {
  Real w2 = SQR(x) + SQR(y);
  Real r2 = w2 + SQR(z);
  return x * M_PI * SQR(r_0)*I_0 / pow(SQR(r_0) + r2, 1.5) *
         (1.0 + 15.0/8.0*SQR(r_0)*(SQR(r_0)+w2)/SQR(SQR(r_0)+r2));
}