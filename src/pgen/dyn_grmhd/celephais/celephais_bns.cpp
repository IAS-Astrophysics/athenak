//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file celephais_bns.cpp
//  \brief Initial data reader for binary neutron star data from the Celephais
//         GR apps `BNS` (bi-spheric symmetric) and `BNS_nosym` (no equatorial /
//         orbital-plane symmetry).
//
//  Reads the TOML configuration (`kadath_config<BIN_INFO>`) and the matching
//  spectral space/field dump (`BeFileSource`, `.dat`) produced by those apps,
//  then inlines the spectral interpolation from
//  src/Utilities/Exporters/export_bns.cpp / export_bns_nosym.cpp directly into
//  the AthenaK fill loop — eliminating the large intermediate output arrays and
//  performing a single pass over all grid points.
//
//  This supersedes the previous FUKa (`kadath_config_boost`, `.info`) reader:
//  the config is now the repo's TOML `kadath_config`, the file is opened via
//  `BeFileSource` (not `fopen`), and the velocity/spin model matches the repo
//  exporters (tilted spins `angs1`/`angs2`, COM offsets `xaxis`/`yaxis`,
//  eccentricity-reduction radial velocity `adot`, and the corotating branch).
//
//  Select the symmetry of the source data with:
//    <problem>
//      initial_data_file = path/to/bns.toml   # Celephais BNS config (.toml)
//      kadath_symmetry   = sym                # "sym" -> Space_bin_ns (BNS app)
//                                             # "nosym" -> Space_bin_ns_nosym
//                                             #            (BNS_nosym app)
//  The Kadath space/field file must reside next to the config file (same stem,
//  resolved through `kadath_config::space_filename()`).
//
//  NOTE ON PARALLELISM: Kadath's MemoryMapper and coef_1d scratch pools are
//  thread_local (patched in memory.hpp/.cpp and coef_1d.cpp), so val_point() and
//  Point() are safe to call concurrently.  The per-point loop runs via
//  Kokkos::parallel_for on DefaultHostExecutionSpace (OpenMP threads).  A serial
//  warmup call before the loop initialises the summation_1d static dispatch table
//  on the main thread, preventing a first-call race among OMP threads.
//
//  NOTE ON INCLUDE PATHS: the Kadath headers below use the Celephais canonical
//  `For_Kadath/...` / `Hydro/...` layout (matching the in-repo exporters and the
//  `2sacra` tools).  The AthenaK build must add the Celephais `include/`
//  directory to its include path (`-I<celephais>/include`).

#include <cmath>
#include <cstdio>

#include <array>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
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
#include "z4c/z4c.hpp"
#include "z4c/z4c_amr.hpp"
#include "utils/tov/tov_utils.hpp"
#include "utils/tov/tov_polytrope.hpp"
#include "utils/tov/tov_piecewise_poly.hpp"
#include "utils/tov/tov_tabulated.hpp"

// Celephais (BNS / BNS_nosym GR initial data)
#include "For_Kadath/Kadath_point_h/kadath_bin_ns.hpp"
#include "For_Kadath/Kadath_point_h/kadath_bin_ns_nosym.hpp"
#include "Hydro/EOS.hh"
#include "For_Kadath/Utilities/Exporters/coord_fields.hpp"
#include "For_Kadath/Config/config_binary.hpp"
#include "For_Kadath/Utilities/exporter_utilities.hpp"
#include "For_Kadath/Utilities/Exporters/bco_geometry.hpp"
#include "For_Kadath/IO/be_file_source.hpp"

void KadathBNSHistory(HistoryData *pdata, Mesh *pm);
void KadathBNSRefinementCondition(MeshBlockPack *pmbp);

//----------------------------------------------------------------------------------------
//! \fn SetupBNS()
//! \brief Fill the AthenaK grid from a Celephais BNS / BNS_nosym solution.
//!
//! Templated on the Kadath space type (\c Space_bin_ns for the \c BNS app,
//! \c Space_bin_ns_nosym for the \c BNS_nosym app) and on the AthenaK 1D EOS.
//! The system-of-equations setup mirrors src/Utilities/Exporters/export_bns.cpp
//! so the interpolated fields are identical to the in-repo exporter.
template<class KadathSpace, class TOVEOS>
void SetupBNS(ParameterInput *pin, Mesh* pmy_mesh_) {
  using namespace Kadath;

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
  // coord_vector / to_int / vec_ary_t / CoordFields live in the global namespace
  // (declared at file scope in coord_fields.hpp); resolved by unqualified lookup.

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
  // Kadath BNS setup (inlined from export_bns.cpp / export_bns_nosym.cpp)
  // =========================================================================

  if (global_variable::my_rank == 0) {
    std::cout << "Reading Celephais BNS config from " << fname << " ..." << std::endl;
  }

  kadath_config<BIN_INFO> bconfig(fname);

  const double h_cut      = bconfig.eos<double>(HCUT, BCO1);
  const std::string eos_file = bconfig.eos<std::string>(EOSFILE, BCO1);
  const std::string eos_type = bconfig.eos<std::string>(EOSTYPE, BCO1);

  const double units    = 4.0 * M_PI;
  const double omega    = bconfig(GOMEGA);
  const bool is_corotating = bconfig.control(COROT_BIN);
  double &ome1     = bconfig(OMEGA, BCO1);
  double &ang1     = bconfig(DEG,   BCO1);
  double &ome2     = bconfig(OMEGA, BCO2);
  double &ang2     = bconfig(DEG,   BCO2);
  const double axis     = bconfig(COM);
  double yaxis = 0.;
  if (!std::isnan(bconfig.set(COMY)))
    yaxis = bconfig(COMY);

  std::string kadath_filename = bconfig.space_filename();

  BeFileSource fin(kadath_filename);
  KadathSpace space(fin);
  Scalar conf (space, fin);
  Scalar lapse(space, fin);
  Vector shift(space, fin);
  Scalar logh (space, fin);
  Scalar phi  (space, fin);

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
  int ndom = space.get_nbr_domains();

  double xc1 = bco_utils::get_center(space, space.NS1);
  double xc2 = bco_utils::get_center(space, space.NS2);
  double xo  = bco_utils::get_center(space, ndom - 1);

  Metric_flat fmet(space, basis);

  CoordFields<KadathSpace> cfields(space);
  vec_ary_t coord_vectors{default_binary_vector_ary(space)};
  update_fields(cfields, coord_vectors, {}, xo, xc1, xc2);
  Vector cart(space, CON, basis);
  cart = cfields.cart();

  System_of_eqs syst(space, 0, ndom - 1);
  fmet.set_system(syst, "f");

  Param p;
  if (eos_type == "Cold_Table") {
    using namespace Kadath::Margherita;
    using eos_t = Kadath::Margherita::Cold_Table;
    const int interp_pts = (bconfig.eos<int>(INTERP_PTS, BCO1) == 0)
                           ? 2000 : bconfig.eos<int>(INTERP_PTS, BCO1);
    EOS<eos_t, eos_var_t::PRESSURE>::init(eos_file, h_cut, interp_pts);
    syst.add_ope("eps",   &EOS<eos_t, eos_var_t::EPSILON>::action,  &p);
    syst.add_ope("press", &EOS<eos_t, eos_var_t::PRESSURE>::action, &p);
    syst.add_ope("rho",   &EOS<eos_t, eos_var_t::DENSITY>::action,  &p);
  } else if (eos_type == "Cold_PWPoly") {
    using namespace Kadath::Margherita;
    using eos_t = Kadath::Margherita::Cold_PWPoly;
    EOS<eos_t, eos_var_t::PRESSURE>::init(eos_file, h_cut);
    syst.add_ope("eps",   &EOS<eos_t, eos_var_t::EPSILON>::action,  &p);
    syst.add_ope("press", &EOS<eos_t, eos_var_t::PRESSURE>::action, &p);
    syst.add_ope("rho",   &EOS<eos_t, eos_var_t::DENSITY>::action,  &p);
  } else {
    throw std::invalid_argument("Unsupported EOS type for BNS export: " + eos_type);
  }

  syst.add_cst("4piG",  units);
  syst.add_cst("PI",    M_PI);

  syst.add_cst("omes1", ome1);
  syst.add_cst("omes2", ome2);
  syst.add_cst("angs1", ang1);
  syst.add_cst("angs2", ang2);

  syst.add_cst("mg",  *coord_vectors[to_int(coord_vector::GLOBAL_ROT)]);
  syst.add_cst("mmx", *coord_vectors[to_int(coord_vector::BCO1_ROTx)]);
  syst.add_cst("mmz", *coord_vectors[to_int(coord_vector::BCO1_ROTz)]);
  syst.add_cst("mpx", *coord_vectors[to_int(coord_vector::BCO2_ROTx)]);
  syst.add_cst("mpz", *coord_vectors[to_int(coord_vector::BCO2_ROTz)]);

  syst.add_cst("ex",   *coord_vectors[to_int(coord_vector::EX)]);
  syst.add_cst("ey",   *coord_vectors[to_int(coord_vector::EY)]);
  syst.add_cst("ez",   *coord_vectors[to_int(coord_vector::EZ)]);

  syst.add_cst("sm",   *coord_vectors[to_int(coord_vector::S_BCO1)]);
  syst.add_cst("sp",   *coord_vectors[to_int(coord_vector::S_BCO2)]);
  syst.add_cst("einf", *coord_vectors[to_int(coord_vector::S_INF)]);

  syst.add_cst("xaxis", axis);
  syst.add_cst("yaxis", yaxis);
  syst.add_cst("ome",   omega);

  syst.add_cst("P",   conf);
  syst.add_cst("N",   lapse);
  syst.add_cst("bet", shift);
  syst.add_cst("phi", phi);
  syst.add_cst("H",   logh);

  syst.add_def("NP = P*N");
  syst.add_def("Ntilde = N / P^6");

  syst.add_def("Morb^i = mg^i + xaxis * ey^i + yaxis * ex^i");
  std::string orbital_shift{"omega^i = bet^i + ome * Morb^i"};
  if (!std::isnan(bconfig.set(ADOT))) {
    syst.add_cst("adot", bconfig(ADOT));
    syst.add_cst("r", cart);
    syst.add_def("comr^i = r^i - xaxis * ex^i + yaxis * ey^i");
    orbital_shift += " + adot * comr^i";
  }
  syst.add_def(orbital_shift.c_str());

  for (int d = space.NS1; d <= space.ADAPTED1; ++d)
    syst.add_def(d, "s^i  = omes1 * ( cos(angs1) * mmz^i + sin(angs1) * mmx^i ) ");
  for (int d = space.NS2; d <= space.ADAPTED2; ++d)
    syst.add_def(d, "s^i  = omes2 * ( cos(angs2) * mpz^i + sin(angs2) * mpx^i ) ");

  syst.add_def("A_ij = (D_i bet_j + D_j bet_i - 2. / 3.* D^k bet_k * f_ij) /2. / N");
  syst.add_def("h = exp(H)");

  if (is_corotating) {
    syst.add_def("U^i = omega^i / N");
  } else {
    for (int d = 0; d < ndom; ++d) {
      if ((d <= space.ADAPTED1) || (d >= space.NS2 && d <= space.ADAPTED2))
        syst.add_def(d, "eta_i = D_i phi + P^4 * s_i");
      else
        syst.add_def(d, "eta_i = D_i phi");
    }

    syst.add_def("Wsquare = eta^i * eta_i / h^2 / P^4 + 1.");
    syst.add_def("W = sqrt(Wsquare)");
    syst.add_def("U^i = eta^i / P^4 / h / W");
  }

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
    pt_warm.set(1) = xc1;
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
    Point pt(3);
    pt.set(1) = static_cast<double>(x) - axis;
    pt.set(2) = static_cast<double>(y) - yaxis;
    pt.set(3) = static_cast<double>(z);

    // Evaluate all spectral quantities at this point.
    double qv[NUM_QUANTS];
    for (int kq = 0; kq < NUM_QUANTS; ++kq) {
      qv[kq] = quants[kq].get().val_point(pt);
    }

    // Conformal factor and derived powers.
    const double psi  = qv[PSI];
    const double psi4 = psi * psi * psi * psi;

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
    const double h_enth = Kokkos::exp(qv[H]);
    if (h_enth <= 1.) {
      // Vacuum: set to atmosphere values.
      host_w0(m, IDN, k, j, i) = 0.0;
      host_w0(m, IPR, k, j, i) = 0.0;
    } else {
      if (use_cold_table) {
        using namespace Kadath::Margherita;
        using eos_t = Kadath::Margherita::Cold_Table;
        host_w0(m, IDN, k, j, i) = EOS<eos_t, eos_var_t::DENSITY>::get(h_enth);
        host_w0(m, IPR, k, j, i) = EOS<eos_t, eos_var_t::PRESSURE>::get(h_enth);
      } else if (use_cold_pwpoly) {
        using namespace Kadath::Margherita;
        using eos_t = Kadath::Margherita::Cold_PWPoly;
        host_w0(m, IDN, k, j, i) = EOS<eos_t, eos_var_t::DENSITY>::get(h_enth);
        host_w0(m, IPR, k, j, i) = EOS<eos_t, eos_var_t::PRESSURE>::get(h_enth);
      }
    }

    if constexpr (use_ye) {
      if (read_ye) {
        Real& rho = host_w0(m, IDN, k, j, i);
        host_w0(m, IYF, k, j, i) = eos.template
                                   GetYeFromRho<tov::LocationTag::Host>(rho);
      }
    }

    // Velocity: qv[UX..UZ] is the Eulerian three-velocity U^i; vacuum -> 0.
    Real vu[3];
    if (h_enth <= 1.) {
      vu[0] = 0.0;
      vu[1] = 0.0;
      vu[2] = 0.0;
    } else {
      vu[0] = static_cast<Real>(qv[UX]);
      vu[1] = static_cast<Real>(qv[UY]);
      vu[2] = static_cast<Real>(qv[UZ]);
    }

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

  // TODO(user): Add magnetic field initialization (e.g., current-loop model).
  // For now, initialize face-centered and cell-centered B fields to zero.
  auto &b0 = pmbp->pmhd->b0;
  par_for(
      "pgen_Bfc", DevExeSpace(), 0, nmb - 1, ks, ke, js, je, is, ie,
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
        b0.x1f(m, k, j, i) = 0.0;
        b0.x2f(m, k, j, i) = 0.0;
        b0.x3f(m, k, j, i) = 0.0;

        if (i == ie) b0.x1f(m, k, j, i + 1) = 0.0;
        if (j == je) b0.x2f(m, k, j + 1, i) = 0.0;
        if (k == ke) b0.x3f(m, k + 1, j, i) = 0.0;
      });

  auto &bcc0 = pmbp->pmhd->bcc0;
  par_for(
      "pgen_bcc", DevExeSpace(), 0, nmb - 1, ks, ke, js, je, is, ie,
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
        bcc0(m, IBX, k, j, i) =
            0.5 * (b0.x1f(m, k, j, i) + b0.x1f(m, k, j, i + 1));
        bcc0(m, IBY, k, j, i) =
            0.5 * (b0.x2f(m, k, j, i) + b0.x2f(m, k, j + 1, i));
        bcc0(m, IBZ, k, j, i) =
            0.5 * (b0.x3f(m, k, j, i) + b0.x3f(m, k + 1, j, i));
      });

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
//! \fn DispatchBNSEOS()
//! \brief Select the AthenaK 1D EOS template for the requested EOS policy and
//!        forward to SetupBNS for the given Kadath space symmetry.
template<class KadathSpace>
void DispatchBNSEOS(ParameterInput *pin, Mesh *pmy_mesh_, MeshBlockPack *pmbp) {
  if (pmbp->pdyngr->eos_policy == DynGRMHD_EOS::eos_ideal) {
    SetupBNS<KadathSpace, tov::PolytropeEOS>(pin, pmy_mesh_);
  } else if (pmbp->pdyngr->eos_policy == DynGRMHD_EOS::eos_compose) {
    SetupBNS<KadathSpace, tov::TabulatedEOS>(pin, pmy_mesh_);
  } else if (pmbp->pdyngr->eos_policy == DynGRMHD_EOS::eos_hybrid) {
    SetupBNS<KadathSpace, tov::TabulatedEOS>(pin, pmy_mesh_);
  } else if (pmbp->pdyngr->eos_policy == DynGRMHD_EOS::eos_piecewise_poly) {
    SetupBNS<KadathSpace, tov::PiecewisePolytropeEOS>(pin, pmy_mesh_);
  } else {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Unknown EOS requested for Kadath BNS problem" << std::endl;
    exit(EXIT_FAILURE);
  }
}

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem()
//! \brief Problem generator for BNS with Celephais (BNS / BNS_nosym GR apps).
void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (!pmbp->pcoord->is_dynamical_relativistic) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Kadath BNS problem must have <adm> block to run"
              << std::endl;
    exit(EXIT_FAILURE);
  }
  user_hist_func = &KadathBNSHistory;
  user_ref_func = &KadathBNSRefinementCondition;

  if (restart) return;

  // Select the Kadath space symmetry of the source data:
  //   "sym"   -> Space_bin_ns        (apps/BNS)
  //   "nosym" -> Space_bin_ns_nosym  (apps/BNS_nosym)
  std::string symmetry = pin->GetOrAddString("problem", "kadath_symmetry", "sym");
  if (symmetry == "sym") {
    DispatchBNSEOS<Kadath::Space_bin_ns>(pin, pmy_mesh_, pmbp);
  } else if (symmetry == "nosym") {
    DispatchBNSEOS<Kadath::Space_bin_ns_nosym>(pin, pmy_mesh_, pmbp);
  } else {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Unknown <problem>/kadath_symmetry '" << symmetry
              << "' (expected 'sym' or 'nosym')" << std::endl;
    exit(EXIT_FAILURE);
  }

  return;
}


//----------------------------------------------------------------------------------------
//! \fn KadathBNSHistory()
//! \brief History function: tracks maximum rest-mass density and minimum lapse
void KadathBNSHistory(HistoryData *pdata, Mesh *pm) {
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
      "KadathBNSHistSums",
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
void KadathBNSRefinementCondition(MeshBlockPack *pmbp) {
  pmbp->pz4c->pamr->Refine(pmbp);
}
