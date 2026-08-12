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
//  NOTE ON PARALLELISM: all mutable spectral coefficients are prepared before
//  the fill, and spectral summation uses worker-local scratch.  The per-point
//  loop runs via Kokkos::parallel_for on DefaultHostExecutionSpace.  A serial
//  warmup call before the loop initialises the summation_1d static dispatch table
//  on the main thread, preventing a first-call race among host threads.
//
//  NOTE ON INCLUDE PATHS: the Kadath headers below use the Celephais canonical
//  `For_Kadath/...` / `Hydro/...` layout (matching the in-repo exporters and the
//  `2sacra` tools).  The AthenaK build must add the Celephais `include/`
//  directory to its include path (`-I<celephais>/include`).

#include <cmath>
#include <cstdio>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cstdint>
#include <functional>
#include <iostream>
#include <limits>
#include <span>
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
#include "Apps/Formalism/Shared/scalar_point_batch.hpp"

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
  const bool batch_fields = pin->GetOrAddBoolean("problem", "batch_fields", true);
  const bool skip_vacuum_velocity =
      pin->GetOrAddBoolean("problem", "skip_vacuum_velocity", true);

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
  const double ang1_rad = bconfig(DEG, BCO1) * M_PI / 180.;
  double &ome2     = bconfig(OMEGA, BCO2);
  const double ang2_rad = bconfig(DEG, BCO2) * M_PI / 180.;
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
  syst.add_cst("angs1", ang1_rad);
  syst.add_cst("angs2", ang2_rad);

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

  std::array<const Scalar*, NUM_QUANTS> quant_fields{};
  for (int kq = 0; kq < NUM_QUANTS; ++kq)
    quant_fields[kq] = &quants[kq].get();
  bns_field_transfer::scalar_source_batch quant_batch(
      std::span<const Scalar* const>(quant_fields.data(), quant_fields.size()));
  // Prepare all mutable coefficient state before the host-parallel fill.
  quant_batch.prepare_coefficients();

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

  const auto interpolation_start = std::chrono::steady_clock::now();

  // Ghost zones of rank-local MeshBlocks can visit exactly the same Cartesian
  // point.  Identify those points by their shifted IEEE-754 coordinate bits,
  // retain the first visit as the canonical destination, and evaluate it once.
  // Keeping signed-zero bits distinct avoids changing odd-basis behaviour.
  struct CoordinateVisit {
    std::array<std::uint64_t, 3> coordinate_bits;
    int visit;
  };
  std::vector<CoordinateVisit> sorted_visits(static_cast<std::size_t>(width));
  for (int idx = 0; idx < width; ++idx) {
    const int m = idx / ncells_per_mb;
    int rem = idx - m * ncells_per_mb;
    const int k = rem / (ncells2 * ncells1);
    rem -= k * ncells2 * ncells1;
    const int j = rem / ncells1;
    const int i = rem % ncells1;
    const double x = static_cast<double>(CellCenterX(
        i - is, indcs.nx1, size.h_view(m).x1min, size.h_view(m).x1max)) - axis;
    const double y = static_cast<double>(CellCenterX(
        j - js, indcs.nx2, size.h_view(m).x2min, size.h_view(m).x2max)) - yaxis;
    const double z = static_cast<double>(CellCenterX(
        k - ks, indcs.nx3, size.h_view(m).x3min, size.h_view(m).x3max));
    sorted_visits[static_cast<std::size_t>(idx)] = {
        {std::bit_cast<std::uint64_t>(x), std::bit_cast<std::uint64_t>(y),
         std::bit_cast<std::uint64_t>(z)}, idx};
  }
  std::sort(sorted_visits.begin(), sorted_visits.end(),
            [](const CoordinateVisit &lhs, const CoordinateVisit &rhs) {
              return lhs.coordinate_bits < rhs.coordinate_bits;
            });

  std::vector<int> canonical_for_visit(static_cast<std::size_t>(width), -1);
  for (std::size_t begin = 0; begin < sorted_visits.size();) {
    std::size_t end = begin + 1;
    int canonical = sorted_visits[begin].visit;
    while (end < sorted_visits.size() &&
           sorted_visits[end].coordinate_bits == sorted_visits[begin].coordinate_bits) {
      canonical = std::min(canonical, sorted_visits[end].visit);
      ++end;
    }
    for (std::size_t visit = begin; visit < end; ++visit)
      canonical_for_visit[static_cast<std::size_t>(sorted_visits[visit].visit)] = canonical;
    begin = end;
  }

  std::vector<int> canonical_visits;
  std::vector<std::array<int, 2>> duplicate_visits;
  canonical_visits.reserve(static_cast<std::size_t>(width));
  duplicate_visits.reserve(static_cast<std::size_t>(width));
  for (int idx = 0; idx < width; ++idx) {
    const int canonical = canonical_for_visit[static_cast<std::size_t>(idx)];
    if (canonical == idx)
      canonical_visits.push_back(idx);
    else
      duplicate_visits.push_back({canonical, idx});
  }
  std::vector<CoordinateVisit>().swap(sorted_visits);
  std::vector<int>().swap(canonical_for_visit);
  const auto interpolation_plan_end = std::chrono::steady_clock::now();

  constexpr int point_lane_width = 4;
  const int canonical_count = static_cast<int>(canonical_visits.size());
  const int point_tiles = (canonical_count + point_lane_width - 1) / point_lane_width;
  Kokkos::parallel_for("celephais_fill",
      Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, point_tiles),
      [&](const int tile) {
    const int first_idx = tile * point_lane_width;
    const int lane_count = std::min(point_lane_width, canonical_count - first_idx);
    std::array<int, point_lane_width> m_values{};
    std::array<int, point_lane_width> k_values{};
    std::array<int, point_lane_width> j_values{};
    std::array<int, point_lane_width> i_values{};
    std::array<Point, point_lane_width> points{
        Point(3), Point(3), Point(3), Point(3)};
    for (int lane = 0; lane < point_lane_width; ++lane) {
      const int canonical_lane = first_idx + std::min(lane, lane_count - 1);
      const int idx = canonical_visits[static_cast<std::size_t>(canonical_lane)];
      int m = idx / ncells_per_mb;
      int rem = idx - m * ncells_per_mb;
      int k = rem / (ncells2 * ncells1);
      rem -= k * ncells2 * ncells1;
      int j = rem / ncells1;
      int i = rem % ncells1;
      m_values[lane] = m;
      k_values[lane] = k;
      j_values[lane] = j;
      i_values[lane] = i;

      const Real x = CellCenterX(i - is, indcs.nx1,
                                 size.h_view(m).x1min, size.h_view(m).x1max);
      const Real y = CellCenterX(j - js, indcs.nx2,
                                 size.h_view(m).x2min, size.h_view(m).x2max);
      const Real z = CellCenterX(k - ks, indcs.nx3,
                                 size.h_view(m).x3min, size.h_view(m).x3max);
      points[lane].set(1) = static_cast<double>(x) - axis;
      points[lane].set(2) = static_cast<double>(y) - yaxis;
      points[lane].set(3) = static_cast<double>(z);
    }

    // Evaluate one field at four adjacent Cartesian points. Same-domain tiles
    // share the coefficient stream and expose four independent recurrences;
    // domain-crossing tiles retain exact scalar evaluation.
    std::array<std::array<double, NUM_QUANTS>, point_lane_width> qv{};
    std::array<double, point_lane_width> h_enth{};
    std::array<bool, point_lane_width> is_vacuum{};
    if (batch_fields) {
      std::array<bns_field_transfer::located_source_point, point_lane_width> located{
          quant_batch.locate(points[0]), quant_batch.locate(points[1]),
          quant_batch.locate(points[2]), quant_batch.locate(points[3])};
      const std::array<const bns_field_transfer::located_source_point*, point_lane_width>
          located_ptrs{&located[0], &located[1], &located[2], &located[3]};
      std::array<double, point_lane_width> field_values{};
      if (skip_vacuum_velocity) {
        for (int kq = PSI; kq <= H; ++kq) {
          quant_batch.value_points4(located_ptrs, kq, field_values);
          for (int lane = 0; lane < point_lane_width; ++lane)
            qv[lane][kq] = field_values[lane];
        }
        for (int lane = 0; lane < lane_count; ++lane) {
          h_enth[lane] = Kokkos::exp(qv[lane][H]);
          is_vacuum[lane] = (h_enth[lane] <= 1.);
          if (!is_vacuum[lane]) {
            for (int kq = UX; kq <= UZ; ++kq)
              qv[lane][kq] = quant_batch.value(located[lane], kq);
          }
        }
      } else {
        for (int kq = 0; kq < NUM_QUANTS; ++kq) {
          quant_batch.value_points4(located_ptrs, kq, field_values);
          for (int lane = 0; lane < point_lane_width; ++lane)
            qv[lane][kq] = field_values[lane];
        }
      }
    } else {
      for (int lane = 0; lane < lane_count; ++lane)
        for (int kq = 0; kq < NUM_QUANTS; ++kq)
          qv[lane][kq] = quants[kq].get().val_point(points[lane]);
    }

    for (int lane = 0; lane < lane_count; ++lane) {
      const int m = m_values[lane];
      const int k = k_values[lane];
      const int j = j_values[lane];
      const int i = i_values[lane];

      // Conformal factor and derived powers.
      const double psi  = qv[lane][PSI];
      const double psi4 = psi * psi * psi * psi;

      // Lapse and shift.
      host_adm.alpha(m, k, j, i)      = qv[lane][ALP];
      host_adm.beta_u(m, 0, k, j, i) = qv[lane][BETX];
      host_adm.beta_u(m, 1, k, j, i) = qv[lane][BETY];
      host_adm.beta_u(m, 2, k, j, i) = qv[lane][BETZ];

      // Spatial metric: g_ij = psi^4 * delta_ij
      Real g3d[NSPMETRIC];
      host_adm.g_dd(m, 0, 0, k, j, i) = g3d[S11] = static_cast<Real>(psi4);
      host_adm.g_dd(m, 0, 1, k, j, i) = g3d[S12] = 0.0;
      host_adm.g_dd(m, 0, 2, k, j, i) = g3d[S13] = 0.0;
      host_adm.g_dd(m, 1, 1, k, j, i) = g3d[S22] = static_cast<Real>(psi4);
      host_adm.g_dd(m, 1, 2, k, j, i) = g3d[S23] = 0.0;
      host_adm.g_dd(m, 2, 2, k, j, i) = g3d[S33] = static_cast<Real>(psi4);

      // Extrinsic curvature: K_ij = psi^4 * A_ij
      host_adm.vK_dd(m, 0, 0, k, j, i) = qv[lane][AXX] * psi4;
      host_adm.vK_dd(m, 0, 1, k, j, i) = qv[lane][AXY] * psi4;
      host_adm.vK_dd(m, 0, 2, k, j, i) = qv[lane][AXZ] * psi4;
      host_adm.vK_dd(m, 1, 1, k, j, i) = qv[lane][AYY] * psi4;
      host_adm.vK_dd(m, 1, 2, k, j, i) = qv[lane][AYZ] * psi4;
      host_adm.vK_dd(m, 2, 2, k, j, i) = qv[lane][AZZ] * psi4;

      // Hydro: qv[H] = log(h), h = specific enthalpy.
      if (!batch_fields || !skip_vacuum_velocity) {
        h_enth[lane] = Kokkos::exp(qv[lane][H]);
        is_vacuum[lane] = (h_enth[lane] <= 1.);
      }
      if (is_vacuum[lane]) {
        host_w0(m, IDN, k, j, i) = 0.0;
        host_w0(m, IPR, k, j, i) = 0.0;
      } else {
        if (use_cold_table) {
          using namespace Kadath::Margherita;
          using eos_t = Kadath::Margherita::Cold_Table;
          host_w0(m, IDN, k, j, i) = EOS<eos_t, eos_var_t::DENSITY>::get(h_enth[lane]);
          host_w0(m, IPR, k, j, i) = EOS<eos_t, eos_var_t::PRESSURE>::get(h_enth[lane]);
        } else if (use_cold_pwpoly) {
          using namespace Kadath::Margherita;
          using eos_t = Kadath::Margherita::Cold_PWPoly;
          host_w0(m, IDN, k, j, i) = EOS<eos_t, eos_var_t::DENSITY>::get(h_enth[lane]);
          host_w0(m, IPR, k, j, i) = EOS<eos_t, eos_var_t::PRESSURE>::get(h_enth[lane]);
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
      if (is_vacuum[lane]) {
        vu[0] = 0.0;
        vu[1] = 0.0;
        vu[2] = 0.0;
      } else {
        vu[0] = static_cast<Real>(qv[lane][UX]);
        vu[1] = static_cast<Real>(qv[lane][UY]);
        vu[2] = static_cast<Real>(qv[lane][UZ]);
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
    }
  });
  Kokkos::fence();

  const auto interpolation_evaluation_end = std::chrono::steady_clock::now();
  const int duplicate_count = static_cast<int>(duplicate_visits.size());
  Kokkos::parallel_for("celephais_scatter_duplicates",
      Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, duplicate_count),
      [&](const int duplicate) {
    const std::array<int, 2> visits =
        duplicate_visits[static_cast<std::size_t>(duplicate)];
    std::array<int, 2> m_values{};
    std::array<int, 2> k_values{};
    std::array<int, 2> j_values{};
    std::array<int, 2> i_values{};
    for (int endpoint = 0; endpoint < 2; ++endpoint) {
      const int idx = visits[endpoint];
      const int m = idx / ncells_per_mb;
      int rem = idx - m * ncells_per_mb;
      const int k = rem / (ncells2 * ncells1);
      rem -= k * ncells2 * ncells1;
      m_values[endpoint] = m;
      k_values[endpoint] = k;
      j_values[endpoint] = rem / ncells1;
      i_values[endpoint] = rem % ncells1;
    }

    const int src_m = m_values[0];
    const int src_k = k_values[0];
    const int src_j = j_values[0];
    const int src_i = i_values[0];
    const int dst_m = m_values[1];
    const int dst_k = k_values[1];
    const int dst_j = j_values[1];
    const int dst_i = i_values[1];
    host_adm.alpha(dst_m, dst_k, dst_j, dst_i) =
        host_adm.alpha(src_m, src_k, src_j, src_i);
    for (int component = 0; component < 3; ++component)
      host_adm.beta_u(dst_m, component, dst_k, dst_j, dst_i) =
          host_adm.beta_u(src_m, component, src_k, src_j, src_i);
    for (int row = 0; row < 3; ++row) {
      for (int column = row; column < 3; ++column) {
        host_adm.g_dd(dst_m, row, column, dst_k, dst_j, dst_i) =
            host_adm.g_dd(src_m, row, column, src_k, src_j, src_i);
        host_adm.vK_dd(dst_m, row, column, dst_k, dst_j, dst_i) =
            host_adm.vK_dd(src_m, row, column, src_k, src_j, src_i);
      }
    }
    host_w0(dst_m, IDN, dst_k, dst_j, dst_i) =
        host_w0(src_m, IDN, src_k, src_j, src_i);
    host_w0(dst_m, IPR, dst_k, dst_j, dst_i) =
        host_w0(src_m, IPR, src_k, src_j, src_i);
    if constexpr (use_ye) {
      if (read_ye)
        host_w0(dst_m, IYF, dst_k, dst_j, dst_i) =
            host_w0(src_m, IYF, src_k, src_j, src_i);
    }
    for (int component = 0; component < 3; ++component)
      host_w0(dst_m, IVX + component, dst_k, dst_j, dst_i) =
          host_w0(src_m, IVX + component, src_k, src_j, src_i);
  });
  Kokkos::fence();

  const auto interpolation_end = std::chrono::steady_clock::now();
  const double interpolation_seconds = std::chrono::duration<double>(
      interpolation_end - interpolation_start).count();
  const double interpolation_plan_seconds = std::chrono::duration<double>(
      interpolation_plan_end - interpolation_start).count();
  const double interpolation_evaluation_seconds = std::chrono::duration<double>(
      interpolation_evaluation_end - interpolation_plan_end).count();
  const double interpolation_scatter_seconds = std::chrono::duration<double>(
      interpolation_end - interpolation_evaluation_end).count();
  double interpolation_min_seconds = interpolation_seconds;
  double interpolation_mean_seconds = interpolation_seconds;
  double interpolation_max_seconds = interpolation_seconds;
  double interpolation_plan_mean_seconds = interpolation_plan_seconds;
  double interpolation_plan_max_seconds = interpolation_plan_seconds;
  double interpolation_evaluation_mean_seconds = interpolation_evaluation_seconds;
  double interpolation_evaluation_max_seconds = interpolation_evaluation_seconds;
  double interpolation_scatter_mean_seconds = interpolation_scatter_seconds;
  double interpolation_scatter_max_seconds = interpolation_scatter_seconds;
  int interpolation_max_rank = 0;
  int interpolation_max_rank_blocks = nmb;
  int interpolation_max_rank_points = width;
  int interpolation_max_rank_canonical_points = canonical_count;
#if MPI_PARALLEL_ENABLED
  constexpr int fill_data_width = 7;
  const std::array<double, fill_data_width> local_fill_data = {
      interpolation_seconds, interpolation_plan_seconds,
      interpolation_evaluation_seconds, interpolation_scatter_seconds,
      static_cast<double>(nmb), static_cast<double>(width),
      static_cast<double>(canonical_count)};
  std::vector<double> all_fill_data;
  if (global_variable::my_rank == 0)
    all_fill_data.resize(fill_data_width * global_variable::nranks);
  MPI_Gather(local_fill_data.data(), fill_data_width, MPI_DOUBLE,
             all_fill_data.data(), fill_data_width, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  if (global_variable::my_rank == 0) {
    interpolation_min_seconds = std::numeric_limits<double>::max();
    interpolation_max_seconds = std::numeric_limits<double>::lowest();
    double interpolation_sum_seconds = 0.0;
    double interpolation_plan_sum_seconds = 0.0;
    double interpolation_evaluation_sum_seconds = 0.0;
    double interpolation_scatter_sum_seconds = 0.0;
    interpolation_plan_max_seconds = 0.0;
    interpolation_evaluation_max_seconds = 0.0;
    interpolation_scatter_max_seconds = 0.0;
    for (int rank = 0; rank < global_variable::nranks; ++rank) {
      const int offset = fill_data_width * rank;
      const double rank_seconds = all_fill_data[offset];
      interpolation_min_seconds = std::min(interpolation_min_seconds, rank_seconds);
      interpolation_sum_seconds += rank_seconds;
      interpolation_plan_sum_seconds += all_fill_data[offset + 1];
      interpolation_evaluation_sum_seconds += all_fill_data[offset + 2];
      interpolation_scatter_sum_seconds += all_fill_data[offset + 3];
      interpolation_plan_max_seconds =
          std::max(interpolation_plan_max_seconds, all_fill_data[offset + 1]);
      interpolation_evaluation_max_seconds =
          std::max(interpolation_evaluation_max_seconds, all_fill_data[offset + 2]);
      interpolation_scatter_max_seconds =
          std::max(interpolation_scatter_max_seconds, all_fill_data[offset + 3]);
      if (rank_seconds > interpolation_max_seconds) {
        interpolation_max_seconds = rank_seconds;
        interpolation_max_rank = rank;
        interpolation_max_rank_blocks = static_cast<int>(all_fill_data[offset + 4]);
        interpolation_max_rank_points = static_cast<int>(all_fill_data[offset + 5]);
        interpolation_max_rank_canonical_points =
            static_cast<int>(all_fill_data[offset + 6]);
      }
    }
    interpolation_mean_seconds =
        interpolation_sum_seconds / static_cast<double>(global_variable::nranks);
    interpolation_plan_mean_seconds =
        interpolation_plan_sum_seconds / static_cast<double>(global_variable::nranks);
    interpolation_evaluation_mean_seconds =
        interpolation_evaluation_sum_seconds / static_cast<double>(global_variable::nranks);
    interpolation_scatter_mean_seconds =
        interpolation_scatter_sum_seconds / static_cast<double>(global_variable::nranks);
  }
#endif

  if (global_variable::my_rank == 0) {
    std::printf("[celephais-timing] batch_fields=%s skip_vacuum_velocity=%s "
                "celephais_fill_min_seconds=%.9f celephais_fill_mean_seconds=%.9f "
                "celephais_fill_max_seconds=%.9f celephais_fill_max_rank=%d "
                "celephais_fill_max_rank_blocks=%d celephais_fill_max_rank_points=%d "
                "celephais_fill_max_rank_canonical_points=%d "
                "celephais_fill_plan_mean_seconds=%.9f "
                "celephais_fill_plan_max_seconds=%.9f "
                "celephais_fill_evaluation_mean_seconds=%.9f "
                "celephais_fill_evaluation_max_seconds=%.9f "
                "celephais_fill_scatter_mean_seconds=%.9f "
                "celephais_fill_scatter_max_seconds=%.9f\n",
                batch_fields ? "true" : "false",
                skip_vacuum_velocity ? "true" : "false",
                interpolation_min_seconds, interpolation_mean_seconds,
                interpolation_max_seconds, interpolation_max_rank,
                interpolation_max_rank_blocks, interpolation_max_rank_points,
                interpolation_max_rank_canonical_points,
                interpolation_plan_mean_seconds, interpolation_plan_max_seconds,
                interpolation_evaluation_mean_seconds,
                interpolation_evaluation_max_seconds,
                interpolation_scatter_mean_seconds,
                interpolation_scatter_max_seconds);
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
