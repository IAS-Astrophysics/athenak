//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file chemistry.cpp
//! \brief implementation of Chemistry class constructor and assorted other
//! functions
#include "chemistry/chemistry.hpp"

#include <algorithm>
#include <iostream>
#include <limits>
#include <map>
#include <string>
#include <tuple>
#include <vector>

#include "athena.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "ode_solvers/ode_solvers.hpp"

namespace chemistry {
//----------------------------------------------------------------------------------------
// Constructor, initializes data structures and parameters
//----------------------------------------------------------------------------------------
Chemistry::Chemistry(MeshBlockPack* ppack, ParameterInput* pin)
    : pmy_pack(ppack),
      is_hydro_enabled(pin->DoesBlockExist("hydro")),
      is_mhd_enabled(pin->DoesBlockExist("mhd")),
      nscalars_chemistry(SetupGetNumChemistryScalars(ppack, pin, -1, false)),
      mu_H(pin->GetOrAddReal("chemistry", "mu_H", 1.4)),
      chemistry_scalars_first_idx(ComputeChemistryScalarsStartIndex()),
      my_pin(pin) {
  // Verify that units are enables
  if (!pin->DoesBlockExist("units")) {
    std::cerr
        << "### FATAL ERROR: The chemistry module requires that the units "
           "module be enabled. Please enable it in the athinput file."
        << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

//----------------------------------------------------------------------------------------
// Destructor, primarily frees memory
//----------------------------------------------------------------------------------------
Chemistry::~Chemistry() {}

// ================
// Member Functions
// ================
/*!
 * \brief Selects the proper template of Chemistry::UpdateChemistry to call and
 * passes in the proper arguments
 */
TaskStatus Chemistry::UpdateChemistryTask(Driver* d, int stage) {
  const std::string network = my_pin->GetString("chemistry", "network");
  const std::string ode_solver = my_pin->GetString("chemistry", "ode_solver");

  if (network == "H2") {
    auto h2_settings = H2Network::GetSettings(my_pin);
    if (ode_solver == "forward_euler") {
      auto fe_settings = ode_solvers::ForwardEuler<H2Network>::GetSettings(
          my_pin, "chemistry");
      UpdateChemistry<ode_solvers::ForwardEuler<H2Network>, H2Network>(
          fe_settings, h2_settings);
    }
  }

  return TaskStatus::complete;
}

/*!
 * \brief Updates the chemistry scalars and internal energy
 *
 * \tparam ODE_Solver_t The ODE solver to ues
 * \tparam Network_t The chemistry network to use
 * \tparam ODESettings The type of the settings struct to pass to the ODE solver
 * \tparam NetworkSettings The type of the settings struct to pass to the ODE
 * solver
 * \param ode_settings The settings struct to pass to the ODE solver
 * \param network_settings The settings struct to pass to the ODE solver
 */
template <typename ODE_Solver_t, typename Network_t, typename ODESettings,
          typename NetworkSettings>
void Chemistry::UpdateChemistry(ODESettings const& ode_settings,
                                NetworkSettings const& network_settings) {
  // ------ Collect variables that we'll need -----
  // The primitive grid
  auto w0 = GetW0();
  // The time at the beginning of this timestep
  Real const t_start = pmy_pack->pmesh->time;
  // The timestep
  Real const dt = pmy_pack->pmesh->dt;

  // ----- Variables for the ODE solver -----
  // For reporting if the ODE solver doesn't converge
  DvceArray0D<bool> chemisty_ode_failure("chemisty_ode_failure", false);

  // ----- Get the unit conversions and constants we'll need -----
  Real const time_cgs = pmy_pack->punit->time_cgs();
  Real const energy_density_cgs = pmy_pack->punit->pressure_cgs();
  Real const density_cgs = pmy_pack->punit->density_cgs();
  Real const hydrogen_mass_cgs = pmy_pack->punit->hydrogen_mass_cgs;
  Real const gamma = pmy_pack->phydro->peos->eos_data.gamma;
  Real const mu_H_local = mu_H;

  // ----- Get all the loop limits and generate the parallel policy ------
  // NOLINTNEXTLINE(whitespace/braces)
  auto const [start_limit, end_limit] = LoopLimitsAllCells();
  int const species_start_idx = chemistry_scalars_first_idx;
  auto const policy = Kokkos::MDRangePolicy<Kokkos::Rank<4>>(
      DevExeSpace(), start_limit, end_limit);

  Kokkos::parallel_for(
      "Chemistry_ODE_Solve", policy,
      KOKKOS_LAMBDA(const int& mb_idx, const int& k, const int& j,
                    const int& i) {
        // Create the chemisty object
        Network_t chem_net(network_settings, w0(mb_idx, IDN, k, j, i),
                           density_cgs, mu_H_local, gamma, hydrogen_mass_cgs,
                           time_cgs, energy_density_cgs);

        // ------ Load cell values ------
        // Chemistry scalars. The loop is based off of the chemical
        // network's number of equations since that's known at compile time,
        // enabling more loop optimizations. The minus 1 is because internal
        // energy occupies the last slot in the array
        int grid_idx = species_start_idx;
        for (int s_idx = 0; s_idx < Network_t::neqs - 1; s_idx++) {
          chem_net.y(s_idx) = w0(mb_idx, grid_idx, k, j, i);
          grid_idx += 1;
        }

        // Load internal energy
        chem_net.y(Network_t::IIE) = w0(mb_idx, IEN, k, j, i);

        // ------ Solve the ODEs ------
        ODE_Solver_t ode_solver(ode_settings, chem_net, t_start, dt);
        ode_solver.SolveODE();

        // check if the ODE solver failed
        if (ode_solver.failed) {
          chemisty_ode_failure() = ode_solver.failed;
        }

        // ------ Write cell values back out ------
        // Chemistry scalars
        grid_idx = species_start_idx;
        for (int s_idx = 0; s_idx < Network_t::neqs - 1; s_idx++) {
          w0(mb_idx, grid_idx, k, j, i) = chem_net.y(s_idx);
          grid_idx += 1;
        }

        // Write internal energy
        w0(mb_idx, IEN, k, j, i) = chem_net.y(Network_t::IIE);
      });

  // Get the failure flag and check for failure
  bool chemisty_ode_failure_h;
  Kokkos::deep_copy(chemisty_ode_failure_h, chemisty_ode_failure);
  if (chemisty_ode_failure_h) {
    std::cerr << "The chemistry ODE solver failed to converge." << std::endl;
  }
}

/*!
 * \brief Syncs the conserved array to the values in the primitive array.
 * Primarily intended to update the energy since the chemistry solve updates the
 * internal energy.
 */
TaskStatus Chemistry::PrimToCons(Driver* pdrive, int stage) {
  auto& indcs = pmy_pack->pmesh->mb_indcs;
  int& ng = indcs.ng;
  int n1m1 = indcs.nx1 + 2 * ng - 1;
  int n2m1 = (indcs.nx2 > 1) ? (indcs.nx2 + 2 * ng - 1) : 0;
  int n3m1 = (indcs.nx3 > 1) ? (indcs.nx3 + 2 * ng - 1) : 0;

  if (is_hydro_enabled) {
    auto peos = pmy_pack->phydro->peos;
    auto u0 = pmy_pack->phydro->u0;
    auto w0 = pmy_pack->phydro->w0;
    peos->PrimToCons(w0, u0, 0, n1m1, 0, n2m1, 0, n3m1);
  } else {  // if (is_mhd_enabled) {
    auto peos = pmy_pack->pmhd->peos;
    auto u0 = pmy_pack->pmhd->u0;
    auto bcc = pmy_pack->pmhd->bcc0;
    auto w0 = pmy_pack->pmhd->w0;
    peos->PrimToCons(w0, bcc, u0, 0, n1m1, 0, n2m1, 0, n3m1);
  }

  return TaskStatus::complete;
}

/*!
 * \brief Return the name of the chemical species at scalar_idx
 *
 * \param scalar_idx The index of the chemistry scalar who's name is needed
 * \return std::string The name of the chemistry species
 */
std::string Chemistry::GetSpeciesNames(int const& scalar_idx) {
  // Only the first time this is called create the mapping between species names
  // and grid index
  static std::map<int, std::string> species_names_map;
  if (species_names_map.size() == 0) {
    // std::vector of scalar names
    const std::string network = my_pin->GetString("chemistry", "network");
    std::vector<std::string_view> species_names;
    if (network == "H2") {
      species_names.assign(H2Network::species_names.begin(),
                           H2Network::species_names.end());
    }

    // Create the mapping
    int name_idx = 0;
    for (size_t i = get_chemistry_scalars_first_idx();
         i < get_chemistry_scalars_last_idx() + 1; i++) {
      species_names_map[i] = "chem_" + std::string(species_names[name_idx]);
      name_idx++;
    }
  }

  // Verify that this is a chemistry scalar
  if (scalar_idx < get_chemistry_scalars_first_idx() ||
      scalar_idx > get_chemistry_scalars_last_idx()) {
    std::stringstream msg;
    msg << "Attempted to output the field at index " << scalar_idx
        << " as a passive scalar for the chemistry module but it is not one of "
           "the scalars managed by the chemistry module.";
    throw std::runtime_error(msg.str());
  }

  // Return the proper name
  return species_names_map[scalar_idx];
}

/*!
 * \brief Get the conserved array. Correctly sources the array from the hydro or
 * MHD classes.
 *
 * \return DvceArray5D<Real> The conserved array
 */
DvceArray5D<Real> Chemistry::GetU0() {
  if (is_hydro_enabled) {
    return pmy_pack->phydro->u0;
  } else {  // if (is_mhd_enabled) {
    return pmy_pack->pmhd->u0;
  }
}

/*!
 * \brief Get the primitive array. Correctly sources the array from the hydro or
 * MHD classes.
 *
 * \return DvceArray5D<Real> The primitive array
 */
DvceArray5D<Real> Chemistry::GetW0() {
  if (is_hydro_enabled) {
    return pmy_pack->phydro->w0;
  } else {  // if (is_mhd_enabled) {
    return pmy_pack->pmhd->w0;
  }
}

/*!
 * \brief Gets the first index for a chemistry scalar
 *
 * \return int
 */
int Chemistry::ComputeChemistryScalarsStartIndex() {
  if (is_hydro_enabled) {
    return pmy_pack->phydro->nhydro + nscalars_pre_chemistry;
  } else if (is_mhd_enabled) {
    return pmy_pack->pmhd->nmhd + nscalars_pre_chemistry;
  } else {
    throw std::runtime_error(
        "The chemistry module requires that either the hydro or MHD "
        "integrators be used and neither was requested in the input file.");
  }
}

/*!
 * \brief Returns loop limits for the chemistry solver to use with
 * MDRangePolicy. These limits loop over the entire grid, including ghost cells,
 * to avoid having to re-comminicate the updated values of internal energy.
 *
 * \return std::tuple<Kokkos::Array<int, 4>, Kokkos::Array<int, 4>> The start
 * and end limits in that order
 */
std::tuple<Kokkos::Array<int, 4>, Kokkos::Array<int, 4>>
Chemistry::LoopLimitsAllCells() {
  // This creates loop bounds over the entire grid, including the ghost cells.
  // This requires extra computation but doesn't require that the ghost cells be
  // re-communicated or that chemistry be inserted during the last stage of the
  // integrator before boundary conditions are communicated
  Kokkos::Array<int, 4> const start = {
      0,  // meshblock start
      0,  // k start
      0,  // j start
      0   // i start
  };

  // Check if the dimension is active and if it's not set the upper limit to 1
  const int ke =
      (pmy_pack->pmesh->mb_indcs.ke == 0)
          ? 1
          : pmy_pack->pmesh->mb_indcs.ke + 1 + pmy_pack->pmesh->mb_indcs.ng;
  const int je =
      (pmy_pack->pmesh->mb_indcs.je == 0)
          ? 1
          : pmy_pack->pmesh->mb_indcs.je + 1 + pmy_pack->pmesh->mb_indcs.ng;
  const int ie =
      (pmy_pack->pmesh->mb_indcs.ie == 0)
          ? 1
          : pmy_pack->pmesh->mb_indcs.ie + 1 + pmy_pack->pmesh->mb_indcs.ng;

  Kokkos::Array<int, 4> const end = {
      pmy_pack->nmb_thispack,  // meshblock end
      ke,                      // k end
      je,                      // j end
      ie                       // i end
  };
  return {start, end};
}

}  // namespace chemistry
