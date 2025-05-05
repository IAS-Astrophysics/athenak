//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_source.cpp

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "driver/driver.hpp"
#include "coordinates/cartesian_ks.hpp"
#include "coordinates/coordinates.hpp"
#include "coordinates/cell_locations.hpp"
#include "geodesic-grid/geodesic_grid.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "units/units.hpp"
#include "radiation.hpp"

#include "radiation/radiation_tetrad.hpp"
#include "radiation/radiation_opacities.hpp"

namespace radiation {
//----------------------------------------------------------------------------------------
//! \fn  void Radiation::SetFrequencyGrid()
//! \brief Set frequency grid for radiation

void Radiation::SetFrequencyGrid() {
  bool &are_units_enabled_ = are_units_enabled;
  Real nu_unit = 1.0;
  if (are_units_enabled_) {
    Real h_p = 6.62607015e-27; // Planck constant
    Real k_b = 1.380649e-16;   // Boltzman constant
    Real temp_unit = pmy_pack->punit->temperature_cgs();
    nu_unit = k_b*temp_unit/h_p;
  }

  // Extract auxiliary quantities for frequency grid
  int  &nfreq_ = nfreq;
  int  &freq_scale = flag_fscale;
  Real &nu_min_ = nu_min;
  Real &nu_max_ = nu_max;
  auto &freq_grid_ = freq_grid;

  // convert frequency from cgs into sim unit
  Real freq_min = nu_min_/nu_unit; // from cgs to sim unit
  Real freq_max = nu_max_/nu_unit; // from cgs to sim unit

  // assign freq_min and freq_max
  int nfreq_grid = nfreq_-1; // frequency grid is defined with both 0 and inf excluded
  freq_grid_(0) = freq_min;
  freq_grid_(nfreq_grid-1) = freq_max;

  // partition frequency domain within [freq_min, freq_max]
  if (nfreq_grid > 2) {
    if (freq_scale == 0) { // linear frequency grid
      Real del_freq = (freq_max-freq_min) / (nfreq_-2);
      for (int f=1; f<nfreq_grid-1; ++f)
        freq_grid_(f) = f*del_freq + freq_min;
    } else if (freq_scale == 1) { // log frequency grid
      Real log_freq_max = log(freq_max);
      Real log_freq_min = log(freq_min);
      Real del_log_freq = (log_freq_max-log_freq_min) / (nfreq_-2);
      for (int f=1; f<nfreq_grid-1; ++f) {
        Real log_freq = f*del_log_freq + log_freq_min;
        freq_grid_(f) = exp(log_freq);
      } // endfor f
    } // endelse freq_scale
  } // endif (nfreq_ > 2)

  return;
}


//----------------------------------------------------------------------------------------
//! \fn TaskStatus Radiation::AddMultiFreqRadiationSourceTerm(Driver *pdriver, int stage)
// \brief Add implicit multi-frequency radiation source term.

// TaskStatus Radiation::AddMultiFreqRadiationSourceTerm(Driver *pdriver, int stage) {
//   // Return if radiation source term disabled
//   if (!(rad_source)) {
//     return TaskStatus::complete;
//   }
//
//   // Extract indices, size data, hydro/mhd/units flags, and coupling flags
//   auto &indcs = pmy_pack->pmesh->mb_indcs;
//   int &is = indcs.is, &ie = indcs.ie;
//   int &js = indcs.js, &je = indcs.je;
//   int &ks = indcs.ks, &ke = indcs.ke;
//   int nmb1 = pmy_pack->nmb_thispack - 1;
//   int &nang_  = prgeo->nangles;
//   int &nfreq_ = nfreq;
//   auto &size  = pmy_pack->pmb->mb_size;
//
//   // Extract coordinate/excision data
//   auto &coord = pmy_pack->pcoord->coord_data;
//   bool &flat = coord.is_minkowski;
//   Real &spin = coord.bh_spin;
//
//
//   // Extract radiation constant and units
//   Real temp_unit = 1.0;
//
//   // Real &arad_ = arad;
//   // Real density_scale_ = 1.0, temperature_scale_ = 1.0, length_scale_ = 1.0;
//   // Real mean_mol_weight_ = 1.0;
//   // Real rosseland_coef_ = 1.0, planck_minus_rosseland_coef_ = 0.0;
//   // Real inv_t_electron_ = 1.0;
//   // if (are_units_enabled_) {
//   //   density_scale_ = pmy_pack->punit->density_cgs();
//   //   temperature_scale_ = pmy_pack->punit->temperature_cgs();
//   //   length_scale_ = pmy_pack->punit->length_cgs();
//   //   mean_mol_weight_ = pmy_pack->punit->mu();
//   //   rosseland_coef_ = pmy_pack->punit->rosseland_coef_cgs;
//   //   planck_minus_rosseland_coef_ = pmy_pack->punit->planck_minus_rosseland_coef_cgs;
//   //   inv_t_electron_ = temperature_scale_/pmy_pack->punit->electron_rest_mass_energy_cgs;
//   // }
//
//
//
//
//
//
//
//   /********** Compute Implicit Source Term **********/
//   par_for("multi_freq_radsrc",DevExeSpace(),0,nmb1,ks,ke,js,je,is,ie,KOKKOS_LAMBDA(int m, int k, int j, int i) {
//     // extract spatial position
//     Real &x1min = size.d_view(m).x1min;
//     Real &x1max = size.d_view(m).x1max;
//     Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);
//
//     Real &x2min = size.d_view(m).x2min;
//     Real &x2max = size.d_view(m).x2max;
//     Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);
//
//     Real &x3min = size.d_view(m).x3min;
//     Real &x3max = size.d_view(m).x3max;
//     Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);
//
//     // compute metric and inverse
//     Real glower[4][4], gupper[4][4];
//     ComputeMetricAndInverse(x1v,x2v,x3v,flat,spin,glower,gupper);
//     Real alpha = sqrt(-1.0/gupper[0][0]);
//
//     // fluid state
//     Real &wdn = w0_(m,IDN,k,j,i);
//     Real &wvx = w0_(m,IVX,k,j,i);
//     Real &wvy = w0_(m,IVY,k,j,i);
//     Real &wvz = w0_(m,IVZ,k,j,i);
//     Real &wen = w0_(m,IEN,k,j,i);
//
//     // derived quantities
//     Real pgas = gm1*wen;
//     Real tgas = pgas/wdn;
//     Real q = glower[1][1]*wvx*wvx + 2.0*glower[1][2]*wvx*wvy + 2.0*glower[1][3]*wvx*wvz
//            + glower[2][2]*wvy*wvy + 2.0*glower[2][3]*wvy*wvz
//            + glower[3][3]*wvz*wvz;
//     Real gamma = sqrt(1.0 + q);
//     Real u0 = gamma/alpha;
//
//     // compute fluid velocity in tetrad frame
//     Real u_tet[4];
//     u_tet[0] = (norm_to_tet_(m,0,0,k,j,i)*gamma + norm_to_tet_(m,0,1,k,j,i)*wvx +
//                 norm_to_tet_(m,0,2,k,j,i)*wvy   + norm_to_tet_(m,0,3,k,j,i)*wvz);
//     u_tet[1] = (norm_to_tet_(m,1,0,k,j,i)*gamma + norm_to_tet_(m,1,1,k,j,i)*wvx +
//                 norm_to_tet_(m,1,2,k,j,i)*wvy   + norm_to_tet_(m,1,3,k,j,i)*wvz);
//     u_tet[2] = (norm_to_tet_(m,2,0,k,j,i)*gamma + norm_to_tet_(m,2,1,k,j,i)*wvx +
//                 norm_to_tet_(m,2,2,k,j,i)*wvy   + norm_to_tet_(m,2,3,k,j,i)*wvz);
//     u_tet[3] = (norm_to_tet_(m,3,0,k,j,i)*gamma + norm_to_tet_(m,3,1,k,j,i)*wvx +
//                 norm_to_tet_(m,3,2,k,j,i)*wvy   + norm_to_tet_(m,3,3,k,j,i)*wvz);
//
//     // coordinate component n^0
//     Real n0 = tt(m,0,0,k,j,i);
//
//
//
//     // compute frequency and angle indices
//     // int ifr  = n / nang_;
//     // int iang = n - ifr*nang_;
//
//
//   });
//
//
//   return TaskStatus::complete;
// }


} // namespace radiation
