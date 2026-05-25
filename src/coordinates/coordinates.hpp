#ifndef COORDINATES_COORDINATES_HPP_
#define COORDINATES_COORDINATES_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file coordinates.hpp
//! \brief implemention of light-weight coordinates class.  Provides data structure that
//! stores array of RegionSizes over (# of MeshBlocks), and inline functions for
//! computing positions.  In GR, also provides inline metric functions (currently only
//! Cartesian Kerr-Schild)

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"

// forward declarations
struct EOS_Data;

// Enumerator for the excision method
enum class ExcisionScheme {
  fixed,
  lapse,
  puncture
};

enum class SmoothExcisionPunctureWeight {
  smoothstep,
  slow_start,
  smoother_start
};

KOKKOS_INLINE_FUNCTION
Real ExcisionKSRXSpin(const Real x1, const Real x2, const Real x3,
                      const Real ax, const Real ay, const Real az) {
  Real a2 = SQR(ax) + SQR(ay) + SQR(az);
  if (a2 <= 0.0) {
    return sqrt(SQR(x1) + SQR(x2) + SQR(x3));
  }
  Real a = sqrt(a2);
  Real rad2 = SQR(x1) + SQR(x2) + SQR(x3);
  Real zspin = (x1*ax + x2*ay + x3*az)/a;
  return sqrt((rad2 - a2 + sqrt(SQR(rad2 - a2) + 4.0*a2*SQR(zspin)))/2.0);
}

KOKKOS_INLINE_FUNCTION
void ExcisionBoostedDisplacement(const Real x1, const Real x2, const Real x3,
                                 const Real cx, const Real cy, const Real cz,
                                 const Real vx, const Real vy, const Real vz,
                                 Real *xbh, Real *ybh, Real *zbh) {
  Real dx = x1 - cx, dy = x2 - cy, dz = x3 - cz;
  Real v2 = SQR(vx) + SQR(vy) + SQR(vz);
  Real q;
  if (v2 < 1.0e-12) {
    q = 0.5 + 0.375*v2 + 0.3125*SQR(v2);
  } else {
    Real gamma = 1.0/sqrt(fmax(1.0 - v2, 1.0e-300));
    q = (gamma - 1.0)/v2;
  }
  Real vd = vx*dx + vy*dy + vz*dz;
  *xbh = dx + q*vx*vd;
  *ybh = dy + q*vy*vd;
  *zbh = dz + q*vz*vd;
}

//----------------------------------------------------------------------------------------
//! \struct CoordData
//! \brief container for Coordinate variables and functions needed inside kernels. Storing
//! everything in a container makes them easier to capture, and pass to inline functions,
//! inside kernels.

struct CoordData {
  // following data is only used in GR calculations to compute metric
  bool is_minkowski;               // flag to specify Minkowski (flat) space
  Real bh_spin;                    // needed for GR metric
  bool bh_excise;                  // flag to specify excision
  Real rexcise;                    // excision radius (SKS)
  Real dexcise;                    // rest-mass density inside excised region
  Real pexcise;                    // pressure inside excised region
  Real texcise;                    // temperature target inside excised region
  Real flux_excise_r;              // reduce to first-order inside this radius
  ExcisionScheme excision_scheme;  // excision method
  Real excise_lapse;               // if excision_scheme = lapse, excise under this lapse
  Real punc_0[3];                  // if excision_scheme = puncture, excise within a radius
  Real punc_1[3];                  // of punc_n_rad of punc_n
  Real punc_0_spin[3];             // Kerr a-vector used for puncture excision shape
  Real punc_1_spin[3];
  Real punc_0_vel[3];              // boost velocity used for puncture excision shape
  Real punc_1_vel[3];
  Real punc_0_rad;
  Real punc_1_rad;
  Real punc_flux_rad_factor;        // puncture flux-excision radius / punc_rad
  bool smooth_excise;              // smoothly drain primitive variables inside horizon
  SmoothExcisionPunctureWeight smooth_excise_puncture_weight;
  Real smooth_excise_width;        // radial width of drain layer inside geometric masks
  Real smooth_excise_puncture_width_fraction; // puncture smooth width / punc_rad
  Real smooth_excise_lapse_width;  // lapse width of drain layer for lapse masks
  Real smooth_excise_sigma_max;    // optional B^2/rho cap inside smooth excision
  Real smooth_excise_temp_ceil;     // optional hard p/rho ceiling inside smooth region
  bool smooth_excise_inflow;        // enforce minimum puncture-frame radial inflow
  Real smooth_excise_inflow_speed;  // coordinate radial inflow speed target
};

//----------------------------------------------------------------------------------------
//! \class Coordinates
//! \brief data and functions for coordinates

class Coordinates {
 public:
  explicit Coordinates(ParameterInput *pin, MeshBlockPack *ppack);
  ~Coordinates() {}

  // flags to denote relativistic dynamics in these coordinates
  bool is_special_relativistic = false;
  bool is_general_relativistic = false;
  bool is_dynamical_relativistic = false;

  // data needed to compute metric in GR
  CoordData coord_data;

  // excision masks
  DvceArray4D<bool> excision_floor;  // cell-centered mask for C2P flooring about horizon
  DvceArray4D<bool> excision_flux;   // cell-centered mask for FOFC about horizon
  DvceArray4D<Real> excision_weight;  // 0 outside, 1 in fully drained excision core

  // functions
  void CoordSrcTerms(const DvceArray5D<Real> &w0, const EOS_Data &eos, const Real dt,
                     DvceArray5D<Real> &u0);
  void CoordSrcTerms(const DvceArray5D<Real> &w0, const DvceArray5D<Real> &bcc,
                     const EOS_Data &eos, const Real dt, DvceArray5D<Real> &u0);
  void SetExcisionMasks(DvceArray4D<bool> &floor, DvceArray4D<bool> &flux);

  void UpdateExcisionMasks();

 private:
  MeshBlockPack* pmy_pack;
};

#endif // COORDINATES_COORDINATES_HPP_
