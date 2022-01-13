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

//----------------------------------------------------------------------------------------
//! \struct CoordData
//! \brief container for Coordinate variables and functions needed inside kernels. Storing
//! everything in a container makes them easier to capture, and pass to inline functions,
//! inside kernels.

struct CoordData {
  // following data is only used in GR calculations to compute metric
  bool is_minkowski;               // flag to specify Minkowski (flat) space
  Real bh_mass;                    // needed for GR metric
  Real bh_spin;                    // needed for GR metric
  Real bh_rmin;                    // needed for GR cons2prim
};

//----------------------------------------------------------------------------------------
//! \class Coordinates
//! \brief data and functions for coordinates

class Coordinates {
 public:
  explicit Coordinates(MeshBlockPack *ppack);
  ~Coordinates() {}

  CoordData coord_data;

  // functions
  void InitMetric(ParameterInput *pin);
  void AddCoordTerms(const DvceArray5D<Real> &w0, const EOS_Data &eos, const Real dt,
                     DvceArray5D<Real> &u0);
  void AddCoordTerms(const DvceArray5D<Real> &w0, const DvceArray5D<Real> &bcc,
                     const EOS_Data &eos, const Real dt, DvceArray5D<Real> &u0);

 private:
  MeshBlockPack* pmy_pack;
};

#endif // COORDINATES_COORDINATES_HPP_
