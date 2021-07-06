#ifndef COORDINATES_COORDINATES_HPP_
#define COORDINATES_COORDINATES_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file coordinates.hpp
//! \brief implemention of coordinates for GR.  Currently only Cartesian Kerr-Schild

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"

//----------------------------------------------------------------------------------------
//! \class Coordinates
//! \brief data and functions for coordinates

class SourceTerms
{
 public:
  Coordinates(std::string block, MeshBlockPack *pp, ParameterInput *pin);
  ~Coordinates();

  // data

  // functions
  void AddCoordTerms(DvceArray5D<Real> &u0,const DvceArray5D<Real> &w0,const Real dt);

 private:
  MeshBlockPack* pmy_pack;
};

#endif // COORDINATES_COORDINATES_HPP_
