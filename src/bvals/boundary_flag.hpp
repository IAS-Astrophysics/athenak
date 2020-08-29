//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file boundary_flag.cpp
//  \brief utilities for processing the user's input <mesh> ixn_bc, oxn_bc parameters and
// the associated internal BoundaryFlag enumerators

#include <iostream>
#include <sstream>
//#include <string>

#include "bvals.hpp"

// identifiers for boundary conditions
enum class BoundaryFlag {block=0, reflect, outflow, user, periodic};

//----------------------------------------------------------------------------------------
//! \fn GetBoundaryFlag(std::string input_string)
//  \brief Parses input string to return scoped enumerator flag specifying boundary
//  condition. Typically called in Mesh() ctor and in pgen/*.cpp files.

static BoundaryFlag GetBoundaryFlag(const std::string& input_string) {
  if (input_string == "reflecting") {
    return BoundaryFlag::reflect;
  } else if (input_string == "outflow") {
    return BoundaryFlag::outflow;
  } else if (input_string == "user") {
    return BoundaryFlag::user;
  } else if (input_string == "periodic") {
    return BoundaryFlag::periodic;
  } else {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Input string = '" << input_string << "' is an invalid boundary type"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

//----------------------------------------------------------------------------------------
//! \fn GetBoundaryString(BoundaryFlag input_flag)
//  \brief Parses enumerated type BoundaryFlag internal integer representation to return
//  string describing the boundary condition. Typicall used to format descriptive errors
//  or diagnostics. Inverse of GetBoundaryFlag().

static std::string GetBoundaryString(BoundaryFlag input_flag) {
  switch (input_flag) {
    case BoundaryFlag::block:  // 0
      return "block";
    case BoundaryFlag::reflect:
      return "reflecting";
    case BoundaryFlag::outflow:
      return "outflow";
    case BoundaryFlag::user:
      return "user";
    case BoundaryFlag::periodic:
      return "periodic";
    default:
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
         << std::endl << "Input enum class BoundaryFlag=" << static_cast<int>(input_flag)
         << " is an invalid boundary type" << std::endl;
      std::exit(EXIT_FAILURE);
      break;
  }
}
