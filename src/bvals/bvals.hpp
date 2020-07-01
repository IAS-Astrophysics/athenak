#ifndef BVALS_BVALS_HPP_
#define BVALS_BVALS_HPP_
//==================================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//==================================================================================================
//! \file bvals.hpp
//  \brief defines BoundaryBase, BoundaryValues classes used for setting BCs on all data


// identifiers for all 6 faces of a MeshBlock
enum BoundaryFace {undef=-1, inner_x1, outer_x1, inner_x2, outer_x2, inner_x3, outer_x3};

// identifiers for boundary conditions
enum class BoundaryFlag {block=-1, undef, reflect, outflow, user, periodic};

// identifiers for types of neighbor blocks (connectivity with current MeshBlock)
enum class NeighborConnection {none, face, edge, corner};
  
// identifiers for status of MPI boundary communications
enum class BoundaryStatus {waiting, arrived, completed};


#endif // BVALS_BVALS_HPP_

