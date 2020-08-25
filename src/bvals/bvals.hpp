#ifndef BVALS_BVALS_HPP_
#define BVALS_BVALS_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
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

// free functions to return boundary flag given input string, and vice versa
BoundaryFlag GetBoundaryFlag(const std::string& input_string);
std::string GetBoundaryString(BoundaryFlag input_flag);

#include "athena.hpp"
#include "athena_arrays.hpp"
#include "parameter_input.hpp"

// Forward delcarations
class Mesh;

//----------------------------------------------------------------------------------------
//! \class BoundaryBase
//  \brief

class BoundaryValues {
 public:
  BoundaryValues(Mesh* pm, std::unique_ptr<ParameterInput> &pin, int gid, BoundaryFlag *bcs, int maxvar);
  ~BoundaryValues();

  // data

  BoundaryFlag bflags[6]; // enums specifying BCs at all 6 faces of this MeshBlock
  AthenaArray<Real> cc_bbuf_x1face, cc_bbuf_x2face, cc_bbuf_x3face;
  AthenaArray<Real> cc_bbuf_x1x2ed, cc_bbuf_x3x1ed, cc_bbuf_x2x3ed;
  AthenaArray<Real> cc_bbuf_corner;

  // functions
  void SendCellCenteredVariables(AthenaArray<Real> &a, int nvar);
  void ReceiveCellCenteredVariables(AthenaArray<Real> &a, int nvar);

 private:
  Mesh *pmesh_;
  int my_mbgid_;

};



#endif // BVALS_BVALS_HPP_

