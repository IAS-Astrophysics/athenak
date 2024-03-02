#ifndef SHEARING_BOX_SHEARING_BOX_HPP_
#define SHEARING_BOX_SHEARING_BOX_HPP_
//========================================================================================
// AthenaK astrophysical fluid dynamics code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file shearing_box.hpp
//  \brief definitions for ShearingBox class

#include "athena.hpp"
#include "parameter_input.hpp"
#include "tasklist/task_list.hpp"
#include "bvals/bvals.hpp"

// forward declarations


//----------------------------------------------------------------------------------------
//! \struct ShearingBoxTaskIDs
//  \brief container to hold TaskIDs of all shearing_box tasks

struct ShearingBoxTaskIDs {
  TaskID irecv;
  TaskID copyu;
  TaskID flux;
  TaskID sendf;
  TaskID recvf;
  TaskID csend;
  TaskID crecv;
};

namespace shearing_box {

//----------------------------------------------------------------------------------------
//! \class ShearingBoc

class ShearingBox {
 public:
  ShearingBox(Mesh *pm, ParameterInput *pin);
  ~ShearingBox();

  // data
  Real qshear, omega0;     // shearing box parameters
  int jshift;              // integer offset for orbital advection

  // container to hold names of TaskIDs
  ShearingBoxTaskIDs id;

  // functions...

 private:
  Mesh *pmy_mesh;
};

} // namespace shearing_box
#endif // SHEARING_BOX_SHEARING_BOX_HPP_
