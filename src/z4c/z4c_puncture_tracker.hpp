//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================

#ifndef Z4C_Z4C_PUNCTURE_TRACKER_HPP_
#define Z4C_Z4C_PUNCTURE_TRACKER_HPP_

#include <cstdio>
#include <string>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "z4c_macros.hpp"

// Forward declaration
class Mesh;
class ParameterInput;

namespace z4c {

//! \class PunctureTracker
//! \brief Tracks a single puncture
class PunctureTracker {
 public:
  //! Initialize a tracker
  PunctureTracker(Mesh *pmesh, ParameterInput *pin, int n);
  //! Destructor (will close output file)
  ~PunctureTracker();
  //! Interpolate the shift vector to the puncture position
  void InterpolateShift(MeshBlockPack *pmbp);
  //! Update and broadcast the puncture position
  void EvolveTracker();
  //! Write data to file
  void WriteTracker() const;
  //! Get position
  inline Real GetPos(int a) {
    return pos[a];
  }
 private:
  bool owns_puncture;
  bool bitant;
  Real pos[NDIM];
  Real betap[NDIM];
  Mesh const *pmesh;
  std::string ofname;
  FILE *pofile;
};

} // end namespace z4c

#endif // Z4C_Z4C_PUNCTURE_TRACKER_HPP_
