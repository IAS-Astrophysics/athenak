//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================

#ifndef Z4C_COMPACT_OBJECT_TRACKER_HPP_
#define Z4C_COMPACT_OBJECT_TRACKER_HPP_

#include <cstdio>
#include <fstream>
#include <string>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "z4c_macros.hpp"

// Forward declaration
class Mesh;
class ParameterInput;

//! \class CompactObjectTracker
//! \brief Tracks a single puncture
class CompactObjectTracker {
  enum CompactObjectType { BlackHole, NeutronStar };
  enum TrackerMode { ODE, Walk };

 public:
  //! Initialize a tracker
  CompactObjectTracker(Mesh *pmesh, ParameterInput *pin, int n);
  //! Destructor (will close output file)
  ~CompactObjectTracker();
  //! Interpolate the shift vector to the puncture position
  void InterpolateVelocity(MeshBlockPack *pmbp);
  //! Update and broadcast the puncture position
  void EvolveTracker(MeshBlockPack *pmbp);
  //! Write data to file
  void WriteTracker();
  //! Get position array
  inline Real * GetPos() {
    return &pos[0];
  }
  //! Get position
  inline Real GetPos(int a) const {
    return pos[a];
  }
  //! Set the position of the CO
  inline void SetPos(Real npos[NDIM]) {
    std::memcpy(pos, npos, NDIM*sizeof(Real));
  }
  //! Get wanted refinement level
  inline int GetReflevel() const {
    return reflevel;
  }
  //! Get radius
  inline Real GetRadius() const {
    return radius;
  }

 private:
  bool owns_compact_object;
  CompactObjectType type;
  TrackerMode mode;
  Real vel[NDIM];
  int reflevel;         // requested minimum refinement level (-1 for infinity)
  Real radius;          // nominal radius of the object (for the AMR driver)
  Mesh const *pmesh;
  int out_every;
  std::ofstream ofile;
  Real pos[NDIM];
};

#endif // Z4C_COMPACT_OBJECT_TRACKER_HPP_
