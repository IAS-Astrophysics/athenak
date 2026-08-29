#ifndef PARTICLES_LAGRANGIAN_MC_HPP_
#define PARTICLES_LAGRANGIAN_MC_HPP_
//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file lagrangian_mc.hpp
//! \brief particle array indices for Lagrangian Monte Carlo particles

namespace particles {
namespace lagrangian_mc {

// The common particle framework requires the first three real fields to be the x1, x2,
// and x3 positions, and the first three integer fields to be the owning GID, durable tag,
// and lifecycle status. The minimum fields record the smallest Euclidean coordinate
// radius from the origin reached by the particle.
enum RealIndex {
  IPX=0, IPY=1, IPZ=2, IPXMIN=3, IPYMIN=4, IPZMIN=5, IPTMIN=6, NREAL=7
};
// The refinement correction uses the packed source-cell state after generic particle
// migration has assigned the destination MeshBlock.
enum IntIndex {
  PGID=0, PTAG=1, PSTATUS=2, PLASTMOVE=3, PSOURCECELL=4, NINT=5
};
inline constexpr const char *real_names[NREAL] = {
  "x", "y", "z", "x_min", "y_min", "z_min", "t_min"
};
inline constexpr const char *int_names[NINT] = {
  "gid", "ptag", "status", "last_move", "source_cell"
};
inline constexpr bool real_output[NREAL] = {
  true, true, true, true, true, true, true
};
inline constexpr bool int_output[NINT] = {
  false, true, true, false, false
};
constexpr int RESTART_LAYOUT_VERSION = 1;
enum MoveDirection {
  PMOVE_NONE=0, PMOVE_X1_LEFT, PMOVE_X1_RIGHT, PMOVE_X2_LEFT,
  PMOVE_X2_RIGHT, PMOVE_X3_LEFT, PMOVE_X3_RIGHT
};
enum SourceCellParityBit {
  PSOURCE_X1_ODD=1, PSOURCE_X2_ODD=2, PSOURCE_X3_ODD=4
};
// Bit 3 distinguishes a newly moved particle from retained particles whose last move has
// already been corrected; the source refinement level occupies the remaining high bits.
constexpr int PSOURCE_PARITY_MASK = 7;
constexpr int PSOURCE_CORRECTION_PENDING = 8;
constexpr int PSOURCE_LEVEL_SHIFT = 4;

KOKKOS_INLINE_FUNCTION
int EncodeSourceCell(const int level, const int i, const int j, const int k) {
  const int parity = ((i & 1) ? PSOURCE_X1_ODD : 0) |
                     ((j & 1) ? PSOURCE_X2_ODD : 0) |
                     ((k & 1) ? PSOURCE_X3_ODD : 0);
  return (level << PSOURCE_LEVEL_SHIFT) | parity;
}

KOKKOS_INLINE_FUNCTION
int SourceLevel(const int source_cell) {
  return source_cell >> PSOURCE_LEVEL_SHIFT;
}

KOKKOS_INLINE_FUNCTION
int SourceParity(const int source_cell) {
  return source_cell & PSOURCE_PARITY_MASK;
}

KOKKOS_INLINE_FUNCTION
bool SourceCorrectionPending(const int source_cell) {
  return (source_cell & PSOURCE_CORRECTION_PENDING) != 0;
}

KOKKOS_INLINE_FUNCTION
int MarkSourceCorrectionPending(const int source_cell) {
  return source_cell | PSOURCE_CORRECTION_PENDING;
}

KOKKOS_INLINE_FUNCTION
int MarkSourceCorrectionComplete(const int source_cell) {
  return source_cell & ~PSOURCE_CORRECTION_PENDING;
}

} // namespace lagrangian_mc
} // namespace particles

#endif // PARTICLES_LAGRANGIAN_MC_HPP_
