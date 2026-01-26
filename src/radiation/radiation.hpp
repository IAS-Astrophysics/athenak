#ifndef RADIATION_RADIATION_HPP_
#define RADIATION_RADIATION_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation.hpp
//  \brief definitions for Radiation class

#include <map>
#include <memory>
#include <string>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "tasklist/task_list.hpp"
#include "bvals/bvals.hpp"

// forward declarations
class EquationOfState;
class Coordinates;
class SourceTerms;
class GeodesicGrid;
class Driver;

//----------------------------------------------------------------------------------------
//! \struct RadiationTaskIDs
//  \brief container to hold TaskIDs of all radiation tasks

struct RadiationTaskIDs {
  TaskID rad_irecv;
  TaskID mhd_irecv;
  TaskID hyd_irecv;
  TaskID copyu;
  TaskID rad_flux;
  TaskID mhd_flux;
  TaskID hyd_flux;
  TaskID rad_sendf;
  TaskID mhd_sendf;
  TaskID hyd_sendf;
  TaskID rad_recvf;
  TaskID mhd_recvf;
  TaskID hyd_recvf;
  TaskID rad_rkupdt;
  TaskID mhd_rkupdt;
  TaskID hyd_rkupdt;
  TaskID rad_src;
  TaskID mhd_src;
  TaskID hyd_src;
  TaskID rad_coupl;
  TaskID rad_resti;
  TaskID hyd_restu;
  TaskID mhd_restu;
  TaskID rad_sendi;
  TaskID mhd_sendu;
  TaskID hyd_sendu;
  TaskID rad_recvi;
  TaskID mhd_recvu;
  TaskID hyd_recvu;
  TaskID mhd_efld;
  TaskID mhd_sende;
  TaskID mhd_recve;
  TaskID mhd_ct;
  TaskID mhd_restb;
  TaskID mhd_sendb;
  TaskID mhd_recvb;
  TaskID bcs;
  TaskID rad_prol;
  TaskID mhd_prol;
  TaskID hyd_prol;
  TaskID mhd_c2p;
  TaskID hyd_c2p;
  TaskID rad_csend;
  TaskID mhd_csend;
  TaskID hyd_csend;
  TaskID rad_crecv;
  TaskID mhd_crecv;
  TaskID hyd_crecv;
};

namespace radiation {

//----------------------------------------------------------------------------------------
//! \class Radiation

class Radiation {
 public:
  Radiation(MeshBlockPack *ppack, ParameterInput *pin);
  ~Radiation();

  // flags to denote hydro/mhd is enabled or units enabled
  bool is_hydro_enabled;
  bool is_mhd_enabled;
  bool are_units_enabled;

  // Radiation coupling term parameters
  bool rad_source;          // flag to enable/disable radiation source term
  bool fixed_fluid;         // flag to enable/disable fluid integration
  bool affect_fluid;        // flag to enable/disable feedback of rad field on fluid
  Real arad;                // radiation constant
  Real kappa_a;             // constant Rosseland mean absoprtion coefficient
  Real kappa_s;             // constant scattering coefficient
  Real kappa_p;             // Planck - Rosseland mean coefficient
  bool power_opacity;       // flag to enable Kramer's law opacity for kappa_a
  bool is_compton_enabled;  // flag to enable/disable compton

  // Flags and parameters for ad hoc fixes
  bool correct_radsrc_velocity;
  bool correct_radsrc_opacity;
  Real dfloor_opacity;
  Real dens_trunc_max;
  Real tau_truncation;
  Real sigmoid_residual; // sigmoid residual must be less than 1./3

  // radiation source term (i.e., beam)
  SourceTerms *psrc = nullptr;

  // Angular mesh
  bool rotate_geo;                    // rotate geodesic mesh
  bool angular_fluxes;                // flag to enable/disable angular fluxes
  Real n_0_floor;                     // floor on n_0
  GeodesicGrid *prgeo = nullptr;      // pointer to radiation angular mesh

  // Tetrad arrays and functions
  DualArray2D<Real> nh_c;             // normal vector computed at face center
  DualArray3D<Real> nh_f;             // normal vector computed at face edges
  DvceArray6D<Real> tet_c;            // tetrad components at cell centers
  DvceArray6D<Real> tetcov_c;         // covariant tetrad components at cell centers
  DvceArray5D<Real> tet_d1_x1f;       // tetrad components (subset) at x1f
  DvceArray5D<Real> tet_d2_x2f;       // tetrad components (subset) at x2f
  DvceArray5D<Real> tet_d3_x3f;       // tetrad components (subset) at x3f
  DvceArray6D<Real> na;               // n^a
  DvceArray6D<Real> norm_to_tet;      // used in transform b/w normal frame and tet frame
  void SetOrthonormalTetrad();

  // intensity arrays
  DvceArray5D<Real> i0;         // intensities
  DvceArray5D<Real> coarse_i0;  // intensities on 2x coarser grid (for SMR/AMR)

  // Boundary communication buffers and functions for i
  MeshBoundaryValuesCC *pbval_i;

  // following only used for time-evolving flow
  DvceArray5D<Real> i1;         // intensity at intermediate step
  DvceFaceFld5D<Real> iflx;     // spatial fluxes on zone faces
  DvceArray5D<Real> divfa;      // angular flux divergence
  Real dtnew;

  // reconstruction method
  ReconstructionMethod recon_method;

  // container to hold names of TaskIDs
  RadiationTaskIDs id;

  // functions...
  void AssembleRadTasks(std::map<std::string, std::shared_ptr<TaskList>> tl);
  // ...in "before_stagen_tl" task list
  TaskStatus InitRecv(Driver *d, int stage);
  // ...in "stagen_tl" task list
  TaskStatus CopyCons(Driver *d, int stage);
  TaskStatus CalculateFluxes(Driver *d, int stage);
  TaskStatus SendFlux(Driver *d, int stage);
  TaskStatus RecvFlux(Driver *d, int stage);
  TaskStatus RKUpdate(Driver *d, int stage);
  TaskStatus RadSrcTerms(Driver *d, int stage);
  TaskStatus RadFluidCoupling(Driver *d, int stage);
  TaskStatus RestrictI(Driver *d, int stage);
  TaskStatus SendI(Driver *d, int stage);
  TaskStatus RecvI(Driver *d, int stage);
  TaskStatus ApplyPhysicalBCs(Driver* pdrive, int stage);
  TaskStatus Prolongate(Driver* pdrive, int stage);
  TaskStatus NewTimeStep(Driver *d, int stage);
  // ...in "after_stagen_tl" task list
  TaskStatus ClearSend(Driver *d, int stage);
  TaskStatus ClearRecv(Driver *d, int stage);

 private:
  MeshBlockPack* pmy_pack;  // ptr to MeshBlockPack containing this Radiation
};

} // namespace radiation
#endif // RADIATION_RADIATION_HPP_
