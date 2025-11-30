#ifndef MESH_CYCLIC_ZOOM_HPP_
#define MESH_CYCLIC_ZOOM_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file cyclic_zoom.hpp
//  \brief definitions for CyclicZoom class

//----------------------------------------------------------------------------------------
//! \struct ZoomState
//! \brief runtime state of cyclic zoom AMR, updated during simulation, used for restart

typedef struct ZoomState {
  int id;                       // zoom event id
  int zone;                     // zone number = level_max - level
  int last_zone;                // last zone number
  int direction;                // direction of zooming (+1: coarsen, -1: refine)
  Real next_time;               // time of next zoom
} ZoomState;

//----------------------------------------------------------------------------------------
//! \struct ZoomAMR
//! \brief parameters of cyclic zoom AMR

typedef struct ZoomAMR {
  int nlevels;                  // number of levels
  int max_level;                // maximum level number
  int min_level;                // minimum level number
  int level;                    // level number = max_level - zone
  int refine_flag;              // flag for refinement (+1: refine, -1: coarsen)
  bool zooming_in;              // flag for performing zoom in (refinement)
  bool zooming_out;             // flag for performing zoom out (coarsening)
  bool first_emf;               // flag for first electric field
  bool dump_rst;                // flag for dumping restart file
} ZoomAMR;

//----------------------------------------------------------------------------------------
//! \struct ZoomRegion
//! \brief parameters of zoom region

typedef struct ZoomRegion {
  // TODO(@mhguo): may add box region later
  // Real x1min, x2min, x3min;     // box minimum of zoom region
  // Real x1max, x2max, x3max;     // box maximum of zoom region
  Real x1c, x2c, x3c;           // center of zoom region
  Real r_0;                     // radius of zoom region at zone 0
  Real radius;                  // radius of zoom region
} ZoomRegion;

//----------------------------------------------------------------------------------------
//! \struct ZoomInterval
//! \brief parameters of zoom interval

typedef struct ZoomInterval {
  Real t_run_fac;               // interval factor
  Real t_run_pow;               // interval power law
  Real t_run_max;               // maximum interval
  std::vector<Real> t_run_fac_zones;        // runtime factors for zones (dynamic)
  // Constructor for initialization
  ZoomInterval() = default;
  // Initialize with given number of zones
  void initialize(int num_zones) {
    t_run_fac_zones.resize(num_zones);
  }
  Real runtime;                 // interval for zoom
} ZoomInterval;

// Forward declaration
class ZoomData;

//----------------------------------------------------------------------------------------
//! \class CyclicZoom

class CyclicZoom
{
  friend class ZoomData;
 public:
  CyclicZoom(Mesh *pmesh, ParameterInput *pin);
  ~CyclicZoom() = default;

  // data
  bool is_set;
  bool read_rst;           // flag for reading zoom data restart file
  bool write_rst;          // flag for writing zoom data restart file
  bool zoom_bcs;           // flag for zoom boundary conditions
  bool zoom_ref;           // flag for zoom refinement
  bool zoom_dt;            // flag for zoom time step
  bool fix_efield;         // flag for fixing electric field
  bool dump_diag;          // flag for dumping diagnostic output
  int ndiag;               // cycles between diagostic output
  int mzoom;               // number of zoom meshblocks
  int nleaf;               // number of zoom meshblocks on each level
  int nvars;               // number of variables
  int nflux;               // number of fluxes through spherical surfaces
  int emf_flag;            // flag for modifying electric field
  Real d_zoom;             // density within inner boundary
  Real p_zoom;             // pressure within inner boundary
  Real emf_f0, emf_f1;     // electric field factor, e = f0 * e0 + f1 * e1
  Real emf_fmax;           // maximum electric field factor
  int  emf_zmax;           // maximum zone number for electric field
  Real re_fac;             // factor for electric field
  Real r0_efld;            // modify e if r < r0_efld

  ZoomAMR zamr;            // zoom AMR parameters
  ZoomInterval zint;       // zoom interval parameters
  ZoomRegion zregion;      // zoom region parameters
  ZoomState zstate;        // zoom runtime state

  // array_sum::GlobalSum nc1, nc2, nc3, em1, em2, em3;
  ZoomData *pzdata;        // zoom data

  // functions
  void Initialize();
  void Update(const bool restart);
  void PrintInfo();
  // AMR functions
  void CheckRefinement();
  void UpdateState();
  void SetRegionAndInterval();
  void SetRefinementFlags();
  void UpdateVariables();
  void UpdateHydroVariables(int zm, int m);
  void SyncVariables();
  void UpdateGhostVariables();
  void ApplyVariables();
  // void StoreZoomRegion(){};
  // void InitZoomRegion(){};
  void WorkBeforeAMR();
  void WorkAfterAMR();
  // Boundary conditions, fluxes and source terms
  void BoundaryConditions();
  void FixEField(DvceEdgeFld4D<Real> emf);
  // void AddEField(DvceEdgeFld4D<Real> emf);
  void AddDeltaEField(DvceEdgeFld4D<Real> emf);
  void UpdateDeltaEField(DvceEdgeFld4D<Real> emf);
  void SyncZoomEField(DvceEdgeFld4D<Real> emf, int zid);
  void SetMaxEField();
  // timestep functions
  Real NewTimeStep(Mesh* pm);
  Real GRTimeStep(Mesh* pm);
  Real EMFTimeStep(Mesh* pm);
  // restart functions
  IOWrapperSizeT RestartFileSize();
  void WriteRestartFile(IOWrapper &resfile);
  void ReadRestartFile(IOWrapper &resfile);

 private:
  // data
  Mesh *pmesh;            // ptr to Mesh containing this CyclicZoom module
};

//----------------------------------------------------------------------------------------
//! \class ZoomMesh
class ZoomMesh
{
 public:
  ZoomMesh() = default;
  ~ZoomMesh() = default;

  // data
  int zm_id;               // zoom mesh id
  int zm_level;            // zoom mesh level
  int zm_nx1, zm_nx2, zm_nx3; // zoom mesh dimensions
  Real zm_dx1, zm_dx2, zm_dx3; // zoom mesh cell sizes
  Real zm_x1min, zm_x1max; // zoom mesh x1 min/max
  Real zm_x2min, zm_x2max; // zoom mesh x2 min/max
  Real zm_x3min, zm_x3max; // zoom mesh x3 min/max

 private:
  CyclicZoom *pzoom;       // ptr to CyclicZoom containing this ZoomMesh module
};

//----------------------------------------------------------------------------------------
//! \class ZoomData
class ZoomData
{
  friend class CyclicZoom;
 public:
  ZoomData(CyclicZoom *pz);
  ~ZoomData() = default;
  // data
  // std::vector<HostArray5D<Real>> vu0;  // Vector of conserved variables
  // std::vector<HostArray5D<Real>> vw0;  // Vector of primitive variables
  // std::vector<HostArray5D<Real>> vcoarse_u0;  // Vector of coarse conserved variables
  // std::vector<HostArray5D<Real>> vcoarse_w0;  // Vector of coarse primitive variables
  // std::vector<HostEdgeFld4D<Real>> vef0; // Vector of edge-centered electric fields just after zoom
  // std::vector<HostEdgeFld4D<Real>> vdelta_efld; // Vector of change in electric fields
  
  DvceArray5D<Real> u0;    // conserved variables
  DvceArray5D<Real> w0;    // primitive variables
  DvceArray5D<Real> coarse_u0;  // coarse conserved variables
  DvceArray5D<Real> coarse_w0;  // coarse primitive variables
  // DvceArray5D<Real> coarse_wuh; // coarse primitive variables from hydro conserved variables

  // following only used for time-evolving flow
  DvceEdgeFld4D<Real> efld;   // edge-centered electric fields (fluxes of B)
  DvceEdgeFld4D<Real> emf0;   // edge-centered electric fields just after zoom
  DvceEdgeFld4D<Real> delta_efld; // change in electric fields

  HostArray2D<Real> max_emf0;  // maximum electric field

  HostArray5D<Real> harr_5d;  // host copy of 5D arrays
  HostArray4D<Real> harr_4d;  // host copy of 4D arrays

  // functions
  void Initialize();
  void DumpData();

 private:
  CyclicZoom *pzoom;       // ptr to CyclicZoom containing this ZoomVariable module
};

#endif // MESH_CYCLIC_ZOOM_HPP_
