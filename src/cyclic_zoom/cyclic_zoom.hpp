#ifndef CYCLIC_ZOOM_CYCLIC_ZOOM_HPP_
#define CYCLIC_ZOOM_CYCLIC_ZOOM_HPP_
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

  // TODO(@mhguo): may need to check whether in previous zoom region
  // Kokkos inline function to check if a location is within the zoom region
  KOKKOS_INLINE_FUNCTION
  bool IsInZoomRegion(Real x1, Real x2, Real x3) const {
    return (SQR(x1 - x1c) + SQR(x2 - x2c) + SQR(x3 - x3c) <= SQR(radius));
  }
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
class ZoomMesh;
class ZoomData;

//----------------------------------------------------------------------------------------
//! \class CyclicZoom
//! \brief Cyclic Zoom AMR module

class CyclicZoom
{
  friend class ZoomMesh;
  friend class ZoomData;
 public:
  CyclicZoom(Mesh *pmesh, ParameterInput *pin);
  ~CyclicZoom() = default;

  // data
  std::string block_name;  // block name for reading parameters
  bool is_set;
  bool read_rst;           // flag for reading zoom data restart file
  bool write_rst;          // flag for writing zoom data restart file
  bool zoom_bcs;           // flag for zoom boundary conditions
  bool zoom_ref;           // flag for zoom refinement
  bool zoom_dt;            // flag for zoom time step
  bool add_emf;            // flag for fixing electric field
  bool dump_diag;          // flag for dumping diagnostic output
  int ndiag;               // cycles between diagostic output
  int nflux;               // number of fluxes through spherical surfaces
  int emf_flag;            // flag for modifying electric field
  Real emf_f0, emf_f1;     // electric field factor, e = f0 * e0 + f1 * e1
  Real emf_fmax;           // maximum electric field factor
  int  emf_zmax;           // maximum zone number for electric field
  Real re_fac;             // factor for electric field
  Real r0_efld;            // modify e if r < r0_efld

  ZoomAMR zamr;            // zoom AMR parameters
  ZoomInterval zint;       // zoom interval parameters
  ZoomRegion zregion;      // zoom region parameters
  ZoomRegion old_zregion;  // previous zoom region parameters
  ZoomState zstate;        // zoom runtime state

  // array_sum::GlobalSum nc1, nc2, nc3, em1, em2, em3;
  ZoomMesh *pzmesh;        // zoom mesh
  ZoomData *pzdata;        // zoom data

  // functions
  void Initialize(ParameterInput *pin);
  void Update(const bool restart);
  void PrintCyclicZoomDiagnostics();
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
  void WorkAfterAMR(Driver *pdriver);
  // For zoom refinement
  // TODO(@mhguo): need to reorganize these functions, now simply for compilation
  void StoreVariables();
  void CorrectVariables();
  void ReinitVariables();
  void MaskVariables();
  void FindMaskRegion();
  void FindReinitRegion();
  bool CheckStoreFlag(int m);
  void UpdateElectricFields(Driver *pdriver);
  void SourceTermsFC(DvceEdgeFld4D<Real> emf);
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
//! \brief Handles Zoom Mesh structures
class ZoomMesh
{
 public:
  ZoomMesh(CyclicZoom *pz, ParameterInput *pin);
  ~ZoomMesh();

  // data
  int max_level;           // maximum zoom mesh level
  int min_level;           // minimum zoom mesh level
  int nlevels;             // number of zoom mesh levels
  // int mzoom;               // number of zoom meshblocks
  int nleaf;               // number of zoom meshblocks on each level
  int nzmb_total;          // total number of Zoom MeshBlocks across all levels/ranks
  // int nmb_thisrank;        // number of MeshBlocks on this MPI rank (local)
  int nzmb_thisdvce;       // number of Zoom MeshBlocks on this device (local)
  int nzmb_thishost;       // number of Zoom MeshBlocks on this host (local)
  int nzmb_max_perdvce;    // max allowed number of Zoom MBs per device (memory limit for AMR)
  int nzmb_max_perhost;    // max allowed number of Zoom MBs per host (memory limit for AMR)
  // following 2x arrays allocated with length [nranks] in BuildTreeFromXXXX()
  // TODO(@mhguo): do you really need so many lists here?
  int *gids_eachlevel;     // starting global ID of Zoom MeshBlocks in each level
  int *nzmb_eachlevel;     // number of Zoom MeshBlocks on each level
  int *gids_eachdvce;      // starting global ID of MeshBlocks in each device
  int *nzmb_eachdvce;      // number of MeshBlocks on each device
  std::vector<int> rank_eachmb;        // rank of each MeshBlock that contains this zoom MeshBlock
  std::vector<int> lid_eachmb;         // local ID of each MeshBlock that contains this zoom MeshBlock
  std::vector<int> rank_eachzmb;        // rank of each Zoom MeshBlock
  std::vector<int> lid_eachzmb;         // local ID of each Zoom MeshBlock
  std::vector<LogicalLocation> lloc_eachzmb;  // LogicalLocations for each MeshBlock

  // functions
  // TODO(@mhguo): think whether there is a better design
  void GatherZMB(int zm_count, int zone);
  void UpdateMeshData();
  void SyncMBLists();
  void SyncLogicalLocations();
  int FindMB(int gzm);
  void FindRegion(int zone);

 private:
  CyclicZoom *pzoom;       // ptr to CyclicZoom containing this ZoomMesh module
};

//----------------------------------------------------------------------------------------
//! \class ZoomData
//! \brief Handles storage of data during cyclic zoom AMR
class ZoomData
{
  friend class CyclicZoom;
 public:
  ZoomData(CyclicZoom *pz, ParameterInput *pin);
  ~ZoomData() = default;
  // data
  int nvars;               // number of variables
  int zmb_data_cnt;        // count of data elements per Zoom MeshBlock needed for zooming
  Real d_zoom;             // density within inner boundary
  Real p_zoom;             // pressure within inner boundary

  DvceArray5D<Real> u0;    // conserved variables
  DvceArray5D<Real> w0;    // primitive variables
  DvceArray5D<Real> coarse_u0;  // coarse conserved variables
  DvceArray5D<Real> coarse_w0;  // coarse primitive variables
  // DvceArray5D<Real> coarse_wuh; // coarse primitive variables from hydro conserved variables

  DvceEdgeFld4D<Real> efld_pre;   // coarse edge-centered electric fields before zoom
  DvceEdgeFld4D<Real> efld_aft;   // coarse edge-centered electric fields after zoom
  // DvceEdgeFld4D<Real> efld;   // edge-centered electric fields (fluxes of B)
  // DvceEdgeFld4D<Real> emf0;   // edge-centered electric fields just after zoom
  DvceEdgeFld4D<Real> delta_efld; // change in electric fields
  DvceEdgeFld4D<Real> efld_buf;   // buffer for electric fields during zoom

  // DualView for device ↔ host mirrored packing buffer (replaces dzbuf+hzbuf)
  // Syncs only used portion via subviews for bandwidth efficiency
  DualArray1D<Real> zbuf;
  
  // Separate host buffer for MPI receive - NOT a mirror of zbuf!
  // Contains different ZMBs after redistribution due to load balancing
  HostArray1D<Real> zdata;   // host array for persistent storage with load balancing

  HostArray2D<Real> max_emf0;  // maximum electric field

  HostArray5D<Real> harr_5d;  // host copy of 5D arrays
  HostArray4D<Real> harr_4d;  // host copy of 4D arrays

#if MPI_PARALLEL_ENABLED
  int ndata;               // size of send/recv data
  MPI_Comm zoom_comm;                       // unique communicator for zoom refinement
  // DualArray1D<AMRBuffer> sendbuf, recvbuf; // send/recv buffers
  MPI_Request *send_req, *recv_req;
#endif

  // functions
  void Initialize();
  void ResetDataEC(DvceEdgeFld4D<Real> ec);
  void DumpData();
  void StoreDataToZoomData(int zm, int m);
  // TODO(@mhguo): u or u0_?
  void StoreCCData(int zm, int m, DvceArray5D<Real> u, DvceArray5D<Real> w);
  void StoreHydroData(int zm, int m, DvceArray5D<Real> u0_, DvceArray5D<Real> w0_);
  // TODO(@mhguo): find a better name
  void StoreEFieldsBeforeAMR(int zm, int m, DvceEdgeFld4D<Real> efld);
  void StoreFinerEFields(int zmc, int zm, DvceEdgeFld4D<Real> efld);
  void StoreEFieldsAfterAMR(int zm, int m, DvceEdgeFld4D<Real> efld);
  void LimitEFields();
  void MeshBlockDataSize();
  void PackBuffer();
  void PackBuffersCC(DvceArray1D<Real> packed_data, DvceArray5D<Real> a0,
                     size_t offset_a0, const int m);
  void PackBuffersFC(DvceArray1D<Real> packed_data, DvceFaceFld4D<Real> fc,
                     size_t offset_fc, const int m);
  void PackBuffersEC(DvceArray1D<Real> packed_data, DvceEdgeFld4D<Real> ec,
                     size_t offset_ec, const int m);
  void UnpackBuffer();
  void UnpackBuffersCC(DvceArray1D<Real> packed_data, DvceArray5D<Real> a0,
                       size_t offset_a0, const int m);
  void UnpackBuffersFC(DvceArray1D<Real> packed_data, DvceFaceFld4D<Real> fc,
                       size_t offset_fc, const int m);
  void UnpackBuffersEC(DvceArray1D<Real> packed_data, DvceEdgeFld4D<Real> ec,
                       size_t offset_ec, const int m);
  void RedistributeZMBs(int zone,
                        HostArray1D<Real> src_buf, std::vector<int>& src_ranks, std::vector<int>* src_lids,
                        HostArray1D<Real> dst_buf, std::vector<int>& dst_ranks, std::vector<int>* dst_lids);
  void SaveToStorage(int zone);
  void LoadFromStorage(int zone);
  void ExtractEField(DvceEdgeFld4D<Real> ec);
  void LoadDataFromZoomData(int m, int zm);
  void LoadCCData(int m, int zm);
  void LoadHydroData(int m, int zm);
  void MaskDataInZoomRegion(int m, int zm);
  void AddSrcTermsFC(int m, int zm, DvceEdgeFld4D<Real> emf);

 private:
  CyclicZoom *pzoom;       // ptr to CyclicZoom containing this ZoomData module
  ZoomMesh   *pzmesh;      // ptr to ZoomMesh containing this ZoomData module
};

#endif // CYCLIC_ZOOM_CYCLIC_ZOOM_HPP_
