#ifndef OUTPUTS_OUTPUTS_HPP_
#define OUTPUTS_OUTPUTS_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file outputs.hpp
//  \brief provides classes to handle ALL types of data output

#include <string>
#include <vector>

#include "athena.hpp"
#include "io_wrapper.hpp"

#define NHISTORY_VARIABLES 12
#if NHISTORY_VARIABLES > NREDUCTION_VARIABLES
    #error NHISTORY > NREDUCTION in outputs.hpp
#endif

#define NOUTPUT_CHOICES 139
// choices for output variables used in <ouput> blocks in input file
// TO ADD MORE CHOICES:
//   - add more strings to array below, change NOUTPUT_CHOICES above appropriately
//   - add code to load new variables in BaseOutputType constructor
//   - may need to change index limits that test whether physics is defined for
//     requested output variable near start of BaseOutputType constructor
static const char *var_choice[NOUTPUT_CHOICES] = {
  "hydro_u_d", "hydro_u_m1", "hydro_u_m2", "hydro_u_m3", "hydro_u_e", "hydro_u",
  "hydro_w_d", "hydro_w_vx", "hydro_w_vy", "hydro_w_vz", "hydro_w_e", "hydro_w",
  "hydro_u_s", "hydro_w_s",  "hydro_wz",   "hydro_w2",
  "mhd_u_d",   "mhd_u_m1",   "mhd_u_m2",   "mhd_u_m3",   "mhd_u_e",   "mhd_u",
  "mhd_w_d",   "mhd_w_vx",   "mhd_w_vy",   "mhd_w_vz",   "mhd_w_e",   "mhd_w",
  "mhd_u_s",   "mhd_w_s",    "mhd_wz",     "mhd_w2",
  "mhd_bcc1",  "mhd_bcc2",   "mhd_bcc3",   "mhd_bcc",    "mhd_u_bcc", "mhd_w_bcc",
  "mhd_jz",    "mhd_j2",     "mhd_divb",
  "turb_force",
  "rad_coord",     "rad_fluid",      "rad_coord_fluid",
  "rad_hydro_u_d", "rad_hydro_u_m1", "rad_hydro_u_m2", "rad_hydro_u_m3", "rad_hydro_u_e",
  "rad_hydro_u",   "rad_hydro_w_d",  "rad_hydro_w_vx", "rad_hydro_w_vy", "rad_hydro_w_vz",
  "rad_hydro_w_e", "rad_hydro_w",    "rad_hydro_u_s",  "rad_hydro_w_s",
  "rad_mhd_u_d",   "rad_mhd_u_m1",   "rad_mhd_u_m2",   "rad_mhd_u_m3",   "rad_mhd_u_e",
  "rad_mhd_u",     "rad_mhd_w_d",    "rad_mhd_w_vx",   "rad_mhd_w_vy",   "rad_mhd_w_vz",
  "rad_mhd_w_e",   "rad_mhd_w",      "rad_mhd_u_s",    "rad_mhd_w_s",    "rad_mhd_bcc1",
  "rad_mhd_bcc2",  "rad_mhd_bcc3",   "rad_mhd_bcc",    "rad_mhd_u_bcc",  "rad_mhd_w_bcc",


  "adm_gxx", "adm_gxy", "adm_gxz", "adm_gyy", "adm_gyz", "adm_gzz",
  "adm_Kxx", "adm_Kxy", "adm_Kxz", "adm_Kyy", "adm_Kyz", "adm_Kzz",
  "adm_psi4",
  "adm_alpha", "adm_betax", "adm_betay", "adm_betaz",
  "adm",

  "z4c_chi",
  "z4c_gxx", "z4c_gxy", "z4c_gxz", "z4c_gyy", "z4c_gyz", "z4c_gzz",
  "z4c_Khat",
  "z4c_Axx", "z4c_Axy", "z4c_Axz", "z4c_Ayy", "z4c_Ayz", "z4c_Azz",
  "z4c_Gamx", "z4c_Gamy", "z4c_Gamz",
  "z4c_Theta",
  "z4c_alpha",
  "z4c_betax", "z4c_betay", "z4c_betaz",
  "z4c",

  "con_C",
  "con_H",
  "con_M",
  "con_Z",
  "con_Mx", "con_My", "con_Mz",
  "con",

  "mat_rho",
  "mat_Sx", "mat_Sy", "mat_Sz",
  "mat_Sxx", "mat_Sxy", "mat_Sxz", "mat_Syy", "mat_Syz", "mat_Szz",
  "mat"
};

// forward declarations
class Mesh;
class ParameterInput;

//----------------------------------------------------------------------------------------
//! \struct OutputParameters
//  \brief  container for parameters read from <output> block in the input file

struct OutputParameters {
  int block_number;
  std::string block_name;
  std::string file_basename;
  std::string file_id;
  std::string file_type;
  std::string variable;
  std::string data_format;
  Real last_time, dt;
  int dcycle;  // enables outputs every 'dcycle'
  int file_number;
  int gid;
  bool include_gzs;
  bool slice1, slice2, slice3;
  Real slice_x1, slice_x2, slice_x3;
  bool user_hist_only;
  bool contains_derived=false;
};

//----------------------------------------------------------------------------------------
//! \struct OutputVariableInfo
//  \brief  container for various properties of each output variable

struct OutputVariableInfo {
  std::string label;             // "name" of variable
  int data_index;                // index of variable in device array
  DvceArray5D<Real> *data_ptr;   // ptr to device array containing variable
  // constructor(s)
  OutputVariableInfo(std::string lab, int indx, DvceArray5D<Real> *ptr) :
    label(lab), data_index(indx), data_ptr(ptr) {}
};

//----------------------------------------------------------------------------------------
//! \struct OutputMeshBlockInfo
//  \brief  container for various properties of each output MeshBlock

struct OutputMeshBlockInfo {
  int mb_gid;                        // gid of output MB
  int ois, oie, ojs, oje, oks, oke;  // start/end indices of data to be output on MB
  Real x1min, x1max;  // physical X1 size of output MB
  Real x2min, x2max;  // physical X2 size of output MB
  Real x3min, x3max;  // physical X3 size of output MB
  // constructor
  OutputMeshBlockInfo(int id, int is, int ie, int js, int je, int ks, int ke,
    Real x1min, Real x1max, Real x2min, Real x2max, Real x3min, Real x3max) :
    mb_gid(id), ois(is), oie(ie), ojs(js), oje(je), oks(ks), oke(ke),
    x1min(x1min), x1max(x1max), x2min(x2min), x2max(x2max), x3min(x3min), x3max(x3max) {}
};

//----------------------------------------------------------------------------------------
//! \struct HistoryData
//  \brief  container for history data for different physics modules

struct HistoryData {
  int nhist;
  PhysicsModule physics;
  std::string label[NHISTORY_VARIABLES];
  Real hdata[NHISTORY_VARIABLES];
  bool header_written;
  // constructor
  explicit HistoryData(PhysicsModule name) : physics(name), header_written(false) {}
};

//----------------------------------------------------------------------------------------
// \brief abstract base class for different output types (modes/formats); node in
//        std::list of BaseTypeOutput created & stored in the Outputs class

class BaseTypeOutput {
 public:
  BaseTypeOutput(OutputParameters oparams, Mesh *pm);
  virtual ~BaseTypeOutput() = default;
  // copy constructor and assignment operator
  BaseTypeOutput(const BaseTypeOutput& copy_other) = default;
  BaseTypeOutput& operator=(const BaseTypeOutput& copy_other) = default;
  // move constructor and assignment operator
  BaseTypeOutput(BaseTypeOutput&&) = default;
  BaseTypeOutput& operator=(BaseTypeOutput&&) = default;

  // data
  OutputParameters out_params;   // params read from <output> block for this type
  DvceArray5D<Real> derived_var; // array to store output variables computed from u0/b0

  // function which computes derived output variables like vorticity and current density
  void ComputeDerivedVariable(std::string name, Mesh *pm);

  // virtual functions may be over-ridden in derived classes
  virtual void LoadOutputData(Mesh *pm);
  virtual void WriteOutputFile(Mesh *pm, ParameterInput *pin) = 0;

 protected:
  // CC output data on host with dims (n,m,k,j,i) except
  // for restarts, where dims are (m,n,k,j,i)
  HostArray5D<Real> outarray;
  HostArray5D<Real> outarray_hyd, outarray_mhd, outarray_rad,
                    outarray_force, outarray_z4c, outarray_adm;
  HostFaceFld4D<Real> outfield;  // FC output field on host
  std::vector<int> noutmbs;   // with MPI, number of output MBs across all ranks
  int noutmbs_min;            // with MPI, minimum number of output MBs across all ranks
  int noutmbs_max;            // with MPI, maximum number of output MBs across all ranks

  // Following vector will be of length (# output MeshBlocks)
  // With slicing, this may not be same as # of MeshBlocks in calculation
  std::vector<OutputMeshBlockInfo> outmbs;

  // Following vector will be of length (# output variables)
  std::vector<OutputVariableInfo> outvars;
};

//----------------------------------------------------------------------------------------
//! \class FormattedTableOutput
//  \brief derived BaseTypeOutput class for formatted table (tabular) data

class FormattedTableOutput : public BaseTypeOutput {
 public:
  FormattedTableOutput(OutputParameters oparams, Mesh *pm);
  void WriteOutputFile(Mesh *pm, ParameterInput *pin) override;
};

//----------------------------------------------------------------------------------------
//! \class HistoryOutput
//  \brief derived BaseTypeOutput class for history data

class HistoryOutput : public BaseTypeOutput {
 public:
  HistoryOutput(OutputParameters oparams, Mesh *pm);

  // vector of length [# of physics modules] containing hdata arrays
  std::vector<HistoryData> hist_data;

  void LoadOutputData(Mesh *pm) override;
  void LoadHydroHistoryData(HistoryData *pdata, Mesh *pm);
  void LoadMHDHistoryData(HistoryData *pdata, Mesh *pm);
  void LoadZ4cHistoryData(HistoryData *pdata, Mesh *pm);
  void WriteOutputFile(Mesh *pm, ParameterInput *pin) override;
};

//----------------------------------------------------------------------------------------
//! \class VTKOutput
//  \brief derived BaseTypeOutput class for vtk binary data (VTK legacy format)

class VTKOutput : public BaseTypeOutput {
 public:
  VTKOutput(OutputParameters oparams, Mesh *pm);
  void WriteOutputFile(Mesh *pm, ParameterInput *pin) override;
};

//----------------------------------------------------------------------------------------
//! \class BinaryOutput
//  \brief derived BaseTypeOutput class for binary grid data (nbf format in pegasus++)

class BinaryOutput : public BaseTypeOutput {
 public:
  BinaryOutput(OutputParameters oparams, Mesh *pm);
  void WriteOutputFile(Mesh *pm, ParameterInput *pin) override;
};

//----------------------------------------------------------------------------------------
//! \class RestartOutput
//  \brief derived BaseTypeOutput class for restarts

class RestartOutput : public BaseTypeOutput {
 public:
  RestartOutput(OutputParameters oparams, Mesh *pm);
  void LoadOutputData(Mesh *pm) override;
  void WriteOutputFile(Mesh *pm, ParameterInput *pin) override;
};

//----------------------------------------------------------------------------------------
//! \class EventLogOutput
//  \brief derived BaseTypeOutput class for event counter data

class EventLogOutput : public BaseTypeOutput {
 public:
  EventLogOutput(OutputParameters oparams, Mesh *pm);

  // various flags to denote output status
  bool header_written=false;
  bool no_output=true;

  void LoadOutputData(Mesh *pm) override;
  void WriteOutputFile(Mesh *pm, ParameterInput *pin) override;
};

//----------------------------------------------------------------------------------------
//! \class Outputs
//  \brief root class for all Athena++ outputs. Provides a std::vector of BaseTypeOutputs,
//   with each element representing one mode/format of output to be made.

class Outputs {
 public:
  Outputs(ParameterInput *pin, Mesh *pm);
  ~Outputs();

  // use vector of pointers to BaseTypeOutputs since it is an abstract base class
  std::vector<BaseTypeOutput*> pout_list;
};

#endif // OUTPUTS_OUTPUTS_HPP_
