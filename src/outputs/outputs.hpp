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

// identifiers for output variables
enum class OutputVariable {undef=-1,
  hydro_u_d, hydro_u_m1, hydro_u_m2, hydro_u_m3, hydro_u_e, hydro_u,
  hydro_w_d, hydro_w_vx, hydro_w_vy, hydro_w_vz, hydro_w_p, hydro_w,
  mhd_u_d,   mhd_u_m1,   mhd_u_m2,   mhd_u_m3,   mhd_u_e,   mhd_u,
  mhd_w_d,   mhd_w_vx,   mhd_w_vy,   mhd_w_vz,   mhd_w_p,   mhd_w,
  mhd_bcc1,  mhd_bcc2,   mhd_bcc3,   mhd_bcc,    mhd_b_x1f, mhd_b_x2f, mhd_b_x3f,
  turb_force};

// forward declarations, and two utility function prototypes
class Mesh;
class ParameterInput;
OutputVariable GetOutputVariable(const std::string& input_string);
std::string GetOutputVariableString(OutputVariable input_flag);

//----------------------------------------------------------------------------------------
//! \struct OutputParameters
//  \brief  container for parameters read from <output> block in the input file

struct OutputParameters
{
  int block_number;
  std::string block_name;
  std::string file_basename;
  std::string file_id;
  std::string file_type;
  OutputVariable variable;
  std::string data_format;
  Real last_time, dt;
  int file_number;
  bool include_gzs;
  bool slice1, slice2, slice3;
  Real slice_x1, slice_x2, slice_x3;
};

//----------------------------------------------------------------------------------------
//! \struct OutputVariableInfo
//  \brief  container for various properties of each output variable

struct OutputVariableInfo
{
  std::string label;             // "name" of variable
  int data_index;                // index of variable in device array
  DvceArray5D<Real> *data_ptr;   // ptr to device array containing variable
  // constructor
  OutputVariableInfo(std::string lab, int indx, DvceArray5D<Real> *ptr) :
    label(lab), data_index(indx), data_ptr(ptr) {}
};

//----------------------------------------------------------------------------------------
//! \struct OutputMeshBlockInfo
//  \brief  container for various properties of each output MeshBlock

struct OutputMeshBlockInfo
{ 
  int mb_gid;                        // gid of output MB
  int ois, oie, ojs, oje, oks, oke;  // start/end indices of data to be output on MB
  // constructor
  OutputMeshBlockInfo(int id, int is, int ie, int js, int je, int ks, int ke) :
    mb_gid(id), ois(is), oie(ie), ojs(js), oje(je), oks(ks), oke(ke) {}
};

//----------------------------------------------------------------------------------------
//! \struct HistoryData
//  \brief  container for history data for different physics modules

struct HistoryData
{
  int nhist;
  PhysicsModule physics;
  std::string label[NHISTORY_VARIABLES];
  Real hdata[NHISTORY_VARIABLES];
  bool header_written;
  // constructor
  HistoryData(PhysicsModule name) : physics(name), header_written(false) {}
};

//----------------------------------------------------------------------------------------
// \brief abstract base class for different output types (modes/formats); node in
//        std::list of OutputType created & stored in the Outputs class

class OutputType
{
 public:
  OutputType(OutputParameters oparams, Mesh *pm);
  virtual ~OutputType() = default;
  // copy constructor and assignment operator
  OutputType(const OutputType& copy_other) = default;
  OutputType& operator=(const OutputType& copy_other) = default;
  // move constructor and assignment operator
  OutputType(OutputType&&) = default;
  OutputType& operator=(OutputType&&) = default;

  // data
  OutputParameters out_params;    // data read from <output> block for this type

  // virtual functions may be over-ridden in derived classes
  virtual void LoadOutputData(Mesh *pm);
  virtual void WriteOutputFile(Mesh *pm, ParameterInput *pin) = 0;
  void ErrHydroOutput(std::string block);
  void ErrMHDOutput(std::string block);
  void ErrForceOutput(std::string block);

 protected:
  HostArray5D<Real> outdata;       // container for data on host with dims (n,m,k,j,i)
  // Following vector will be of length (# output MeshBlocks)
  // With slicing, this may not be same as # of MeshBlocks in calculation
  std::vector<OutputMeshBlockInfo> outmbs; 
  // Following vector will be of length (# output variables)
  std::vector<OutputVariableInfo> outvars;
};

//----------------------------------------------------------------------------------------
//! \class FormattedTableOutput
//  \brief derived OutputType class for formatted table (tabular) data

class FormattedTableOutput : public OutputType
{
 public:
  FormattedTableOutput(OutputParameters oparams, Mesh *pm);
  void WriteOutputFile(Mesh *pm, ParameterInput *pin) override;
};

//----------------------------------------------------------------------------------------
//! \class HistoryOutput
//  \brief derived OutputType class for history data
    
class HistoryOutput : public OutputType
{   
 public:
  HistoryOutput(OutputParameters oparams, Mesh *pm);

  // vector of length [# of physics modules] containing hdata arrays
  std::vector<HistoryData> hist_data;

  void LoadOutputData(Mesh *pm) override;
  void LoadHydroHistoryData(HistoryData *pdata, Mesh *pm);
  void LoadMHDHistoryData(HistoryData *pdata, Mesh *pm);
  void WriteOutputFile(Mesh *pm, ParameterInput *pin) override;
};  

//----------------------------------------------------------------------------------------
//! \class VTKOutput
//  \brief derived OutputType class for vtk binary data (VTK legacy format)

class VTKOutput : public OutputType
{
 public:
  VTKOutput(OutputParameters oparams, Mesh *pm);

  void WriteOutputFile(Mesh *pm, ParameterInput *pin) override;
};

//----------------------------------------------------------------------------------------
//! \class Outputs
//  \brief root class for all Athena++ outputs. Provides a std::vector of OutputTypes,
//   with each element representing one mode/format of output to be made.

class Outputs
{
 public:
  Outputs(ParameterInput *pin, Mesh *pm);
  ~Outputs();

  // use vector of pointers to OutputTypes since it is an abstract base class 
  std::vector<OutputType*> pout_list;  
};

#endif // OUTPUTS_OUTPUTS_HPP_
