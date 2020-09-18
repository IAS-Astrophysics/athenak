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

// forward declarations
class Mesh;
class ParameterInput;

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
  std::string variable;
  std::string data_format;
  Real last_time, dt;
  int file_number;
  bool include_gzs;
  bool slice1, slice2, slice3;
  Real slice_x1, slice_x2, slice_x3;
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
  int nout1, nout2, nout3;        // dimensions of output arrays for this type
  int ois, ojs, oks;              // starting indices of data to be output
  OutputParameters out_params;    // data read from <output> block for this type

  HostArray1D<bool> hydro_cons_out_vars;
  HostArray1D<bool> hydro_prim_out_vars;

  // virtual functions over-ridden in derived classes
  virtual void LoadOutputData(Mesh *pm);
  virtual void WriteOutputFile(Mesh *pm, ParameterInput *pin) = 0;

 protected:
  // output data stored in vector of HostArrays which will be of length (# output MBs) 
  // With slicing this may be different than nmbthisrank since not all MBs are output
  // This vector is stored in another vector which will be of length (# of output vars)
  std::vector< std::vector<HostArray3D<Real>> > out_data_;
  // coordinates of out_data_ stored in vectors which will be of length (# output MBs)
  std::vector<HostArray1D<Real>> x1_cc_, x1_fc_;
  std::vector<HostArray1D<Real>> x2_cc_, x2_fc_;
  std::vector<HostArray1D<Real>> x3_cc_, x3_fc_;
  std::vector<int> out_gids_;  // GridIDs of output MBs
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
  void LoadOutputData(Mesh *pm) override;
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
  std::vector<OutputType*> pout_list_;  

 private:

};

#endif // OUTPUTS_OUTPUTS_HPP_
