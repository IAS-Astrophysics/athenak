#ifndef PARAMETER_INPUT_HPP_
#define PARAMETER_INPUT_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file parameter_input.hpp
//  \brief definition of class ParameterInput
// Contains data structures used to store, and functions used to access, parameters
// read from the input file.  See comments at start of parameter_input.cpp for more
// information on the Athena++ input file format.

#include <cstddef>  // std::size_t
#include <ostream>  // ostream
#include <string>   // string
#include <cstdlib>
#include <list>

#include "athena.hpp"
#include "outputs/io_wrapper.hpp"

#if OPENMP_PARALLEL_ENABLED
#include <omp.h>
#endif

//----------------------------------------------------------------------------------------
//! \struct InputLine
//  \brief  node in a std::list of parameters contained within 1x input block

struct InputLine {
 public:
  InputLine(std::string name, std::string value, std::string comment)
    : param_name{name}, param_value{value}, param_comment{comment} {};
  ~InputLine() {}

  std::string param_name;
  std::string param_value;   // value of the parameter is stored as a string!
  std::string param_comment;
};

//----------------------------------------------------------------------------------------
//! \class InputBlock
//  \brief node in a std::list of all input blocks contained within input file

class InputBlock {
 public:
  explicit InputBlock(std::string name) : block_name{name} {};
  ~InputBlock() {}

  // data
  std::string block_name;
  std::size_t max_len_parname;  // length of longest param_name, for nice-looking output
  std::size_t max_len_parvalue; // length of longest param_value, to format outputs
  std::list<InputLine> line;    // singly linked list of input lines (in this block)

  // functions
  InputLine* GetPtrToLine(std::string name);
};

//----------------------------------------------------------------------------------------
//! \class ParameterInput
//  \brief data and definitions of functions used to store and access input parameters
//  Functions are implemented in parameter_input.cpp

class ParameterInput {
 public:
  // constructor(s)/destructor
  ParameterInput();
  explicit ParameterInput(std::string input_filename);
  ~ParameterInput();

  // data
  std::list<InputBlock> block;

  // functions
  void CheckBlockNames();
  void LoadFromStream(std::istream &is);
  void LoadFromFile(IOWrapper &input, bool single_file_per_rank=false);
  void ModifyFromCmdline(int argc, char *argv[]);
  void ParameterDump(std::ostream& os);
  bool DoesBlockExist(std::string name);
  bool DoesParameterExist(std::string block, std::string name);
  int  GetInteger(std::string block, std::string name);
  int  GetOrAddInteger(std::string block, std::string name, int value);
  int  SetInteger(std::string block, std::string name, int value);
  Real GetReal(std::string block, std::string name);
  Real GetOrAddReal(std::string block, std::string name, Real value);
  Real SetReal(std::string block, std::string name, Real value);
  bool GetBoolean(std::string block, std::string name);
  bool GetOrAddBoolean(std::string block, std::string name, bool value);
  bool SetBoolean(std::string block, std::string name, bool value);
  std::string GetString(std::string block, std::string name);
  std::string GetOrAddString(std::string block, std::string name, std::string value);
  std::string SetString(std::string block, std::string name, std::string value);

 private:
  std::string last_filename;  // last input file opened, to prevent duplicate reads

  InputBlock* FindOrAddBlock(std::string name);
  InputBlock* GetPtrToBlock(std::string name);
  void ParseLine(std::string line, std::string &name, std::string &val,
                 std::string &comment);
  void AddParameter(InputBlock *pib, std::string name, std::string val, std::string comm);

#if OPENMP_PARALLEL_ENABLED
  // lock to implement OpenMP thread safety
  omp_lock_t lock_;
#endif

  // functions to set/unset thread locks for reading and writing
  inline void Lock() {
#if OPENMP_PARALLEL_ENABLED
    omp_set_lock(&lock_);
#endif
    return;
  }
  inline void Unlock() {
#if OPENMP_PARALLEL_ENABLED
    omp_unset_lock(&lock_);
#endif
    return;
  }
};
#endif // PARAMETER_INPUT_HPP_
