//==================================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//==================================================================================================
//! \file parameter_input.cpp
//  \brief implementation of functions in class ParameterInput
//
// PURPOSE: Member functions of this class are used to read and parse the input file.
//   Functionality is loosely modeled after FORTRAN namelist.
//
// EXAMPLE of input file in 'Athena++' format:
//   <blockname1>      # block name; must be on a line by itself
//                     # everything after a hash symbol is a comment and is ignored
//   name1=value       # each parameter name must be on a line by itself
//   name2 = value1    # whitespace around the = is optional
//                     # blank lines are OK
//   # my comment here   comment lines are OK
//   # name3 = value3    values (and blocks) that are commented out are ignored
//
//   <blockname2>      # start new block
//   name1 = value1    # note that same parameter names can appear in different blocks
//   name2 = value2    # empty lines (like following) are OK
//
//   <blockname1>      # same blockname can re-appear, although NOT recommended
//   name3 = value3    # this would be the 3rd parameter name in blockname1
//   name1 = value4    # if parameter name is repeated, previous value is overwritten!
//
// LIMITATIONS:
//   - parameter specification (name=val #comment) must all be on a single line
//
// HISTORY:
//   - Nov 2002:  Created for Athena1.0/Cambridge release by Peter Teuben
//   - 2003-2008: Many improvements and extensions by T. Gardiner and J.M. Stone
//   - Jan 2014:  Rewritten in C++ for the Athena++ code by J.M. Stone
//==================================================================================================

#include "parameter_input.hpp"

#include <algorithm>  // transform
#include <cmath>      // std::fmod()
#include <cstdlib>    // atoi(), atof(), nullptr, std::size_t
#include <fstream>    // ifstream
#include <iostream>   // endl, ostream
#include <sstream>    // stringstream
#include <stdexcept>  // runtime_error
#include <string>     // string

#if OPENMP_PARALLEL_ENABLED
#include <omp.h>
#endif

//--------------------------------------------------------------------------------------------------
// ParameterInput constructor

ParameterInput::ParameterInput() : last_filename_{} {
#if OPENMP_PARALLEL_ENABLED
  omp_init_lock(&lock_);
#endif
}

// ParameterInput destructor

ParameterInput::~ParameterInput() {
#if OPENMP_PARALLEL_ENABLED
  omp_destroy_lock(&lock_);
#endif
}

//--------------------------------------------------------------------------------------------------
//! \fn InputLine* InputBlock::GetPtrToLine(std::string name)
//  \brief return pointer to InputLine containing specified parameter if it exists

InputLine* InputBlock::GetPtrToLine(std::string name) {
  for (auto it = line.begin(); it != line.end(); ++it) {
    if (name.compare(it->param_name) == 0) return &*it;
  }
  return nullptr;
}

//--------------------------------------------------------------------------------------------------
//! \fn InputBlock* ParameterInput::GetPtrToBlock(std::string name)
//  \brief return pointer to specified InputBlock if it exists

InputBlock* ParameterInput::GetPtrToBlock(std::string name) {
  for (auto it = block.begin(); it != block.end(); ++it) {
    if (name.compare(it->block_name) == 0) return &*it;
  }
  return nullptr;
}

//--------------------------------------------------------------------------------------------------
//! \fn  void ParameterInput::LoadFromStream(std::istream &is)
//  \brief Load input parameters from a stream

//  Block names are allocated and stored in a singly linked list of InputBlocks. Within each
//  InputBlock the names, values, and comments of each parameter are allocated and stored in a
//  singly linked list of InputLines.

void ParameterInput::LoadFromStream(std::istream &is) {
  std::string line, block_name, param_name, param_value, param_comment;
  std::size_t first_char, last_char;
  std::stringstream msg;
  InputBlock *pib{};
  int line_num{-1}, blocks_found{0};

  while (is.good()) {
    std::getline(is, line);
    line_num++;
    if (line.find('\t') != std::string::npos) {
      line.erase(std::remove(line.begin(), line.end(), '\t'), line.end());
    }
    if (line.empty()) continue;                             // skip blank line
    first_char = line.find_first_not_of(" ");               // skip white space
    if (first_char == std::string::npos) continue;          // line is all white space
    if (line.compare(first_char, 1, "#") == 0) continue;      // skip comments
    if (line.compare(first_char, 9, "<par_end>") == 0) break; // stop on <par_end>

    if (line.compare(first_char, 1, "<") == 0) {              // a new block
      first_char++;
      last_char = (line.find_first_of(">", first_char));
      block_name.assign(line, first_char, last_char-1);       // extract block name

      if (last_char == std::string::npos) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                  << "Block name '" << block_name << "' not properly ended" << std::endl;
        std::exit(EXIT_FAILURE);
      }

      pib = FindOrAddBlock(block_name);  // find or add block to singly linked list

      if (pib == nullptr) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                  << std::endl << "Block name '" << block_name << "' could not be found/added";
        std::exit(EXIT_FAILURE);
      }
      blocks_found++;
      continue;  // skip to next line if block name was found
    } // end "a new block was found"

    // if line does not contain a block name or skippable information (comments,
    // whitespace), it must contain a parameter value
    if (blocks_found == 0) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                  << "Input must specify a block name before first param = value line" << std::endl;
        std::exit(EXIT_FAILURE);
    }
    // parse line and add name/value/comment strings (if found) to current block name
    ParseLine(pib, line, param_name, param_value, param_comment);
    AddParameter(pib, param_name, param_value, param_comment);
  }
  return;
}

//--------------------------------------------------------------------------------------------------
//! \fn  void ParameterInput::LoadFromFile(IOWrapper &input)
//  \brief Read the parameters from an input or restart file.

void ParameterInput::LoadFromFile(IOWrapper &input) {
  std::stringstream par, msg;
  constexpr int kBufSize = 4096;
  char buf[kBufSize];
  IOWrapperSizeT header = 0, ret, loc;

  // search for <par_end> (reading from restart files) or EOF (reading from input file).
  do { 
    if (global_variable::my_rank == 0) // only the master process reads the header from the file
      ret = input.Read(buf, sizeof(char), kBufSize);
#if MPI_PARALLEL_ENABLED
    // then broadcasts it
    MPI_Bcast(&ret, sizeof(IOWrapperSizeT), MPI_BYTE, 0, MPI_COMM_WORLD);
    MPI_Bcast(buf, ret, MPI_BYTE, 0, MPI_COMM_WORLD);
#endif
    par.write(buf, ret); // add the buffer into the stream
    header += ret;
    std::string sbuf = par.str(); // create string for search
    loc = sbuf.find("<par_end>", 0); // search from the top of the stream
    if (loc != std::string::npos) { // found <par_end>
      header = loc + 10; // store the header length
      break;
    }
    if (header > kBufSize*10) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                << "<par_end> is not found in the first 40KBytes." << std::endl
                << "Probably the file is broken or the wrong file is specified" << std::endl;
      std::exit(EXIT_FAILURE);
    }
  } while (ret == kBufSize); // till EOF (or par_end is found)

  // Now par contains the parameter inputs + some additional including <par_end>
  // Read the stream and load the parameters
  LoadFromStream(par);
  // Seek the file to the end of the header
  input.Seek(header);

  return;
}

//--------------------------------------------------------------------------------------------------
//! \fn InputBlock* ParameterInput::FindOrAddBlock(std::string name)
//  \brief find or add specified InputBlock.  Returns pointer to block.

InputBlock* ParameterInput::FindOrAddBlock(std::string name) {

  // Search singly linked list of InputBlocks to see if name exists, return if found.
  for (auto it = block.begin(); it != block.end(); ++it) {
    if (name.compare(it->block_name) == 0) return &*it;
  }

  // Create new block at end of list if not found above, and return pointer to it
  block.emplace_after(block.end(),name);
//  return &*block.begin();
  auto it = block.end();
  return &*it;
}

//--------------------------------------------------------------------------------------------------
//! \fn void ParameterInput::ParseLine(InputBlock *pib, std::string line,
//           std::string& name, std::string& value, std::string& comment)
//  \brief parse "name = value # comment" format, return name/value/comment strings.

void ParameterInput::ParseLine(InputBlock *pib, std::string line, std::string& name,
                               std::string& value, std::string& comment) {
  std::size_t first_char, last_char, equal_char, hash_char, len;

  first_char = line.find_first_not_of(" ");   // find first non-white space
  equal_char = line.find_first_of("=");       // find "=" char
  hash_char  = line.find_first_of("#");       // find "#" (optional)

  // copy substring into name, remove white space at end of name
  len = equal_char - first_char;
  name.assign(line, first_char, len);

  last_char = name.find_last_not_of(" ");
  name.erase(last_char+1, std::string::npos);

  // copy substring into value, remove white space at start and end
  len = hash_char - equal_char - 1;
  value.assign(line, equal_char+1, len);

  first_char = value.find_first_not_of(" ");
  value.erase(0, first_char);

  last_char = value.find_last_not_of(" ");
  value.erase(last_char+1, std::string::npos);

  // copy substring into comment, if present
  if (hash_char != std::string::npos) {
    comment = line.substr(hash_char);
  } else {
    comment = "";
  }
}

//--------------------------------------------------------------------------------------------------
//! \fn void ParameterInput::AddParameter(InputBlock *pb, std::string name,
//   std::string value, std::string comment)
//  \brief add name/value/comment tuple to the InputLine singly linked list in block *pb.
//  If a parameter with the same name already exists, the value and comment strings
//  are replaced (overwritten).

void ParameterInput::AddParameter(InputBlock *pb, std::string name, std::string value,
                                  std::string comment) {

  // Search singly linked list of InputBlocks to see if name exists, replace contents with new
  // values if found and return.
  for (auto it = pb->line.begin(); it != pb->line.end(); ++it) {
    if (name.compare(it->param_name) == 0) {   // param name already exists
      it->param_value.assign(value);           // replace existing param value
      it->param_comment.assign(comment);       // replace exisiting param comment
      if (value.length() > pb->max_len_parvalue) pb->max_len_parvalue = value.length();
      return;
    }
  }

  // Parameter not found, so create new node in singly linked list
//  InputLine new_line;
//  new_line.param_name.assign(name);
//  new_line.param_value.assign(value);
//  new_line.param_comment.assign(comment);
  pb->line.emplace_after(pb->line.end(),name,value,comment);
  if (name.length() > pb->max_len_parname) pb->max_len_parname = name.length();
  if (value.length() > pb->max_len_parvalue) pb->max_len_parvalue = value.length();
  
  return;
}

//--------------------------------------------------------------------------------------------------
//! \fn int ParameterInput::DoesParameterExist(std::string block, std::string name)
//  \brief check whether parameter of given name in given block exists

int ParameterInput::DoesParameterExist(std::string block, std::string name) {
  InputLine *pl;
  InputBlock *pb;
  pb = GetPtrToBlock(block);
  if (pb == nullptr) return 0;
  pl = pb->GetPtrToLine(name);
  return (pl == nullptr ? 0 : 1);
}

//--------------------------------------------------------------------------------------------------
//! \fn void ParameterInput::ParameterDump(std::ostream& os)
//  \brief output entire InputBlock/InputLine hierarchy to specified stream

void ParameterInput::ParameterDump(std::ostream& os) {
  std::string param_name,param_value;
  std::size_t len;

  os<< "#------------------------- PAR_DUMP -------------------------" << std::endl;

  for (auto itb = block.begin(); itb != block.end(); ++itb) {            // loop over InputBlocks
    os<< "<" << itb->block_name << ">" << std::endl;                     // write block name
    for (auto itl = itb->line.begin(); itl != itb->line.end(); ++itl) {  // loop over InputLines
      param_name.assign(itl->param_name);
      param_value.assign(itl->param_value);

      len = itb->max_len_parname - param_name.length() + 1;
      param_name.append(len,' ');                             // pad name to align vertically
      len = itb->max_len_parvalue - param_value.length() + 1;
      param_value.append(len,' ');                            // pad value to align vertically

      os<< param_name << "= " << param_value << itl->param_comment <<  std::endl;
    }
  }

  os<< "#------------------------- PAR_DUMP -------------------------" << std::endl;
  os<< "<par_end>" << std::endl;    // finish with par-end (needed for restart files)
}

//--------------------------------------------------------------------------------------------------
//! \fn void ParameterInput::Lock()
//  \brief Lock ParameterInput for reading and writing

void ParameterInput::Lock() {
#if OPENMP_PARALLEL_ENABLED
  omp_set_lock(&lock_);
#endif
  return;
}

//--------------------------------------------------------------------------------------------------
//! \fn void ParameterInput::Unlock()
//  \brief Unlock ParameterInput for reading and writing

void ParameterInput::Unlock() {
#if OPENMP_PARALLEL_ENABLED
  omp_unset_lock(&lock_);
#endif
  return;
}
