//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
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
//========================================================================================

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"

#if OPENMP_PARALLEL_ENABLED
#include <omp.h>
#endif

//----------------------------------------------------------------------------------------
// ParameterInput constructor(s)

ParameterInput::ParameterInput() : last_filename{} {
#if OPENMP_PARALLEL_ENABLED
  omp_init_lock(&lock_);
#endif
}

// this constructor automatically loads data from input_filename in argument
ParameterInput::ParameterInput(std::string input_filename) : last_filename{} {
#ifdef OPENMP_PARALLEL
  omp_init_lock(&lock_);
#endif
  IOWrapper infile;
  infile.Open(input_filename.c_str(), IOWrapper::FileMode::read);
  LoadFromFile(infile);
  infile.Close();
}

//----------------------------------------------------------------------------------------
// ParameterInput destructor

ParameterInput::~ParameterInput() {
#if OPENMP_PARALLEL_ENABLED
  omp_destroy_lock(&lock_);
#endif
}

//----------------------------------------------------------------------------------------
//! \fn InputLine* InputBlock::GetPtrToLine(std::string name)
//  \brief return pointer to InputLine containing specified parameter if it exists

InputLine* InputBlock::GetPtrToLine(std::string name) {
  for (auto it = line.begin(); it != line.end(); ++it) {
    if (name.compare(it->param_name) == 0) return &*it;
  }
  return nullptr;
}

//----------------------------------------------------------------------------------------
//! \fn InputBlock* ParameterInput::GetPtrToBlock(std::string name)
//  \brief return pointer to specified InputBlock if it exists

InputBlock* ParameterInput::GetPtrToBlock(std::string name) {
  for (auto it = block.begin(); it != block.end(); ++it) {
    if (name.compare(it->block_name) == 0) return &*it;
  }
  return nullptr;
}

//----------------------------------------------------------------------------------------
//! \fn  void ParameterInput::LoadFromStream(std::istream &is)
//  \brief Load input parameters from a stream

// Block names are allocated and stored in a linked list of InputBlocks. Within each
// InputBlock the names, values, and comments of each parameter are allocated and stored
// in a linked list of InputLines.

void ParameterInput::LoadFromStream(std::istream &is) {
  std::string line, block_name, param_name, param_value, param_comment;
  std::size_t first_char, last_char;
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
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "Block name '" << block_name
                  << "' not properly ended" << std::endl;
        std::exit(EXIT_FAILURE);
      }

      pib = FindOrAddBlock(block_name);  // find or add block to linked list

      if (pib == nullptr) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "Block name '" << block_name
                  << "' could not be found/added" << std::endl;
        std::exit(EXIT_FAILURE);
      }
      blocks_found++;
      continue;  // skip to next line if block name was found
    } // end "a new block was found"

    // if line does not contain a block name or skippable information (comments,
    // whitespace), it must contain a parameter value
    if (blocks_found == 0) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "Input must specify a block name before first "
                  << "param = value line" << std::endl;
        std::exit(EXIT_FAILURE);
    }
    // parse line and add name/value/comment strings (if found) to current block name
    ParseLine(line, param_name, param_value, param_comment);
    AddParameter(pib, param_name, param_value, param_comment);
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn  void ParameterInput::LoadFromFile(IOWrapper &input)
//  \brief Read the parameters from an input or restart file.

void ParameterInput::LoadFromFile(IOWrapper &input, bool single_file_per_rank) {
  std::stringstream par;
  constexpr int kBufSize = 4096;
  char buf[kBufSize];
  IOWrapperSizeT header = 0, ret, loc;

  // search for <par_end> (reading from restart files) or EOF (reading from input file).
  do {
    if (global_variable::my_rank == 0 || single_file_per_rank) {
      ret = input.Read_bytes(buf, sizeof(char), kBufSize, single_file_per_rank);
    }
#if MPI_PARALLEL_ENABLED
    // then broadcasts it
  if (!single_file_per_rank) {
    MPI_Bcast(&ret, sizeof(IOWrapperSizeT), MPI_BYTE, 0, MPI_COMM_WORLD);
    if (ret == 0) {
      break;
    }
    MPI_Bcast(buf, ret, MPI_BYTE, 0, MPI_COMM_WORLD);
  }
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
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "<par_end> is not found in the first 40KBytes."
                << std::endl << "Probably the file is broken or the wrong file is "
                << "specified" << std::endl;
      std::exit(EXIT_FAILURE);
    }
  } while (ret == kBufSize); // till EOF (or par_end is found)

  // Now par contains the parameter inputs + some additional including <par_end>
  // Read the stream and load the parameters
  LoadFromStream(par);
  // Seek the file to the end of the header
  input.Seek(header, single_file_per_rank);

  return;
}

//----------------------------------------------------------------------------------------
//! \fn InputBlock* ParameterInput::FindOrAddBlock(std::string name)
//  \brief find or add specified InputBlock.  Returns pointer to block.

InputBlock* ParameterInput::FindOrAddBlock(std::string name) {
  // if block contains no elements, create the first one
  if (block.empty()) {
    block.emplace_front(name);
    return &block.front();

  // else search linked list of InputBlocks to see if name exists, return if found.
  } else {
    for (auto it = block.begin(); it != block.end(); ++it) {
      if (name.compare(it->block_name) == 0) return &*it;
    }

    // Create new block at end of list if not found above, and return pointer to it
    block.emplace_back(name);
    return &block.back();
  }
}

//----------------------------------------------------------------------------------------
//! \fn void ParameterInput::ParseLine(std::string line,
//           std::string& name, std::string& value, std::string& comment)
//  \brief parse "name = value # comment" format, return name/value/comment strings.

void ParameterInput::ParseLine(std::string line, std::string& name, std::string& value,
                               std::string& comment) {
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

//----------------------------------------------------------------------------------------
//! \fn void ParameterInput::AddParameter(InputBlock *pb, std::string name,
//   std::string value, std::string comment)
//  \brief add name/value/comment tuple to the InputLine linked list in block *pb.
//  If a parameter with the same name already exists, the value and comment strings
//  are replaced (overwritten).

void ParameterInput::AddParameter(InputBlock *pb, std::string name, std::string value,
                                  std::string comment) {
  // if line contains no elements, create the first one
  if (pb->line.empty()) {
    pb->line.emplace_front(name,value,comment);
    pb->max_len_parname = name.length();
    pb->max_len_parvalue = value.length();
    return;

  // else search linked list of InputBlocks to see if name exists, replace contents
  // with new values if found and return.
  } else {
    for (auto it = pb->line.begin(); it != pb->line.end(); ++it) {
      if (name.compare(it->param_name) == 0) {   // param name already exists
        it->param_value.assign(value);           // replace existing param value
        it->param_comment.assign(comment);       // replace exisiting param comment
        if (value.length() > pb->max_len_parvalue) pb->max_len_parvalue = value.length();
        return;
      }
    }

  // Parameter not found, so create new node in linked list
    pb->line.emplace_back(name,value,comment);
    if (name.length() > pb->max_len_parname) pb->max_len_parname = name.length();
    if (value.length() > pb->max_len_parvalue) pb->max_len_parvalue = value.length();
  }

  return;
}

//----------------------------------------------------------------------------------------
//! void ParameterInput::ModifyFromCmdline(int argc, char *argv[])
//  \brief parse commandline for changes to input parameters
// Note this function is very forgiving (no warnings!) if there is an error in format

void ParameterInput::ModifyFromCmdline(int argc, char *argv[]) {
  std::string input_text, block,name, value;
  InputBlock *pb;
  InputLine *pl;

  for (int i=1; i<argc; i++) {
    input_text = argv[i];
    std::size_t slash_posn = input_text.find_first_of("/");   // find "/" character
    std::size_t equal_posn = input_text.find_first_of("=");   // find "=" character

    // skip if either "/" or "=" do not exist in input
    if ((slash_posn == std::string::npos) || (equal_posn == std::string::npos)) continue;

    // extract block/name/value strings
    block = input_text.substr(0, slash_posn);
    name  = input_text.substr(slash_posn+1, (equal_posn - slash_posn - 1));
    value = input_text.substr(equal_posn+1, std::string::npos);

    // get pointer to node with same block name in linked list of InputBlocks
    pb = GetPtrToBlock(block);
    if (pb == nullptr) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Block name '" << block << "' on command line not found"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }

    // get pointer to node with same parameter name in linked list of InputLines
    pl = pb->GetPtrToLine(name);
    if (pl == nullptr) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Parameter '" << name << "' in block '" << block
                << "' on command line not found" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    pl->param_value.assign(value);   // replace existing value

    if (value.length() > pb->max_len_parvalue) pb->max_len_parvalue = value.length();
  }
}


//----------------------------------------------------------------------------------------
//! \fn bool ParameterInput::DoesBlockExist(std::string name)
//  \brief check whether block of given name exists

bool ParameterInput::DoesBlockExist(std::string name) {
  InputBlock *pb;
  pb = GetPtrToBlock(name);
  return (pb == nullptr ? false : true);
}

//----------------------------------------------------------------------------------------
//! \fn bool ParameterInput::DoesParameterExist(std::string block, std::string name)
//  \brief check whether parameter of given name in given block exists

bool ParameterInput::DoesParameterExist(std::string block, std::string name) {
  InputLine *pl;
  InputBlock *pb;
  pb = GetPtrToBlock(block);
  if (pb == nullptr) return 0;
  pl = pb->GetPtrToLine(name);
  return (pl == nullptr ? false : true);
}

//----------------------------------------------------------------------------------------
//! \fn int ParameterInput::GetInteger(std::string block, std::string name)
//  \brief returns integer value of string stored in block/name

int ParameterInput::GetInteger(std::string block, std::string name) {
  InputBlock* pb;
  InputLine* pl;

  Lock();

  // get pointer to node with same block name in linked list of InputBlocks
  pb = GetPtrToBlock(block);
  if (pb == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Block name '" << block << "' not found when trying to set value "
              << "for parameter '" << name << "'" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // get pointer to node with same parameter name in linked list of InputLines
  pl = pb->GetPtrToLine(name);
  if (pl == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Parameter name '" << name << "' not found in block '" << block
              << "'" <<std::endl;
    std::exit(EXIT_FAILURE);
  }

  std::string val=pl->param_value;
  Unlock();

  // Convert string to integer and return value
  return atoi(val.c_str());
}

//----------------------------------------------------------------------------------------
//! \fn Real ParameterInput::GetReal(std::string block, std::string name)
//  \brief returns real value of string stored in block/name

Real ParameterInput::GetReal(std::string block, std::string name) {
  InputBlock* pb;
  InputLine* pl;

  Lock();

  // get pointer to node with same block name in linked list of InputBlocks
  pb = GetPtrToBlock(block);
  if (pb == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Block name '" << block << "' not found when trying to set value "
              << "for parameter '" << name << "'" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // get pointer to node with same parameter name in linked list of InputLines
  pl = pb->GetPtrToLine(name);
  if (pl == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Parameter name '" << name << "' not found in block '" << block
              << "'" <<std::endl;
    std::exit(EXIT_FAILURE);
  }

  std::string val=pl->param_value;
  Unlock();

  // Convert string to real and return value
  return static_cast<Real>(atof(val.c_str()));
}

//----------------------------------------------------------------------------------------
//! \fn bool ParameterInput::GetBoolean(std::string block, std::string name)
//  \brief returns boolean value of string stored in block/name

bool ParameterInput::GetBoolean(std::string block, std::string name) {
  InputBlock* pb;
  InputLine* pl;

  Lock();

  // get pointer to node with same block name in linked list of InputBlocks
  pb = GetPtrToBlock(block);
  if (pb == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Block name '" << block << "' not found when trying to set value "
              << "for parameter '" << name << "'" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // get pointer to node with same parameter name in linked list of InputLines
  pl = pb->GetPtrToLine(name);
  if (pl == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Parameter name '" << name << "' not found in block '" << block
              << "'"<< std::endl;
    std::exit(EXIT_FAILURE);
  }

  std::string val=pl->param_value;
  Unlock();

  // check is string contains integers 0 or 1 (instead of true or false) and return
  if (val.compare(0, 1, "0")==0 || val.compare(0, 1, "1")==0) {
    return static_cast<bool>(atoi(val.c_str()));
  }

  // convert string to all lower case
  std::transform(val.begin(), val.end(), val.begin(), ::tolower);
  // Convert string to bool and return value
  bool b;
  std::istringstream is(val);
  is >> std::boolalpha >> b;

  return (b);
}

//----------------------------------------------------------------------------------------
//! \fn std::string ParameterInput::GetString(std::string block, std::string name)
//  \brief returns string stored in block/name

std::string ParameterInput::GetString(std::string block, std::string name) {
  InputBlock* pb;
  InputLine* pl;

  Lock();

  // get pointer to node with same block name in linked list of InputBlocks
  pb = GetPtrToBlock(block);
  if (pb == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Block name '" << block << "' not found when trying to set value "
              << "for parameter '" << name << "'" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // get pointer to node with same parameter name in linked list of InputLines
  pl = pb->GetPtrToLine(name);
  if (pl == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Parameter name '" << name << "' not found in block '" << block
              << "'"<< std::endl;
    std::exit(EXIT_FAILURE);
  }

  std::string val=pl->param_value;
  Unlock();

  // return value
  return val;
}

//----------------------------------------------------------------------------------------
//! \fn int ParameterInput::GetOrAddInteger(std::string block, std::string name,
//    int default_value)
//  \brief returns integer value stored in block/name if it exists, or creates and sets
//  value to def_value if it does not exist

int ParameterInput::GetOrAddInteger(std::string block, std::string name, int def_value) {
  InputBlock* pb;
  InputLine *pl;
  std::stringstream ss_value;
  int ret;

  Lock();
  if (DoesParameterExist(block, name)) {
    pb = GetPtrToBlock(block);
    pl = pb->GetPtrToLine(name);
    std::string val = pl->param_value;
    ret = atoi(val.c_str());
  } else {
    pb = FindOrAddBlock(block);
    ss_value << def_value;
    AddParameter(pb, name, ss_value.str(), "# Default value added at run time");
    ret = def_value;
  }
  Unlock();
  return ret;
}

//----------------------------------------------------------------------------------------
//! \fn Real ParameterInput::GetOrAddReal(std::string block, std::string name,
//    Real def_value)
//  \brief returns real value stored in block/name if it exists, or creates and sets
//  value to def_value if it does not exist

Real ParameterInput::GetOrAddReal(std::string block, std::string name, Real def_value) {
  InputBlock* pb;
  InputLine *pl;
  std::stringstream ss_value;
  Real ret;

  Lock();
  if (DoesParameterExist(block, name)) {
    pb = GetPtrToBlock(block);
    pl = pb->GetPtrToLine(name);
    std::string val = pl->param_value;
    ret = static_cast<Real>(atof(val.c_str()));
  } else {
    pb = FindOrAddBlock(block);
    ss_value << def_value;
    AddParameter(pb, name, ss_value.str(), "# Default value added at run time");
    ret = def_value;
  }
  Unlock();
  return ret;
}

//----------------------------------------------------------------------------------------
//! \fn bool ParameterInput::GetOrAddBoolean(std::string block, std::string name,
//    bool def_value)
//  \brief returns boolean value stored in block/name if it exists, or creates and sets
//  value to def_value if it does not exist

bool ParameterInput::GetOrAddBoolean(std::string block,std::string name, bool def_value) {
  InputBlock* pb;
  InputLine *pl;
  std::stringstream ss_value;
  bool ret;

  Lock();
  if (DoesParameterExist(block, name)) {
    pb = GetPtrToBlock(block);
    pl = pb->GetPtrToLine(name);
    std::string val = pl->param_value;
    if (val.compare(0, 1, "0")==0 || val.compare(0, 1, "1")==0) {
      ret = static_cast<bool>(atoi(val.c_str()));
    } else {
      std::transform(val.begin(), val.end(), val.begin(), ::tolower);
      std::istringstream is(val);
      is >> std::boolalpha >> ret;
    }
  } else {
    pb = FindOrAddBlock(block);
    ss_value << def_value;
    AddParameter(pb, name, ss_value.str(), "# Default value added at run time");
    ret = def_value;
  }
  Unlock();
  return ret;
}

//----------------------------------------------------------------------------------------
//! \fn std::string ParameterInput::GetOrAddString(std::string block, std::string name,
//                                                 std::string def_value)
//  \brief returns string value stored in block/name if it exists, or creates and sets
//  value to def_value if it does not exist

std::string ParameterInput::GetOrAddString(std::string block, std::string name,
                                           std::string def_value) {
  InputBlock* pb;
  InputLine *pl;
  std::stringstream ss_value;
  std::string ret;

  Lock();
  if (DoesParameterExist(block, name)) {
    pb = GetPtrToBlock(block);
    pl = pb->GetPtrToLine(name);
    ret = pl->param_value;
  } else {
    pb = FindOrAddBlock(block);
    AddParameter(pb, name, def_value, "# Default value added at run time");
    ret = def_value;
  }
  Unlock();
  return ret;
}

//----------------------------------------------------------------------------------------
//! \fn int ParameterInput::SetInteger(std::string block, std::string name, int value)
//  \brief updates an integer parameter; creates it if it does not exist

int ParameterInput::SetInteger(std::string block, std::string name, int value) {
  InputBlock* pb;
  std::stringstream ss_value;

  Lock();
  pb = FindOrAddBlock(block);
  ss_value << value;
  AddParameter(pb, name, ss_value.str(), "# Updated during run time");
  Unlock();
  return value;
}

//----------------------------------------------------------------------------------------
//! \fn Real ParameterInput::SetReal(std::string block, std::string name, Real value)
//  \brief updates a real parameter; creates it if it does not exist

Real ParameterInput::SetReal(std::string block, std::string name, Real value) {
  InputBlock* pb;
  std::stringstream ss_value;

  Lock();
  pb = FindOrAddBlock(block);
  ss_value << value;
  AddParameter(pb, name, ss_value.str(), "# Updated during run time");
  Unlock();
  return value;
}

//----------------------------------------------------------------------------------------
//! \fn bool ParameterInput::SetBoolean(std::string block, std::string name, bool value)
//  \brief updates a boolean parameter; creates it if it does not exist

bool ParameterInput::SetBoolean(std::string block, std::string name, bool value) {
  InputBlock* pb;
  std::stringstream ss_value;

  Lock();
  pb = FindOrAddBlock(block);
  ss_value << value;
  AddParameter(pb, name, ss_value.str(), "# Updated during run time");
  Unlock();
  return value;
}

//----------------------------------------------------------------------------------------
//! \fn std::string ParameterInput::SetString(std::string block, std::string name,
//                                            std::string  value)
//  \brief updates a string parameter; creates it if it does not exist

std::string ParameterInput::SetString(std::string block, std::string name,
                                      std::string value) {
  InputBlock* pb;

  Lock();
  pb = FindOrAddBlock(block);
  AddParameter(pb, name, value, "# Updated during run time");
  Unlock();
  return value;
}


//----------------------------------------------------------------------------------------
//! \fn void ParameterInput::ParameterDump(std::ostream& os)
//  \brief output entire InputBlock/InputLine hierarchy to specified stream

void ParameterInput::ParameterDump(std::ostream& os) {
  std::string param_name,param_value;
  std::size_t len;

  os<< "#------------------------- PAR_DUMP -------------------------" << std::endl;

  for (auto itb = block.begin(); itb != block.end(); ++itb) {     // loop over InputBlocks
    os<< "<" << itb->block_name << ">" << std::endl;              // write block name
    // loop over InputLines and write each parameter name/value
    for (auto itl = itb->line.begin(); itl != itb->line.end(); ++itl) {
      param_name.assign(itl->param_name);
      param_value.assign(itl->param_value);

      len = itb->max_len_parname - param_name.length() + 1;
      param_name.append(len,' ');                         // pad name  to align vertically
      len = itb->max_len_parvalue - param_value.length() + 1;
      param_value.append(len,' ');                        // pad value to align vertically

      os<< param_name << "= " << param_value << itl->param_comment <<  std::endl;
    }
  }

  os<< "#------------------------- PAR_DUMP -------------------------" << std::endl;
  os<< "<par_end>" << std::endl;    // finish with par-end (needed for restart files)
}
