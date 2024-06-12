//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file tr_table.cpp
//! \brief Implementation of Table class
#include <string>
#include <fstream>
#include <sstream>
#include <map>
#include <vector>

#include "tr_table.hpp"
#include "tr_utils.hpp"

using namespace TableReader; // NOLINT

Table::Table() : ndim(0), npoints(0), mem_size(0), initialized(false) {
}

Table::~Table() {
  if (initialized) {
    delete[] data;
  }
}

ReadResult Table::ReadTable(const std::string fname) {
  ReadResult result;

  std::ifstream file;
  try {
    file.open(fname.c_str(), std::ifstream::in);
  } catch (std::ifstream::failure& e) {
    result.error = ReadResult::BAD_FILENAME;
    std::stringstream ss;
    ss << "Could not read '" << fname << "'\n"
       << "open() returned the following error:\n"
       << e.what();
    result.message = ss.str();
    return result;
  }

  // Something bizarre happened while reading the file.
  if (!file.is_open()) {
    result.error = ReadResult::BAD_FILENAME;
    std::stringstream ss;
    ss << "No exception occurred, but ReadTable() failed to open '" << fname << "'\n";
    result.message = ss.str();
    return result;
  }

  // HEADER PARSING

  // Read in the metadata
  std::vector<std::string> block_lines;

  result = ExtractBlock(file, "metadata", block_lines);
  if (result.error != ReadResult::SUCCESS) {
    file.close();
    return result;
  }
  result = ParseBlock("metadata", block_lines,
  [&](const std::string& k, const std::string& v) {
    metadata[k] = v;
  });
  if (result.error != ReadResult::SUCCESS) {
    file.close();
    return result;
  }
  block_lines.clear();

  // Read in the scalars
  result = ExtractBlock(file, "scalars", block_lines);
  if (result.error != ReadResult::SUCCESS) {
    file.close();
    return result;
  }
  result = ParseBlock("scalars", block_lines,
  [&](const std::string& k, const std::string& v) {
    scalars[k] = std::stod(v);
  });
  if (result.error != ReadResult::SUCCESS) {
    file.close();
    return result;
  }
  block_lines.clear();

  // Read in the points
  result = ExtractBlock(file, "points", block_lines);
  if (result.error != ReadResult::SUCCESS) {
    file.close();
    return result;
  }
  result = ParseBlock("points", block_lines,
  [&](const std::string& k, const std::string& v) {
    point_info.push_back({k, std::stoi(v)});
  });
  if (result.error != ReadResult::SUCCESS) {
    file.close();
    return result;
  }
  block_lines.clear();
  ndim = point_info.size();

  // Read in the fields
  result = ExtractBlock(file, "fields", block_lines);
  if (result.error != ReadResult::SUCCESS) {
    file.close();
    return result;
  }
  for (auto line : block_lines) {
    TrimWhiteSpace(line);
    field_names.push_back(line);
  }

  size_t header_size = file.tellg();
  file.close();

  // Allocate memory for the fields.
  npoints = 1;
  mem_size = 0;
  for (auto& p : point_info) {
    npoints *= p.second;
    mem_size += p.second;
  }
  mem_size += npoints*field_names.size();
  data = new double[mem_size];

  // Set the memory offsets for all the fields.
  size_t offset = 0;
  for (auto &p : point_info) {
    fields[p.first] = &data[offset];
    offset += p.second;
  }
  for (auto &s : field_names) {
    fields[s] = &data[offset];
    offset += npoints;
  }

  initialized = true;

  // Now we need to load the table memory itself. We reopen the file as a binary.
  try {
    file.open(fname.c_str(), std::ifstream::in | std::ifstream::binary);
  } catch (std::ifstream::failure& e) {
    result.error = ReadResult::BAD_FILENAME;
    std::stringstream ss;
    ss << "Could not read '" << fname << "' as a binary file\n"
       << "open() returned the following error:\n"
       << e.what();
    result.message = ss.str();
    return result;
  }
  if (!file.is_open()) {
    result.error = ReadResult::BAD_FILENAME;
    std::stringstream ss;
    ss << "No exception occurred, but ReadTable() failed to open '" << fname << "'\n"
       << "as a binary file\n";
    result.message = ss.str();
    return result;
  }

  // Because we've already read the header, we skip ahead to the binary section.
  file.seekg(header_size);
  // Dump all the memory into the data.
  // FIXME(JMF): This doesn't handle single precision data correctly!
  char *memblock = reinterpret_cast<char*>(data);
  file.read(memblock, mem_size*sizeof(double));

  // Now we need to check for endianness.
  if ((!metadata["endianness"].compare("little") && !IsLittleEndian()) ||
      (!metadata["endianness"].compare("big") && IsLittleEndian())) {
    for (size_t i = 0; i < mem_size; i++) {
      data[i] = SwapEndianness(data[i]);
    }
    result.message = "Swapped endianness of data.\n";
  }

  file.close();

  result.error = ReadResult::SUCCESS;

  return result;
}

ReadResult Table::ExtractBlock(std::ifstream& file, const std::string name,
                               std::vector<std::string>& lines) {
  ReadResult result;
  // Read the first block
  std::stringstream ss;
  ss << "<" << name << "begin" << ">";
  char buffer[256];
  file.getline(buffer, 256);

  std::string line = std::string(buffer);
  if (line.compare(ss.str()) != 0) {
    ss.str("");
    result.error = ReadResult::BAD_HEADER;
    ss << "Header is either missing '" << name << "' or is in the wrong order.\n";
    result.message = ss.str();
    return result;
  }

  ss.str("");
  ss << "<" << name << "end" << ">";
  while(!file.eof()) {
    file.getline(buffer, 256);
    // Check if we're at the end of this block.
    line = std::string(buffer);
    if (line[0] == '<') {
      if (line.compare(ss.str()) != 0) {
        ss.str("");
        result.error = ReadResult::BAD_HEADER;
        ss << "Unexpected new block before reaching end of '" << name << "' in header.\n";
        result.message = ss.str();
        return result;
      } else {
        break;
      }
    } else {
      lines.push_back(line);
    }
  }

  if (file.eof()) {
    // We reached the end of the file, but we shouldn't have.
    ss.str("");
    result.error = ReadResult::BAD_HEADER;
    ss << "Unexpected end of file while reading '" << name << "' in header.\n";
    result.message = ss.str();
    return result;
  }

  result.error = ReadResult::SUCCESS;
  return result;
}

bool Table::SplitToken(const std::string& in, std::string& key, std::string& value) {
  size_t pos = in.find('=');
  // The equals sign does not exist or is in the wrong location.
  if (pos == std::string::npos) {
    return false;
  } else if (in.back() == '=' || in.front() == '=') {
    return false;
  }

  key = in.substr(0, pos);
  value = in.substr(pos+1, in.size());

  TrimWhiteSpace(key);
  TrimWhiteSpace(value);

  return true;
}

void Table::TrimWhiteSpace(std::string& str) {
  while(str.front() == ' ' || str.front() == '\t') {
    str.erase(0,1);
  }
  while(str.back() == ' ' || str.front() == '\t') {
    str.pop_back();
  }
}
