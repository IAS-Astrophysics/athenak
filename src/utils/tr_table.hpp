#ifndef UTILS_TR_TABLE_HPP_
#define UTILS_TR_TABLE_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file tr_table.hpp
//! \brief Declares Table class

#include <string>
#include <map>
#include <vector>
#include <fstream>
#include <sstream>
#include <utility>

namespace TableReader {

struct ReadResult {
  enum ErrorCode {
    SUCCESS,
    BAD_FILENAME,
    BAD_HEADER
  };
  ErrorCode error;
  std::string message;
};

class Table {
 public:
  Table();
  ~Table();

  ReadResult ReadTable(const std::string fname);

  inline const std::map<std::string, std::string> GetMetadata() {
    return metadata;
  }

  inline const std::vector<std::pair<std::string, size_t>>& GetPointInfo() {
    return point_info;
  }

  inline const std::map<std::string, double> GetScalars() {
    return scalars;
  }

  inline size_t GetNDimensions() {
    return ndim;
  }

  inline size_t GetNPoints() {
    return npoints;
  }

  inline size_t GetMemSize() {
    return mem_size;
  }

  inline bool HasField(const std::string& field) {
    return (fields.find(field) != fields.end());
  }

  inline const std::vector<std::string>& GetFieldNames() {
    return field_names;
  }

  inline double * operator[](const std::string& field) {
    return fields[field];
  }

  inline bool IsInitialize() {
    return initialized;
  }

  inline const double * GetRawData() {
    return data;
  }

 private:
  template<typename F>
  ReadResult ParseBlock(std::string name, std::vector<std::string>& block_lines, F add) {
    ReadResult result;
    for (auto line : block_lines) {
      std::string key, value;
      bool success = SplitToken(line, key, value);
      if (!success) {
        result.error = ReadResult::BAD_HEADER;
        std::stringstream ss;
        ss << "'" << line << "' is not a valid " << name << " line.\n";
        result.message = ss.str();
        return result;
      } else {
        add(key, value);
      }
    }
    result.error = ReadResult::SUCCESS;
    return result;
  }

  ReadResult ExtractBlock(std::ifstream& file, const std::string name,
                          std::vector<std::string>& lines);

  bool SplitToken(const std::string& in, std::string& key, std::string& value);

  void TrimWhiteSpace(std::string& str);

  std::map<std::string, std::string> metadata;
  std::vector<std::pair<std::string, size_t>> point_info;
  std::vector<std::string> field_names;
  std::map<std::string, double> scalars;
  std::map<std::string, double*> fields;
  double * data;
  size_t ndim;
  size_t npoints;
  size_t mem_size;
  bool initialized;
};

} // namespace TableReader

#endif // UTILS_TR_TABLE_HPP_
