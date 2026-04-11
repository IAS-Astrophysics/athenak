#ifndef FILE_SHARDING_HPP_
#define FILE_SHARDING_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file file_sharding.hpp
//! \brief helpers for shared, per-node, and per-rank output sharding.

#include <cstdio>
#include <string>
#include <vector>

#include "globals.hpp"

enum class FileShardMode {
  shared,
  per_node,
  per_rank
};

inline bool UsesSerialIO(FileShardMode mode) {
  return (mode == FileShardMode::per_rank);
}

inline bool IsPartitioned(FileShardMode mode) {
  return (mode != FileShardMode::shared);
}

inline const char *ShardDistributionName(FileShardMode mode) {
  switch (mode) {
    case FileShardMode::shared:
      return "shared";
    case FileShardMode::per_node:
      return "node";
    case FileShardMode::per_rank:
      return "rank";
    default:
      return "shared";
  }
}

inline bool IsShardWriter(FileShardMode mode) {
  switch (mode) {
    case FileShardMode::shared:
      return (global_variable::my_rank == 0);
    case FileShardMode::per_node:
      return (global_variable::rank_in_node == 0);
    case FileShardMode::per_rank:
      return true;
    default:
      return false;
  }
}

inline int ShardWriterId(FileShardMode mode) {
  switch (mode) {
    case FileShardMode::shared:
      return 0;
    case FileShardMode::per_node:
      return global_variable::node_id;
    case FileShardMode::per_rank:
      return global_variable::my_rank;
    default:
      return 0;
  }
}

inline int ShardLocalIndex(FileShardMode mode) {
  switch (mode) {
    case FileShardMode::shared:
      return global_variable::my_rank;
    case FileShardMode::per_node:
      return global_variable::rank_in_node;
    case FileShardMode::per_rank:
      return 0;
    default:
      return 0;
  }
}

inline int ShardParticipantCount(FileShardMode mode) {
  switch (mode) {
    case FileShardMode::shared:
      return global_variable::nranks;
    case FileShardMode::per_node:
      return global_variable::ranks_per_node;
    case FileShardMode::per_rank:
      return 1;
    default:
      return 1;
  }
}

inline std::string ShardDirectoryName(FileShardMode mode) {
  if (mode == FileShardMode::per_rank) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "rank_%08d/", global_variable::my_rank);
    return std::string(buf);
  }
  if (mode == FileShardMode::per_node) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "node_%08d/", global_variable::node_id);
    return std::string(buf);
  }
  return "";
}

inline std::string CanonicalShardDirectoryName(FileShardMode mode) {
  if (mode == FileShardMode::per_rank) {
    return "rank_00000000/";
  }
  if (mode == FileShardMode::per_node) {
    return "node_00000000/";
  }
  return "";
}

#if MPI_PARALLEL_ENABLED
inline MPI_Comm ShardCommunicator(FileShardMode mode) {
  if (mode == FileShardMode::per_node) {
    return global_variable::node_comm;
  }
  return MPI_COMM_WORLD;
}
#endif

inline std::vector<int> GatherShardCounts(int local_count, FileShardMode mode) {
  std::vector<int> counts(ShardParticipantCount(mode), local_count);
#if MPI_PARALLEL_ENABLED
  if (mode == FileShardMode::shared) {
    MPI_Allgather(&local_count, 1, MPI_INT, counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
  } else if (mode == FileShardMode::per_node) {
    MPI_Allgather(&local_count, 1, MPI_INT, counts.data(), 1, MPI_INT,
                  global_variable::node_comm);
  }
#endif
  return counts;
}

inline int PrefixCountBeforeMe(const std::vector<int> &counts, FileShardMode mode) {
  int prefix = 0;
  int my_index = ShardLocalIndex(mode);
  for (int i = 0; i < my_index; ++i) {
    prefix += counts[i];
  }
  return prefix;
}

#endif // FILE_SHARDING_HPP_
