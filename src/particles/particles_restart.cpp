//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file particles_restart.cpp
//! \brief particle restart sidecar input and output

#include <sys/stat.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "outputs/io_wrapper.hpp"
#include "cosmic_ray.hpp"
#include "lagrangian_mc.hpp"
#include "particles.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

namespace particles {
namespace {

constexpr char restart_magic[8] = {'A', 'T', 'H', 'P', 'R', 'S', 'T', '1'};
constexpr std::uint32_t restart_format_version = 1;
constexpr std::uint32_t endian_marker = 0x01020304;
constexpr std::size_t population_name_size = 64;
constexpr std::uint64_t restart_header_size =
    8 + 5*sizeof(std::uint32_t) + 2*sizeof(std::uint64_t);
constexpr std::uint64_t population_entry_size =
    population_name_size + 4*sizeof(std::uint32_t) + sizeof(std::uint64_t);
constexpr std::uint64_t block_entry_size = sizeof(std::uint64_t);

struct BlockRestartData {
  std::uint64_t count;
  std::uint64_t rdata_offset;
  std::uint64_t idata_offset;
};

struct PopulationRestartData {
  ParticlePopulation *population;
  HostArray2D<Real> rdata;
  HostArray2D<int> idata;
  std::vector<std::vector<int>> ordered_particles;
  std::vector<std::uint64_t> counts;
  std::vector<BlockRestartData> blocks;
  std::vector<char> metadata;
  std::uint64_t metadata_offset;
  std::uint64_t directory_offset;
};

struct PopulationRestartHeader {
  std::string name;
  std::uint32_t particle_type;
  std::uint32_t layout_version;
  std::uint32_t nrdata;
  std::uint32_t nidata;
  std::uint64_t metadata_size;
};

//----------------------------------------------------------------------------------------
// Stop on a malformed or incompatible particle restart.

void RestartError(const std::string &message) {
  std::cout << "### FATAL ERROR in particle restart" << std::endl
            << message << std::endl;
#if MPI_PARALLEL_ENABLED
  MPI_Abort(MPI_COMM_WORLD, 1);
#endif
  std::exit(EXIT_FAILURE);
}

//----------------------------------------------------------------------------------------
// Checked arithmetic for file sizes and offsets.

std::uint64_t CheckedAdd(std::uint64_t a, std::uint64_t b) {
  if (b > std::numeric_limits<std::uint64_t>::max() - a) {
    RestartError("Particle restart byte offset overflow.");
  }
  return a + b;
}

std::uint64_t CheckedMultiply(std::uint64_t a, std::uint64_t b) {
  if (a != 0 && b > std::numeric_limits<std::uint64_t>::max()/a) {
    RestartError("Particle restart byte count overflow.");
  }
  return a*b;
}

//----------------------------------------------------------------------------------------
// Check that a particle position is compatible with its MeshBlock assignment.

bool PositionInMeshBlock(Real x1, Real x2, Real x3, const RegionSize &block_size) {
  return (x1 >= block_size.x1min && x1 < block_size.x1max &&
          x2 >= block_size.x2min && x2 < block_size.x2max &&
          x3 >= block_size.x3min && x3 < block_size.x3max);
}

//----------------------------------------------------------------------------------------
// Check for a retained particle beyond a nonperiodic global boundary.

bool RetainedBeyondPhysicalBoundary(Real x1, Real x2, Real x3, int status,
                                    const Mesh *pm) {
  if (status != PFROZEN && status != PDELETE_PENDING &&
      status != PDELETE_AFTER_SNAPSHOT) {
    return false;
  }
  if (!std::isfinite(x1) || !std::isfinite(x2) || !std::isfinite(x3)) return false;

  auto periodic = [](BoundaryFlag flag) {
    return flag == BoundaryFlag::periodic || flag == BoundaryFlag::shear_periodic;
  };
  const RegionSize &mesh_size = pm->mesh_size;
  return ((x1 < mesh_size.x1min &&
           !periodic(pm->mesh_bcs[BoundaryFace::inner_x1])) ||
          (x1 >= mesh_size.x1max &&
           !periodic(pm->mesh_bcs[BoundaryFace::outer_x1])) ||
          (x2 < mesh_size.x2min &&
           !periodic(pm->mesh_bcs[BoundaryFace::inner_x2])) ||
          (x2 >= mesh_size.x2max &&
           !periodic(pm->mesh_bcs[BoundaryFace::outer_x2])) ||
          (x3 < mesh_size.x3min &&
           !periodic(pm->mesh_bcs[BoundaryFace::inner_x3])) ||
          (x3 >= mesh_size.x3max &&
           !periodic(pm->mesh_bcs[BoundaryFace::outer_x3])));
}

//----------------------------------------------------------------------------------------
// Append and recover fixed-width values without writing C++ structs as binary blobs.

void AppendBytes(std::vector<char> &buffer, const void *data, std::size_t size) {
  const char *bytes = reinterpret_cast<const char*>(data);
  buffer.insert(buffer.end(), bytes, bytes + size);
}

void AppendUInt32(std::vector<char> &buffer, std::uint32_t value) {
  AppendBytes(buffer, &value, sizeof(value));
}

void AppendUInt64(std::vector<char> &buffer, std::uint64_t value) {
  AppendBytes(buffer, &value, sizeof(value));
}

std::uint32_t ReadUInt32(const std::vector<char> &buffer, std::size_t &offset) {
  if (offset > buffer.size() || sizeof(std::uint32_t) > buffer.size() - offset) {
    RestartError("Particle restart header or directory is truncated.");
  }
  std::uint32_t value;
  std::memcpy(&value, buffer.data() + offset, sizeof(value));
  offset += sizeof(value);
  return value;
}

std::uint64_t ReadUInt64(const std::vector<char> &buffer, std::size_t &offset) {
  if (offset > buffer.size() || sizeof(std::uint64_t) > buffer.size() - offset) {
    RestartError("Particle restart header or directory is truncated.");
  }
  std::uint64_t value;
  std::memcpy(&value, buffer.data() + offset, sizeof(value));
  offset += sizeof(value);
  return value;
}

//----------------------------------------------------------------------------------------
// Read and write byte ranges in chunks accepted by MPI count arguments.

void WriteBytesAt(IOWrapper &file, const void *data, std::uint64_t size,
                  std::uint64_t offset) {
  const char *bytes = reinterpret_cast<const char*>(data);
  std::uint64_t written = 0;
  while (written < size) {
    std::uint64_t chunk = std::min<std::uint64_t>(size - written, INT_MAX);
    if (file.Write_any_type_at(bytes + written, chunk, offset + written, "byte") != chunk) {
      RestartError("Particle data were not written correctly to the restart sidecar.");
    }
    written += chunk;
  }
}

void ReadBytesAt(IOWrapper &file, void *data, std::uint64_t size, std::uint64_t offset) {
  char *bytes = reinterpret_cast<char*>(data);
  std::uint64_t nread = 0;
  while (nread < size) {
    std::uint64_t chunk = std::min<std::uint64_t>(size - nread, INT_MAX);
    if (file.Read_bytes_at(bytes + nread, 1, chunk, offset + nread) != chunk) {
      RestartError("Particle data were not read correctly from the restart sidecar.");
    }
    nread += chunk;
  }
}

//----------------------------------------------------------------------------------------
// Return the stable on-disk identifiers for particle types.

std::uint32_t ParticleTypeID(ParticleType type) {
  switch (type) {
    case ParticleType::cosmic_ray:
      return 1;
    case ParticleType::lagrangian_mc:
      return 2;
    default:
      RestartError("Particle type has no restart identifier.");
  }
  return 0;
}

//----------------------------------------------------------------------------------------
// Return the size of a completed sidecar.

std::uint64_t FileSize(const std::string &filename) {
  struct stat file_info;
  if (stat(filename.c_str(), &file_info) != 0 || file_info.st_size < 0) {
    RestartError("Unable to inspect particle restart file '" + filename + "'.");
  }
  return static_cast<std::uint64_t>(file_info.st_size);
}

} // namespace

//----------------------------------------------------------------------------------------
//! \brief Return the version of this particle type's durable array layout.

int ParticlePopulation::RestartLayoutVersion() const {
  switch (particle_type) {
    case ParticleType::cosmic_ray:
      return cosmic_ray::RESTART_LAYOUT_VERSION;
    case ParticleType::lagrangian_mc:
      return lagrangian_mc::RESTART_LAYOUT_VERSION;
    default:
      RestartError("Particle type has no restart layout version.");
  }
  return 0;
}

//----------------------------------------------------------------------------------------
//! \brief Pack durable model state that is not stored in the particle arrays.

std::vector<char> ParticlePopulation::RestartMetadata() const {
  return {};
}

//----------------------------------------------------------------------------------------
//! \brief Validate durable model state against the configured particle model.

void ParticlePopulation::ValidateRestartMetadata(const std::vector<char> &metadata) const {
  if (particle_type != ParticleType::cosmic_ray &&
      particle_type != ParticleType::lagrangian_mc) {
    RestartError("Particle type has no restart metadata contract.");
  }
  if (!metadata.empty()) {
    RestartError("Particle restart contains unexpected model metadata.");
  }
}

//----------------------------------------------------------------------------------------
//! \fn Particles::WriteRestart
//! \brief Write all particle populations to a shared, MeshBlock-indexed sidecar.

void Particles::WriteRestart(const std::string &filename) const {
  const int nmb = pmy_pack_->pmesh->nmb_total;
  const int gids = pmy_pack_->gids;
  const int gide = pmy_pack_->gide;
  std::vector<PopulationRestartData> restart_data;

  for (auto *population : populations_) {
    if (population->nidata <= 1) {
      RestartError("Particle layout does not contain the common integer fields.");
    }
    PopulationRestartData data;
    data.population = population;
    data.metadata = population->RestartMetadata();
    data.ordered_particles.resize(nmb);
    data.counts.assign(nmb, 0);
    data.blocks.resize(nmb);

    const int npart = population->nprtcl_thispack;
    data.rdata = HostArray2D<Real>("particle restart real data", population->nrdata, npart);
    data.idata = HostArray2D<int>("particle restart int data", population->nidata, npart);
    Kokkos::deep_copy(data.rdata, population->prtcl_rdata);
    Kokkos::deep_copy(data.idata, population->prtcl_idata);

    for (int p=0; p<npart; ++p) {
      const int gid = data.idata(PGID,p);
      const int tag = data.idata(PTAG,p);
      const int status = data.idata(PSTATUS,p);
      if (gid < gids || gid > gide) {
        RestartError("A particle is not owned by a local MeshBlock at checkpoint time.");
      }
      if (tag < 0) {
        RestartError("A particle tag is negative at checkpoint time.");
      }
      if (status < PACTIVE || status > PDELETE_AFTER_SNAPSHOT) {
        RestartError("A particle has an invalid lifecycle status at checkpoint time.");
      }
      const RegionSize &block_size = pmy_pack_->pmb->mb_size.h_view(gid - gids);
      if (!PositionInMeshBlock(data.rdata(IPX,p), data.rdata(IPY,p),
                               data.rdata(IPZ,p), block_size) &&
          !RetainedBeyondPhysicalBoundary(
              data.rdata(IPX,p), data.rdata(IPY,p), data.rdata(IPZ,p), status,
              pmy_pack_->pmesh)) {
        RestartError("A particle position is outside its MeshBlock at checkpoint time "
                     "(GID " + std::to_string(gid) + ").");
      }
      data.ordered_particles[gid].push_back(p);
      data.counts[gid]++;
    }

    if (restart_sort_by_tag_) {
      for (int gid=gids; gid<=gide; ++gid) {
        auto &order = data.ordered_particles[gid];
        std::sort(order.begin(), order.end(), [&data](int a, int b) {
          return data.idata(PTAG,a) < data.idata(PTAG,b);
        });
      }
    }

#if MPI_PARALLEL_ENABLED
    MPI_Allreduce(MPI_IN_PLACE, data.counts.data(), nmb, MPI_UINT64_T, MPI_SUM,
                  MPI_COMM_WORLD);
#endif
    restart_data.push_back(data);
  }

  const std::uint64_t npop = restart_data.size();
  std::uint64_t next_offset = CheckedAdd(
      restart_header_size, CheckedMultiply(npop, population_entry_size));
  for (auto &data : restart_data) {
    data.directory_offset = next_offset;
    next_offset = CheckedAdd(next_offset, CheckedMultiply(nmb, block_entry_size));
  }
  for (auto &data : restart_data) {
    data.metadata_offset = next_offset;
    next_offset = CheckedAdd(next_offset, data.metadata.size());
  }
  for (auto &data : restart_data) {
    for (int gid=0; gid<nmb; ++gid) {
      BlockRestartData &block = data.blocks[gid];
      block.count = data.counts[gid];
      block.rdata_offset = next_offset;
      std::uint64_t rsize = CheckedMultiply(block.count, data.population->nrdata);
      rsize = CheckedMultiply(rsize, sizeof(Real));
      next_offset = CheckedAdd(next_offset, rsize);
      block.idata_offset = next_offset;
      std::uint64_t isize = CheckedMultiply(block.count, data.population->nidata - 1);
      isize = CheckedMultiply(isize, sizeof(int));
      next_offset = CheckedAdd(next_offset, isize);
    }
  }

  std::vector<char> header;
  AppendBytes(header, restart_magic, sizeof(restart_magic));
  AppendUInt32(header, restart_format_version);
  AppendUInt32(header, endian_marker);
  AppendUInt32(header, static_cast<std::uint32_t>(sizeof(Real)));
  AppendUInt32(header, static_cast<std::uint32_t>(sizeof(int)));
  AppendUInt32(header, static_cast<std::uint32_t>(npop));
  AppendUInt64(header, static_cast<std::uint64_t>(nmb));
  AppendUInt64(header, static_cast<std::uint64_t>(next_tag_));
  if (header.size() != restart_header_size) {
    RestartError("Internal particle restart header size is inconsistent.");
  }

  std::vector<char> population_entries;
  for (const auto &data : restart_data) {
    const ParticlePopulation *population = data.population;
    if (population->name.size() >= population_name_size) {
      RestartError("Particle population name is too long for the restart format.");
    }
    char name[population_name_size] = {};
    std::memcpy(name, population->name.data(), population->name.size());
    AppendBytes(population_entries, name, population_name_size);
    AppendUInt32(population_entries, ParticleTypeID(population->particle_type));
    AppendUInt32(population_entries,
                 static_cast<std::uint32_t>(population->RestartLayoutVersion()));
    AppendUInt32(population_entries, static_cast<std::uint32_t>(population->nrdata));
    AppendUInt32(population_entries, static_cast<std::uint32_t>(population->nidata));
    AppendUInt64(population_entries, static_cast<std::uint64_t>(data.metadata.size()));
  }

  IOWrapper file;
  file.Open(filename.c_str(), IOWrapper::FileMode::write);
  if (global_variable::my_rank == 0) {
    WriteBytesAt(file, header.data(), header.size(), 0);
    WriteBytesAt(file, population_entries.data(), population_entries.size(),
                 restart_header_size);
    for (const auto &data : restart_data) {
      std::vector<char> directory;
      for (const auto count : data.counts) {
        AppendUInt64(directory, count);
      }
      WriteBytesAt(file, directory.data(), directory.size(), data.directory_offset);
      WriteBytesAt(file, data.metadata.data(), data.metadata.size(), data.metadata_offset);
    }
  }

#if MPI_PARALLEL_ENABLED
  MPI_Barrier(MPI_COMM_WORLD);
#endif
  for (const auto &data : restart_data) {
    const ParticlePopulation *population = data.population;
    for (int gid=gids; gid<=gide; ++gid) {
      const std::vector<int> &order = data.ordered_particles[gid];
      if (order.empty()) continue;
      std::vector<Real> rbuffer(order.size()*population->nrdata);
      const int stored_nidata = population->nidata - 1;
      std::vector<int> ibuffer(order.size()*stored_nidata);
      for (std::size_t p=0; p<order.size(); ++p) {
        for (int n=0; n<population->nrdata; ++n) {
          rbuffer[p*population->nrdata + n] = data.rdata(n,order[p]);
        }
        int stored_n = 0;
        for (int n=0; n<population->nidata; ++n) {
          if (n == PGID) continue;
          ibuffer[p*stored_nidata + stored_n] = data.idata(n,order[p]);
          ++stored_n;
        }
      }
      WriteBytesAt(file, rbuffer.data(), rbuffer.size()*sizeof(Real),
                   data.blocks[gid].rdata_offset);
      WriteBytesAt(file, ibuffer.data(), ibuffer.size()*sizeof(int),
                   data.blocks[gid].idata_offset);
    }
  }
  file.Close();
}

//----------------------------------------------------------------------------------------
//! \fn Particles::LoadRestart
//! \brief Validate and restore particle populations owned by this rank.

void Particles::LoadRestart(const std::string &filename) {
  if (filename.empty()) {
    RestartError("A particle restart file must be specified with '-p <file>'.");
  }
  const std::uint64_t actual_file_size = FileSize(filename);
  IOWrapper file;
  file.Open(filename.c_str(), IOWrapper::FileMode::read);

  std::vector<char> header(restart_header_size);
  ReadBytesAt(file, header.data(), header.size(), 0);
  if (std::memcmp(header.data(), restart_magic, sizeof(restart_magic)) != 0) {
    RestartError("Particle restart magic value is invalid.");
  }
  std::size_t offset = sizeof(restart_magic);
  const std::uint32_t format_version = ReadUInt32(header, offset);
  const std::uint32_t stored_endian = ReadUInt32(header, offset);
  const std::uint32_t real_size = ReadUInt32(header, offset);
  const std::uint32_t int_size = ReadUInt32(header, offset);
  const std::uint32_t npop = ReadUInt32(header, offset);
  const std::uint64_t nmb = ReadUInt64(header, offset);
  const std::uint64_t stored_next_tag = ReadUInt64(header, offset);

  const Mesh *pm = pmy_pack_->pmesh;
  if (format_version != restart_format_version || stored_endian != endian_marker ||
      real_size != sizeof(Real) || int_size != sizeof(int)) {
    RestartError("Particle restart binary format is incompatible with this executable.");
  }
  if (nmb != static_cast<std::uint64_t>(pm->nmb_total)) {
    RestartError("Particle restart MeshBlock count does not match the current mesh.");
  }
  if (npop != populations_.size()) {
    RestartError("Particle restart population count does not match the input model.");
  }

  std::uint64_t entries_size = CheckedMultiply(npop, population_entry_size);
  std::vector<char> entries(entries_size);
  ReadBytesAt(file, entries.data(), entries.size(), restart_header_size);
  std::vector<PopulationRestartHeader> population_headers(npop);
  offset = 0;
  for (std::uint32_t p=0; p<npop; ++p) {
    PopulationRestartHeader &entry = population_headers[p];
    const char *name = entries.data() + offset;
    std::size_t name_length = 0;
    while (name_length < population_name_size && name[name_length] != '\0') ++name_length;
    if (name_length == population_name_size) {
      RestartError("Particle restart population name is not terminated.");
    }
    entry.name.assign(name, name_length);
    offset += population_name_size;
    entry.particle_type = ReadUInt32(entries, offset);
    entry.layout_version = ReadUInt32(entries, offset);
    entry.nrdata = ReadUInt32(entries, offset);
    entry.nidata = ReadUInt32(entries, offset);
    entry.metadata_size = ReadUInt64(entries, offset);

    ParticlePopulation *population = populations_[p];
    if (population->nidata <= 1) {
      RestartError("Particle layout does not contain the common integer fields.");
    }
    if (entry.name != population->name ||
        entry.particle_type != ParticleTypeID(population->particle_type) ||
        entry.layout_version !=
            static_cast<std::uint32_t>(population->RestartLayoutVersion()) ||
        entry.nrdata != static_cast<std::uint32_t>(population->nrdata) ||
        entry.nidata != static_cast<std::uint32_t>(population->nidata)) {
      RestartError("Particle restart population layout does not match the input model.");
    }
  }

  std::vector<std::vector<BlockRestartData>> directories(npop);
  std::uint64_t next_offset = CheckedAdd(
      restart_header_size, CheckedMultiply(npop, population_entry_size));
  for (std::uint32_t p=0; p<npop; ++p) {
    std::uint64_t directory_size = CheckedMultiply(nmb, block_entry_size);
    std::vector<char> directory(directory_size);
    ReadBytesAt(file, directory.data(), directory.size(), next_offset);
    directories[p].resize(nmb);
    std::size_t directory_offset = 0;
    for (std::uint64_t gid=0; gid<nmb; ++gid) {
      directories[p][gid].count = ReadUInt64(directory, directory_offset);
    }
    next_offset = CheckedAdd(next_offset, directory_size);
  }

  for (std::uint32_t p=0; p<npop; ++p) {
    const PopulationRestartHeader &entry = population_headers[p];
    if (next_offset > actual_file_size ||
        entry.metadata_size > actual_file_size - next_offset) {
      RestartError("Particle restart metadata size is invalid.");
    }
    std::vector<char> metadata(entry.metadata_size);
    ReadBytesAt(file, metadata.data(), metadata.size(), next_offset);
    populations_[p]->ValidateRestartMetadata(metadata);
    next_offset = CheckedAdd(next_offset, entry.metadata_size);
  }

  for (std::uint32_t p=0; p<npop; ++p) {
    ParticlePopulation *population = populations_[p];
    for (std::uint64_t gid=0; gid<nmb; ++gid) {
      BlockRestartData &block = directories[p][gid];
      block.rdata_offset = next_offset;
      std::uint64_t rsize = CheckedMultiply(block.count, population->nrdata);
      rsize = CheckedMultiply(rsize, sizeof(Real));
      next_offset = CheckedAdd(next_offset, rsize);
      block.idata_offset = next_offset;
      std::uint64_t isize = CheckedMultiply(block.count, population->nidata - 1);
      isize = CheckedMultiply(isize, sizeof(int));
      next_offset = CheckedAdd(next_offset, isize);
    }
  }
  if (next_offset != actual_file_size) {
    RestartError("Particle restart size does not match its counts and particle layout.");
  }

  const int gids = pmy_pack_->gids;
  const int gide = pmy_pack_->gide;
  for (std::uint32_t p=0; p<npop; ++p) {
    ParticlePopulation *population = populations_[p];
    std::uint64_t local_count = 0;
    for (int gid=gids; gid<=gide; ++gid) {
      local_count = CheckedAdd(local_count, directories[p][gid].count);
    }
    if (local_count > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
      RestartError("Local particle restart count exceeds the in-memory integer limit.");
    }

    const int npart = static_cast<int>(local_count);
    HostArray2D<Real> rdata("particle restart real input", population->nrdata, npart);
    HostArray2D<int> idata("particle restart int input", population->nidata, npart);
    int local_offset = 0;
    for (int gid=gids; gid<=gide; ++gid) {
      const BlockRestartData &block = directories[p][gid];
      const RegionSize &block_size = pmy_pack_->pmb->mb_size.h_view(gid - gids);
      if (block.count > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
        RestartError("MeshBlock particle count exceeds the in-memory integer limit.");
      }
      std::size_t block_count = static_cast<std::size_t>(block.count);
      std::vector<Real> rbuffer(block_count*population->nrdata);
      const int stored_nidata = population->nidata - 1;
      std::vector<int> ibuffer(block_count*stored_nidata);
      ReadBytesAt(file, rbuffer.data(), rbuffer.size()*sizeof(Real), block.rdata_offset);
      ReadBytesAt(file, ibuffer.data(), ibuffer.size()*sizeof(int), block.idata_offset);
      for (std::size_t q=0; q<block_count; ++q) {
        for (int n=0; n<population->nrdata; ++n) {
          rdata(n,local_offset) = rbuffer[q*population->nrdata + n];
        }
        idata(PGID,local_offset) = gid;
        int stored_n = 0;
        for (int n=0; n<population->nidata; ++n) {
          if (n == PGID) continue;
          idata(n,local_offset) = ibuffer[q*stored_nidata + stored_n];
          ++stored_n;
        }
        const int tag = idata(PTAG,local_offset);
        const int status = idata(PSTATUS,local_offset);
        const Real x1 = rdata(IPX,local_offset);
        const Real x2 = rdata(IPY,local_offset);
        const Real x3 = rdata(IPZ,local_offset);
        if (tag < 0) {
          RestartError("Particle restart contains a negative tag.");
        }
        if (status < PACTIVE || status > PDELETE_AFTER_SNAPSHOT) {
          RestartError("Particle restart contains an invalid lifecycle status.");
        }
        if (!PositionInMeshBlock(x1, x2, x3, block_size) &&
            !RetainedBeyondPhysicalBoundary(x1, x2, x3, status,
                                            pmy_pack_->pmesh)) {
          RestartError("Particle restart contains a position outside its MeshBlock "
                       "section (GID " + std::to_string(gid) + ").");
        }
        ++local_offset;
      }
    }

    population->nprtcl_thispack = npart;
    Kokkos::realloc(population->prtcl_rdata, population->nrdata, npart);
    Kokkos::realloc(population->prtcl_idata, population->nidata, npart);
    Kokkos::deep_copy(population->prtcl_rdata, rdata);
    Kokkos::deep_copy(population->prtcl_idata, idata);
  }

  const std::int64_t minimum_next_tag = NextTagFromParticles();
  const std::uint64_t max_next_tag =
      static_cast<std::uint64_t>(std::numeric_limits<int>::max()) + 1;
  if (stored_next_tag > max_next_tag ||
      stored_next_tag < static_cast<std::uint64_t>(minimum_next_tag)) {
    RestartError("Particle restart next-tag value is inconsistent with its particles.");
  }
  next_tag_ = static_cast<std::int64_t>(stored_next_tag);
  file.Close();
}

} // namespace particles
