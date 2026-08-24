//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file particle_track.cpp
//! \brief append-only binary histories for selected particles

#include <sys/stat.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "particles/particles.hpp"
#include "pgen/pgen.hpp"
#include "outputs.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

namespace {

constexpr int track_format_version = 1;
constexpr int track_block_marker_size = 16;
constexpr char track_block_header[] = "ATHKPARTBLOCK";
constexpr char track_block_footer[] = "ATHKPARTEND";

struct ParsedTagSelection {
  std::string canonical;
  ParticleTrackSelection mode;
  int start;
  int stop;
  int step;
  bool has_stop;
  std::vector<int> explicit_tags;
};

struct TrackFileSchema {
  std::string population_name;
  std::string particle_type;
  std::string tag_selection;
  std::vector<std::string> int_names;
  std::vector<std::string> real_names;
};

struct TrackFileHeader {
  int version;
  int real_size;
  int int_size;
  int population_name_size;
  int particle_type_size;
  int tag_selection_size;
  int int_names_size;
  int real_names_size;
  int n_int_fields;
  int n_real_fields;
};

std::string Trim(const std::string &text) {
  const std::size_t first = text.find_first_not_of(" \t\n\r");
  if (first == std::string::npos) return "";
  const std::size_t last = text.find_last_not_of(" \t\n\r");
  return text.substr(first, last-first+1);
}

[[noreturn]] void FatalTrackOutput(const std::string &message) {
  std::cout << "### FATAL ERROR in particle_track.cpp" << std::endl
            << message << std::endl;
#if MPI_PARALLEL_ENABLED
  MPI_Abort(MPI_COMM_WORLD, 1);
#endif
  std::exit(EXIT_FAILURE);
}

int ParseNonnegativeInteger(const std::string &text, const std::string &description) {
  const std::string value_text = Trim(text);
  if (value_text.empty()) {
    FatalTrackOutput(description + " is empty");
  }
  try {
    std::size_t parsed = 0;
    const long long value = std::stoll(value_text, &parsed);
    if (parsed != value_text.size() || value < 0 ||
        value > std::numeric_limits<int>::max()) {
      FatalTrackOutput(description + " must be a non-negative integer");
    }
    return static_cast<int>(value);
  } catch (...) {
    FatalTrackOutput(description + " must be a non-negative integer");
  }
}

std::vector<std::string> Split(const std::string &text, char delimiter) {
  std::vector<std::string> pieces;
  std::size_t first = 0;
  while (true) {
    const std::size_t next = text.find(delimiter, first);
    pieces.push_back(text.substr(first, next-first));
    if (next == std::string::npos) break;
    first = next+1;
  }
  return pieces;
}

ParsedTagSelection ParseTagSelection(const std::string &requested) {
  ParsedTagSelection parsed = {"all", ParticleTrackSelection::all,
                               0, 0, 1, false, {}};
  if (requested == "all") return parsed;

  if (requested.find(':') != std::string::npos) {
    if (requested.find(',') != std::string::npos) {
      FatalTrackOutput("particle tag lists and slices cannot be mixed");
    }
    const std::vector<std::string> pieces = Split(requested, ':');
    if (pieces.size() < 2 || pieces.size() > 3) {
      FatalTrackOutput("particle tag slice must have the form start:stop[:step]");
    }
    parsed.mode = ParticleTrackSelection::slice;
    if (!Trim(pieces[0]).empty()) {
      parsed.start = ParseNonnegativeInteger(pieces[0], "particle tag slice start");
    }
    if (!Trim(pieces[1]).empty()) {
      parsed.stop = ParseNonnegativeInteger(pieces[1], "particle tag slice stop");
      parsed.has_stop = true;
    }
    if (pieces.size() == 3 && !Trim(pieces[2]).empty()) {
      parsed.step = ParseNonnegativeInteger(pieces[2], "particle tag slice step");
    }
    if (parsed.step == 0) {
      FatalTrackOutput("particle tag slice step cannot be zero");
    }
    std::stringstream canonical;
    canonical << parsed.start << ":";
    if (parsed.has_stop) canonical << parsed.stop;
    canonical << ":" << parsed.step;
    parsed.canonical = canonical.str();
    return parsed;
  }

  parsed.mode = ParticleTrackSelection::list;
  for (const std::string &piece : Split(requested, ',')) {
    parsed.explicit_tags.push_back(
        ParseNonnegativeInteger(piece, "particle tag list entry"));
  }
  std::sort(parsed.explicit_tags.begin(), parsed.explicit_tags.end());
  parsed.explicit_tags.erase(
      std::unique(parsed.explicit_tags.begin(), parsed.explicit_tags.end()),
      parsed.explicit_tags.end());
  std::stringstream canonical;
  for (std::size_t n=0; n<parsed.explicit_tags.size(); ++n) {
    if (n > 0) canonical << ",";
    canonical << parsed.explicit_tags[n];
  }
  parsed.canonical = canonical.str();
  return parsed;
}

std::string JoinNames(const std::vector<std::string> &names) {
  std::string joined;
  for (std::size_t n=0; n<names.size(); ++n) {
    if (n > 0) joined += "\n";
    joined += names[n];
  }
  return joined;
}

TrackFileHeader MakeFileHeader(const TrackFileSchema &schema) {
  const std::string int_blob = JoinNames(schema.int_names);
  const std::string real_blob = JoinNames(schema.real_names);
  return {track_format_version, static_cast<int>(sizeof(Real)),
          static_cast<int>(sizeof(int)),
          static_cast<int>(schema.population_name.size()),
          static_cast<int>(schema.particle_type.size()),
          static_cast<int>(schema.tag_selection.size()),
          static_cast<int>(int_blob.size()), static_cast<int>(real_blob.size()),
          static_cast<int>(schema.int_names.size()),
          static_cast<int>(schema.real_names.size())};
}

void WriteFileHeader(FILE *file, const TrackFileSchema &schema) {
  char magic[32] = {};
  std::strncpy(magic, "ATHENAK_PARTICLE_TRACK", sizeof(magic)-1);
  const std::string int_blob = JoinNames(schema.int_names);
  const std::string real_blob = JoinNames(schema.real_names);
  const TrackFileHeader header = MakeFileHeader(schema);
  const int serialized_header[10] = {
      header.version, header.real_size, header.int_size,
      header.population_name_size, header.particle_type_size,
      header.tag_selection_size, header.int_names_size, header.real_names_size,
      header.n_int_fields, header.n_real_fields};
  if (std::fwrite(magic, sizeof(char), sizeof(magic), file) != sizeof(magic) ||
      std::fwrite(serialized_header, sizeof(int), 10, file) != 10 ||
      (!schema.population_name.empty() &&
       std::fwrite(schema.population_name.data(), sizeof(char),
                   schema.population_name.size(), file) != schema.population_name.size()) ||
      (!schema.particle_type.empty() &&
       std::fwrite(schema.particle_type.data(), sizeof(char),
                   schema.particle_type.size(), file) != schema.particle_type.size()) ||
      (!schema.tag_selection.empty() &&
       std::fwrite(schema.tag_selection.data(), sizeof(char),
                   schema.tag_selection.size(), file) != schema.tag_selection.size()) ||
      (!int_blob.empty() &&
       std::fwrite(int_blob.data(), sizeof(char), int_blob.size(), file) !=
       int_blob.size()) ||
      (!real_blob.empty() &&
       std::fwrite(real_blob.data(), sizeof(char), real_blob.size(), file) !=
       real_blob.size())) {
    FatalTrackOutput("failed to write particle track file header");
  }
}

std::string ReadString(FILE *file, int size, const std::string &description) {
  if (size < 0) FatalTrackOutput("invalid " + description + " size in file header");
  std::string value(size, '\0');
  if (size > 0 &&
      std::fread(value.data(), sizeof(char), size, file) != static_cast<std::size_t>(size)) {
    FatalTrackOutput("existing particle track file has a truncated " + description);
  }
  return value;
}

bool HeadersMatch(const TrackFileHeader &left, const TrackFileHeader &right) {
  return left.version == right.version &&
         left.real_size == right.real_size &&
         left.int_size == right.int_size &&
         left.population_name_size == right.population_name_size &&
         left.particle_type_size == right.particle_type_size &&
         left.tag_selection_size == right.tag_selection_size &&
         left.int_names_size == right.int_names_size &&
         left.real_names_size == right.real_names_size &&
         left.n_int_fields == right.n_int_fields &&
         left.n_real_fields == right.n_real_fields;
}

bool EndsWithBlockFooter(FILE *file, long file_size) {
  if (file_size < track_block_marker_size) return false;
  if (std::fseek(file, file_size-track_block_marker_size, SEEK_SET) != 0) {
    FatalTrackOutput("failed to inspect particle track output file");
  }
  char footer[track_block_marker_size];
  char expected[track_block_marker_size] = {};
  std::strncpy(expected, track_block_footer, sizeof(expected)-1);
  if (std::fread(footer, sizeof(char), sizeof(footer), file) != sizeof(footer)) {
    FatalTrackOutput("failed to inspect particle track output file");
  }
  return std::memcmp(footer, expected, sizeof(footer)) == 0;
}

std::uint64_t EnsureFileHeader(FILE *file, const TrackFileSchema &schema,
                               const std::string &filename) {
  const TrackFileHeader expected = MakeFileHeader(schema);
  const std::string int_blob = JoinNames(schema.int_names);
  const std::string real_blob = JoinNames(schema.real_names);
  if (std::fseek(file, 0, SEEK_END) != 0) {
    FatalTrackOutput("failed to seek in particle track output file");
  }
  const long file_size = std::ftell(file);
  if (file_size < 0) FatalTrackOutput("failed to inspect particle track output file");
  long header_end = 0;
  if (file_size == 0) {
    WriteFileHeader(file, schema);
  } else {
    if (std::fseek(file, 0, SEEK_SET) != 0) {
      FatalTrackOutput("failed to seek in particle track output file");
    }
    char magic[32];
    int serialized_header[10];
    if (std::fread(magic, sizeof(char), sizeof(magic), file) != sizeof(magic) ||
        std::fread(serialized_header, sizeof(int), 10, file) != 10) {
      FatalTrackOutput("existing particle track file has a truncated header");
    }
    if (std::strncmp(magic, "ATHENAK_PARTICLE_TRACK", 22) != 0) {
      FatalTrackOutput("existing particle track file has an unrecognized header");
    }
    const TrackFileHeader header = {
        serialized_header[0], serialized_header[1], serialized_header[2],
        serialized_header[3], serialized_header[4], serialized_header[5],
        serialized_header[6], serialized_header[7], serialized_header[8],
        serialized_header[9]};
    if (!HeadersMatch(header, expected)) {
      FatalTrackOutput("existing particle track file schema does not match this output");
    }
    if (ReadString(file, header.population_name_size, "population name") !=
            schema.population_name ||
        ReadString(file, header.particle_type_size, "particle type") !=
            schema.particle_type ||
        ReadString(file, header.tag_selection_size, "tag selection") !=
            schema.tag_selection ||
        ReadString(file, header.int_names_size, "integer field list") != int_blob ||
        ReadString(file, header.real_names_size, "real field list") != real_blob) {
      FatalTrackOutput("existing particle track file schema does not match this output");
    }
    header_end = std::ftell(file);
    if (header_end < 0) {
      FatalTrackOutput("failed to inspect particle track output file");
    }
    if (std::fseek(file, 0, SEEK_END) != 0) {
      FatalTrackOutput("failed to seek in particle track output file");
    }
  }
  const long position = std::ftell(file);
  if (position < 0) FatalTrackOutput("failed to inspect particle track output file");
  if (file_size > header_end && !EndsWithBlockFooter(file, position)) {
    std::cout << "### WARNING in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Particle track file '" << filename << "' ends with an incomplete "
              << "output block. The existing bytes will be preserved, and new output "
              << "will be appended after them." << std::endl;
    if (std::fseek(file, 0, SEEK_END) != 0) {
      FatalTrackOutput("failed to seek in particle track output file");
    }
  }
  return static_cast<std::uint64_t>(position);
}

void WriteBlockHeader(FILE *file, int nrecords, int int_per_record,
                      int real_per_record, int cycle, Real time) {
  char magic[track_block_marker_size] = {};
  std::strncpy(magic, track_block_header, sizeof(magic)-1);
  int header[5] = {track_format_version, nrecords, int_per_record,
                   real_per_record, cycle};
  if (std::fwrite(magic, sizeof(char), sizeof(magic), file) != sizeof(magic) ||
      std::fwrite(header, sizeof(int), 5, file) != 5 ||
      std::fwrite(&time, sizeof(Real), 1, file) != 1) {
    FatalTrackOutput("failed to write particle track block header");
  }
}

void WriteBlockFooter(FILE *file) {
  char magic[track_block_marker_size] = {};
  std::strncpy(magic, track_block_footer, sizeof(magic)-1);
  if (std::fwrite(magic, sizeof(char), sizeof(magic), file) != sizeof(magic)) {
    FatalTrackOutput("failed to write particle track block footer");
  }
}

void WarnAboutMissingTags(const particles::ParticlePopulation *population,
                          const std::vector<int> &requested_tags,
                          const std::string &block_name) {
  std::vector<int> local_present(requested_tags.size(), 0);
  std::vector<int> present(requested_tags.size(), 0);
  const int npart = population->nprtcl_thispack;
  HostArray1D<int> particle_tags("particle_track_startup_tags", npart);
  if (npart > 0) {
    auto tags = Kokkos::subview(population->prtcl_idata, static_cast<int>(PTAG),
                                Kokkos::ALL);
    Kokkos::deep_copy(particle_tags, tags);
    for (int p=0; p<npart; ++p) {
      auto found = std::lower_bound(requested_tags.begin(), requested_tags.end(),
                                    particle_tags(p));
      if (found != requested_tags.end() && *found == particle_tags(p)) {
        local_present[found-requested_tags.begin()] = 1;
      }
    }
  }
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(local_present.data(), present.data(),
                static_cast<int>(requested_tags.size()), MPI_INT, MPI_MAX,
                MPI_COMM_WORLD);
#else
  present = local_present;
#endif

  std::vector<int> missing;
  for (std::size_t n=0; n<requested_tags.size(); ++n) {
    if (present[n] == 0) missing.push_back(requested_tags[n]);
  }
  if (global_variable::my_rank != 0 || missing.empty()) return;

  constexpr std::size_t max_print = 16;
  std::cout << "### WARNING in " << __FILE__ << " at line " << __LINE__ << std::endl
            << "Particle track output '" << block_name
            << "' requested tags not present at startup: ";
  for (std::size_t n=0; n<std::min(missing.size(), max_print); ++n) {
    if (n > 0) std::cout << ",";
    std::cout << missing[n];
  }
  if (missing.size() > max_print) {
    std::cout << " (and " << missing.size()-max_print << " more)";
  }
  std::cout << ". They will be recorded if they appear later." << std::endl;
}

KOKKOS_INLINE_FUNCTION
bool TagIsSelected(int tag, ParticleTrackSelection mode, int slice_start, int slice_stop,
                   int slice_step, bool slice_has_stop,
                   DvceArray1D<int> selected_tags) {
  if (mode == ParticleTrackSelection::all) return true;
  if (mode == ParticleTrackSelection::slice) {
    return tag >= slice_start && (!slice_has_stop || tag < slice_stop) &&
           ((tag-slice_start) % slice_step == 0);
  }
  int lower = 0;
  int upper = selected_tags.extent_int(0);
  while (lower < upper) {
    const int middle = lower + (upper-lower)/2;
    if (selected_tags(middle) < tag) {
      lower = middle+1;
    } else {
      upper = middle;
    }
  }
  return lower < selected_tags.extent_int(0) && selected_tags(lower) == tag;
}

} // namespace

//----------------------------------------------------------------------------------------
// ctor

ParticleTrackOutput::ParticleTrackOutput(ParameterInput *pin, Mesh *pm,
                                         OutputParameters op) :
    BaseTypeOutput(pin, pm, op),
    population_name(pin->GetOrAddString(op.block_name, "population", "particles")),
    selection_mode(ParticleTrackSelection::all),
    slice_start(0),
    slice_stop(0),
    slice_step(1),
    slice_has_stop(false),
    npout_thisrank(0),
    npout_total(0) {
  mkdir("particle_track", 0775);
  if (pm->pmb_pack->ppart == nullptr) {
    FatalTrackOutput("file_type=particle_track requires particles");
  }
  particles::ParticlePopulation *population =
      pm->pmb_pack->ppart->FindPopulation(population_name);
  if (population == nullptr) {
    FatalTrackOutput("particle population '" + population_name + "' was not found");
  }

  const ParsedTagSelection parsed = ParseTagSelection(
      Trim(pin->GetOrAddString(op.block_name, "tags", "all")));
  tag_selection = parsed.canonical;
  selection_mode = parsed.mode;
  slice_start = parsed.start;
  slice_stop = parsed.stop;
  slice_step = parsed.step;
  slice_has_stop = parsed.has_stop;
  explicit_tags = parsed.explicit_tags;
  if (selection_mode == ParticleTrackSelection::list) {
    Kokkos::realloc(selected_tags, explicit_tags.size());
    auto host_tags = Kokkos::create_mirror_view(selected_tags);
    for (std::size_t n=0; n<explicit_tags.size(); ++n) {
      host_tags(n) = explicit_tags[n];
    }
    Kokkos::deep_copy(selected_tags, host_tags);
  }

  int_fields.push_back(PTAG);
  for (int n=0; n<population->nidata; ++n) {
    if (n != PTAG && population->int_output[n]) int_fields.push_back(n);
  }
  for (int n=0; n<population->nrdata; ++n) {
    if (population->real_output[n]) real_fields.push_back(n);
  }
  if (pm->pgen != nullptr) {
    for (const auto &variable : pm->pgen->user_particle_output_variables) {
      for (int field : int_fields) {
        if (variable.name == population->int_names[field]) {
          FatalTrackOutput("user particle output variable '" + variable.name +
                           "' conflicts with a stored particle field");
        }
      }
      for (int field : real_fields) {
        if (variable.name == population->real_names[field]) {
          FatalTrackOutput("user particle output variable '" + variable.name +
                           "' conflicts with a stored particle field");
        }
      }
      user_real_names.push_back(variable.name);
    }
  }
  npout_eachrank.resize(global_variable::nranks);

  // Missing explicit tags remain valid and will be written if they are created later.
  if (selection_mode == ParticleTrackSelection::list) {
    WarnAboutMissingTags(population, explicit_tags, op.block_name);
  }
}

//----------------------------------------------------------------------------------------
// Load selected particle fields into host arrays.

void ParticleTrackOutput::LoadOutputData(Mesh *pm) {
  particles::ParticlePopulation *population =
      pm->pmb_pack->ppart->FindPopulation(population_name);
  const int npart = population->nprtcl_thispack;
  auto pi = population->prtcl_idata;
  auto pr = population->prtcl_rdata;
  const ParticleTrackSelection mode = selection_mode;
  const int start = slice_start;
  const int stop = slice_stop;
  const int step = slice_step;
  const bool has_stop = slice_has_stop;
  auto tags = selected_tags;

  Kokkos::parallel_reduce(
      "particle_track_count", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
      KOKKOS_LAMBDA(const int p, int &count) {
        if (TagIsSelected(pi(PTAG,p), mode, start, stop, step, has_stop, tags)) ++count;
      }, npout_thisrank);

  npout_eachrank[global_variable::my_rank] = npout_thisrank;
#if MPI_PARALLEL_ENABLED
  MPI_Allgather(&npout_thisrank, 1, MPI_INT, npout_eachrank.data(), 1, MPI_INT,
                MPI_COMM_WORLD);
#endif
  npout_total = 0;
  for (int count : npout_eachrank) npout_total += count;

  Kokkos::realloc(outpart_idata, population->nidata, npout_thisrank);
  Kokkos::realloc(outpart_rdata, population->nrdata, npout_thisrank);
  Kokkos::realloc(outpart_user_rdata, user_real_names.size(), npout_thisrank);

  DvceArray2D<int> device_idata("particle_track_idata", population->nidata,
                                npout_thisrank);
  DvceArray2D<Real> device_rdata("particle_track_rdata", population->nrdata,
                                 npout_thisrank);
  const int ni = population->nidata;
  const int nr = population->nrdata;
  int ngathered = 0;
  if (npout_thisrank > 0) {
    Kokkos::parallel_scan(
        "particle_track_gather", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
        KOKKOS_LAMBDA(const int p, int &offset, const bool final) {
          if (TagIsSelected(pi(PTAG,p), mode, start, stop, step, has_stop, tags)) {
            if (final) {
              for (int n=0; n<ni; ++n) device_idata(n,offset) = pi(n,p);
              for (int n=0; n<nr; ++n) device_rdata(n,offset) = pr(n,p);
            }
            ++offset;
          }
        }, ngathered);
  }
  if (ngathered != npout_thisrank) {
    FatalTrackOutput("particle selection changed while gathering output data");
  }

  const std::size_t registered_user_fields = (pm->pgen == nullptr) ? 0 :
      pm->pgen->user_particle_output_variables.size();
  if (registered_user_fields != user_real_names.size()) {
    FatalTrackOutput("user particle output registry changed after output setup");
  }
  if (!user_real_names.empty()) {
    DvceArray2D<Real> device_user_rdata(
        "particle_track_user_rdata", user_real_names.size(), npout_thisrank);
    if (npout_thisrank > 0) {
      Kokkos::deep_copy(device_user_rdata,
                        std::numeric_limits<Real>::quiet_NaN());
    }
    for (std::size_t n=0; n<user_real_names.size(); ++n) {
      const auto &variable = pm->pgen->user_particle_output_variables[n];
      if (variable.name != user_real_names[n] || variable.function == nullptr) {
        FatalTrackOutput("user particle output registry changed after output setup");
      }
      particles::ParticleOutputData output(
          pm->pmb_pack, population, device_rdata, device_idata,
          device_user_rdata, static_cast<int>(n), npout_thisrank);
      variable.function(&output);
    }
    if (npout_thisrank > 0) {
      Kokkos::deep_copy(outpart_user_rdata, device_user_rdata);
    }
  }
  if (npout_thisrank > 0) {
    Kokkos::deep_copy(outpart_idata, device_idata);
    Kokkos::deep_copy(outpart_rdata, device_rdata);
  }
}

//----------------------------------------------------------------------------------------
// Append one time/cycle block to the shared particle track file.

void ParticleTrackOutput::WriteOutputFile(Mesh *pm, ParameterInput *pin) {
  const particles::ParticlePopulation *population =
      pm->pmb_pack->ppart->FindPopulation(population_name);
  std::vector<std::string> int_names;
  std::vector<std::string> real_names;
  for (int field : int_fields) int_names.emplace_back(population->int_names[field]);
  for (int field : real_fields) real_names.emplace_back(population->real_names[field]);
  real_names.insert(real_names.end(), user_real_names.begin(), user_real_names.end());

  const int int_per_record = static_cast<int>(int_fields.size());
  const int real_per_record =
      static_cast<int>(real_fields.size() + user_real_names.size());
  std::vector<int> local_ints(
      std::max(static_cast<int>(npout_thisrank*int_per_record), 1));
  std::vector<Real> local_reals(
      std::max(static_cast<int>(npout_thisrank*real_per_record), 1));
  for (int p=0; p<npout_thisrank; ++p) {
    for (int n=0; n<int_per_record; ++n) {
      local_ints[int_per_record*p+n] = outpart_idata(int_fields[n],p);
    }
    for (int n=0; n<real_per_record; ++n) {
      if (n < static_cast<int>(real_fields.size())) {
        local_reals[real_per_record*p+n] = outpart_rdata(real_fields[n],p);
      } else {
        local_reals[real_per_record*p+n] =
            outpart_user_rdata(n-static_cast<int>(real_fields.size()),p);
      }
    }
  }

  const std::string filename = "particle_track/" + out_params.file_basename + "." +
                               out_params.file_id + ".part_track";
  std::uint64_t block_offset = 0;
  if (global_variable::my_rank == 0) {
    FILE *file = std::fopen(filename.c_str(), "ab+");
    if (file == nullptr) {
      FatalTrackOutput("output file '" + filename + "' could not be opened");
    }
    const TrackFileSchema schema = {
        population_name, population->type_name, tag_selection, int_names, real_names};
    block_offset = EnsureFileHeader(file, schema, filename);
    WriteBlockHeader(file, npout_total, int_per_record, real_per_record,
                     pm->ncycle, pm->time);
    if (std::fclose(file) != 0) {
      FatalTrackOutput("failed to close particle track output file");
    }
  }
#if MPI_PARALLEL_ENABLED
  unsigned long long shared_offset = block_offset;
  MPI_Bcast(&shared_offset, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
  block_offset = shared_offset;
#endif

  int rank_offset = 0;
  for (int rank=0; rank<global_variable::my_rank; ++rank) {
    rank_offset += npout_eachrank[rank];
  }
  const IOWrapperSizeT block_header_size = 16 + 5*sizeof(int) + sizeof(Real);
  const IOWrapperSizeT integer_data_offset = block_offset + block_header_size;
  const IOWrapperSizeT real_data_offset = integer_data_offset +
      static_cast<IOWrapperSizeT>(npout_total)*int_per_record*sizeof(int);

  IOWrapper file;
  file.Open(filename.c_str(), IOWrapper::FileMode::read_write);
  const IOWrapperSizeT my_int_offset = integer_data_offset +
      static_cast<IOWrapperSizeT>(rank_offset)*int_per_record*sizeof(int);
  const IOWrapperSizeT my_real_offset = real_data_offset +
      static_cast<IOWrapperSizeT>(rank_offset)*real_per_record*sizeof(Real);
  const IOWrapperSizeT local_int_count =
      static_cast<IOWrapperSizeT>(npout_thisrank)*int_per_record;
  const IOWrapperSizeT local_real_count =
      static_cast<IOWrapperSizeT>(npout_thisrank)*real_per_record;
  const bool ints_written =
      file.Write_any_type_at_all(local_ints.data(), local_int_count,
                                 my_int_offset, "int") == local_int_count;
  const bool reals_written =
      file.Write_any_type_at_all(local_reals.data(), local_real_count,
                                 my_real_offset, "Real") == local_real_count;
  bool write_succeeded = ints_written && reals_written;
#if MPI_PARALLEL_ENABLED
  int write_status = write_succeeded ? 1 : 0;
  MPI_Allreduce(MPI_IN_PLACE, &write_status, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  write_succeeded = (write_status == 1);
#endif
  if (!write_succeeded) {
    FatalTrackOutput("particle data were not written correctly to '" + filename + "'");
  }
  bool close_succeeded = (file.Close() == 0);
#if MPI_PARALLEL_ENABLED
  int close_status = close_succeeded ? 1 : 0;
  MPI_Allreduce(MPI_IN_PLACE, &close_status, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  close_succeeded = (close_status == 1);
#endif
  if (!close_succeeded) {
    FatalTrackOutput("failed to close particle track output file");
  }

  if (global_variable::my_rank == 0) {
    FILE *footer_file = std::fopen(filename.c_str(), "ab");
    if (footer_file == nullptr) {
      FatalTrackOutput("output file '" + filename + "' could not be opened");
    }
    WriteBlockFooter(footer_file);
    if (std::fclose(footer_file) != 0) {
      FatalTrackOutput("failed to close particle track output file");
    }
  }
#if MPI_PARALLEL_ENABLED
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  out_params.file_number++;
  if (out_params.last_time < 0.0) {
    out_params.last_time = pm->time;
  } else {
    out_params.last_time += out_params.dt;
  }
  pin->SetInteger(out_params.block_name, "file_number", out_params.file_number);
  pin->SetReal(out_params.block_name, "last_time", out_params.last_time);
}
