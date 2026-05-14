//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file z4c_speck_cart_reader.cpp
//! \brief Initialize Z4c from a SpECK/AthenaK Cartesian GH data dump.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <hdf5.h>

#include "athena.hpp"
#include "coordinates/adm.hpp"
#include "coordinates/cell_locations.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "pgen/pgen.hpp"
#include "z4c/z4c.hpp"

namespace {

struct BinaryCartMetadata {
  int cycle;
  float time;
  float center[3];
  float extent[3];
  int numpoints[3];
  bool is_cheb;
  int noutvars;
};

struct CartMetadata {
  int cycle = 0;
  Real time = 0.0;
  Real center[3] = {0.0, 0.0, 0.0};
  Real extent[3] = {0.0, 0.0, 0.0};
  int numpoints[3] = {0, 0, 0};
  bool is_cheb = false;
  int noutvars = 0;
};

struct CartGridDeviceSpec {
  Real center[3];
  Real extent[3];
  int numpoints[3];
  bool is_cheb;
};

struct GhIndexMap {
  int psi[4][4];
  int pi[4][4];
  int phi[3][4][4];
};

struct ConstraintSummary {
  Real c_rms = 0.0;
  Real h_rms = 0.0;
  Real m_rms = 0.0;
  Real z_rms = 0.0;
  Real volume = 0.0;
};

std::string SymmetricLabel(const std::string &prefix, const int a,
                           const int b) {
  return prefix + std::to_string(a) + std::to_string(b);
}

std::vector<std::string> SplitLabels(const std::string &labels) {
  std::istringstream stream(labels);
  std::vector<std::string> result;
  std::string word;
  while (stream >> word) {
    result.push_back(word);
  }
  return result;
}

int FindLabel(const std::unordered_map<std::string, int> &labels,
              const std::string &label) {
  const auto it = labels.find(label);
  return it == labels.end() ? -1 : it->second;
}

GhIndexMap BuildGhIndexMap(const std::vector<std::string> &labels) {
  std::unordered_map<std::string, int> by_name;
  for (int n = 0; n < static_cast<int>(labels.size()); ++n) {
    by_name[labels[n]] = n;
  }

  GhIndexMap map{};
  for (int a = 0; a < 4; ++a) {
    for (int b = 0; b < 4; ++b) {
      map.psi[a][b] = -1;
      map.pi[a][b] = -1;
      for (int d = 0; d < 3; ++d) {
        map.phi[d][a][b] = -1;
      }
    }
  }

  for (int a = 0; a < 4; ++a) {
    for (int b = a; b < 4; ++b) {
      const int psi = FindLabel(by_name, SymmetricLabel("psi", a, b));
      const int pi = FindLabel(by_name, SymmetricLabel("pi", a, b));
      map.psi[a][b] = map.psi[b][a] = psi;
      map.pi[a][b] = map.pi[b][a] = pi;
      for (int d = 0; d < 3; ++d) {
        const int phi =
            FindLabel(by_name, "phi" + std::to_string(d) +
                                   SymmetricLabel("_", a, b));
        map.phi[d][a][b] = map.phi[d][b][a] = phi;
      }
    }
  }

  std::vector<std::string> missing;
  for (int a = 0; a < 4; ++a) {
    for (int b = a; b < 4; ++b) {
      if (map.psi[a][b] < 0) {
        missing.push_back(SymmetricLabel("psi", a, b));
      }
      if (map.pi[a][b] < 0) {
        missing.push_back(SymmetricLabel("pi", a, b));
      }
      for (int d = 0; d < 3; ++d) {
        if (map.phi[d][a][b] < 0) {
          missing.push_back("phi" + std::to_string(d) +
                            SymmetricLabel("_", a, b));
        }
      }
    }
  }
  if (!missing.empty()) {
    std::ostringstream message;
    message << "SpECK cart GH input is missing required label(s):";
    const int count = std::min<int>(static_cast<int>(missing.size()), 12);
    for (int n = 0; n < count; ++n) {
      message << ' ' << missing[n];
    }
    if (static_cast<int>(missing.size()) > count) {
      message << " ...";
    }
    throw std::runtime_error(message.str());
  }
  return map;
}

KOKKOS_INLINE_FUNCTION Real CartCoordinate(const CartGridDeviceSpec &grid,
                                           const int dim, const int index) {
  if (grid.is_cheb) {
    constexpr Real pi = 3.141592653589793238462643383279502884;
    return grid.center[dim] +
           grid.extent[dim] *
               Kokkos::cos(static_cast<Real>(index) * pi /
                           static_cast<Real>(grid.numpoints[dim] - 1));
  }
  const Real min_x = grid.center[dim] - grid.extent[dim];
  const Real max_x = grid.center[dim] + grid.extent[dim];
  return min_x + static_cast<Real>(index) * (max_x - min_x) /
                     static_cast<Real>(grid.numpoints[dim] - 1);
}

KOKKOS_INLINE_FUNCTION void LocateCartesianIndex(const CartGridDeviceSpec &grid,
                                                 const int dim,
                                                 const Real x, int &i0,
                                                 Real &weight,
                                                 int &out_of_domain) {
  const int n = grid.numpoints[dim];
  const Real x_left = CartCoordinate(grid, dim, 0);
  const Real x_right = CartCoordinate(grid, dim, n - 1);
  const Real x_min = x_left < x_right ? x_left : x_right;
  const Real x_max = x_left < x_right ? x_right : x_left;
  constexpr Real tol_factor =
      sizeof(Real) == sizeof(float) ? static_cast<Real>(1.0e-5)
                                    : static_cast<Real>(1.0e-6);
  const Real tol =
      tol_factor * Kokkos::fmax(static_cast<Real>(1.0),
                                Kokkos::fmax(Kokkos::abs(x_min),
                                             Kokkos::abs(x_max)));
  Real xc = x;
  if (xc < x_min - tol || xc > x_max + tol) {
    out_of_domain = 1;
  }
  if (xc < x_min) {
    xc = x_min;
  }
  if (xc > x_max) {
    xc = x_max;
  }

  if (!grid.is_cheb) {
    const Real min_x = grid.center[dim] - grid.extent[dim];
    const Real max_x = grid.center[dim] + grid.extent[dim];
    const Real scaled =
        (xc - min_x) * static_cast<Real>(n - 1) / (max_x - min_x);
    int lo = static_cast<int>(Kokkos::floor(scaled));
    if (lo < 0) {
      lo = 0;
    }
    if (lo > n - 2) {
      lo = n - 2;
    }
    i0 = lo;
    weight = scaled - static_cast<Real>(lo);
    if (weight < 0.0) {
      weight = 0.0;
    }
    if (weight > 1.0) {
      weight = 1.0;
    }
    return;
  }

  i0 = 0;
  weight = 0.0;
  for (int q = 0; q < n - 1; ++q) {
    const Real x0 = CartCoordinate(grid, dim, q);
    const Real x1 = CartCoordinate(grid, dim, q + 1);
    const Real lo = x0 < x1 ? x0 : x1;
    const Real hi = x0 < x1 ? x1 : x0;
    if (xc >= lo - tol && xc <= hi + tol) {
      i0 = q;
      weight = (xc - x0) / (x1 - x0);
      if (weight < 0.0) {
        weight = 0.0;
      }
      if (weight > 1.0) {
        weight = 1.0;
      }
      return;
    }
  }
  i0 = n - 2;
  weight = 1.0;
}

KOKKOS_INLINE_FUNCTION Real InterpolateCartValue(
    const DvceArray4D<Real> &data, const CartGridDeviceSpec &grid,
    const int var, const Real x, const Real y, const Real z,
    int &out_of_domain) {
  int i0 = 0;
  int j0 = 0;
  int k0 = 0;
  Real wx = 0.0;
  Real wy = 0.0;
  Real wz = 0.0;
  LocateCartesianIndex(grid, 0, x, i0, wx, out_of_domain);
  LocateCartesianIndex(grid, 1, y, j0, wy, out_of_domain);
  LocateCartesianIndex(grid, 2, z, k0, wz, out_of_domain);

  Real value = 0.0;
  for (int dk = 0; dk < 2; ++dk) {
    const Real wk = dk == 0 ? 1.0 - wz : wz;
    for (int dj = 0; dj < 2; ++dj) {
      const Real wj = dj == 0 ? 1.0 - wy : wy;
      for (int di = 0; di < 2; ++di) {
        const Real wi = di == 0 ? 1.0 - wx : wx;
        value += wk * wj * wi * data(var, k0 + dk, j0 + dj, i0 + di);
      }
    }
  }
  return value;
}

KOKKOS_INLINE_FUNCTION Real GhValue(const DvceArray4D<Real> &data,
                                    const CartGridDeviceSpec &grid,
                                    const int var, const Real x,
                                    const Real y, const Real z,
                                    int &out_of_domain) {
  return InterpolateCartValue(data, grid, var, x, y, z, out_of_domain);
}

KOKKOS_INLINE_FUNCTION void FillSymmetricMatrixFromSpatialMetric(
    const Real gamma[6], Real matrix[3][3]) {
  matrix[0][0] = gamma[0];
  matrix[0][1] = matrix[1][0] = gamma[1];
  matrix[0][2] = matrix[2][0] = gamma[2];
  matrix[1][1] = gamma[3];
  matrix[1][2] = matrix[2][1] = gamma[4];
  matrix[2][2] = gamma[5];
}

bool EndsWith(const std::string &value, const std::string &suffix) {
  return value.size() >= suffix.size() &&
         value.compare(value.size() - suffix.size(), suffix.size(), suffix) ==
             0;
}

void RequireHdf5(const herr_t status, const std::string &action) {
  if (status < 0) {
    throw std::runtime_error("HDF5 failure while " + action);
  }
}

std::size_t Hdf5ElementCount(const hid_t space, const std::string &path) {
  const hssize_t count = H5Sget_simple_extent_npoints(space);
  if (count < 0) {
    throw std::runtime_error("failed to get HDF5 element count for " + path);
  }
  return static_cast<std::size_t>(count);
}

std::vector<int> ReadHdf5Ints(const hid_t file, const std::string &path,
                              const int expected_count) {
  const hid_t dataset = H5Dopen2(file, path.c_str(), H5P_DEFAULT);
  if (dataset < 0) {
    throw std::runtime_error("failed to open HDF5 dataset " + path);
  }
  const hid_t space = H5Dget_space(dataset);
  if (space < 0) {
    H5Dclose(dataset);
    throw std::runtime_error("failed to get HDF5 dataspace " + path);
  }
  if (Hdf5ElementCount(space, path) !=
      static_cast<std::size_t>(expected_count)) {
    H5Sclose(space);
    H5Dclose(dataset);
    throw std::runtime_error("unexpected HDF5 element count for " + path);
  }
  std::vector<int> values(static_cast<std::size_t>(expected_count));
  RequireHdf5(H5Dread(dataset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                      values.data()),
              "reading dataset " + path);
  RequireHdf5(H5Sclose(space), "closing dataspace " + path);
  RequireHdf5(H5Dclose(dataset), "closing dataset " + path);
  return values;
}

std::vector<double> ReadHdf5Doubles(const hid_t file, const std::string &path,
                                    const int expected_count) {
  const hid_t dataset = H5Dopen2(file, path.c_str(), H5P_DEFAULT);
  if (dataset < 0) {
    throw std::runtime_error("failed to open HDF5 dataset " + path);
  }
  const hid_t space = H5Dget_space(dataset);
  if (space < 0) {
    H5Dclose(dataset);
    throw std::runtime_error("failed to get HDF5 dataspace " + path);
  }
  if (Hdf5ElementCount(space, path) !=
      static_cast<std::size_t>(expected_count)) {
    H5Sclose(space);
    H5Dclose(dataset);
    throw std::runtime_error("unexpected HDF5 element count for " + path);
  }
  std::vector<double> values(static_cast<std::size_t>(expected_count));
  RequireHdf5(H5Dread(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
                      H5P_DEFAULT, values.data()),
              "reading dataset " + path);
  RequireHdf5(H5Sclose(space), "closing dataspace " + path);
  RequireHdf5(H5Dclose(dataset), "closing dataset " + path);
  return values;
}

std::string ReadHdf5LabelString(const hid_t file) {
  const hid_t dataset = H5Dopen2(file, "/labels", H5P_DEFAULT);
  if (dataset < 0) {
    throw std::runtime_error("failed to open HDF5 dataset /labels");
  }
  const hid_t space = H5Dget_space(dataset);
  if (space < 0) {
    H5Dclose(dataset);
    throw std::runtime_error("failed to get HDF5 dataspace /labels");
  }
  if (H5Sget_simple_extent_ndims(space) != 1) {
    H5Sclose(space);
    H5Dclose(dataset);
    throw std::runtime_error("SpECK HDF5 cart /labels must be rank 1");
  }
  const std::size_t length = Hdf5ElementCount(space, "/labels");
  std::string labels(static_cast<std::size_t>(length), '\0');
  RequireHdf5(H5Dread(dataset, H5T_NATIVE_CHAR, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                      labels.data()),
              "reading dataset /labels");
  RequireHdf5(H5Sclose(space), "closing dataspace /labels");
  RequireHdf5(H5Dclose(dataset), "closing dataset /labels");
  return labels;
}

void ValidateCartMetadata(const std::string &filename,
                          const CartMetadata &metadata) {
  if (metadata.noutvars <= 0 || metadata.numpoints[0] < 2 ||
      metadata.numpoints[1] < 2 || metadata.numpoints[2] < 2) {
    throw std::runtime_error("invalid SpECK cart metadata in: " + filename);
  }
}

void ReadSpeckCartBinaryFile(const std::string &filename,
                             CartMetadata &metadata,
                             std::vector<std::string> &labels,
                             DvceArray4D<Real> &device_data) {
  std::ifstream input(filename, std::ios::binary);
  if (!input) {
    throw std::runtime_error("failed to open SpECK cart input file: " +
                             filename);
  }
  BinaryCartMetadata binary_metadata{};
  input.read(reinterpret_cast<char *>(&binary_metadata),
             sizeof(binary_metadata));
  if (!input) {
    throw std::runtime_error("failed to read SpECK cart metadata from: " +
                             filename);
  }
  metadata.cycle = binary_metadata.cycle;
  metadata.time = static_cast<Real>(binary_metadata.time);
  metadata.is_cheb = binary_metadata.is_cheb;
  metadata.noutvars = binary_metadata.noutvars;
  for (int d = 0; d < 3; ++d) {
    metadata.center[d] = static_cast<Real>(binary_metadata.center[d]);
    metadata.extent[d] = static_cast<Real>(binary_metadata.extent[d]);
    metadata.numpoints[d] = binary_metadata.numpoints[d];
  }
  ValidateCartMetadata(filename, metadata);

  int label_length = 0;
  input.read(reinterpret_cast<char *>(&label_length), sizeof(int));
  if (!input || label_length <= 0) {
    throw std::runtime_error("invalid SpECK cart label string in: " +
                             filename);
  }
  std::string label_text(static_cast<std::size_t>(label_length), '\0');
  input.read(label_text.data(), label_length);
  labels = SplitLabels(label_text);
  if (static_cast<int>(labels.size()) != metadata.noutvars) {
    throw std::runtime_error("SpECK cart label count does not match metadata");
  }

  const int nx = metadata.numpoints[0];
  const int ny = metadata.numpoints[1];
  const int nz = metadata.numpoints[2];
  HostArray4D<Real> host_data("speck_cart_host", metadata.noutvars, nz, ny,
                              nx);
  for (int v = 0; v < metadata.noutvars; ++v) {
    for (int k = 0; k < nz; ++k) {
      for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
          float raw = 0.0F;
          input.read(reinterpret_cast<char *>(&raw), sizeof(float));
          if (!input) {
            throw std::runtime_error("truncated SpECK cart payload in: " +
                                     filename);
          }
          host_data(v, k, j, i) = static_cast<Real>(raw);
        }
      }
    }
  }
  device_data = DvceArray4D<Real>("speck_cart_device", metadata.noutvars, nz,
                                  ny, nx);
  Kokkos::deep_copy(device_data, host_data);
}

void ReadSpeckCartHdf5File(const std::string &filename,
                           CartMetadata &metadata,
                           std::vector<std::string> &labels,
                           DvceArray4D<Real> &device_data) {
  const hid_t file = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  if (file < 0) {
    throw std::runtime_error("failed to open SpECK HDF5 cart input file: " +
                             filename);
  }
  try {
    metadata.cycle = ReadHdf5Ints(file, "/cycle", 1)[0];
    metadata.time = static_cast<Real>(ReadHdf5Doubles(file, "/time", 1)[0]);
    const std::vector<double> center = ReadHdf5Doubles(file, "/center", 3);
    const std::vector<double> extent = ReadHdf5Doubles(file, "/extent", 3);
    const std::vector<int> npts = ReadHdf5Ints(file, "/numpoints", 3);
    metadata.is_cheb = ReadHdf5Ints(file, "/is_cheb", 1)[0] != 0;
    metadata.noutvars = ReadHdf5Ints(file, "/noutvars", 1)[0];
    for (int d = 0; d < 3; ++d) {
      metadata.center[d] = static_cast<Real>(center[d]);
      metadata.extent[d] = static_cast<Real>(extent[d]);
      metadata.numpoints[d] = npts[d];
    }
    ValidateCartMetadata(filename, metadata);
    labels = SplitLabels(ReadHdf5LabelString(file));
    if (static_cast<int>(labels.size()) != metadata.noutvars) {
      throw std::runtime_error(
          "SpECK HDF5 cart label count does not match metadata");
    }

    const hid_t dataset = H5Dopen2(file, "/data", H5P_DEFAULT);
    if (dataset < 0) {
      throw std::runtime_error("failed to open HDF5 dataset /data");
    }
    const hid_t space = H5Dget_space(dataset);
    if (space < 0) {
      H5Dclose(dataset);
      throw std::runtime_error("failed to get HDF5 dataspace /data");
    }
    hsize_t dims[4] = {0, 0, 0, 0};
    if (H5Sget_simple_extent_ndims(space) != 4) {
      H5Sclose(space);
      H5Dclose(dataset);
      throw std::runtime_error("SpECK HDF5 cart /data must be rank 4");
    }
    RequireHdf5(H5Sget_simple_extent_dims(space, dims, nullptr),
                "reading /data extent");
    if (dims[0] != static_cast<hsize_t>(metadata.noutvars) ||
        dims[1] != static_cast<hsize_t>(metadata.numpoints[2]) ||
        dims[2] != static_cast<hsize_t>(metadata.numpoints[1]) ||
        dims[3] != static_cast<hsize_t>(metadata.numpoints[0])) {
      H5Sclose(space);
      H5Dclose(dataset);
      throw std::runtime_error(
          "SpECK HDF5 cart /data shape does not match metadata");
    }
    const std::size_t count =
        static_cast<std::size_t>(dims[0] * dims[1] * dims[2] * dims[3]);
    std::vector<double> raw(count);
    RequireHdf5(H5Dread(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
                        H5P_DEFAULT, raw.data()),
                "reading dataset /data");
    RequireHdf5(H5Sclose(space), "closing dataspace /data");
    RequireHdf5(H5Dclose(dataset), "closing dataset /data");

    const int nx = metadata.numpoints[0];
    const int ny = metadata.numpoints[1];
    const int nz = metadata.numpoints[2];
    HostArray4D<Real> host_data("speck_hdf5_cart_host", metadata.noutvars, nz,
                                ny, nx);
    for (int v = 0; v < metadata.noutvars; ++v) {
      for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
          for (int i = 0; i < nx; ++i) {
            const std::size_t idx =
                ((static_cast<std::size_t>(v) * nz + k) * ny + j) * nx + i;
            host_data(v, k, j, i) = static_cast<Real>(raw[idx]);
          }
        }
      }
    }
    device_data = DvceArray4D<Real>("speck_hdf5_cart_device",
                                    metadata.noutvars, nz, ny, nx);
    Kokkos::deep_copy(device_data, host_data);
  } catch (...) {
    H5Fclose(file);
    throw;
  }
  RequireHdf5(H5Fclose(file), "closing HDF5 cart file");
}

void ReadSpeckCartFile(const std::string &filename, CartMetadata &metadata,
                       std::vector<std::string> &labels,
                       DvceArray4D<Real> &device_data) {
  if (EndsWith(filename, ".h5") || EndsWith(filename, ".hdf5")) {
    ReadSpeckCartHdf5File(filename, metadata, labels, device_data);
  } else {
    ReadSpeckCartBinaryFile(filename, metadata, labels, device_data);
  }
}

void FillAdmFromSpeckGhCart(MeshBlockPack *pmbp, const CartMetadata &metadata,
                            const GhIndexMap &index_map,
                            const DvceArray4D<Real> &cart_data,
                            const bool require_ghost_coverage) {
  auto &indcs = pmbp->pmesh->mb_indcs;
  auto &size = pmbp->pmb->mb_size;
  auto &adm_vars = pmbp->padm->adm;
  auto &z4c_vars = pmbp->pz4c->z4c;
  auto &u0 = pmbp->pz4c->u0;

  const int is = indcs.is;
  const int js = indcs.js;
  const int ks = indcs.ks;
  const int ng = indcs.ng;
  const int n1 = indcs.nx1 + 2 * ng;
  const int n2 = indcs.nx2 + 2 * ng;
  const int n3 = indcs.nx3 + 2 * ng;
  const int nmb = pmbp->nmb_thispack;

  CartGridDeviceSpec grid{};
  for (int d = 0; d < 3; ++d) {
    grid.center[d] = static_cast<Real>(metadata.center[d]);
    grid.extent[d] = static_cast<Real>(metadata.extent[d]);
    grid.numpoints[d] = metadata.numpoints[d];
  }
  grid.is_cheb = metadata.is_cheb;

  DvceArray1D<int> import_status("speck_cart_import_status", 2);
  Kokkos::deep_copy(import_status, 0);
  Kokkos::deep_copy(u0, 0.0);

  par_for("speck_cart_to_adm", DevExeSpace(), 0, nmb - 1, 0, n3 - 1, 0,
          n2 - 1, 0, n1 - 1,
          KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
            const Real x = CellCenterX(i - is, indcs.nx1,
                                       size.d_view(m).x1min,
                                       size.d_view(m).x1max);
            const Real y = CellCenterX(j - js, indcs.nx2,
                                       size.d_view(m).x2min,
                                       size.d_view(m).x2max);
            const Real z = CellCenterX(k - ks, indcs.nx3,
                                       size.d_view(m).x3min,
                                       size.d_view(m).x3max);
            int out_of_domain = 0;

            Real psi[4][4];
            Real pi[4][4];
            Real phi[3][4][4];
            for (int a = 0; a < 4; ++a) {
              for (int b = 0; b < 4; ++b) {
                psi[a][b] =
                    GhValue(cart_data, grid, index_map.psi[a][b], x, y, z,
                            out_of_domain);
                pi[a][b] =
                    GhValue(cart_data, grid, index_map.pi[a][b], x, y, z,
                            out_of_domain);
                for (int d = 0; d < 3; ++d) {
                  phi[d][a][b] = GhValue(cart_data, grid,
                                         index_map.phi[d][a][b], x, y, z,
                                         out_of_domain);
                }
              }
            }
            if (out_of_domain != 0) {
              Kokkos::atomic_max(&import_status(0), 1);
            }

            Real gamma[6];
            gamma[0] = psi[1][1];
            gamma[1] = psi[1][2];
            gamma[2] = psi[1][3];
            gamma[3] = psi[2][2];
            gamma[4] = psi[2][3];
            gamma[5] = psi[3][3];
            const Real detg = adm::SpatialDet(gamma[0], gamma[1], gamma[2],
                                              gamma[3], gamma[4], gamma[5]);
            if (!(detg > 0.0) || detg != detg) {
              Kokkos::atomic_max(&import_status(1), 1);
            }

            Real inv_gamma[6];
            adm::SpatialInv(1.0 / detg, gamma[0], gamma[1], gamma[2],
                            gamma[3], gamma[4], gamma[5], &inv_gamma[0],
                            &inv_gamma[1], &inv_gamma[2], &inv_gamma[3],
                            &inv_gamma[4], &inv_gamma[5]);
            Real inv[3][3];
            FillSymmetricMatrixFromSpatialMetric(inv_gamma, inv);

            Real beta_d[3] = {psi[0][1], psi[0][2], psi[0][3]};
            Real beta_u[3] = {0.0, 0.0, 0.0};
            Real beta2 = 0.0;
            for (int a = 0; a < 3; ++a) {
              for (int b = 0; b < 3; ++b) {
                beta_u[a] += inv[a][b] * beta_d[b];
              }
              beta2 += beta_d[a] * beta_u[a];
            }
            const Real alpha2 = beta2 - psi[0][0];
            if (!(alpha2 > 0.0) || alpha2 != alpha2) {
              Kokkos::atomic_max(&import_status(1), 1);
            }
            const Real alpha = Kokkos::sqrt(
                Kokkos::fmax(alpha2, static_cast<Real>(1.0e-30)));

            Real gamma_matrix[3][3];
            FillSymmetricMatrixFromSpatialMetric(gamma, gamma_matrix);

            Real dgamma[3][3][3];
            for (int c = 0; c < 3; ++c) {
              for (int a = 0; a < 3; ++a) {
                for (int b = 0; b < 3; ++b) {
                  dgamma[c][a][b] = phi[c][a + 1][b + 1];
                }
              }
            }

            Real gamma_udd[3][3][3];
            for (int a = 0; a < 3; ++a) {
              for (int b = 0; b < 3; ++b) {
                for (int c = 0; c < 3; ++c) {
                  gamma_udd[a][b][c] = 0.0;
                  for (int d = 0; d < 3; ++d) {
                    gamma_udd[a][b][c] +=
                        0.5 * inv[a][d] *
                        (dgamma[b][c][d] + dgamma[c][b][d] -
                         dgamma[d][b][c]);
                  }
                }
              }
            }

            adm_vars.alpha(m, k, j, i) = alpha;
            for (int a = 0; a < 3; ++a) {
              adm_vars.beta_u(m, a, k, j, i) = beta_u[a];
            }
            adm_vars.psi4(m, k, j, i) = 1.0;

            for (int a = 0; a < 3; ++a) {
              for (int b = a; b < 3; ++b) {
                Real beta_phi = 0.0;
                for (int d = 0; d < 3; ++d) {
                  beta_phi += beta_u[d] * phi[d][a + 1][b + 1];
                }
                Real cov_beta_deriv = phi[a][0][b + 1] + phi[b][0][a + 1];
                for (int c = 0; c < 3; ++c) {
                  cov_beta_deriv -=
                      2.0 * gamma_udd[c][a][b] * beta_d[c];
                }
                const Real k_ab =
                    0.5 * pi[a + 1][b + 1] -
                    0.5 * beta_phi / alpha +
                    0.5 * cov_beta_deriv / alpha;
                adm_vars.g_dd(m, a, b, k, j, i) = gamma_matrix[a][b];
                adm_vars.vK_dd(m, a, b, k, j, i) = k_ab;
              }
            }
            z4c_vars.vTheta(m, k, j, i) = 0.0;
            for (int a = 0; a < 3; ++a) {
              z4c_vars.vB_d(m, a, k, j, i) = 0.0;
            }
          });
  Kokkos::fence();

  auto host_status =
      Kokkos::create_mirror_view_and_copy(HostMemSpace(), import_status);
  if (require_ghost_coverage && host_status(0) != 0) {
    throw std::runtime_error(
        "SpECK cart input grid does not cover all AthenaK interior and ghost "
        "cell centers");
  }
  if (host_status(1) != 0) {
    throw std::runtime_error(
        "SpECK cart input produced invalid ADM lapse or spatial metric");
  }
}

void RecomputeAdmConstraints(MeshBlockPack *pmbp) {
  const int ng = pmbp->pmesh->mb_indcs.ng;
  switch (ng) {
    case 2:
      pmbp->pz4c->ADMConstraints<2>(pmbp);
      break;
    case 3:
      pmbp->pz4c->ADMConstraints<3>(pmbp);
      break;
    case 4:
      pmbp->pz4c->ADMConstraints<4>(pmbp);
      break;
    default:
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line "
                << __LINE__ << std::endl
                << "z4c_speck_cart_reader supports nghost = 2, 3, or 4"
                << std::endl;
      std::exit(EXIT_FAILURE);
  }
}

ConstraintSummary ComputeConstraintSummary(Mesh *pm) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  auto &indcs = pm->mb_indcs;
  const int nx1 = indcs.nx1;
  const int nx2 = indcs.nx2;
  const int nx3 = indcs.nx3;
  const int is = indcs.is;
  const int js = indcs.js;
  const int ks = indcs.ks;
  const int nmkji = pmbp->nmb_thispack * nx3 * nx2 * nx1;
  const int nkji = nx3 * nx2 * nx1;
  const int nji = nx2 * nx1;
  auto &u_con = pmbp->pz4c->u_con;
  auto &adm_vars = pmbp->padm->adm;
  auto &size = pmbp->pmb->mb_size;

  array_sum::GlobalSum local_sum;
  Kokkos::parallel_reduce(
      "speck_cart_constraint_summary",
      Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
      KOKKOS_LAMBDA(const int idx, array_sum::GlobalSum &sum) {
        const int m = idx / nkji;
        const int k0 = (idx - m * nkji) / nji;
        const int j0 = (idx - m * nkji - k0 * nji) / nx1;
        const int i = (idx - m * nkji - k0 * nji - j0 * nx1) + is;
        const int j = j0 + js;
        const int k = k0 + ks;
        const Real detg = adm::SpatialDet(
            adm_vars.g_dd(m, 0, 0, k, j, i),
            adm_vars.g_dd(m, 0, 1, k, j, i),
            adm_vars.g_dd(m, 0, 2, k, j, i),
            adm_vars.g_dd(m, 1, 1, k, j, i),
            adm_vars.g_dd(m, 1, 2, k, j, i),
            adm_vars.g_dd(m, 2, 2, k, j, i));
        const Real vol = size.d_view(m).dx1 * size.d_view(m).dx2 *
                         size.d_view(m).dx3 *
                         Kokkos::sqrt(Kokkos::abs(detg));
        array_sum::GlobalSum cell_sum;
        cell_sum.the_array[0] = vol * u_con(m, z4c::Z4c::I_CON_C, k, j, i);
        cell_sum.the_array[1] =
            vol * SQR(u_con(m, z4c::Z4c::I_CON_H, k, j, i));
        cell_sum.the_array[2] = vol * u_con(m, z4c::Z4c::I_CON_M, k, j, i);
        cell_sum.the_array[3] = vol * u_con(m, z4c::Z4c::I_CON_Z, k, j, i);
        cell_sum.the_array[4] = vol;
        for (int n = 5; n < NREDUCTION_VARIABLES; ++n) {
          cell_sum.the_array[n] = 0.0;
        }
        sum += cell_sum;
      },
      Kokkos::Sum<array_sum::GlobalSum>(local_sum));

  Real totals[5];
  for (int n = 0; n < 5; ++n) {
    totals[n] = local_sum.the_array[n];
  }
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, totals, 5, MPI_ATHENA_REAL, MPI_SUM,
                MPI_COMM_WORLD);
#endif

  ConstraintSummary summary;
  summary.volume = totals[4];
  if (summary.volume > 0.0) {
    summary.c_rms = std::sqrt(totals[0] / summary.volume);
    summary.h_rms = std::sqrt(totals[1] / summary.volume);
    summary.m_rms = std::sqrt(totals[2] / summary.volume);
    summary.z_rms = std::sqrt(totals[3] / summary.volume);
  }
  return summary;
}

void WriteConstraintSummary(ParameterInput *pin, Mesh *pm,
                            const ConstraintSummary &summary) {
  if (global_variable::my_rank != 0) {
    return;
  }
  std::string fname = pin->GetString("job", "basename");
  fname.append("-speck-cart-constraints.dat");
  FILE *pfile = std::fopen(fname.c_str(), "r");
  if (pfile != nullptr) {
    pfile = std::freopen(fname.c_str(), "a", pfile);
  } else {
    pfile = std::fopen(fname.c_str(), "w");
    if (pfile != nullptr) {
      std::fprintf(pfile,
                   "# Nx1  Nx2  Nx3   Ncycle  C_rms  H_rms  M_rms  Z_rms  "
                   "Volume\n");
    }
  }
  if (pfile == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "SpECK cart constraint output file could not be opened"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  std::fprintf(pfile,
               "%04d  %04d  %04d  %05d  %.16e  %.16e  %.16e  %.16e  "
               "%.16e\n",
               pm->mesh_indcs.nx1, pm->mesh_indcs.nx2,
               pm->mesh_indcs.nx3, pm->ncycle, summary.c_rms,
               summary.h_rms, summary.m_rms, summary.z_rms, summary.volume);
  std::fclose(pfile);
}

void EnforceConstraintThresholds(ParameterInput *pin,
                                 const ConstraintSummary &summary) {
  const Real c_threshold = pin->GetOrAddReal(
      "problem", "fail_if_c_rms_above", std::numeric_limits<Real>::infinity());
  const Real h_threshold = pin->GetOrAddReal(
      "problem", "fail_if_h_rms_above", std::numeric_limits<Real>::infinity());
  const Real m_threshold = pin->GetOrAddReal(
      "problem", "fail_if_m_rms_above", std::numeric_limits<Real>::infinity());
  const Real z_threshold = pin->GetOrAddReal(
      "problem", "fail_if_z_rms_above", std::numeric_limits<Real>::infinity());
  if (summary.c_rms > c_threshold || summary.h_rms > h_threshold ||
      summary.m_rms > m_threshold || summary.z_rms > z_threshold) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "SpECK cart imported constraints exceeded threshold: C="
              << summary.c_rms << " H=" << summary.h_rms
              << " M=" << summary.m_rms << " Z=" << summary.z_rms
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

void SpeckCartConstraintReport(ParameterInput *pin, Mesh *pm) {
  if (pm->pmb_pack->pz4c == nullptr) {
    return;
  }
  RecomputeAdmConstraints(pm->pmb_pack);
  const ConstraintSummary summary = ComputeConstraintSummary(pm);
  if (pin->GetOrAddBoolean("problem", "write_constraint_summary", true)) {
    WriteConstraintSummary(pin, pm, summary);
  }
  EnforceConstraintThresholds(pin, summary);
}

void WriteAdmPsi4XyPlane(ParameterInput *pin, MeshBlockPack *pmbp) {
  const std::string path =
      pin->GetOrAddString("problem", "xy_plane_output", "EMPTY");
  if (path == "EMPTY" || path.empty()) {
    return;
  }
  if (global_variable::my_rank != 0) {
    return;
  }
  const Real z_plane = pin->GetOrAddReal("problem", "xy_plane_z", 0.0);
  const std::filesystem::path output_path(path);
  if (!output_path.parent_path().empty()) {
    std::filesystem::create_directories(output_path.parent_path());
  }
  std::ofstream output(path);
  if (!output) {
    throw std::runtime_error("failed to open xy-plane output: " + path);
  }
  auto host_adm =
      Kokkos::create_mirror_view_and_copy(HostMemSpace(), pmbp->padm->u_adm);
  pmbp->pmb->mb_size.sync_host();
  auto size = pmbp->pmb->mb_size.h_view;
  auto &indcs = pmbp->pmesh->mb_indcs;
  output << "# x y z adm_psi4 block i j k\n";
  for (int m = 0; m < pmbp->nmb_thispack; ++m) {
    int k_plane = indcs.ks;
    Real best = std::numeric_limits<Real>::infinity();
    for (int k = indcs.ks; k <= indcs.ke; ++k) {
      const Real z =
          CellCenterX(k - indcs.ks, indcs.nx3, size(m).x3min,
                      size(m).x3max);
      const Real distance = std::abs(z - z_plane);
      if (distance < best) {
        best = distance;
        k_plane = k;
      }
    }
    for (int j = indcs.js; j <= indcs.je; ++j) {
      const Real y =
          CellCenterX(j - indcs.js, indcs.nx2, size(m).x2min,
                      size(m).x2max);
      for (int i = indcs.is; i <= indcs.ie; ++i) {
        const Real x =
            CellCenterX(i - indcs.is, indcs.nx1, size(m).x1min,
                        size(m).x1max);
        const Real z =
            CellCenterX(k_plane - indcs.ks, indcs.nx3, size(m).x3min,
                        size(m).x3max);
        output << x << ' ' << y << ' ' << z << ' '
               << host_adm(m, adm::ADM::I_ADM_PSI4, k_plane, j, i) << ' '
               << m << ' ' << i << ' ' << j << ' ' << k_plane << '\n';
      }
    }
  }
}

} // namespace

void ProblemGenerator::Z4cSpeckCartReader(ParameterInput *pin,
                                          const bool restart) {
  pgen_final_func = SpeckCartConstraintReport;
  if (restart) {
    return;
  }

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->pz4c == nullptr || pmbp->padm == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "z4c_speck_cart_reader requires <z4c> and <adm> blocks"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }

  const std::string filename =
      pin->GetOrAddString("problem", "speck_cart_file", "EMPTY");
  const bool require_ghost_coverage =
      pin->GetOrAddBoolean("problem", "require_ghost_coverage", true);
  CartMetadata metadata{};
  std::vector<std::string> labels;
  DvceArray4D<Real> cart_data;
  try {
    ReadSpeckCartFile(filename, metadata, labels, cart_data);
    const GhIndexMap index_map = BuildGhIndexMap(labels);
    FillAdmFromSpeckGhCart(pmbp, metadata, index_map, cart_data,
                           require_ghost_coverage);
  } catch (const std::exception &error) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "Failed to initialize from SpECK cart input: " << error.what()
              << std::endl;
    std::exit(EXIT_FAILURE);
  }

  switch (pmy_mesh_->mb_indcs.ng) {
    case 2:
      pmbp->pz4c->ADMToZ4c<2>(pmbp, pin);
      break;
    case 3:
      pmbp->pz4c->ADMToZ4c<3>(pmbp, pin);
      break;
    case 4:
      pmbp->pz4c->ADMToZ4c<4>(pmbp, pin);
      break;
    default:
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line "
                << __LINE__ << std::endl
                << "z4c_speck_cart_reader supports nghost = 2, 3, or 4"
                << std::endl;
      std::exit(EXIT_FAILURE);
  }
  pmbp->pz4c->Z4cToADM(pmbp);
  RecomputeAdmConstraints(pmbp);
  const ConstraintSummary summary = ComputeConstraintSummary(pmy_mesh_);
  EnforceConstraintThresholds(pin, summary);
  try {
    WriteAdmPsi4XyPlane(pin, pmbp);
  } catch (const std::exception &error) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "Failed to write AthenaK SpECK cart xy-plane diagnostic: "
              << error.what() << std::endl;
    std::exit(EXIT_FAILURE);
  }

  if (global_variable::my_rank == 0) {
    std::cout << "Initialized Z4c from SpECK cart GH data: " << filename
              << " labels=" << labels.size() << " grid=("
              << metadata.numpoints[0] << "," << metadata.numpoints[1]
              << "," << metadata.numpoints[2] << ")"
              << " C_rms=" << summary.c_rms << " H_rms=" << summary.h_rms
              << " M_rms=" << summary.m_rms << " Z_rms=" << summary.z_rms
              << std::endl;
  }
}
