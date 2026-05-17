//========================================================================================
// Convert a uniform SpECK ADM Cartesian HDF5 dump into the id_solve HDF5 layout.
//
// Build:
//   h5c++ -std=c++17 -O2 scripts/z4c/convert_speck_adm_cart_to_id_solve.cpp -o convert_speck_adm_cart_to_id_solve
//
// Usage:
//   ./convert_speck_adm_cart_to_id_solve speck.adm.h5 id_solve.h5
//========================================================================================

#include <hdf5.h>

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

void Require(const herr_t status, const std::string &context) {
  if (status < 0) {
    throw std::runtime_error("HDF5 failure while " + context);
  }
}

std::size_t ElementCount(const hid_t space, const std::string &name) {
  const int rank = H5Sget_simple_extent_ndims(space);
  if (rank < 0) {
    throw std::runtime_error("failed to get rank for " + name);
  }
  std::vector<hsize_t> dims(static_cast<std::size_t>(rank));
  Require(H5Sget_simple_extent_dims(space, dims.data(), nullptr),
          "reading extent for " + name);
  std::size_t count = 1;
  for (const hsize_t dim : dims) {
    count *= static_cast<std::size_t>(dim);
  }
  return count;
}

std::vector<double> ReadDoubles(const hid_t file, const std::string &name,
                                std::vector<hsize_t> *dims_out = nullptr) {
  const hid_t dataset = H5Dopen2(file, name.c_str(), H5P_DEFAULT);
  if (dataset < 0) {
    throw std::runtime_error("failed to open " + name);
  }
  const hid_t space = H5Dget_space(dataset);
  if (space < 0) {
    H5Dclose(dataset);
    throw std::runtime_error("failed to get dataspace for " + name);
  }
  const int rank = H5Sget_simple_extent_ndims(space);
  std::vector<hsize_t> dims(static_cast<std::size_t>(rank));
  Require(H5Sget_simple_extent_dims(space, dims.data(), nullptr),
          "reading extent for " + name);
  const std::size_t count = ElementCount(space, name);
  std::vector<double> values(count);
  Require(H5Dread(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                  values.data()),
          "reading " + name);
  Require(H5Sclose(space), "closing dataspace for " + name);
  Require(H5Dclose(dataset), "closing dataset " + name);
  if (dims_out != nullptr) {
    *dims_out = std::move(dims);
  }
  return values;
}

std::vector<int> ReadInts(const hid_t file, const std::string &name,
                          const std::size_t expected) {
  const hid_t dataset = H5Dopen2(file, name.c_str(), H5P_DEFAULT);
  if (dataset < 0) {
    throw std::runtime_error("failed to open " + name);
  }
  const hid_t space = H5Dget_space(dataset);
  if (space < 0) {
    H5Dclose(dataset);
    throw std::runtime_error("failed to get dataspace for " + name);
  }
  const std::size_t count = ElementCount(space, name);
  if (count != expected) {
    throw std::runtime_error("unexpected element count for " + name);
  }
  std::vector<int> values(count);
  Require(H5Dread(dataset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                  values.data()),
          "reading " + name);
  Require(H5Sclose(space), "closing dataspace for " + name);
  Require(H5Dclose(dataset), "closing dataset " + name);
  return values;
}

std::vector<std::string> SplitLabels(const std::string &text) {
  std::istringstream stream(text);
  std::vector<std::string> labels;
  std::string label;
  while (stream >> label) {
    labels.push_back(label);
  }
  return labels;
}

std::vector<std::string> ReadLabels(const hid_t file) {
  const hid_t dataset = H5Dopen2(file, "/labels", H5P_DEFAULT);
  if (dataset < 0) {
    throw std::runtime_error("failed to open /labels");
  }
  const hid_t space = H5Dget_space(dataset);
  if (space < 0) {
    H5Dclose(dataset);
    throw std::runtime_error("failed to get dataspace for /labels");
  }
  const std::size_t count = ElementCount(space, "/labels");
  std::string text(count, '\0');
  Require(H5Dread(dataset, H5T_NATIVE_CHAR, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                  text.data()),
          "reading /labels");
  Require(H5Sclose(space), "closing dataspace for /labels");
  Require(H5Dclose(dataset), "closing dataset /labels");
  return SplitLabels(text);
}

int LabelIndex(const std::unordered_map<std::string, int> &labels,
               const std::string &name) {
  const auto it = labels.find(name);
  if (it == labels.end()) {
    throw std::runtime_error("missing required SpECK ADM label: " + name);
  }
  return it->second;
}

void WriteDoubles(const hid_t file, const std::string &name,
                  const std::vector<hsize_t> &dims,
                  const std::vector<double> &values) {
  hsize_t count = 1;
  for (const hsize_t dim : dims) {
    count *= dim;
  }
  if (count != values.size()) {
    throw std::runtime_error("shape/value mismatch for " + name);
  }
  const hid_t space = H5Screate_simple(static_cast<int>(dims.size()),
                                       dims.data(), nullptr);
  if (space < 0) {
    throw std::runtime_error("failed to create dataspace for " + name);
  }
  const hid_t dataset = H5Dcreate2(file, name.c_str(), H5T_IEEE_F64LE, space,
                                   H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  if (dataset < 0) {
    H5Sclose(space);
    throw std::runtime_error("failed to create dataset " + name);
  }
  Require(H5Dwrite(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                   values.data()),
          "writing " + name);
  Require(H5Dclose(dataset), "closing dataset " + name);
  Require(H5Sclose(space), "closing dataspace " + name);
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 3) {
    std::cerr << "usage: " << argv[0] << " speck_adm_cart.h5 id_solve.h5\n";
    return EXIT_FAILURE;
  }
  try {
    const std::string input = argv[1];
    const std::string output = argv[2];

    const hid_t in_file = H5Fopen(input.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (in_file < 0) {
      throw std::runtime_error("failed to open input file: " + input);
    }

    const std::vector<int> npts = ReadInts(in_file, "/numpoints", 3);
    const std::vector<int> is_cheb = ReadInts(in_file, "/is_cheb", 1);
    if (is_cheb[0] != 0) {
      throw std::runtime_error("id_solve conversion requires a uniform SpECK cart dump");
    }
    const std::vector<double> center = ReadDoubles(in_file, "/center");
    const std::vector<double> extent = ReadDoubles(in_file, "/extent");
    std::vector<hsize_t> data_dims;
    const std::vector<double> data = ReadDoubles(in_file, "/data", &data_dims);
    const std::vector<std::string> label_list = ReadLabels(in_file);
    Require(H5Fclose(in_file), "closing input file");

    if (data_dims.size() != 4 ||
        data_dims[1] != static_cast<hsize_t>(npts[2]) ||
        data_dims[2] != static_cast<hsize_t>(npts[1]) ||
        data_dims[3] != static_cast<hsize_t>(npts[0])) {
      throw std::runtime_error("SpECK /data shape does not match /numpoints");
    }

    std::unordered_map<std::string, int> labels;
    for (int n = 0; n < static_cast<int>(label_list.size()); ++n) {
      labels[label_list[n]] = n;
    }
    const char *metric_labels[6] = {"adm_gxx", "adm_gxy", "adm_gxz",
                                    "adm_gyy", "adm_gyz", "adm_gzz"};
    const char *extrin_labels[6] = {"adm_Kxx", "adm_Kxy", "adm_Kxz",
                                    "adm_Kyy", "adm_Kyz", "adm_Kzz"};
    int metric_index[6];
    int extrin_index[6];
    for (int c = 0; c < 6; ++c) {
      metric_index[c] = LabelIndex(labels, metric_labels[c]);
      extrin_index[c] = LabelIndex(labels, extrin_labels[c]);
    }

    const int nx = npts[0];
    const int ny = npts[1];
    const int nz = npts[2];
    const int nblocks = 1;
    std::vector<double> x1v(static_cast<std::size_t>(nx));
    std::vector<double> x2v(static_cast<std::size_t>(ny));
    std::vector<double> x3v(static_cast<std::size_t>(nz));
    for (int i = 0; i < nx; ++i) {
      x1v[i] = (center[0] - extent[0]) +
               static_cast<double>(i) * (2.0 * extent[0]) /
                   static_cast<double>(nx - 1);
    }
    for (int j = 0; j < ny; ++j) {
      x2v[j] = (center[1] - extent[1]) +
               static_cast<double>(j) * (2.0 * extent[1]) /
                   static_cast<double>(ny - 1);
    }
    for (int k = 0; k < nz; ++k) {
      x3v[k] = (center[2] - extent[2]) +
               static_cast<double>(k) * (2.0 * extent[2]) /
                   static_cast<double>(nz - 1);
    }

    const std::size_t block_vol =
        static_cast<std::size_t>(nx) * ny * nz;
    std::vector<double> metric(6 * nblocks * block_vol);
    std::vector<double> extrin(6 * nblocks * block_vol);
    auto speck = [&](const int v, const int k, const int j, const int i) {
      return data[((static_cast<std::size_t>(v) * nz + k) * ny + j) * nx + i];
    };
    auto idsolve = [&](const int c, const int k, const int j, const int i) {
      return static_cast<std::size_t>(c) * nblocks * block_vol +
             static_cast<std::size_t>(i) +
             static_cast<std::size_t>(nx) *
                 (static_cast<std::size_t>(j) +
                  static_cast<std::size_t>(ny) * static_cast<std::size_t>(k));
    };
    for (int c = 0; c < 6; ++c) {
      for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
          for (int i = 0; i < nx; ++i) {
            metric[idsolve(c, k, j, i)] = speck(metric_index[c], k, j, i);
            extrin[idsolve(c, k, j, i)] = speck(extrin_index[c], k, j, i);
          }
        }
      }
    }

    const hid_t out_file =
        H5Fcreate(output.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (out_file < 0) {
      throw std::runtime_error("failed to create output file: " + output);
    }
    WriteDoubles(out_file, "/x1v", {static_cast<hsize_t>(nblocks),
                                    static_cast<hsize_t>(nx)}, x1v);
    WriteDoubles(out_file, "/x2v", {static_cast<hsize_t>(nblocks),
                                    static_cast<hsize_t>(ny)}, x2v);
    WriteDoubles(out_file, "/x3v", {static_cast<hsize_t>(nblocks),
                                    static_cast<hsize_t>(nz)}, x3v);
    WriteDoubles(out_file, "/metric",
                 {6, static_cast<hsize_t>(nblocks), static_cast<hsize_t>(nx),
                  static_cast<hsize_t>(ny), static_cast<hsize_t>(nz)},
                 metric);
    WriteDoubles(out_file, "/extrin",
                 {6, static_cast<hsize_t>(nblocks), static_cast<hsize_t>(nx),
                  static_cast<hsize_t>(ny), static_cast<hsize_t>(nz)},
                 extrin);
    Require(H5Fclose(out_file), "closing output file");
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
