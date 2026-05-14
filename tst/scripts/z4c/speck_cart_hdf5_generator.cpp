#include <hdf5.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kNumGhVars = 50;

std::vector<std::pair<int, int>> SymPairs() {
  std::vector<std::pair<int, int>> pairs;
  for (int a = 0; a < 4; ++a) {
    for (int b = a; b < 4; ++b) {
      pairs.push_back({a, b});
    }
  }
  return pairs;
}

std::string Labels() {
  std::ostringstream labels;
  bool first = true;
  const auto pairs = SymPairs();
  for (const std::string prefix : {"psi", "pi"}) {
    for (const auto &pair : pairs) {
      if (!first) {
        labels << ' ';
      }
      first = false;
      labels << prefix << pair.first << pair.second;
    }
  }
  for (int d = 0; d < 3; ++d) {
    for (const auto &pair : pairs) {
      labels << ' ' << "phi" << d << '_' << pair.first << pair.second;
    }
  }
  return labels.str();
}

std::vector<double> KerrSchildGhValues(const double x, const double y,
                                       const double z) {
  const double mass = 1.0;
  const double radius = std::sqrt(x * x + y * y + z * z);
  const double normal[3] = {x / radius, y / radius, z / radius};
  const double h = 2.0 * mass / radius;

  double psi[4][4] = {};
  psi[0][0] = -1.0 + h;
  for (int a = 0; a < 3; ++a) {
    psi[0][a + 1] = h * normal[a];
    psi[a + 1][0] = h * normal[a];
  }
  for (int a = 0; a < 3; ++a) {
    for (int b = 0; b < 3; ++b) {
      psi[a + 1][b + 1] = (a == b ? 1.0 : 0.0) + h * normal[a] * normal[b];
    }
  }

  double phi[3][4][4] = {};
  for (int d = 0; d < 3; ++d) {
    const double dh = -h * normal[d] / radius;
    double dn[3] = {};
    for (int a = 0; a < 3; ++a) {
      dn[a] = ((d == a ? 1.0 : 0.0) - normal[d] * normal[a]) / radius;
    }
    phi[d][0][0] = dh;
    for (int a = 0; a < 3; ++a) {
      const double value = dh * normal[a] + h * dn[a];
      phi[d][0][a + 1] = value;
      phi[d][a + 1][0] = value;
    }
    for (int a = 0; a < 3; ++a) {
      for (int b = 0; b < 3; ++b) {
        phi[d][a + 1][b + 1] =
            dh * normal[a] * normal[b] +
            h * (dn[a] * normal[b] + normal[a] * dn[b]);
      }
    }
  }

  const double alpha = 1.0 / std::sqrt(1.0 + h);
  const double beta[3] = {h / (1.0 + h) * normal[0],
                          h / (1.0 + h) * normal[1],
                          h / (1.0 + h) * normal[2]};
  double pi[4][4] = {};
  for (int a = 0; a < 4; ++a) {
    for (int b = 0; b < 4; ++b) {
      for (int d = 0; d < 3; ++d) {
        pi[a][b] += beta[d] * phi[d][a][b] / alpha;
      }
    }
  }

  std::vector<double> values;
  values.reserve(kNumGhVars);
  const auto pairs = SymPairs();
  for (const auto &pair : pairs) {
    values.push_back(psi[pair.first][pair.second]);
  }
  for (const auto &pair : pairs) {
    values.push_back(pi[pair.first][pair.second]);
  }
  for (int d = 0; d < 3; ++d) {
    for (const auto &pair : pairs) {
      values.push_back(phi[d][pair.first][pair.second]);
    }
  }
  return values;
}

void Require(const herr_t status, const std::string &what) {
  if (status < 0) {
    throw std::runtime_error("HDF5 failure while " + what);
  }
}

template <typename T>
void WriteDataset(const hid_t file, const std::string &name,
                  const hid_t hdf5_type, const std::vector<T> &values,
                  const std::vector<hsize_t> &dims) {
  const hid_t space =
      H5Screate_simple(static_cast<int>(dims.size()), dims.data(), nullptr);
  if (space < 0) {
    throw std::runtime_error("failed to create dataspace " + name);
  }
  const hid_t dataset =
      H5Dcreate2(file, name.c_str(), hdf5_type, space, H5P_DEFAULT,
                 H5P_DEFAULT, H5P_DEFAULT);
  if (dataset < 0) {
    H5Sclose(space);
    throw std::runtime_error("failed to create dataset " + name);
  }
  Require(H5Dwrite(dataset, hdf5_type, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                   values.data()),
          "writing " + name);
  Require(H5Dclose(dataset), "closing " + name);
  Require(H5Sclose(space), "closing dataspace " + name);
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 4) {
    std::cerr << "usage: " << argv[0]
              << " output.h5 active_resolution nghost\n";
    return 2;
  }
  const std::string path = argv[1];
  const int resolution = std::atoi(argv[2]);
  const int nghost = std::atoi(argv[3]);
  if (resolution <= 0 || nghost < 0) {
    std::cerr << "invalid resolution or nghost\n";
    return 2;
  }

  try {
    const int n = resolution + 2 * nghost;
    const double xmin[3] = {3.0, -1.0, -1.0};
    const double xmax[3] = {5.0, 1.0, 1.0};
    std::vector<std::vector<double>> axes(3);
    double center[3] = {};
    double extent[3] = {};
    for (int d = 0; d < 3; ++d) {
      axes[d].resize(static_cast<std::size_t>(n));
      const double dx = (xmax[d] - xmin[d]) / resolution;
      for (int q = 0; q < n; ++q) {
        axes[d][q] = xmin[d] + ((q - nghost) + 0.5) * dx;
      }
      center[d] = 0.5 * (axes[d].front() + axes[d].back());
      extent[d] = 0.5 * (axes[d].back() - axes[d].front());
    }

    std::vector<double> data(static_cast<std::size_t>(kNumGhVars) * n * n * n);
    for (int k = 0; k < n; ++k) {
      for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
          const auto values =
              KerrSchildGhValues(axes[0][i], axes[1][j], axes[2][k]);
          for (int v = 0; v < kNumGhVars; ++v) {
            const std::size_t index =
                ((static_cast<std::size_t>(v) * n + k) * n + j) * n + i;
            data[index] = values[static_cast<std::size_t>(v)];
          }
        }
      }
    }

    const hid_t file = H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT,
                                 H5P_DEFAULT);
    if (file < 0) {
      throw std::runtime_error("failed to create " + path);
    }
    WriteDataset<double>(file, "/center", H5T_NATIVE_DOUBLE,
                         {center[0], center[1], center[2]}, {3});
    WriteDataset<double>(file, "/extent", H5T_NATIVE_DOUBLE,
                         {extent[0], extent[1], extent[2]}, {3});
    WriteDataset<double>(file, "/time", H5T_NATIVE_DOUBLE, {0.0}, {1});
    WriteDataset<int>(file, "/cycle", H5T_NATIVE_INT, {0}, {1});
    WriteDataset<int>(file, "/numpoints", H5T_NATIVE_INT, {n, n, n}, {3});
    WriteDataset<int>(file, "/is_cheb", H5T_NATIVE_INT, {0}, {1});
    WriteDataset<int>(file, "/noutvars", H5T_NATIVE_INT, {kNumGhVars}, {1});
    const std::string label_text = Labels();
    std::vector<char> labels(label_text.begin(), label_text.end());
    WriteDataset<char>(file, "/labels", H5T_NATIVE_CHAR, labels,
                       {static_cast<hsize_t>(labels.size())});
    WriteDataset<double>(
        file, "/data", H5T_NATIVE_DOUBLE, data,
        {static_cast<hsize_t>(kNumGhVars), static_cast<hsize_t>(n),
         static_cast<hsize_t>(n), static_cast<hsize_t>(n)});
    Require(H5Fclose(file), "closing " + path);
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  return 0;
}
