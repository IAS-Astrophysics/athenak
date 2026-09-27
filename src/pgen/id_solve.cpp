//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2026 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file id_solve.cpp
//! \brief Import Cartesian id_solve HDF5 data with checked fourth-order interpolation.

#include <hdf5.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <climits>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/adm.hpp"
#include "coordinates/cell_locations.hpp"
#include "z4c/z4c.hpp"
#include "z4c/z4c_amr.hpp"

namespace {
// Close HDF5 handles even if validation or allocation throws.
struct Handle {
  hid_t id;
  herr_t (*close)(hid_t);
  ~Handle() { if (id >= 0) close(id); }
};
struct Dataset {
  std::vector<hsize_t> shape;
  std::vector<double> data;
};
Dataset Read(hid_t file, const char *name, int rank) {
  Handle ds{H5Dopen2(file, name, H5P_DEFAULT), H5Dclose};
  if (ds.id < 0) throw std::runtime_error(std::string("Missing dataset: ")+name);
  Handle space{H5Dget_space(ds.id), H5Sclose};
  if (H5Sget_simple_extent_ndims(space.id) != rank) {
    throw std::runtime_error(std::string("Invalid dataset rank: ")+name);
  }
  Dataset result;
  result.shape.resize(rank);
  H5Sget_simple_extent_dims(space.id, result.shape.data(), nullptr);
  std::size_t count = 1;
  for (auto dim : result.shape) {
    if (dim == 0 || dim > std::numeric_limits<std::size_t>::max()/count) {
      throw std::runtime_error("Invalid HDF5 extent");
    }
    count *= dim;
  }
  result.data.resize(count);
  if (H5Dread(ds.id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
              result.data.data()) < 0) throw std::runtime_error("HDF5 read failed");
  return result;
}

// A centered five-point stencil, shifted only where a block edge requires it.
int Weights(const double *x, int n, double xp, double w[5]) {
  int near = std::lower_bound(x, x+n, xp)-x;
  near = std::min(near, n-1);
  if (near > 0 && xp-x[near-1] < x[near]-xp) --near;
  int start = std::clamp(near-2, 0, n-5);
  for (int a=0; a<5; ++a) {
    w[a] = 1.0;
    for (int b=0; b<5; ++b) {
      if (a != b) w[a] *= (xp-x[start+b])/(x[start+a]-x[start+b]);
    }
  }
  return start;
}
void IDRefinement(MeshBlockPack *pack) { pack->pz4c->pamr->Refine(pack); }

void Load(MeshBlockPack *pack, ParameterInput *pin) {
  auto filename = pin->GetString("problem", "id_filename");
  Handle file{H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), H5Fclose};
  if (file.id < 0) throw std::runtime_error("Cannot open id_solve HDF5 file");
  std::array<Dataset, 3> coords = {Read(file.id,"x1v",2), Read(file.id,"x2v",2),
                                  Read(file.id,"x3v",2)};
  Dataset metric = Read(file.id,"metric",5), extrin = Read(file.id,"extrin",5);
  // Athena/HDF5 layout: [xx,xy,xz,yy,yz,zz][block][z][y][x].
  std::size_t nb = coords[0].shape[0];
  int n[3];
  int source_ng = pin->GetOrAddInteger("problem", "id_source_nghost", 0);
  for (int a=0; a<3; ++a) {
    if (coords[a].shape[0] != nb || coords[a].shape[1] > INT_MAX) {
      throw std::runtime_error("Coordinate block count/extent mismatch");
    }
    n[a] = static_cast<int>(coords[a].shape[1]);
    if (n[a] < 5 || source_ng < 0 || 2*source_ng >= n[a]) {
      throw std::runtime_error("Need at least five coordinate points and valid ghosts");
    }
    for (std::size_t b=0; b<nb; ++b) {
      const double *x = coords[a].data.data()+b*n[a];
      for (int q=0; q<n[a]; ++q) {
        if (!std::isfinite(x[q]) || (q>0 && x[q]<=x[q-1])) {
          throw std::runtime_error("Coordinates must be finite and strictly increasing");
        }
      }
    }
  }
  std::vector<hsize_t> expected{6,nb,static_cast<hsize_t>(n[2]),
                                   static_cast<hsize_t>(n[1]),
                                   static_cast<hsize_t>(n[0])};
  if (metric.shape != expected || extrin.shape != expected) {
    throw std::runtime_error("Expected metric/extrin shape [6,block,z,y,x]");
  }
  Real fill = pin->GetOrAddReal("z4c", "r_fill", 0.0);
  Real radius_floor = pin->GetOrAddReal("problem", "puncture_radius_floor", 1e-8);
  Real mass[2], center[2][3];
  for (int h=0; h<2; ++h) {
    mass[h] = pin->GetOrAddReal("z4c", "M_fill_"+std::to_string(h), 0.5);
    for (int a=0; a<3; ++a) {
      center[h][a] = pin->GetOrAddReal("z4c", "co_"+std::to_string(h)+"_"+
                                      std::string(1,'x'+a), 0.0);
      if (!std::isfinite(center[h][a])) throw std::runtime_error("Invalid fill center");
    }
    if (!std::isfinite(mass[h]) || mass[h]<0) {
      throw std::runtime_error("Invalid fill mass");
    }
  }
  if (!std::isfinite(fill) || fill<0 || !std::isfinite(radius_floor) || radius_floor<=0) {
    throw std::runtime_error("Invalid fill radius");
  }
  auto ind = pack->pmesh->mb_indcs;
  auto &sizes = pack->pmb->mb_size;
  sizes.sync<HostMemSpace>();
  auto host = Kokkos::create_mirror_view(pack->padm->u_adm);
  Kokkos::deep_copy(host, 0.0);
  std::size_t volume = static_cast<std::size_t>(n[0])*n[1]*n[2];
  for (int m=0; m<pack->nmb_thispack; ++m) {
    auto size = sizes.h_view(m);
    for (int k=0; k<ind.nx3+2*ind.ng; ++k)
    for (int j=0; j<ind.nx2+2*ind.ng; ++j)
    for (int i=0; i<ind.nx1+2*ind.ng; ++i) {
      double pos[3] = {CellCenterX(i-ind.is,ind.nx1,size.x1min,size.x1max),
                       CellCenterX(j-ind.js,ind.nx2,size.x2min,size.x2max),
                       CellCenterX(k-ind.ks,ind.nx3,size.x3min,size.x3max)};
      std::size_t chosen = nb;
      // Prefer active coverage, then source ghosts; prefer finer overlapping blocks.
      for (int pass=0; pass<2 && chosen==nb; ++pass) {
        double best = std::numeric_limits<double>::infinity();
        for (std::size_t b=0; b<nb; ++b) {
          bool inside = true;
          double volume_cell = 1.0;
          for (int a=0; a<3; ++a) {
            const double *x = coords[a].data.data()+b*n[a];
            int g = pass == 0 ? source_ng : 0;
            inside &= pos[a]>=x[g] && pos[a]<=x[n[a]-1-g];
            volume_cell *= (x[n[a]-1]-x[0])/(n[a]-1);
          }
          if (inside && volume_cell<best) {
            chosen=b; best=volume_cell;
          }
        }
      }
      if (chosen==nb) {
        throw std::runtime_error("id_solve data do not cover destination ghosts");
      }
      int start[3]; double w[3][5];
      for (int a=0; a<3; ++a) {
        start[a] = Weights(coords[a].data.data()+chosen*n[a],n[a],pos[a],w[a]);
      }
      double values[12] = {};
      for (int c=0; c<12; ++c) {
        const auto &data = c<6 ? metric.data : extrin.data;
        for (int az=0; az<5; ++az)
        for (int ay=0; ay<5; ++ay)
        for (int ax=0; ax<5; ++ax) {
          std::size_t index = ((c%6)*nb+chosen)*volume+
              (static_cast<std::size_t>(start[2]+az)*n[1]+start[1]+ay)*n[0]+start[0]+ax;
          values[c] += w[0][ax]*w[1][ay]*w[2][az]*data[index];
        }
      }
      if (fill>0) {
        double r[2] = {};
        for (int h=0; h<2; ++h) {
          for (int a=0; a<3; ++a) r[h] += SQR(pos[a]-center[h][a]);
          r[h] = std::max(static_cast<double>(radius_floor),std::sqrt(r[h]));
        }
        double u = std::min(r[0],r[1])/fill;
        if (u<1) {
          double weight = 1-u*u*u*(10-15*u+6*u*u);
          double psi = 1+mass[0]/(2*r[0])+mass[1]/(2*r[1]);
          for (int c=0; c<12; ++c) {
            values[c] = (1-weight)*values[c]+
                         ((c==0 || c==3 || c==5) ? weight*SQR(SQR(psi)) : 0.0);
          }
        }
      }
      for (double value : values) {
        if (!std::isfinite(value)) {
          throw std::runtime_error("Nonfinite interpolated data");
        }
      }
      double det = adm::SpatialDet(values[0],values[1],values[2],values[3],
                                   values[4],values[5]);
      if (values[0]<=0 || values[0]*values[3]<=SQR(values[1]) || det<=0) {
        throw std::runtime_error("Interpolated spatial metric is not positive definite");
      }
      for (int c=0; c<6; ++c) {
        host(m,adm::ADM::I_ADM_GXX+c,k,j,i) = values[c];
        host(m,adm::ADM::I_ADM_KXX+c,k,j,i) = values[c+6];
      }
      host(m,adm::ADM::I_ADM_ALPHA,k,j,i) = 1.0;
    }
  }
  Kokkos::deep_copy(pack->padm->u_adm,host);
}
} // namespace

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  user_ref_func = IDRefinement;
  if (restart) return;
  auto *pack = pmy_mesh_->pmb_pack;
  try {
    if (!pack->pz4c || !pmy_mesh_->three_d) {
      throw std::runtime_error("id_solve requires three dimensions and <z4c>");
    }
    Load(pack,pin);
  } catch (const std::exception &e) {
    std::cerr << "id_solve: " << e.what() << std::endl;
    std::exit(EXIT_FAILURE);
  }
  switch (pmy_mesh_->mb_indcs.ng) {
    case 2: pack->pz4c->ADMToZ4c<2>(pack,pin); break;
    case 3: pack->pz4c->ADMToZ4c<3>(pack,pin); break;
    case 4: pack->pz4c->ADMToZ4c<4>(pack,pin); break;
  }
  pack->pz4c->Z4cToADM(pack);
  pack->pz4c->GaugePreCollapsedLapse(pack,pin);
}
