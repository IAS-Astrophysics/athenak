#ifndef ATHENA_TENSOR_HPP_
#define ATHENA_TENSOR_HPP_
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file athena_tensor.hpp
//  \brief provides classes for tensor-like fields
//
//  Convention: indices a,b,c,d are tensor indices. Indices n,i,j,k are grid indices.

#include <cassert> // assert
#include <utility>
#include "athena.hpp"

// tensor symmetries
enum class TensorSymm {
  NONE,     // no symmetries
  SYM2,     // symmetric in the last 2 indices
  ISYM2,    // symmetric in the first 2 indices
  SYM22,    // symmetric in the last 2 pairs of indices
};


using sub_DvceArray5D_2D = decltype(Kokkos::subview(
                           std::declval<DvceArray5D<Real>>(),
                           Kokkos::ALL,std::make_pair(0,6),
                           Kokkos::ALL,Kokkos::ALL,Kokkos::ALL));
using sub_DvceArray5D_1D = decltype(Kokkos::subview(
                           std::declval<DvceArray5D<Real>>(),
                           Kokkos::ALL,std::make_pair(0,3),
                           Kokkos::ALL,Kokkos::ALL,Kokkos::ALL));
using sub_DvceArray5D_0D = decltype(Kokkos::subview(
                           std::declval<DvceArray5D<Real>>(),
                           Kokkos::ALL,1,
                           Kokkos::ALL,Kokkos::ALL,Kokkos::ALL));

using sub_HostArray5D_2D = decltype(Kokkos::subview(
                           std::declval<HostArray5D<Real>>(),
                           Kokkos::ALL,std::make_pair(0,6),
                           Kokkos::ALL,Kokkos::ALL,Kokkos::ALL));
using sub_HostArray5D_1D = decltype(Kokkos::subview(
                           std::declval<HostArray5D<Real>>(),
                           Kokkos::ALL,std::make_pair(0,3),
                           Kokkos::ALL,Kokkos::ALL,Kokkos::ALL));
using sub_HostArray5D_0D = decltype(Kokkos::subview(
                           std::declval<HostArray5D<Real>>(),
                           Kokkos::ALL,1,
                           Kokkos::ALL,Kokkos::ALL,Kokkos::ALL));

// this is the abstract base class
// This now works only for spatially 3D data
template<typename T, TensorSymm sym, int ndim, int rank>
class AthenaHostTensor;

//----------------------------------------------------------------------------------------
// rank 0 AthenaHostTensor: 3D scalar fields
// This is simply a DvceArray3D
template<typename T, TensorSymm sym, int ndim>
class AthenaHostTensor<T, sym, ndim, 0> {
 public:
  // the default constructor/destructor/copy operators are sufficient
  AthenaHostTensor() = default;
  ~AthenaHostTensor() = default;
  AthenaHostTensor(AthenaHostTensor<T, sym, ndim, 0> const &) = default;
  AthenaHostTensor<T, sym, ndim, 0> & operator=
  (AthenaHostTensor<T, sym, ndim, 0> const &) = default;
  // operators to access the data
  KOKKOS_INLINE_FUNCTION
  decltype(auto) operator() (int const m, int const k, int const j, int const i) const {
    return data_(m,k,j,i);
  }
  //KOKKOS_INLINE_FUNCTION
  void InitWithShallowSlice(HostArray5D<Real> src, const int indx) {
    data_ = Kokkos::subview(src,Kokkos::ALL,indx,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
  }

 private:
  sub_HostArray5D_0D data_;
};
//----------------------------------------------------------------------------------------
// rank 1 AthenaTensor, e.g., the lapse
template<typename T, TensorSymm sym, int ndim>
class AthenaHostTensor<T, sym, ndim, 1> {
 public:
  // the default constructor/destructor/copy operators are sufficient
  AthenaHostTensor() = default;
  ~AthenaHostTensor() = default;
  AthenaHostTensor(AthenaHostTensor<T, sym, ndim, 1> const &) = default;
  AthenaHostTensor<T, sym, ndim, 1> & operator =
  (AthenaHostTensor<T, sym, ndim, 1> const &) = default;
  // operators to access the data
  KOKKOS_INLINE_FUNCTION
  decltype(auto) operator() (int const m, int const a,
                             int const k, int const j, int const i) const {
    return data_(m,a,k,j,i);
  }
  void InitWithShallowSlice(HostArray5D<Real> src, const int indx1, const int indx2) {
    data_ = Kokkos::subview(src, Kokkos::ALL, std::make_pair(indx1, indx2+1),
                                 Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
  }

 private:
  sub_HostArray5D_1D data_;
};
//----------------------------------------------------------------------------------------
// rank 2 AthenaTensor, e.g., the metric or the extrinsic curvature
template<typename T, TensorSymm sym, int ndim>
class AthenaHostTensor<T, sym, ndim, 2> {
 public:
  AthenaHostTensor();
  // the default destructor/copy operators are sufficient
  ~AthenaHostTensor() = default;
  AthenaHostTensor(AthenaHostTensor<T, sym, ndim, 2> const &) = default;
  AthenaHostTensor<T, sym, ndim, 2> & operator=
  (AthenaHostTensor<T, sym, ndim, 2> const &) = default;

  int idxmap(int const a, int const b) const {
    return idxmap_[a][b];
  }
  // operators to access the data
  KOKKOS_INLINE_FUNCTION
  decltype(auto) operator() (int const m, int const a, int const b,
                             int const k, int const j, int const i) const {
    return data_(m,idxmap_[a][b],k,j,i);
  }
  //KOKKOS_INLINE_FUNCTION
  void InitWithShallowSlice(HostArray5D<Real> src, const int indx1, const int indx2) {
    data_ = Kokkos::subview(src, Kokkos::ALL, std::make_pair(indx1, indx2+1),
                                 Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
  }

 private:
  sub_HostArray5D_2D data_;
  int idxmap_[3][3];
  int ndof_;
};

//----------------------------------------------------------------------------------------
// Implementation details
// They are all duplicated to account for dim 0
template<typename T, TensorSymm sym, int ndim>
AthenaHostTensor<T, sym, ndim, 2>::AthenaHostTensor() {
  switch(sym) {
    case TensorSymm::NONE:
      ndof_ = 0;
      for(int a = 0; a < ndim; ++a)
      for(int b = 0; b < ndim; ++b) {
        idxmap_[a][b] = ndof_++;
      }
      break;
    case TensorSymm::SYM2:
    case TensorSymm::ISYM2:
      ndof_ = 0;
      for(int a = 0; a < ndim; ++a)
      for(int b = a; b < ndim; ++b) {
        idxmap_[a][b] = ndof_++;
        idxmap_[b][a] = idxmap_[a][b];
      }
      break;
  }
}


// this is the abstract base class
// This now works only for spatially 3D data
template<typename T, TensorSymm sym, int ndim, int rank>
class AthenaTensor;

//----------------------------------------------------------------------------------------
// rank 0 AthenaTensor: 3D scalar fields
// This is simply a DvceArray3D
template<typename T, TensorSymm sym, int ndim>
class AthenaTensor<T, sym, ndim, 0> {
 public:
  // the default constructor/destructor/copy operators are sufficient
  AthenaTensor() = default;
  ~AthenaTensor() = default;
  AthenaTensor(AthenaTensor<T, sym, ndim, 0> const &) = default;
  AthenaTensor<T, sym, ndim, 0> & operator=
  (AthenaTensor<T, sym, ndim, 0> const &) = default;
  // operators to access the data
  KOKKOS_INLINE_FUNCTION
  decltype(auto) operator() (int const m, int const k, int const j, int const i) const {
    return data_(m,k,j,i);
  }
  //KOKKOS_INLINE_FUNCTION
  void InitWithShallowSlice(DvceArray5D<Real> src, const int indx) {
    data_ = Kokkos::subview(src,Kokkos::ALL,indx,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
  }
 private:
  sub_DvceArray5D_0D data_;
};

//----------------------------------------------------------------------------------------
// rank 1 AthenaTensor: 3D vector and co-vector fields
// This is a 4D AthenaTensor
template<typename T, TensorSymm sym, int ndim>
class AthenaTensor<T, sym, ndim, 1> {
 public:
  // the default constructor/destructor/copy operators are sufficient
  AthenaTensor() = default;
  ~AthenaTensor() = default;
  AthenaTensor(AthenaTensor<T, sym, ndim, 1> const &) = default;
  AthenaTensor<T, sym, ndim, 1> & operator=
  (AthenaTensor<T, sym, ndim, 1> const &) = default;
  // operators to access the data
  KOKKOS_INLINE_FUNCTION
  decltype(auto) operator() (int const m, int const a,
                             int const k, int const j, int const i) const {
    return data_(m,a,k,j,i);
  }
  //KOKKOS_INLINE_FUNCTION
  void InitWithShallowSlice(DvceArray5D<Real> src, const int indx1, const int indx2) {
    data_ = Kokkos::subview(src, Kokkos::ALL, std::make_pair(indx1, indx2+1),
                                 Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
  }
 private:
  sub_DvceArray5D_1D data_;
};

//----------------------------------------------------------------------------------------
// rank 2 AthenaTensor, e.g., the metric or the extrinsic curvature
template<typename T, TensorSymm sym, int ndim>
class AthenaTensor<T, sym, ndim, 2> {
 public:
  AthenaTensor();
  // the default destructor/copy operators are sufficient
  ~AthenaTensor() = default;
  AthenaTensor(AthenaTensor<T, sym, ndim, 2> const &) = default;
  AthenaTensor<T, sym, ndim, 2> & operator=
  (AthenaTensor<T, sym, ndim, 2> const &) = default;

  int idxmap(int const a, int const b) const {
    return idxmap_[a][b];
  }
  // operators to access the data
  KOKKOS_INLINE_FUNCTION
  decltype(auto) operator() (int const m, int const a, int const b,
                             int const k, int const j, int const i) const {
    return data_(m,idxmap_[a][b],k,j,i);
  }
  //KOKKOS_INLINE_FUNCTION
  void InitWithShallowSlice(DvceArray5D<Real> src, const int indx1, const int indx2) {
    data_ = Kokkos::subview(src, Kokkos::ALL, std::make_pair(indx1, indx2+1),
                                 Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
  }

 private:
  sub_DvceArray5D_2D data_;
  int idxmap_[3][3];
  int ndof_;
};

//----------------------------------------------------------------------------------------
// Implementation details
// They are all duplicated to account for dim 0
template<typename T, TensorSymm sym, int ndim>
AthenaTensor<T, sym, ndim, 2>::AthenaTensor() {
  switch(sym) {
    case TensorSymm::NONE:
      ndof_ = 0;
      for(int a = 0; a < ndim; ++a)
      for(int b = 0; b < ndim; ++b) {
        idxmap_[a][b] = ndof_++;
      }
      break;
    case TensorSymm::SYM2:
    case TensorSymm::ISYM2:
      ndof_ = 0;
      for(int a = 0; a < ndim; ++a)
      for(int b = a; b < ndim; ++b) {
        idxmap_[a][b] = ndof_++;
        idxmap_[b][a] = idxmap_[a][b];
      }
      break;
  }
}


// Here tensors are defined as static 1D arrays, with compile-time dimension calculated as
// dim**rank
// this is the abstract base class
template<typename T, TensorSymm sym, int ndim, int rank>
class AthenaScratchTensor;

//----------------------------------------------------------------------------------------
// rank 1 AthenaScratchTensor: spatially 0D vector and co-vector fields
// This is a 1D AthenaScratchTensor
template<typename T, TensorSymm sym, int ndim>
class AthenaScratchTensor<T, sym, ndim, 1> {
 public:
  // the default constructor/destructor/copy operators are sufficient
  AthenaScratchTensor() = default;
  ~AthenaScratchTensor() = default;
  AthenaScratchTensor(AthenaScratchTensor<T, sym, ndim, 1> const &) = default;
  AthenaScratchTensor<T, sym, ndim, 1> & operator=
  (AthenaScratchTensor<T, sym, ndim, 1> const &) = default;

  KOKKOS_INLINE_FUNCTION
  Real operator()(int const a) const {
    return data_[a];
  }
  KOKKOS_INLINE_FUNCTION
  Real & operator()(int const a) {
    return data_[a];
  }
  KOKKOS_INLINE_FUNCTION
  void ZeroClear() {
    for (int i = 0; i < ndim; ++i) {
      data_[i] = 0;
    }
  }

 private:
  Real data_[3];
};

//----------------------------------------------------------------------------------------
// rank 2 AthenaScratchTensor
// This is a 0D AthenaScratchTensor
template<typename T, TensorSymm sym, int ndim>
class AthenaScratchTensor<T, sym, ndim, 2> {
 public:
  KOKKOS_INLINE_FUNCTION
  AthenaScratchTensor();
  // the default destructor/copy operators are sufficient
  ~AthenaScratchTensor() = default;
  AthenaScratchTensor(AthenaScratchTensor<T, sym, ndim, 2> const &) = default;
  AthenaScratchTensor<T, sym, ndim, 2> & operator=
  (AthenaScratchTensor<T, sym, ndim, 2> const &) = default;
  KOKKOS_INLINE_FUNCTION
  int idxmap(int const a, int const b) const {
    return idxmap_[a][b];
  }
  KOKKOS_INLINE_FUNCTION
  Real operator()(int const a, int const b) const {
    return data_[idxmap_[a][b]];
  }
  KOKKOS_INLINE_FUNCTION
  Real & operator()(int const a, int const b) {
    return data_[idxmap_[a][b]];
  }
  KOKKOS_INLINE_FUNCTION
  void ZeroClear() {
    for (int i = 0; i < ndim*ndim; ++i) {
      data_[i] = 0.0;
    }
  }

 private:
  Real data_[9];
  int idxmap_[3][3];
  int ndof_;
};

//----------------------------------------------------------------------------------------
// Implementation details
template<typename T, TensorSymm sym, int ndim>
KOKKOS_INLINE_FUNCTION
AthenaScratchTensor<T, sym, ndim, 2>::AthenaScratchTensor() {
switch(sym) {
    case TensorSymm::NONE:
      ndof_ = 0;
      for(int a = 0; a < ndim; ++a)
      for(int b = 0; b < ndim; ++b) {
        idxmap_[a][b] = ndof_++;
      }
      break;
    case TensorSymm::SYM2:
    case TensorSymm::ISYM2:
      ndof_ = 0;
      for(int a = 0; a < ndim; ++a)
      for(int b = a; b < ndim; ++b) {
        idxmap_[a][b] = ndof_++;
        idxmap_[b][a] = idxmap_[a][b];
      }
      break;
  }
}

//----------------------------------------------------------------------------------------
// rank 3 AthenaScratchTensor
// This is a 0D AthenaScratchTensor
template<typename T, TensorSymm sym, int ndim>
class AthenaScratchTensor<T, sym, ndim, 3> {
 public:
  KOKKOS_INLINE_FUNCTION
  AthenaScratchTensor();
  // the default destructor/copy operators are sufficient
  ~AthenaScratchTensor() = default;
  AthenaScratchTensor(AthenaScratchTensor<T, sym, ndim, 3> const &) = default;
  AthenaScratchTensor<T, sym, ndim, 3> & operator=
  (AthenaScratchTensor<T, sym, ndim, 3> const &) = default;
  KOKKOS_INLINE_FUNCTION
  int idxmap(int const a, int const b, int const c) const {
    return idxmap_[a][b][c];
  }
  KOKKOS_INLINE_FUNCTION
  Real operator()(int const a, int const b, int const c) const {
    return data_[idxmap_[a][b][c]];
  }
  KOKKOS_INLINE_FUNCTION
  Real & operator()(int const a, int const b, int const c) {
    return data_[idxmap_[a][b][c]];
  }
  KOKKOS_INLINE_FUNCTION
  void ZeroClear() {
    for (int i = 0; i < ndim*ndim*ndim; ++i) {
      data_[i] = 0.0;
    }
  }

 private:
  Real data_[27];
  int idxmap_[3][3][3];
  int ndof_;
};

//----------------------------------------------------------------------------------------
// Implementation details
template<typename T, TensorSymm sym, int ndim>
KOKKOS_INLINE_FUNCTION
AthenaScratchTensor<T, sym, ndim, 3>::AthenaScratchTensor() {
  switch(sym) {
    case TensorSymm::NONE:
      ndof_ = 0;
      for(int a = 0; a < ndim; ++a)
      for(int b = 0; b < ndim; ++b)
      for(int c = 0; c < ndim; ++c) {
        idxmap_[a][b][c] = ndof_++;
      }
      break;
    case TensorSymm::SYM2:
      ndof_ = 0;
      for(int a = 0; a < ndim; ++a)
      for(int b = 0; b < ndim; ++b)
      for(int c = b; c < ndim; ++c) {
        idxmap_[a][b][c] = ndof_++;
        idxmap_[a][c][b] = idxmap_[a][b][c];
      }
      break;
    case TensorSymm::ISYM2:
      ndof_ = 0;
      for(int a = 0; a < ndim; ++a)
      for(int b = a; b < ndim; ++b)
      for(int c = 0; c < ndim; ++c) {
        idxmap_[a][b][c] = ndof_++;
        idxmap_[b][a][c] = idxmap_[a][b][c];
      }
      break;
  }
}

//----------------------------------------------------------------------------------------
// rank 4 AthenaScratchTensor
// This is a 0D AthenaScratchTensor
template<typename T, TensorSymm sym, int ndim>
class AthenaScratchTensor<T, sym, ndim, 4> {
 public:
  KOKKOS_INLINE_FUNCTION
  AthenaScratchTensor();
  // the default destructor/copy operators are sufficient
  ~AthenaScratchTensor() = default;
  AthenaScratchTensor(AthenaScratchTensor<T, sym, ndim, 4> const &) = default;
  AthenaScratchTensor<T, sym, ndim, 4> & operator=
  (AthenaScratchTensor<T, sym, ndim, 4> const &) = default;
  KOKKOS_INLINE_FUNCTION
  int idxmap(int const a, int const b, int const c, int const d) const {
    return idxmap_[a][b][c][d];
  }
  KOKKOS_INLINE_FUNCTION
  Real operator()(int const a, int const b,
                  int const c, int const d) const {
    return data_[idxmap_[a][b][c][d]];
  }
  KOKKOS_INLINE_FUNCTION
  Real & operator()(int const a, int const b,
                    int const c, int const d) {
    return data_[idxmap_[a][b][c][d]];
  }
  KOKKOS_INLINE_FUNCTION
  void ZeroClear() {
    for (int i = 0; i < ndim*ndim*ndim*ndim; ++i) {
      data_[i] = 0.0;
    }
  }

 private:
  Real data_[81];
  int idxmap_[3][3][3][3];
  int ndof_;
};

//----------------------------------------------------------------------------------------
// Implementation details
template<typename T, TensorSymm sym, int ndim>
KOKKOS_INLINE_FUNCTION
AthenaScratchTensor<T, sym, ndim, 4>::AthenaScratchTensor() {
  switch(sym) {
    case TensorSymm::NONE:
      ndof_ = 0;
      for(int a = 0; a < ndim; ++a)
      for(int b = 0; b < ndim; ++b)
      for(int c = 0; c < ndim; ++c)
      for(int d = 0; d < ndim; ++d) {
        idxmap_[a][b][c][d] = ndof_++;
      }
      break;
    case TensorSymm::SYM2:
      ndof_ = 0;
      for(int a = 0; a < ndim; ++a)
      for(int b = 0; b < ndim; ++b)
      for(int c = 0; c < ndim; ++c)
      for(int d = c; d < ndim; ++d) {
        idxmap_[a][b][c][d] = ndof_++;
        idxmap_[a][b][d][c] = idxmap_[a][b][c][d];
      }
      break;
    case TensorSymm::ISYM2:
      ndof_ = 0;
      for(int a = 0; a < ndim; ++a)
      for(int b = a; b < ndim; ++b)
      for(int c = 0; c < ndim; ++c)
      for(int d = 0; d < ndim; ++d) {
        idxmap_[a][b][c][d] = ndof_++;
        idxmap_[b][a][c][d] = idxmap_[a][b][c][d];
      }
      break;
    case TensorSymm::SYM22:
      ndof_ = 0;
      for(int a = 0; a < ndim; ++a)
      for(int b = a; b < ndim; ++b)
      for(int c = 0; c < ndim; ++c)
      for(int d = c; d < ndim; ++d) {
        idxmap_[a][b][c][d] = ndof_++;
        idxmap_[b][a][c][d] = idxmap_[a][b][c][d];
        idxmap_[a][b][d][c] = idxmap_[a][b][c][d];
        idxmap_[b][a][d][c] = idxmap_[a][b][c][d];
      }
      break;
  }
}

#endif // ATHENA_TENSOR_HPP_
