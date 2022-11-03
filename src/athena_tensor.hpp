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

#include "athena.hpp"

// tensor symmetries
enum class TensorSymm {
  NONE,     // no symmetries
  SYM2,     // symmetric in the last 2 indices
  ISYM2,    // symmetric in the first 2 indices
  SYM22,    // symmetric in the last 2 pairs of indices
};


using sub_DvceArray5D_2D = decltype(Kokkos::subview(std::declval<DvceArray5D<Real>>(),Kokkos::ALL,std::make_pair(0,5),Kokkos::ALL,Kokkos::ALL,Kokkos::ALL));
using sub_DvceArray5D_1D = decltype(Kokkos::subview(std::declval<DvceArray5D<Real>>(),Kokkos::ALL,std::make_pair(0,2),Kokkos::ALL,Kokkos::ALL,Kokkos::ALL));
using sub_DvceArray5D_0D = decltype(Kokkos::subview(std::declval<DvceArray5D<Real>>(),Kokkos::ALL,1,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL));


// this is the abstract base class
// This now works only for spatially 3D data
template<typename T, TensorSymm sym, int ndim, int rank>
class AthenaTensorField;


//----------------------------------------------------------------------------------------
// rank 0, dim 3 AthenaTensorField: 3D scalar fields
// This is simply a DvceArray3D
template<typename T, TensorSymm sym>
class AthenaTensorField<T, sym, 3, 0> {
public:
  // the default constructor/destructor/copy operators are sufficient
  AthenaTensorField() = default;
  ~AthenaTensorField() = default;
  AthenaTensorField(AthenaTensorField<T, sym, 3, 0> const &) = default;
  AthenaTensorField<T, sym, 3, 0> & operator=(AthenaTensorField<T, sym, 3, 0> const &) = default;
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
// rank 1, dim 3 AthenaTensorField: 3D vector and co-vector fields
// This is a 4D AthenaTensorField
template<typename T, TensorSymm sym>
class AthenaTensorField<T, sym, 3, 1> {
public:
  // the default constructor/destructor/copy operators are sufficient
  AthenaTensorField() = default;
  ~AthenaTensorField() = default;
  AthenaTensorField(AthenaTensorField<T, sym, 3, 1> const &) = default;
  AthenaTensorField<T, sym, 3, 1> & operator=(AthenaTensorField<T, sym, 3, 1> const &) = default;
  // operators to access the data
KOKKOS_INLINE_FUNCTION
    decltype(auto) operator() (int const m, int const a, 
                               int const k, int const j, int const i) const {
    return data_(m,a,k,j,i);
  }
//KOKKOS_INLINE_FUNCTION
  void InitWithShallowSlice(DvceArray5D<Real> src, const int indx1, const int indx2) {
    data_ = Kokkos::subview(src, Kokkos::ALL, std::make_pair(indx1, indx2),
                                 Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
  }
private:
  sub_DvceArray5D_1D data_;
};

//----------------------------------------------------------------------------------------
// rank 2, dim 3 AthenaTensorField, e.g., the metric or the extrinsic curvature
template<typename T, TensorSymm sym>
class AthenaTensorField<T, sym, 3, 2> {
public:
  AthenaTensorField();
  // the default destructor/copy operators are sufficient
  ~AthenaTensorField() = default;
  AthenaTensorField(AthenaTensorField<T, sym, 3, 2> const &) = default;
  AthenaTensorField<T, sym, 3, 2> & operator=(AthenaTensorField<T, sym, 3, 2> const &) = default;
  
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
    data_ = Kokkos::subview(src, Kokkos::ALL, std::make_pair(indx1, indx2),
                                 Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
  }
private:
  sub_DvceArray5D_2D data_;
  int idxmap_[3][3];
  int ndof_;
};

//----------------------------------------------------------------------------------------
// Implementation details
// They are all duplicated to account for dim 0, dim 3
#include <cassert>
template<typename T, TensorSymm sym>
AthenaTensorField<T, sym, 3, 2>::AthenaTensorField() {
  switch(sym) {
    case TensorSymm::NONE:
      ndof_ = 0;
      for(int a = 0; a < 3; ++a)
      for(int b = 0; b < 3; ++b) {
        idxmap_[a][b] = ndof_++;
      }
      break;
    case TensorSymm::SYM2:
    case TensorSymm::ISYM2:
      ndof_ = 0;
      for(int a = 0; a < 3; ++a)
      for(int b = a; b < 3; ++b) {
        idxmap_[a][b] = ndof_++;
        idxmap_[b][a] = idxmap_[a][b];
      }
      break;
/*
    default:
      assert(false); // you shouldn't be here
      abort();
*/
  }
}


// Here tensors are defined as static 1D arrays, with compile-time dimension calculated as 
// dim**rank
// this is the abstract base class
template<typename T, TensorSymm sym, int ndim, int rank>
class AthenaTensor;

//----------------------------------------------------------------------------------------
// rank 0, dim 1 AthenaTensor: spatially 0D vector and co-vector fields
// This is a 1D AthenaTensor
template<typename T, TensorSymm sym>
class AthenaTensor<T, sym, 1, 0> {
public:
  // the default constructor/destructor/copy operators are sufficient
  AthenaTensor() = default;
  ~AthenaTensor() = default;
  AthenaTensor(AthenaTensor<T, sym, 1, 0> const &) = default;
  AthenaTensor<T, sym, 1, 0> & operator=(AthenaTensor<T, sym, 1, 0> const &) = default;

KOKKOS_INLINE_FUNCTION
  decltype(auto) operator()(int const i) const {
    return data_(i);
  }
KOKKOS_INLINE_FUNCTION
  void NewAthenaTensor(const TeamMember_t &member, int scr_level, int nx) {
    data_ = ScrArray1D<T>(member.team_scratch(scr_level), nx);
  }
KOKKOS_INLINE_FUNCTION
  void ZeroClear() {
    Kokkos::Experimental::local_deep_copy(data_, 0.);
  }
private:
  ScrArray1D<T> data_;
};

//----------------------------------------------------------------------------------------
// rank 1, dim 1 AthenaTensor: spatially 0D vector and co-vector fields
// This is a 1D AthenaTensor
template<typename T, TensorSymm sym>
class AthenaTensor<T, sym, 1, 1> {
public:
  // the default constructor/destructor/copy operators are sufficient
  AthenaTensor() = default;
  ~AthenaTensor() = default;
  AthenaTensor(AthenaTensor<T, sym, 1, 1> const &) = default;
  AthenaTensor<T, sym, 1, 1> & operator=(AthenaTensor<T, sym, 1, 1> const &) = default;

KOKKOS_INLINE_FUNCTION
  decltype(auto) operator()(int const a, int const i) const {
    return data_(a, i);
  }
KOKKOS_INLINE_FUNCTION
  void NewAthenaTensor(const TeamMember_t & member, int scr_level, int nx) {
    data_ = ScrArray2D<T>(member.team_scratch(scr_level), 3, nx);
  }
KOKKOS_INLINE_FUNCTION
  void ZeroClear() {
    Kokkos::Experimental::local_deep_copy(data_, 0.);
  }
private:
  ScrArray2D<T> data_;
};
//----------------------------------------------------------------------------------------
// rank 2, dim 1 AthenaTensor
// This is a 1D AthenaTensor
template<typename T, TensorSymm sym>
class AthenaTensor<T, sym, 1, 2> {
public:
  // the default constructor/destructor/copy operators are sufficient
#ifdef __CUDA_ARCH__
__device__ __host__ AthenaTensor();
#else
  AthenaTensor();
#endif

  ~AthenaTensor() = default;
  AthenaTensor(AthenaTensor<T, sym, 1, 2> const &) = default;
  AthenaTensor<T, sym, 1, 2> & operator=(AthenaTensor<T, sym, 1, 2> const &) = default;
KOKKOS_INLINE_FUNCTION
  int idxmap(int const a, int const b) const {
    return idxmap_[a][b];
  }
KOKKOS_INLINE_FUNCTION
  decltype(auto) operator()(int const a, int const b, int const i) const {
    return data_(idxmap_[a][b], i);
  }
KOKKOS_INLINE_FUNCTION
  void NewAthenaTensor(const TeamMember_t & member, int scr_level, int nx) {
    data_ = ScrArray2D<T>(member.team_scratch(scr_level), ndof_, nx);
  }
KOKKOS_INLINE_FUNCTION
  void ZeroClear() {
    Kokkos::Experimental::local_deep_copy(data_, 0);
  }
private:
  ScrArray2D<T> data_;
  int idxmap_[3][3];
  int ndof_;
};
//----------------------------------------------------------------------------------------
// Implementation details
#include <cassert>
template<typename T, TensorSymm sym>
#ifdef __CUDA_ARCH__
__device__ __host__ AthenaTensor<T, sym, 1, 2>::AthenaTensor() {
#else
  AthenaTensor<T, sym, 1, 2>::AthenaTensor() {
#endif
switch(sym) {
    case TensorSymm::NONE:
      ndof_ = 0;
      for(int a = 0; a < 3; ++a)
      for(int b = 0; b < 3; ++b) {
        idxmap_[a][b] = ndof_++;
      }
      break;
    case TensorSymm::SYM2:
    case TensorSymm::ISYM2:
      ndof_ = 0;
      for(int a = 0; a < 3; ++a)
      for(int b = a; b < 3; ++b) {
        idxmap_[a][b] = ndof_++;
        idxmap_[b][a] = idxmap_[a][b];
      }
      break;
/*
    default:
      assert(false); // you shouldn't be here
      abort();
*/
  }
}
//----------------------------------------------------------------------------------------
// rank 3, dim 1 AthenaTensor
// This is a 1D AthenaTensor
template<typename T, TensorSymm sym>
class AthenaTensor<T, sym, 1, 3> {
public:
  // the default constructor/destructor/copy operators are sufficient
#ifdef __CUDA_ARCH__
__device__ __host__ AthenaTensor();
#else
  AthenaTensor();
#endif
  ~AthenaTensor() = default;
  AthenaTensor(AthenaTensor<T, sym, 1, 3> const &) = default;
  AthenaTensor<T, sym, 1, 3> & operator=(AthenaTensor<T, sym, 1, 3> const &) = default;
KOKKOS_INLINE_FUNCTION
  int idxmap(int const a, int const b, int const c) const {
    return idxmap_[a][b][c];
  }
KOKKOS_INLINE_FUNCTION
  decltype(auto) operator()(int const a, int const b, int const c, int const i) const {
    return data_(idxmap_[a][b][c], i);
  }
KOKKOS_INLINE_FUNCTION
  void NewAthenaTensor(const TeamMember_t & member, int scr_level, int nx) {
    data_ = ScrArray2D<T>(member.team_scratch(scr_level), ndof_, nx);
  }
KOKKOS_INLINE_FUNCTION
  void ZeroClear() {
    Kokkos::Experimental::local_deep_copy(data_, 0);
  }
private:
  ScrArray2D<T> data_;
  int idxmap_[3][3][3];
  int ndof_;
};
//----------------------------------------------------------------------------------------
// Implementation details
// They are all duplicated to account for dim 0, dim 3
#include <cassert>
template<typename T, TensorSymm sym>
#ifdef __CUDA_ARCH__
__device__ __host__ AthenaTensor<T, sym, 1, 3>::AthenaTensor() {
#else
  AthenaTensor<T, sym, 1, 3>::AthenaTensor() {
#endif
  switch(sym) {
    case TensorSymm::NONE:
      ndof_ = 0;
      for(int a = 0; a < 3; ++a)
      for(int b = 0; b < 3; ++b)
      for(int c = 0; c < 3; ++c) {
        idxmap_[a][b][c] = ndof_++;
      }
      break;
    case TensorSymm::SYM2:
      ndof_ = 0;
      for(int a = 0; a < 3; ++a)
      for(int b = 0; b < 3; ++b)
      for(int c = b; c < 3; ++c) {
        idxmap_[a][b][c] = ndof_++;
        idxmap_[a][c][b] = idxmap_[a][b][c];
      }
      break;
    case TensorSymm::ISYM2:
      ndof_ = 0;
      for(int a = 0; a < 3; ++a)
      for(int b = a; b < 3; ++b)
      for(int c = 0; c < 3; ++c) {
        idxmap_[a][b][c] = ndof_++;
        idxmap_[b][a][c] = idxmap_[a][b][c];
      }
      break;
/*
    default:
      assert(false); // you shouldn't be here
      abort();
*/
  }
}
//----------------------------------------------------------------------------------------
// rank 4, dim 1 AthenaTensor
// This is a 1D AthenaTensor
template<typename T, TensorSymm sym>
class AthenaTensor<T, sym, 1, 4> {
public:
  // the default constructor/destructor/copy operators are sufficient
#ifdef __CUDA_ARCH__
__device__ __host__ AthenaTensor();
#else
  AthenaTensor();
#endif
  ~AthenaTensor() = default;
  AthenaTensor(AthenaTensor<T, sym, 1, 4> const &) = default;
  AthenaTensor<T, sym, 1, 4> & operator=(AthenaTensor<T, sym, 1, 4> const &) = default;
KOKKOS_INLINE_FUNCTION
  int idxmap(int const a, int const b, int const c, int const d) const {
    return idxmap_[a][b][c][d];
  }
KOKKOS_INLINE_FUNCTION
  decltype(auto) operator()(int const a, int const b, int const c, int const d, int const i) const {
    return data_(idxmap_[a][b][c][d], i);
  }
KOKKOS_INLINE_FUNCTION
  void NewAthenaTensor(const TeamMember_t & member, int scr_level, int nx) {
    data_ = ScrArray2D<T>(member.team_scratch(scr_level), ndof_, nx);
  }
KOKKOS_INLINE_FUNCTION
  void ZeroClear() {
    Kokkos::Experimental::local_deep_copy(data_, 0);
  }
private:
  ScrArray2D<T> data_;
  int idxmap_[3][3][3][3];
  int ndof_;
};
//----------------------------------------------------------------------------------------
// Implementation details
// They are all duplicated to account for dim 0, dim 3
#include <cassert>
template<typename T, TensorSymm sym>
#ifdef __CUDA_ARCH__
__device__ __host__ AthenaTensor<T, sym, 1, 4>::AthenaTensor() {
#else
  AthenaTensor<T, sym, 1, 4>::AthenaTensor() {
#endif
  switch(sym) {
    case TensorSymm::NONE:
      ndof_ = 0;
      for(int a = 0; a < 3; ++a)
      for(int b = 0; b < 3; ++b)
      for(int c = 0; c < 3; ++c)
      for(int d = 0; d < 3; ++d) {
        idxmap_[a][b][c][d] = ndof_++;
      }
      break;
    case TensorSymm::SYM2:
      ndof_ = 0;
      for(int a = 0; a < 3; ++a)
      for(int b = 0; b < 3; ++b)
      for(int c = 0; c < 3; ++c)
      for(int d = c; d < 3; ++d) {
        idxmap_[a][b][c][d] = ndof_++;
        idxmap_[a][b][d][c] = idxmap_[a][b][c][d];
      }
      break;
    case TensorSymm::ISYM2:
      ndof_ = 0;
      for(int a = 0; a < 3; ++a)
      for(int b = a; b < 3; ++b)
      for(int c = 0; c < 3; ++c)
      for(int d = 0; d < 3; ++d) {
        idxmap_[a][b][c][d] = ndof_++;
        idxmap_[b][a][c][d] = idxmap_[a][b][c][d];
      }
      break;
    case TensorSymm::SYM22:
      ndof_ = 0;
      for(int a = 0; a < 3; ++a)
      for(int b = a; b < 3; ++b)
      for(int c = 0; c < 3; ++c)
      for(int d = c; d < 3; ++d) {
        idxmap_[a][b][c][d] = ndof_++;
        idxmap_[b][a][c][d] = idxmap_[a][b][c][d];
        idxmap_[a][b][d][c] = idxmap_[a][b][c][d];
        idxmap_[b][a][d][c] = idxmap_[a][b][c][d];
      }
      break;
/*
    default:
      assert(false); // you shouldn't be here
      abort();
*/
  }
}

#endif
