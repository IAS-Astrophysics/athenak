#ifndef ATHENA_ARRAYS_HPP_
#define ATHENA_ARRAYS_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file athena_arrays.hpp
//  \brief provides array classes valid in 1D to 6D.
//
//  The operator() is overloaded, e.g. elements of a 4D array of size [N4xN3xN2xN1]
//  are accessed as:  A(n,k,j,i) = A[i + N1*(j + N2*(k + N3*n))]
//  NOTE THE TRAILING INDEX INSIDE THE PARENTHESES IS INDEXED FASTEST

#include <cstddef> 

template <typename T>
class AthenaCenterArray {
 public:

  // ctors with various arguments
  // "label" argument added for compatibility with Kokkos view API
  AthenaCenterArray() : label_(""),nx1_(0),nx2_(0),nx3_(0),nx4_(0),nx5_(0),nx6_(0) {}

  AthenaCenterArray(const std::string &label, int nx6, int nx5, int nx4, int nx3, int nx2,
                    int nx1) :
    label_(label), nx1_(nx1), nx2_(nx2), nx3_(nx3), nx4_(nx4), nx5_(nx5), nx6_(nx6) {
    assert(nx6 > 0 && nx5 > 0 && nx4 > 0 && nx3 > 0 && nx2 > 0 && nx1 > 0);
    pdata_ = new T[nx1_*nx2_*nx3_*nx4_*nx5_*nx6_]();
  }
  AthenaCenterArray(const std::string &label, int nx5, int nx4, int nx3, int nx2,
                    int nx1) :
    label_(label), nx1_(nx1), nx2_(nx2), nx3_(nx3), nx4_(nx4), nx5_(nx5), nx6_(1) {
    assert(nx5 > 0 && nx4 > 0 && nx3 > 0 && nx2 > 0 && nx1 > 0);
    pdata_ = new T[nx1_*nx2_*nx3_*nx4_*nx5_*nx6_]();
  }
  AthenaCenterArray(const std::string &label, int nx4, int nx3, int nx2, int nx1) :
    label_(label), nx1_(nx1), nx2_(nx2), nx3_(nx3), nx4_(nx4), nx5_(1), nx6_(1) {
    assert(nx4 > 0 && nx3 > 0 && nx2 > 0 && nx1 > 0);
    pdata_ = new T[nx1_*nx2_*nx3_*nx4_*nx5_*nx6_]();
  }
  AthenaCenterArray(const std::string &label, int nx3, int nx2, int nx1) :
    label_(label), nx1_(nx1), nx2_(nx2), nx3_(nx3), nx4_(1), nx5_(1), nx6_(1) {
    assert(nx3 > 0 && nx2 > 0 && nx1 > 0);
    pdata_ = new T[nx1_*nx2_*nx3_*nx4_*nx5_*nx6_]();
  }
  AthenaCenterArray(const std::string &label, int nx2, int nx1) :
    label_(label), nx1_(nx1), nx2_(nx2), nx3_(1), nx4_(1), nx5_(1), nx6_(1) {
    assert(nx2 > 0 && nx1 > 0);
    pdata_ = new T[nx1_*nx2_*nx3_*nx4_*nx5_*nx6_]();
  }
  AthenaCenterArray(const std::string &label, int nx1) :
    label_(label), nx1_(nx1), nx2_(1), nx3_(1), nx4_(1), nx5_(1), nx6_(1) {
    assert(nx1 > 0);
    pdata_ = new T[nx1_*nx2_*nx3_*nx4_*nx5_*nx6_]();
  }

  // dtor, copy & move ctors, move and overload assignment operators defined below
  __attribute__((nothrow)) AthenaCenterArray(const AthenaCenterArray<T> &t);
  __attribute__((nothrow)) ~AthenaCenterArray();
  __attribute__((nothrow)) AthenaCenterArray<T> &operator=(const AthenaCenterArray<T> &t);
  __attribute__((nothrow)) AthenaCenterArray(AthenaCenterArray<T> &&t);
  __attribute__((nothrow)) AthenaCenterArray<T> &operator=
                           (AthenaCenterArray<T> &&t);

  // getter functions
  int GetDim(int i) const {
    assert (i>0 && i<7);
    if (i==1) { return nx1_; }
    else if (i==2) { return nx2_; }
    else if (i==3) { return nx3_; }
    else if (i==4) { return nx4_; }
    else if (i==5) { return nx5_; }
    else { return nx6_; }
  }
  int GetSize() const { return nx1_*nx2_*nx3_*nx4_*nx5_*nx6_; }
  std::size_t GetSizeInBytes() const { return nx1_*nx2_*nx3_*nx4_*nx5_*nx6_*sizeof(T); }

  // "getter" function to access private data member
  // TODO(felker): Replace this unrestricted "getter" with a limited, safer alternative.
  // TODO(felker): Rename function. Conflicts with "AthenaArray<> data" OutputData member.
//  T *data() { return pdata_; }
//  const T *data() const { return pdata_; }

  // overload "function call" operator() to access 1d-6d data
  // provides Fortran-like syntax for multidimensional arrays vs. "subscript" operator[]
  // "const" and "non-const" variants provided for return by value or reference, rspctvly
  T &operator() (const int i) { return pdata_[i]; }
  T operator() (const int i) const { return pdata_[i]; }

  T &operator() (const int j, const int i) { return pdata_[i + nx1_*j]; }
  T operator() (const int j, const int i) const { return pdata_[i + nx1_*j]; }

  T &operator() (const int k, const int j, const int i) {
    return pdata_[i + nx1_*(j + nx2_*k)]; }
  T operator() (const int k, const int j, const int i) const {
    return pdata_[i + nx1_*(j + nx2_*k)]; }

  T &operator() (const int l, const int k, const int j, const int i) {
    return pdata_[i + nx1_*(j + nx2_*(k + nx3_*l))]; }
  T operator() (const int l, const int k, const int j, const int i) const {
    return pdata_[i + nx1_*(j + nx2_*(k + nx3_*l))]; }

  T &operator() (const int m, const int l, const int k, const int j, const int i) {
    return pdata_[i + nx1_*(j + nx2_*(k + nx3_*(l + nx4_*m)))]; }
  T operator() (const int m, const int l, const int k, const int j, const int i) const {
    return pdata_[i + nx1_*(j + nx2_*(k + nx3_*(l + nx4_*m)))]; }

  // int l?, int o?
  T &operator() (const int n, const int m, const int l, const int k, const int j,
                 const int i) {
    return pdata_[i + nx1_*(j + nx2_*(k + nx3_*(l + nx4_*(m + nx5_*n))))]; }
  T operator() (const int n, const int m, const int l, const int k, const int j,
                const int i) const {
    return pdata_[i + nx1_*(j + nx2_*(k + nx3_*(l + nx4_*(m + nx5_*n))))]; }

 private:
  T *pdata_;
  int nx1_, nx2_, nx3_, nx4_, nx5_, nx6_;
  std::string label_;
};


//----------------------------------------------------------------------------------------
// destructor

template<typename T>
AthenaCenterArray<T>::~AthenaCenterArray() {
  delete[] pdata_;
}

//----------------------------------------------------------------------------------------
// copy constructor (does a deep copy)

template<typename T>
__attribute__((nothrow)) 
AthenaCenterArray<T>::AthenaCenterArray(const AthenaCenterArray<T>& src) {
  nx1_ = src.nx1_;
  nx2_ = src.nx2_;
  nx3_ = src.nx3_;
  nx4_ = src.nx4_;
  nx5_ = src.nx5_;
  nx6_ = src.nx6_;
  if (src.pdata_) {
    std::size_t size = (src.nx1_)*(src.nx2_)*(src.nx3_)*(src.nx4_)*(src.nx5_);
    pdata_ = new T[size];        // allocate memory for array data
    for (std::size_t i=0; i<size; ++i) {
      pdata_[i] = src.pdata_[i]; // copy data (not just addresses!) into new memory
    }
  }
}

//----------------------------------------------------------------------------------------
// copy assignment operator (does a deep copy). Does not allocate memory for destination.
// THIS REQUIRES THAT THE DESTINATION ARRAY IS ALREADY ALLOCATED & THE SAME SIZE AS SOURCE

template<typename T>
__attribute__((nothrow))
AthenaCenterArray<T> &AthenaCenterArray<T>::operator= (const AthenaCenterArray<T> &src) {
  assert (nx1_ == src.nx1_ && nx2_ == src.nx2_ && nx3_ == src.nx3_ && 
          nx4_ == src.nx4_ && nx5_ == src.nx5_ && nx6_ == src.nx6_);
  if (this != &src) {
    std::size_t size = (src.nx1_)*(src.nx2_)*(src.nx3_)*(src.nx4_)*(src.nx5_)*(src.nx6_);
    for (std::size_t i=0; i<size; ++i) {
      this->pdata_[i] = src.pdata_[i]; // copy data (not just addresses!)
    }
  }
  return *this;
}

//----------------------------------------------------------------------------------------
// move constructor (transfers ownership of data)

template<typename T>
__attribute__((nothrow)) 
AthenaCenterArray<T>::AthenaCenterArray(AthenaCenterArray<T>&& src) {
  nx1_ = src.nx1_;
  nx2_ = src.nx2_;
  nx3_ = src.nx3_;
  nx4_ = src.nx4_;
  nx5_ = src.nx5_;
  nx6_ = src.nx6_;
  if (src.pdata_) {
    pdata_ = src.pdata_;
    // remove ownership of data from src to prevent it from free'ing the resources
    src.pdata_ = nullptr;
    src.nx1_ = 0;
    src.nx2_ = 0;
    src.nx3_ = 0;
    src.nx4_ = 0;
    src.nx5_ = 0;
    src.nx6_ = 0;
  }
}

//----------------------------------------------------------------------------------------
// move assignment operator (transfers ownership of data)

template<typename T>
__attribute__((nothrow))
AthenaCenterArray<T> &AthenaCenterArray<T>::operator= (AthenaCenterArray<T> &&src) {
  if (this != &src) {
    delete [] pdata_;   // free the target to prepare to receive src pdata_
    if (src.pdata_) {
      nx1_ = src.nx1_;
      nx2_ = src.nx2_;
      nx3_ = src.nx3_;
      nx4_ = src.nx4_;
      nx5_ = src.nx5_;
      nx6_ = src.nx6_;
      pdata_ = src.pdata_;

      src.pdata_ = nullptr;
      src.nx1_ = 0;
      src.nx2_ = 0;
      src.nx3_ = 0;
      src.nx4_ = 0;
      src.nx5_ = 0;
      src.nx6_ = 0;
    }
  }
  return *this;
}

#endif // ATHENA_ARRAYS_HPP_
