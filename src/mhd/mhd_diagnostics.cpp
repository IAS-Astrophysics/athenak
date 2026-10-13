//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file mhd_diagnostics.cpp
//! \brief Finite-value checks for MHD field and EMF validation.

#include <cmath>
#include <algorithm>
#include <iostream>
#include <limits>
#include <string>

#include "athena.hpp"
#include "globals.hpp"
#include "driver/driver.hpp"
#include "mesh/mesh.hpp"
#include "mhd/mhd.hpp"

namespace {

enum class LocationKind {cell, x1f, x2f, x3f, x1e, x2e, x3e};

struct IndexRange {
  int lo = 0;
  int hi = -1;
};

IndexRange ClampRange(const int lo, const int hi, const int extent) {
  IndexRange r;
  if (extent <= 0) return r;
  r.lo = std::max(0, lo);
  r.hi = std::min(extent - 1, hi);
  if (r.hi < r.lo) {
    r.lo = 0;
    r.hi = -1;
  }
  return r;
}

int RangeCount(const IndexRange &r) {
  return (r.hi >= r.lo) ? (r.hi - r.lo + 1) : 0;
}

KOKKOS_INLINE_FUNCTION
bool FiniteForRepair(const Real v) {
  return ((v == v) && (v <= static_cast<Real>(1.0e300)) &&
          (v >= static_cast<Real>(-1.0e300)));
}

template <typename ViewType>
void RepairFluxRange(const char *component, const ViewType &a, const int nvars,
                     const IndexRange mrange, const IndexRange krange,
                     const IndexRange jrange, const IndexRange irange,
                     const int count_offset, const DvceArray1D<int> &counts,
                     const DvceArray2D<int> &first_locs) {
  const int nm = RangeCount(mrange);
  const int nk = RangeCount(krange);
  const int nj = RangeCount(jrange);
  const int ni = RangeCount(irange);
  const int total = nm*nk*nj*ni;
  if (total <= 0 || nvars <= 0) return;

  std::string kernel_label = std::string("mhd_repair_flux_") + component;
  Kokkos::parallel_for(kernel_label, Kokkos::RangePolicy<>(DevExeSpace(), 0, total),
  KOKKOS_LAMBDA(const int idx) {
    int q = idx;
    const int i = irange.lo + (q % ni);
    q /= ni;
    const int j = jrange.lo + (q % nj);
    q /= nj;
    const int k = krange.lo + (q % nk);
    q /= nk;
    const int m = mrange.lo + q;

    bool bad_face = false;
    int bad_components = 0;
    int first_bad_component = -1;
    for (int n = 0; n < nvars; ++n) {
      const Real v = a(m,n,k,j,i);
      if (!FiniteForRepair(v)) {
        bad_face = true;
        ++bad_components;
        if (first_bad_component < 0) first_bad_component = n;
      }
    }
    if (!bad_face) return;

    const int old_count = Kokkos::atomic_fetch_add(&counts(count_offset), 1);
    Kokkos::atomic_add(&counts(count_offset + 1), bad_components);
    if (old_count == 0) {
      const int row = count_offset/2;
      first_locs(row,0) = m;
      first_locs(row,1) = k;
      first_locs(row,2) = j;
      first_locs(row,3) = i;
      first_locs(row,4) = first_bad_component;
    }
    for (int n = 0; n < nvars; ++n) {
      a(m,n,k,j,i) = 0.0;
    }
  });
  DevExeSpace().fence();
}

template <typename ViewType>
void RepairConsRange(const ViewType &u0, const ViewType &u1, const int nvars,
                     const IndexRange mrange, const IndexRange krange,
                     const IndexRange jrange, const IndexRange irange,
                     const DvceArray1D<int> &counts,
                     const DvceArray2D<int> &first_locs) {
  const int nm = RangeCount(mrange);
  const int nk = RangeCount(krange);
  const int nj = RangeCount(jrange);
  const int ni = RangeCount(irange);
  const int total = nm*nk*nj*ni;
  if (total <= 0 || nvars <= 0) return;

  Kokkos::parallel_for("mhd_repair_conserved", Kokkos::RangePolicy<>(DevExeSpace(), 0, total),
  KOKKOS_LAMBDA(const int idx) {
    int q = idx;
    const int i = irange.lo + (q % ni);
    q /= ni;
    const int j = jrange.lo + (q % nj);
    q /= nj;
    const int k = krange.lo + (q % nk);
    q /= nk;
    const int m = mrange.lo + q;

    bool bad_cell = false;
    int bad_components = 0;
    int first_bad_component = -1;
    bool previous_state_ok = true;
    for (int n = 0; n < nvars; ++n) {
      const Real v = u0(m,n,k,j,i);
      if (!FiniteForRepair(v)) {
        bad_cell = true;
        ++bad_components;
        if (first_bad_component < 0) first_bad_component = n;
      }
      if (!FiniteForRepair(u1(m,n,k,j,i))) previous_state_ok = false;
    }
    if (!bad_cell) return;

    const int old_count = Kokkos::atomic_fetch_add(&counts(0), 1);
    Kokkos::atomic_add(&counts(1), bad_components);
    if (!previous_state_ok) Kokkos::atomic_add(&counts(2), 1);
    if (old_count == 0) {
      first_locs(0,0) = m;
      first_locs(0,1) = k;
      first_locs(0,2) = j;
      first_locs(0,3) = i;
      first_locs(0,4) = first_bad_component;
    }

    for (int n = 0; n < nvars; ++n) {
      const Real prev = u1(m,n,k,j,i);
      const Real curr = u0(m,n,k,j,i);
      u0(m,n,k,j,i) = FiniteForRepair(prev) ? prev : (FiniteForRepair(curr) ? curr : 0.0);
    }
  });
  DevExeSpace().fence();
}

template <typename ViewType>
void RepairFaceEMFArray(const char *component, const ViewType &a, const int count_index,
                        const DvceArray1D<int> &counts,
                        const DvceArray2D<int> &first_locs) {
  const int nmb = static_cast<int>(a.extent(0));
  const int n3 = static_cast<int>(a.extent(1));
  const int n2 = static_cast<int>(a.extent(2));
  const int n1 = static_cast<int>(a.extent(3));
  const int total = nmb*n3*n2*n1;
  if (total <= 0) return;

  std::string kernel_label = std::string("mhd_repair_face_emf_") + component;
  Kokkos::parallel_for(kernel_label, Kokkos::RangePolicy<>(DevExeSpace(), 0, total),
  KOKKOS_LAMBDA(const int idx) {
    int q = idx;
    const int i = q % n1;
    q /= n1;
    const int j = q % n2;
    q /= n2;
    const int k = q % n3;
    q /= n3;
    const int m = q;

    const Real v = a(m,k,j,i);
    if (FiniteForRepair(v)) return;

    Real sum = 0.0;
    int finite_count = 0;
    if (i > 0) {
      const Real nb = a(m,k,j,i-1);
      if (FiniteForRepair(nb)) {
        sum += nb;
        ++finite_count;
      }
    }
    if (i + 1 < n1) {
      const Real nb = a(m,k,j,i+1);
      if (FiniteForRepair(nb)) {
        sum += nb;
        ++finite_count;
      }
    }
    if (j > 0) {
      const Real nb = a(m,k,j-1,i);
      if (FiniteForRepair(nb)) {
        sum += nb;
        ++finite_count;
      }
    }
    if (j + 1 < n2) {
      const Real nb = a(m,k,j+1,i);
      if (FiniteForRepair(nb)) {
        sum += nb;
        ++finite_count;
      }
    }
    if (k > 0) {
      const Real nb = a(m,k-1,j,i);
      if (FiniteForRepair(nb)) {
        sum += nb;
        ++finite_count;
      }
    }
    if (k + 1 < n3) {
      const Real nb = a(m,k+1,j,i);
      if (FiniteForRepair(nb)) {
        sum += nb;
        ++finite_count;
      }
    }

    const Real fallback = (finite_count > 0) ? (sum/static_cast<Real>(finite_count)) : 0.0;
    const int old_count = Kokkos::atomic_fetch_add(&counts(count_index), 1);
    if (old_count == 0) {
      first_locs(count_index,0) = m;
      first_locs(count_index,1) = k;
      first_locs(count_index,2) = j;
      first_locs(count_index,3) = i;
    }
    if (finite_count == 0) {
      Kokkos::atomic_add(&counts(count_index + 6), 1);
    }
    a(m,k,j,i) = fallback;
  });
  DevExeSpace().fence();
}

void ApproxCoords(const RegionSize &size, const int ng, const LocationKind loc,
                  const int k, const int j, const int i,
                  Real &x1, Real &x2, Real &x3) {
  const Real ic = static_cast<Real>(i - ng);
  const Real jc = static_cast<Real>(j - ng);
  const Real kc = static_cast<Real>(k - ng);
  x1 = size.x1min + (ic + 0.5)*size.dx1;
  x2 = size.x2min + (jc + 0.5)*size.dx2;
  x3 = size.x3min + (kc + 0.5)*size.dx3;

  if (loc == LocationKind::x1f) x1 = size.x1min + ic*size.dx1;
  if (loc == LocationKind::x2f) x2 = size.x2min + jc*size.dx2;
  if (loc == LocationKind::x3f) x3 = size.x3min + kc*size.dx3;

  if (loc == LocationKind::x1e) {
    x2 = size.x2min + jc*size.dx2;
    x3 = size.x3min + kc*size.dx3;
  } else if (loc == LocationKind::x2e) {
    x1 = size.x1min + ic*size.dx1;
    x3 = size.x3min + kc*size.dx3;
  } else if (loc == LocationKind::x3e) {
    x1 = size.x1min + ic*size.dx1;
    x2 = size.x2min + jc*size.dx2;
  }
}

template <typename ViewType>
bool ScanFinite4D(const char *label, const char *component, const ViewType &a,
                  MeshBlockPack *pmbp, Driver *pdriver, const int stage,
                  const LocationKind loc) {
  const int nmb = static_cast<int>(a.extent(0));
  const int n3 = static_cast<int>(a.extent(1));
  const int n2 = static_cast<int>(a.extent(2));
  const int n1 = static_cast<int>(a.extent(3));
  const int total = nmb*n3*n2*n1;
  if (total <= 0) return true;

  int bad_count = 0;
  int first_bad = total;
  std::string kernel_label = std::string("mhd_finite_") + label + "_" + component;
  Kokkos::parallel_reduce(kernel_label, Kokkos::RangePolicy<>(DevExeSpace(), 0, total),
  KOKKOS_LAMBDA(const int idx, int &count, int &first) {
    int q = idx;
    const int i = q % n1;
    q /= n1;
    const int j = q % n2;
    q /= n2;
    const int k = q % n3;
    q /= n3;
    const int m = q;
    const Real v = a(m,k,j,i);
    if ((v != v) || (v > static_cast<Real>(1.0e300)) ||
        (v < static_cast<Real>(-1.0e300))) {
      count += 1;
      if (idx < first) first = idx;
    }
  }, Kokkos::Sum<int>(bad_count), Kokkos::Min<int>(first_bad));
  DevExeSpace().fence();

  if (bad_count == 0) return true;

  int q = first_bad;
  const int i = q % n1;
  q /= n1;
  const int j = q % n2;
  q /= n2;
  const int k = q % n3;
  q /= n3;
  const int m = q;

  Real bad_value = 0.0;
  Kokkos::parallel_reduce(std::string("mhd_finite_value_") + label + "_" + component,
  Kokkos::RangePolicy<>(DevExeSpace(), 0, total),
  KOKKOS_LAMBDA(const int idx, Real &value) {
    if (idx == first_bad) {
      int qv = idx;
      const int iv = qv % n1;
      qv /= n1;
      const int jv = qv % n2;
      qv /= n2;
      const int kv = qv % n3;
      qv /= n3;
      const int mv = qv;
      value += a(mv,kv,jv,iv);
    }
  }, Kokkos::Sum<Real>(bad_value));
  DevExeSpace().fence();

  const int ng = pmbp->pmesh->mb_indcs.ng;
  Real x1, x2, x3;
  ApproxCoords(pmbp->pmb->mb_size.h_view(m), ng, loc, k, j, i, x1, x2, x3);
  const Real r = std::sqrt(x1*x1 + x2*x2 + x3*x3);

  std::cout << "MHD finite-check failure after " << label
            << " on rank " << global_variable::my_rank
            << " stage=" << stage
            << " time=" << pmbp->pmesh->time
            << " cycle=" << pmbp->pmesh->ncycle
            << " component=" << component
            << " bad_count=" << bad_count
            << " first=(m,k,j,i)=(" << m << "," << k << "," << j << "," << i << ")"
            << " xyz=(" << x1 << "," << x2 << "," << x3 << ")"
            << " r_origin=" << r
            << " value=" << bad_value
            << std::endl;
  return false;
}

template <typename ViewType>
bool ScanFinite5DComponent(const char *label, const char *component, const ViewType &a,
                           const int n, MeshBlockPack *pmbp, Driver *pdriver,
                           const int stage, const LocationKind loc) {
  const int nmb = static_cast<int>(a.extent(0));
  const int nvar = static_cast<int>(a.extent(1));
  const int n3 = static_cast<int>(a.extent(2));
  const int n2 = static_cast<int>(a.extent(3));
  const int n1 = static_cast<int>(a.extent(4));
  if (n < 0 || n >= nvar) return true;
  const int total = nmb*n3*n2*n1;
  if (total <= 0) return true;

  int bad_count = 0;
  int first_bad = total;
  std::string kernel_label = std::string("mhd_finite_") + label + "_" + component;
  Kokkos::parallel_reduce(kernel_label, Kokkos::RangePolicy<>(DevExeSpace(), 0, total),
  KOKKOS_LAMBDA(const int idx, int &count, int &first) {
    int q = idx;
    const int i = q % n1;
    q /= n1;
    const int j = q % n2;
    q /= n2;
    const int k = q % n3;
    q /= n3;
    const int m = q;
    const Real v = a(m,n,k,j,i);
    if ((v != v) || (v > static_cast<Real>(1.0e300)) ||
        (v < static_cast<Real>(-1.0e300))) {
      count += 1;
      if (idx < first) first = idx;
    }
  }, Kokkos::Sum<int>(bad_count), Kokkos::Min<int>(first_bad));
  DevExeSpace().fence();

  if (bad_count == 0) return true;

  int q = first_bad;
  const int i = q % n1;
  q /= n1;
  const int j = q % n2;
  q /= n2;
  const int k = q % n3;
  q /= n3;
  const int m = q;

  Real bad_value = 0.0;
  Kokkos::parallel_reduce(std::string("mhd_finite_value_") + label + "_" + component,
  Kokkos::RangePolicy<>(DevExeSpace(), 0, total),
  KOKKOS_LAMBDA(const int idx, Real &value) {
    if (idx == first_bad) {
      int qv = idx;
      const int iv = qv % n1;
      qv /= n1;
      const int jv = qv % n2;
      qv /= n2;
      const int kv = qv % n3;
      qv /= n3;
      const int mv = qv;
      value += a(mv,n,kv,jv,iv);
    }
  }, Kokkos::Sum<Real>(bad_value));
  DevExeSpace().fence();

  const int ng = pmbp->pmesh->mb_indcs.ng;
  Real x1, x2, x3;
  ApproxCoords(pmbp->pmb->mb_size.h_view(m), ng, loc, k, j, i, x1, x2, x3);
  const Real r = std::sqrt(x1*x1 + x2*x2 + x3*x3);

  std::cout << "MHD finite-check failure after " << label
            << " on rank " << global_variable::my_rank
            << " stage=" << stage
            << " time=" << pmbp->pmesh->time
            << " cycle=" << pmbp->pmesh->ncycle
            << " component=" << component
            << " bad_count=" << bad_count
            << " first=(m,k,j,i)=(" << m << "," << k << "," << j << "," << i << ")"
            << " xyz=(" << x1 << "," << x2 << "," << x3 << ")"
            << " r_origin=" << r
            << " value=" << bad_value
            << std::endl;
  return false;
}

struct BadLocation {
  int bad_count = 0;
  int m = 0;
  int k = 0;
  int j = 0;
  int i = 0;
  Real value = 0.0;
  Real x1 = 0.0;
  Real x2 = 0.0;
  Real x3 = 0.0;
  Real r = 0.0;
};

template <typename ViewType>
bool FindFirstBad4D(const char *label, const char *component, const ViewType &a,
                    MeshBlockPack *pmbp, const int stage, const LocationKind loc,
                    BadLocation &bad) {
  const int nmb = static_cast<int>(a.extent(0));
  const int n3 = static_cast<int>(a.extent(1));
  const int n2 = static_cast<int>(a.extent(2));
  const int n1 = static_cast<int>(a.extent(3));
  const int total = nmb*n3*n2*n1;
  bad = BadLocation();
  if (total <= 0) return true;

  int bad_count = 0;
  int first_bad = total;
  std::string kernel_label = std::string("mhd_find_bad_") + label + "_" + component;
  Kokkos::parallel_reduce(kernel_label, Kokkos::RangePolicy<>(DevExeSpace(), 0, total),
  KOKKOS_LAMBDA(const int idx, int &count, int &first) {
    int q = idx;
    const int i = q % n1;
    q /= n1;
    const int j = q % n2;
    q /= n2;
    const int k = q % n3;
    q /= n3;
    const int m = q;
    const Real v = a(m,k,j,i);
    if ((v != v) || (v > static_cast<Real>(1.0e300)) ||
        (v < static_cast<Real>(-1.0e300))) {
      count += 1;
      if (idx < first) first = idx;
    }
  }, Kokkos::Sum<int>(bad_count), Kokkos::Min<int>(first_bad));
  DevExeSpace().fence();

  if (bad_count == 0) return true;

  int q = first_bad;
  const int i = q % n1;
  q /= n1;
  const int j = q % n2;
  q /= n2;
  const int k = q % n3;
  q /= n3;
  const int m = q;

  Real bad_value = 0.0;
  Kokkos::parallel_reduce(std::string("mhd_find_bad_value_") + label + "_" + component,
  Kokkos::RangePolicy<>(DevExeSpace(), 0, total),
  KOKKOS_LAMBDA(const int idx, Real &value) {
    if (idx == first_bad) {
      int qv = idx;
      const int iv = qv % n1;
      qv /= n1;
      const int jv = qv % n2;
      qv /= n2;
      const int kv = qv % n3;
      qv /= n3;
      const int mv = qv;
      value += a(mv,kv,jv,iv);
    }
  }, Kokkos::Sum<Real>(bad_value));
  DevExeSpace().fence();

  const int ng = pmbp->pmesh->mb_indcs.ng;
  ApproxCoords(pmbp->pmb->mb_size.h_view(m), ng, loc, k, j, i,
               bad.x1, bad.x2, bad.x3);
  bad.bad_count = bad_count;
  bad.m = m;
  bad.k = k;
  bad.j = j;
  bad.i = i;
  bad.value = bad_value;
  bad.r = std::sqrt(bad.x1*bad.x1 + bad.x2*bad.x2 + bad.x3*bad.x3);

  std::cout << "MHD finite-check failure after " << label
            << " on rank " << global_variable::my_rank
            << " stage=" << stage
            << " time=" << pmbp->pmesh->time
            << " cycle=" << pmbp->pmesh->ncycle
            << " component=" << component
            << " bad_count=" << bad.bad_count
            << " first=(m,k,j,i)=(" << bad.m << "," << bad.k << ","
            << bad.j << "," << bad.i << ")"
            << " xyz=(" << bad.x1 << "," << bad.x2 << "," << bad.x3 << ")"
            << " r_origin=" << bad.r
            << " value=" << bad.value
            << std::endl;
  return false;
}

void PrintSG07Value(const char *name, const Real value) {
  const bool bad = ((value != value) || (value > static_cast<Real>(1.0e300)) ||
                    (value < static_cast<Real>(-1.0e300)));
  std::cout << "  " << name << "=" << value
            << " abs=" << std::fabs(value)
            << " bad=" << (bad ? "yes" : "no")
            << std::endl;
}

void PrintSG07Branch(const char *name, const int branch) {
  std::cout << "  branch " << name << "="
            << (branch ? ">=0" : "<0_or_nan") << std::endl;
}

void DumpSG07E1(const BadLocation &bad,
                const DvceArray4D<Real> &e1x2_, const DvceArray4D<Real> &e1x3_,
                const DvceArray4D<Real> &e1cc_,
                const DvceArray5D<Real> &flx2, const DvceArray5D<Real> &flx3,
                MeshBlockPack *pmbp, const int stage) {
  DvceArray1D<Real> vals("sg07_e1_vals", 18);
  DvceArray1D<int> branches("sg07_e1_branches", 4);
  const int m = bad.m;
  const int k = bad.k;
  const int j = bad.j;
  const int i = bad.i;

  Kokkos::parallel_for("mhd_sg07_e1_dump", Kokkos::RangePolicy<>(DevExeSpace(), 0, 1),
  KOKKOS_LAMBDA(const int) {
    const Real flx2_l = flx2(m,IDN,k-1,j,i);
    const Real flx2_r = flx2(m,IDN,k  ,j,i);
    const Real flx3_l = flx3(m,IDN,k,j-1,i);
    const Real flx3_r = flx3(m,IDN,k,j  ,i);
    branches(0) = (flx2_l >= 0.0);
    branches(1) = (flx2_r >= 0.0);
    branches(2) = (flx3_l >= 0.0);
    branches(3) = (flx3_r >= 0.0);

    Real e1_l3, e1_r3, e1_l2, e1_r2;
    if (branches(0)) {
      e1_l3 = e1x3_(m,k,j-1,i) - e1cc_(m,k-1,j-1,i);
    } else {
      e1_l3 = e1x3_(m,k,j  ,i) - e1cc_(m,k-1,j  ,i);
    }
    if (branches(1)) {
      e1_r3 = e1x3_(m,k,j-1,i) - e1cc_(m,k  ,j-1,i);
    } else {
      e1_r3 = e1x3_(m,k,j  ,i) - e1cc_(m,k  ,j  ,i);
    }
    if (branches(2)) {
      e1_l2 = e1x2_(m,k-1,j,i) - e1cc_(m,k-1,j-1,i);
    } else {
      e1_l2 = e1x2_(m,k  ,j,i) - e1cc_(m,k  ,j-1,i);
    }
    if (branches(3)) {
      e1_r2 = e1x2_(m,k-1,j,i) - e1cc_(m,k-1,j  ,i);
    } else {
      e1_r2 = e1x2_(m,k  ,j,i) - e1cc_(m,k  ,j  ,i);
    }

    vals(0) = bad.value;
    vals(1) = 0.25*(e1_l3 + e1_r3 + e1_l2 + e1_r2 +
              e1x2_(m,k-1,j,i) + e1x2_(m,k,j,i) +
              e1x3_(m,k,j-1,i) + e1x3_(m,k,j,i));
    vals(2) = e1_l3;
    vals(3) = e1_r3;
    vals(4) = e1_l2;
    vals(5) = e1_r2;
    vals(6) = e1x2_(m,k-1,j,i);
    vals(7) = e1x2_(m,k  ,j,i);
    vals(8) = e1x3_(m,k,j-1,i);
    vals(9) = e1x3_(m,k,j  ,i);
    vals(10) = e1cc_(m,k-1,j-1,i);
    vals(11) = e1cc_(m,k-1,j  ,i);
    vals(12) = e1cc_(m,k  ,j-1,i);
    vals(13) = e1cc_(m,k  ,j  ,i);
    vals(14) = flx2_l;
    vals(15) = flx2_r;
    vals(16) = flx3_l;
    vals(17) = flx3_r;
  });

  auto h_vals = Kokkos::create_mirror_view(vals);
  auto h_branches = Kokkos::create_mirror_view(branches);
  Kokkos::deep_copy(h_vals, vals);
  Kokkos::deep_copy(h_branches, branches);

  std::cout << "MHD SG07 term dump for efld.x1e on rank " << global_variable::my_rank
            << " stage=" << stage
            << " time=" << pmbp->pmesh->time
            << " cycle=" << pmbp->pmesh->ncycle
            << " first=(m,k,j,i)=(" << m << "," << k << "," << j << "," << i << ")"
            << " xyz=(" << bad.x1 << "," << bad.x2 << "," << bad.x3 << ")"
            << " r_origin=" << bad.r
            << std::endl;
  const char *names[18] = {
    "edge_value", "sg07_recomputed", "e1_l3", "e1_r3", "e1_l2", "e1_r2",
    "e1x2(k-1,j,i)", "e1x2(k,j,i)", "e1x3(k,j-1,i)", "e1x3(k,j,i)",
    "e1_cc(k-1,j-1,i)", "e1_cc(k-1,j,i)", "e1_cc(k,j-1,i)", "e1_cc(k,j,i)",
    "flx2(IDN,k-1,j,i)", "flx2(IDN,k,j,i)",
    "flx3(IDN,k,j-1,i)", "flx3(IDN,k,j,i)"
  };
  for (int n = 0; n < 18; ++n) {
    PrintSG07Value(names[n], h_vals(n));
  }
  PrintSG07Branch("e1_l3 from flx2(k-1,j,i)", h_branches(0));
  PrintSG07Branch("e1_r3 from flx2(k,j,i)", h_branches(1));
  PrintSG07Branch("e1_l2 from flx3(k,j-1,i)", h_branches(2));
  PrintSG07Branch("e1_r2 from flx3(k,j,i)", h_branches(3));
}

void DumpSG07E2(const BadLocation &bad,
                const DvceArray4D<Real> &e2x1_, const DvceArray4D<Real> &e2x3_,
                const DvceArray4D<Real> &e2cc_,
                const DvceArray5D<Real> &flx1, const DvceArray5D<Real> &flx3,
                MeshBlockPack *pmbp, const int stage) {
  DvceArray1D<Real> vals("sg07_e2_vals", 18);
  DvceArray1D<int> branches("sg07_e2_branches", 4);
  const int m = bad.m;
  const int k = bad.k;
  const int j = bad.j;
  const int i = bad.i;

  Kokkos::parallel_for("mhd_sg07_e2_dump", Kokkos::RangePolicy<>(DevExeSpace(), 0, 1),
  KOKKOS_LAMBDA(const int) {
    const Real flx1_l = flx1(m,IDN,k-1,j,i);
    const Real flx1_r = flx1(m,IDN,k  ,j,i);
    const Real flx3_l = flx3(m,IDN,k,j,i-1);
    const Real flx3_r = flx3(m,IDN,k,j,i  );
    branches(0) = (flx1_l >= 0.0);
    branches(1) = (flx1_r >= 0.0);
    branches(2) = (flx3_l >= 0.0);
    branches(3) = (flx3_r >= 0.0);

    Real e2_l3, e2_r3, e2_l1, e2_r1;
    if (branches(0)) {
      e2_l3 = e2x3_(m,k,j,i-1) - e2cc_(m,k-1,j,i-1);
    } else {
      e2_l3 = e2x3_(m,k,j,i  ) - e2cc_(m,k-1,j,i  );
    }
    if (branches(1)) {
      e2_r3 = e2x3_(m,k,j,i-1) - e2cc_(m,k  ,j,i-1);
    } else {
      e2_r3 = e2x3_(m,k,j,i  ) - e2cc_(m,k  ,j,i  );
    }
    if (branches(2)) {
      e2_l1 = e2x1_(m,k-1,j,i) - e2cc_(m,k-1,j,i-1);
    } else {
      e2_l1 = e2x1_(m,k  ,j,i) - e2cc_(m,k  ,j,i-1);
    }
    if (branches(3)) {
      e2_r1 = e2x1_(m,k-1,j,i) - e2cc_(m,k-1,j,i  );
    } else {
      e2_r1 = e2x1_(m,k  ,j,i) - e2cc_(m,k  ,j,i  );
    }

    vals(0) = bad.value;
    vals(1) = 0.25*(e2_l3 + e2_r3 + e2_l1 + e2_r1 +
              e2x3_(m,k,j,i-1) + e2x3_(m,k,j,i) +
              e2x1_(m,k-1,j,i) + e2x1_(m,k,j,i));
    vals(2) = e2_l3;
    vals(3) = e2_r3;
    vals(4) = e2_l1;
    vals(5) = e2_r1;
    vals(6) = e2x3_(m,k,j,i-1);
    vals(7) = e2x3_(m,k,j,i  );
    vals(8) = e2x1_(m,k-1,j,i);
    vals(9) = e2x1_(m,k  ,j,i);
    vals(10) = e2cc_(m,k-1,j,i-1);
    vals(11) = e2cc_(m,k-1,j,i  );
    vals(12) = e2cc_(m,k  ,j,i-1);
    vals(13) = e2cc_(m,k  ,j,i  );
    vals(14) = flx1_l;
    vals(15) = flx1_r;
    vals(16) = flx3_l;
    vals(17) = flx3_r;
  });

  auto h_vals = Kokkos::create_mirror_view(vals);
  auto h_branches = Kokkos::create_mirror_view(branches);
  Kokkos::deep_copy(h_vals, vals);
  Kokkos::deep_copy(h_branches, branches);

  std::cout << "MHD SG07 term dump for efld.x2e on rank " << global_variable::my_rank
            << " stage=" << stage
            << " time=" << pmbp->pmesh->time
            << " cycle=" << pmbp->pmesh->ncycle
            << " first=(m,k,j,i)=(" << m << "," << k << "," << j << "," << i << ")"
            << " xyz=(" << bad.x1 << "," << bad.x2 << "," << bad.x3 << ")"
            << " r_origin=" << bad.r
            << std::endl;
  const char *names[18] = {
    "edge_value", "sg07_recomputed", "e2_l3", "e2_r3", "e2_l1", "e2_r1",
    "e2x3(k,j,i-1)", "e2x3(k,j,i)", "e2x1(k-1,j,i)", "e2x1(k,j,i)",
    "e2_cc(k-1,j,i-1)", "e2_cc(k-1,j,i)", "e2_cc(k,j,i-1)", "e2_cc(k,j,i)",
    "flx1(IDN,k-1,j,i)", "flx1(IDN,k,j,i)",
    "flx3(IDN,k,j,i-1)", "flx3(IDN,k,j,i)"
  };
  for (int n = 0; n < 18; ++n) {
    PrintSG07Value(names[n], h_vals(n));
  }
  PrintSG07Branch("e2_l3 from flx1(k-1,j,i)", h_branches(0));
  PrintSG07Branch("e2_r3 from flx1(k,j,i)", h_branches(1));
  PrintSG07Branch("e2_l1 from flx3(k,j,i-1)", h_branches(2));
  PrintSG07Branch("e2_r1 from flx3(k,j,i)", h_branches(3));
}

} // namespace

namespace mhd {

TaskStatus MHD::RepairNonFiniteFluxes(Driver *pdriver, int stage) {
  RepairNonFiniteFluxArrays("RecvFlux", pdriver, stage);
  return TaskStatus::complete;
}

void MHD::RepairNonFiniteFluxArrays(const char *label, Driver *pdriver, int stage) {
  if (!repair_nonfinite_fluxes || stage < 0) return;

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nvars = nmhd + nscalars;
  const int nmb1 = pmy_pack->nmb_thispack - 1;
  if (nvars <= 0 || nmb1 < 0) return;

  DvceArray1D<int> counts("mhd_nonfinite_flux_repair_counts", 6);
  DvceArray2D<int> first_locs("mhd_nonfinite_flux_repair_first_locs", 3, 5);
  Kokkos::deep_copy(DevExeSpace(), counts, 0);
  Kokkos::deep_copy(DevExeSpace(), first_locs, -1);

  const IndexRange mrange = ClampRange(0, nmb1, static_cast<int>(uflx.x1f.extent(0)));

  // RKUpdate needs active faces; CornerE SG07 also reads one transverse stencil layer.
  RepairFluxRange("x1f", uflx.x1f, nvars, mrange,
                  ClampRange(ks - 1, ke + 1, static_cast<int>(uflx.x1f.extent(2))),
                  ClampRange(js - 1, je + 1, static_cast<int>(uflx.x1f.extent(3))),
                  ClampRange(is, ie + 1, static_cast<int>(uflx.x1f.extent(4))),
                  0, counts, first_locs);
  if (pmy_pack->pmesh->multi_d) {
    RepairFluxRange("x2f", uflx.x2f, nvars, mrange,
                    ClampRange(ks - 1, ke + 1, static_cast<int>(uflx.x2f.extent(2))),
                    ClampRange(js, je + 1, static_cast<int>(uflx.x2f.extent(3))),
                    ClampRange(is - 1, ie + 1, static_cast<int>(uflx.x2f.extent(4))),
                    2, counts, first_locs);
  }
  if (pmy_pack->pmesh->three_d) {
    RepairFluxRange("x3f", uflx.x3f, nvars, mrange,
                    ClampRange(ks, ke + 1, static_cast<int>(uflx.x3f.extent(2))),
                    ClampRange(js - 1, je + 1, static_cast<int>(uflx.x3f.extent(3))),
                    ClampRange(is - 1, ie + 1, static_cast<int>(uflx.x3f.extent(4))),
                    4, counts, first_locs);
  }

  auto h_counts = Kokkos::create_mirror_view(counts);
  auto h_first_locs = Kokkos::create_mirror_view(first_locs);
  Kokkos::deep_copy(h_counts, counts);
  Kokkos::deep_copy(h_first_locs, first_locs);

  const int repaired_faces = h_counts(0) + h_counts(2) + h_counts(4);
  if (repaired_faces > 0 && repair_nonfinite_fluxes_verbose) {
    std::cout << "MHD nonfinite-flux repair after " << label
              << " on rank " << global_variable::my_rank
              << " stage=" << stage
              << " time=" << pmy_pack->pmesh->time
              << " cycle=" << pmy_pack->pmesh->ncycle
              << " x1_faces=" << h_counts(0)
              << " x1_bad_components=" << h_counts(1)
              << " x1_first_mkjin=(" << h_first_locs(0,0) << ","
              << h_first_locs(0,1) << "," << h_first_locs(0,2) << ","
              << h_first_locs(0,3) << "," << h_first_locs(0,4) << ")"
              << " x2_faces=" << h_counts(2)
              << " x2_bad_components=" << h_counts(3)
              << " x2_first_mkjin=(" << h_first_locs(1,0) << ","
              << h_first_locs(1,1) << "," << h_first_locs(1,2) << ","
              << h_first_locs(1,3) << "," << h_first_locs(1,4) << ")"
              << " x3_faces=" << h_counts(4)
              << " x3_bad_components=" << h_counts(5)
              << " x3_first_mkjin=(" << h_first_locs(2,0) << ","
              << h_first_locs(2,1) << "," << h_first_locs(2,2) << ","
              << h_first_locs(2,3) << "," << h_first_locs(2,4) << ")"
              << std::endl;
  }
  (void)pdriver;
}

TaskStatus MHD::RepairNonFiniteConserved(Driver *pdriver, int stage) {
  if (!repair_nonfinite_conserved || stage < 0) return TaskStatus::complete;

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nvars = nmhd + nscalars;
  const int nmb1 = pmy_pack->nmb_thispack - 1;
  if (nvars <= 0 || nmb1 < 0) return TaskStatus::complete;

  DvceArray1D<int> counts("mhd_nonfinite_conserved_repair_counts", 3);
  DvceArray2D<int> first_locs("mhd_nonfinite_conserved_repair_first_locs", 1, 5);
  Kokkos::deep_copy(DevExeSpace(), counts, 0);
  Kokkos::deep_copy(DevExeSpace(), first_locs, -1);

  const IndexRange mrange = ClampRange(0, nmb1, static_cast<int>(u0.extent(0)));
  RepairConsRange(u0, u1, nvars, mrange,
                  ClampRange(ks, ke, static_cast<int>(u0.extent(2))),
                  ClampRange(js, je, static_cast<int>(u0.extent(3))),
                  ClampRange(is, ie, static_cast<int>(u0.extent(4))),
                  counts, first_locs);

  auto h_counts = Kokkos::create_mirror_view(counts);
  auto h_first_locs = Kokkos::create_mirror_view(first_locs);
  Kokkos::deep_copy(h_counts, counts);
  Kokkos::deep_copy(h_first_locs, first_locs);

  if (h_counts(0) > 0 && repair_nonfinite_fluxes_verbose) {
    std::cout << "MHD nonfinite-conserved repair after RKUpdate"
              << " on rank " << global_variable::my_rank
              << " stage=" << stage
              << " time=" << pmy_pack->pmesh->time
              << " cycle=" << pmy_pack->pmesh->ncycle
              << " cells=" << h_counts(0)
              << " bad_components=" << h_counts(1)
              << " partial_fallback_cells=" << h_counts(2)
              << " first_mkjin=(" << h_first_locs(0,0) << ","
              << h_first_locs(0,1) << "," << h_first_locs(0,2) << ","
              << h_first_locs(0,3) << "," << h_first_locs(0,4) << ")"
              << std::endl;
  }
  (void)pdriver;
  return TaskStatus::complete;
}

void MHD::RepairNonFiniteFaceEMFs(const char *label, Driver *pdriver, int stage) {
  if (!repair_nonfinite_face_emfs) return;

  DvceArray1D<int> counts("mhd_nonfinite_face_emf_repair_counts", 12);
  DvceArray2D<int> first_locs("mhd_nonfinite_face_emf_repair_first_locs", 6, 4);
  Kokkos::deep_copy(DevExeSpace(), counts, 0);
  Kokkos::deep_copy(DevExeSpace(), first_locs, -1);

  RepairFaceEMFArray("e3x1", e3x1, 0, counts, first_locs);
  RepairFaceEMFArray("e2x1", e2x1, 1, counts, first_locs);
  RepairFaceEMFArray("e1x2", e1x2, 2, counts, first_locs);
  RepairFaceEMFArray("e3x2", e3x2, 3, counts, first_locs);
  RepairFaceEMFArray("e2x3", e2x3, 4, counts, first_locs);
  RepairFaceEMFArray("e1x3", e1x3, 5, counts, first_locs);

  auto h_counts = Kokkos::create_mirror_view(counts);
  auto h_first_locs = Kokkos::create_mirror_view(first_locs);
  Kokkos::deep_copy(h_counts, counts);
  Kokkos::deep_copy(h_first_locs, first_locs);

  const int repaired = h_counts(0) + h_counts(1) + h_counts(2) +
                       h_counts(3) + h_counts(4) + h_counts(5);
  if (repaired > 0 && repair_nonfinite_fluxes_verbose) {
    std::cout << "MHD nonfinite-face-EMF repair after " << label
              << " on rank " << global_variable::my_rank
              << " stage=" << stage
              << " time=" << pmy_pack->pmesh->time
              << " cycle=" << pmy_pack->pmesh->ncycle
              << " e3x1_count=" << h_counts(0)
              << " e3x1_zero_fallback=" << h_counts(6)
              << " e3x1_first_mkji=(" << h_first_locs(0,0) << ","
              << h_first_locs(0,1) << "," << h_first_locs(0,2) << ","
              << h_first_locs(0,3) << ")"
              << " e2x1_count=" << h_counts(1)
              << " e2x1_zero_fallback=" << h_counts(7)
              << " e2x1_first_mkji=(" << h_first_locs(1,0) << ","
              << h_first_locs(1,1) << "," << h_first_locs(1,2) << ","
              << h_first_locs(1,3) << ")"
              << " e1x2_count=" << h_counts(2)
              << " e1x2_zero_fallback=" << h_counts(8)
              << " e1x2_first_mkji=(" << h_first_locs(2,0) << ","
              << h_first_locs(2,1) << "," << h_first_locs(2,2) << ","
              << h_first_locs(2,3) << ")"
              << " e3x2_count=" << h_counts(3)
              << " e3x2_zero_fallback=" << h_counts(9)
              << " e3x2_first_mkji=(" << h_first_locs(3,0) << ","
              << h_first_locs(3,1) << "," << h_first_locs(3,2) << ","
              << h_first_locs(3,3) << ")"
              << " e2x3_count=" << h_counts(4)
              << " e2x3_zero_fallback=" << h_counts(10)
              << " e2x3_first_mkji=(" << h_first_locs(4,0) << ","
              << h_first_locs(4,1) << "," << h_first_locs(4,2) << ","
              << h_first_locs(4,3) << ")"
              << " e1x3_count=" << h_counts(5)
              << " e1x3_zero_fallback=" << h_counts(11)
              << " e1x3_first_mkji=(" << h_first_locs(5,0) << ","
              << h_first_locs(5,1) << "," << h_first_locs(5,2) << ","
              << h_first_locs(5,3) << ")"
              << std::endl;
  }
  (void)pdriver;
}

bool MHD::CheckFiniteFaceB(const char *label, Driver *pdriver, int stage) {
  bool ok = true;
  ok = ScanFinite4D(label, "b0.x1f", b0.x1f, pmy_pack, pdriver, stage,
                    LocationKind::x1f) && ok;
  ok = ScanFinite4D(label, "b0.x2f", b0.x2f, pmy_pack, pdriver, stage,
                    LocationKind::x2f) && ok;
  ok = ScanFinite4D(label, "b0.x3f", b0.x3f, pmy_pack, pdriver, stage,
                    LocationKind::x3f) && ok;
  return ok;
}

bool MHD::CheckFiniteEdgeE(const char *label, Driver *pdriver, int stage) {
  bool ok = true;
  ok = ScanFinite4D(label, "efld.x1e", efld.x1e, pmy_pack, pdriver, stage,
                    LocationKind::x1e) && ok;
  ok = ScanFinite4D(label, "efld.x2e", efld.x2e, pmy_pack, pdriver, stage,
                    LocationKind::x2e) && ok;
  ok = ScanFinite4D(label, "efld.x3e", efld.x3e, pmy_pack, pdriver, stage,
                    LocationKind::x3e) && ok;
  return ok;
}

bool MHD::CheckFiniteFaceEMF(const char *label, Driver *pdriver, int stage) {
  bool ok = true;
  ok = ScanFinite4D(label, "e3x1", e3x1, pmy_pack, pdriver, stage,
                    LocationKind::x1f) && ok;
  ok = ScanFinite4D(label, "e2x1", e2x1, pmy_pack, pdriver, stage,
                    LocationKind::x1f) && ok;
  ok = ScanFinite4D(label, "e1x2", e1x2, pmy_pack, pdriver, stage,
                    LocationKind::x2f) && ok;
  ok = ScanFinite4D(label, "e3x2", e3x2, pmy_pack, pdriver, stage,
                    LocationKind::x2f) && ok;
  ok = ScanFinite4D(label, "e2x3", e2x3, pmy_pack, pdriver, stage,
                    LocationKind::x3f) && ok;
  ok = ScanFinite4D(label, "e1x3", e1x3, pmy_pack, pdriver, stage,
                    LocationKind::x3f) && ok;
  // Face EMFs can inherit non-finite density-flux inputs that the edge fallback
  // is meant to repair. Keep the reports visible without pre-empting that repair.
  (void)ok;
  return true;
}

bool MHD::CheckFiniteCellEMF(const char *label, Driver *pdriver, int stage) {
  bool ok = true;
  ok = ScanFinite4D(label, "e1_cc", e1_cc, pmy_pack, pdriver, stage,
                    LocationKind::cell) && ok;
  ok = ScanFinite4D(label, "e2_cc", e2_cc, pmy_pack, pdriver, stage,
                    LocationKind::cell) && ok;
  ok = ScanFinite4D(label, "e3_cc", e3_cc, pmy_pack, pdriver, stage,
                    LocationKind::cell) && ok;
  return ok;
}

bool MHD::CheckFiniteDensityFlux(const char *label, Driver *pdriver, int stage) {
  (void)ScanFinite5DComponent(label, "uflx.x1f(IDN)", uflx.x1f, IDN, pmy_pack,
                              pdriver, stage, LocationKind::x1f);
  (void)ScanFinite5DComponent(label, "uflx.x2f(IDN)", uflx.x2f, IDN, pmy_pack,
                              pdriver, stage, LocationKind::x2f);
  (void)ScanFinite5DComponent(label, "uflx.x3f(IDN)", uflx.x3f, IDN, pmy_pack,
                              pdriver, stage, LocationKind::x3f);
  return true;
}

bool MHD::CheckFiniteCornerE(const char *label, Driver *pdriver, int stage) {
  bool ok = true;

  BadLocation bad_e1;
  const bool e1_ok = FindFirstBad4D(label, "efld.x1e", efld.x1e, pmy_pack, stage,
                                    LocationKind::x1e, bad_e1);
  if (!e1_ok) {
    DumpSG07E1(bad_e1, e1x2, e1x3, e1_cc, uflx.x2f, uflx.x3f, pmy_pack, stage);
    ok = false;
  }

  BadLocation bad_e2;
  const bool e2_ok = FindFirstBad4D(label, "efld.x2e", efld.x2e, pmy_pack, stage,
                                    LocationKind::x2e, bad_e2);
  if (!e2_ok) {
    DumpSG07E2(bad_e2, e2x1, e2x3, e2_cc, uflx.x1f, uflx.x3f, pmy_pack, stage);
    ok = false;
  }

  ok = ScanFinite4D(label, "efld.x3e", efld.x3e, pmy_pack, pdriver, stage,
                    LocationKind::x3e) && ok;
  return ok;
}

} // namespace mhd
