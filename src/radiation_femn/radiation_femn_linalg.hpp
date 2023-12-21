#ifndef ATHENA_SRC_RADIATION_FEMN_RADIATION_FEMN_LINALG_HPP_
#define ATHENA_SRC_RADIATION_FEMN_RADIATION_FEMN_LINALG_HPP_

#include "athena.hpp"

namespace radiationfemn {

void LUDecomposition(DvceArray2D<Real> square_matrix, DvceArray2D<Real> lu_matrix, DvceArray1D<int> pivot_indices);
void LUSolve(DvceArray2D<Real> lu_matrix, DvceArray1D<int> pivot_indices, DvceArray1D<Real> b_array, DvceArray1D<Real> x_array);
void LUInverse(DvceArray2D<Real> A_matrix, DvceArray2D<Real> A_matrix_inverse);
void MatMultiplyHost(HostArray2D<Real> A_matrix, HostArray2D<Real> B_matrix, HostArray2D<Real> result);
void MatMatMultiply(DvceArray2D<Real> A_matrix, DvceArray2D<Real> B_matrix, DvceArray2D<Real> result);
void MatVecMultiply(DvceArray2D<Real> A_matrix, DvceArray1D<Real> B_array, DvceArray1D<Real> result);
void MatMultiplyComplex(std::vector<std::vector<std::complex<Real>>> &A_matrix, std::vector<std::vector<std::complex<Real>>> &B_matrix,
                        std::vector<std::vector<std::complex<Real>>> &result);
void MatLumping(DvceArray2D<Real> A_matrix);
void MatEig(std::vector<std::vector<Real>> &matrix, std::vector<std::complex<Real>> &eigval, std::vector<std::vector<std::complex<Real>>> &eigvec);
void ZeroSpeedCorrection(HostArray2D<Real> matrix, HostArray2D<Real> matrix_corrected, Real v);

}

#endif //ATHENA_SRC_RADIATION_FEMN_RADIATION_FEMN_LINALG_HPP_