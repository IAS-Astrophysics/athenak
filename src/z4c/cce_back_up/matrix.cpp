#include <iostream>
#include <complex>
#include <cmath>
#include "myassert.hpp"
#include "matrix.hpp"

#if MATRIX_DECOMP_USE_GSL_MATRIX_INVERTER
#  include <gsl/gsl_linalg.h>
#  include <gsl/gsl_permutation.h>
#  include <gsl/gsl_vector.h>
#  include <gsl/gsl_matrix.h>
#endif

namespace decomp_matrix_class {
using namespace std;

template <class T>
int row<T>::length()
{
  return size;
}

template <class T>
row<T>& row<T>::operator=(const row<T> &src)
{
  int i;
  if (this == &src)
  {
    return *this;
  }

  if (data)
  {
    delete [] data;
    data = NULL;
  }

  size = src.size;
  data = new T[size];
  myassert(data);

  for (i=0; i < size; i++)
  {
    data[i] = src.data[i];
  }
  return *this;
}

template <class T>
row<T> row<T>::operator+(const row<T> &other) const
{
  int i;
  class row<T> nw;


  if (this->size == other.size)
  {
    nw = *this;
    for (i=0; i < other.size; i++)
    {
      nw.data[i] += other.data[i];
    }
  }
  else
  {
    cerr << "Rows of different dimension cannnot be combined" << endl;
    myassert(this->size == other.size);
  }
  return nw;
}

template <class T>
row<T> row<T>::operator-(const row<T> &other) const
{
  int i;
  class row<T> nw;


  if (this->size == other.size)
  {
    nw = *this;
    for (i=0; i < other.size; i++)
    {
      nw.data[i] -= other.data[i];
    }
  }
  else
  {
    cerr << "Rows of different dimension cannnot be combined" << endl;
    myassert(this->size == other.size);
  }
  return nw;
}



template <class T>
row<T> row<T>::operator*(const T val) const
{
  int i;
  class row <T> nw(*this);
  for (i=0; i < size; i++)
  {
    nw.data[i] *= val;
  }
  return nw;
}

template <class T>
row<T> row<T>::operator/(const T val) const
{
  int i;
  class row <T> nw(*this);

  for (i=0; i < size; i++)
  {
    nw.data[i] /= val;
  }
  return nw;
}


template <class T>
row<T>::row(int sz)
{
  size = sz;
  data = new T[sz];
  myassert(data);
  for (int i=0; i < sz; i++)
  {
    data[i] = (T) 0;
  }
}

template <class T>
row<T>::row(void)
{
  size = 0;
  data = NULL;
}


template <class T>
row<T>::row(const row<T> &other)
{
  int i;
  size = other.size;
  data = new T[size];
  myassert(data);
  for (i=0; i < size; i++)
  {
    data[i] = other.data[i];
  }
}

template <class T>
row<T>::~row()
{
//  cout << "dest called " << data << " " << size << endl;
  size = 0;
  if (data)
  {
    delete [] data;
  }
  data = NULL;
}

template <class T>
void row<T>::print() const
{
  int i;
  if (!size)
  {
    cout << "empty row"<< endl;
    return;
  }
  cout.precision(16);
  for (i=0; i < size-1; i++)
  {
    cout << data[i] << ", ";
  }
  cout << data[size-1] << endl;
//  cout << size << "pointer: " <<data << endl;
}

template <class T>
void row<T>::set_val(int i, T val)
{
  myassert(i<size);
  data[i] = val;
}


template <class T>
matrix <T> matrix<T>::left_identity(void) const
{
  matrix<T> nw(nrows, nrows);
  for (int i=0; i < nrows; i++)
  {
    nw.set_val(i,i, (T)1);
  }
  return nw;
}

template <class T>
T matrix<T>::determinant(void) const
{
  if (nrows == ncolumns)
  {
    matrix<T> nw = *this;
    return nw.row_reduce();
  }
  else
  {
    cerr << "Determinant on non-square matrix undefined" << endl;
  }
  return (T) 0;
}

template <class T>
void matrix<T>::zero_matrix(void)
{
  for (int i=0; i < nrows; i++)
  {
    for (int j=0; j < ncolumns; j++)
    {
      set_val(i,j, (T) 0);
    }
  }
}

template <class T>
matrix <T> matrix<T>::sub_matrix(int rs, int re, int cs, int ce) const
{

  if (rs < 0 || cs < 0)
  {
    cerr << "lower bounds of sub_matrix must be larger than 0" << endl;
    myassert(rs>=0);
    myassert(cs>=0);
  }

  if (re < rs || ce < cs)
  {
    cerr << "upper bounds of sub_matrix must be at least as large as lower bounds" << endl;
    myassert(re>=rs);
    myassert(ce>=cs);
  }

  if (re >= nrows || ce >= ncolumns)
  {
    cerr << "upper bounds of sub_matrix must be contained in matrix" << endl;
    myassert(re<nrows);
    myassert(ce<ncolumns);
  }

  matrix<T> nw(re-rs+1,ce-cs+1);
  for (int i=rs; i<=re; i++)
  {
    for (int j=cs; j<=ce; j++)
    {
      nw.set_val(i-rs,j-cs, this->get_val(i,j));
    }
  } 
  return nw;
}

#if MATRIX_DECOMP_USE_GSL_MATRIX_INVERTER
// for double precision matrices just use GSL package to invert matrix
template<>
int matrix<double>::invert(void)
{
# ifdef DEBUG_
  cout << "calling optimized matrix inverter" << endl;
# endif
  if (nrows != ncolumns)
  {
    this->zero_matrix();
    cerr << "Cannot invert non-square Matrix !!!!! " << endl;
    return -1;
  }

  gsl_matrix *mat = gsl_matrix_alloc(nrows, ncolumns);
  gsl_matrix *imat = gsl_matrix_alloc(nrows, ncolumns);

  myassert(mat);
  myassert(imat);

  for (int i=0; i < nrows; i++)
  {
    for (int j = 0; j < ncolumns; j++)
    {
      gsl_matrix_set(mat, i, j, get_val(i,j));
    }
  }

  int s;

  gsl_permutation * p = gsl_permutation_alloc (nrows);

  gsl_linalg_LU_decomp (mat, p, &s);

  gsl_linalg_LU_invert(mat, p, imat);

  gsl_permutation_free (p);

  for (int i=0; i < nrows; i++)
  {
    for (int j = 0; j < ncolumns; j++)
    {
      set_val(i,j, gsl_matrix_get(imat, i, j));
    }
  }

  gsl_matrix_free(mat);
  gsl_matrix_free(imat);
  return 0;
}

// for complex<double> matrices just use GSL package to invert matrix
template<>
int matrix< complex<double> >::invert(void)
{
# ifdef DEBUG_
  cout << "calling optimized matrix inverter" << endl;
# endif
  if (nrows != ncolumns)
  {
    this->zero_matrix();
    cerr << "Cannot invert non-square Matrix !!!!! " << endl;
    return -1;
  }

  gsl_matrix_complex *mat = gsl_matrix_complex_alloc(nrows, ncolumns);
  gsl_matrix_complex *imat = gsl_matrix_complex_alloc(nrows, ncolumns);

  myassert(mat);
  myassert(imat);

  for (int i=0; i < nrows; i++)
  {
    for (int j = 0; j < ncolumns; j++)
    {
      complex<double> v = get_val(i,j);
      gsl_complex val = {{v.real(), v.imag()}};
      gsl_matrix_complex_set(mat, i, j, val);
    }
  }

  int s;

  gsl_permutation * p = gsl_permutation_alloc (nrows);

  gsl_linalg_complex_LU_decomp (mat, p, &s);

  gsl_linalg_complex_LU_invert(mat, p, imat);

  gsl_permutation_free (p);

  for (int i=0; i < nrows; i++)
  {
    for (int j = 0; j < ncolumns; j++)
    {
      gsl_complex val = gsl_matrix_complex_get(imat, i, j);
      complex<double> v = complex<double>(val.dat[0], val.dat[1]);
      set_val(i,j, v);
    }
  }

  gsl_matrix_complex_free(mat);
  gsl_matrix_complex_free(imat);
  return 0;
}

#endif

// Generic matrix inverter.
// invert matrix via Gauss-Jordan ellimination. Calls row_reduce
// which uses partial pivoting. Returns 0 on success, -1 on failure.
template <class T>
int matrix<T>::invert(void)
{
# ifdef DEBUG_
  cout << "calling UNoptimized matrix inverter" << endl;
# endif

  if (nrows == ncolumns)
  {
    matrix<T> aug = *this | this->left_identity();
    
    T det = aug.row_reduce();

    if (abs(det) <= 0.0)
    {
      this->zero_matrix();
      cerr << "Failed to Invert Matrix !!!!! " << endl;
      return -1;
    }

    *this = aug.sub_matrix(0, nrows - 1, ncolumns, 2*ncolumns-1);
    return 0;
  }
  else
  {
    this->zero_matrix();
    cerr << "Cannot invert non-square Matrix !!!!! " << endl;
    return -1;
  }
}

template <class T>
void matrix<T>::print() const
{
  if (nrows)
  {
    for (int i =0; i< nrows; i++)
    {
      data[i].row<T>::print();
    }
  }
}

template <class T>
matrix<T>::~matrix()
{
  if (nrows)
  {
    delete [] data;
    data = NULL;
    nrows = 0;
    ncolumns = 0;
  }
}

template <class T>
matrix<T>::matrix(int nr, int nc, T *dat)
{
  nrows = nr;
  ncolumns = nc;
  row<T> empty(ncolumns);
  data = new row<T> [nrows];
  myassert(data);

  for (int i=0; i < nrows; i++)
  {
    data[i] = empty;
  }

  for (int i=0; i < nrows; i++)
  {
    for (int j=0; j < ncolumns; j++)
    {
      set_val(i, j, dat[i*ncolumns+j]);
    }
  }
}

template <class T>
matrix<T>::matrix()
{
  nrows = 0;
  ncolumns = 0;
  data = NULL;
}

template <class T>
matrix<T>::matrix(int nr, int nc)
{
  nrows = nr;
  ncolumns = nc;
  data = new row<T>[nr];
  myassert(data);
  for (int i=0; i < nr; i++)
  {
    row<T> empty(nc);
    data[i] = empty;
  }
}

template <class T>
matrix<T>::matrix(const matrix<T> &src)
{
  nrows = src.nrows;
  ncolumns = src.ncolumns;

  data = new row<T> [nrows];
  myassert(data);
  
  for (int i=0; i < nrows; i ++)
  {
    data[i] = src.data[i];
  }
}


template <class T>
matrix <T>& matrix<T>::operator=(const matrix<T> &src)
{
  if (this == &src)
  {
    return *this;
  }

  this->~matrix();

  this->nrows=src.nrows;
  this->ncolumns=src.ncolumns;
  this->data = new row<T> [nrows];
  myassert(this->data);

  for (int i=0; i < nrows; i ++)
  {
    this->data[i] = src.data[i];
  }
  return *this;
}

template <class T>
matrix <T> matrix<T>::operator|(const matrix<T> &src) const
{
  if (nrows != src.nrows)
  {
    cerr << "Cannot augment two incompatible matrices" << endl;
    myassert(nrows == src.nrows);
  }

  matrix<T> nw(nrows,ncolumns+src.ncolumns);

  for (int i=0; i < nrows; i++)
  {
    for (int j=0; j < ncolumns; j++)
    {
      nw.set_val(i,j,this->get_val(i,j));
    }
    for (int j=0; j < src.ncolumns; j++)
    {
      nw.set_val(i,j+ncolumns,src.get_val(i,j));
    }
  }
  return nw;
}

template <class T>
matrix <T> matrix<T>::operator-(const matrix<T> &other) const
{
  if (nrows != other.nrows || ncolumns != other.ncolumns)
  {
    cerr << "Cannot combine matrices of different orders with '-'" << endl;
    myassert(nrows == other.nrows);
    myassert(ncolumns == other.ncolumns);
  }

  matrix<T> nw(nrows,ncolumns);

  for (int i=0; i < nrows; i++)
  {
    for (int j=0; j < ncolumns; j++)
    {
      nw.set_val(i,j,this->get_val(i,j)-other.get_val(i,j));
    }
  }
  return nw;
}


template <class T>
matrix <T> matrix<T>::operator*(const matrix<T> &rtmat) const
{
  if (ncolumns != rtmat.nrows)
  {
    cerr << "Incompatible matrices used in matrix multiplication" << endl;
    myassert(ncolumns == rtmat.nrows);
  }

  matrix<T> nw(nrows,rtmat.ncolumns);
  for (int i=0; i < nrows; i++)
  {
    for (int j=0; j < rtmat.ncolumns; j++)
    {
      T val = 0;
      for (int k=0; k < ncolumns; k++)
      {
        val += this->get_val(i,k) * rtmat.get_val(k,j);
      }
      nw.set_val(i,j,val);
    }
  }
  return nw;
}


template <class T>
void matrix<T>::set_val(int i, int j, T val)
{
  myassert(i>=0 && i<nrows);
  myassert(j>=0 && j<ncolumns);
  data[i].set_val(j, val);
}

template <class T>
T matrix<T>::get_val(int i, int j) const
{
  myassert(i>=0 && i<nrows);
  myassert(j>=0 && j<ncolumns);
  return data[i].get_val(j);
}


template <class T>
void matrix<T>::clean_small_numbers(void)
{
  for (int row=0; row < nrows; row++)
  {
    for (int col=0; col < ncolumns; col++)
    {
      if (abs(data[row].get_val(col)) <= MATRIX_ELEMENT_ROUND_OFF_LIMIT)
      {
        set_val(row,col, (T) 0);
      }
    }
  }
}

// Row reduce using partial pivoting. Returns the determinant of
// the matrix if the matrix is square. Otherwise the returned value
// is not significant.
template <class T>
T matrix<T>::row_reduce()
{
  T D = (T)1;
  int rrow = 0;
  for (int col=0; col < ncolumns; col++)
  {
    //T val = data[rrow].get_val(col);
    T val = get_val(rrow, col);

    // partial pivoting
    for (int row=rrow+1; row < nrows; row++)
    {
      if (abs(val) < abs(get_val(row, col)))
      {
        swap_rows(rrow,row);
        D*=(T)(-1);
        val = get_val(rrow, col);
      }
    }

    // column may contain all zeros, if it does we move to next column
    // of same row, otherwise we reduce on this column.
    if (abs(val) > MATRIX_ELEMENT_ROUND_OFF_LIMIT)
    {
      // set leading element to 1
      data[rrow] = data[rrow] / val;
      D*=val;
      // make all other elements in this column zero
      for (int row=0; row < nrows; row++)
      {
        if (rrow == row)
        {
          continue;
        }
        data[row] = data[row] - data[rrow]*get_val(row, col);
      }
      rrow++;
    }
    
    if (rrow >= nrows)
    {
      break;
    }
  }
  // matrix should now be in reduced row echelon form

  for (int i=0; i < nrows && i < ncolumns; i++)
  {
    D*= data[i].get_val(i);
  }
  this->clean_small_numbers();
  return D; 
}

template class matrix< complex<double> >;
template class matrix< double >;
}

#if 0

int main(int argc, char **argv)
{
  using namespace decomp_matrix_class;
  double re[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  double im[] = {1, 2, 3, 4, 5, 6, 7, 8, 1};
  const complex<double> I(0,1);

  complex <double> cdata[9] ;

  for (int i=0; i < 9; i++)
  {
    cdata[i] = re[i] + I*im[i];
  }

  class matrix <complex<double> > cm(3, 3, cdata);


  cm.print();
  cout << "------------" << endl;
  cm.invert();
  cm.print();

  cm.sub_matrix(0,2,0,0).print();
}

#endif
