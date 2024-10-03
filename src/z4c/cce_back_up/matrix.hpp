#ifndef MATRIX_HH_
#define MATRIX_HH_
#include "myassert.hpp"
#define MATRIX_ELEMENT_ROUND_OFF_LIMIT 1.0e-12

#define MATRIX_DECOMP_USE_GSL_MATRIX_INVERTER 1

namespace decomp_matrix_class {
template <class T>
class row {
  private:
    int size;
    T *data;
  public:
    int length();
    row(void);
    row(int sz);
    row(const row<T> &other);
    ~row();
    void print() const;
    void set_val(int pos, T val);
    T get_val(int pos) const {myassert(pos>-1 && pos < size); 
           return data[pos];};
    row <T> & operator=(const row<T> &src);
    row <T>  operator*(const T val) const;
    row <T>  operator/(const T val) const;
    row <T>  operator+(const row<T> &other) const;
    row <T>  operator-(const row<T> &other) const;
};


template <class T>
class matrix {
  private:
    int nrows;
    int ncolumns;
    row<T> *data;
  public:
    ~matrix();
    matrix();
    matrix(int sr, int sc);
    matrix(int nr, int nc, T *dat);
    matrix(const matrix<T> &other);
    void set_val(int nr, int nc, T val);
    T get_val(int nr, int nc) const;
    void print() const;
    matrix <T> & operator=(const matrix<T> &src);
    matrix <T> operator|(const matrix<T> &src) const;
    matrix <T> operator*(const matrix<T> &rtmat) const;
    matrix <T> operator-(const matrix<T> &other) const;
    void swap_rows(int row1, int row2)
         { row<T> temp = data[row1];
           data[row1]=data[row2]; data[row2]=temp;};
    void scale_row(int row, T scale)
         { data[row] = data[row]*scale;};
    void add_row_to_row(int src, int dest, T scale)
         { data[dest] = data[dest]+data[src]*scale;};
    T row_reduce(void);
    void clean_small_numbers(void);
    T determinant(void) const;
    void zero_matrix(void);
    int invert(void);
    matrix <T> sub_matrix(int rs, int re, int cs, int ce) const;
    matrix <T> left_identity(void) const;
};
}
#endif

