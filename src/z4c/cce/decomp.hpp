#ifndef DECOMP_DECOMP_H
#define DECOMP_DECOMP_H
#include <complex>
#include "matrix.hpp"
#include "sYlm.hpp"

#ifdef USE_LEGENDRE
# ERROR: Do not use Legendre
#  include "Legendre.hpp"
#else
#  include "Chebyshev.hpp"
#endif


using namespace decomp_matrix_class;
using namespace decomp_sYlm;
#ifdef USE_LEGENDRE
using namespace decomp_Legendre;
#else
using namespace decomp_Chebyshev;
#endif
using namespace std;

namespace decomp_decompose
{

class decomp_info
{
  private:
    int s;
    int nl;
    int nn;
    int nmu;
    int nphi;
    int nx;
    double *ncolloc;
    double *mucolloc;
    double *phicolloc;
    matrix <double> lmmat;
    matrix <double> nmat;
    double **Pn;
    double **Yr;
    double **Yi;
  public:
    decomp_info(int ss, int nll, int nnn, int nnmu, int nnphi, int nnx);
    ~decomp_info();
    int spin() const { return s;};
    int get_nmu() const { return nmu;};
    int get_nl() const { return nl;};
    int get_nphi() const { return nphi;};
    int get_nx() const { return nx;};
    int get_ncolloc(int n, double *dst) const;
    int get_mucolloc(int n, double *dst) const;
    int get_phicolloc(int n, double *dst) const;
    friend int decompose2D (const decomp_info *dinfo_p, const double *re, const double *im, double *ore, double *oim);
    friend int decompose3D (const decomp_info *dinfo_p, const double *re, const double *im, double *ore, double *oim);
};

int decompose2D (const decomp_info *dinfo_p, const double *re, const double *im, double *ore, double *oim);
int decompose3D (const decomp_info *dinfo_p, const double *re, const double *im, double *ore, double *oim);
const decomp_info *initialize(int s, int nl, int nn, int nmu, int nphi, int nx);

}
#endif
