#include <iostream>
#include <complex>
#include <cmath>
#include "myassert.hpp"
#include "sYlm.hpp"
#include "decomp.hpp"

#define Max(a_,b_) ((a_)>(b_)? (a_):(b_))
#define Min(a_,b_) ((a_)<(b_)? (a_):(b_))
#define ABS(x_) ((x_)>0 ? (x_) : (-(x_)))

#define PI (3.1415926535897932384626433832795)
#define TwoPI (6.2831853071795864769252867665590)

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

decomp_info::decomp_info(int ss, int nnl, int nnn, int nnmu, int nnphi, int nnx)
{
  const complex<double> I(0,1);
  
  s = ss;
  nl = nnl;
  nn = nnn;

  nmu = nnmu;
  nphi = nnphi;
  nx = nnx;

  const int nlmmodes = nl*(nl+2*ABS(s));
 
  const double dmu = 2.0 / (double)(nmu);
  const double dphi = TwoPI / (double)(nphi);
  const double dx = 2.0 / (double)(nx);

  ncolloc = new double [nx];
  mucolloc = new double [nmu*nphi];
  phicolloc = new double [nmu*nphi];


  myassert(ncolloc);
  myassert(mucolloc);
  myassert(phicolloc);

  Yr = new double *[nlmmodes];
  Yi = new double *[nlmmodes];

  myassert(Yr);
  myassert(Yi);

  for (int i=0; i < nlmmodes; i++)
  {
    Yr[i] = new double [nmu*nphi];
    Yi[i] = new double [nmu*nphi];

    myassert(Yr[i]);
    myassert(Yi[i]);
  }

  Pn = new double *[nn];
  myassert(Pn);

  for (int i=0; i < nn; i++)
  {
    Pn[i] = new double [nx];
    myassert(Pn[i]); 
  }

  for (int i = 0; i < nx; i++)
  {
    ncolloc[i] = -1.0 + (i+0.5) * dx;
  }

  for (int i = 0; i < nmu; i++)
  {
    for (int j = 0; j < nphi; j++)
    {
      const int indx = j + i*nphi;
      mucolloc[indx] = -1.0 + (i+0.5) * dmu;
      phicolloc[indx] = j*dphi;
    }
  }

  int sfun = 0;
  for (int ll = 0; ll < nl; ll++)
  {
    const int l = ABS(s) + ll;
    for (int m=-l; m < l+1; m++, sfun++)
    {
      for (int indx = 0; indx < nmu*nphi; indx ++)
      {
        const double mu = mucolloc[indx];
        const double phi = phicolloc[indx];
        complex<double> ylm = sYlm_mu(s, l, m, mu, phi);
        Yr[sfun][indx] = ylm.real();
        Yi[sfun][indx] = ylm.imag();
      }
    }
  }

  for (int fun = 0; fun < nn; fun++)
  {
    for (int i=0; i < nx; i++)
    {
#ifdef USE_LEGENDRE
      Pn[fun][i] = LegendreP(fun, ncolloc[i]);
#else
      Pn[fun][i] = ChebyshevU(fun, ncolloc[i]);
#endif
    }
  }


  matrix <double> lsmat(nlmmodes*2,nlmmodes*2);
  for (int row = 0; row < nlmmodes; row++)
  {
    for (int col = 0; col < nlmmodes; col ++)
    {
      double el_r_c = 0; 
      double el_r_cp = 0; 
      double el_rp_c = 0; 
      double el_rp_cp = 0; 
      for (int i=0; i < nmu*nphi; i++)
      {
        el_r_c += Yr[row][i]*Yr[col][i] + Yi[row][i]*Yi[col][i];
        el_r_cp += Yi[row][i]*Yr[col][i] - Yr[row][i]*Yi[col][i];

        el_rp_c += Yr[row][i]*Yi[col][i] - Yi[row][i]*Yr[col][i];
        el_rp_cp += Yr[row][i]*Yr[col][i] + Yi[row][i]*Yi[col][i];
      }
      lsmat.set_val(row, col, el_r_c);
      lsmat.set_val(row, col+nlmmodes, el_r_cp);
      lsmat.set_val(row+nlmmodes, col, el_rp_c);
      lsmat.set_val(row+nlmmodes, col+nlmmodes, el_rp_cp);
    }
  }

  myassert(lsmat.invert()==0);

  lmmat = lsmat;

  matrix <double> nnw(nn,nn);

  for (int row = 0; row < nn; row++)
  {
    for (int col = 0; col < nn; col ++)
    {
      double el_r_c = 0; 
      for (int i=0; i < nx; i++)
      {
        el_r_c += Pn[row][i]*Pn[col][i];
      }
      nnw.set_val(row, col, el_r_c);
    }
  }
  myassert(nnw.invert()==0);
  nmat = nnw;
}

decomp_info::~decomp_info()
{
  const int nlmmodes = nl*(nl+2*ABS(s));
  delete [] ncolloc;
  delete [] mucolloc;
  delete [] phicolloc;

  for (int i=0; i < nlmmodes; i++)
  {
    delete [] Yr[i];
    delete [] Yi[i];
  }
  delete [] Yr;
  delete [] Yi;

  Yr = NULL;
  Yi = NULL;

  for (int i=0; i < nn; i++)
  {
    delete [] Pn[i];
  }
  delete [] Pn;
  Pn = NULL;

  nl = 0;
  nn = 0;
  nmu = 0;
  nphi = 0;
  nx = 0;
}

int decomp_info::get_ncolloc(int n, double *dst) const
{
  myassert(n == nx);
  for (int i=0; i < n; i++)
  {
    dst[i] = ncolloc[i];
  }
   return 0;
}

int decomp_info::get_mucolloc(int n, double *dst) const
{
  myassert(n == nmu*nphi);
  for (int i=0; i < n; i++)
  {
    dst[i] = mucolloc[i];
  }
   return 0;
}

int decomp_info::get_phicolloc(int n, double *dst) const
{
  myassert(n == nmu*nphi);
  for (int i=0; i < n; i++)
  {
    dst[i] = phicolloc[i];
  }
   return 0;
}


const decomp_info *initialize(int s, int nl, int nn, int nmu, int nphi, int nx)
{
  decomp_info *dinfo_p = NULL;
  dinfo_p = new decomp_info(s, nl, nn, nmu, nphi, nx);
  return dinfo_p;
}

/* re and im are 1d arrays with the following ordering

   re[0] -> (mu,phi0)
   re[1] -> (mu,phi0)
   re[2] -> (theta2,phi0)
   re[nl] -> (theta0,phi1)
   re[nl+1] -> (theta1,phi1) ....
*/

int decompose2D (const decomp_info *dinfo_p, const double *re, const double *im, double *ore, double *oim)
{
  myassert(dinfo_p);
  const int nmu = dinfo_p->nmu;
  const int nphi = dinfo_p->nphi;
  const int nl = dinfo_p->nl;
  const int s = dinfo_p->s;
  double **Yr = dinfo_p->Yr;
  double **Yi = dinfo_p->Yi;

  const int nlmmodes = nl*(nl+2*ABS(s));
  myassert(re);
  myassert(im);
  myassert(ore);
  myassert(oim);

  double *B =  new double[2*nlmmodes];
  myassert(B);

  for (int row=0; row < nlmmodes; row++)
  {
    B[row] = 0;
    B[row+nlmmodes] = 0;
    for (int i = 0; i < nmu*nphi; i++)
    {
      B[row] += Yr[row][i]*re[i] + Yi[row][i]*im[i];
      B[row+nlmmodes] += Yr[row][i]*im[i] - Yi[row][i]*re[i];
    }
  }

  for (int row=0; row < nlmmodes; row++)
  {
    ore[row] = 0;
    oim[row] = 0;

    for (int col = 0; col < 2*nlmmodes; col++)
    {
      ore[row] += dinfo_p->lmmat.get_val(row,col) * B[col];
      oim[row] += dinfo_p->lmmat.get_val(row+nlmmodes,col) * B[col];
    }
  }

  delete [] B;
  return 0;
}

int decompose3D (const decomp_info *dinfo_p, const double *re, const double *im, double *ore, double *oim)
{
  myassert(dinfo_p);
  const int nmu = dinfo_p->nmu;
  const int nphi = dinfo_p->nphi;
  const int nx = dinfo_p->nx;
  const int nl = dinfo_p->nl;
  const int nn = dinfo_p->nn;
  const int s = dinfo_p->s;
  double **Pn = dinfo_p->Pn;

  const int nlmmodes = nl*(nl+2*ABS(s));

  myassert(re && im && ore && oim);

  for (int k=0; k < nx; k++)
  {
    const int indx = k*nmu*nphi;
    const int oindx = k*nlmmodes;
    
    decompose2D (dinfo_p, re+indx, im+indx, ore+oindx, oim+oindx);
  }

  double *Br = new double [nn];
  double *Bi = new double [nn];


  for (int a = 0; a < nlmmodes; a++)
  {
    for (int k=0; k < nn; k++)
    {
      Br[k] = 0;
      Bi[k] = 0;

      for (int i=0; i < nx; i++)
      {
        Br[k] += ore[a + i*nlmmodes] * Pn[k][i];
        Bi[k] += oim[a + i*nlmmodes] * Pn[k][i];
      }
    }

    for (int row=0; row < nn; row++)
    {
      ore[row*nlmmodes+a] = 0;
      oim[row*nlmmodes+a] = 0;

      for (int col = 0; col < nn; col++)
      {
        ore[row*nlmmodes+a] += dinfo_p->nmat.get_val(row,col) * Br[col];
        oim[row*nlmmodes+a] += dinfo_p->nmat.get_val(row,col) * Bi[col];
      }
    }
  }
  delete [] Br;
  delete [] Bi;
  return 0;
}
}
