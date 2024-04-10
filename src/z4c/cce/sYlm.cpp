#include <complex>
#include <cmath>
#include "myassert.hpp"
#include "sYlm.hpp"

#define Max(a_,b_) ((a_)>(b_)? (a_):(b_))
#define Min(a_,b_) ((a_)<(b_)? (a_):(b_))
#define FourPi 12.566370614359172953850573533118
#define ABS(x_) ((x_)>0 ? (x_) : (-(x_)))

#define RO (1.0e-9)

namespace decomp_sYlm
{
using namespace std;
static long double fact(int i)
{
  int j;
  long double temp = 1;

  myassert(i>-1);

  for (j=2; j <=i; j++)
  {
    temp *=j;
  }
  return temp;
}

double sPlm(int s, int l, int m, double theta)
{
  int k;
  long double temp = 0;
  s= -s;  /* oops below is really the definition for
            -s not s */

  myassert(l>=ABS(s));
  myassert(l>=ABS(m));

  const long double sc = (1-2*(ABS(s)%2))*
        sqrt((2*l+1)/(FourPi)*fact(l+m)*fact(l-m)*fact(l+s)*fact(l-s));

  for (k= Max(0, m-s); k <= Min(l+m, l-s); k++)
  {
    long double t = sc;

    t/= fact(l+m-k);
    t/= fact(l-s-k);
    t/= fact(k);
    t/= fact(k+s-m);

    temp += (1-2*(ABS(k)%2))*t*
       pow(cos(0.5*theta),(double)(2*l+m-s-2*k))*
       pow(sin(0.5*theta),(double)(2*k+s-m));
  }

  return (double)temp;
}

complex <double> sYlm (int s, int l, int m, double theta, double phi)
{
  const complex<double> I(0,1);
  return sPlm(s,l,m,theta)*exp(I*(complex<double>)(m*phi));
}

complex <double> sYlm_mu (int s, int l, int m, double mu, double phi)
{
  const complex<double> I(0,1);
  return sPlm_mu(s,l,m,mu)*exp(I*(complex<double>)(m*phi));
}

double sPlm_mu(int s, int l, int m, double mu)
{
  int k;
  long double temp = 0;
  myassert(mu > -1 - RO);
  myassert(mu < 1 + RO);
  double beta = sqrt(0.5*(1+mu));
  double alpha = sqrt(0.5*(1-mu));

  s= -s;  /* oops below is really the definition for
            -s not s */

  myassert(l>=ABS(s));
  myassert(l>=ABS(m));

  const long double sc = (1-2*(ABS(s)%2))*
        sqrt((2*l+1)/(FourPi)*fact(l+m)*fact(l-m)*fact(l+s)*fact(l-s));

  for (k= Max(0, m-s); k <= Min(l+m, l-s); k++)
  {
    long double t = sc;

    t/= fact(l+m-k);
    t/= fact(l-s-k);
    t/= fact(k);
    t/= fact(k+s-m);

    temp += (1-2*(ABS(k)%2)) *t*
       pow(beta,(double)(2*l+m-s-2*k))*
       pow(alpha,(double)(2*k+s-m));
  }

  return (double)temp;
}
}
