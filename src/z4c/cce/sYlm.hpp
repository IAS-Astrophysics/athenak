#ifndef DECOMP_SYLM_HH
#define DECOMP_SYLM_HH
#include <complex>
#include <cmath>
namespace decomp_sYlm
{
  using namespace std;
  double sPlm(int s, int l, int m, double theta);
  double sPlm_mu(int s, int l, int m, double mu);
  complex <double> sYlm(int s, int l, int m, double theta, double phi);
  complex <double> sYlm_mu(int s, int l, int m, double mu, double phi);
}
#endif
