#include <array>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

namespace {

constexpr double kAmp = 1.0e-3;
constexpr double kOmega = 0.2;
constexpr double kSupport = 8.0;
constexpr double kBumpSteepness = 0.35;
constexpr double kHelicity = 1.0;
constexpr double kRadialCoeff[3] = {1.0, -0.2, 0.03};

using Sym6 = std::array<double, 6>;

int SymIdx(int a, int b) {
  const int lo = a < b ? a : b;
  const int hi = a < b ? b : a;
  if (lo == 0 && hi == 0) return 0;
  if (lo == 0 && hi == 1) return 1;
  if (lo == 0 && hi == 2) return 2;
  if (lo == 1 && hi == 1) return 3;
  if (lo == 1 && hi == 2) return 4;
  return 5;
}

void LegendreAndDerivative(int n, double x, double &p, double &dpdx) {
  if (n == 0) {
    p = 1.0;
    dpdx = 0.0;
    return;
  }
  double p_nm2 = 1.0;
  double dp_nm2 = 0.0;
  double p_nm1 = x;
  double dp_nm1 = 1.0;
  if (n == 1) {
    p = p_nm1;
    dpdx = dp_nm1;
    return;
  }
  for (int l = 2; l <= n; ++l) {
    const double lr = static_cast<double>(l);
    p = ((2.0*lr - 1.0)*x*p_nm1 - (lr - 1.0)*p_nm2)/lr;
    dpdx = ((2.0*lr - 1.0)*(p_nm1 + x*dp_nm1) -
            (lr - 1.0)*dp_nm2)/lr;
    p_nm2 = p_nm1;
    dp_nm2 = dp_nm1;
    p_nm1 = p;
    dp_nm1 = dpdx;
  }
}

double Det3(const double g[3][3]) {
  return g[0][0]*(g[1][1]*g[2][2] - g[1][2]*g[2][1]) -
         g[0][1]*(g[1][0]*g[2][2] - g[1][2]*g[2][0]) +
         g[0][2]*(g[1][0]*g[2][1] - g[1][1]*g[2][0]);
}

void RadialProfile(double r, double &value, double &dr_value) {
  value = 0.0;
  dr_value = 0.0;
  if (r <= 0.0 || r >= kSupport) return;

  const double rho = r/kSupport;
  const double one_minus_rho = 1.0 - rho;
  const double denom = rho*one_minus_rho;
  if (denom <= 0.0) return;

  const double bump = std::exp(-kBumpSteepness*(1.0/denom - 4.0));
  const double dbump_dr =
      bump*kBumpSteepness*(1.0 - 2.0*rho)/
      (rho*rho*one_minus_rho*one_minus_rho*kSupport);
  const double x = 2.0*rho - 1.0;

  for (int n = 0; n < 3; ++n) {
    double legendre = 0.0;
    double dlegendre_dx = 0.0;
    LegendreAndDerivative(n, x, legendre, dlegendre_dx);
    value += kRadialCoeff[n]*bump*legendre;
    dr_value += kRadialCoeff[n]*(dbump_dr*legendre +
                                 bump*dlegendre_dx*2.0/kSupport);
  }
  value *= kAmp;
  dr_value *= kAmp;
}

Sym6 Metric(double x, double y, double z) {
  const double r = std::sqrt(x*x + y*y + z*z);
  double f = 0.0;
  double df_dr = 0.0;
  RadialProfile(r, f, df_dr);

  const double inv_r0_sq = 1.0/(kSupport*kSupport);
  const double solid_c = (x*x - y*y)*inv_r0_sq;
  const double solid_s = 2.0*x*y*inv_r0_sq;
  const double phase = kOmega*r;
  const double cos_phase = std::cos(phase);
  const double sin_phase = std::sin(phase);

  const double pol_e[3][3] = {{1.0, 0.0, 0.0},
                              {0.0, -1.0, 0.0},
                              {0.0, 0.0, 0.0}};
  const double pol_b[3][3] = {{0.0, 1.0, 0.0},
                              {1.0, 0.0, 0.0},
                              {0.0, 0.0, 0.0}};

  double h[3][3] = {};
  for (int a = 0; a < 3; ++a) {
    for (int b = 0; b < 3; ++b) {
      const double y_e = solid_c*pol_e[a][b];
      const double y_b = solid_s*pol_b[a][b];
      const double tensor = cos_phase*y_e + kHelicity*sin_phase*y_b;
      h[a][b] = f*tensor;
    }
  }

  double q[3][3] = {};
  for (int a = 0; a < 3; ++a) {
    for (int b = 0; b < 3; ++b) {
      double h2 = 0.0;
      for (int c = 0; c < 3; ++c) h2 += h[a][c]*h[c][b];
      q[a][b] = (a == b ? 1.0 : 0.0) + h[a][b] + 0.5*h2;
    }
  }

  const double q_det = Det3(q);
  const double det_scale = q_det > 0.0 ? std::pow(q_det, -1.0/3.0) : 1.0;
  Sym6 out{};
  for (int a = 0; a < 3; ++a) {
    for (int b = a; b < 3; ++b) {
      out[SymIdx(a, b)] = det_scale*q[a][b];
    }
  }
  return out;
}

struct Result {
  int nghost;
  int n;
  double dx;
  double metric_rms;
  double grad_rms;
  double lap_rms;
};

std::size_t Idx(int i, int j, int k, int n_with_ghost) {
  return static_cast<std::size_t>((k*n_with_ghost + j)*n_with_ghost + i);
}

Result Measure(int nghost, int n) {
  const int radius = nghost - 1;
  const int nt = n + 2*radius;
  const double xmin = -16.0;
  const double xmax = 16.0;
  const double dx = (xmax - xmin)/static_cast<double>(n);
  const double inv_dx = 1.0/dx;
  const double inv_dx2 = inv_dx*inv_dx;
  std::vector<Sym6> g(static_cast<std::size_t>(nt)*nt*nt);

  for (int k = 0; k < nt; ++k) {
    const double z = xmin + (static_cast<double>(k - radius) + 0.5)*dx;
    for (int j = 0; j < nt; ++j) {
      const double y = xmin + (static_cast<double>(j - radius) + 0.5)*dx;
      for (int i = 0; i < nt; ++i) {
        const double x = xmin + (static_cast<double>(i - radius) + 0.5)*dx;
        g[Idx(i, j, k, nt)] = Metric(x, y, z);
      }
    }
  }

  double metric_sum = 0.0;
  double grad_sum = 0.0;
  double lap_sum = 0.0;
  const double diag[6] = {1.0, 0.0, 0.0, 1.0, 0.0, 1.0};
  const double dV = dx*dx*dx;
  const double V = 32.0*32.0*32.0;

  for (int k = radius; k < radius + n; ++k) {
    for (int j = radius; j < radius + n; ++j) {
      for (int i = radius; i < radius + n; ++i) {
        for (int c = 0; c < 6; ++c) {
          const double gc = g[Idx(i, j, k, nt)][c] - diag[c];
          metric_sum += dV*gc*gc;
          double grad[3] = {};
          double second[3] = {};
          if (nghost == 2) {
            grad[0] = 0.5*(g[Idx(i+1, j, k, nt)][c] - g[Idx(i-1, j, k, nt)][c])*inv_dx;
            grad[1] = 0.5*(g[Idx(i, j+1, k, nt)][c] - g[Idx(i, j-1, k, nt)][c])*inv_dx;
            grad[2] = 0.5*(g[Idx(i, j, k+1, nt)][c] - g[Idx(i, j, k-1, nt)][c])*inv_dx;
            second[0] = (g[Idx(i+1, j, k, nt)][c] - 2.0*g[Idx(i, j, k, nt)][c] +
                         g[Idx(i-1, j, k, nt)][c])*inv_dx2;
            second[1] = (g[Idx(i, j+1, k, nt)][c] - 2.0*g[Idx(i, j, k, nt)][c] +
                         g[Idx(i, j-1, k, nt)][c])*inv_dx2;
            second[2] = (g[Idx(i, j, k+1, nt)][c] - 2.0*g[Idx(i, j, k, nt)][c] +
                         g[Idx(i, j, k-1, nt)][c])*inv_dx2;
          } else if (nghost == 3) {
            grad[0] = ((1.0/12.0)*g[Idx(i-2, j, k, nt)][c] -
                       (2.0/3.0)*g[Idx(i-1, j, k, nt)][c] +
                       (2.0/3.0)*g[Idx(i+1, j, k, nt)][c] -
                       (1.0/12.0)*g[Idx(i+2, j, k, nt)][c])*inv_dx;
            grad[1] = ((1.0/12.0)*g[Idx(i, j-2, k, nt)][c] -
                       (2.0/3.0)*g[Idx(i, j-1, k, nt)][c] +
                       (2.0/3.0)*g[Idx(i, j+1, k, nt)][c] -
                       (1.0/12.0)*g[Idx(i, j+2, k, nt)][c])*inv_dx;
            grad[2] = ((1.0/12.0)*g[Idx(i, j, k-2, nt)][c] -
                       (2.0/3.0)*g[Idx(i, j, k-1, nt)][c] +
                       (2.0/3.0)*g[Idx(i, j, k+1, nt)][c] -
                       (1.0/12.0)*g[Idx(i, j, k+2, nt)][c])*inv_dx;
            second[0] = (-(1.0/12.0)*g[Idx(i-2, j, k, nt)][c] +
                         (4.0/3.0)*g[Idx(i-1, j, k, nt)][c] -
                         (5.0/2.0)*g[Idx(i, j, k, nt)][c] +
                         (4.0/3.0)*g[Idx(i+1, j, k, nt)][c] -
                         (1.0/12.0)*g[Idx(i+2, j, k, nt)][c])*inv_dx2;
            second[1] = (-(1.0/12.0)*g[Idx(i, j-2, k, nt)][c] +
                         (4.0/3.0)*g[Idx(i, j-1, k, nt)][c] -
                         (5.0/2.0)*g[Idx(i, j, k, nt)][c] +
                         (4.0/3.0)*g[Idx(i, j+1, k, nt)][c] -
                         (1.0/12.0)*g[Idx(i, j+2, k, nt)][c])*inv_dx2;
            second[2] = (-(1.0/12.0)*g[Idx(i, j, k-2, nt)][c] +
                         (4.0/3.0)*g[Idx(i, j, k-1, nt)][c] -
                         (5.0/2.0)*g[Idx(i, j, k, nt)][c] +
                         (4.0/3.0)*g[Idx(i, j, k+1, nt)][c] -
                         (1.0/12.0)*g[Idx(i, j, k+2, nt)][c])*inv_dx2;
          } else {
            grad[0] = (-(1.0/60.0)*g[Idx(i-3, j, k, nt)][c] +
                       (3.0/20.0)*g[Idx(i-2, j, k, nt)][c] -
                       (3.0/4.0)*g[Idx(i-1, j, k, nt)][c] +
                       (3.0/4.0)*g[Idx(i+1, j, k, nt)][c] -
                       (3.0/20.0)*g[Idx(i+2, j, k, nt)][c] +
                       (1.0/60.0)*g[Idx(i+3, j, k, nt)][c])*inv_dx;
            grad[1] = (-(1.0/60.0)*g[Idx(i, j-3, k, nt)][c] +
                       (3.0/20.0)*g[Idx(i, j-2, k, nt)][c] -
                       (3.0/4.0)*g[Idx(i, j-1, k, nt)][c] +
                       (3.0/4.0)*g[Idx(i, j+1, k, nt)][c] -
                       (3.0/20.0)*g[Idx(i, j+2, k, nt)][c] +
                       (1.0/60.0)*g[Idx(i, j+3, k, nt)][c])*inv_dx;
            grad[2] = (-(1.0/60.0)*g[Idx(i, j, k-3, nt)][c] +
                       (3.0/20.0)*g[Idx(i, j, k-2, nt)][c] -
                       (3.0/4.0)*g[Idx(i, j, k-1, nt)][c] +
                       (3.0/4.0)*g[Idx(i, j, k+1, nt)][c] -
                       (3.0/20.0)*g[Idx(i, j, k+2, nt)][c] +
                       (1.0/60.0)*g[Idx(i, j, k+3, nt)][c])*inv_dx;
            second[0] = ((1.0/90.0)*g[Idx(i-3, j, k, nt)][c] -
                         (3.0/20.0)*g[Idx(i-2, j, k, nt)][c] +
                         (3.0/2.0)*g[Idx(i-1, j, k, nt)][c] -
                         (49.0/18.0)*g[Idx(i, j, k, nt)][c] +
                         (3.0/2.0)*g[Idx(i+1, j, k, nt)][c] -
                         (3.0/20.0)*g[Idx(i+2, j, k, nt)][c] +
                         (1.0/90.0)*g[Idx(i+3, j, k, nt)][c])*inv_dx2;
            second[1] = ((1.0/90.0)*g[Idx(i, j-3, k, nt)][c] -
                         (3.0/20.0)*g[Idx(i, j-2, k, nt)][c] +
                         (3.0/2.0)*g[Idx(i, j-1, k, nt)][c] -
                         (49.0/18.0)*g[Idx(i, j, k, nt)][c] +
                         (3.0/2.0)*g[Idx(i, j+1, k, nt)][c] -
                         (3.0/20.0)*g[Idx(i, j+2, k, nt)][c] +
                         (1.0/90.0)*g[Idx(i, j+3, k, nt)][c])*inv_dx2;
            second[2] = ((1.0/90.0)*g[Idx(i, j, k-3, nt)][c] -
                         (3.0/20.0)*g[Idx(i, j, k-2, nt)][c] +
                         (3.0/2.0)*g[Idx(i, j, k-1, nt)][c] -
                         (49.0/18.0)*g[Idx(i, j, k, nt)][c] +
                         (3.0/2.0)*g[Idx(i, j, k+1, nt)][c] -
                         (3.0/20.0)*g[Idx(i, j, k+2, nt)][c] +
                         (1.0/90.0)*g[Idx(i, j, k+3, nt)][c])*inv_dx2;
          }
          grad_sum += dV*(grad[0]*grad[0] + grad[1]*grad[1] + grad[2]*grad[2]);
          const double lap = second[0] + second[1] + second[2];
          lap_sum += dV*lap*lap;
        }
      }
    }
  }

  return {nghost, n, dx, std::sqrt(metric_sum/V), std::sqrt(grad_sum/V),
          std::sqrt(lap_sum/V)};
}

} // namespace

int main() {
  std::cout << "nghost,N,dx,metric_rms,grad_rms,lap_rms\n";
  for (int nghost : {2, 3, 4}) {
    for (int n : {32, 48, 64, 96, 128}) {
      const Result r = Measure(nghost, n);
      std::cout << r.nghost << ',' << r.n << ','
                << std::setprecision(17) << r.dx << ','
                << r.metric_rms << ',' << r.grad_rms << ','
                << r.lap_rms << '\n';
    }
  }
  return 0;
}
