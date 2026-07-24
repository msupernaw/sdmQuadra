#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

#ifndef R_NO_REMAP
#define R_NO_REMAP
#endif
#include <R.h>
#include <Rinternals.h>

#include "core/autodiff.hpp"
#include "core/inference/ad_delta_method_vector.hpp"
#include "core/inference/fixed_effect_covariance.hpp"
#include "core/laplace/laplace_fixed_gradient.hpp"
#include "core/laplace/laplace_implicit_workspace.hpp"
#include "core/laplace/sparse_huu_factorization.hpp"
#include "core/uncertainty/linear_predictor_marginal.hpp"
#include "core/uncertainty/random_effect_marginal.hpp"
#include "math/distributions.hpp"

DECLARE_ADGRAPH()

namespace {

template <typename Type>
Type observation_nll(const Type &eta, double y, double weight, bool nb2,
                     bool gaussian, const Type &likelihood_parameter) {
  if (gaussian) {
    constexpr double log_sqrt_2pi = 0.91893853320467274178;
    const Type residual = (Type(y) - eta) / likelihood_parameter;
    return Type(weight) *
           (Type(0.5) * residual * residual + log(likelihood_parameter) +
            Type(log_sqrt_2pi));
  }
  if (nb2)
    return -Type(weight) *
           quadra::dnbinom2(static_cast<int>(y), exp(eta),
                            likelihood_parameter, true);
  return Type(weight) *
         (exp(eta) - Type(y) * eta + Type(std::lgamma(y + 1.0)));
}

struct ADGraphReset {
  ~ADGraphReset() { had::g_ADGraph = nullptr; }
};

template <typename Type>
Type gaussian_nll(const std::vector<Type> &beta, const double *x,
                  const double *y, const double *offset, int n, int p) {
  Type nll = Type(0.0);
  constexpr double log_sqrt_2pi = 0.91893853320467274178;

  for (int i = 0; i < n; ++i) {
    Type eta = Type(offset[i]);
    for (int j = 0; j < p; ++j) {
      eta = eta + beta[j] * x[i + n * j];
    }
    const Type residual = Type(y[i]) - eta;
    nll = nll + Type(0.5) * residual * residual + Type(log_sqrt_2pi);
  }

  return nll;
}

class GaussianRandomInterceptModel {
public:
  GaussianRandomInterceptModel(const double *x, const double *y,
                               const double *offset, const int *group, int n,
                               int p, int n_group)
      : x_(x), y_(y), offset_(offset), group_(group), n_(n), p_(p),
        n_group_(n_group) {}

  void initialize(quadra::ModelReportContext &ctx) const { ctx.clear(); }

  template <typename Type>
  Type evaluate(const std::vector<Type> &parameters,
                quadra::ModelReportContext &) const {
    const Type sigma_obs = exp(parameters[static_cast<size_t>(p_)]);
    const Type sigma_group = exp(parameters[static_cast<size_t>(p_ + 1)]);
    Type nll = Type(0.0);
    constexpr double log_sqrt_2pi = 0.91893853320467274178;

    for (int i = 0; i < n_; ++i) {
      Type eta = Type(offset_[i]);
      for (int j = 0; j < p_; ++j) {
        eta = eta + parameters[static_cast<size_t>(j)] * x_[i + n_ * j];
      }
      eta = eta + parameters[static_cast<size_t>(p_ + 2 + group_[i])];
      const Type z = (Type(y_[i]) - eta) / sigma_obs;
      nll = nll + Type(0.5) * z * z + log(sigma_obs) +
            Type(log_sqrt_2pi);
    }

    for (int g = 0; g < n_group_; ++g) {
      const Type u = parameters[static_cast<size_t>(p_ + 2 + g)];
      const Type z = u / sigma_group;
      nll = nll + Type(0.5) * z * z + log(sigma_group) +
            Type(log_sqrt_2pi);
    }

    return nll;
  }

private:
  const double *x_;
  const double *y_;
  const double *offset_;
  const int *group_;
  int n_;
  int p_;
  int n_group_;
};

class GaussianSparseFieldModel {
public:
  GaussianSparseFieldModel(const double *x, const double *y,
                           const double *offset, const int *node, int n, int p,
                           int n_node, const int *q_i, const int *q_j,
                           const double *q_x, int n_q, double logdet_q)
      : x_(x), y_(y), offset_(offset), node_(node), n_(n), p_(p),
        n_node_(n_node), q_i_(q_i), q_j_(q_j), q_x_(q_x), n_q_(n_q),
        logdet_q_(logdet_q) {}

  void initialize(quadra::ModelReportContext &ctx) const { ctx.clear(); }

  template <typename Type>
  Type evaluate(const std::vector<Type> &parameters,
                quadra::ModelReportContext &) const {
    const Type sigma_obs = exp(parameters[static_cast<size_t>(p_)]);
    const Type tau = exp(parameters[static_cast<size_t>(p_ + 1)]);
    Type nll = Type(0.0);
    constexpr double log_sqrt_2pi = 0.91893853320467274178;

    for (int i = 0; i < n_; ++i) {
      Type eta = Type(offset_[i]);
      for (int j = 0; j < p_; ++j) {
        eta = eta + parameters[static_cast<size_t>(j)] * x_[i + n_ * j];
      }
      eta = eta + parameters[static_cast<size_t>(p_ + 2 + node_[i])];
      const Type z = (Type(y_[i]) - eta) / sigma_obs;
      nll = nll + Type(0.5) * z * z + log(sigma_obs) +
            Type(log_sqrt_2pi);
    }

    Type quadratic = Type(0.0);
    for (int k = 0; k < n_q_; ++k) {
      const Type ui = parameters[static_cast<size_t>(p_ + 2 + q_i_[k])];
      const Type uj = parameters[static_cast<size_t>(p_ + 2 + q_j_[k])];
      quadratic = quadratic + Type(q_x_[k]) * ui * uj;
    }

    nll = nll + Type(0.5) * tau * tau * quadratic -
          Type(n_node_) * log(tau) - Type(0.5 * logdet_q_) +
          Type(n_node_ * log_sqrt_2pi);
    return nll;
  }

private:
  const double *x_;
  const double *y_;
  const double *offset_;
  const int *node_;
  int n_;
  int p_;
  int n_node_;
  const int *q_i_;
  const int *q_j_;
  const double *q_x_;
  int n_q_;
  double logdet_q_;
};

class GaussianProjectedFieldModel {
public:
  GaussianProjectedFieldModel(
      const double *x, const double *y, const double *offset, int n, int p,
      int n_node, const int *a_i, const int *a_j, const double *a_x, int n_a,
      const int *q_i, const int *q_j, const double *q_x, int n_q,
      double logdet_q)
      : x_(x), y_(y), offset_(offset), n_(n), p_(p), n_node_(n_node),
        a_i_(a_i), a_j_(a_j), a_x_(a_x), n_a_(n_a), q_i_(q_i), q_j_(q_j),
        q_x_(q_x), n_q_(n_q), logdet_q_(logdet_q) {}

  void initialize(quadra::ModelReportContext &ctx) const { ctx.clear(); }

  template <typename Type>
  Type evaluate(const std::vector<Type> &parameters,
                quadra::ModelReportContext &) const {
    const Type sigma_obs = exp(parameters[static_cast<size_t>(p_)]);
    const Type tau = exp(parameters[static_cast<size_t>(p_ + 1)]);
    std::vector<Type> projected(static_cast<size_t>(n_), Type(0.0));
    for (int k = 0; k < n_a_; ++k) {
      projected[static_cast<size_t>(a_i_[k])] =
          projected[static_cast<size_t>(a_i_[k])] +
          Type(a_x_[k]) *
              parameters[static_cast<size_t>(p_ + 2 + a_j_[k])];
    }

    Type nll = Type(0.0);
    constexpr double log_sqrt_2pi = 0.91893853320467274178;
    for (int i = 0; i < n_; ++i) {
      Type eta = Type(offset_[i]) + projected[static_cast<size_t>(i)];
      for (int j = 0; j < p_; ++j) {
        eta = eta + parameters[static_cast<size_t>(j)] * x_[i + n_ * j];
      }
      const Type z = (Type(y_[i]) - eta) / sigma_obs;
      nll = nll + Type(0.5) * z * z + log(sigma_obs) +
            Type(log_sqrt_2pi);
    }

    Type quadratic = Type(0.0);
    for (int k = 0; k < n_q_; ++k) {
      const Type ui = parameters[static_cast<size_t>(p_ + 2 + q_i_[k])];
      const Type uj = parameters[static_cast<size_t>(p_ + 2 + q_j_[k])];
      quadratic = quadratic + Type(q_x_[k]) * ui * uj;
    }
    nll = nll + Type(0.5) * tau * tau * quadratic -
          Type(n_node_) * log(tau) - Type(0.5 * logdet_q_) +
          Type(n_node_ * log_sqrt_2pi);
    return nll;
  }

private:
  const double *x_;
  const double *y_;
  const double *offset_;
  int n_;
  int p_;
  int n_node_;
  const int *a_i_;
  const int *a_j_;
  const double *a_x_;
  int n_a_;
  const int *q_i_;
  const int *q_j_;
  const double *q_x_;
  int n_q_;
  double logdet_q_;
};

class GaussianSpdeFieldModel {
public:
  GaussianSpdeFieldModel(
      const double *x, const double *y, const double *offset, int n, int p,
      int n_node, const int *a_i, const int *a_j, const double *a_x, int n_a,
      const int *m0_i, const int *m0_j, const double *m0_x, int n_m0,
      const int *m1_i, const int *m1_j, const double *m1_x, int n_m1,
      const int *m2_i, const int *m2_j, const double *m2_x, int n_m2)
      : x_(x), y_(y), offset_(offset), n_(n), p_(p), n_node_(n_node),
        a_i_(a_i), a_j_(a_j), a_x_(a_x), n_a_(n_a), m0_i_(m0_i),
        m0_j_(m0_j), m0_x_(m0_x), n_m0_(n_m0), m1_i_(m1_i),
        m1_j_(m1_j), m1_x_(m1_x), n_m1_(n_m1), m2_i_(m2_i),
        m2_j_(m2_j), m2_x_(m2_x), n_m2_(n_m2) {}

  void initialize(quadra::ModelReportContext &ctx) const { ctx.clear(); }

  template <typename Type>
  Type evaluate(const std::vector<Type> &parameters,
                quadra::ModelReportContext &) const {
    const Type sigma_obs = exp(parameters[static_cast<size_t>(p_)]);
    const Type tau = exp(parameters[static_cast<size_t>(p_ + 1)]);
    const Type log_kappa = parameters[static_cast<size_t>(p_ + 2)];
    const double kappa2 =
        std::exp(2.0 * quadra::value_of_arithmetic_or_ad(log_kappa));
    const double kappa4 = kappa2 * kappa2;

    std::vector<Type> projected(static_cast<size_t>(n_), Type(0.0));
    for (int k = 0; k < n_a_; ++k) {
      projected[static_cast<size_t>(a_i_[k])] =
          projected[static_cast<size_t>(a_i_[k])] +
          Type(a_x_[k]) *
              parameters[static_cast<size_t>(p_ + 3 + a_j_[k])];
    }

    Type nll = Type(0.0);
    constexpr double log_sqrt_2pi = 0.91893853320467274178;
    for (int i = 0; i < n_; ++i) {
      Type eta = Type(offset_[i]) + projected[static_cast<size_t>(i)];
      for (int j = 0; j < p_; ++j) {
        eta = eta + parameters[static_cast<size_t>(j)] * x_[i + n_ * j];
      }
      const Type z = (Type(y_[i]) - eta) / sigma_obs;
      nll = nll + Type(0.5) * z * z + log(sigma_obs) +
            Type(log_sqrt_2pi);
    }

    Type quadratic = Type(0.0);
    add_quadratic(parameters, m0_i_, m0_j_, m0_x_, n_m0_, kappa4,
                  quadratic);
    add_quadratic(parameters, m1_i_, m1_j_, m1_x_, n_m1_, 2.0 * kappa2,
                  quadratic);
    add_quadratic(parameters, m2_i_, m2_j_, m2_x_, n_m2_, 1.0,
                  quadratic);

    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(static_cast<size_t>(n_m0_ + n_m1_ + n_m2_));
    add_triplets(triplets, m0_i_, m0_j_, m0_x_, n_m0_, kappa4);
    add_triplets(triplets, m1_i_, m1_j_, m1_x_, n_m1_, 2.0 * kappa2);
    add_triplets(triplets, m2_i_, m2_j_, m2_x_, n_m2_, 1.0);
    Eigen::SparseMatrix<double> q(n_node_, n_node_);
    q.setFromTriplets(triplets.begin(), triplets.end());
    bool logdet_ok = false;
    const double logdet_q = quadra::sparse_ldlt_logdet(q, logdet_ok);
    if (!logdet_ok) {
      throw std::runtime_error("SPDE precision matrix is not positive definite");
    }

    nll = nll + Type(0.5) * tau * tau * quadratic -
          Type(n_node_) * log(tau) - Type(0.5 * logdet_q) +
          Type(n_node_ * log_sqrt_2pi);
    return nll;
  }

private:
  template <typename Type>
  void add_quadratic(const std::vector<Type> &parameters, const int *row,
                     const int *col, const double *value, int count,
                     double scale, Type &quadratic) const {
    for (int k = 0; k < count; ++k) {
      const Type ui =
          parameters[static_cast<size_t>(p_ + 3 + row[k])];
      const Type uj =
          parameters[static_cast<size_t>(p_ + 3 + col[k])];
      quadratic = quadratic + Type(scale * value[k]) * ui * uj;
    }
  }

  static void add_triplets(std::vector<Eigen::Triplet<double>> &out,
                           const int *row, const int *col,
                           const double *value, int count, double scale) {
    for (int k = 0; k < count; ++k) {
      out.emplace_back(row[k], col[k], scale * value[k]);
    }
  }

  const double *x_, *y_, *offset_;
  int n_, p_, n_node_;
  const int *a_i_, *a_j_;
  const double *a_x_;
  int n_a_;
  const int *m0_i_, *m0_j_;
  const double *m0_x_;
  int n_m0_;
  const int *m1_i_, *m1_j_;
  const double *m1_x_;
  int n_m1_;
  const int *m2_i_, *m2_j_;
  const double *m2_x_;
  int n_m2_;
};

class PoissonSpdeFieldModel {
public:
  PoissonSpdeFieldModel(
      const double *x, const double *y, const double *offset,
      const double *weights, int n, int p,
      int n_node, const int *a_i, const int *a_j, const double *a_x, int n_a,
      const int *m0_i, const int *m0_j, const double *m0_x, int n_m0,
      const int *m1_i, const int *m1_j, const double *m1_x, int n_m1,
      const int *m2_i, const int *m2_j, const double *m2_x, int n_m2,
      bool nb2 = false, bool gaussian = false)
      : x_(x), y_(y), offset_(offset), weights_(weights), n_(n), p_(p),
        n_node_(n_node),
        a_i_(a_i), a_j_(a_j), a_x_(a_x), n_a_(n_a), m0_i_(m0_i),
        m0_j_(m0_j), m0_x_(m0_x), n_m0_(n_m0), m1_i_(m1_i),
        m1_j_(m1_j), m1_x_(m1_x), n_m1_(n_m1), m2_i_(m2_i),
        m2_j_(m2_j), m2_x_(m2_x), n_m2_(n_m2), nb2_(nb2),
        gaussian_(gaussian) {}

  void initialize(quadra::ModelReportContext &ctx) const { ctx.clear(); }

  template <typename Type>
  Type evaluate(const std::vector<Type> &parameters,
                quadra::ModelReportContext &) const {
    const Type tau = exp(parameters[static_cast<size_t>(p_)]);
    const Type kappa2 =
        exp(Type(2.0) * parameters[static_cast<size_t>(p_ + 1)]);
    const Type kappa4 = kappa2 * kappa2;
    const int random_base =
        p_ + 2 + static_cast<int>(nb2_ || gaussian_);
    const Type likelihood_parameter = (nb2_ || gaussian_)
        ? exp(parameters[static_cast<size_t>(p_ + 2)])
        : Type(1.0);
    std::vector<Type> projected(static_cast<size_t>(n_), Type(0.0));
    for (int k = 0; k < n_a_; ++k) {
      projected[static_cast<size_t>(a_i_[k])] =
          projected[static_cast<size_t>(a_i_[k])] +
          Type(a_x_[k]) *
              parameters[static_cast<size_t>(random_base + a_j_[k])];
    }
    Type nll = Type(0.0);
    for (int i = 0; i < n_; ++i) {
      Type eta = Type(offset_[i]) + projected[static_cast<size_t>(i)];
      for (int j = 0; j < p_; ++j) {
        eta = eta + parameters[static_cast<size_t>(j)] * x_[i + n_ * j];
      }
      nll = nll + observation_nll(
          eta, y_[i], weights_ == nullptr ? 1.0 : weights_[i], nb2_,
          gaussian_,
          likelihood_parameter);
    }
    Type quadratic = Type(0.0);
    add_quadratic(parameters, m0_i_, m0_j_, m0_x_, n_m0_, kappa4,
                  quadratic);
    add_quadratic(parameters, m1_i_, m1_j_, m1_x_, n_m1_, Type(2.0) * kappa2,
                  quadratic);
    add_quadratic(parameters, m2_i_, m2_j_, m2_x_, n_m2_, 1.0,
                  quadratic);
    std::vector<Eigen::Triplet<double>> entries;
    entries.reserve(static_cast<size_t>(n_m0_ + n_m1_ + n_m2_));
    add_triplets(entries, m0_i_, m0_j_, m0_x_, n_m0_,
                 quadra::value_of_arithmetic_or_ad(kappa4));
    add_triplets(entries, m1_i_, m1_j_, m1_x_, n_m1_,
                 2.0 * quadra::value_of_arithmetic_or_ad(kappa2));
    add_triplets(entries, m2_i_, m2_j_, m2_x_, n_m2_, 1.0);
    Eigen::SparseMatrix<double> q(n_node_, n_node_);
    q.setFromTriplets(entries.begin(), entries.end());
    bool ok = false;
    const double logdet_q = quadra::sparse_ldlt_logdet(q, ok);
    if (!ok) throw std::runtime_error("SPDE precision is not positive definite");
    constexpr double log_sqrt_2pi = 0.91893853320467274178;
    nll = nll + Type(0.5) * tau * tau * quadratic -
          Type(n_node_) * log(tau) - Type(0.5 * logdet_q) +
          Type(n_node_ * log_sqrt_2pi);
    return nll;
  }

private:
  template <typename Type, typename Scale>
  void add_quadratic(const std::vector<Type> &parameters, const int *row,
                     const int *col, const double *value, int count,
                     const Scale &scale, Type &out) const {
    for (int k = 0; k < count; ++k) {
      const int random_base =
          p_ + 2 + static_cast<int>(nb2_ || gaussian_);
      const Type ui = parameters[static_cast<size_t>(random_base + row[k])];
      const Type uj = parameters[static_cast<size_t>(random_base + col[k])];
      out = out + Type(value[k]) * scale * ui * uj;
    }
  }
  static void add_triplets(std::vector<Eigen::Triplet<double>> &out,
                           const int *row, const int *col,
                           const double *value, int count, double scale) {
    for (int k = 0; k < count; ++k)
      out.emplace_back(row[k], col[k], scale * value[k]);
  }
  const double *x_, *y_, *offset_, *weights_;
  int n_, p_, n_node_;
  const int *a_i_, *a_j_;
  const double *a_x_;
  int n_a_;
  const int *m0_i_, *m0_j_;
  const double *m0_x_;
  int n_m0_;
  const int *m1_i_, *m1_j_;
  const double *m1_x_;
  int n_m1_;
  const int *m2_i_, *m2_j_;
  const double *m2_x_;
  int n_m2_;
  bool nb2_;
  bool gaussian_;
};

class PoissonSpatiotemporalIidModel {
public:
  PoissonSpatiotemporalIidModel(
      const double *x, const double *y, const double *offset,
      const double *weights, const int *time, int n, int p, int n_node,
      int n_time, const int *a_i, const int *a_j,
      const double *a_x, int n_a, const int *m0_i, const int *m0_j,
      const double *m0_x, int n_m0, const int *m1_i, const int *m1_j,
      const double *m1_x, int n_m1, const int *m2_i, const int *m2_j,
      const double *m2_x, int n_m2, bool ar1 = false, bool rw = false,
      bool separate_range = false, bool nb2 = false, bool gaussian = false)
      : x_(x), y_(y), offset_(offset), weights_(weights), time_(time),
        n_(n), p_(p),
        n_node_(n_node), n_time_(n_time), a_i_(a_i), a_j_(a_j), a_x_(a_x),
        n_a_(n_a), m0_i_(m0_i), m0_j_(m0_j), m0_x_(m0_x), n_m0_(n_m0),
        m1_i_(m1_i), m1_j_(m1_j), m1_x_(m1_x), n_m1_(n_m1),
        m2_i_(m2_i), m2_j_(m2_j), m2_x_(m2_x), n_m2_(n_m2), ar1_(ar1),
        rw_(rw), separate_range_(separate_range), nb2_(nb2),
        gaussian_(gaussian) {}

  void initialize(quadra::ModelReportContext &ctx) const { ctx.clear(); }

  template <typename Type>
  Type evaluate(const std::vector<Type> &parameters,
                quadra::ModelReportContext &) const {
    const Type tau_s = exp(parameters[static_cast<size_t>(p_)]);
    const Type tau_st = exp(parameters[static_cast<size_t>(p_ + 1)]);
    const Type kappa_s2 =
        exp(Type(2.0) * parameters[static_cast<size_t>(p_ + 2)]);
    const Type kappa_st2 =
        separate_range_
            ? exp(Type(2.0) * parameters[static_cast<size_t>(p_ + 3)])
            : kappa_s2;
    const Type kappa_s4 = kappa_s2 * kappa_s2;
    const Type kappa_st4 = kappa_st2 * kappa_st2;
    const int temporal_parameter_base =
        p_ + 3 + static_cast<int>(separate_range_);
    const int random_base =
        temporal_parameter_base + static_cast<int>(ar1_) +
        static_cast<int>(nb2_ || gaussian_);
    const Type likelihood_parameter = (nb2_ || gaussian_)
        ? exp(parameters[static_cast<size_t>(random_base - 1)])
        : Type(1.0);
    std::vector<Type> projected(static_cast<size_t>(n_), Type(0.0));
    for (int k = 0; k < n_a_; ++k) {
      const int observation = a_i_[k];
      const int node = a_j_[k];
      const int st_index =
          random_base + n_node_ * (1 + time_[observation]) + node;
      projected[static_cast<size_t>(observation)] =
          projected[static_cast<size_t>(observation)] +
          Type(a_x_[k]) *
              (parameters[static_cast<size_t>(random_base + node)] +
               parameters[static_cast<size_t>(st_index)]);
    }
    Type nll = Type(0.0);
    for (int i = 0; i < n_; ++i) {
      Type eta = Type(offset_[i]) + projected[static_cast<size_t>(i)];
      for (int j = 0; j < p_; ++j)
        eta = eta + parameters[static_cast<size_t>(j)] * x_[i + n_ * j];
      nll = nll + observation_nll(
          eta, y_[i], weights_[i], nb2_, gaussian_, likelihood_parameter);
    }
    Type spatial_quadratic = Type(0.0);
    Type st_quadratic = Type(0.0);
    add_all_quadratics(parameters, random_base, kappa_s4, kappa_s2,
                       spatial_quadratic);
    Type ar1_log_normalizer = Type(0.0);
    if (ar1_) {
      const Type rho =
          Type(2.0) /
              (Type(1.0) +
               exp(-parameters[static_cast<size_t>(
                   temporal_parameter_base)])) -
          Type(1.0);
      const Type one_minus_rho2 = Type(1.0) - rho * rho;
      add_all_quadratics(parameters, random_base + n_node_, kappa_st4,
                         kappa_st2, st_quadratic);
      for (int t = 1; t < n_time_; ++t)
        add_all_difference_quadratics(
            parameters, random_base + n_node_ * (t + 1),
            random_base + n_node_ * t, rho, kappa_st4, kappa_st2,
            st_quadratic);
      st_quadratic = st_quadratic / one_minus_rho2;
      // The first slice is marginal rather than conditional.
      Type first = Type(0.0);
      add_all_quadratics(parameters, random_base + n_node_, kappa_st4,
                         kappa_st2, first);
      st_quadratic = st_quadratic - first / one_minus_rho2 + first;
      ar1_log_normalizer =
          Type(0.5 * n_node_ * (n_time_ - 1)) * log(one_minus_rho2);
    } else if (rw_) {
      add_all_quadratics(parameters, random_base + n_node_, kappa_st4,
                         kappa_st2, st_quadratic);
      for (int t = 1; t < n_time_; ++t)
        add_all_difference_quadratics(
            parameters, random_base + n_node_ * (t + 1),
            random_base + n_node_ * t, Type(1.0), kappa_st4, kappa_st2,
            st_quadratic);
    } else {
      for (int t = 0; t < n_time_; ++t)
        add_all_quadratics(parameters, random_base + n_node_ * (t + 1),
                           kappa_st4, kappa_st2, st_quadratic);
    }

    std::vector<Eigen::Triplet<double>> entries;
    entries.reserve(static_cast<size_t>(n_m0_ + n_m1_ + n_m2_));
    add_triplets(entries, m0_i_, m0_j_, m0_x_, n_m0_,
                 quadra::value_of_arithmetic_or_ad(kappa_s4));
    add_triplets(entries, m1_i_, m1_j_, m1_x_, n_m1_,
                 2.0 * quadra::value_of_arithmetic_or_ad(kappa_s2));
    add_triplets(entries, m2_i_, m2_j_, m2_x_, n_m2_, 1.0);
    Eigen::SparseMatrix<double> q_s(n_node_, n_node_);
    q_s.setFromTriplets(entries.begin(), entries.end());
    entries.clear();
    add_triplets(entries, m0_i_, m0_j_, m0_x_, n_m0_,
                 quadra::value_of_arithmetic_or_ad(kappa_st4));
    add_triplets(entries, m1_i_, m1_j_, m1_x_, n_m1_,
                 2.0 * quadra::value_of_arithmetic_or_ad(kappa_st2));
    add_triplets(entries, m2_i_, m2_j_, m2_x_, n_m2_, 1.0);
    Eigen::SparseMatrix<double> q_st(n_node_, n_node_);
    q_st.setFromTriplets(entries.begin(), entries.end());
    bool ok = false;
    const double logdet_q_s = quadra::sparse_ldlt_logdet(q_s, ok);
    if (!ok) throw std::runtime_error("SPDE precision is not positive definite");
    const double logdet_q_st = quadra::sparse_ldlt_logdet(q_st, ok);
    if (!ok) throw std::runtime_error("SPDE precision is not positive definite");
    constexpr double log_sqrt_2pi = 0.91893853320467274178;
    nll = nll + Type(0.5) * tau_s * tau_s * spatial_quadratic -
          Type(n_node_) * log(tau_s);
    nll = nll + Type(0.5) * tau_st * tau_st * st_quadratic -
          Type(n_node_ * n_time_) * log(tau_st) + ar1_log_normalizer;
    nll = nll - Type(0.5 * logdet_q_s) -
          Type(0.5 * n_time_ * logdet_q_st) +
          Type(n_node_ * (n_time_ + 1) * log_sqrt_2pi);
    return nll;
  }

private:
  template <typename Type>
  void add_all_quadratics(const std::vector<Type> &parameters, int base,
                          const Type &kappa4, const Type &kappa2,
                          Type &out) const {
    add_quadratic(parameters, base, m0_i_, m0_j_, m0_x_, n_m0_, kappa4,
                  out);
    add_quadratic(parameters, base, m1_i_, m1_j_, m1_x_, n_m1_,
                  Type(2.0) * kappa2, out);
    add_quadratic(parameters, base, m2_i_, m2_j_, m2_x_, n_m2_, Type(1.0),
                  out);
  }
  template <typename Type>
  void add_all_difference_quadratics(
      const std::vector<Type> &parameters, int current_base, int previous_base,
      const Type &rho, const Type &kappa4, const Type &kappa2,
      Type &out) const {
    add_difference_quadratic(parameters, current_base, previous_base, m0_i_,
                             m0_j_, m0_x_, n_m0_, rho, kappa4, out);
    add_difference_quadratic(parameters, current_base, previous_base, m1_i_,
                             m1_j_, m1_x_, n_m1_, rho,
                             Type(2.0) * kappa2, out);
    add_difference_quadratic(parameters, current_base, previous_base, m2_i_,
                             m2_j_, m2_x_, n_m2_, rho, Type(1.0), out);
  }
  template <typename Type>
  static void add_quadratic(const std::vector<Type> &parameters, int base,
                            const int *row, const int *col,
                            const double *value, int count, const Type &scale,
                            Type &out) {
    for (int k = 0; k < count; ++k)
      out = out + Type(value[k]) * scale *
                      parameters[static_cast<size_t>(base + row[k])] *
                      parameters[static_cast<size_t>(base + col[k])];
  }
  template <typename Type>
  static void add_difference_quadratic(
      const std::vector<Type> &parameters, int current_base, int previous_base,
      const int *row, const int *col, const double *value, int count,
      const Type &rho, const Type &scale, Type &out) {
    for (int k = 0; k < count; ++k) {
      const Type left =
          parameters[static_cast<size_t>(current_base + row[k])] -
          rho * parameters[static_cast<size_t>(previous_base + row[k])];
      const Type right =
          parameters[static_cast<size_t>(current_base + col[k])] -
          rho * parameters[static_cast<size_t>(previous_base + col[k])];
      out = out + Type(value[k]) * scale * left * right;
    }
  }
  static void add_triplets(std::vector<Eigen::Triplet<double>> &out,
                           const int *row, const int *col,
                           const double *value, int count, double scale) {
    for (int k = 0; k < count; ++k)
      out.emplace_back(row[k], col[k], scale * value[k]);
  }

  const double *x_, *y_, *offset_, *weights_;
  const int *time_;
  int n_, p_, n_node_, n_time_;
  const int *a_i_, *a_j_;
  const double *a_x_;
  int n_a_;
  const int *m0_i_, *m0_j_;
  const double *m0_x_;
  int n_m0_;
  const int *m1_i_, *m1_j_;
  const double *m1_x_;
  int n_m1_;
  const int *m2_i_, *m2_j_;
  const double *m2_x_;
  int n_m2_;
  bool ar1_;
  bool rw_;
  bool separate_range_;
  bool nb2_;
  bool gaussian_;
};

struct PersistentPoissonResult {
  quadra::LaplaceObjectiveResult objective;
  std::vector<double> gradient;
};

struct BridgeDerivedInference {
  std::vector<std::string> names;
  quadra::ADDeltaMethodVectorResult result;
};

struct BridgeRandomEffectInference {
  quadra::RandomEffectMarginalResult result;
};

class PersistentPoissonSpdeState {
public:
  PersistentPoissonSpdeState(
      const std::vector<double> &fixed, const std::vector<double> &random,
      const double *x, const double *y, const double *offset,
      const double *weights, int n, int p,
      int n_node, const int *a_i, const int *a_j, const double *a_x, int n_a,
      const int *m0_i, const int *m0_j, const double *m0_x, int n_m0,
      const int *m1_i, const int *m1_j, const double *m1_x, int n_m1,
      const int *m2_i, const int *m2_j, const double *m2_x, int n_m2,
      bool nb2 = false, bool gaussian = false)
      : x_(x, x + static_cast<size_t>(n * p)), y_(y, y + n),
        offset_(offset, offset + n), weights_(weights, weights + n),
        a_i_(a_i, a_i + n_a),
        a_j_(a_j, a_j + n_a), a_x_(a_x, a_x + n_a),
        m0_i_(m0_i, m0_i + n_m0), m0_j_(m0_j, m0_j + n_m0),
        m0_x_(m0_x, m0_x + n_m0), m1_i_(m1_i, m1_i + n_m1),
        m1_j_(m1_j, m1_j + n_m1), m1_x_(m1_x, m1_x + n_m1),
        m2_i_(m2_i, m2_i + n_m2), m2_j_(m2_j, m2_j + n_m2),
        m2_x_(m2_x, m2_x + n_m2), n_(n), p_(p), n_node_(n_node),
        random_(random), nb2_(nb2), gaussian_(gaussian) {
    model_.reset(new PoissonSpdeFieldModel(
        x_.data(), y_.data(), offset_.data(), weights_.data(), n_, p_,
        n_node_, a_i_.data(),
        a_j_.data(), a_x_.data(), static_cast<int>(a_x_.size()),
        m0_i_.data(), m0_j_.data(), m0_x_.data(),
        static_cast<int>(m0_x_.size()), m1_i_.data(), m1_j_.data(),
        m1_x_.data(), static_cast<int>(m1_x_.size()), m2_i_.data(),
        m2_j_.data(), m2_x_.data(), static_cast<int>(m2_x_.size()), nb2_,
        gaussian_));
    M0_cache_ = matrix(m0_i_, m0_j_, m0_x_);
    M1_cache_ = matrix(m1_i_, m1_j_, m1_x_);
    M2_cache_ = matrix(m2_i_, m2_j_, m2_x_);
    const int n_fixed =
        p_ + 2 + static_cast<int>(nb2_ || gaussian_);
    for (int j = 0; j < n_fixed; ++j)
      partition_.fixed_indices_m.push_back(static_cast<size_t>(j));
    for (int s = 0; s < n_node_; ++s)
      partition_.random_indices_m.push_back(
          static_cast<size_t>(n_fixed + s));
    anchor_logdet_q_ = precision_logdet(fixed);
    workspace_.reset(new quadra::RandomEffectHessianWorkspace<
                     PoissonSpdeFieldModel>(
        *model_, fixed, random_, partition_));
  }

  PersistentPoissonResult Evaluate(const std::vector<double> &fixed,
                                   bool gradient_requested) {
    if (has_cached_center_ && fixed == cached_fixed_) {
      PersistentPoissonResult cached;
      cached.objective = cached_center_;
      cached.gradient.assign(
          static_cast<size_t>(
              p_ + 2 + static_cast<int>(nb2_ || gaussian_)),
          NA_REAL);
      if (gradient_requested && cached_center_.logdet_ok_m)
        cached.gradient = exact_gradient(fixed, cached_center_);
      return cached;
    }
    quadra::RandomEffectNewtonOptions newton_options;
    newton_options.max_iterations_m = 100;
    newton_options.gradient_tolerance_m = 1e-6;
    newton_options.step_tolerance_m = 1e-12;
    quadra::RandomEffectNewtonResult newton =
        quadra::optimize_random_effects_newton(
            *model_, *workspace_, fixed, random_, partition_, newton_options);
    if (!newton.converged_m) {
      const std::vector<double> cold_start(static_cast<size_t>(n_node_), 0.0);
      quadra::RandomEffectNewtonResult retry =
          quadra::optimize_random_effects_newton(
              *model_, *workspace_, fixed, cold_start, partition_,
              newton_options);
      if (retry.converged_m ||
          retry.gradient_norm_m < newton.gradient_norm_m)
        newton = std::move(retry);
    }
    if (newton.converged_m) random_ = newton.u_hat_m;

    quadra::LaplaceObjectiveResult center;
    center.fixed_m = fixed;
    center.u_hat_m = newton.u_hat_m;
    center.full_m = newton.full_m;
    center.joint_objective_m = newton.objective_value_m;
    center.gradient_norm_random_m = newton.gradient_norm_m;
    center.random_step_norm_m = newton.step_norm_m;
    center.newton_iterations_m = newton.iterations_m;
    center.converged_m = newton.converged_m;
    center.message_m = newton.message_m;
    center.hessian_random_m = newton.hessian_random_m;
    center.gradient_random_m = newton.gradient_random_m;
    center.reports_m = newton.reports_m;
    center.n_random_m = n_node_;
    bool logdet_ok = false;
    try {
      center.log_det_hessian_m =
          workspace_->factorize_terminal_hessian(center.hessian_random_m);
      logdet_ok = true;
    } catch (const std::exception &) {
      center.log_det_hessian_m =
          quadra::sparse_ldlt_logdet(center.hessian_random_m, logdet_ok);
    }
    center.logdet_ok_m = logdet_ok;
    const double current_logdet_q = precision_logdet(fixed);
    center.joint_objective_m -=
        0.5 * (current_logdet_q - anchor_logdet_q_);
    center.laplace_objective_m =
        center.joint_objective_m + 0.5 * center.log_det_hessian_m -
        0.5 * static_cast<double>(n_node_) * std::log(2.0 * M_PI);
    cached_fixed_ = fixed;
    cached_center_ = center;
    has_cached_center_ = true;

    PersistentPoissonResult result;
    result.objective = center;
    result.gradient.assign(
        static_cast<size_t>(
            p_ + 2 + static_cast<int>(nb2_ || gaussian_)), NA_REAL);
    if (gradient_requested && logdet_ok)
      result.gradient = exact_gradient(fixed, center);
    return result;
  }

  quadra::FixedEffectCovarianceResult
  FixedCovariance(const std::vector<double> &fixed) {
    auto gradient = [this](const std::vector<double> &theta) {
      return Evaluate(theta, true).gradient;
    };
    auto result =
        quadra::estimate_fixed_effect_covariance_from_gradient(gradient, fixed);
    Evaluate(fixed, false);
    return result;
  }

  BridgeDerivedInference DerivedInference(
      const std::vector<double> &fixed,
      const quadra::FixedEffectCovarianceResult &covariance) const {
    BridgeDerivedInference out;
    out.names = {"spatial_precision_scale", "spatial_range"};
    if (gaussian_) out.names.push_back("sigma");
    if (nb2_) out.names.push_back("phi");
    auto transform = [this](const std::vector<quadra::AD> &theta) {
      std::vector<quadra::AD> value;
      value.push_back(exp(theta[static_cast<size_t>(p_)]));
      value.push_back(quadra::AD(std::sqrt(8.0)) *
                      exp(-theta[static_cast<size_t>(p_ + 1)]));
      if (gaussian_ || nb2_)
        value.push_back(exp(theta[static_cast<size_t>(p_ + 2)]));
      return value;
    };
    if (covariance.success_m)
      out.result = quadra::ad_delta_method_vector(
          transform, fixed, covariance.covariance_m);
    else
      out.result.message_m = covariance.message_m;
    return out;
  }

  BridgeRandomEffectInference RandomEffectInference(
      const std::vector<double> &fixed,
      const quadra::FixedEffectCovarianceResult &covariance) {
    BridgeRandomEffectInference out;
    if (!covariance.success_m) {
      out.result.message_m = covariance.message_m;
      return out;
    }
    if (!has_cached_center_ || cached_fixed_ != fixed) Evaluate(fixed, false);
    auto implicit = quadra::LaplaceImplicitWorkspace();
    implicit.fixed_m = fixed;
    implicit.u_hat_m = cached_center_.u_hat_m;
    implicit.full_m = cached_center_.full_m;
    implicit.H_uu_m = cached_center_.hessian_random_m;
    implicit.H_u_theta_m = quadra::ad_mixed_hessian(
        *model_, fixed, implicit.u_hat_m, partition_);
    implicit.du_dtheta_m =
        -workspace_->terminal_solve(implicit.H_u_theta_m);
    implicit.converged_m = cached_center_.converged_m;
    implicit.success_m = true;
    implicit.message_m =
        "Implicit derivatives reused the terminal factorization.";
    had::g_ADGraph = nullptr;
    implicit_cache_.reset(
        new quadra::LaplaceImplicitWorkspace(std::move(implicit)));
    covariance_cache_ = covariance.covariance_m;
    out.result = quadra::random_effect_marginal_diagonal(
        workspace_->terminal_selected_inverse(),
        implicit_cache_->du_dtheta_m,
        covariance_cache_);
    return out;
  }

  quadra::LinearPredictorMarginalResult PredictionInference(
      const Eigen::MatrixXd &X_theta,
      const Eigen::SparseMatrix<double, Eigen::RowMajor> &Z) const {
    if (!implicit_cache_)
      throw std::runtime_error(
          "Prediction uncertainty requires covariance inference first.");
    return quadra::linear_predictor_marginal_diagonal(
        workspace_->terminal_factorization(),
        workspace_->terminal_selected_inverse(),
        implicit_cache_->du_dtheta_m,
        covariance_cache_, X_theta, Z);
  }

private:
  Eigen::SparseMatrix<double>
  matrix(const std::vector<int> &row, const std::vector<int> &col,
         const std::vector<double> &value) const {
    std::vector<Eigen::Triplet<double>> entries;
    entries.reserve(value.size());
    for (size_t k = 0; k < value.size(); ++k)
      entries.emplace_back(row[k], col[k], value[k]);
    Eigen::SparseMatrix<double> out(n_node_, n_node_);
    out.setFromTriplets(entries.begin(), entries.end());
    return out;
  }

  Eigen::SparseMatrix<double> precision(const std::vector<double> &fixed) const {
    const double kappa2 =
        std::exp(2.0 * fixed[static_cast<size_t>(p_ + 1)]);
    return (kappa2 * kappa2) * M0_cache_ +
           (2.0 * kappa2) * M1_cache_ + M2_cache_;
  }

  double precision_logdet(const std::vector<double> &fixed) const {
    bool ok = false;
    const double value = quadra::sparse_ldlt_logdet(precision(fixed), ok);
    if (!ok) throw std::runtime_error("SPDE precision is not positive definite");
    return value;
  }

  std::vector<double>
  exact_gradient(const std::vector<double> &fixed,
                 const quadra::LaplaceObjectiveResult &center) const {
    std::vector<Eigen::Triplet<double>> a_entries;
    a_entries.reserve(a_x_.size());
    std::vector<std::vector<std::pair<int, double>>> a_rows(
        static_cast<size_t>(n_));
    for (size_t k = 0; k < a_x_.size(); ++k) {
      a_entries.emplace_back(a_i_[k], a_j_[k], a_x_[k]);
      a_rows[static_cast<size_t>(a_i_[k])].emplace_back(a_j_[k], a_x_[k]);
    }
    Eigen::SparseMatrix<double> A(n_, n_node_);
    A.setFromTriplets(a_entries.begin(), a_entries.end());
    const double tau = std::exp(fixed[static_cast<size_t>(p_)]);
    const double tau2 = tau * tau;
    const double kappa2 =
        std::exp(2.0 * fixed[static_cast<size_t>(p_ + 1)]);
    const double kappa4 = kappa2 * kappa2;
    const Eigen::SparseMatrix<double> &M0 = M0_cache_;
    const Eigen::SparseMatrix<double> &M1 = M1_cache_;
    const Eigen::SparseMatrix<double> &M2 = M2_cache_;
    const Eigen::SparseMatrix<double> Q =
        kappa4 * M0 + (2.0 * kappa2) * M1 + M2;
    const Eigen::SparseMatrix<double> dQ =
        (4.0 * kappa4) * M0 + (4.0 * kappa2) * M1;
    quadra::laplace::SparseHuuFactorization q_factor(Q);
    const auto h_selected = workspace_->terminal_selected_inverse();
    const auto q_selected = q_factor.selected_inverse();
    Eigen::VectorXd leverage(n_);
    Eigen::VectorXd rhs = Eigen::VectorXd::Zero(n_node_);
    for (int i = 0; i < n_; ++i) {
      leverage[i] = 0.0;
      bool selected_supported = true;
      for (const auto &left : a_rows[static_cast<size_t>(i)])
        for (const auto &right : a_rows[static_cast<size_t>(i)])
          selected_supported =
              selected_supported &&
              h_selected.contains(left.first, right.first);
      if (selected_supported) {
        for (const auto &left : a_rows[static_cast<size_t>(i)])
          for (const auto &right : a_rows[static_cast<size_t>(i)])
            leverage[i] += left.second * right.second *
                           h_selected.value(left.first, right.first);
      } else {
        for (const auto &entry : a_rows[static_cast<size_t>(i)])
          rhs[entry.first] += entry.second;
        leverage[i] = rhs.dot(workspace_->terminal_solve(rhs));
        for (const auto &entry : a_rows[static_cast<size_t>(i)])
          rhs[entry.first] = 0.0;
      }
    }
    std::vector<double> h_traces;
    if (h_selected.supports(Q) && h_selected.supports(dQ)) {
      h_traces = h_selected.traces_inverse_times({Q, dQ});
    } else {
      quadra::laplace::SparseHuuFactorization fallback(
          center.hessian_random_m);
      h_traces = fallback.traces_inverse_times_streaming({Q, dQ});
    }
    const double trace_hq = h_traces[0];
    const double trace_hdq = h_traces[1];
    const double trace_qdq =
        q_selected.supports(dQ)
            ? q_selected.trace_inverse_times(dQ)
            : q_factor.trace_inverse_times_streaming(dQ);
    const Eigen::VectorXd u = Eigen::Map<const Eigen::VectorXd>(
        center.u_hat_m.data(), n_node_);
    const Eigen::VectorXd projected = A * u;
    const double likelihood_parameter = (nb2_ || gaussian_)
        ? std::exp(fixed[static_cast<size_t>(p_ + 2)])
        : 1.0;
    Eigen::VectorXd mu(n_), residual(n_), third(n_), phi_cross(n_),
        phi_hpartial(n_), phi_score(n_);
    for (int i = 0; i < n_; ++i) {
      double eta = offset_[static_cast<size_t>(i)] + projected[i];
      for (int j = 0; j < p_; ++j)
        eta += x_[static_cast<size_t>(i + n_ * j)] *
               fixed[static_cast<size_t>(j)];
      const double mean = std::exp(eta);
      const double weight = weights_[static_cast<size_t>(i)];
      const double response = y_[static_cast<size_t>(i)];
      if (nb2_) {
        const double phi = likelihood_parameter;
        const double denominator = phi + mean;
        residual[i] =
            weight * phi * (mean - response) / denominator;
        mu[i] = weight * phi * mean * (phi + response) /
                (denominator * denominator);
        third[i] = mu[i] * (phi - mean) / denominator;
        phi_cross[i] =
            weight * phi * mean * (mean - response) /
            (denominator * denominator);
        phi_hpartial[i] =
            mu[i] * (1.0 + phi / (phi + response) -
                     2.0 * phi / denominator);
        phi_score[i] = weight * phi *
            (-digamma(response + phi) + digamma(phi) -
             std::log(phi) - 1.0 + std::log(denominator) +
             (response + phi) / denominator);
      } else if (gaussian_) {
        const double sigma2 =
            likelihood_parameter * likelihood_parameter;
        residual[i] = weight * (eta - response) / sigma2;
        mu[i] = weight / sigma2;
        third[i] = 0.0;
        phi_cross[i] = -2.0 * residual[i];
        phi_hpartial[i] = -2.0 * mu[i];
        const double standardized =
            (response - eta) / likelihood_parameter;
        phi_score[i] = weight * (1.0 - standardized * standardized);
      } else {
        mu[i] = weight * mean;
        residual[i] = weight * (mean - response);
        third[i] = mu[i];
        phi_cross[i] = phi_hpartial[i] = phi_score[i] = 0.0;
      }
    }
    const int n_fixed =
        p_ + 2 + static_cast<int>(nb2_ || gaussian_);
    std::vector<double> out(static_cast<size_t>(n_fixed));
    for (int j = 0; j < n_fixed; ++j) {
      Eigen::VectorXd direct = Eigen::VectorXd::Zero(n_);
      Eigen::VectorXd cross = Eigen::VectorXd::Zero(n_node_);
      double joint = 0.0, prior_trace = 0.0;
      if (j < p_) {
        for (int i = 0; i < n_; ++i) {
          direct[i] = x_[static_cast<size_t>(i + n_ * j)];
          joint += direct[i] * residual[i];
        }
        cross = A.transpose() * mu.cwiseProduct(direct);
      } else if (j == p_) {
        joint = tau2 * u.dot(Q * u) - static_cast<double>(n_node_);
        cross = 2.0 * tau2 * (Q * u);
        prior_trace = 2.0 * tau2 * trace_hq;
      } else if (j == p_ + 1) {
        joint = 0.5 * tau2 * u.dot(dQ * u) - 0.5 * trace_qdq;
        cross = tau2 * (dQ * u);
        prior_trace = tau2 * trace_hdq;
      } else {
        joint = phi_score.sum();
        cross = A.transpose() * phi_cross;
      }
      const Eigen::VectorXd deta =
          direct - A * workspace_->terminal_solve(cross);
      Eigen::VectorXd likelihood_hdot = third.cwiseProduct(deta);
      if ((nb2_ || gaussian_) && j == p_ + 2)
        likelihood_hdot += phi_hpartial;
      out[static_cast<size_t>(j)] =
          joint + 0.5 * likelihood_hdot.dot(leverage) +
          0.5 * prior_trace;
    }
    return out;
  }

  std::vector<double> x_, y_, offset_, weights_;
  std::vector<int> a_i_, a_j_;
  std::vector<double> a_x_;
  std::vector<int> m0_i_, m0_j_;
  std::vector<double> m0_x_;
  std::vector<int> m1_i_, m1_j_;
  std::vector<double> m1_x_;
  std::vector<int> m2_i_, m2_j_;
  std::vector<double> m2_x_;
  int n_, p_, n_node_;
  std::vector<double> random_;
  double anchor_logdet_q_;
  bool nb2_;
  bool gaussian_;
  quadra::ParameterPartition partition_;
  std::unique_ptr<PoissonSpdeFieldModel> model_;
  std::unique_ptr<
      quadra::RandomEffectHessianWorkspace<PoissonSpdeFieldModel>>
      workspace_;
  std::unique_ptr<quadra::LaplaceImplicitWorkspace> implicit_cache_;
  Eigen::MatrixXd covariance_cache_;
  Eigen::SparseMatrix<double> M0_cache_, M1_cache_, M2_cache_;
  std::vector<double> cached_fixed_;
  quadra::LaplaceObjectiveResult cached_center_;
  bool has_cached_center_ = false;
};

class PersistentPoissonSpatiotemporalIidState {
public:
  PersistentPoissonSpatiotemporalIidState(
      const std::vector<double> &fixed, const std::vector<double> &random,
      const double *x, const double *y, const double *offset,
      const double *weights, const int *time, int n, int p, int n_node,
      int n_time, const int *a_i, const int *a_j,
      const double *a_x, int n_a, const int *m0_i, const int *m0_j,
      const double *m0_x, int n_m0, const int *m1_i, const int *m1_j,
      const double *m1_x, int n_m1, const int *m2_i, const int *m2_j,
      const double *m2_x, int n_m2, bool ar1 = false, bool rw = false,
      bool separate_range = false, bool nb2 = false, bool gaussian = false)
      : x_(x, x + static_cast<size_t>(n * p)), y_(y, y + n),
        offset_(offset, offset + n), weights_(weights, weights + n),
        time_(time, time + n),
        a_i_(a_i, a_i + n_a), a_j_(a_j, a_j + n_a),
        a_x_(a_x, a_x + n_a), m0_i_(m0_i, m0_i + n_m0),
        m0_j_(m0_j, m0_j + n_m0), m0_x_(m0_x, m0_x + n_m0),
        m1_i_(m1_i, m1_i + n_m1), m1_j_(m1_j, m1_j + n_m1),
        m1_x_(m1_x, m1_x + n_m1), m2_i_(m2_i, m2_i + n_m2),
        m2_j_(m2_j, m2_j + n_m2), m2_x_(m2_x, m2_x + n_m2),
        n_(n), p_(p), n_node_(n_node), n_time_(n_time), random_(random),
        ar1_(ar1), rw_(rw), separate_range_(separate_range), nb2_(nb2),
        gaussian_(gaussian) {
    model_.reset(new PoissonSpatiotemporalIidModel(
        x_.data(), y_.data(), offset_.data(), weights_.data(), time_.data(),
        n_, p_, n_node_, n_time_, a_i_.data(), a_j_.data(), a_x_.data(),
        static_cast<int>(a_x_.size()), m0_i_.data(), m0_j_.data(),
        m0_x_.data(), static_cast<int>(m0_x_.size()), m1_i_.data(),
        m1_j_.data(), m1_x_.data(), static_cast<int>(m1_x_.size()),
        m2_i_.data(), m2_j_.data(), m2_x_.data(),
        static_cast<int>(m2_x_.size()), ar1_, rw_, separate_range_, nb2_,
        gaussian_));
    M0_cache_ = matrix(m0_i_, m0_j_, m0_x_);
    M1_cache_ = matrix(m1_i_, m1_j_, m1_x_);
    M2_cache_ = matrix(m2_i_, m2_j_, m2_x_);
    const int n_fixed =
        p_ + 3 + static_cast<int>(separate_range_) +
        static_cast<int>(ar1_) + static_cast<int>(nb2_ || gaussian_);
    for (int j = 0; j < n_fixed; ++j)
      partition_.fixed_indices_m.push_back(static_cast<size_t>(j));
    for (size_t j = 0; j < random_.size(); ++j)
      partition_.random_indices_m.push_back(
          static_cast<size_t>(n_fixed) + j);
    anchor_logdet_q_s_ = precision_logdet(fixed, false);
    anchor_logdet_q_st_ = precision_logdet(fixed, true);
    workspace_.reset(new quadra::RandomEffectHessianWorkspace<
                     PoissonSpatiotemporalIidModel>(
        *model_, fixed, random_, partition_));
  }

  PersistentPoissonResult Evaluate(const std::vector<double> &fixed,
                                   bool gradient_requested) {
    if (has_cached_center_ && fixed == cached_fixed_) {
      PersistentPoissonResult cached;
      cached.objective = cached_center_;
      cached.gradient.assign(
          static_cast<size_t>(
              p_ + 3 + static_cast<int>(separate_range_) +
              static_cast<int>(ar1_) +
              static_cast<int>(nb2_ || gaussian_)),
          NA_REAL);
      if (gradient_requested && cached_center_.logdet_ok_m)
        cached.gradient = exact_gradient(fixed, cached_center_);
      return cached;
    }
    quadra::RandomEffectNewtonOptions options;
    options.max_iterations_m = 100;
    options.gradient_tolerance_m = 1e-6;
    options.step_tolerance_m = 1e-12;
    quadra::RandomEffectNewtonResult newton =
        quadra::optimize_random_effects_newton(
            *model_, *workspace_, fixed, random_, partition_, options);
    if (!newton.converged_m) {
      const std::vector<double> cold(random_.size(), 0.0);
      quadra::RandomEffectNewtonResult retry =
          quadra::optimize_random_effects_newton(
              *model_, *workspace_, fixed, cold, partition_, options);
      if (retry.converged_m ||
          retry.gradient_norm_m < newton.gradient_norm_m)
        newton = std::move(retry);
    }
    if (newton.converged_m) random_ = newton.u_hat_m;

    quadra::LaplaceObjectiveResult center;
    center.fixed_m = fixed;
    center.u_hat_m = newton.u_hat_m;
    center.full_m = newton.full_m;
    center.joint_objective_m = newton.objective_value_m;
    center.gradient_norm_random_m = newton.gradient_norm_m;
    center.random_step_norm_m = newton.step_norm_m;
    center.newton_iterations_m = newton.iterations_m;
    center.converged_m = newton.converged_m;
    center.message_m = newton.message_m;
    center.hessian_random_m = newton.hessian_random_m;
    center.gradient_random_m = newton.gradient_random_m;
    center.reports_m = newton.reports_m;
    center.n_random_m = static_cast<int>(random_.size());
    bool logdet_ok = false;
    try {
      center.log_det_hessian_m =
          workspace_->factorize_terminal_hessian(center.hessian_random_m);
      logdet_ok = true;
    } catch (const std::exception &) {
      center.log_det_hessian_m =
          quadra::sparse_ldlt_logdet(center.hessian_random_m, logdet_ok);
    }
    center.logdet_ok_m = logdet_ok;
    center.joint_objective_m -=
        0.5 * (precision_logdet(fixed, false) - anchor_logdet_q_s_) +
        0.5 * static_cast<double>(n_time_) *
            (precision_logdet(fixed, true) - anchor_logdet_q_st_);
    center.laplace_objective_m =
        center.joint_objective_m + 0.5 * center.log_det_hessian_m -
        0.5 * static_cast<double>(random_.size()) * std::log(2.0 * M_PI);
    cached_fixed_ = fixed;
    cached_center_ = center;
    has_cached_center_ = true;

    PersistentPoissonResult result;
    result.objective = center;
    result.gradient.assign(
        static_cast<size_t>(
            p_ + 3 + static_cast<int>(separate_range_) +
            static_cast<int>(ar1_)),
        NA_REAL);
    if (gradient_requested && logdet_ok)
      result.gradient = exact_gradient(fixed, center);
    return result;
  }

  quadra::FixedEffectCovarianceResult
  FixedCovariance(const std::vector<double> &fixed) {
    auto gradient = [this](const std::vector<double> &theta) {
      return Evaluate(theta, true).gradient;
    };
    auto result =
        quadra::estimate_fixed_effect_covariance_from_gradient(gradient, fixed);
    Evaluate(fixed, false);
    return result;
  }

  BridgeDerivedInference DerivedInference(
      const std::vector<double> &fixed,
      const quadra::FixedEffectCovarianceResult &covariance) const {
    BridgeDerivedInference out;
    out.names = {"spatial_precision_scale",
                 "spatiotemporal_precision_scale", "spatial_range"};
    if (separate_range_)
      out.names.push_back("spatiotemporal_range");
    if (ar1_) out.names.push_back("rho");
    if (gaussian_) out.names.push_back("sigma");
    if (nb2_) out.names.push_back("phi");
    auto transform = [this](const std::vector<quadra::AD> &theta) {
      std::vector<quadra::AD> value;
      value.push_back(exp(theta[static_cast<size_t>(p_)]));
      value.push_back(exp(theta[static_cast<size_t>(p_ + 1)]));
      value.push_back(quadra::AD(std::sqrt(8.0)) *
                      exp(-theta[static_cast<size_t>(p_ + 2)]));
      int next = p_ + 3;
      if (separate_range_) {
        value.push_back(quadra::AD(std::sqrt(8.0)) *
                        exp(-theta[static_cast<size_t>(next)]));
        ++next;
      }
      if (ar1_) {
        value.push_back(quadra::AD(2.0) /
                            (quadra::AD(1.0) +
                             exp(-theta[static_cast<size_t>(next)])) -
                        quadra::AD(1.0));
        ++next;
      }
      if (gaussian_ || nb2_)
        value.push_back(exp(theta[static_cast<size_t>(next)]));
      return value;
    };
    if (covariance.success_m)
      out.result = quadra::ad_delta_method_vector(
          transform, fixed, covariance.covariance_m);
    else
      out.result.message_m = covariance.message_m;
    return out;
  }

  BridgeRandomEffectInference RandomEffectInference(
      const std::vector<double> &fixed,
      const quadra::FixedEffectCovarianceResult &covariance) {
    BridgeRandomEffectInference out;
    if (!covariance.success_m) {
      out.result.message_m = covariance.message_m;
      return out;
    }
    if (!has_cached_center_ || cached_fixed_ != fixed) Evaluate(fixed, false);
    auto implicit = quadra::LaplaceImplicitWorkspace();
    implicit.fixed_m = fixed;
    implicit.u_hat_m = cached_center_.u_hat_m;
    implicit.full_m = cached_center_.full_m;
    implicit.H_uu_m = cached_center_.hessian_random_m;
    implicit.H_u_theta_m = quadra::ad_mixed_hessian(
        *model_, fixed, implicit.u_hat_m, partition_);
    implicit.du_dtheta_m =
        -workspace_->terminal_solve(implicit.H_u_theta_m);
    implicit.converged_m = cached_center_.converged_m;
    implicit.success_m = true;
    implicit.message_m =
        "Implicit derivatives reused the terminal factorization.";
    had::g_ADGraph = nullptr;
    implicit_cache_.reset(
        new quadra::LaplaceImplicitWorkspace(std::move(implicit)));
    covariance_cache_ = covariance.covariance_m;
    out.result = quadra::random_effect_marginal_diagonal(
        workspace_->terminal_selected_inverse(),
        implicit_cache_->du_dtheta_m,
        covariance_cache_);
    return out;
  }

  quadra::LinearPredictorMarginalResult PredictionInference(
      const Eigen::MatrixXd &X_theta,
      const Eigen::SparseMatrix<double, Eigen::RowMajor> &Z) const {
    if (!implicit_cache_)
      throw std::runtime_error(
          "Prediction uncertainty requires covariance inference first.");
    return quadra::linear_predictor_marginal_diagonal(
        workspace_->terminal_factorization(),
        workspace_->terminal_selected_inverse(),
        implicit_cache_->du_dtheta_m,
        covariance_cache_, X_theta, Z);
  }

private:
  Eigen::SparseMatrix<double>
  matrix(const std::vector<int> &row, const std::vector<int> &col,
         const std::vector<double> &value) const {
    std::vector<Eigen::Triplet<double>> entries;
    entries.reserve(value.size());
    for (size_t k = 0; k < value.size(); ++k)
      entries.emplace_back(row[k], col[k], value[k]);
    Eigen::SparseMatrix<double> out(n_node_, n_node_);
    out.setFromTriplets(entries.begin(), entries.end());
    return out;
  }

  Eigen::SparseMatrix<double> precision(const std::vector<double> &fixed,
                                        bool spatiotemporal) const {
    const int kappa_index =
        p_ + 2 +
        static_cast<int>(spatiotemporal && separate_range_);
    const double kappa2 =
        std::exp(2.0 * fixed[static_cast<size_t>(kappa_index)]);
    return (kappa2 * kappa2) * M0_cache_ +
           (2.0 * kappa2) * M1_cache_ + M2_cache_;
  }

  double precision_logdet(const std::vector<double> &fixed,
                          bool spatiotemporal) const {
    bool ok = false;
    const double value =
        quadra::sparse_ldlt_logdet(precision(fixed, spatiotemporal), ok);
    if (!ok) throw std::runtime_error("SPDE precision is not positive definite");
    return value;
  }

  std::vector<double>
  exact_gradient(const std::vector<double> &fixed,
                 const quadra::LaplaceObjectiveResult &center) const {
    const int n_random = n_node_ * (n_time_ + 1);
    std::vector<Eigen::Triplet<double>> z_entries;
    z_entries.reserve(a_x_.size() * 2);
    std::vector<std::vector<std::pair<int, double>>> z_rows(
        static_cast<size_t>(n_));
    for (size_t k = 0; k < a_x_.size(); ++k) {
      const int observation = a_i_[k], node = a_j_[k];
      z_entries.emplace_back(observation, node, a_x_[k]);
      const int st_node =
          n_node_ * (1 + time_[static_cast<size_t>(observation)]) + node;
      z_entries.emplace_back(observation, st_node, a_x_[k]);
      z_rows[static_cast<size_t>(observation)].emplace_back(node, a_x_[k]);
      z_rows[static_cast<size_t>(observation)].emplace_back(st_node, a_x_[k]);
    }
    Eigen::SparseMatrix<double> Z(n_, n_random);
    Z.setFromTriplets(z_entries.begin(), z_entries.end());
    const double tau_s2 =
        std::exp(2.0 * fixed[static_cast<size_t>(p_)]);
    const double tau_st2 =
        std::exp(2.0 * fixed[static_cast<size_t>(p_ + 1)]);
    const double kappa_s2 =
        std::exp(2.0 * fixed[static_cast<size_t>(p_ + 2)]);
    const int kappa_st_index =
        p_ + 2 + static_cast<int>(separate_range_);
    const double kappa_st2 =
        std::exp(2.0 * fixed[static_cast<size_t>(kappa_st_index)]);
    const double kappa_s4 = kappa_s2 * kappa_s2;
    const double kappa_st4 = kappa_st2 * kappa_st2;
    const Eigen::SparseMatrix<double> &M0 = M0_cache_;
    const Eigen::SparseMatrix<double> &M1 = M1_cache_;
    const Eigen::SparseMatrix<double> &M2 = M2_cache_;
    const Eigen::SparseMatrix<double> Q_s =
        kappa_s4 * M0 + (2.0 * kappa_s2) * M1 + M2;
    const Eigen::SparseMatrix<double> dQ_s =
        (4.0 * kappa_s4) * M0 + (4.0 * kappa_s2) * M1;
    const Eigen::SparseMatrix<double> Q_st =
        kappa_st4 * M0 + (2.0 * kappa_st2) * M1 + M2;
    const Eigen::SparseMatrix<double> dQ_st =
        (4.0 * kappa_st4) * M0 + (4.0 * kappa_st2) * M1;
    Eigen::MatrixXd temporal =
        Eigen::MatrixXd::Identity(n_time_, n_time_);
    Eigen::MatrixXd dtemporal =
        Eigen::MatrixXd::Zero(n_time_, n_time_);
    double rho = 0.0;
    if (ar1_ && n_time_ > 1) {
      const double phi = fixed[static_cast<size_t>(
          p_ + 3 + static_cast<int>(separate_range_))];
      rho = 2.0 / (1.0 + std::exp(-phi)) - 1.0;
      const double denominator = 1.0 - rho * rho;
      temporal.setZero();
      for (int t = 0; t < n_time_; ++t) {
        const bool boundary = t == 0 || t == n_time_ - 1;
        temporal(t, t) =
            boundary ? 1.0 / denominator
                     : (1.0 + rho * rho) / denominator;
        dtemporal(t, t) =
            boundary ? rho / denominator : 2.0 * rho / denominator;
      }
      for (int t = 1; t < n_time_; ++t) {
        temporal(t - 1, t) = temporal(t, t - 1) = -rho / denominator;
        dtemporal(t - 1, t) = dtemporal(t, t - 1) =
            -(1.0 + rho * rho) / (2.0 * denominator);
      }
    } else if (rw_ && n_time_ > 1) {
      temporal.setZero();
      for (int t = 0; t < n_time_; ++t)
        temporal(t, t) = t == n_time_ - 1 ? 1.0 : 2.0;
      for (int t = 1; t < n_time_; ++t)
        temporal(t - 1, t) = temporal(t, t - 1) = -1.0;
    }
    const int n_st = n_node_ * n_time_;
    auto temporal_kronecker =
        [this, n_st](const Eigen::MatrixXd &time_matrix,
                     const Eigen::SparseMatrix<double> &space_matrix) {
          std::vector<Eigen::Triplet<double>> entries;
          entries.reserve(static_cast<size_t>(
              time_matrix.rows() * space_matrix.nonZeros() * 3));
          for (int t = 0; t < time_matrix.rows(); ++t) {
            for (int s = 0; s < time_matrix.cols(); ++s) {
              const double coefficient = time_matrix(t, s);
              if (coefficient == 0.0)
                continue;
              for (int outer = 0; outer < space_matrix.outerSize(); ++outer) {
                for (Eigen::SparseMatrix<double>::InnerIterator it(
                         space_matrix, outer);
                     it; ++it)
                  entries.emplace_back(t * n_node_ + it.row(),
                                       s * n_node_ + it.col(),
                                       coefficient * it.value());
              }
            }
          }
          Eigen::SparseMatrix<double> out(n_st, n_st);
          out.setFromTriplets(entries.begin(), entries.end());
          return out;
        };
    const Eigen::SparseMatrix<double> temporal_q =
        temporal_kronecker(temporal, Q_st);
    const Eigen::SparseMatrix<double> temporal_dq =
        temporal_kronecker(temporal, dQ_st);
    const Eigen::SparseMatrix<double> dtemporal_q =
        temporal_kronecker(dtemporal, Q_st);
    auto embed_blocks =
        [n_random, this](const Eigen::SparseMatrix<double> *spatial,
                         double spatial_scale,
                         const Eigen::SparseMatrix<double> *st,
                         double st_scale) {
          std::vector<Eigen::Triplet<double>> entries;
          if (spatial != nullptr) {
            for (int outer = 0; outer < spatial->outerSize(); ++outer)
              for (Eigen::SparseMatrix<double>::InnerIterator it(*spatial,
                                                                  outer);
                   it; ++it)
                entries.emplace_back(it.row(), it.col(),
                                     spatial_scale * it.value());
          }
          if (st != nullptr) {
            for (int outer = 0; outer < st->outerSize(); ++outer)
              for (Eigen::SparseMatrix<double>::InnerIterator it(*st, outer);
                   it; ++it)
                entries.emplace_back(n_node_ + it.row(),
                                     n_node_ + it.col(),
                                     st_scale * it.value());
          }
          Eigen::SparseMatrix<double> out(n_random, n_random);
          out.setFromTriplets(entries.begin(), entries.end());
          return out;
        };
    const Eigen::SparseMatrix<double> tau_s_hdot =
        embed_blocks(&Q_s, 2.0 * tau_s2, nullptr, 0.0);
    const Eigen::SparseMatrix<double> tau_st_hdot =
        embed_blocks(nullptr, 0.0, &temporal_q, 2.0 * tau_st2);
    const Eigen::SparseMatrix<double> kappa_s_hdot =
        embed_blocks(&dQ_s, tau_s2,
                     separate_range_ ? nullptr : &temporal_dq,
                     separate_range_ ? 0.0 : tau_st2);
    const Eigen::SparseMatrix<double> kappa_st_hdot =
        embed_blocks(nullptr, 0.0, &temporal_dq, tau_st2);
    const Eigen::SparseMatrix<double> rho_hdot =
        embed_blocks(nullptr, 0.0, &dtemporal_q, tau_st2);
    quadra::laplace::SparseHuuFactorization q_s_factor(Q_s);
    quadra::laplace::SparseHuuFactorization q_st_factor(Q_st);
    const auto h_selected = workspace_->terminal_selected_inverse();
    const auto q_s_selected = q_s_factor.selected_inverse();
    const auto q_st_selected = q_st_factor.selected_inverse();
    Eigen::VectorXd leverage(n_);
    Eigen::VectorXd leverage_rhs = Eigen::VectorXd::Zero(n_random);
    for (int i = 0; i < n_; ++i) {
      leverage[i] = 0.0;
      bool selected_supported = true;
      for (const auto &left : z_rows[static_cast<size_t>(i)])
        for (const auto &right : z_rows[static_cast<size_t>(i)])
          selected_supported =
              selected_supported &&
              h_selected.contains(left.first, right.first);
      if (selected_supported) {
        for (const auto &left : z_rows[static_cast<size_t>(i)])
          for (const auto &right : z_rows[static_cast<size_t>(i)])
            leverage[i] += left.second * right.second *
                           h_selected.value(left.first, right.first);
      } else {
        for (const auto &entry : z_rows[static_cast<size_t>(i)])
          leverage_rhs[entry.first] += entry.second;
        leverage[i] =
            leverage_rhs.dot(workspace_->terminal_solve(leverage_rhs));
        for (const auto &entry : z_rows[static_cast<size_t>(i)])
          leverage_rhs[entry.first] = 0.0;
      }
    }
    const double trace_qs_dqs =
        q_s_selected.supports(dQ_s)
            ? q_s_selected.trace_inverse_times(dQ_s)
            : q_s_factor.trace_inverse_times_streaming(dQ_s);
    const double trace_qst_dqst =
        q_st_selected.supports(dQ_st)
            ? q_st_selected.trace_inverse_times(dQ_st)
            : q_st_factor.trace_inverse_times_streaming(dQ_st);
    const std::vector<Eigen::SparseMatrix<double>> prior_derivatives = {
        tau_s_hdot, tau_st_hdot, kappa_s_hdot, kappa_st_hdot, rho_hdot};
    bool selected_traces_supported = true;
    for (const auto &derivative : prior_derivatives)
      selected_traces_supported =
          selected_traces_supported && h_selected.supports(derivative);
    std::vector<double> prior_traces;
    if (selected_traces_supported) {
      prior_traces = h_selected.traces_inverse_times(prior_derivatives);
    } else {
      quadra::laplace::SparseHuuFactorization fallback(
          center.hessian_random_m);
      prior_traces =
          fallback.traces_inverse_times_streaming(prior_derivatives);
    }
    const double trace_tau_s = prior_traces[0];
    const double trace_tau_st = prior_traces[1];
    const double trace_kappa_s = prior_traces[2];
    const double trace_kappa_st = prior_traces[3];
    const double trace_rho = ar1_ ? prior_traces[4] : 0.0;
    const Eigen::VectorXd u = Eigen::Map<const Eigen::VectorXd>(
        center.u_hat_m.data(), n_random);
    const Eigen::VectorXd projected = Z * u;
    const int rho_index =
        p_ + 3 + static_cast<int>(separate_range_);
    const int likelihood_index = rho_index + static_cast<int>(ar1_);
    const double likelihood_parameter = (nb2_ || gaussian_)
        ? std::exp(fixed[static_cast<size_t>(likelihood_index)])
        : 1.0;
    Eigen::VectorXd mu(n_), residual(n_), third(n_), phi_cross(n_),
        phi_hpartial(n_), phi_score(n_);
    for (int i = 0; i < n_; ++i) {
      double eta = offset_[static_cast<size_t>(i)] + projected[i];
      for (int j = 0; j < p_; ++j)
        eta += x_[static_cast<size_t>(i + n_ * j)] *
               fixed[static_cast<size_t>(j)];
      const double mean = std::exp(eta);
      const double weight = weights_[static_cast<size_t>(i)];
      const double response = y_[static_cast<size_t>(i)];
      if (nb2_) {
        const double phi = likelihood_parameter;
        const double denominator = phi + mean;
        residual[i] = weight * phi * (mean - response) / denominator;
        mu[i] = weight * phi * mean * (phi + response) /
                (denominator * denominator);
        third[i] = mu[i] * (phi - mean) / denominator;
        phi_cross[i] =
            weight * phi * mean * (mean - response) /
            (denominator * denominator);
        phi_hpartial[i] =
            mu[i] * (1.0 + phi / (phi + response) -
                     2.0 * phi / denominator);
        phi_score[i] = weight * phi *
            (-digamma(response + phi) + digamma(phi) -
             std::log(phi) - 1.0 + std::log(denominator) +
             (response + phi) / denominator);
      } else if (gaussian_) {
        const double sigma2 =
            likelihood_parameter * likelihood_parameter;
        residual[i] = weight * (eta - response) / sigma2;
        mu[i] = weight / sigma2;
        third[i] = 0.0;
        phi_cross[i] = -2.0 * residual[i];
        phi_hpartial[i] = -2.0 * mu[i];
        const double standardized = (response - eta) / likelihood_parameter;
        phi_score[i] = weight * (1.0 - standardized * standardized);
      } else {
        mu[i] = weight * mean;
        residual[i] = weight * (mean - response);
        third[i] = mu[i];
        phi_cross[i] = phi_hpartial[i] = phi_score[i] = 0.0;
      }
    }

    const Eigen::VectorXd st_fields = u.segment(n_node_, n_st);
    std::vector<double> out(
        static_cast<size_t>(
            p_ + 3 + static_cast<int>(separate_range_) +
            static_cast<int>(ar1_) +
            static_cast<int>(nb2_ || gaussian_)));
    for (int j = 0;
         j < p_ + 3 + static_cast<int>(separate_range_) +
                 static_cast<int>(ar1_) +
                 static_cast<int>(nb2_ || gaussian_);
         ++j) {
      Eigen::VectorXd direct = Eigen::VectorXd::Zero(n_);
      Eigen::VectorXd cross = Eigen::VectorXd::Zero(n_random);
      double joint = 0.0, prior_trace = 0.0;
      if (j < p_) {
        for (int i = 0; i < n_; ++i) {
          direct[i] = x_[static_cast<size_t>(i + n_ * j)];
          joint += direct[i] * residual[i];
        }
        cross = Z.transpose() * mu.cwiseProduct(direct);
      } else if (j == p_) {
        const Eigen::VectorXd field = u.segment(0, n_node_);
        joint = tau_s2 * field.dot(Q_s * field) - n_node_;
        cross.segment(0, n_node_) = 2.0 * tau_s2 * (Q_s * field);
        prior_trace = trace_tau_s;
      } else if (j == p_ + 1) {
        joint = tau_st2 * st_fields.dot(temporal_q * st_fields) -
                static_cast<double>(n_node_ * n_time_);
        cross.segment(n_node_, n_st) =
            2.0 * tau_st2 * (temporal_q * st_fields);
        prior_trace = trace_tau_st;
      } else if (j == p_ + 2) {
        const Eigen::VectorXd spatial = u.segment(0, n_node_);
        joint =
            0.5 * tau_s2 * spatial.dot(dQ_s * spatial) -
            0.5 * trace_qs_dqs;
        cross.segment(0, n_node_) = tau_s2 * (dQ_s * spatial);
        prior_trace = trace_kappa_s;
        if (!separate_range_) {
          joint +=
              0.5 * tau_st2 * st_fields.dot(temporal_dq * st_fields) -
              0.5 * static_cast<double>(n_time_) * trace_qst_dqst;
          cross.segment(n_node_, n_st) =
              tau_st2 * (temporal_dq * st_fields);
        }
      } else if (separate_range_ && j == kappa_st_index) {
        joint =
            0.5 * tau_st2 * st_fields.dot(temporal_dq * st_fields) -
            0.5 * static_cast<double>(n_time_) * trace_qst_dqst;
        cross.segment(n_node_, n_st) =
            tau_st2 * (temporal_dq * st_fields);
        prior_trace = trace_kappa_st;
      } else if (ar1_ && j == rho_index) {
        joint =
            0.5 * tau_st2 *
                st_fields.dot(dtemporal_q * st_fields) -
            0.5 * static_cast<double>(n_node_ * (n_time_ - 1)) * rho;
        cross.segment(n_node_, n_st) =
            tau_st2 * (dtemporal_q * st_fields);
        prior_trace = trace_rho;
      } else if ((nb2_ || gaussian_) && j == likelihood_index) {
        joint = phi_score.sum();
        cross = Z.transpose() * phi_cross;
      }
      const Eigen::VectorXd deta =
          direct - Z * workspace_->terminal_solve(cross);
      Eigen::VectorXd likelihood_hdot = third.cwiseProduct(deta);
      if ((nb2_ || gaussian_) && j == likelihood_index)
        likelihood_hdot += phi_hpartial;
      out[static_cast<size_t>(j)] =
          joint + 0.5 * likelihood_hdot.dot(leverage) +
          0.5 * prior_trace;
    }
    return out;
  }

  std::vector<double> x_, y_, offset_, weights_;
  std::vector<int> time_, a_i_, a_j_;
  std::vector<double> a_x_;
  std::vector<int> m0_i_, m0_j_;
  std::vector<double> m0_x_;
  std::vector<int> m1_i_, m1_j_;
  std::vector<double> m1_x_;
  std::vector<int> m2_i_, m2_j_;
  std::vector<double> m2_x_;
  int n_, p_, n_node_, n_time_;
  std::vector<double> random_;
  double anchor_logdet_q_s_, anchor_logdet_q_st_;
  bool ar1_;
  bool rw_;
  bool separate_range_;
  bool nb2_;
  bool gaussian_;
  quadra::ParameterPartition partition_;
  std::unique_ptr<PoissonSpatiotemporalIidModel> model_;
  std::unique_ptr<
      quadra::RandomEffectHessianWorkspace<PoissonSpatiotemporalIidModel>>
      workspace_;
  std::unique_ptr<quadra::LaplaceImplicitWorkspace> implicit_cache_;
  Eigen::MatrixXd covariance_cache_;
  Eigen::SparseMatrix<double> M0_cache_, M1_cache_, M2_cache_;
  std::vector<double> cached_fixed_;
  quadra::LaplaceObjectiveResult cached_center_;
  bool has_cached_center_ = false;
};

void validate_inputs(SEXP beta, SEXP x, SEXP y, SEXP offset) {
  if (TYPEOF(beta) != REALSXP || TYPEOF(x) != REALSXP ||
      TYPEOF(y) != REALSXP || TYPEOF(offset) != REALSXP) {
    throw std::invalid_argument("beta, X, y, and offset must be double");
  }

  SEXP dims = Rf_getAttrib(x, R_DimSymbol);
  if (Rf_length(dims) != 2) {
    throw std::invalid_argument("X must be a matrix");
  }

  const int n = INTEGER(dims)[0];
  const int p = INTEGER(dims)[1];
  if (Rf_xlength(beta) != p || Rf_xlength(y) != n ||
      Rf_xlength(offset) != n) {
    throw std::invalid_argument("non-conformable beta, X, y, or offset");
  }
}

void validate_random_intercept_inputs(SEXP fixed, SEXP random, SEXP x, SEXP y,
                                      SEXP offset, SEXP group) {
  if (TYPEOF(fixed) != REALSXP || TYPEOF(random) != REALSXP ||
      TYPEOF(x) != REALSXP || TYPEOF(y) != REALSXP ||
      TYPEOF(offset) != REALSXP || TYPEOF(group) != INTSXP) {
    throw std::invalid_argument(
        "fixed, random, X, y, and offset must be numeric; group must be integer");
  }

  SEXP dims = Rf_getAttrib(x, R_DimSymbol);
  if (Rf_length(dims) != 2) {
    throw std::invalid_argument("X must be a matrix");
  }

  const int n = INTEGER(dims)[0];
  const int p = INTEGER(dims)[1];
  const int n_group = static_cast<int>(Rf_xlength(random));
  if (Rf_xlength(fixed) != p + 2 || Rf_xlength(y) != n ||
      Rf_xlength(offset) != n || Rf_xlength(group) != n || n_group < 1) {
    throw std::invalid_argument(
        "non-conformable fixed, random, X, y, offset, or group");
  }

  for (int i = 0; i < n; ++i) {
    if (INTEGER(group)[i] < 0 || INTEGER(group)[i] >= n_group) {
      throw std::invalid_argument("group indices must be zero-based and valid");
    }
  }
}

void validate_sparse_field_inputs(SEXP fixed, SEXP random, SEXP x, SEXP y,
                                  SEXP offset, SEXP node, SEXP q_i, SEXP q_j,
                                  SEXP q_x, SEXP logdet_q) {
  if (TYPEOF(fixed) != REALSXP || TYPEOF(random) != REALSXP ||
      TYPEOF(x) != REALSXP || TYPEOF(y) != REALSXP ||
      TYPEOF(offset) != REALSXP || TYPEOF(node) != INTSXP ||
      TYPEOF(q_i) != INTSXP || TYPEOF(q_j) != INTSXP ||
      TYPEOF(q_x) != REALSXP || TYPEOF(logdet_q) != REALSXP) {
    throw std::invalid_argument("invalid sparse-field input types");
  }

  SEXP dims = Rf_getAttrib(x, R_DimSymbol);
  if (Rf_length(dims) != 2) {
    throw std::invalid_argument("X must be a matrix");
  }

  const int n = INTEGER(dims)[0];
  const int p = INTEGER(dims)[1];
  const int n_node = static_cast<int>(Rf_xlength(random));
  const R_xlen_t n_q = Rf_xlength(q_x);
  if (Rf_xlength(fixed) != p + 2 || Rf_xlength(y) != n ||
      Rf_xlength(offset) != n || Rf_xlength(node) != n || n_node < 1 ||
      Rf_xlength(q_i) != n_q || Rf_xlength(q_j) != n_q ||
      Rf_xlength(logdet_q) != 1) {
    throw std::invalid_argument("non-conformable sparse-field inputs");
  }
  if (!std::isfinite(REAL(logdet_q)[0])) {
    throw std::invalid_argument("logdet_q must be finite");
  }

  for (int i = 0; i < n; ++i) {
    if (INTEGER(node)[i] < 0 || INTEGER(node)[i] >= n_node) {
      throw std::invalid_argument("node indices must be zero-based and valid");
    }
  }
  for (R_xlen_t k = 0; k < n_q; ++k) {
    if (INTEGER(q_i)[k] < 0 || INTEGER(q_i)[k] >= n_node ||
        INTEGER(q_j)[k] < 0 || INTEGER(q_j)[k] >= n_node ||
        !std::isfinite(REAL(q_x)[k])) {
      throw std::invalid_argument("invalid sparse precision triplet");
    }
  }
}

void validate_projected_field_inputs(
    SEXP fixed, SEXP random, SEXP x, SEXP y, SEXP offset, SEXP a_i, SEXP a_j,
    SEXP a_x, SEXP q_i, SEXP q_j, SEXP q_x, SEXP logdet_q) {
  if (TYPEOF(fixed) != REALSXP || TYPEOF(random) != REALSXP ||
      TYPEOF(x) != REALSXP || TYPEOF(y) != REALSXP ||
      TYPEOF(offset) != REALSXP || TYPEOF(a_i) != INTSXP ||
      TYPEOF(a_j) != INTSXP || TYPEOF(a_x) != REALSXP ||
      TYPEOF(q_i) != INTSXP || TYPEOF(q_j) != INTSXP ||
      TYPEOF(q_x) != REALSXP || TYPEOF(logdet_q) != REALSXP) {
    throw std::invalid_argument("invalid projected-field input types");
  }

  SEXP dims = Rf_getAttrib(x, R_DimSymbol);
  if (Rf_length(dims) != 2) {
    throw std::invalid_argument("X must be a matrix");
  }
  const int n = INTEGER(dims)[0];
  const int p = INTEGER(dims)[1];
  const int n_node = static_cast<int>(Rf_xlength(random));
  const R_xlen_t n_a = Rf_xlength(a_x);
  const R_xlen_t n_q = Rf_xlength(q_x);
  if (Rf_xlength(fixed) != p + 2 || Rf_xlength(y) != n ||
      Rf_xlength(offset) != n || n_node < 1 || Rf_xlength(a_i) != n_a ||
      Rf_xlength(a_j) != n_a || Rf_xlength(q_i) != n_q ||
      Rf_xlength(q_j) != n_q || Rf_xlength(logdet_q) != 1) {
    throw std::invalid_argument("non-conformable projected-field inputs");
  }
  for (R_xlen_t k = 0; k < n_a; ++k) {
    if (INTEGER(a_i)[k] < 0 || INTEGER(a_i)[k] >= n ||
        INTEGER(a_j)[k] < 0 || INTEGER(a_j)[k] >= n_node ||
        !std::isfinite(REAL(a_x)[k])) {
      throw std::invalid_argument("invalid sparse projection triplet");
    }
  }
  for (R_xlen_t k = 0; k < n_q; ++k) {
    if (INTEGER(q_i)[k] < 0 || INTEGER(q_i)[k] >= n_node ||
        INTEGER(q_j)[k] < 0 || INTEGER(q_j)[k] >= n_node ||
        !std::isfinite(REAL(q_x)[k])) {
      throw std::invalid_argument("invalid sparse precision triplet");
    }
  }
  if (!std::isfinite(REAL(logdet_q)[0])) {
    throw std::invalid_argument("logdet_q must be finite");
  }
}

void validate_spde_field_inputs(
    SEXP fixed, SEXP random, SEXP x, SEXP y, SEXP offset, SEXP a_i, SEXP a_j,
    SEXP a_x, SEXP m0_i, SEXP m0_j, SEXP m0_x, SEXP m1_i, SEXP m1_j,
    SEXP m1_x, SEXP m2_i, SEXP m2_j, SEXP m2_x, int fixed_extra) {
  const SEXP inputs[] = {fixed, random, x,      y,      offset, a_x,
                         m0_x,  m1_x,   m2_x};
  for (SEXP input : inputs) {
    if (TYPEOF(input) != REALSXP) {
      throw std::invalid_argument("invalid SPDE-field numeric input type");
    }
  }
  const SEXP indices[] = {a_i, a_j, m0_i, m0_j, m1_i, m1_j, m2_i, m2_j};
  for (SEXP index : indices) {
    if (TYPEOF(index) != INTSXP) {
      throw std::invalid_argument("invalid SPDE-field index input type");
    }
  }
  SEXP dims = Rf_getAttrib(x, R_DimSymbol);
  if (Rf_length(dims) != 2) {
    throw std::invalid_argument("X must be a matrix");
  }
  const int n = INTEGER(dims)[0];
  const int p = INTEGER(dims)[1];
  const int n_node = static_cast<int>(Rf_xlength(random));
  if (Rf_xlength(fixed) != p + fixed_extra || Rf_xlength(y) != n ||
      Rf_xlength(offset) != n || n_node < 1 ||
      Rf_xlength(a_i) != Rf_xlength(a_x) ||
      Rf_xlength(a_j) != Rf_xlength(a_x)) {
    throw std::invalid_argument("non-conformable SPDE-field inputs");
  }
  for (R_xlen_t k = 0; k < Rf_xlength(a_x); ++k) {
    if (INTEGER(a_i)[k] < 0 || INTEGER(a_i)[k] >= n ||
        INTEGER(a_j)[k] < 0 || INTEGER(a_j)[k] >= n_node ||
        !std::isfinite(REAL(a_x)[k])) {
      throw std::invalid_argument("invalid sparse projection triplet");
    }
  }
  const SEXP matrix_i[] = {m0_i, m1_i, m2_i};
  const SEXP matrix_j[] = {m0_j, m1_j, m2_j};
  const SEXP matrix_x[] = {m0_x, m1_x, m2_x};
  for (int matrix = 0; matrix < 3; ++matrix) {
    const R_xlen_t count = Rf_xlength(matrix_x[matrix]);
    if (Rf_xlength(matrix_i[matrix]) != count ||
        Rf_xlength(matrix_j[matrix]) != count) {
      throw std::invalid_argument("non-conformable SPDE matrix triplet");
    }
    for (R_xlen_t k = 0; k < count; ++k) {
      if (INTEGER(matrix_i[matrix])[k] < 0 ||
          INTEGER(matrix_i[matrix])[k] >= n_node ||
          INTEGER(matrix_j[matrix])[k] < 0 ||
          INTEGER(matrix_j[matrix])[k] >= n_node ||
          !std::isfinite(REAL(matrix_x[matrix])[k])) {
        throw std::invalid_argument("invalid SPDE matrix triplet");
      }
    }
  }
}

} // namespace

SEXP derived_inference_to_sexp(const BridgeDerivedInference &derived) {
  const int n = static_cast<int>(derived.names.size());
  SEXP result = PROTECT(Rf_allocVector(VECSXP, 5));
  SEXP names = PROTECT(Rf_allocVector(STRSXP, 5));
  SEXP quantity_names = PROTECT(Rf_allocVector(STRSXP, n));
  SEXP estimate = PROTECT(Rf_allocVector(REALSXP, n));
  SEXP std_error = PROTECT(Rf_allocVector(REALSXP, n));
  for (int i = 0; i < n; ++i) {
    SET_STRING_ELT(quantity_names, i, Rf_mkChar(derived.names[i].c_str()));
    REAL(estimate)[i] =
        derived.result.estimate_m.size() == n
            ? derived.result.estimate_m[i]
            : NA_REAL;
    REAL(std_error)[i] =
        derived.result.std_error_m.size() == n
            ? derived.result.std_error_m[i]
            : NA_REAL;
  }
  SET_VECTOR_ELT(result, 0, quantity_names);
  SET_VECTOR_ELT(result, 1, estimate);
  SET_VECTOR_ELT(result, 2, std_error);
  SET_VECTOR_ELT(result, 3, Rf_ScalarLogical(derived.result.success_m));
  SET_VECTOR_ELT(result, 4, Rf_mkString(derived.result.message_m.c_str()));
  const char *labels[] = {"name", "estimate", "std_error", "success",
                          "message"};
  for (int i = 0; i < 5; ++i)
    SET_STRING_ELT(names, i, Rf_mkChar(labels[i]));
  Rf_setAttrib(result, R_NamesSymbol, names);
  UNPROTECT(5);
  return result;
}

SEXP prediction_inference_to_sexp(
    const quadra::LinearPredictorMarginalResult &value) {
  const int n = static_cast<int>(value.std_error_m.size());
  SEXP result = PROTECT(Rf_allocVector(VECSXP, 6));
  SEXP names = PROTECT(Rf_allocVector(STRSXP, 6));
  auto vector_to_sexp = [n](const Eigen::VectorXd &x) {
    SEXP out = Rf_allocVector(REALSXP, n);
    for (int i = 0; i < n; ++i)
      REAL(out)[i] = x.size() == n ? x[i] : NA_REAL;
    return out;
  };
  SEXP conditional = PROTECT(vector_to_sexp(value.conditional_variance_m));
  SEXP parameter = PROTECT(vector_to_sexp(value.parameter_variance_m));
  SEXP marginal = PROTECT(vector_to_sexp(value.marginal_variance_m));
  SEXP std_error = PROTECT(vector_to_sexp(value.std_error_m));
  SET_VECTOR_ELT(result, 0, conditional);
  SET_VECTOR_ELT(result, 1, parameter);
  SET_VECTOR_ELT(result, 2, marginal);
  SET_VECTOR_ELT(result, 3, std_error);
  SET_VECTOR_ELT(result, 4, Rf_ScalarLogical(value.success_m));
  SET_VECTOR_ELT(result, 5, Rf_mkString(value.message_m.c_str()));
  const char *labels[] = {"conditional_variance", "parameter_variance",
                          "marginal_variance", "std_error", "success",
                          "message"};
  for (int i = 0; i < 6; ++i)
    SET_STRING_ELT(names, i, Rf_mkChar(labels[i]));
  Rf_setAttrib(result, R_NamesSymbol, names);
  UNPROTECT(6);
  return result;
}

template <class State>
SEXP state_prediction_inference(State *state, SEXP x_theta, SEXP z_i,
                                SEXP z_j, SEXP z_x, SEXP n_random) {
  if (state == nullptr || TYPEOF(x_theta) != REALSXP ||
      TYPEOF(z_i) != INTSXP || TYPEOF(z_j) != INTSXP ||
      TYPEOF(z_x) != REALSXP || TYPEOF(n_random) != INTSXP ||
      Rf_xlength(n_random) != 1)
    throw std::invalid_argument("invalid prediction uncertainty inputs");
  SEXP dims = Rf_getAttrib(x_theta, R_DimSymbol);
  if (Rf_length(dims) != 2 || Rf_xlength(z_i) != Rf_xlength(z_x) ||
      Rf_xlength(z_j) != Rf_xlength(z_x))
    throw std::invalid_argument("non-conformable prediction inputs");
  const int n = INTEGER(dims)[0];
  const int p = INTEGER(dims)[1];
  const int q = INTEGER(n_random)[0];
  Eigen::Map<const Eigen::MatrixXd> X(REAL(x_theta), n, p);
  std::vector<Eigen::Triplet<double>> entries;
  entries.reserve(static_cast<size_t>(Rf_xlength(z_x)));
  for (R_xlen_t k = 0; k < Rf_xlength(z_x); ++k) {
    const int row = INTEGER(z_i)[k];
    const int col = INTEGER(z_j)[k];
    const double x = REAL(z_x)[k];
    if (row < 0 || row >= n || col < 0 || col >= q ||
        !std::isfinite(x))
      throw std::invalid_argument("invalid prediction projection triplet");
    entries.emplace_back(row, col, x);
  }
  Eigen::SparseMatrix<double, Eigen::RowMajor> Z(n, q);
  Z.setFromTriplets(entries.begin(), entries.end());
  return prediction_inference_to_sexp(state->PredictionInference(X, Z));
}

SEXP fixed_covariance_to_sexp(
    const quadra::FixedEffectCovarianceResult &value,
    const BridgeDerivedInference &derived,
    const BridgeRandomEffectInference &random_effects) {
  const int n = static_cast<int>(value.hessian_m.rows());
  SEXP result = PROTECT(Rf_allocVector(VECSXP, 11));
  SEXP names = PROTECT(Rf_allocVector(STRSXP, 11));
  auto matrix_to_sexp = [n](const Eigen::MatrixXd &matrix) {
    SEXP out = Rf_allocMatrix(REALSXP, n, n);
    for (int j = 0; j < n; ++j)
      for (int i = 0; i < n; ++i)
        REAL(out)[i + n * j] = matrix(i, j);
    return out;
  };
  SEXP hessian = PROTECT(matrix_to_sexp(value.hessian_m));
  SEXP covariance = PROTECT(matrix_to_sexp(value.covariance_m));
  SEXP correlation = PROTECT(matrix_to_sexp(value.correlation_m));
  SEXP eigenvalues = PROTECT(Rf_allocVector(REALSXP, n));
  SEXP steps = PROTECT(Rf_allocVector(REALSXP, n));
  SEXP derived_sexp = PROTECT(derived_inference_to_sexp(derived));
  const int n_random =
      static_cast<int>(random_effects.result.std_error_m.size());
  SEXP random_sexp = PROTECT(Rf_allocVector(VECSXP, 6));
  SEXP random_names = PROTECT(Rf_allocVector(STRSXP, 6));
  auto random_vector = [n_random](const Eigen::VectorXd &x) {
    SEXP out = Rf_allocVector(REALSXP, n_random);
    for (int i = 0; i < n_random; ++i)
      REAL(out)[i] = x.size() == n_random ? x[i] : NA_REAL;
    return out;
  };
  SEXP conditional = PROTECT(
      random_vector(random_effects.result.conditional_variance_m));
  SEXP parameter = PROTECT(
      random_vector(random_effects.result.parameter_variance_m));
  SEXP marginal = PROTECT(
      random_vector(random_effects.result.marginal_variance_m));
  SEXP random_std_error =
      PROTECT(random_vector(random_effects.result.std_error_m));
  SET_VECTOR_ELT(random_sexp, 0, conditional);
  SET_VECTOR_ELT(random_sexp, 1, parameter);
  SET_VECTOR_ELT(random_sexp, 2, marginal);
  SET_VECTOR_ELT(random_sexp, 3, random_std_error);
  SET_VECTOR_ELT(
      random_sexp, 4, Rf_ScalarLogical(random_effects.result.success_m));
  SET_VECTOR_ELT(
      random_sexp, 5,
      Rf_mkString(random_effects.result.message_m.c_str()));
  const char *random_labels[] = {
      "conditional_variance", "parameter_variance", "marginal_variance",
      "std_error", "success", "message"};
  for (int i = 0; i < 6; ++i)
    SET_STRING_ELT(random_names, i, Rf_mkChar(random_labels[i]));
  Rf_setAttrib(random_sexp, R_NamesSymbol, random_names);
  for (int i = 0; i < n; ++i) {
    REAL(eigenvalues)[i] =
        value.eigenvalues_m.size() == n ? value.eigenvalues_m[i] : NA_REAL;
    REAL(steps)[i] = value.steps_m.size() == n ? value.steps_m[i] : NA_REAL;
  }
  SET_VECTOR_ELT(result, 0, hessian);
  SET_VECTOR_ELT(result, 1, covariance);
  SET_VECTOR_ELT(result, 2, correlation);
  SET_VECTOR_ELT(result, 3, eigenvalues);
  SET_VECTOR_ELT(result, 4, steps);
  SET_VECTOR_ELT(result, 5, Rf_ScalarLogical(value.success_m));
  SET_VECTOR_ELT(result, 6, Rf_ScalarLogical(value.positive_definite_m));
  SET_VECTOR_ELT(result, 7, Rf_ScalarReal(value.condition_number_m));
  SET_VECTOR_ELT(result, 8, Rf_mkString(value.message_m.c_str()));
  SET_VECTOR_ELT(result, 9, derived_sexp);
  SET_VECTOR_ELT(result, 10, random_sexp);
  const char *labels[] = {
      "hessian", "covariance", "correlation", "eigenvalues", "steps",
      "success", "positive_definite", "condition_number", "message",
      "derived", "random_effects"};
  for (int i = 0; i < 11; ++i)
    SET_STRING_ELT(names, i, Rf_mkChar(labels[i]));
  Rf_setAttrib(result, R_NamesSymbol, names);
  UNPROTECT(14);
  return result;
}

extern "C" SEXP sdmTMB_quadra_gaussian(SEXP beta, SEXP x, SEXP y,
                                         SEXP offset) {
  try {
    validate_inputs(beta, x, y, offset);

    SEXP dims = Rf_getAttrib(x, R_DimSymbol);
    const int n = INTEGER(dims)[0];
    const int p = INTEGER(dims)[1];

    ADGraphReset graph_reset;
    quadra::TapeContext tape;
    quadra::ADScope scope(tape.graph);
    std::vector<quadra::AD> beta_ad;
    beta_ad.reserve(static_cast<size_t>(p));
    for (int j = 0; j < p; ++j) {
      beta_ad.emplace_back(REAL(beta)[j]);
    }

    const quadra::AD nll =
        gaussian_nll(beta_ad, REAL(x), REAL(y), REAL(offset), n, p);
    const double value = scope.value(nll);
    scope.backward(nll);

    SEXP gradient = PROTECT(Rf_allocVector(REALSXP, p));
    for (int j = 0; j < p; ++j) {
      REAL(gradient)[j] = scope.grad(beta_ad[j]);
    }

    SEXP result = PROTECT(Rf_allocVector(VECSXP, 2));
    SET_VECTOR_ELT(result, 0, Rf_ScalarReal(value));
    SET_VECTOR_ELT(result, 1, gradient);

    SEXP names = PROTECT(Rf_allocVector(STRSXP, 2));
    SET_STRING_ELT(names, 0, Rf_mkChar("value"));
    SET_STRING_ELT(names, 1, Rf_mkChar("gradient"));
    Rf_setAttrib(result, R_NamesSymbol, names);

    UNPROTECT(3);
    return result;
  } catch (const std::exception &e) {
    Rf_error("Quadra Gaussian evaluation failed: %s", e.what());
  } catch (...) {
    Rf_error("Quadra Gaussian evaluation failed with an unknown error");
  }

  return R_NilValue;
}

extern "C" SEXP sdmTMB_quadra_gaussian_random_intercept(
    SEXP fixed, SEXP random, SEXP x, SEXP y, SEXP offset, SEXP group) {
  try {
    validate_random_intercept_inputs(fixed, random, x, y, offset, group);

    SEXP dims = Rf_getAttrib(x, R_DimSymbol);
    const int n = INTEGER(dims)[0];
    const int p = INTEGER(dims)[1];
    const int n_group = static_cast<int>(Rf_xlength(random));

    GaussianRandomInterceptModel model(REAL(x), REAL(y), REAL(offset),
                                       INTEGER(group), n, p, n_group);

    std::vector<double> fixed_values(REAL(fixed),
                                     REAL(fixed) + Rf_xlength(fixed));
    std::vector<double> random_values(REAL(random),
                                      REAL(random) + Rf_xlength(random));

    quadra::ParameterPartition partition;
    for (int j = 0; j < p + 2; ++j) {
      partition.fixed_indices_m.push_back(static_cast<size_t>(j));
    }
    for (int g = 0; g < n_group; ++g) {
      partition.random_indices_m.push_back(static_cast<size_t>(p + 2 + g));
    }

    ADGraphReset graph_reset;
    quadra::LaplaceFixedGradientOptions options;
    options.objective_m.newton_m.gradient_tolerance_m = 1e-10;
    options.objective_m.newton_m.step_tolerance_m = 1e-12;
    options.relative_step_m = 1e-5;
    options.absolute_step_m = 1e-7;

    const quadra::LaplaceFixedGradientResult fit =
        quadra::evaluate_laplace_fixed_gradient(
            model, fixed_values, random_values, partition, options);

    const int n_result = 10;
    SEXP result = PROTECT(Rf_allocVector(VECSXP, n_result));
    SEXP names = PROTECT(Rf_allocVector(STRSXP, n_result));
    SEXP gradient =
        PROTECT(Rf_allocVector(REALSXP, static_cast<R_xlen_t>(p + 2)));
    SEXP u_hat =
        PROTECT(Rf_allocVector(REALSXP, static_cast<R_xlen_t>(n_group)));

    for (int j = 0; j < p + 2; ++j) {
      REAL(gradient)[j] = fit.gradient_fixed_m[static_cast<size_t>(j)];
    }
    for (int g = 0; g < n_group; ++g) {
      REAL(u_hat)[g] = fit.u_hat_m[static_cast<size_t>(g)];
    }

    SET_VECTOR_ELT(result, 0, Rf_ScalarReal(fit.laplace_objective_m));
    SET_VECTOR_ELT(result, 1, gradient);
    SET_VECTOR_ELT(result, 2, u_hat);
    SET_VECTOR_ELT(
        result, 3,
        Rf_ScalarReal(fit.objective_result_m.joint_objective_m));
    SET_VECTOR_ELT(
        result, 4,
        Rf_ScalarReal(fit.objective_result_m.log_det_hessian_m));
    SET_VECTOR_ELT(
        result, 5,
        Rf_ScalarReal(fit.objective_result_m.gradient_norm_random_m));
    SET_VECTOR_ELT(result, 6, Rf_ScalarInteger(
                                  fit.objective_result_m.newton_iterations_m));
    SET_VECTOR_ELT(result, 7, Rf_ScalarLogical(fit.converged_m));
    SET_VECTOR_ELT(result, 8, Rf_ScalarLogical(fit.logdet_ok_m));
    SET_VECTOR_ELT(
        result, 9,
        Rf_ScalarInteger(fit.objective_result_m.hessian_random_m.nonZeros()));

    const char *result_names[n_result] = {
        "value",           "gradient",       "u_hat",
        "joint_objective", "logdet_hessian", "random_gradient_norm",
        "iterations",      "converged",      "logdet_ok",
        "hessian_nonzeros"};
    for (int i = 0; i < n_result; ++i) {
      SET_STRING_ELT(names, i, Rf_mkChar(result_names[i]));
    }
    Rf_setAttrib(result, R_NamesSymbol, names);

    UNPROTECT(4);
    return result;
  } catch (const std::exception &e) {
    Rf_error("Quadra random-intercept evaluation failed: %s", e.what());
  } catch (...) {
    Rf_error(
        "Quadra random-intercept evaluation failed with an unknown error");
  }

  return R_NilValue;
}

extern "C" SEXP sdmTMB_quadra_gaussian_sparse_field(
    SEXP fixed, SEXP random, SEXP x, SEXP y, SEXP offset, SEXP node, SEXP q_i,
    SEXP q_j, SEXP q_x, SEXP logdet_q) {
  try {
    validate_sparse_field_inputs(fixed, random, x, y, offset, node, q_i, q_j,
                                 q_x, logdet_q);

    SEXP dims = Rf_getAttrib(x, R_DimSymbol);
    const int n = INTEGER(dims)[0];
    const int p = INTEGER(dims)[1];
    const int n_node = static_cast<int>(Rf_xlength(random));
    const int n_q = static_cast<int>(Rf_xlength(q_x));

    GaussianSparseFieldModel model(
        REAL(x), REAL(y), REAL(offset), INTEGER(node), n, p, n_node,
        INTEGER(q_i), INTEGER(q_j), REAL(q_x), n_q, REAL(logdet_q)[0]);

    std::vector<double> fixed_values(REAL(fixed),
                                     REAL(fixed) + Rf_xlength(fixed));
    std::vector<double> random_values(REAL(random),
                                      REAL(random) + Rf_xlength(random));
    quadra::ParameterPartition partition;
    for (int j = 0; j < p + 2; ++j) {
      partition.fixed_indices_m.push_back(static_cast<size_t>(j));
    }
    for (int s = 0; s < n_node; ++s) {
      partition.random_indices_m.push_back(static_cast<size_t>(p + 2 + s));
    }

    ADGraphReset graph_reset;
    quadra::LaplaceFixedGradientOptions options;
    options.objective_m.newton_m.gradient_tolerance_m = 1e-10;
    options.objective_m.newton_m.step_tolerance_m = 1e-12;
    options.relative_step_m = 1e-5;
    options.absolute_step_m = 1e-7;

    const quadra::LaplaceFixedGradientResult fit =
        quadra::evaluate_laplace_fixed_gradient(
            model, fixed_values, random_values, partition, options);

    const int n_result = 10;
    SEXP result = PROTECT(Rf_allocVector(VECSXP, n_result));
    SEXP names = PROTECT(Rf_allocVector(STRSXP, n_result));
    SEXP gradient =
        PROTECT(Rf_allocVector(REALSXP, static_cast<R_xlen_t>(p + 2)));
    SEXP u_hat =
        PROTECT(Rf_allocVector(REALSXP, static_cast<R_xlen_t>(n_node)));

    for (int j = 0; j < p + 2; ++j) {
      REAL(gradient)[j] = fit.gradient_fixed_m[static_cast<size_t>(j)];
    }
    for (int s = 0; s < n_node; ++s) {
      REAL(u_hat)[s] = fit.u_hat_m[static_cast<size_t>(s)];
    }

    SET_VECTOR_ELT(result, 0, Rf_ScalarReal(fit.laplace_objective_m));
    SET_VECTOR_ELT(result, 1, gradient);
    SET_VECTOR_ELT(result, 2, u_hat);
    SET_VECTOR_ELT(
        result, 3,
        Rf_ScalarReal(fit.objective_result_m.joint_objective_m));
    SET_VECTOR_ELT(
        result, 4,
        Rf_ScalarReal(fit.objective_result_m.log_det_hessian_m));
    SET_VECTOR_ELT(
        result, 5,
        Rf_ScalarReal(fit.objective_result_m.gradient_norm_random_m));
    SET_VECTOR_ELT(result, 6, Rf_ScalarInteger(
                                  fit.objective_result_m.newton_iterations_m));
    SET_VECTOR_ELT(result, 7, Rf_ScalarLogical(fit.converged_m));
    SET_VECTOR_ELT(result, 8, Rf_ScalarLogical(fit.logdet_ok_m));
    SET_VECTOR_ELT(
        result, 9,
        Rf_ScalarInteger(fit.objective_result_m.hessian_random_m.nonZeros()));

    const char *result_names[n_result] = {
        "value",           "gradient",       "u_hat",
        "joint_objective", "logdet_hessian", "random_gradient_norm",
        "iterations",      "converged",      "logdet_ok",
        "hessian_nonzeros"};
    for (int i = 0; i < n_result; ++i) {
      SET_STRING_ELT(names, i, Rf_mkChar(result_names[i]));
    }
    Rf_setAttrib(result, R_NamesSymbol, names);

    UNPROTECT(4);
    return result;
  } catch (const std::exception &e) {
    Rf_error("Quadra sparse-field evaluation failed: %s", e.what());
  } catch (...) {
    Rf_error("Quadra sparse-field evaluation failed with an unknown error");
  }

  return R_NilValue;
}

extern "C" SEXP sdmTMB_quadra_gaussian_projected_field(
    SEXP fixed, SEXP random, SEXP x, SEXP y, SEXP offset, SEXP a_i, SEXP a_j,
    SEXP a_x, SEXP q_i, SEXP q_j, SEXP q_x, SEXP logdet_q) {
  try {
    validate_projected_field_inputs(fixed, random, x, y, offset, a_i, a_j, a_x,
                                    q_i, q_j, q_x, logdet_q);
    SEXP dims = Rf_getAttrib(x, R_DimSymbol);
    const int n = INTEGER(dims)[0];
    const int p = INTEGER(dims)[1];
    const int n_node = static_cast<int>(Rf_xlength(random));

    GaussianProjectedFieldModel model(
        REAL(x), REAL(y), REAL(offset), n, p, n_node, INTEGER(a_i),
        INTEGER(a_j), REAL(a_x), static_cast<int>(Rf_xlength(a_x)),
        INTEGER(q_i), INTEGER(q_j), REAL(q_x),
        static_cast<int>(Rf_xlength(q_x)), REAL(logdet_q)[0]);
    std::vector<double> fixed_values(REAL(fixed),
                                     REAL(fixed) + Rf_xlength(fixed));
    std::vector<double> random_values(REAL(random),
                                      REAL(random) + Rf_xlength(random));
    quadra::ParameterPartition partition;
    for (int j = 0; j < p + 2; ++j) {
      partition.fixed_indices_m.push_back(static_cast<size_t>(j));
    }
    for (int s = 0; s < n_node; ++s) {
      partition.random_indices_m.push_back(static_cast<size_t>(p + 2 + s));
    }

    ADGraphReset graph_reset;
    quadra::LaplaceFixedGradientOptions options;
    options.objective_m.newton_m.gradient_tolerance_m = 1e-7;
    options.objective_m.newton_m.step_tolerance_m = 1e-12;
    options.relative_step_m = 1e-5;
    options.absolute_step_m = 1e-7;
    const quadra::LaplaceFixedGradientResult fit =
        quadra::evaluate_laplace_fixed_gradient(
            model, fixed_values, random_values, partition, options);

    const int n_result = 10;
    SEXP result = PROTECT(Rf_allocVector(VECSXP, n_result));
    SEXP names = PROTECT(Rf_allocVector(STRSXP, n_result));
    SEXP gradient =
        PROTECT(Rf_allocVector(REALSXP, static_cast<R_xlen_t>(p + 2)));
    SEXP u_hat =
        PROTECT(Rf_allocVector(REALSXP, static_cast<R_xlen_t>(n_node)));
    for (int j = 0; j < p + 2; ++j) {
      REAL(gradient)[j] = fit.gradient_fixed_m[static_cast<size_t>(j)];
    }
    for (int s = 0; s < n_node; ++s) {
      REAL(u_hat)[s] = fit.u_hat_m[static_cast<size_t>(s)];
    }
    SET_VECTOR_ELT(result, 0, Rf_ScalarReal(fit.laplace_objective_m));
    SET_VECTOR_ELT(result, 1, gradient);
    SET_VECTOR_ELT(result, 2, u_hat);
    SET_VECTOR_ELT(
        result, 3,
        Rf_ScalarReal(fit.objective_result_m.joint_objective_m));
    SET_VECTOR_ELT(
        result, 4,
        Rf_ScalarReal(fit.objective_result_m.log_det_hessian_m));
    SET_VECTOR_ELT(
        result, 5,
        Rf_ScalarReal(fit.objective_result_m.gradient_norm_random_m));
    SET_VECTOR_ELT(result, 6, Rf_ScalarInteger(
                                  fit.objective_result_m.newton_iterations_m));
    SET_VECTOR_ELT(result, 7, Rf_ScalarLogical(fit.converged_m));
    SET_VECTOR_ELT(result, 8, Rf_ScalarLogical(fit.logdet_ok_m));
    SET_VECTOR_ELT(
        result, 9,
        Rf_ScalarInteger(fit.objective_result_m.hessian_random_m.nonZeros()));
    const char *result_names[n_result] = {
        "value",           "gradient",       "u_hat",
        "joint_objective", "logdet_hessian", "random_gradient_norm",
        "iterations",      "converged",      "logdet_ok",
        "hessian_nonzeros"};
    for (int i = 0; i < n_result; ++i) {
      SET_STRING_ELT(names, i, Rf_mkChar(result_names[i]));
    }
    Rf_setAttrib(result, R_NamesSymbol, names);
    UNPROTECT(4);
    return result;
  } catch (const std::exception &e) {
    Rf_error("Quadra projected-field evaluation failed: %s", e.what());
  } catch (...) {
    Rf_error("Quadra projected-field evaluation failed with an unknown error");
  }
  return R_NilValue;
}

extern "C" SEXP sdmTMB_quadra_gaussian_spde_field(
    SEXP fixed, SEXP random, SEXP x, SEXP y, SEXP offset, SEXP a_i, SEXP a_j,
    SEXP a_x, SEXP m0_i, SEXP m0_j, SEXP m0_x, SEXP m1_i, SEXP m1_j,
    SEXP m1_x, SEXP m2_i, SEXP m2_j, SEXP m2_x, SEXP gradient_requested) {
  try {
    if (TYPEOF(gradient_requested) != LGLSXP ||
        Rf_xlength(gradient_requested) != 1) {
      throw std::invalid_argument("gradient_requested must be one logical");
    }
    validate_spde_field_inputs(fixed, random, x, y, offset, a_i, a_j, a_x,
                               m0_i, m0_j, m0_x, m1_i, m1_j, m1_x, m2_i,
                               m2_j, m2_x, 3);
    SEXP dims = Rf_getAttrib(x, R_DimSymbol);
    const int n = INTEGER(dims)[0];
    const int p = INTEGER(dims)[1];
    const int n_node = static_cast<int>(Rf_xlength(random));
    GaussianSpdeFieldModel model(
        REAL(x), REAL(y), REAL(offset), n, p, n_node, INTEGER(a_i),
        INTEGER(a_j), REAL(a_x), static_cast<int>(Rf_xlength(a_x)),
        INTEGER(m0_i), INTEGER(m0_j), REAL(m0_x),
        static_cast<int>(Rf_xlength(m0_x)), INTEGER(m1_i), INTEGER(m1_j),
        REAL(m1_x), static_cast<int>(Rf_xlength(m1_x)), INTEGER(m2_i),
        INTEGER(m2_j), REAL(m2_x), static_cast<int>(Rf_xlength(m2_x)));
    std::vector<double> fixed_values(REAL(fixed),
                                     REAL(fixed) + Rf_xlength(fixed));
    std::vector<double> random_values(REAL(random),
                                      REAL(random) + Rf_xlength(random));
    quadra::ParameterPartition partition;
    for (int j = 0; j < p + 3; ++j) {
      partition.fixed_indices_m.push_back(static_cast<size_t>(j));
    }
    for (int s = 0; s < n_node; ++s) {
      partition.random_indices_m.push_back(static_cast<size_t>(p + 3 + s));
    }

    ADGraphReset graph_reset;
    quadra::LaplaceFixedGradientOptions options;
    options.objective_m.newton_m.gradient_tolerance_m = 1e-7;
    options.objective_m.newton_m.step_tolerance_m = 1e-12;
    options.relative_step_m = 1e-5;
    options.absolute_step_m = 1e-7;
    const quadra::LaplaceObjectiveResult center =
        quadra::evaluate_laplace_objective(
            model, fixed_values, random_values, partition,
            options.objective_m);
    quadra::LaplaceFixedGradientResult fit;
    fit.fixed_m = fixed_values;
    fit.u_hat_m = center.u_hat_m;
    fit.full_m = center.full_m;
    fit.gradient_fixed_m.assign(fixed_values.size(), NA_REAL);
    fit.laplace_objective_m = center.laplace_objective_m;
    fit.converged_m = center.converged_m;
    fit.logdet_ok_m = center.logdet_ok_m;
    fit.message_m = center.message_m;
    fit.objective_result_m = center;

    if (LOGICAL(gradient_requested)[0] && center.converged_m &&
        center.logdet_ok_m) {
      auto sparse_from_triplets =
          [n_node](SEXP row, SEXP col, SEXP value, double scale) {
            std::vector<Eigen::Triplet<double>> entries;
            entries.reserve(static_cast<size_t>(Rf_xlength(value)));
            for (R_xlen_t k = 0; k < Rf_xlength(value); ++k) {
              entries.emplace_back(INTEGER(row)[k], INTEGER(col)[k],
                                   scale * REAL(value)[k]);
            }
            Eigen::SparseMatrix<double> out(n_node, n_node);
            out.setFromTriplets(entries.begin(), entries.end());
            return out;
          };

      const double sigma = std::exp(fixed_values[static_cast<size_t>(p)]);
      const double sigma2 = sigma * sigma;
      const double tau = std::exp(fixed_values[static_cast<size_t>(p + 1)]);
      const double tau2 = tau * tau;
      const double log_kappa = fixed_values[static_cast<size_t>(p + 2)];
      const double kappa2 = std::exp(2.0 * log_kappa);
      const double kappa4 = kappa2 * kappa2;
      const Eigen::SparseMatrix<double> M0 =
          sparse_from_triplets(m0_i, m0_j, m0_x, 1.0);
      const Eigen::SparseMatrix<double> M1 =
          sparse_from_triplets(m1_i, m1_j, m1_x, 1.0);
      const Eigen::SparseMatrix<double> M2 =
          sparse_from_triplets(m2_i, m2_j, m2_x, 1.0);
      const Eigen::SparseMatrix<double> Q =
          kappa4 * M0 + (2.0 * kappa2) * M1 + M2;
      const Eigen::SparseMatrix<double> dQ =
          (4.0 * kappa4) * M0 + (4.0 * kappa2) * M1;
      std::vector<Eigen::Triplet<double>> a_entries;
      a_entries.reserve(static_cast<size_t>(Rf_xlength(a_x)));
      for (R_xlen_t k = 0; k < Rf_xlength(a_x); ++k) {
        a_entries.emplace_back(INTEGER(a_i)[k], INTEGER(a_j)[k],
                               REAL(a_x)[k]);
      }
      Eigen::SparseMatrix<double> A(n, n_node);
      A.setFromTriplets(a_entries.begin(), a_entries.end());
      const Eigen::MatrixXd AtA =
          Eigen::MatrixXd(A.transpose() * A);
      const Eigen::MatrixXd H =
          Eigen::MatrixXd(center.hessian_random_m);
      const Eigen::MatrixXd Qdense(Q);
      const Eigen::MatrixXd dQdense(dQ);
      Eigen::LDLT<Eigen::MatrixXd> hessian_ldlt(H);
      Eigen::LDLT<Eigen::MatrixXd> q_ldlt(Qdense);
      if (hessian_ldlt.info() != Eigen::Success ||
          q_ldlt.info() != Eigen::Success) {
        throw std::runtime_error(
            "exact SPDE gradient factorization failed");
      }
      const Eigen::MatrixXd identity =
          Eigen::MatrixXd::Identity(n_node, n_node);
      const Eigen::MatrixXd Hinv = hessian_ldlt.solve(identity);
      const Eigen::MatrixXd Qinv = q_ldlt.solve(identity);
      const Eigen::VectorXd u =
          Eigen::Map<const Eigen::VectorXd>(center.u_hat_m.data(), n_node);
      const Eigen::VectorXd projected = A * u;

      Eigen::VectorXd residual(n);
      for (int i = 0; i < n; ++i) {
        double eta = REAL(offset)[i] + projected[i];
        for (int j = 0; j < p; ++j) {
          eta += REAL(x)[i + n * j] * fixed_values[static_cast<size_t>(j)];
        }
        residual[i] = REAL(y)[i] - eta;
      }
      for (int j = 0; j < p; ++j) {
        double gradient_beta = 0.0;
        for (int i = 0; i < n; ++i) {
          gradient_beta -= REAL(x)[i + n * j] * residual[i] / sigma2;
        }
        fit.gradient_fixed_m[static_cast<size_t>(j)] = gradient_beta;
      }

      const double observation_trace =
          (Hinv * (-2.0 / sigma2 * AtA)).trace();
      fit.gradient_fixed_m[static_cast<size_t>(p)] =
          static_cast<double>(n) - residual.squaredNorm() / sigma2 +
          0.5 * observation_trace;

      fit.gradient_fixed_m[static_cast<size_t>(p + 1)] =
          tau2 * u.dot(Q * u) - static_cast<double>(n_node) +
          tau2 * (Hinv * Qdense).trace();
      fit.gradient_fixed_m[static_cast<size_t>(p + 2)] =
          0.5 * tau2 * u.dot(dQ * u) -
          0.5 * (Qinv * dQdense).trace() +
          0.5 * tau2 * (Hinv * dQdense).trace();
      fit.gradient_norm_m = quadra::norm2(fit.gradient_fixed_m);
    }

    const int n_result = 10;
    SEXP result = PROTECT(Rf_allocVector(VECSXP, n_result));
    SEXP names = PROTECT(Rf_allocVector(STRSXP, n_result));
    SEXP gradient =
        PROTECT(Rf_allocVector(REALSXP, static_cast<R_xlen_t>(p + 3)));
    SEXP u_hat =
        PROTECT(Rf_allocVector(REALSXP, static_cast<R_xlen_t>(n_node)));
    for (int j = 0; j < p + 3; ++j) {
      REAL(gradient)[j] = fit.gradient_fixed_m[static_cast<size_t>(j)];
    }
    for (int s = 0; s < n_node; ++s) {
      REAL(u_hat)[s] = fit.u_hat_m[static_cast<size_t>(s)];
    }
    SET_VECTOR_ELT(result, 0, Rf_ScalarReal(fit.laplace_objective_m));
    SET_VECTOR_ELT(result, 1, gradient);
    SET_VECTOR_ELT(result, 2, u_hat);
    SET_VECTOR_ELT(
        result, 3,
        Rf_ScalarReal(fit.objective_result_m.joint_objective_m));
    SET_VECTOR_ELT(
        result, 4,
        Rf_ScalarReal(fit.objective_result_m.log_det_hessian_m));
    SET_VECTOR_ELT(
        result, 5,
        Rf_ScalarReal(fit.objective_result_m.gradient_norm_random_m));
    SET_VECTOR_ELT(result, 6, Rf_ScalarInteger(
                                  fit.objective_result_m.newton_iterations_m));
    SET_VECTOR_ELT(result, 7, Rf_ScalarLogical(fit.converged_m));
    SET_VECTOR_ELT(result, 8, Rf_ScalarLogical(fit.logdet_ok_m));
    SET_VECTOR_ELT(
        result, 9,
        Rf_ScalarInteger(fit.objective_result_m.hessian_random_m.nonZeros()));
    const char *result_names[n_result] = {
        "value",           "gradient",       "u_hat",
        "joint_objective", "logdet_hessian", "random_gradient_norm",
        "iterations",      "converged",      "logdet_ok",
        "hessian_nonzeros"};
    for (int i = 0; i < n_result; ++i) {
      SET_STRING_ELT(names, i, Rf_mkChar(result_names[i]));
    }
    Rf_setAttrib(result, R_NamesSymbol, names);
    UNPROTECT(4);
    return result;
  } catch (const std::exception &e) {
    Rf_error("Quadra SPDE-field evaluation failed: %s", e.what());
  } catch (...) {
    Rf_error("Quadra SPDE-field evaluation failed with an unknown error");
  }
  return R_NilValue;
}

extern "C" SEXP sdmTMB_quadra_poisson_spde_field(
    SEXP fixed, SEXP random, SEXP x, SEXP y, SEXP offset, SEXP a_i, SEXP a_j,
    SEXP a_x, SEXP m0_i, SEXP m0_j, SEXP m0_x, SEXP m1_i, SEXP m1_j,
    SEXP m1_x, SEXP m2_i, SEXP m2_j, SEXP m2_x, SEXP gradient_requested) {
  try {
    validate_spde_field_inputs(fixed, random, x, y, offset, a_i, a_j, a_x,
                               m0_i, m0_j, m0_x, m1_i, m1_j, m1_x, m2_i,
                               m2_j, m2_x, 2);
    if (TYPEOF(gradient_requested) != LGLSXP ||
        Rf_xlength(gradient_requested) != 1)
      throw std::invalid_argument("gradient_requested must be one logical");
    SEXP dims = Rf_getAttrib(x, R_DimSymbol);
    const int n = INTEGER(dims)[0], p = INTEGER(dims)[1];
    const int n_node = static_cast<int>(Rf_xlength(random));
    for (int i = 0; i < n; ++i) {
      if (REAL(y)[i] < 0.0 || REAL(y)[i] != std::floor(REAL(y)[i]))
        throw std::invalid_argument("Poisson response must be nonnegative integers");
    }
    PoissonSpdeFieldModel model(
        REAL(x), REAL(y), REAL(offset), nullptr, n, p, n_node, INTEGER(a_i),
        INTEGER(a_j), REAL(a_x), static_cast<int>(Rf_xlength(a_x)),
        INTEGER(m0_i), INTEGER(m0_j), REAL(m0_x),
        static_cast<int>(Rf_xlength(m0_x)), INTEGER(m1_i), INTEGER(m1_j),
        REAL(m1_x), static_cast<int>(Rf_xlength(m1_x)), INTEGER(m2_i),
        INTEGER(m2_j), REAL(m2_x), static_cast<int>(Rf_xlength(m2_x)));
    std::vector<double> fixed_values(REAL(fixed),
                                     REAL(fixed) + Rf_xlength(fixed));
    std::vector<double> random_values(REAL(random),
                                      REAL(random) + Rf_xlength(random));
    quadra::ParameterPartition partition;
    for (int j = 0; j < p + 2; ++j)
      partition.fixed_indices_m.push_back(static_cast<size_t>(j));
    for (int s = 0; s < n_node; ++s)
      partition.random_indices_m.push_back(static_cast<size_t>(p + 2 + s));
    ADGraphReset graph_reset;
    quadra::LaplaceObjectiveOptions options;
    options.newton_m.max_iterations_m = 100;
    options.newton_m.gradient_tolerance_m = 1e-7;
    options.newton_m.step_tolerance_m = 1e-12;
    const quadra::LaplaceObjectiveResult center =
        quadra::evaluate_laplace_objective(model, fixed_values, random_values,
                                           partition, options);
    std::vector<double> gradient(static_cast<size_t>(p + 2), NA_REAL);

    if (LOGICAL(gradient_requested)[0] && center.logdet_ok_m) {
      auto matrix_from_triplets =
          [n_node](SEXP row, SEXP col, SEXP value, double scale) {
            std::vector<Eigen::Triplet<double>> entries;
            entries.reserve(static_cast<size_t>(Rf_xlength(value)));
            for (R_xlen_t k = 0; k < Rf_xlength(value); ++k)
              entries.emplace_back(INTEGER(row)[k], INTEGER(col)[k],
                                   scale * REAL(value)[k]);
            Eigen::SparseMatrix<double> out(n_node, n_node);
            out.setFromTriplets(entries.begin(), entries.end());
            return out;
          };
      std::vector<Eigen::Triplet<double>> a_entries;
      a_entries.reserve(static_cast<size_t>(Rf_xlength(a_x)));
      for (R_xlen_t k = 0; k < Rf_xlength(a_x); ++k)
        a_entries.emplace_back(INTEGER(a_i)[k], INTEGER(a_j)[k], REAL(a_x)[k]);
      Eigen::SparseMatrix<double> A(n, n_node);
      A.setFromTriplets(a_entries.begin(), a_entries.end());
      const double tau = std::exp(fixed_values[static_cast<size_t>(p)]);
      const double tau2 = tau * tau;
      const double kappa2 =
          std::exp(2.0 * fixed_values[static_cast<size_t>(p + 1)]);
      const double kappa4 = kappa2 * kappa2;
      const Eigen::SparseMatrix<double> M0 =
          matrix_from_triplets(m0_i, m0_j, m0_x, 1.0);
      const Eigen::SparseMatrix<double> M1 =
          matrix_from_triplets(m1_i, m1_j, m1_x, 1.0);
      const Eigen::SparseMatrix<double> M2 =
          matrix_from_triplets(m2_i, m2_j, m2_x, 1.0);
      const Eigen::SparseMatrix<double> Q =
          kappa4 * M0 + (2.0 * kappa2) * M1 + M2;
      const Eigen::SparseMatrix<double> dQ =
          (4.0 * kappa4) * M0 + (4.0 * kappa2) * M1;
      const Eigen::MatrixXd H(center.hessian_random_m);
      const Eigen::MatrixXd Qdense(Q), dQdense(dQ);
      Eigen::LDLT<Eigen::MatrixXd> h_ldlt(H), q_ldlt(Qdense);
      if (h_ldlt.info() != Eigen::Success ||
          q_ldlt.info() != Eigen::Success)
        throw std::runtime_error("exact Poisson gradient factorization failed");
      const Eigen::MatrixXd identity =
          Eigen::MatrixXd::Identity(n_node, n_node);
      const Eigen::MatrixXd Hinv = h_ldlt.solve(identity);
      const Eigen::MatrixXd Qinv = q_ldlt.solve(identity);
      const Eigen::MatrixXd Adense(A);
      const Eigen::VectorXd spatial_leverage =
          (Adense * Hinv).cwiseProduct(Adense).rowwise().sum();
      const double trace_hinv_q =
          Hinv.cwiseProduct(Qdense.transpose()).sum();
      const double trace_hinv_dq =
          Hinv.cwiseProduct(dQdense.transpose()).sum();
      const double trace_qinv_dq =
          Qinv.cwiseProduct(dQdense.transpose()).sum();
      const Eigen::VectorXd u =
          Eigen::Map<const Eigen::VectorXd>(center.u_hat_m.data(), n_node);
      const Eigen::VectorXd projected = A * u;
      Eigen::VectorXd eta(n), mu(n), residual(n);
      for (int i = 0; i < n; ++i) {
        eta[i] = REAL(offset)[i] + projected[i];
        for (int j = 0; j < p; ++j)
          eta[i] += REAL(x)[i + n * j] *
                    fixed_values[static_cast<size_t>(j)];
        mu[i] = std::exp(eta[i]);
        residual[i] = mu[i] - REAL(y)[i];
      }
      for (int j = 0; j < p + 2; ++j) {
        Eigen::VectorXd direct = Eigen::VectorXd::Zero(n);
        Eigen::VectorXd cross = Eigen::VectorXd::Zero(n_node);
        double joint = 0.0;
        double prior_trace = 0.0;
        if (j < p) {
          for (int i = 0; i < n; ++i) {
            direct[i] = REAL(x)[i + n * j];
            joint += direct[i] * residual[i];
          }
          cross = A.transpose() * mu.cwiseProduct(direct);
        } else if (j == p) {
          joint = tau2 * u.dot(Q * u) - static_cast<double>(n_node);
          cross = 2.0 * tau2 * (Q * u);
          prior_trace = 2.0 * tau2 * trace_hinv_q;
        } else {
          joint = 0.5 * tau2 * u.dot(dQ * u) -
                  0.5 * trace_qinv_dq;
          cross = tau2 * (dQ * u);
          prior_trace = tau2 * trace_hinv_dq;
        }
        const Eigen::VectorXd du = -Hinv * cross;
        const Eigen::VectorXd deta = direct + A * du;
        gradient[static_cast<size_t>(j)] =
            joint +
            0.5 * (mu.cwiseProduct(deta)).dot(spatial_leverage) +
            0.5 * prior_trace;
      }
    }

    const int n_result = 10;
    SEXP result = PROTECT(Rf_allocVector(VECSXP, n_result));
    SEXP names = PROTECT(Rf_allocVector(STRSXP, n_result));
    SEXP gradient_sexp = PROTECT(Rf_allocVector(REALSXP, p + 2));
    SEXP u_hat = PROTECT(Rf_allocVector(REALSXP, n_node));
    for (int j = 0; j < p + 2; ++j)
      REAL(gradient_sexp)[j] = gradient[static_cast<size_t>(j)];
    for (int s = 0; s < n_node; ++s)
      REAL(u_hat)[s] = center.u_hat_m[static_cast<size_t>(s)];
    SET_VECTOR_ELT(result, 0, Rf_ScalarReal(center.laplace_objective_m));
    SET_VECTOR_ELT(result, 1, gradient_sexp);
    SET_VECTOR_ELT(result, 2, u_hat);
    SET_VECTOR_ELT(result, 3, Rf_ScalarReal(center.joint_objective_m));
    SET_VECTOR_ELT(result, 4, Rf_ScalarReal(center.log_det_hessian_m));
    SET_VECTOR_ELT(result, 5, Rf_ScalarReal(center.gradient_norm_random_m));
    SET_VECTOR_ELT(result, 6, Rf_ScalarInteger(center.newton_iterations_m));
    SET_VECTOR_ELT(result, 7, Rf_ScalarLogical(center.converged_m));
    SET_VECTOR_ELT(result, 8, Rf_ScalarLogical(center.logdet_ok_m));
    SET_VECTOR_ELT(result, 9,
                   Rf_ScalarInteger(center.hessian_random_m.nonZeros()));
    const char *result_names[n_result] = {
        "value", "gradient", "u_hat", "joint_objective", "logdet_hessian",
        "random_gradient_norm", "iterations", "converged", "logdet_ok",
        "hessian_nonzeros"};
    for (int i = 0; i < n_result; ++i)
      SET_STRING_ELT(names, i, Rf_mkChar(result_names[i]));
    Rf_setAttrib(result, R_NamesSymbol, names);
    UNPROTECT(4);
    return result;
  } catch (const std::exception &e) {
    Rf_error("Quadra Poisson SPDE evaluation failed: %s", e.what());
  } catch (...) {
    Rf_error("Quadra Poisson SPDE evaluation failed with an unknown error");
  }
  return R_NilValue;
}

extern "C" void sdmTMB_quadra_poisson_state_finalizer(SEXP pointer) {
  auto *state = static_cast<PersistentPoissonSpdeState *>(
      R_ExternalPtrAddr(pointer));
  if (state != nullptr) {
    delete state;
    R_ClearExternalPtr(pointer);
  }
}

extern "C" SEXP sdmTMB_quadra_poisson_state_create(
    SEXP fixed, SEXP random, SEXP x, SEXP y, SEXP offset, SEXP weights,
    SEXP a_i, SEXP a_j, SEXP a_x, SEXP m0_i, SEXP m0_j, SEXP m0_x,
    SEXP m1_i, SEXP m1_j, SEXP m1_x, SEXP m2_i, SEXP m2_j, SEXP m2_x,
    SEXP nb2, SEXP gaussian) {
  try {
    if (TYPEOF(nb2) != LGLSXP || Rf_xlength(nb2) != 1 ||
        TYPEOF(gaussian) != LGLSXP || Rf_xlength(gaussian) != 1)
      throw std::invalid_argument("likelihood flags must be logical scalars");
    const bool use_nb2 = LOGICAL(nb2)[0];
    const bool use_gaussian = LOGICAL(gaussian)[0];
    validate_spde_field_inputs(fixed, random, x, y, offset, a_i, a_j, a_x,
                               m0_i, m0_j, m0_x, m1_i, m1_j, m1_x, m2_i,
                               m2_j, m2_x,
                               2 + static_cast<int>(use_nb2 || use_gaussian));
    SEXP dims = Rf_getAttrib(x, R_DimSymbol);
    const int n = INTEGER(dims)[0], p = INTEGER(dims)[1];
    const int n_node = static_cast<int>(Rf_xlength(random));
    if (TYPEOF(weights) != REALSXP || Rf_xlength(weights) != n)
      throw std::invalid_argument("weights must be a double vector of length n");
    for (int i = 0; i < n; ++i)
      if (!R_FINITE(REAL(weights)[i]) || REAL(weights)[i] < 0.0)
        throw std::invalid_argument("weights must be finite and nonnegative");
    std::vector<double> fixed_values(REAL(fixed),
                                     REAL(fixed) + Rf_xlength(fixed));
    std::vector<double> random_values(REAL(random),
                                      REAL(random) + Rf_xlength(random));
    auto *state = new PersistentPoissonSpdeState(
        fixed_values, random_values, REAL(x), REAL(y), REAL(offset),
        REAL(weights), n, p, n_node, INTEGER(a_i), INTEGER(a_j), REAL(a_x),
        static_cast<int>(Rf_xlength(a_x)), INTEGER(m0_i), INTEGER(m0_j),
        REAL(m0_x), static_cast<int>(Rf_xlength(m0_x)), INTEGER(m1_i),
        INTEGER(m1_j), REAL(m1_x), static_cast<int>(Rf_xlength(m1_x)),
        INTEGER(m2_i), INTEGER(m2_j), REAL(m2_x),
        static_cast<int>(Rf_xlength(m2_x)), use_nb2, use_gaussian);
    SEXP pointer = PROTECT(R_MakeExternalPtr(state, R_NilValue, R_NilValue));
    R_RegisterCFinalizerEx(pointer, sdmTMB_quadra_poisson_state_finalizer,
                          TRUE);
    UNPROTECT(1);
    return pointer;
  } catch (const std::exception &e) {
    Rf_error("Quadra Poisson state creation failed: %s", e.what());
  }
  return R_NilValue;
}

extern "C" SEXP sdmTMB_quadra_poisson_state_evaluate(
    SEXP pointer, SEXP fixed, SEXP gradient_requested) {
  try {
    if (TYPEOF(pointer) != EXTPTRSXP || TYPEOF(fixed) != REALSXP ||
        TYPEOF(gradient_requested) != LGLSXP)
      throw std::invalid_argument("invalid persistent Poisson state input");
    auto *state = static_cast<PersistentPoissonSpdeState *>(
        R_ExternalPtrAddr(pointer));
    if (state == nullptr)
      throw std::runtime_error("persistent Poisson state has been finalized");
    std::vector<double> fixed_values(REAL(fixed),
                                     REAL(fixed) + Rf_xlength(fixed));
    PersistentPoissonResult fit =
        state->Evaluate(fixed_values, LOGICAL(gradient_requested)[0]);
    const auto &center = fit.objective;
    const int n_fixed = static_cast<int>(fit.gradient.size());
    const int n_node = static_cast<int>(center.u_hat_m.size());
    const int n_result = 10;
    SEXP result = PROTECT(Rf_allocVector(VECSXP, n_result));
    SEXP names = PROTECT(Rf_allocVector(STRSXP, n_result));
    SEXP gradient = PROTECT(Rf_allocVector(REALSXP, n_fixed));
    SEXP u_hat = PROTECT(Rf_allocVector(REALSXP, n_node));
    for (int j = 0; j < n_fixed; ++j)
      REAL(gradient)[j] = fit.gradient[static_cast<size_t>(j)];
    for (int j = 0; j < n_node; ++j)
      REAL(u_hat)[j] = center.u_hat_m[static_cast<size_t>(j)];
    SET_VECTOR_ELT(result, 0, Rf_ScalarReal(center.laplace_objective_m));
    SET_VECTOR_ELT(result, 1, gradient);
    SET_VECTOR_ELT(result, 2, u_hat);
    SET_VECTOR_ELT(result, 3, Rf_ScalarReal(center.joint_objective_m));
    SET_VECTOR_ELT(result, 4, Rf_ScalarReal(center.log_det_hessian_m));
    SET_VECTOR_ELT(result, 5, Rf_ScalarReal(center.gradient_norm_random_m));
    SET_VECTOR_ELT(result, 6, Rf_ScalarInteger(center.newton_iterations_m));
    SET_VECTOR_ELT(result, 7, Rf_ScalarLogical(center.converged_m));
    SET_VECTOR_ELT(result, 8, Rf_ScalarLogical(center.logdet_ok_m));
    SET_VECTOR_ELT(result, 9,
                   Rf_ScalarInteger(center.hessian_random_m.nonZeros()));
    const char *result_names[n_result] = {
        "value", "gradient", "u_hat", "joint_objective", "logdet_hessian",
        "random_gradient_norm", "iterations", "converged", "logdet_ok",
        "hessian_nonzeros"};
    for (int i = 0; i < n_result; ++i)
      SET_STRING_ELT(names, i, Rf_mkChar(result_names[i]));
    Rf_setAttrib(result, R_NamesSymbol, names);
    UNPROTECT(4);
    return result;
  } catch (const std::exception &e) {
    Rf_error("Quadra Poisson state evaluation failed: %s", e.what());
  }
  return R_NilValue;
}

extern "C" SEXP sdmTMB_quadra_poisson_state_covariance(
    SEXP pointer, SEXP fixed) {
  try {
    auto *state = static_cast<PersistentPoissonSpdeState *>(
        R_ExternalPtrAddr(pointer));
    if (state == nullptr || TYPEOF(fixed) != REALSXP)
      throw std::invalid_argument("invalid persistent Poisson state");
    std::vector<double> values(REAL(fixed),
                               REAL(fixed) + Rf_xlength(fixed));
    const auto covariance = state->FixedCovariance(values);
    const auto derived = state->DerivedInference(values, covariance);
    const auto random_effects =
        state->RandomEffectInference(values, covariance);
    return fixed_covariance_to_sexp(covariance, derived, random_effects);
  } catch (const std::exception &e) {
    Rf_error("Quadra covariance estimation failed: %s", e.what());
  }
  return R_NilValue;
}

extern "C" SEXP sdmTMB_quadra_poisson_state_prediction_uncertainty(
    SEXP pointer, SEXP x_theta, SEXP z_i, SEXP z_j, SEXP z_x,
    SEXP n_random) {
  try {
    auto *state = static_cast<PersistentPoissonSpdeState *>(
        R_ExternalPtrAddr(pointer));
    return state_prediction_inference(
        state, x_theta, z_i, z_j, z_x, n_random);
  } catch (const std::exception &e) {
    Rf_error("Quadra prediction uncertainty failed: %s", e.what());
  }
  return R_NilValue;
}

extern "C" void sdmTMB_quadra_poisson_st_iid_state_finalizer(SEXP pointer) {
  auto *state = static_cast<PersistentPoissonSpatiotemporalIidState *>(
      R_ExternalPtrAddr(pointer));
  if (state != nullptr) {
    delete state;
    R_ClearExternalPtr(pointer);
  }
}

extern "C" SEXP sdmTMB_quadra_poisson_st_iid_state_create(
    SEXP fixed, SEXP random, SEXP x, SEXP y, SEXP offset, SEXP weights,
    SEXP time, SEXP a_i, SEXP a_j, SEXP a_x, SEXP m0_i, SEXP m0_j,
    SEXP m0_x, SEXP m1_i, SEXP m1_j, SEXP m1_x, SEXP m2_i, SEXP m2_j,
    SEXP m2_x, SEXP temporal_model, SEXP separate_range, SEXP nb2,
    SEXP gaussian) {
  try {
    if (TYPEOF(time) != INTSXP || TYPEOF(temporal_model) != INTSXP ||
        Rf_xlength(temporal_model) != 1 ||
        TYPEOF(separate_range) != LGLSXP || Rf_xlength(separate_range) != 1 ||
        TYPEOF(nb2) != LGLSXP || Rf_xlength(nb2) != 1 ||
        TYPEOF(gaussian) != LGLSXP || Rf_xlength(gaussian) != 1)
      throw std::invalid_argument(
          "time index and temporal model must be integer");
    const int temporal_mode = INTEGER(temporal_model)[0];
    if (temporal_mode < 0 || temporal_mode > 2)
      throw std::invalid_argument("invalid temporal model");
    const bool use_ar1 = temporal_mode == 1;
    const bool use_rw = temporal_mode == 2;
    const bool use_separate_range = LOGICAL(separate_range)[0];
    const bool use_nb2 = LOGICAL(nb2)[0];
    const bool use_gaussian = LOGICAL(gaussian)[0];
    if (use_nb2 && use_gaussian)
      throw std::invalid_argument("NB2 and Gaussian cannot both be selected");
    const R_xlen_t random_size = Rf_xlength(random);
    const int n_time =
        Rf_xlength(time) ? *std::max_element(INTEGER(time),
                                             INTEGER(time) + Rf_xlength(time)) +
                              1
                         : 0;
    if (n_time < 1 || random_size % (n_time + 1) != 0)
      throw std::invalid_argument("invalid spatiotemporal random dimensions");
    const int n_node = static_cast<int>(random_size / (n_time + 1));
    SEXP dims = Rf_getAttrib(x, R_DimSymbol);
    if (TYPEOF(fixed) != REALSXP || TYPEOF(random) != REALSXP ||
        TYPEOF(x) != REALSXP || TYPEOF(y) != REALSXP ||
        TYPEOF(offset) != REALSXP || TYPEOF(weights) != REALSXP ||
        Rf_length(dims) != 2)
      throw std::invalid_argument("invalid spatiotemporal state input types");
    const int n = INTEGER(dims)[0], p = INTEGER(dims)[1];
    if (Rf_xlength(fixed) !=
            p + 3 + static_cast<int>(use_separate_range) +
                static_cast<int>(use_ar1) +
                static_cast<int>(use_nb2 || use_gaussian) ||
        Rf_xlength(y) != n ||
        Rf_xlength(offset) != n || Rf_xlength(weights) != n ||
        Rf_xlength(time) != n)
      throw std::invalid_argument("non-conformable spatiotemporal inputs");
    for (int i = 0; i < n; ++i) {
      if (INTEGER(time)[i] < 0 || INTEGER(time)[i] >= n_time)
        throw std::invalid_argument("time indices must be zero-based");
      if (!use_gaussian &&
          (REAL(y)[i] < 0.0 || REAL(y)[i] != std::floor(REAL(y)[i])))
        throw std::invalid_argument(
            "Poisson response must be nonnegative integers");
      if (!R_FINITE(REAL(weights)[i]) || REAL(weights)[i] < 0.0)
        throw std::invalid_argument("weights must be finite and nonnegative");
    }
    const SEXP numeric_triplets[] = {a_x, m0_x, m1_x, m2_x};
    const SEXP integer_triplets[] = {a_i, a_j, m0_i, m0_j,
                                     m1_i, m1_j, m2_i, m2_j};
    for (SEXP value : numeric_triplets)
      if (TYPEOF(value) != REALSXP)
        throw std::invalid_argument("triplet values must be double");
    for (SEXP index : integer_triplets)
      if (TYPEOF(index) != INTSXP)
        throw std::invalid_argument("triplet indices must be integer");
    if (Rf_xlength(a_i) != Rf_xlength(a_x) ||
        Rf_xlength(a_j) != Rf_xlength(a_x))
      throw std::invalid_argument("non-conformable projection triplet");
    for (R_xlen_t k = 0; k < Rf_xlength(a_x); ++k) {
      if (INTEGER(a_i)[k] < 0 || INTEGER(a_i)[k] >= n ||
          INTEGER(a_j)[k] < 0 || INTEGER(a_j)[k] >= n_node)
        throw std::invalid_argument("invalid projection triplet index");
    }
    const SEXP mi[] = {m0_i, m1_i, m2_i};
    const SEXP mj[] = {m0_j, m1_j, m2_j};
    const SEXP mx[] = {m0_x, m1_x, m2_x};
    for (int matrix_index = 0; matrix_index < 3; ++matrix_index) {
      if (Rf_xlength(mi[matrix_index]) != Rf_xlength(mx[matrix_index]) ||
          Rf_xlength(mj[matrix_index]) != Rf_xlength(mx[matrix_index]))
        throw std::invalid_argument("non-conformable SPDE triplet");
      for (R_xlen_t k = 0; k < Rf_xlength(mx[matrix_index]); ++k) {
        if (INTEGER(mi[matrix_index])[k] < 0 ||
            INTEGER(mi[matrix_index])[k] >= n_node ||
            INTEGER(mj[matrix_index])[k] < 0 ||
            INTEGER(mj[matrix_index])[k] >= n_node)
          throw std::invalid_argument("invalid SPDE triplet index");
      }
    }
    std::vector<double> fixed_values(REAL(fixed),
                                     REAL(fixed) + Rf_xlength(fixed));
    std::vector<double> random_values(REAL(random),
                                      REAL(random) + random_size);
    auto *state = new PersistentPoissonSpatiotemporalIidState(
        fixed_values, random_values, REAL(x), REAL(y), REAL(offset),
        REAL(weights), INTEGER(time), n, p, n_node, n_time, INTEGER(a_i),
        INTEGER(a_j),
        REAL(a_x), static_cast<int>(Rf_xlength(a_x)), INTEGER(m0_i),
        INTEGER(m0_j), REAL(m0_x), static_cast<int>(Rf_xlength(m0_x)),
        INTEGER(m1_i), INTEGER(m1_j), REAL(m1_x),
        static_cast<int>(Rf_xlength(m1_x)), INTEGER(m2_i), INTEGER(m2_j),
        REAL(m2_x), static_cast<int>(Rf_xlength(m2_x)), use_ar1, use_rw,
        use_separate_range, use_nb2, use_gaussian);
    SEXP pointer = PROTECT(R_MakeExternalPtr(state, R_NilValue, R_NilValue));
    R_RegisterCFinalizerEx(
        pointer, sdmTMB_quadra_poisson_st_iid_state_finalizer, TRUE);
    UNPROTECT(1);
    return pointer;
  } catch (const std::exception &e) {
    Rf_error("Quadra Poisson IID spatiotemporal state creation failed: %s",
             e.what());
  }
  return R_NilValue;
}

extern "C" SEXP sdmTMB_quadra_poisson_st_iid_state_evaluate(
    SEXP pointer, SEXP fixed, SEXP gradient_requested) {
  try {
    if (TYPEOF(pointer) != EXTPTRSXP || TYPEOF(fixed) != REALSXP ||
        TYPEOF(gradient_requested) != LGLSXP)
      throw std::invalid_argument("invalid persistent state input");
    auto *state = static_cast<PersistentPoissonSpatiotemporalIidState *>(
        R_ExternalPtrAddr(pointer));
    if (state == nullptr)
      throw std::runtime_error("persistent state has been finalized");
    const std::vector<double> fixed_values(
        REAL(fixed), REAL(fixed) + Rf_xlength(fixed));
    const PersistentPoissonResult fit =
        state->Evaluate(fixed_values, LOGICAL(gradient_requested)[0]);
    const auto &center = fit.objective;
    const int n_fixed = static_cast<int>(fit.gradient.size());
    const int n_random = static_cast<int>(center.u_hat_m.size());
    const int n_result = 10;
    SEXP result = PROTECT(Rf_allocVector(VECSXP, n_result));
    SEXP names = PROTECT(Rf_allocVector(STRSXP, n_result));
    SEXP gradient = PROTECT(Rf_allocVector(REALSXP, n_fixed));
    SEXP u_hat = PROTECT(Rf_allocVector(REALSXP, n_random));
    for (int j = 0; j < n_fixed; ++j)
      REAL(gradient)[j] = fit.gradient[static_cast<size_t>(j)];
    for (int j = 0; j < n_random; ++j)
      REAL(u_hat)[j] = center.u_hat_m[static_cast<size_t>(j)];
    SET_VECTOR_ELT(result, 0, Rf_ScalarReal(center.laplace_objective_m));
    SET_VECTOR_ELT(result, 1, gradient);
    SET_VECTOR_ELT(result, 2, u_hat);
    SET_VECTOR_ELT(result, 3, Rf_ScalarReal(center.joint_objective_m));
    SET_VECTOR_ELT(result, 4, Rf_ScalarReal(center.log_det_hessian_m));
    SET_VECTOR_ELT(result, 5, Rf_ScalarReal(center.gradient_norm_random_m));
    SET_VECTOR_ELT(result, 6, Rf_ScalarInteger(center.newton_iterations_m));
    SET_VECTOR_ELT(result, 7, Rf_ScalarLogical(center.converged_m));
    SET_VECTOR_ELT(result, 8, Rf_ScalarLogical(center.logdet_ok_m));
    SET_VECTOR_ELT(result, 9,
                   Rf_ScalarInteger(center.hessian_random_m.nonZeros()));
    const char *result_names[n_result] = {
        "value", "gradient", "u_hat", "joint_objective", "logdet_hessian",
        "random_gradient_norm", "iterations", "converged", "logdet_ok",
        "hessian_nonzeros"};
    for (int i = 0; i < n_result; ++i)
      SET_STRING_ELT(names, i, Rf_mkChar(result_names[i]));
    Rf_setAttrib(result, R_NamesSymbol, names);
    UNPROTECT(4);
    return result;
  } catch (const std::exception &e) {
    Rf_error("Quadra Poisson IID spatiotemporal evaluation failed: %s",
             e.what());
  }
  return R_NilValue;
}

extern "C" SEXP sdmTMB_quadra_poisson_st_iid_state_covariance(
    SEXP pointer, SEXP fixed) {
  try {
    auto *state = static_cast<PersistentPoissonSpatiotemporalIidState *>(
        R_ExternalPtrAddr(pointer));
    if (state == nullptr || TYPEOF(fixed) != REALSXP)
      throw std::invalid_argument("invalid persistent spatiotemporal state");
    std::vector<double> values(REAL(fixed),
                               REAL(fixed) + Rf_xlength(fixed));
    const auto covariance = state->FixedCovariance(values);
    const auto derived = state->DerivedInference(values, covariance);
    const auto random_effects =
        state->RandomEffectInference(values, covariance);
    return fixed_covariance_to_sexp(covariance, derived, random_effects);
  } catch (const std::exception &e) {
    Rf_error("Quadra spatiotemporal covariance failed: %s", e.what());
  }
  return R_NilValue;
}

extern "C" SEXP
sdmTMB_quadra_poisson_st_iid_state_prediction_uncertainty(
    SEXP pointer, SEXP x_theta, SEXP z_i, SEXP z_j, SEXP z_x,
    SEXP n_random) {
  try {
    auto *state = static_cast<PersistentPoissonSpatiotemporalIidState *>(
        R_ExternalPtrAddr(pointer));
    return state_prediction_inference(
        state, x_theta, z_i, z_j, z_x, n_random);
  } catch (const std::exception &e) {
    Rf_error("Quadra spatiotemporal prediction uncertainty failed: %s",
             e.what());
  }
  return R_NilValue;
}
