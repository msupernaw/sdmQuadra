#include <TMB.hpp>

template <class Type>
Type objective_function<Type>::operator()() {
  DATA_VECTOR(y);
  DATA_MATRIX(X);
  DATA_VECTOR(offset);
  DATA_VECTOR(weights);
  DATA_IVECTOR(time);
  DATA_SPARSE_MATRIX(A);
  DATA_SPARSE_MATRIX(M0);
  DATA_SPARSE_MATRIX(M1);
  DATA_SPARSE_MATRIX(M2);
  PARAMETER_VECTOR(beta);
  PARAMETER(log_tau_spatial);
  PARAMETER(log_tau_spatiotemporal);
  PARAMETER(log_kappa_spatial);
  PARAMETER(log_kappa_spatiotemporal);
  PARAMETER(ar1_phi);
  PARAMETER_VECTOR(spatial);
  PARAMETER_MATRIX(spatiotemporal);

  const Type kappa_s2 = exp(Type(2.0) * log_kappa_spatial);
  const Type kappa_st2 = exp(Type(2.0) * log_kappa_spatiotemporal);
  const Type rho = Type(2.0) / (Type(1.0) + exp(-ar1_phi)) - Type(1.0);
  const Type innovation_scale = sqrt(Type(1.0) - rho * rho);
  Eigen::SparseMatrix<Type> Q_s =
      kappa_s2 * kappa_s2 * M0 + Type(2.0) * kappa_s2 * M1 + M2;
  Eigen::SparseMatrix<Type> Q_st =
      kappa_st2 * kappa_st2 * M0 + Type(2.0) * kappa_st2 * M1 + M2;
  Type nll = density::SCALE(
      density::GMRF(Q_s), Type(1.0) / exp(log_tau_spatial))(spatial);
  vector<Type> first = spatiotemporal.col(0);
  nll += density::SCALE(
      density::GMRF(Q_st),
      Type(1.0) / exp(log_tau_spatiotemporal))(first);
  for (int t = 1; t < spatiotemporal.cols(); ++t) {
    vector<Type> innovation =
        (spatiotemporal.col(t) - rho * spatiotemporal.col(t - 1)) /
        innovation_scale;
    nll += density::SCALE(
        density::GMRF(Q_st),
        Type(1.0) / exp(log_tau_spatiotemporal))(innovation);
  }
  nll += Type((spatiotemporal.cols() - 1) * spatiotemporal.rows()) *
         log(innovation_scale);
  const vector<Type> projected_spatial = A * spatial;
  matrix<Type> projected_st(A.rows(), spatiotemporal.cols());
  for (int t = 0; t < spatiotemporal.cols(); ++t)
    projected_st.col(t) = A * spatiotemporal.col(t);
  for (int i = 0; i < y.size(); ++i) {
    Type eta =
        offset(i) + projected_spatial(i) + projected_st(i, time(i));
    for (int j = 0; j < X.cols(); ++j) eta += X(i, j) * beta(j);
    nll -= weights(i) * dpois(y(i), exp(eta), true);
  }
  return nll;
}
