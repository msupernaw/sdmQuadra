#include <TMB.hpp>

template <class Type>
Type objective_function<Type>::operator()() {
  DATA_VECTOR(y);
  DATA_MATRIX(X);
  DATA_VECTOR(offset);
  DATA_IVECTOR(time);
  DATA_SPARSE_MATRIX(A);
  DATA_SPARSE_MATRIX(M0);
  DATA_SPARSE_MATRIX(M1);
  DATA_SPARSE_MATRIX(M2);
  PARAMETER_VECTOR(beta);
  PARAMETER(log_tau_spatial);
  PARAMETER(log_tau_spatiotemporal);
  PARAMETER(log_kappa);
  PARAMETER_VECTOR(spatial);
  PARAMETER_MATRIX(spatiotemporal);

  const Type kappa2 = exp(Type(2.0) * log_kappa);
  Eigen::SparseMatrix<Type> Q =
      kappa2 * kappa2 * M0 + Type(2.0) * kappa2 * M1 + M2;
  Type nll = density::SCALE(
      density::GMRF(Q), Type(1.0) / exp(log_tau_spatial))(spatial);
  vector<Type> first = spatiotemporal.col(0);
  nll += density::SCALE(
      density::GMRF(Q), Type(1.0) / exp(log_tau_spatiotemporal))(first);
  for (int t = 1; t < spatiotemporal.cols(); ++t) {
    vector<Type> difference =
        spatiotemporal.col(t) - spatiotemporal.col(t - 1);
    nll += density::SCALE(
        density::GMRF(Q), Type(1.0) / exp(log_tau_spatiotemporal))(
        difference);
  }

  const vector<Type> projected_spatial = A * spatial;
  matrix<Type> projected_st(A.rows(), spatiotemporal.cols());
  for (int t = 0; t < spatiotemporal.cols(); ++t) {
    vector<Type> field = spatiotemporal.col(t);
    projected_st.col(t) = A * field;
  }
  for (int i = 0; i < y.size(); ++i) {
    Type eta =
        offset(i) + projected_spatial(i) + projected_st(i, time(i));
    for (int j = 0; j < X.cols(); ++j) eta += X(i, j) * beta(j);
    nll -= dpois(y(i), exp(eta), true);
  }
  return nll;
}
