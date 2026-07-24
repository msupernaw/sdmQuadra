#include <TMB.hpp>

template <class Type>
Type objective_function<Type>::operator()() {
  DATA_VECTOR(y);
  DATA_MATRIX(X);
  DATA_VECTOR(offset);
  DATA_SPARSE_MATRIX(A);
  DATA_SPARSE_MATRIX(M0);
  DATA_SPARSE_MATRIX(M1);
  DATA_SPARSE_MATRIX(M2);

  PARAMETER_VECTOR(beta);
  PARAMETER(log_sigma_obs);
  PARAMETER(log_tau);
  PARAMETER(log_kappa);
  PARAMETER_VECTOR(field);

  const Type sigma_obs = exp(log_sigma_obs);
  const Type tau = exp(log_tau);
  const Type kappa2 = exp(Type(2.0) * log_kappa);
  Eigen::SparseMatrix<Type> Q =
      kappa2 * kappa2 * M0 + Type(2.0) * kappa2 * M1 + M2;
  Eigen::SparseMatrix<Type> scaled_Q = Q * tau * tau;
  Type nll = density::GMRF(scaled_Q)(field);
  vector<Type> projected = A * field;

  for (int i = 0; i < y.size(); ++i) {
    Type eta = offset(i) + projected(i);
    for (int j = 0; j < X.cols(); ++j) {
      eta += X(i, j) * beta(j);
    }
    nll -= dnorm(y(i), eta, sigma_obs, true);
  }

  REPORT(field);
  REPORT(projected);
  return nll;
}
