#include <TMB.hpp>

template <class Type>
Type objective_function<Type>::operator()() {
  DATA_VECTOR(y);
  DATA_MATRIX(X);
  DATA_VECTOR(offset);
  DATA_VECTOR(weights);
  DATA_SPARSE_MATRIX(A);
  DATA_SPARSE_MATRIX(M0);
  DATA_SPARSE_MATRIX(M1);
  DATA_SPARSE_MATRIX(M2);
  PARAMETER_VECTOR(beta);
  PARAMETER(log_tau);
  PARAMETER(log_kappa);
  PARAMETER(log_phi);
  PARAMETER_VECTOR(field);

  const Type tau = exp(log_tau);
  const Type kappa2 = exp(Type(2.0) * log_kappa);
  const Type phi = exp(log_phi);
  Eigen::SparseMatrix<Type> Q =
      kappa2 * kappa2 * M0 + Type(2.0) * kappa2 * M1 + M2;
  Eigen::SparseMatrix<Type> scaled_Q = Q * tau * tau;
  Type nll = density::GMRF(scaled_Q)(field);
  vector<Type> projected = A * field;
  for (int i = 0; i < y.size(); ++i) {
    Type eta = offset(i) + projected(i);
    for (int j = 0; j < X.cols(); ++j) eta += X(i, j) * beta(j);
    const Type mean = exp(eta);
    const Type log_probability =
        lgamma(y(i) + phi) - lgamma(phi) - lgamma(y(i) + Type(1.0)) +
        phi * log(phi / (phi + mean)) +
        y(i) * log(mean / (phi + mean));
    nll -= weights(i) * log_probability;
  }
  return nll;
}
