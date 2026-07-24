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
  Eigen::SparseMatrix<Type> Q_spatial =
      Q * exp(Type(2.0) * log_tau_spatial);
  Eigen::SparseMatrix<Type> Q_spatiotemporal =
      Q * exp(Type(2.0) * log_tau_spatiotemporal);
  Type nll = density::GMRF(Q_spatial)(spatial);
  for (int t = 0; t < spatiotemporal.cols(); ++t) {
    vector<Type> field = spatiotemporal.col(t);
    nll += density::GMRF(Q_spatiotemporal)(field);
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
