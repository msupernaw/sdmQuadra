#include <TMB.hpp>

template <class Type>
Type objective_function<Type>::operator()() {
  DATA_VECTOR(y);
  DATA_MATRIX(X);
  DATA_VECTOR(offset);
  DATA_IVECTOR(group);

  PARAMETER_VECTOR(beta);
  PARAMETER(log_sigma_obs);
  PARAMETER(log_sigma_group);
  PARAMETER_VECTOR(u);

  const Type sigma_obs = exp(log_sigma_obs);
  const Type sigma_group = exp(log_sigma_group);
  Type nll = Type(0.0);

  for (int i = 0; i < y.size(); ++i) {
    Type eta = offset(i) + u(group(i));
    for (int j = 0; j < X.cols(); ++j) {
      eta += X(i, j) * beta(j);
    }
    nll -= dnorm(y(i), eta, sigma_obs, true);
  }

  for (int g = 0; g < u.size(); ++g) {
    nll -= dnorm(u(g), Type(0.0), sigma_group, true);
  }

  REPORT(u);
  return nll;
}
