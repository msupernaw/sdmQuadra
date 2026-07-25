test_that("Quadra Gaussian objective and gradient match analytic values", {
  X <- cbind(1, c(-1, 0, 2, 3))
  y <- c(0.5, 1.25, 2.5, 4)
  offset <- c(0.1, 0, -0.2, 0.3)
  beta <- c(0.75, 0.6)

  obj <- sdmTMB:::.make_quadra_gaussian_object(
    X = X, y = y, offset = offset, start = beta
  )

  eta <- as.vector(X %*% beta + offset)
  expected_value <- sum(0.5 * (y - eta)^2 + 0.5 * log(2 * pi))
  expected_gradient <- as.vector(crossprod(X, eta - y))

  expect_equal(obj$fn(beta), expected_value, tolerance = 1e-10)
  expect_equal(obj$gr(beta), expected_gradient, tolerance = 1e-10)
  expect_equal(obj$report(beta)$eta_i, eta)
})

test_that("Quadra Gaussian object works with nlminb", {
  X <- cbind(1, seq(-1, 1, length.out = 20))
  y <- as.vector(X %*% c(1.5, -0.75))
  obj <- sdmTMB:::.make_quadra_gaussian_object(X, y)

  fit <- stats::nlminb(obj$par, obj$fn, obj$gr)

  expect_equal(fit$par, c(1.5, -0.75), tolerance = 1e-7)
  expect_equal(obj$env$last.par.best, fit$par, tolerance = 1e-7)
})

gaussian_random_intercept_marginal_nll <- function(
    fixed, X, y, group, offset = numeric(length(y))
) {
  beta <- fixed[seq_len(ncol(X))]
  sigma_obs <- exp(fixed[ncol(X) + 1L])
  sigma_group <- exp(fixed[ncol(X) + 2L])
  residual <- y - as.vector(X %*% beta + offset)
  nll <- 0

  for (g in unique(group)) {
    r <- residual[group == g]
    covariance <- diag(sigma_obs^2, length(r)) + sigma_group^2
    root <- chol(covariance)
    nll <- nll + sum(log(diag(root))) +
      0.5 * sum(backsolve(root, r, transpose = TRUE)^2) +
      0.5 * length(r) * log(2 * pi)
  }

  nll
}

test_that("Quadra random-intercept Laplace objective is exact for Gaussian data", {
  group <- rep(letters[1:4], each = 3)
  x <- seq(-1, 1, length.out = length(group))
  X <- cbind(1, x)
  y <- 0.8 - 0.4 * x +
    rep(c(-0.5, 0.2, 0.7, -0.1), each = 3) +
    rep(c(-0.1, 0, 0.1), 4)
  fixed <- c(0.7, -0.3, log(0.35), log(0.6))

  obj <- sdmTMB:::.make_quadra_gaussian_random_intercept_object(
    X, y, group, start = fixed
  )

  expected <- gaussian_random_intercept_marginal_nll(fixed, X, y, group)
  expect_equal(obj$fn(fixed), expected, tolerance = 1e-8)

  h <- 1e-5
  expected_gradient <- vapply(seq_along(fixed), function(i) {
    plus <- minus <- fixed
    plus[i] <- plus[i] + h
    minus[i] <- minus[i] - h
    (
      gaussian_random_intercept_marginal_nll(plus, X, y, group) -
        gaussian_random_intercept_marginal_nll(minus, X, y, group)
    ) / (2 * h)
  }, numeric(1))

  expect_equal(obj$gr(fixed), expected_gradient, tolerance = 1e-5)
  expect_lt(obj$env$last.eval$random_gradient_norm, 1e-8)
  expect_true(obj$env$last.eval$converged)
  expect_true(obj$env$last.eval$logdet_ok)
})

gaussian_sparse_field_marginal_nll <- function(
    fixed, X, y, node, Q, offset = numeric(length(y))
) {
  beta <- fixed[seq_len(ncol(X))]
  sigma_obs <- exp(fixed[ncol(X) + 1L])
  tau <- exp(fixed[ncol(X) + 2L])
  Z <- matrix(0, nrow = length(y), ncol = nrow(Q))
  Z[cbind(seq_along(node), node)] <- 1
  field_covariance <- solve(tau^2 * as.matrix(Q))
  covariance <- sigma_obs^2 * diag(length(y)) +
    Z %*% field_covariance %*% t(Z)
  residual <- y - as.vector(X %*% beta + offset)
  root <- chol(covariance)

  sum(log(diag(root))) +
    0.5 * sum(backsolve(root, residual, transpose = TRUE)^2) +
    0.5 * length(y) * log(2 * pi)
}

test_that("Quadra sparse-field Laplace objective matches exact marginal", {
  n_node <- 8L
  difference <- diff(diag(n_node))
  Q <- Matrix::Matrix(crossprod(difference) + 0.35 * diag(n_node), sparse = TRUE)
  node <- rep(seq_len(n_node), each = 2)
  x <- seq(-1, 1, length.out = length(node))
  X <- cbind(1, x)
  field <- c(-0.6, -0.4, -0.1, 0.25, 0.5, 0.35, 0.1, -0.2)
  y <- 1.2 - 0.45 * x + field[node] + rep(c(-0.08, 0.08), n_node)
  fixed <- c(1, -0.3, log(0.25), log(1.1))

  obj <- sdmTMB:::.make_quadra_gaussian_sparse_field_object(
    X, y, node, Q, start = fixed
  )

  expected <- gaussian_sparse_field_marginal_nll(fixed, X, y, node, Q)
  expect_equal(obj$fn(fixed), expected, tolerance = 1e-8)

  h <- 1e-5
  expected_gradient <- vapply(seq_along(fixed), function(i) {
    plus <- minus <- fixed
    plus[i] <- plus[i] + h
    minus[i] <- minus[i] - h
    (
      gaussian_sparse_field_marginal_nll(plus, X, y, node, Q) -
        gaussian_sparse_field_marginal_nll(minus, X, y, node, Q)
    ) / (2 * h)
  }, numeric(1))

  expect_equal(obj$gr(fixed), expected_gradient, tolerance = 1e-5)
  expect_lt(obj$env$last.eval$random_gradient_norm, 1e-8)
  expect_true(obj$env$last.eval$converged)
  expect_true(obj$env$last.eval$logdet_ok)
  expect_gt(obj$env$last.eval$hessian_nonzeros, n_node)
})

gaussian_projected_field_marginal_nll <- function(
    fixed, X, y, A, Q, offset = numeric(length(y))
) {
  beta <- fixed[seq_len(ncol(X))]
  sigma_obs <- exp(fixed[ncol(X) + 1L])
  tau <- exp(fixed[ncol(X) + 2L])
  covariance <- sigma_obs^2 * diag(length(y)) +
    as.matrix(A %*% solve(tau^2 * as.matrix(Q)) %*% Matrix::t(A))
  residual <- y - as.vector(X %*% beta + offset)
  root <- chol(covariance)
  sum(log(diag(root))) +
    0.5 * sum(backsolve(root, residual, transpose = TRUE)^2) +
    0.5 * length(y) * log(2 * pi)
}

test_that("Quadra projected sparse field matches exact marginal", {
  n_node <- 6L
  difference <- diff(diag(n_node))
  Q <- Matrix::Matrix(crossprod(difference) + 0.4 * diag(n_node), sparse = TRUE)
  locations <- seq(1, n_node, length.out = 11)
  left <- pmin(floor(locations), n_node - 1L)
  weight_right <- locations - left
  A <- Matrix::sparseMatrix(
    i = rep(seq_along(locations), each = 2L),
    j = as.vector(rbind(left, left + 1L)),
    x = as.vector(rbind(1 - weight_right, weight_right)),
    dims = c(length(locations), n_node)
  )
  x <- seq(-1, 1, length.out = nrow(A))
  X <- cbind(1, x)
  field <- c(-0.5, -0.3, 0.05, 0.45, 0.3, -0.15)
  y <- 0.9 - 0.35 * x + as.vector(A %*% field) +
    rep(c(-0.06, 0.06), length.out = nrow(A))
  fixed <- c(0.8, -0.25, log(0.22), log(1.15))

  obj <- sdmTMB:::.make_quadra_gaussian_projected_field_object(
    X, y, A, Q, start = fixed
  )
  expected <- gaussian_projected_field_marginal_nll(fixed, X, y, A, Q)
  expect_equal(obj$fn(fixed), expected, tolerance = 1e-8)

  h <- 1e-5
  expected_gradient <- vapply(seq_along(fixed), function(i) {
    plus <- minus <- fixed
    plus[i] <- plus[i] + h
    minus[i] <- minus[i] - h
    (
      gaussian_projected_field_marginal_nll(plus, X, y, A, Q) -
        gaussian_projected_field_marginal_nll(minus, X, y, A, Q)
    ) / (2 * h)
  }, numeric(1))
  expect_equal(obj$gr(fixed), expected_gradient, tolerance = 1e-5)
  expect_lt(obj$env$last.eval$random_gradient_norm, 1e-8)
  expect_true(obj$env$last.eval$converged)
  expect_true(obj$env$last.eval$logdet_ok)
  expect_gt(obj$env$last.eval$hessian_nonzeros, n_node)
})

test_that("Quadra dynamic SPDE kappa matches exact marginal", {
  n_node <- 6L
  difference <- diff(diag(n_node))
  laplacian <- crossprod(difference)
  spde <- list(
    c0 = Matrix::Diagonal(n_node),
    g1 = Matrix::Matrix(laplacian, sparse = TRUE),
    g2 = Matrix::Matrix(laplacian %*% laplacian + 0.25 * diag(n_node),
      sparse = TRUE
    )
  )
  locations <- seq(1, n_node, length.out = 12)
  left <- pmin(floor(locations), n_node - 1L)
  weight_right <- locations - left
  A <- Matrix::sparseMatrix(
    i = rep(seq_along(locations), each = 2L),
    j = as.vector(rbind(left, left + 1L)),
    x = as.vector(rbind(1 - weight_right, weight_right)),
    dims = c(length(locations), n_node)
  )
  x <- seq(-1, 1, length.out = nrow(A))
  X <- cbind(1, x)
  field <- c(-0.45, -0.25, 0.1, 0.4, 0.25, -0.1)
  y <- 1 - 0.3 * x + as.vector(A %*% field) +
    rep(c(-0.05, 0.05), length.out = nrow(A))
  fixed <- c(0.9, -0.2, log(0.2), log(1.1), log(0.7))

  obj <- sdmTMB:::.make_quadra_gaussian_spde_object(
    X, y, A, spde, start = fixed
  )
  Q <- obj$build_precision(fixed[length(fixed)])
  expected <- gaussian_projected_field_marginal_nll(
    fixed[-length(fixed)], X, y, A, Q
  )
  expect_equal(obj$fn(fixed), expected, tolerance = 1e-8)

  h <- 1e-5
  expected_gradient <- vapply(seq_along(fixed), function(i) {
    marginal <- function(par) {
      kappa2 <- exp(2 * par[length(par)])
      Q <- kappa2^2 * spde$c0 + 2 * kappa2 * spde$g1 + spde$g2
      gaussian_projected_field_marginal_nll(
        par[-length(par)], X, y, A, Q
      )
    }
    plus <- minus <- fixed
    plus[i] <- plus[i] + h
    minus[i] <- minus[i] - h
    (marginal(plus) - marginal(minus)) / (2 * h)
  }, numeric(1))

  expect_equal(obj$gr(fixed), expected_gradient, tolerance = 1e-5)
  expect_lt(obj$env$last.eval$random_gradient_norm, 1e-7)
  expect_true(obj$env$last.eval$converged)
  expect_true(obj$env$last.eval$logdet_ok)
})

test_that("Quadra field and prediction uncertainty match TMB joint precision", {
  fixture <- testthat::test_path(
    "..", "fixtures", paste0("quadra_spde_tmb", .Platform$dynlib.ext)
  )
  skip_if_not(file.exists(fixture))

  n_node <- 6L
  difference <- diff(diag(n_node))
  laplacian <- crossprod(difference)
  spde <- list(
    c0 = Matrix::Diagonal(n_node),
    g1 = Matrix::Matrix(laplacian, sparse = TRUE),
    g2 = Matrix::Matrix(
      laplacian %*% laplacian + 0.25 * diag(n_node), sparse = TRUE
    )
  )
  locations <- seq(1, n_node, length.out = 30)
  left <- pmin(floor(locations), n_node - 1L)
  weight_right <- locations - left
  A <- Matrix::sparseMatrix(
    i = rep(seq_along(locations), each = 2L),
    j = as.vector(rbind(left, left + 1L)),
    x = as.vector(rbind(1 - weight_right, weight_right)),
    dims = c(length(locations), n_node)
  )
  x <- seq(-1, 1, length.out = nrow(A))
  X <- cbind(1, x)
  field <- c(-0.3, -0.15, 0.05, 0.25, 0.15, -0.05)
  y <- as.numeric(
    0.7 - 0.4 * x + A %*% field + rep(c(-0.1, 0.1), 15)
  )

  quadra <- sdmTMB:::.make_quadra_poisson_spde_object(
    X, y, A, spde,
    start = c(0.5, -0.3, 0, log(0.8), log(0.2)),
    gaussian = TRUE
  )
  quadra_fit <- stats::nlminb(quadra$par, quadra$fn, quadra$gr)
  quadra_covariance <- quadra$covariance(quadra_fit$par)
  quadra_prediction <- quadra$prediction_uncertainty(X, A)
  expect_true(isTRUE(quadra_covariance$success))
  expect_true(isTRUE(quadra_prediction$success))

  dyn.load(fixture)
  on.exit(dyn.unload(fixture), add = TRUE)
  tmb <- TMB::MakeADFun(
    data = list(
      y = y, X = X, offset = numeric(length(y)), A = A,
      M0 = spde$c0, M1 = spde$g1, M2 = spde$g2
    ),
    parameters = list(
      beta = c(0.5, -0.3), log_sigma_obs = log(0.2),
      log_tau = 0, log_kappa = log(0.8), field = numeric(n_node)
    ),
    random = "field", DLL = "quadra_spde_tmb", silent = TRUE
  )
  tmb_fit <- stats::nlminb(tmb$par, tmb$fn, tmb$gr)
  tmb_report <- TMB::sdreport(
    tmb, par.fixed = tmb_fit$par, getJointPrecision = TRUE
  )
  joint_precision <- as.matrix(tmb_report$jointPrecision)
  joint_covariance <- solve(joint_precision)
  n_fixed <- length(tmb_fit$par)
  random_index <- n_fixed + seq_len(n_node)
  conditional_covariance <- solve(
    joint_precision[random_index, random_index, drop = FALSE]
  )
  marginal_field_variance <- diag(joint_covariance)[random_index]
  conditional_field_variance <- diag(conditional_covariance)

  prediction_design <- cbind(
    X, matrix(0, nrow(X), n_fixed - ncol(X)), as.matrix(A)
  )
  marginal_prediction_variance <- rowSums(
    (prediction_design %*% joint_covariance) * prediction_design
  )
  conditional_prediction_variance <- rowSums(
    (as.matrix(A) %*% conditional_covariance) * as.matrix(A)
  )

  expect_lt(max(abs(
    quadra_covariance$random_effects$conditional_variance -
      conditional_field_variance
  )), 1e-7)
  expect_lt(max(abs(
    quadra_covariance$random_effects$marginal_variance -
      marginal_field_variance
  )), 1e-7)
  expect_lt(max(abs(
    quadra_covariance$random_effects$parameter_variance -
      (marginal_field_variance - conditional_field_variance)
  )), 1e-7)
  expect_lt(max(abs(
    quadra_prediction$conditional_variance -
      conditional_prediction_variance
  )), 1e-7)
  expect_lt(max(abs(
    quadra_prediction$marginal_variance -
      marginal_prediction_variance
  )), 1e-7)
  expect_lt(max(abs(
    quadra_prediction$parameter_variance -
      (marginal_prediction_variance - conditional_prediction_variance)
  )), 1e-7)
})

test_that("Quadra Poisson uncertainty matches TMB joint precision", {
  fixture <- testthat::test_path(
    "..", "fixtures",
    paste0("quadra_poisson_spde_tmb", .Platform$dynlib.ext)
  )
  skip_if_not(file.exists(fixture))

  set.seed(1)
  n_node <- 30L
  rotation <- qr.Q(qr(matrix(stats::rnorm(n_node^2), n_node)))
  eigenvalues <- exp(seq(-2, 2, length.out = n_node))
  gradient <- rotation %*% diag(eigenvalues) %*% t(rotation)
  spde <- list(
    c0 = Matrix::Diagonal(n_node),
    g1 = Matrix::Matrix(gradient, sparse = TRUE),
    g2 = Matrix::Matrix(
      gradient %*% gradient + 0.05 * diag(n_node), sparse = TRUE
    )
  )
  precision <- spde$c0 + 2 * spde$g1 + spde$g2
  field <- as.numeric(
    solve(chol(as.matrix(precision)), stats::rnorm(n_node))
  ) * 1.2
  n <- 600L
  node <- rep(seq_len(n_node), each = n / n_node)
  A <- Matrix::sparseMatrix(
    i = seq_len(n), j = node, x = 1, dims = c(n, n_node)
  )
  x <- stats::rnorm(n)
  X <- cbind(1, x)
  eta <- 1 + 0.5 * x + field[node]
  y <- stats::rpois(n, exp(eta))

  quadra <- sdmTMB:::.make_quadra_poisson_spde_object(
    X, y, A, spde, start = c(0.8, 0.4, -0.2, 0)
  )
  quadra_fit <- stats::nlminb(
    quadra$par, quadra$fn, quadra$gr,
    lower = rep(-10, 4), upper = rep(10, 4),
    control = list(iter.max = 200, eval.max = 300)
  )
  quadra_covariance <- quadra$covariance(quadra_fit$par)
  prediction_rows <- seq_len(20)
  quadra_prediction <- quadra$prediction_uncertainty(
    X[prediction_rows, , drop = FALSE],
    A[prediction_rows, , drop = FALSE]
  )
  expect_true(isTRUE(quadra_covariance$success))
  expect_true(isTRUE(quadra_prediction$success))

  dyn.load(fixture)
  on.exit(dyn.unload(fixture), add = TRUE)
  tmb <- TMB::MakeADFun(
    data = list(
      y = y, X = X, offset = numeric(n), A = A,
      M0 = spde$c0, M1 = spde$g1, M2 = spde$g2
    ),
    parameters = list(
      beta = quadra_fit$par[1:2],
      log_tau = quadra_fit$par[3],
      log_kappa = quadra_fit$par[4],
      field = numeric(n_node)
    ),
    random = "field", DLL = "quadra_poisson_spde_tmb", silent = TRUE
  )
  tmb_fit <- stats::nlminb(tmb$par, tmb$fn, tmb$gr)
  expect_equal(unname(tmb_fit$par), quadra_fit$par, tolerance = 5e-6)
  tmb_report <- TMB::sdreport(
    tmb, par.fixed = tmb_fit$par, getJointPrecision = TRUE
  )
  joint_precision <- as.matrix(tmb_report$jointPrecision)
  joint_covariance <- solve(joint_precision)
  n_fixed <- length(tmb_fit$par)
  random_index <- n_fixed + seq_len(n_node)
  conditional_covariance <- solve(
    joint_precision[random_index, random_index, drop = FALSE]
  )
  marginal_field_variance <- diag(joint_covariance)[random_index]
  conditional_field_variance <- diag(conditional_covariance)
  prediction_A <- as.matrix(A[prediction_rows, , drop = FALSE])
  prediction_design <- cbind(
    X[prediction_rows, , drop = FALSE],
    matrix(0, length(prediction_rows), n_fixed - ncol(X)),
    prediction_A
  )
  marginal_prediction_variance <- rowSums(
    (prediction_design %*% joint_covariance) * prediction_design
  )
  conditional_prediction_variance <- rowSums(
    (prediction_A %*% conditional_covariance) * prediction_A
  )

  expect_lt(max(abs(
    quadra_covariance$random_effects$conditional_variance -
      conditional_field_variance
  )), 5e-7)
  expect_lt(max(abs(
    quadra_covariance$random_effects$marginal_variance -
      marginal_field_variance
  )), 5e-7)
  expect_lt(max(abs(
    quadra_covariance$random_effects$parameter_variance -
      (marginal_field_variance - conditional_field_variance)
  )), 5e-7)
  expect_lt(max(abs(
    quadra_prediction$conditional_variance -
      conditional_prediction_variance
  )), 5e-7)
  expect_lt(max(abs(
    quadra_prediction$marginal_variance -
      marginal_prediction_variance
  )), 5e-7)
  expect_lt(max(abs(
    quadra_prediction$parameter_variance -
      (marginal_prediction_variance - conditional_prediction_variance)
  )), 5e-7)
})

test_that("Quadra AR1 spatiotemporal uncertainty matches TMB", {
  fixture <- testthat::test_path(
    "..", "fixtures",
    paste0("quadra_gaussian_st_separate_ar1_tmb", .Platform$dynlib.ext)
  )
  skip_if_not(file.exists(fixture))

  set.seed(7)
  n_node <- 8L
  n_time <- 4L
  difference <- diff(diag(n_node))
  laplacian <- crossprod(difference)
  spde <- list(
    c0 = Matrix::Diagonal(n_node),
    g1 = Matrix::Matrix(laplacian, sparse = TRUE),
    g2 = Matrix::Matrix(
      laplacian %*% laplacian + 0.2 * diag(n_node), sparse = TRUE
    )
  )
  n <- 320L
  node <- rep(seq_len(n_node), length.out = n)
  time <- rep(rep(0:(n_time - 1L), each = n_node), length.out = n)
  A <- Matrix::sparseMatrix(
    i = seq_len(n), j = node, x = 1, dims = c(n, n_node)
  )
  x <- stats::rnorm(n)
  X <- cbind(1, x)
  spatial <- stats::rnorm(n_node, 0, 0.35)
  spatiotemporal <- matrix(
    stats::rnorm(n_node * n_time, 0, 0.25), n_node, n_time
  )
  y <- 0.8 - 0.3 * x + spatial[node] +
    spatiotemporal[cbind(node, time + 1L)] + stats::rnorm(n, 0, 0.3)
  start <- c(0.6, -0.2, 0, 0, 0, -0.2, 0.2, log(0.3))

  quadra <- sdmTMB:::.make_quadra_poisson_spatiotemporal_iid_object(
    X, y, A, spde, time, start = start, ar1 = TRUE,
    share_range = FALSE, gaussian = TRUE
  )
  quadra_fit <- stats::nlminb(
    quadra$par, quadra$fn, quadra$gr,
    lower = rep(-10, length(start)), upper = rep(10, length(start)),
    control = list(iter.max = 200, eval.max = 300)
  )
  quadra_covariance <- quadra$covariance(quadra_fit$par)
  prediction_rows <- seq_len(20)
  quadra_prediction <- quadra$prediction_uncertainty(
    X[prediction_rows, , drop = FALSE],
    A[prediction_rows, , drop = FALSE], time[prediction_rows]
  )
  expect_true(isTRUE(quadra_covariance$success))
  expect_true(isTRUE(quadra_prediction$success))

  dyn.load(fixture)
  on.exit(dyn.unload(fixture), add = TRUE)
  tmb <- TMB::MakeADFun(
    data = list(
      y = y, X = X, offset = numeric(n), weights = rep(1, n),
      time = time, A = A, M0 = spde$c0, M1 = spde$g1, M2 = spde$g2
    ),
    parameters = list(
      beta = quadra_fit$par[1:2],
      log_tau_spatial = quadra_fit$par[3],
      log_tau_spatiotemporal = quadra_fit$par[4],
      log_kappa_spatial = quadra_fit$par[5],
      log_kappa_spatiotemporal = quadra_fit$par[6],
      ar1_phi = quadra_fit$par[7], log_sigma = quadra_fit$par[8],
      spatial = numeric(n_node),
      spatiotemporal = matrix(0, n_node, n_time)
    ),
    random = c("spatial", "spatiotemporal"),
    DLL = "quadra_gaussian_st_separate_ar1_tmb", silent = TRUE
  )
  tmb_fit <- stats::nlminb(tmb$par, tmb$fn, tmb$gr)
  expect_equal(unname(tmb_fit$par), quadra_fit$par, tolerance = 5e-6)
  tmb_report <- TMB::sdreport(
    tmb, par.fixed = tmb_fit$par, getJointPrecision = TRUE
  )
  joint_precision <- as.matrix(tmb_report$jointPrecision)
  joint_covariance <- solve(joint_precision)
  n_fixed <- length(tmb_fit$par)
  n_random <- n_node * (n_time + 1L)
  random_index <- n_fixed + seq_len(n_random)
  conditional_covariance <- solve(
    joint_precision[random_index, random_index, drop = FALSE]
  )
  prediction_A <- as.matrix(A[prediction_rows, , drop = FALSE])
  Z <- matrix(0, length(prediction_rows), n_random)
  Z[, seq_len(n_node)] <- prediction_A
  for (i in seq_along(prediction_rows)) {
    temporal_columns <- n_node +
      time[prediction_rows[i]] * n_node + seq_len(n_node)
    Z[i, temporal_columns] <- prediction_A[i, ]
  }
  prediction_design <- cbind(
    X[prediction_rows, , drop = FALSE],
    matrix(0, length(prediction_rows), n_fixed - ncol(X)), Z
  )
  marginal_prediction_variance <- rowSums(
    (prediction_design %*% joint_covariance) * prediction_design
  )
  conditional_prediction_variance <- rowSums(
    (Z %*% conditional_covariance) * Z
  )
  marginal_field_variance <- diag(joint_covariance)[random_index]
  conditional_field_variance <- diag(conditional_covariance)

  expect_lt(max(abs(
    quadra_covariance$random_effects$conditional_variance -
      conditional_field_variance
  )), 5e-7)
  expect_lt(max(abs(
    quadra_covariance$random_effects$marginal_variance -
      marginal_field_variance
  )), 5e-7)
  expect_lt(max(abs(
    quadra_covariance$random_effects$parameter_variance -
      (marginal_field_variance - conditional_field_variance)
  )), 5e-7)
  expect_lt(max(abs(
    quadra_prediction$conditional_variance -
      conditional_prediction_variance
  )), 5e-7)
  expect_lt(max(abs(
    quadra_prediction$marginal_variance -
      marginal_prediction_variance
  )), 5e-7)
  expect_lt(max(abs(
    quadra_prediction$parameter_variance -
      (marginal_prediction_variance - conditional_prediction_variance)
  )), 5e-7)
})

test_that("sdmTMB exposes the experimental Quadra backend", {
  fit <- sdmTMB(
    log(density + 0.1) ~ depth_scaled,
    data = pcod_2011,
    mesh = pcod_mesh_2011,
    family = gaussian(),
    backend = "quadra",
    control = sdmTMBcontrol(
      iter.max = 100, eval.max = 200, getsd = TRUE
    )
  )

  expect_s3_class(fit, "sdmTMB_quadra")
  expect_identical(fit$backend, "quadra")
  expect_equal(length(coef(fit)), 2L)
  expect_equal(length(fitted(fit)), nrow(pcod_2011))
  expect_equal(nrow(predict(fit)), nrow(pcod_2011))
  expect_equal(nobs(fit), nrow(pcod_2011))
  expect_equal(fit$model$convergence, 0L)
  expect_true(fit$hessian_diagnostics$positive_definite)
  expect_true(is.finite(fit$hessian_diagnostics$condition_number))
  expect_true(all(is.finite(vcov(fit, complete = TRUE))))
  expect_equal(fit$vcov, t(fit$vcov), tolerance = 1e-12)
  expect_equal(
    dim(vcov(fit)), c(ncol(fit$X), ncol(fit$X))
  )
  expect_identical(
    fit$derived_uncertainty$term,
    c("spatial_precision_scale", "spatial_range", "sigma")
  )
  expect_true(all(is.finite(fit$derived_uncertainty$estimate)))
  expect_true(all(is.finite(fit$derived_uncertainty$std.error)))
  expect_true(isTRUE(fit$random_effect_uncertainty$success))
  expect_length(fit$field_se, ncol(fit$spde$A_st))
  expect_true(all(is.finite(fit$field_se) & fit$field_se >= 0))
  expect_equal(
    fit$random_effect_uncertainty$marginal_variance,
    fit$random_effect_uncertainty$conditional_variance +
      fit$random_effect_uncertainty$parameter_variance,
    tolerance = 1e-12
  )
  expect_equal(
    fit$field_se^2,
    fit$random_effect_uncertainty$marginal_variance,
    tolerance = 1e-12
  )

  new_prediction <- predict(
    fit, newdata = pcod_2011[1:3, ], type = "response", se_fit = TRUE
  )
  expect_equal(nrow(new_prediction), 3L)
  expect_true(all(is.finite(new_prediction$est)))
  expect_true(all(is.finite(new_prediction$est_se)))
  expect_true(all(new_prediction$est_se >= 0))
  expect_true(all(new_prediction$lwr <= new_prediction$est))
  expect_true(all(new_prediction$upr >= new_prediction$est))
  expect_identical(attr(new_prediction, "level"), 0.95)
  expect_true(isTRUE(attr(new_prediction, "quadra_uncertainty")$success))

  fitted_prediction <- predict(fit, se_fit = TRUE, level = 0.8)
  expect_equal(nrow(fitted_prediction), nrow(pcod_2011))
  expect_true(all(is.finite(fitted_prediction$est_se)))
  expect_identical(attr(fitted_prediction, "level"), 0.8)
  expect_equal(
    fitted_prediction$upr - fitted_prediction$est,
    fitted_prediction$est - fitted_prediction$lwr,
    tolerance = 1e-12
  )
  expect_error(predict(fit, level = 1), "strictly between")
  expect_equal(
    attr(fitted_prediction, "quadra_uncertainty")$marginal_variance,
    attr(fitted_prediction, "quadra_uncertainty")$conditional_variance +
      attr(fitted_prediction, "quadra_uncertainty")$parameter_variance,
    tolerance = 1e-12
  )

  poisson_fit <- sdmTMB(
    present ~ depth_scaled,
    data = pcod_2011,
    mesh = pcod_mesh_2011,
    family = poisson(),
    backend = "quadra",
    control = sdmTMBcontrol(
      iter.max = 100, eval.max = 200, getsd = FALSE
    )
  )
  expect_equal(poisson_fit$model$convergence, 0L)
  expect_true(all(fitted(poisson_fit) > 0))

  expect_error(
    sdmTMB(
      present ~ depth_scaled,
      data = pcod_2011,
      mesh = pcod_mesh_2011,
      family = binomial(),
      backend = "quadra"
    ),
    "gaussian\\(identity\\).*poisson"
  )
})

test_that("Quadra prediction columns and components match TMB", {
  control <- sdmTMBcontrol(
    iter.max = 100, eval.max = 200, getsd = FALSE,
    newton_loops = 0, multiphase = FALSE
  )
  newdata <- pcod_2011[1:10, ]
  for (model in c("spatial", "spatiotemporal")) {
    arguments <- list(
      formula = log(density + 0.1) ~ depth_scaled,
      data = pcod_2011,
      mesh = pcod_mesh_2011,
      family = gaussian(),
      control = control
    )
    if (identical(model, "spatiotemporal")) {
      arguments$time <- "year"
      arguments$spatiotemporal <- "iid"
    }
    quadra_fit <- do.call(sdmTMB, c(arguments, list(backend = "quadra")))
    tmb_fit <- do.call(sdmTMB, c(arguments, list(backend = "tmb")))
    quadra_prediction <- predict(quadra_fit, newdata = newdata)
    tmb_prediction <- predict(tmb_fit, newdata = newdata)

    expect_identical(class(quadra_prediction), class(newdata))
    component_columns <- c("est", "est_non_rf", "est_rf", "omega_s")
    if (identical(model, "spatiotemporal")) {
      component_columns <- c(component_columns, "epsilon_st")
      expect_equal(
        quadra_prediction$est_rf,
        quadra_prediction$omega_s + quadra_prediction$epsilon_st,
        tolerance = 1e-12
      )
    } else {
      expect_false("epsilon_st" %in% names(quadra_prediction))
    }
    expect_true(all(component_columns %in% names(quadra_prediction)))
    expect_equal(
      quadra_prediction$est,
      quadra_prediction$est_non_rf + quadra_prediction$est_rf,
      tolerance = 1e-12
    )
    for (column in component_columns) {
      expect_equal(
        quadra_prediction[[column]], tmb_prediction[[column]],
        tolerance = 2e-4
      )
    }
  }
})

test_that("Quadra computes spatiotemporal newdata prediction uncertainty", {
  fit <- sdmTMB(
    log(density + 0.1) ~ depth_scaled,
    data = pcod_2011,
    mesh = pcod_mesh_2011,
    time = "year",
    spatiotemporal = "iid",
    family = gaussian(),
    backend = "quadra",
    control = sdmTMBcontrol(
      iter.max = 100, eval.max = 200, getsd = TRUE
    )
  )
  prediction <- predict(
    fit, newdata = pcod_2011[1:3, ], se_fit = TRUE
  )
  expect_equal(nrow(prediction), 3L)
  expect_true(all(is.finite(prediction$est_se)))
  expect_true(all(prediction$est_se >= 0))
  expect_true(all(prediction$lwr <= prediction$est))
  expect_true(all(prediction$upr >= prediction$est))
  expect_true(isTRUE(attr(prediction, "quadra_uncertainty")$success))
})

test_that("Quadra supports persistent Poisson IID spatiotemporal fields", {
  fit <- sdmTMB(
    present ~ depth_scaled,
    data = pcod_2011,
    mesh = pcod_mesh_2011,
    time = "year",
    spatiotemporal = "iid",
    family = poisson(),
    backend = "quadra",
    control = sdmTMBcontrol(
      iter.max = 100, eval.max = 200, newton_loops = 0,
      multiphase = FALSE, getsd = FALSE
    )
  )

  n_time <- length(unique(pcod_2011$year))
  n_node <- ncol(pcod_mesh_2011$A_st)
  expect_s3_class(fit, "sdmTMB_quadra")
  expect_identical(fit$spatiotemporal, "iid")
  expect_identical(fit$time, "year")
  expect_equal(fit$model$convergence, 0L)
  expect_length(fit$quadra_obj$env$last.eval$u_hat, n_node * (n_time + 1L))
  expect_equal(
    dim(fit$report$spatiotemporal_field), c(n_node, n_time)
  )
  expect_true(fit$quadra_obj$env$last.eval$hessian_nonzeros > 5000L)
  expect_true(fit$quadra_obj$env$last.eval$converged)
  expect_true(all(is.finite(fitted(fit))))
  expect_equal(
    nrow(predict(fit, pcod_2011[1:4, ], type = "response")), 4L
  )
})

test_that("Quadra supports persistent Poisson AR1 spatiotemporal fields", {
  fit <- sdmTMB(
    present ~ depth_scaled,
    data = pcod_2011,
    mesh = pcod_mesh_2011,
    time = "year",
    spatiotemporal = "ar1",
    family = poisson(),
    backend = "quadra",
    control = sdmTMBcontrol(
      iter.max = 100, eval.max = 200, newton_loops = 0,
      multiphase = FALSE, getsd = FALSE
    )
  )

  expect_identical(fit$spatiotemporal, "ar1")
  expect_equal(fit$model$convergence, 0L)
  expect_true(is.finite(fit$report$rho))
  expect_true(abs(fit$report$rho) < 1)
  expect_true(fit$quadra_obj$env$last.eval$hessian_nonzeros > 10000L)
  expect_true(fit$quadra_obj$env$last.eval$converged)
  expect_equal(
    nrow(predict(fit, pcod_2011[1:4, ], type = "response")), 4L
  )
})

test_that("Quadra supports persistent Poisson RW spatiotemporal fields", {
  fit <- sdmTMB(
    present ~ depth_scaled,
    data = pcod_2011,
    mesh = pcod_mesh_2011,
    time = "year",
    spatiotemporal = "rw",
    family = poisson(),
    backend = "quadra",
    control = sdmTMBcontrol(
      iter.max = 100, eval.max = 200, newton_loops = 0,
      multiphase = FALSE, getsd = FALSE
    )
  )

  expect_identical(fit$spatiotemporal, "rw")
  expect_equal(fit$model$convergence, 0L)
  expect_true(fit$quadra_obj$env$last.eval$hessian_nonzeros > 10000L)
  expect_true(fit$quadra_obj$env$last.eval$converged)
  expect_equal(
    nrow(predict(fit, pcod_2011[1:4, ], type = "response")), 4L
  )
})

test_that("Quadra supports separate spatial and spatiotemporal ranges", {
  for (temporal_model in c("iid", "ar1", "rw")) {
    fit <- sdmTMB(
      present ~ depth_scaled,
      data = pcod_2011,
      mesh = pcod_mesh_2011,
      time = "year",
      spatiotemporal = temporal_model,
      share_range = FALSE,
      family = poisson(),
      backend = "quadra",
      do_fit = FALSE
    )
    expect_false(fit$share_range)
    expect_true("log_kappa_spatiotemporal" %in% names(fit$quadra_obj$par))

    par <- fit$quadra_obj$par
    par["log_kappa"] <- log(0.03)
    par["log_kappa_spatiotemporal"] <- log(0.05)
    expect_true(is.finite(fit$quadra_obj$fn(par)))
    expect_true(all(is.finite(fit$quadra_obj$gr(par))))

    report <- fit$quadra_obj$report(par)
    expect_equal(unname(report$spatial_kappa), 0.03, tolerance = 1e-12)
    expect_equal(
      unname(report$spatiotemporal_kappa), 0.05, tolerance = 1e-12
    )
    expect_false(isTRUE(all.equal(
      report$spatial_range, report$spatiotemporal_range
    )))
  }
})

test_that("Quadra supports Poisson likelihood weights", {
  observation_weights <- seq(0.5, 1.5, length.out = nrow(pcod_2011))
  for (temporal_model in c("iid", "ar1", "rw")) {
    fit <- sdmTMB(
      present ~ depth_scaled,
      data = pcod_2011,
      mesh = pcod_mesh_2011,
      time = "year",
      spatiotemporal = temporal_model,
      share_range = FALSE,
      weights = observation_weights,
      family = poisson(),
      backend = "quadra",
      do_fit = FALSE
    )
    expect_true(is.finite(fit$quadra_obj$fn()))
    expect_true(all(is.finite(fit$quadra_obj$gr())))
  }

  expect_error(
    sdmTMB(
      present ~ depth_scaled, data = pcod_2011, mesh = pcod_mesh_2011,
      weights = rep(-1, nrow(pcod_2011)), family = poisson(),
      backend = "quadra", do_fit = FALSE
    ),
    "finite, nonnegative"
  )
})

test_that("Quadra supports weighted NB2 spatial and spatiotemporal models", {
  observation_weights <- seq(0.5, 1.5, length.out = nrow(pcod_2011))
  for (temporal_model in c("off", "iid", "ar1", "rw")) {
    use_time <- !identical(temporal_model, "off")
    fit <- sdmTMB(
      present ~ depth_scaled,
      data = pcod_2011,
      mesh = pcod_mesh_2011,
      time = if (use_time) "year" else NULL,
      spatiotemporal = temporal_model,
      share_range = FALSE,
      weights = observation_weights,
      family = nbinom2(),
      backend = "quadra",
      do_fit = FALSE
    )
    expect_true("log_phi" %in% names(fit$quadra_obj$par))
    expect_true(is.finite(fit$quadra_obj$fn()))
    expect_true(all(is.finite(fit$quadra_obj$gr())))
    expect_equal(
      unname(fit$quadra_obj$report()$phi), 10, tolerance = 1e-12
    )
  }
})

test_that("Quadra supports weighted Gaussian spatiotemporal models", {
  observation_weights <- seq(0.5, 1.5, length.out = nrow(pcod_2011))
  for (temporal_model in c("iid", "ar1", "rw")) {
    fit <- sdmTMB(
      depth_scaled ~ present,
      data = pcod_2011,
      mesh = pcod_mesh_2011,
      time = "year",
      spatiotemporal = temporal_model,
      share_range = FALSE,
      weights = observation_weights,
      family = gaussian(),
      backend = "quadra",
      do_fit = FALSE
    )
    expect_true("log_sigma" %in% names(fit$quadra_obj$par))
    expect_true(is.finite(fit$quadra_obj$fn()))
    expect_identical(fit$quadra_obj$env$last.eval$iterations, 1L)
    expect_true(all(is.finite(fit$quadra_obj$gr())))
    report <- fit$quadra_obj$report()
    expect_true(is.finite(report$sigma) && report$sigma > 0)
    expect_equal(report$mu_i, report$eta_i)
  }
})

test_that("Quadra native L-BFGS reaches the nlminb spatial optimum", {
  base_control <- list(
    getsd = FALSE, newton_loops = 0, multiphase = FALSE,
    iter.max = 100
  )
  fit_nlminb <- sdmTMB(
    log(density + 0.1) ~ depth_scaled,
    data = pcod_2011, mesh = pcod_mesh_2011, family = gaussian(),
    backend = "quadra",
    control = do.call(sdmTMBcontrol, c(
      base_control, list(quadra_optimizer = "nlminb")
    ))
  )
  fit_lbfgs <- sdmTMB(
    log(density + 0.1) ~ depth_scaled,
    data = pcod_2011, mesh = pcod_mesh_2011, family = gaussian(),
    backend = "quadra",
    control = do.call(sdmTMBcontrol, c(
      base_control, list(quadra_optimizer = "lbfgs")
    ))
  )
  expect_identical(fit_lbfgs$optimizer, "lbfgs")
  expect_identical(fit_lbfgs$model$convergence, 0L)
  expect_equal(
    fit_lbfgs$model$objective, fit_nlminb$model$objective,
    tolerance = 1e-7
  )
  expect_lt(fit_lbfgs$model$gradient_norm, 1e-4)
})
