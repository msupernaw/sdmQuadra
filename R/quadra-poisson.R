.make_quadra_poisson_spde_object <- function(
    X, y, A, spde, offset = NULL, weights = NULL, start = NULL,
    random_start = NULL, nb2 = FALSE, gaussian = FALSE
) {
  X <- as.matrix(X)
  storage.mode(X) <- "double"
  y <- as.double(y)
  A <- Matrix::Matrix(A, sparse = TRUE)
  matrices <- lapply(spde[c("c0", "g1", "g2")], Matrix::Matrix, sparse = TRUE)
  if (length(matrices) != 3L || any(vapply(matrices, is.null, logical(1)))) {
    stop("spde must contain c0, g1, and g2", call. = FALSE)
  }
  n_node <- ncol(A)
  if (length(y) != nrow(X) || nrow(A) != length(y) ||
      (!isTRUE(gaussian) && any(y < 0 | y != floor(y)))) {
    stop("Poisson data and design matrices are not conformable", call. = FALSE)
  }
  if (is.null(offset)) offset <- numeric(length(y))
  offset <- as.double(offset)
  if (is.null(weights)) weights <- rep(1, length(y))
  weights <- as.double(weights)
  if (length(weights) != length(y) ||
      any(!is.finite(weights) | weights < 0)) {
    stop("weights must be finite, nonnegative, and match y", call. = FALSE)
  }
  if (is.null(start)) {
    mean_y <- max(mean(y), 0.1)
    start <- c(
      rep(0, ncol(X)), 0, 0,
      if (isTRUE(nb2)) log(10),
      if (isTRUE(gaussian)) log(max(stats::sd(y), 1e-3))
    )
    if (ncol(X) && all(X[, 1L] == 1)) start[1L] <- log(mean_y)
  }
  start <- as.double(start)
  if (length(start) !=
      ncol(X) + 2L + as.integer(isTRUE(nb2) || isTRUE(gaussian))) {
    stop("start must contain beta, log_precision_scale, log_kappa, and optional log_phi",
      call. = FALSE
    )
  }
  if (is.null(random_start)) random_start <- numeric(n_node)
  random_start <- as.double(random_start)
  a <- .quadra_sparse_triplet(A)
  m0 <- .quadra_sparse_triplet(matrices[[1L]])
  m1 <- .quadra_sparse_triplet(matrices[[2L]])
  m2 <- .quadra_sparse_triplet(matrices[[3L]])
  state <- new.env(parent = emptyenv())
  state$last.par <- state$last.par.best <- start
  state$random <- random_start
  state$last.eval <- NULL
  state$cached.par <- NULL
  state$cached.eval <- NULL
  state$native <- .Call(
    "sdmTMB_quadra_poisson_state_create",
    start, random_start, X, y, offset, weights,
    as.integer(a$i), as.integer(a$j), as.double(a$x),
    as.integer(m0$i), as.integer(m0$j), as.double(m0$x),
    as.integer(m1$i), as.integer(m1$j), as.double(m1$x),
    as.integer(m2$i), as.integer(m2$j), as.double(m2$x),
    as.logical(nb2), as.logical(gaussian),
    PACKAGE = "sdmTMB"
  )
  evaluate <- function(par, gradient = FALSE) {
    par <- as.double(par)
    if (!is.null(state$cached.par) &&
        identical(par, state$cached.par) &&
        (!gradient || all(is.finite(state$cached.eval$gradient)))) {
      state$last.par <- state$last.par.best <- par
      state$last.eval <- state$cached.eval
      return(state$cached.eval)
    }
    result <- .Call(
      "sdmTMB_quadra_poisson_state_evaluate",
      state$native, par, as.logical(gradient), PACKAGE = "sdmTMB"
    )
    state$last.par <- state$last.par.best <- as.double(par)
    if (isTRUE(result$converged)) state$random <- result$u_hat
    state$last.eval <- result
    state$cached.par <- par
    state$cached.eval <- result
    result
  }
  structure(list(
    par = start,
    fn = function(par = state$last.par) evaluate(par, FALSE)$value,
    gr = function(par = state$last.par) evaluate(par, TRUE)$gradient,
    covariance = function(par = state$last.par) {
      .Call(
        "sdmTMB_quadra_poisson_state_covariance",
        state$native, as.double(par), PACKAGE = "sdmTMB"
      )
    },
    prediction_uncertainty = function(X_prediction, A_prediction) {
      X_theta <- matrix(
        0, nrow(X_prediction), length(state$last.par),
        dimnames = list(NULL, NULL)
      )
      X_theta[, seq_len(ncol(X_prediction))] <- X_prediction
      z <- .quadra_sparse_triplet(A_prediction)
      .Call(
        "sdmTMB_quadra_poisson_state_prediction_uncertainty",
        state$native, X_theta, as.integer(z$i), as.integer(z$j),
        as.double(z$x), as.integer(n_node), PACKAGE = "sdmTMB"
      )
    },
    report = function(par = state$last.par) {
      result <- evaluate(par, FALSE)
      projected <- as.vector(A %*% result$u_hat)
      eta <- as.vector(X %*% par[seq_len(ncol(X))] + projected + offset)
      list(
        eta_i = eta,
        mu_i = if (isTRUE(gaussian)) eta else exp(eta),
        field = result$u_hat,
        projected_field = projected,
        precision_scale = exp(par[ncol(X) + 1L]),
        kappa = exp(par[ncol(X) + 2L]),
        range = sqrt(8) / exp(par[ncol(X) + 2L]),
        phi = if (isTRUE(nb2)) exp(par[ncol(X) + 3L]) else Inf,
        sigma = if (isTRUE(gaussian)) {
          exp(par[ncol(X) + 3L])
        } else {
          NA_real_
        }
      )
    },
    env = state, A = A, spde = spde, nb2 = isTRUE(nb2),
    gaussian = isTRUE(gaussian)
  ), class = "sdmTMB_quadra_object")
}

.make_quadra_poisson_spatiotemporal_iid_object <- function(
    X, y, A, spde, time_index, offset = NULL, start = NULL,
    random_start = NULL, ar1 = FALSE, rw = FALSE, share_range = TRUE,
    weights = NULL, nb2 = FALSE, gaussian = FALSE
) {
  if (isTRUE(nb2) && isTRUE(gaussian)) {
    stop("NB2 and Gaussian cannot both be selected", call. = FALSE)
  }
  if (isTRUE(ar1) && isTRUE(rw)) {
    stop("Only one temporal model can be selected", call. = FALSE)
  }
  X <- as.matrix(X)
  storage.mode(X) <- "double"
  y <- as.double(y)
  A <- Matrix::Matrix(A, sparse = TRUE)
  time_index <- as.integer(time_index)
  matrices <- lapply(spde[c("c0", "g1", "g2")], Matrix::Matrix, sparse = TRUE)
  n_node <- ncol(A)
  n_time <- if (length(time_index)) max(time_index) + 1L else 0L
  if (length(y) != nrow(X) || nrow(A) != length(y) ||
      length(time_index) != length(y) || n_time < 1L ||
      any(time_index < 0L) ||
      (!isTRUE(gaussian) && any(y < 0 | y != floor(y)))) {
    stop("Poisson spatiotemporal data are not conformable", call. = FALSE)
  }
  if (is.null(offset)) offset <- numeric(length(y))
  offset <- as.double(offset)
  if (is.null(weights)) weights <- rep(1, length(y))
  weights <- as.double(weights)
  if (length(weights) != length(y) ||
      any(!is.finite(weights) | weights < 0)) {
    stop("weights must be finite, nonnegative, and match y", call. = FALSE)
  }
  if (is.null(start)) {
    start <- c(
      rep(0, ncol(X)), 0, 0, 0,
      if (!isTRUE(share_range)) 0,
      if (isTRUE(ar1)) 0.1,
      if (isTRUE(nb2)) log(10),
      if (isTRUE(gaussian)) log(max(stats::sd(y), 1e-3))
    )
    if (ncol(X) && all(X[, 1L] == 1)) {
      start[1L] <- log(max(mean(y), 0.1))
    }
  }
  start <- as.double(start)
  if (length(start) !=
      ncol(X) + 3L + as.integer(!isTRUE(share_range)) +
        as.integer(isTRUE(ar1)) +
        as.integer(isTRUE(nb2) || isTRUE(gaussian))) {
    stop(
      "start must contain beta, precision scales, log_kappa, and optional AR1 parameter",
      call. = FALSE
    )
  }
  n_random <- n_node * (n_time + 1L)
  if (is.null(random_start)) random_start <- numeric(n_random)
  random_start <- as.double(random_start)
  if (length(random_start) != n_random) {
    stop("random_start has the wrong spatiotemporal size", call. = FALSE)
  }
  a <- .quadra_sparse_triplet(A)
  m0 <- .quadra_sparse_triplet(matrices[[1L]])
  m1 <- .quadra_sparse_triplet(matrices[[2L]])
  m2 <- .quadra_sparse_triplet(matrices[[3L]])
  state <- new.env(parent = emptyenv())
  state$last.par <- state$last.par.best <- start
  state$random <- random_start
  state$last.eval <- state$cached.par <- state$cached.eval <- NULL
  state$native <- .Call(
    "sdmTMB_quadra_poisson_st_iid_state_create",
    start, random_start, X, y, offset, weights, time_index,
    as.integer(a$i), as.integer(a$j), as.double(a$x),
    as.integer(m0$i), as.integer(m0$j), as.double(m0$x),
    as.integer(m1$i), as.integer(m1$j), as.double(m1$x),
    as.integer(m2$i), as.integer(m2$j), as.double(m2$x),
    as.integer(if (isTRUE(ar1)) 1L else if (isTRUE(rw)) 2L else 0L),
    as.logical(!isTRUE(share_range)),
    as.logical(isTRUE(nb2)),
    as.logical(isTRUE(gaussian)),
    PACKAGE = "sdmTMB"
  )
  evaluate <- function(par, gradient = FALSE) {
    par <- as.double(par)
    if (!is.null(state$cached.par) && identical(par, state$cached.par) &&
        (!gradient || all(is.finite(state$cached.eval$gradient)))) {
      state$last.par <- state$last.par.best <- par
      state$last.eval <- state$cached.eval
      return(state$cached.eval)
    }
    result <- .Call(
      "sdmTMB_quadra_poisson_st_iid_state_evaluate",
      state$native, par, as.logical(gradient), PACKAGE = "sdmTMB"
    )
    state$last.par <- state$last.par.best <- par
    if (isTRUE(result$converged)) state$random <- result$u_hat
    state$last.eval <- result
    state$cached.par <- par
    state$cached.eval <- result
    result
  }
  structure(list(
    par = start,
    fn = function(par = state$last.par) evaluate(par, FALSE)$value,
    gr = function(par = state$last.par) evaluate(par, TRUE)$gradient,
    covariance = function(par = state$last.par) {
      .Call(
        "sdmTMB_quadra_poisson_st_iid_state_covariance",
        state$native, as.double(par), PACKAGE = "sdmTMB"
      )
    },
    prediction_uncertainty = function(
        X_prediction, A_prediction, prediction_time_index
    ) {
      X_theta <- matrix(0, nrow(X_prediction), length(state$last.par))
      X_theta[, seq_len(ncol(X_prediction))] <- X_prediction
      a_prediction <- .quadra_sparse_triplet(A_prediction)
      spatial_j <- a_prediction$j
      temporal_j <- a_prediction$j +
        n_node * (prediction_time_index[a_prediction$i + 1L] + 1L)
      z_i <- c(a_prediction$i, a_prediction$i)
      z_j <- c(spatial_j, temporal_j)
      z_x <- c(a_prediction$x, a_prediction$x)
      .Call(
        "sdmTMB_quadra_poisson_st_iid_state_prediction_uncertainty",
        state$native, X_theta, as.integer(z_i), as.integer(z_j),
        as.double(z_x), as.integer(n_random), PACKAGE = "sdmTMB"
      )
    },
    report = function(par = state$last.par) {
      result <- evaluate(par, FALSE)
      spatial <- result$u_hat[seq_len(n_node)]
      st <- matrix(
        result$u_hat[-seq_len(n_node)], nrow = n_node, ncol = n_time
      )
      projected_spatial <- as.vector(A %*% spatial)
      projected_st <- vapply(seq_along(y), function(i) {
        as.numeric(A[i, , drop = FALSE] %*% st[, time_index[i] + 1L])
      }, numeric(1))
      eta <- as.vector(
        X %*% par[seq_len(ncol(X))] + offset +
          projected_spatial + projected_st
      )
      list(
        eta_i = eta,
        mu_i = if (isTRUE(gaussian)) eta else exp(eta),
        field = spatial,
        spatiotemporal_field = st,
        projected_field = projected_spatial,
        projected_spatiotemporal = projected_st,
        spatial_precision_scale = exp(par[ncol(X) + 1L]),
        spatiotemporal_precision_scale = exp(par[ncol(X) + 2L]),
        kappa = exp(par[ncol(X) + 3L]),
        spatial_kappa = exp(par[ncol(X) + 3L]),
        spatiotemporal_kappa = exp(
          par[ncol(X) + 3L + as.integer(!isTRUE(share_range))]
        ),
        range = sqrt(8) / exp(par[ncol(X) + 3L]),
        spatial_range = sqrt(8) / exp(par[ncol(X) + 3L]),
        spatiotemporal_range = sqrt(8) / exp(
          par[ncol(X) + 3L + as.integer(!isTRUE(share_range))]
        ),
        rho = if (isTRUE(ar1)) {
          2 * stats::plogis(
            par[ncol(X) + 4L + as.integer(!isTRUE(share_range))]
          ) - 1
        } else {
          0
        },
        phi = if (isTRUE(nb2)) exp(par[length(par)]) else Inf,
        sigma = if (isTRUE(gaussian)) exp(par[length(par)]) else NA_real_
      )
    },
    env = state, A = A, spde = spde, time_index = time_index,
    n_time = n_time, ar1 = isTRUE(ar1), rw = isTRUE(rw),
    share_range = isTRUE(share_range), nb2 = isTRUE(nb2),
    gaussian = isTRUE(gaussian)
  ), class = "sdmTMB_quadra_object")
}
