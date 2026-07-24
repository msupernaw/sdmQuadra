# Experimental Quadra backend helpers.

.quadra_gaussian_eval <- function(par, X, y, offset) {
  .Call(
    "sdmTMB_quadra_gaussian",
    as.double(par),
    X,
    y,
    as.double(offset),
    PACKAGE = "sdmTMB"
  )
}

.make_quadra_gaussian_object <- function(X, y, offset = NULL, start = NULL) {
  X <- as.matrix(X)
  storage.mode(X) <- "double"
  y <- as.double(y)

  if (length(y) != nrow(X)) {
    stop("length(y) must equal nrow(X)", call. = FALSE)
  }
  if (is.null(offset)) {
    offset <- numeric(length(y))
  }
  offset <- as.double(offset)
  if (length(offset) != length(y)) {
    stop("length(offset) must equal length(y)", call. = FALSE)
  }
  if (is.null(start)) {
    start <- numeric(ncol(X))
  }
  start <- as.double(start)
  if (length(start) != ncol(X)) {
    stop("length(start) must equal ncol(X)", call. = FALSE)
  }

  state <- new.env(parent = emptyenv())
  state$last.par <- start
  state$last.par.best <- start
  state$last.eval <- NULL

  evaluate <- function(par) {
    result <- .quadra_gaussian_eval(par, X, y, offset)
    state$last.par <- as.double(par)
    state$last.par.best <- state$last.par
    state$last.eval <- result
    result
  }

  structure(
    list(
      par = start,
      fn = function(par = state$last.par) evaluate(par)$value,
      gr = function(par = state$last.par) evaluate(par)$gradient,
      report = function(par = state$last.par) {
        eta <- as.vector(X %*% par + offset)
        list(eta_i = eta, mu_i = eta)
      },
      env = state
    ),
    class = "sdmTMB_quadra_object"
  )
}

.quadra_gaussian_random_intercept_eval <- function(
    fixed, random, X, y, offset, group
) {
  .Call(
    "sdmTMB_quadra_gaussian_random_intercept",
    as.double(fixed),
    as.double(random),
    X,
    y,
    as.double(offset),
    as.integer(group),
    PACKAGE = "sdmTMB"
  )
}

.make_quadra_gaussian_random_intercept_object <- function(
    X, y, group, offset = NULL, start = NULL, random_start = NULL
) {
  X <- as.matrix(X)
  storage.mode(X) <- "double"
  y <- as.double(y)

  if (length(y) != nrow(X)) {
    stop("length(y) must equal nrow(X)", call. = FALSE)
  }
  if (length(group) != length(y)) {
    stop("length(group) must equal length(y)", call. = FALSE)
  }
  if (anyNA(group)) {
    stop("group cannot contain missing values", call. = FALSE)
  }

  group_levels <- unique(group)
  group_index <- match(group, group_levels) - 1L
  n_group <- length(group_levels)

  if (is.null(offset)) {
    offset <- numeric(length(y))
  }
  offset <- as.double(offset)
  if (length(offset) != length(y)) {
    stop("length(offset) must equal length(y)", call. = FALSE)
  }

  if (is.null(start)) {
    start <- c(numeric(ncol(X)), log(stats::sd(y) / 2), log(stats::sd(y) / 2))
  }
  start <- as.double(start)
  if (length(start) != ncol(X) + 2L) {
    stop(
      "start must contain beta, log_sigma_obs, and log_sigma_group",
      call. = FALSE
    )
  }
  if (is.null(random_start)) {
    random_start <- numeric(n_group)
  }
  random_start <- as.double(random_start)
  if (length(random_start) != n_group) {
    stop("length(random_start) must equal the number of groups", call. = FALSE)
  }

  state <- new.env(parent = emptyenv())
  state$last.par <- start
  state$last.par.best <- start
  state$random <- random_start
  state$last.eval <- NULL

  evaluate <- function(par) {
    result <- .quadra_gaussian_random_intercept_eval(
      par, state$random, X, y, offset, group_index
    )
    state$last.par <- as.double(par)
    state$last.par.best <- state$last.par
    state$random <- result$u_hat
    state$last.eval <- result
    result
  }

  structure(
    list(
      par = start,
      fn = function(par = state$last.par) evaluate(par)$value,
      gr = function(par = state$last.par) evaluate(par)$gradient,
      report = function(par = state$last.par) {
        result <- evaluate(par)
        eta <- as.vector(X %*% par[seq_len(ncol(X))] +
          result$u_hat[group_index + 1L] + offset)
        list(
          eta_i = eta,
          mu_i = eta,
          random_intercept = stats::setNames(result$u_hat, group_levels),
          sigma = exp(par[ncol(X) + 1L]),
          sigma_group = exp(par[ncol(X) + 2L])
        )
      },
      env = state,
      group_levels = group_levels
    ),
    class = "sdmTMB_quadra_object"
  )
}

.quadra_gaussian_sparse_field_eval <- function(
    fixed, random, X, y, offset, node, q_triplet, logdet_q
) {
  .Call(
    "sdmTMB_quadra_gaussian_sparse_field",
    as.double(fixed),
    as.double(random),
    X,
    y,
    as.double(offset),
    as.integer(node),
    as.integer(q_triplet$i),
    as.integer(q_triplet$j),
    as.double(q_triplet$x),
    as.double(logdet_q),
    PACKAGE = "sdmTMB"
  )
}

.quadra_sparse_triplet <- function(x) {
  x <- methods::as(methods::as(x, "generalMatrix"), "TsparseMatrix")
  list(i = x@i, j = x@j, x = x@x)
}

.make_quadra_gaussian_sparse_field_object <- function(
    X, y, node, Q, offset = NULL, start = NULL, random_start = NULL
) {
  X <- as.matrix(X)
  storage.mode(X) <- "double"
  y <- as.double(y)
  Q <- Matrix::Matrix(Q, sparse = TRUE)

  if (length(y) != nrow(X)) {
    stop("length(y) must equal nrow(X)", call. = FALSE)
  }
  if (length(node) != length(y)) {
    stop("length(node) must equal length(y)", call. = FALSE)
  }
  if (nrow(Q) != ncol(Q) || !isTRUE(Matrix::isSymmetric(Q))) {
    stop("Q must be a symmetric square precision matrix", call. = FALSE)
  }
  n_node <- nrow(Q)
  node <- as.integer(node)
  if (anyNA(node) || any(node < 1L | node > n_node)) {
    stop("node must contain valid one-based indices into Q", call. = FALSE)
  }
  node_index <- node - 1L

  factorization <- try(Matrix::Cholesky(Q, LDL = FALSE), silent = TRUE)
  if (inherits(factorization, "try-error")) {
    stop("Q must be positive definite", call. = FALSE)
  }
  logdet_q <- as.numeric(Matrix::determinant(Q, logarithm = TRUE)$modulus)
  q_triplet <- .quadra_sparse_triplet(Q)

  if (is.null(offset)) {
    offset <- numeric(length(y))
  }
  offset <- as.double(offset)
  if (length(offset) != length(y)) {
    stop("length(offset) must equal length(y)", call. = FALSE)
  }
  if (is.null(start)) {
    response_sd <- stats::sd(y)
    if (!is.finite(response_sd) || response_sd <= 0) {
      response_sd <- 1
    }
    start <- c(numeric(ncol(X)), log(response_sd / 2), 0)
  }
  start <- as.double(start)
  if (length(start) != ncol(X) + 2L) {
    stop(
      "start must contain beta, log_sigma_obs, and log_precision_scale",
      call. = FALSE
    )
  }
  if (is.null(random_start)) {
    random_start <- numeric(n_node)
  }
  random_start <- as.double(random_start)
  if (length(random_start) != n_node) {
    stop("length(random_start) must equal nrow(Q)", call. = FALSE)
  }

  state <- new.env(parent = emptyenv())
  state$last.par <- start
  state$last.par.best <- start
  state$random <- random_start
  state$last.eval <- NULL

  evaluate <- function(par) {
    result <- .quadra_gaussian_sparse_field_eval(
      par, state$random, X, y, offset, node_index, q_triplet, logdet_q
    )
    state$last.par <- as.double(par)
    state$last.par.best <- state$last.par
    state$random <- result$u_hat
    state$last.eval <- result
    result
  }

  structure(
    list(
      par = start,
      fn = function(par = state$last.par) evaluate(par)$value,
      gr = function(par = state$last.par) evaluate(par)$gradient,
      report = function(par = state$last.par) {
        result <- evaluate(par)
        eta <- as.vector(X %*% par[seq_len(ncol(X))] +
          result$u_hat[node] + offset)
        list(
          eta_i = eta,
          mu_i = eta,
          field = result$u_hat,
          sigma = exp(par[ncol(X) + 1L]),
          precision_scale = exp(par[ncol(X) + 2L])
        )
      },
      env = state,
      Q = Q,
      node = node
    ),
    class = "sdmTMB_quadra_object"
  )
}

.quadra_gaussian_projected_field_eval <- function(
    fixed, random, X, y, offset, a_triplet, q_triplet, logdet_q
) {
  .Call(
    "sdmTMB_quadra_gaussian_projected_field",
    as.double(fixed),
    as.double(random),
    X,
    y,
    as.double(offset),
    as.integer(a_triplet$i),
    as.integer(a_triplet$j),
    as.double(a_triplet$x),
    as.integer(q_triplet$i),
    as.integer(q_triplet$j),
    as.double(q_triplet$x),
    as.double(logdet_q),
    PACKAGE = "sdmTMB"
  )
}

.make_quadra_gaussian_projected_field_object <- function(
    X, y, A, Q, offset = NULL, start = NULL, random_start = NULL
) {
  X <- as.matrix(X)
  storage.mode(X) <- "double"
  y <- as.double(y)
  A <- Matrix::Matrix(A, sparse = TRUE)
  Q <- Matrix::Matrix(Q, sparse = TRUE)

  if (length(y) != nrow(X) || nrow(A) != length(y)) {
    stop("nrow(X), nrow(A), and length(y) must match", call. = FALSE)
  }
  if (nrow(Q) != ncol(Q) || ncol(A) != nrow(Q)) {
    stop("A and Q have incompatible dimensions", call. = FALSE)
  }
  if (!isTRUE(Matrix::isSymmetric(Q))) {
    stop("Q must be symmetric", call. = FALSE)
  }
  factorization <- try(Matrix::Cholesky(Q, LDL = FALSE), silent = TRUE)
  if (inherits(factorization, "try-error")) {
    stop("Q must be positive definite", call. = FALSE)
  }

  logdet_q <- as.numeric(Matrix::determinant(Q, logarithm = TRUE)$modulus)
  a_triplet <- .quadra_sparse_triplet(A)
  q_triplet <- .quadra_sparse_triplet(Q)

  if (is.null(offset)) {
    offset <- numeric(length(y))
  }
  offset <- as.double(offset)
  if (length(offset) != length(y)) {
    stop("length(offset) must equal length(y)", call. = FALSE)
  }
  if (is.null(start)) {
    response_sd <- stats::sd(y)
    if (!is.finite(response_sd) || response_sd <= 0) {
      response_sd <- 1
    }
    start <- c(numeric(ncol(X)), log(response_sd / 2), 0)
  }
  start <- as.double(start)
  if (length(start) != ncol(X) + 2L) {
    stop(
      "start must contain beta, log_sigma_obs, and log_precision_scale",
      call. = FALSE
    )
  }
  if (is.null(random_start)) {
    random_start <- numeric(ncol(A))
  }
  random_start <- as.double(random_start)
  if (length(random_start) != ncol(A)) {
    stop("length(random_start) must equal ncol(A)", call. = FALSE)
  }

  state <- new.env(parent = emptyenv())
  state$last.par <- start
  state$last.par.best <- start
  state$random <- random_start
  state$last.eval <- NULL
  evaluate <- function(par) {
    result <- .quadra_gaussian_projected_field_eval(
      par, state$random, X, y, offset, a_triplet, q_triplet, logdet_q
    )
    state$last.par <- as.double(par)
    state$last.par.best <- state$last.par
    state$random <- result$u_hat
    state$last.eval <- result
    result
  }

  structure(
    list(
      par = start,
      fn = function(par = state$last.par) evaluate(par)$value,
      gr = function(par = state$last.par) evaluate(par)$gradient,
      report = function(par = state$last.par) {
        result <- evaluate(par)
        projected <- as.vector(A %*% result$u_hat)
        eta <- as.vector(X %*% par[seq_len(ncol(X))] + projected + offset)
        list(
          eta_i = eta,
          mu_i = eta,
          field = result$u_hat,
          projected_field = projected,
          sigma = exp(par[ncol(X) + 1L]),
          precision_scale = exp(par[ncol(X) + 2L])
        )
      },
      env = state,
      A = A,
      Q = Q
    ),
    class = "sdmTMB_quadra_object"
  )
}

.make_quadra_gaussian_spde_object <- function(
    X, y, A, spde, offset = NULL, start = NULL, random_start = NULL
) {
  required_spde <- c("c0", "g1", "g2")
  if (!all(required_spde %in% names(spde))) {
    stop("spde must contain c0, g1, and g2 matrices", call. = FALSE)
  }

  X <- as.matrix(X)
  storage.mode(X) <- "double"
  y <- as.double(y)
  A <- Matrix::Matrix(A, sparse = TRUE)
  M0 <- Matrix::Matrix(spde$c0, sparse = TRUE)
  M1 <- Matrix::Matrix(spde$g1, sparse = TRUE)
  M2 <- Matrix::Matrix(spde$g2, sparse = TRUE)
  n_node <- ncol(A)

  if (length(y) != nrow(X) || nrow(A) != length(y)) {
    stop("nrow(X), nrow(A), and length(y) must match", call. = FALSE)
  }
  if (any(vapply(
    list(M0, M1, M2),
    function(m) nrow(m) != n_node || ncol(m) != n_node,
    logical(1)
  ))) {
    stop("SPDE matrices must be square and conformable with A", call. = FALSE)
  }
  if (is.null(offset)) {
    offset <- numeric(length(y))
  }
  offset <- as.double(offset)
  if (length(offset) != length(y)) {
    stop("length(offset) must equal length(y)", call. = FALSE)
  }
  if (is.null(start)) {
    response_sd <- stats::sd(y)
    if (!is.finite(response_sd) || response_sd <= 0) {
      response_sd <- 1
    }
    start <- c(numeric(ncol(X)), log(response_sd / 2), 0, 0)
  }
  start <- as.double(start)
  if (length(start) != ncol(X) + 3L) {
    stop(
      paste(
        "start must contain beta, log_sigma_obs, log_precision_scale,",
        "and log_kappa"
      ),
      call. = FALSE
    )
  }
  if (is.null(random_start)) {
    random_start <- numeric(n_node)
  }
  random_start <- as.double(random_start)
  if (length(random_start) != n_node) {
    stop("length(random_start) must equal ncol(A)", call. = FALSE)
  }

  a_triplet <- .quadra_sparse_triplet(A)
  m0_triplet <- .quadra_sparse_triplet(M0)
  m1_triplet <- .quadra_sparse_triplet(M1)
  m2_triplet <- .quadra_sparse_triplet(M2)
  build_precision <- function(log_kappa) {
    kappa2 <- exp(2 * log_kappa)
    Matrix::drop0(kappa2^2 * M0 + 2 * kappa2 * M1 + M2)
  }
  evaluate_native <- function(par, random, gradient) {
    .Call(
      "sdmTMB_quadra_gaussian_spde_field",
      as.double(par),
      as.double(random),
      X,
      y,
      offset,
      as.integer(a_triplet$i),
      as.integer(a_triplet$j),
      as.double(a_triplet$x),
      as.integer(m0_triplet$i),
      as.integer(m0_triplet$j),
      as.double(m0_triplet$x),
      as.integer(m1_triplet$i),
      as.integer(m1_triplet$j),
      as.double(m1_triplet$x),
      as.integer(m2_triplet$i),
      as.integer(m2_triplet$j),
      as.double(m2_triplet$x),
      as.logical(gradient),
      PACKAGE = "sdmTMB"
    )
  }

  state <- new.env(parent = emptyenv())
  state$last.par <- start
  state$last.par.best <- start
  state$random <- random_start
  state$last.eval <- NULL
  evaluate <- function(par, gradient = FALSE) {
    par <- as.double(par)
    center <- evaluate_native(par, state$random, gradient)
    state$random <- center$u_hat
    state$last.par <- par
    state$last.par.best <- par
    state$last.eval <- center
    center
  }

  structure(
    list(
      par = start,
      fn = function(par = state$last.par) evaluate(par, FALSE)$value,
      gr = function(par = state$last.par) evaluate(par, TRUE)$gradient,
      report = function(par = state$last.par) {
        result <- evaluate(par)
        projected <- as.vector(A %*% result$u_hat)
        eta <- as.vector(X %*% par[seq_len(ncol(X))] + projected + offset)
        list(
          eta_i = eta,
          mu_i = eta,
          field = result$u_hat,
          projected_field = projected,
          sigma = exp(par[ncol(X) + 1L]),
          precision_scale = exp(par[ncol(X) + 2L]),
          kappa = exp(par[ncol(X) + 3L]),
          range = sqrt(8) / exp(par[ncol(X) + 3L])
        )
      },
      env = state,
      A = A,
      spde = list(c0 = M0, g1 = M1, g2 = M2),
      build_precision = build_precision
    ),
    class = "sdmTMB_quadra_object"
  )
}
