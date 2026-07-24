# Experimental public sdmTMB interface for the Quadra SPDE backend.

.quadra_native_covariance <- function(object, par, parameter_names) {
  result <- object$covariance(par)
  matrix_names <- list(parameter_names, parameter_names)
  dimnames(result$hessian) <- matrix_names
  dimnames(result$covariance) <- matrix_names
  dimnames(result$correlation) <- matrix_names
  names(result$eigenvalues) <- NULL
  names(result$steps) <- parameter_names
  if (!isTRUE(result$success)) {
    warning(result$message, call. = FALSE)
    result$covariance[,] <- NA_real_
  }
  list(
    hessian = result$hessian,
    covariance = result$covariance,
    diagnostics = list(
      steps = result$steps,
      eigenvalues = result$eigenvalues,
      positive_definite = result$positive_definite,
      condition_number = result$condition_number,
      correlation = result$correlation,
      success = result$success,
      message = result$message
    ),
    derived = data.frame(
      term = result$derived$name,
      estimate = result$derived$estimate,
      std.error = result$derived$std_error,
      row.names = NULL
    ),
    derived_success = result$derived$success,
    derived_message = result$derived$message,
    random_effects = result$random_effects
  )
}

.sdmTMB_quadra_fit <- function(
    formula, data, mesh, family, spatial, spatiotemporal, spatial_model,
    time, weights, offset, reml, anisotropy, share_range, control, do_fit, call
) {
  if (!inherits(data, "data.frame") || !inherits(mesh, "sdmTMBmesh")) {
    stop("Quadra requires a data frame and an SPDE mesh from make_mesh()",
      call. = FALSE
    )
  }
  gaussian_family <- identical(family$family, "gaussian") &&
    identical(family$link, "identity")
  poisson_family <- identical(family$family, "poisson") &&
    identical(family$link, "log")
  nb2_family <- identical(family$family, "nbinom2") &&
    identical(family$link, "log")
  count_family <- poisson_family || nb2_family
  if (!gaussian_family && !count_family) {
    stop("The experimental Quadra backend supports gaussian(identity), poisson(log), and nbinom2(log)",
      call. = FALSE
    )
  }
  spatiotemporal_value <- if (length(spatiotemporal) > 1L && is.null(time)) {
    "off"
  } else {
    tolower(spatiotemporal[1L])
  }
  st_iid <- !is.null(time) && identical(spatiotemporal_value, "iid")
  st_ar1 <- !is.null(time) && identical(spatiotemporal_value, "ar1")
  st_rw <- !is.null(time) && identical(spatiotemporal_value, "rw")
  st_supported <- st_iid || st_ar1 || st_rw
  share_range_value <- isTRUE(share_range[1L])
  if ((!is.null(time) && !st_supported) || isTRUE(reml) ||
      isTRUE(anisotropy) || !identical(tolower(spatial_model[1L]), "spde")) {
    stop(
      paste(
        "The experimental Quadra backend currently supports an isotropic",
        "SPDE with optional count IID, AR1, or RW spatiotemporal fields,",
        "without REML"
      ),
      call. = FALSE
    )
  }
  spatial_value <- spatial[1L]
  if (!identical(tolower(spatial_value), "on")) {
    stop("The experimental Quadra backend currently requires spatial = \"on\"",
      call. = FALSE
    )
  }
  if (!identical(spatiotemporal_value, "off") && !st_supported) {
    stop("The experimental Quadra backend supports IID, AR1, and RW spatiotemporal fields",
      call. = FALSE
    )
  }
  if (st_supported && !count_family && !gaussian_family) {
    stop("Quadra spatiotemporal fields require a supported likelihood",
      call. = FALSE
    )
  }
  if (st_supported && (!is.character(time) || length(time) != 1L ||
      !time %in% names(data))) {
    stop("time must name one column in data", call. = FALSE)
  }

  model_frame <- stats::model.frame(formula, data = data,
    na.action = stats::na.fail
  )
  terms_object <- stats::terms(model_frame)
  response <- as.double(stats::model.response(model_frame))
  X <- stats::model.matrix(terms_object, model_frame)
  formula_offset <- stats::model.offset(model_frame)
  if (is.null(formula_offset)) formula_offset <- numeric(length(response))
  supplied_offset <- if (is.null(offset)) numeric(length(response)) else offset
  supplied_offset <- as.double(supplied_offset)
  if (length(supplied_offset) != length(response)) {
    stop("offset must have one value per observation", call. = FALSE)
  }
  model_offset <- as.double(formula_offset) + supplied_offset
  model_weights <- if (is.null(weights)) {
    rep(1, length(response))
  } else {
    as.double(weights)
  }
  if (length(model_weights) != length(response) ||
      any(!is.finite(model_weights) | model_weights < 0)) {
    stop("weights must be finite, nonnegative, and match the data",
      call. = FALSE
    )
  }

  A <- mesh$A_st
  if (nrow(A) != length(response)) {
    stop(
      paste(
        "mesh$A_st does not match the model data; rebuild the mesh from the",
        "same rows supplied to sdmTMB()"
      ),
      call. = FALSE
    )
  }
  mesh_span <- max(
    diff(range(mesh$mesh$loc[, 1L])),
    diff(range(mesh$mesh$loc[, 2L]))
  )
  initial_range <- if (is.finite(mesh_span) && mesh_span > 0) {
    mesh_span / 3
  } else {
    1
  }
  transformed_response <- if (count_family) log(response + 0.1) else response
  response_sd <- stats::sd(transformed_response)
  if (!is.finite(response_sd) || response_sd <= 0) response_sd <- 1
  beta_start <- stats::coef(stats::lm.fit(X, transformed_response))
  beta_start[!is.finite(beta_start)] <- 0
  if (gaussian_family && !st_supported) {
    start <- c(
      beta_start, 0, log(sqrt(8) / initial_range), log(response_sd)
    )
    quadra_object <- .make_quadra_poisson_spde_object(
      X, response, A, mesh$spde, offset = model_offset,
      weights = model_weights, start = start, gaussian = TRUE
    )
    parameter_names <- c(
      colnames(X), "log_tau", "log_kappa", "log_sigma"
    )
  } else if (st_supported) {
    time_levels <- sort(unique(data[[time]]))
    time_index <- match(data[[time]], time_levels) - 1L
    start <- c(
      beta_start, 0, 0, log(sqrt(8) / initial_range),
      if (!share_range_value) log(sqrt(8) / initial_range),
      if (st_ar1) 0.1,
      if (nb2_family) log(10),
      if (gaussian_family) log(response_sd)
    )
    quadra_object <- .make_quadra_poisson_spatiotemporal_iid_object(
      X, response, A, mesh$spde, time_index = time_index,
      offset = model_offset, start = start, ar1 = st_ar1, rw = st_rw,
      share_range = share_range_value, weights = model_weights,
      nb2 = nb2_family, gaussian = gaussian_family
    )
    parameter_names <- c(
      colnames(X), "log_tau_spatial", "log_tau_spatiotemporal", "log_kappa",
      if (!share_range_value) "log_kappa_spatiotemporal",
      if (st_ar1) "ar1_phi", if (nb2_family) "log_phi",
      if (gaussian_family) "log_sigma"
    )
  } else {
    start <- c(
      beta_start, 0, log(sqrt(8) / initial_range),
      if (nb2_family) log(10)
    )
    quadra_object <- .make_quadra_poisson_spde_object(
      X, response, A, mesh$spde, offset = model_offset,
      weights = model_weights, start = start, nb2 = nb2_family
    )
    parameter_names <- c(
      colnames(X), "log_tau", "log_kappa", if (nb2_family) "log_phi"
    )
  }
  names(quadra_object$par) <- parameter_names

  nlminb_names <- c(
    "eval.max", "iter.max", "trace", "abs.tol", "rel.tol", "x.tol",
    "xf.tol", "step.min", "step.max", "sing.tol", "scale"
  )
  optimizer_control <- control[intersect(names(control), nlminb_names)]
  quadra_optimizer <- match.arg(
    tolower(control$quadra_optimizer %||% "nlminb"),
    c("nlminb", "lbfgs")
  )
  if (isTRUE(do_fit)) {
    lower <- rep(-Inf, length(quadra_object$par))
    upper <- rep(Inf, length(quadra_object$par))
    if (count_family) {
      lower[] <- -20
      upper[] <- 20
      lower[(ncol(X) + 1L):length(lower)] <- -10
      upper[(ncol(X) + 1L):length(upper)] <- 10
      if (st_supported) {
        upper[ncol(X) + 2L] <- 15
      }
      if (st_ar1) {
        rho_position <- length(lower) - as.integer(nb2_family)
        lower[rho_position] <- stats::qlogis((1 - 0.999) / 2)
        upper[rho_position] <- stats::qlogis((1 + 0.999) / 2)
      }
    }
    if (identical(quadra_optimizer, "lbfgs")) {
      model <- quadra_object$lbfgs(
        quadra_object$par,
        max_iterations = as.integer(control$iter.max %||% 100L),
        memory = as.integer(control$quadra_lbfgs_memory %||% 7L),
        gradient_tolerance =
          as.double(control$quadra_gradient_tolerance %||% 1e-4)
      )
    } else {
      model <- stats::nlminb(
        quadra_object$par, quadra_object$fn, quadra_object$gr,
        lower = lower, upper = upper,
        control = optimizer_control
      )
    }
    names(model$par) <- parameter_names
    report <- quadra_object$report(model$par)
    if (isTRUE(control$getsd)) {
      native_uncertainty <- .quadra_native_covariance(
        quadra_object, model$par, parameter_names
      )
      hessian_result <- native_uncertainty$diagnostics
      hessian_result$hessian <- native_uncertainty$hessian
      covariance <- native_uncertainty$covariance
      derived_uncertainty <- native_uncertainty$derived
      random_effect_uncertainty <- native_uncertainty$random_effects
      if (!isTRUE(native_uncertainty$derived_success)) {
        warning(native_uncertainty$derived_message, call. = FALSE)
      }
      if (!isTRUE(random_effect_uncertainty$success)) {
        warning(random_effect_uncertainty$message, call. = FALSE)
      }
    } else {
      covariance <- NULL
      hessian_result <- NULL
      derived_uncertainty <- NULL
      random_effect_uncertainty <- NULL
    }
  } else {
    model <- NULL
    report <- NULL
    covariance <- NULL
    hessian_result <- NULL
    derived_uncertainty <- NULL
    random_effect_uncertainty <- NULL
  }

  structure(
    list(
      call = call,
      formula = formula,
      terms = terms_object,
      xlevels = stats::.getXlevels(terms_object, model_frame),
      contrasts = attr(X, "contrasts"),
      data = data,
      model_frame = model_frame,
      X = X,
      response = response,
      offset = model_offset,
      spde = mesh,
      family = family,
      spatial = "on",
      spatiotemporal = if (st_ar1) {
        "ar1"
      } else if (st_rw) {
        "rw"
      } else if (st_iid) {
        "iid"
      } else {
        "off"
      },
      time = if (st_supported) time else NULL,
      time_levels = if (st_supported) time_levels else NULL,
      share_range = share_range_value,
      backend = "quadra",
      optimizer = quadra_optimizer,
      quadra_obj = quadra_object,
      tmb_obj = quadra_object,
      model = model,
      report = report,
      vcov = covariance,
      hessian = if (is.null(hessian_result)) NULL else hessian_result$hessian,
      hessian_diagnostics = hessian_result,
      derived_uncertainty = derived_uncertainty,
      random_effect_uncertainty = random_effect_uncertainty,
      field_se = if (is.null(random_effect_uncertainty)) {
        NULL
      } else {
        random_effect_uncertainty$std_error[seq_len(ncol(A))]
      },
      spatiotemporal_field_se = if (
        is.null(random_effect_uncertainty) || !st_supported
      ) {
        NULL
      } else {
        matrix(
          random_effect_uncertainty$std_error[-seq_len(ncol(A))],
          nrow = ncol(A), ncol = length(time_levels)
        )
      }
    ),
    class = c("sdmTMB_quadra", "sdmTMB")
  )
}

#' @export
print.sdmTMB_quadra <- function(x, ...) {
  model_type <- if (x$spatiotemporal %in% c("iid", "ar1", "rw")) {
    paste("Spatial and", toupper(x$spatiotemporal), "spatiotemporal model")
  } else {
    "Spatial model"
  }
  cat(model_type, "fit by maximum likelihood with Quadra\n\n")
  cat("Family:", x$family$family, "(", x$family$link, ")\n")
  cat("Formula:", deparse(x$formula), "\n")
  if (is.null(x$model)) {
    cat("Model object created without optimization.\n")
    return(invisible(x))
  }
  cat("\nFixed effects:\n")
  fixed_estimate <- stats::setNames(
    x$model$par[seq_len(ncol(x$X))], colnames(x$X)
  )
  if (!is.null(x$vcov)) {
    fixed_se <- sqrt(diag(vcov(x)))
    print(cbind(Estimate = fixed_estimate, `Std. Error` = fixed_se))
  } else {
    print(fixed_estimate)
  }
  cat("\nSpatial parameters:\n")
  if (identical(x$family$family, "gaussian") &&
      identical(x$spatiotemporal, "off")) {
    cat("  sigma:", x$report$sigma, "\n")
  }
  if (identical(x$family$family, "gaussian") &&
      x$spatiotemporal %in% c("iid", "ar1", "rw")) {
    cat("  sigma:", x$report$sigma, "\n")
  }
  if (x$spatiotemporal %in% c("iid", "ar1", "rw")) {
    cat(
      "  spatiotemporal precision scale:",
      exp(x$model$par[ncol(x$X) + 2L]), "\n"
    )
    if (identical(x$spatiotemporal, "ar1")) {
      cat("  AR1 correlation:", x$report$rho, "\n")
    }
  }
  if (identical(x$family$family, "nbinom2")) {
    cat("  NB2 dispersion:", x$report$phi, "\n")
  }
  if (identical(x$share_range, FALSE) &&
      x$spatiotemporal %in% c("iid", "ar1", "rw")) {
    cat("  spatial range:", x$report$spatial_range, "\n")
    cat("  spatiotemporal range:", x$report$spatiotemporal_range, "\n")
  } else {
    cat("  range:", x$report$range, "\n")
  }
  cat("  objective:", x$model$objective, "\n")
  cat("  convergence:", x$model$convergence, "\n")
  if (!is.null(x$derived_uncertainty)) {
    cat("\nDerived-parameter inference:\n")
    print(x$derived_uncertainty, row.names = FALSE)
  }
  invisible(x)
}

#' @export
coef.sdmTMB_quadra <- function(object, ...) {
  stats::setNames(
    object$model$par[seq_len(ncol(object$X))], colnames(object$X)
  )
}

#' @export
fixef.sdmTMB_quadra <- function(object, ...) coef.sdmTMB_quadra(object)

#' @export
vcov.sdmTMB_quadra <- function(object, complete = FALSE, ...) {
  if (is.null(object$vcov)) {
    stop("Covariance was not calculated because control$getsd was FALSE",
      call. = FALSE
    )
  }
  if (isTRUE(complete)) return(object$vcov)
  object$vcov[seq_len(ncol(object$X)), seq_len(ncol(object$X)), drop = FALSE]
}

#' @export
fitted.sdmTMB_quadra <- function(object, ...) object$report$mu_i

#' @export
predict.sdmTMB_quadra <- function(
    object, newdata = NULL, type = c("link", "response"),
    se_fit = FALSE, level = 0.95, ...
) {
  type <- match.arg(type)
  if (length(level) != 1L || !is.finite(level) ||
      level <= 0 || level >= 1) {
    stop("level must be one finite number strictly between 0 and 1",
      call. = FALSE
    )
  }
  if (is.null(newdata)) {
    estimate <- object$report$eta_i
    prediction_X <- object$X
    prediction_A <- object$quadra_obj$A
    prediction_time_index <- object$quadra_obj$time_index
  } else {
    new_frame <- stats::model.frame(
      stats::delete.response(object$terms), newdata,
      xlev = object$xlevels, na.action = stats::na.pass
    )
    new_X <- stats::model.matrix(
      stats::delete.response(object$terms), new_frame,
      contrasts.arg = object$contrasts
    )
    xy_cols <- object$spde$xy_cols
    if (!all(xy_cols %in% names(newdata))) {
      stop("newdata must contain the mesh coordinate columns", call. = FALSE)
    }
    new_A <- fmesher::fm_basis(
      object$spde$mesh,
      loc = as.matrix(newdata[, xy_cols, drop = FALSE])
    )
    prediction_X <- new_X
    prediction_A <- new_A
    prediction_time_index <- NULL
    estimate <- as.vector(
      new_X %*% coef.sdmTMB_quadra(object) +
        new_A %*% object$report$field
    )
    if (object$spatiotemporal %in% c("iid", "ar1", "rw")) {
      if (!object$time %in% names(newdata)) {
        stop("newdata must contain the fitted time column", call. = FALSE)
      }
      time_index <- match(newdata[[object$time]], object$time_levels)
      if (anyNA(time_index)) {
        stop("newdata contains an unseen time value", call. = FALSE)
      }
      prediction_time_index <- time_index - 1L
      estimate <- estimate + vapply(seq_len(nrow(newdata)), function(i) {
        as.numeric(
          new_A[i, , drop = FALSE] %*%
            object$report$spatiotemporal_field[, time_index[i]]
        )
      }, numeric(1))
    }
  }
  link_estimate <- estimate
  uncertainty <- NULL
  if (isTRUE(se_fit)) {
    if (is.null(object$vcov)) {
      stop(
        "Prediction standard errors require control$getsd = TRUE when fitting",
        call. = FALSE
      )
    }
    if (!isTRUE(object$hessian_diagnostics$success)) {
      stop(
        "Prediction standard errors are unavailable because fixed-parameter covariance inference failed",
        call. = FALSE
      )
    }
    uncertainty <- if (object$spatiotemporal %in% c("iid", "ar1", "rw")) {
      object$quadra_obj$prediction_uncertainty(
        prediction_X, prediction_A, prediction_time_index
      )
    } else {
      object$quadra_obj$prediction_uncertainty(
        prediction_X, prediction_A
      )
    }
    if (!isTRUE(uncertainty$success)) {
      stop(uncertainty$message, call. = FALSE)
    }
  }
  if (identical(type, "response")) estimate <- object$family$linkinv(estimate)
  out <- data.frame(est = estimate)
  if (!is.null(uncertainty)) {
    link_se <- uncertainty$std_error
    critical_value <- stats::qnorm((1 + level) / 2)
    link_lwr <- link_estimate - critical_value * link_se
    link_upr <- link_estimate + critical_value * link_se
    out$est_se <- link_se
    if (identical(type, "response")) {
      out$est_se <- abs(object$family$mu.eta(link_estimate)) * out$est_se
      out$lwr <- object$family$linkinv(link_lwr)
      out$upr <- object$family$linkinv(link_upr)
    } else {
      out$lwr <- link_lwr
      out$upr <- link_upr
    }
    attr(out, "level") <- level
    attr(out, "quadra_uncertainty") <- uncertainty
  }
  out
}

#' @export
logLik.sdmTMB_quadra <- function(object, ...) {
  value <- structure(
    -object$model$objective,
    nobs = length(object$response),
    df = length(object$model$par),
    class = "logLik"
  )
  value
}

#' @export
nobs.sdmTMB_quadra <- function(object, ...) length(object$response)

#' @export
formula.sdmTMB_quadra <- function(x, ...) x$formula

#' @export
family.sdmTMB_quadra <- function(object, ...) object$family
