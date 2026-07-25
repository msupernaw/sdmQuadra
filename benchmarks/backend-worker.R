environment_args <- Sys.getenv(c(
  "SDMTMB_BENCHMARK_BACKEND", "SDMTMB_BENCHMARK_MODEL",
  "SDMTMB_BENCHMARK_PREDICTIONS"
))
args <- if (all(nzchar(environment_args))) {
  unname(environment_args)
} else {
  commandArgs(trailingOnly = TRUE)
}
if (length(args) != 3L) {
  stop("usage: backend-worker.R BACKEND MODEL N_PREDICTIONS")
}

backend <- args[[1L]]
model <- args[[2L]]
n_predictions <- as.integer(args[[3L]])
if (!backend %in% c("quadra", "quadra-lbfgs", "tmb") ||
    !model %in% c("spatial", "spatiotemporal") ||
    is.na(n_predictions) || n_predictions < 1L) {
  stop("invalid benchmark arguments")
}

installed_entry_point <- tryCatch(
  getExportedValue("sdmTMB", "sdmTMB"),
  error = function(...) NULL
)
if (is.null(installed_entry_point) ||
    !"backend" %in% names(formals(installed_entry_point))) {
  if (!file.exists("DESCRIPTION") || !dir.exists("R")) {
    stop(
      paste(
        "the installed sdmTMB predates the Quadra backend and the worker is",
        "not running from an sdmTMB source checkout"
      )
    )
  }
  if (!requireNamespace("pkgload", quietly = TRUE)) {
    stop(
      paste(
        "the installed sdmTMB predates the Quadra backend;",
        "run `R CMD INSTALL .` or install the pkgload package"
      )
    )
  }
  pkgload::load_all(".", export_all = FALSE, helpers = FALSE, quiet = TRUE)
} else {
  library(sdmTMB)
}

if (!"backend" %in% names(formals(sdmTMB))) {
  stop("the loaded sdmTMB does not provide the Quadra backend")
}

control <- sdmTMBcontrol(
  iter.max = 100, eval.max = 200, getsd = TRUE,
  newton_loops = 0, multiphase = FALSE,
  quadra_optimizer = if (identical(backend, "quadra-lbfgs")) {
    "lbfgs"
  } else {
    "nlminb"
  }
)
arguments <- list(
  formula = log(density + 0.1) ~ depth_scaled,
  data = pcod_2011,
  mesh = pcod_mesh_2011,
  family = gaussian(),
  backend = if (identical(backend, "quadra-lbfgs")) "quadra" else backend,
  control = control
)
if (identical(model, "spatiotemporal")) {
  arguments$time <- "year"
  arguments$spatiotemporal <- "iid"
}

fit_time <- system.time({
  fit <- do.call(sdmTMB, arguments)
})[["elapsed"]]
max_gradient <- max(abs(fit$tmb_obj$gr(fit$model$par)))

newdata <- pcod_2011[
  rep(seq_len(nrow(pcod_2011)), length.out = n_predictions), ,
  drop = FALSE
]
prediction <- predict(
  fit, newdata = newdata, se_fit = TRUE, level = 0.95
)
prediction_times <- replicate(3, system.time({
  prediction <- predict(
    fit, newdata = newdata, se_fit = TRUE, level = 0.95
  )
})[["elapsed"]])
stopifnot(
  nrow(prediction) == n_predictions,
  all(is.finite(prediction$est)),
  all(is.finite(prediction$est_se))
)

cat(
  sprintf("backend=%s\n", backend),
  sprintf("optimizer=%s\n", if (identical(backend, "quadra-lbfgs")) {
    "quadra-lbfgs"
  } else {
    "nlminb"
  }),
  sprintf("model=%s\n", model),
  sprintf("predictions=%d\n", n_predictions),
  sprintf("fit_seconds=%.6f\n", fit_time),
  sprintf("objective=%.10f\n", fit$model$objective),
  sprintf("convergence=%d\n", fit$model$convergence),
  sprintf("iterations=%d\n", fit$model$iterations),
  sprintf("max_gradient=%.10g\n", max_gradient),
  sprintf("prediction_median_seconds=%.6f\n", median(prediction_times)),
  sprintf("prediction_minimum_seconds=%.6f\n", min(prediction_times)),
  sprintf("prediction_maximum_seconds=%.6f\n", max(prediction_times)),
  sep = ""
)
