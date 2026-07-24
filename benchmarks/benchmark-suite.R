args <- commandArgs(trailingOnly = TRUE)
n_predictions <- if (length(args)) as.integer(args[[1L]]) else 5000L
if (is.na(n_predictions) || n_predictions < 1L) {
  stop("prediction count must be a positive integer")
}
if (!requireNamespace("ps", quietly = TRUE)) {
  stop("the ps package is required to sample peak RSS")
}

worker <- file.path("benchmarks", "backend-worker.R")
backends <- c("quadra", "quadra-lbfgs", "tmb")
models <- c("spatial", "spatiotemporal")

run_worker <- function(backend, model) {
  job <- parallel::mcparallel(
    capture.output({
      Sys.setenv(
        SDMTMB_BENCHMARK_BACKEND = backend,
        SDMTMB_BENCHMARK_MODEL = model,
        SDMTMB_BENCHMARK_PREDICTIONS = n_predictions
      )
      source(worker, local = new.env(parent = globalenv()))
    }),
    silent = TRUE
  )
  peak_rss <- 0
  repeat {
    root <- try(ps::ps_handle(job$pid), silent = TRUE)
    handles <- if (inherits(root, "try-error")) {
      list()
    } else {
      c(list(root), tryCatch(
        ps::ps_children(root, recursive = TRUE),
        error = function(...) list()
      ))
    }
    rss <- vapply(handles, function(handle) {
      tryCatch(ps::ps_memory_info(handle)[["rss"]], error = function(...) 0)
    }, numeric(1))
    peak_rss <- max(peak_rss, sum(rss))
    collected <- parallel::mccollect(job, wait = FALSE)
    if (!is.null(collected)) break
    Sys.sleep(0.005)
  }
  lines <- unname(collected[[1L]])
  if (inherits(lines, "try-error")) stop(lines)
  values <- strsplit(lines[grepl("^[a-z_]+=", lines)], "=", fixed = TRUE)
  values <- stats::setNames(
    vapply(values, `[[`, character(1), 2L),
    vapply(values, `[[`, character(1), 1L)
  )
  data.frame(
    backend = backend,
    model = model,
    fit_seconds = as.numeric(values[["fit_seconds"]]),
    objective = as.numeric(values[["objective"]]),
    convergence = as.integer(values[["convergence"]]),
    iterations = as.integer(values[["iterations"]]),
    max_gradient = as.numeric(values[["max_gradient"]]),
    prediction_median_seconds =
      as.numeric(values[["prediction_median_seconds"]]),
    prediction_minimum_seconds =
      as.numeric(values[["prediction_minimum_seconds"]]),
    prediction_maximum_seconds =
      as.numeric(values[["prediction_maximum_seconds"]]),
    peak_rss_mib = peak_rss / 1024^2
  )
}

results <- do.call(rbind, lapply(models, function(model) {
  do.call(rbind, lapply(backends, run_worker, model = model))
}))
print(results, row.names = FALSE, digits = 6)
