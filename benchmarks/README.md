# Quadra backend benchmarks

## 2026-07-24: marginal prediction uncertainty

These benchmarks measure the experimental Quadra backend after adding full
marginal latent-field and prediction uncertainty. Prediction variance includes
both the conditional random-field component and fixed-parameter propagation
through the implicit derivative `du/dtheta`.

### Environment

| Item | Value |
|---|---|
| Package | sdmTMB 1.1.0.9000 |
| Git baseline | `891d8e7f` plus the current Quadra working-tree changes |
| R | 4.4.3 |
| Platform | aarch64-apple-darwin20 |
| OS | macOS 15.7.7, Darwin 24.6.0, arm64 |
| Compiler | Apple clang 16, C++17 |

### Workload

Both models use the bundled Pacific cod data and mesh:

| Property | Value |
|---|---:|
| Observations | 969 |
| Mesh nodes | 76 |
| Time levels | 4 |
| Formula | `log(density + 0.1) ~ depth_scaled` |
| Likelihood | Gaussian with identity link |
| Spatial model | Isotropic SPDE |
| Spatiotemporal model | IID |

Complete fit timings include optimization, native fixed-effect covariance,
derived-parameter uncertainty, implicit random-effect derivatives, and marginal
field uncertainty. Prediction timings use `predict(se_fit = TRUE)` and therefore
include mesh projection, sparse conditional-variance solves, fixed-parameter
propagation, and confidence-interval construction.

Each prediction size received one warm-up call. Reported values are three-call
elapsed-time summaries.

### Results

Complete fit with uncertainty:

| Model | Random coefficients | Elapsed seconds |
|---|---:|---:|
| Spatial | 76 | 0.116 |
| Spatial + IID spatiotemporal | 380 | 0.836 |

Spatial prediction uncertainty:

| Predictions | Median seconds | Minimum | Maximum |
|---:|---:|---:|---:|
| 100 | 0.003 | 0.002 | 0.004 |
| 1,000 | 0.006 | 0.005 | 0.006 |
| 5,000 | 0.017 | 0.016 | 0.019 |

Spatial + IID spatiotemporal prediction uncertainty:

| Predictions | Median seconds | Minimum | Maximum |
|---:|---:|---:|---:|
| 100 | 0.008 | 0.007 | 0.008 |
| 1,000 | 0.048 | 0.048 | 0.049 |
| 5,000 | 0.308 | 0.307 | 0.309 |

### Interpretation

Prediction time is approximately linear in the number of requested locations,
as expected from one sparse Hessian solve per projected linear predictor. The
implementation does not construct a dense random-effect covariance matrix.
The 380-coefficient spatiotemporal model is consequently slower per prediction
than the 76-coefficient spatial model, while remaining well below one second
for 5,000 predictions on this workload.

These are development-machine microbenchmarks, not cross-platform performance
guarantees. Future benchmark entries should retain this workload and add larger
meshes so changes in sparse factorization, solve batching, and prediction
chunking can be compared over time.

### Reproduction

Install the current source package, then run:

```r
library(sdmTMB)

benchmark_predictions <- function(fit, data, sizes) {
  do.call(rbind, lapply(sizes, function(n) {
    newdata <- data[
      rep(seq_len(nrow(data)), length.out = n), ,
      drop = FALSE
    ]
    invisible(predict(fit, newdata = newdata, se_fit = TRUE))
    elapsed <- replicate(
      3,
      system.time(
        predict(fit, newdata = newdata, se_fit = TRUE)
      )[["elapsed"]]
    )
    data.frame(
      predictions = n,
      median_seconds = median(elapsed),
      minimum_seconds = min(elapsed),
      maximum_seconds = max(elapsed)
    )
  }))
}

control <- sdmTMBcontrol(
  iter.max = 100, eval.max = 200, getsd = TRUE
)

spatial <- sdmTMB(
  log(density + 0.1) ~ depth_scaled,
  data = pcod_2011,
  mesh = pcod_mesh_2011,
  family = gaussian(),
  backend = "quadra",
  control = control
)

spatiotemporal <- sdmTMB(
  log(density + 0.1) ~ depth_scaled,
  data = pcod_2011,
  mesh = pcod_mesh_2011,
  time = "year",
  spatiotemporal = "iid",
  family = gaussian(),
  backend = "quadra",
  control = control
)

benchmark_predictions(spatial, pcod_2011, c(100, 1000, 5000))
benchmark_predictions(spatiotemporal, pcod_2011, c(100, 1000, 5000))
```

## 2026-07-24: Quadra versus TMB with peak RSS

This comparison uses the same environment and Pacific cod workloads described
above. Each backend ran in a fresh R process. The process performed one complete
fit with uncertainty, one prediction warm-up, and three measured calls for
5,000 predictions with `se_fit = TRUE`.

Peak resident set size (RSS) was sampled every 5 milliseconds from the isolated
child process using `ps::ps_memory_info()`. It is therefore a sampled peak,
rather than the kernel-accounted peak reported by `/usr/bin/time`. The latter
could not read the required macOS kernel counter in the sandbox. RSS covers the
entire fit-and-predict process and includes the R runtime, loaded package code,
model state, warm-up, and prediction results.

The exact worker workload is retained in
[`backend-worker.R`](backend-worker.R).

### Spatial model

| Backend | Fit seconds | Prediction median seconds | Prediction minimum | Prediction maximum | Peak RSS (MiB) |
|---|---:|---:|---:|---:|---:|
| Quadra | 0.056 | 0.005 | 0.005 | 0.006 | 304.9 |
| TMB | 0.131 | 1.356 | 1.348 | 1.370 | 1,000.5 |

For this workload, Quadra prediction was approximately 270 times faster and
used approximately 30% of TMB's peak RSS.

### Spatial + IID spatiotemporal model

| Backend | Fit seconds | Prediction median seconds | Prediction minimum | Prediction maximum | Peak RSS (MiB) |
|---|---:|---:|---:|---:|---:|
| Quadra | 0.371 | 0.202 | 0.200 | 0.203 | 352.0 |
| TMB | 0.320 | 4.168 | 4.114 | 4.185 | 1,204.3 |

TMB completed the spatiotemporal fit approximately 16% faster in this run.
Quadra's subsequent marginal prediction-uncertainty calculation was
approximately 21 times faster and its full-process peak RSS was approximately
29% of TMB's.

### Complete Quadra diagnostics suite

The following sections reproduce the full format of the
[FIMS Quadra diagnostics report](https://github.com/NOAA-FIMS/FIMS/blob/feature/quadra-backend/tests/quadra-diagnostics.md):
model health, optimization, effective structure, backend recommendation,
uncertainty graph, latent states, spectral structure, and the most influential
random effects. Curvature diagnostics analyze the random-effect Hessian,
\(H_{uu}\). Fixed gradients are exact Laplace gradients.

#### Spatial model

##### Executive summary and model health

- Overall status: `HEALTHY`
- Confidence: `HIGH`
- Optimization quality: `EXCELLENT`
- Gradient quality: `PASS`
- Curvature: `PASS`
- Conditioning: `EXCELLENT`

| Check | Status |
|---|---:|
| Optimization | `PASS` |
| Gradient | `PASS` |
| Curvature | `PASS` |
| Conditioning | `EXCELLENT` |
| Overall | `HEALTHY` |

##### Optimization

| Quantity | Value |
|---|---:|
| Laplace objective | `2355.154717` |
| Joint objective | `2437.073811` |
| Fixed gradient norm | `0.000314101` |
| Random gradient norm | `1.336e-19` |
| Joint gradient norm | `0.000314101` |
| Maximum-gradient parameter | `log_sigma` |
| Maximum absolute gradient | `0.000311984` |
| Outer iterations | `18` |
| Objective / gradient evaluations | `22 / 19` |
| Diagnostic convergence | `yes` |

##### Curvature and effective structure

| Quantity | Value |
|---|---:|
| Random effects | `76` |
| Positive definite | `yes` |
| Condition number | `91.8849` |
| Structural density | `0.209141` |
| Structural nonzeros | `1,208` |
| Entries retaining 95% curvature | `286` |
| Bandwidth retaining 95% curvature | `27` |
| Hessian jitter | `0` |

##### Quadra recommendation

- Detected structure: `sparse_pattern`
- Factorization backend: `sparse_ldlt`
- Full structural bandwidth: `71`
- Reason: fill ratio is below the dense threshold and bandwidth exceeds the
  banded-backend threshold; sparse LDLT is preferred.

##### Uncertainty and correlation graph

- Fixed covariance: `SUCCESS`
- Random-effect marginal uncertainty: `SUCCESS`
- Maximum absolute latent correlation: `0.632735`
- Correlation graph nodes / edges: `76 / 4`
- Connected components: `72`
- Largest component: `3`
- Maximum degree: `2`
- Maximum-degree parameter: `spatial_node_56`

##### Latent states

- Count: `76`
- Mean: `0.0696594`
- Standard deviation: `1.67948`
- Range: `[-5.33399, 3.82142]`

##### Spectral structure

- Eigenvalue range: `[0.0429430, 3.94582]`
- Entropy effective rank: `53.4746`
- Largest eigenvalue share: `0.0423232`
- Eigen-directions for 50% curvature: `18`
- Eigen-directions for 90% curvature: `44`
- Eigen-directions for 95% curvature: `51`
- Eigen-directions for 99% curvature: `67`

##### Most influential random effects

| Rank | Parameter | Importance share | Conditional SD |
|---:|---|---:|---:|
| 1 | `spatial_node_41` | `0.0271870` | `0.603420` |
| 2 | `spatial_node_21` | `0.0261753` | `0.678483` |
| 3 | `spatial_node_17` | `0.0247628` | `0.610667` |
| 4 | `spatial_node_33` | `0.0241552` | `0.652813` |
| 5 | `spatial_node_3` | `0.0240511` | `0.623786` |

#### Spatial + IID spatiotemporal model

##### Executive summary and model health

- Overall status: `HEALTHY`
- Confidence: `MODERATE`
- Optimization quality: `GOOD`
- Gradient quality: `PASS`
- Curvature: `PASS`
- Conditioning: `GOOD`

| Check | Status |
|---|---:|
| Optimization | `PASS` |
| Gradient | `PASS` |
| Curvature | `PASS` |
| Conditioning | `GOOD` |
| Overall | `HEALTHY` |

##### Optimization

| Quantity | Value |
|---|---:|
| Laplace objective | `2347.262794` |
| Joint objective | `2683.936238` |
| Fixed gradient norm | `0.000284908` |
| Random gradient norm | `3.348e-19` |
| Joint gradient norm | `0.000284908` |
| Maximum-gradient parameter | `log_sigma` |
| Maximum absolute gradient | `0.000268784` |
| Outer iterations | `24` |
| Objective / gradient evaluations | `36 / 25` |
| Diagnostic convergence | `yes` |

##### Curvature and effective structure

| Quantity | Value |
|---|---:|
| Random effects | `380` |
| Positive definite | `yes` |
| Condition number | `127.159` |
| Structural density | `0.0624515` |
| Structural nonzeros | `9,018` |
| Entries retaining 95% curvature | `3,591` |
| Bandwidth retaining 95% curvature | `249` |
| Hessian jitter | `0` |

##### Quadra recommendation

- Detected structure: `sparse_pattern`
- Factorization backend: `sparse_ldlt`
- Full structural bandwidth: `370`
- Reason: low fill ratio and wide cross-field coupling favor sparse LDLT.

##### Uncertainty and correlation graph

- Fixed covariance: `SUCCESS`
- Random-effect marginal uncertainty: `SUCCESS`
- Maximum absolute latent correlation: `0.667324`
- Correlation graph nodes / edges: `380 / 20`
- Connected components: `360`
- Largest component: `3`
- Maximum degree: `2`
- Maximum-degree parameter: `spatial_node_56`

##### Latent states

- Count: `380`
- Mean: `0.0163418`
- Standard deviation: `0.813674`
- Range: `[-5.07553, 3.90755]`

##### Spectral structure

- Eigenvalue range: `[0.0435668, 5.53990]`
- Entropy effective rank: `310.843`
- Largest eigenvalue share: `0.0106014`
- Eigen-directions for 50% curvature: `107`
- Eigen-directions for 90% curvature: `270`
- Eigen-directions for 95% curvature: `306`
- Eigen-directions for 99% curvature: `350`

##### Most influential random effects

| Rank | Parameter | Importance share | Conditional SD |
|---:|---|---:|---:|
| 1 | `spatial_node_17` | `0.00695661` | `0.742428` |
| 2 | `spatial_node_3` | `0.00663857` | `0.728211` |
| 3 | `spatial_node_33` | `0.00658423` | `0.759298` |
| 4 | `spatial_node_21` | `0.00654241` | `0.787878` |
| 5 | `spatial_node_5` | `0.00632948` | `0.800674` |

### Scope

These numbers compare the current user-facing implementations, not isolated
linear algebra kernels. In particular, both prediction paths include their
backend-specific setup, reporting, projection, and result construction. The
RSS field is intentionally a full-process measure because it captures the
actual memory pressure a package user experiences.

## 2026-07-24: spatiotemporal fit-path optimization

Profiling separated complete optimization from post-fit uncertainty. Before
the change, the Gaussian IID spatiotemporal workload took 0.557 seconds in
Quadra with `getsd = FALSE`, compared with 0.267 seconds in TMB. Both backends
reached objective 2347.263 with similar evaluation counts.

`nlminb` requested 36 objective values but only 25 gradients. Quadra previously
computed the full exact Laplace gradient during every objective call so the
following gradient call could reuse it. The persistent native states now cache
the profiled center—optimized random effects, Hessian, objective, and reports.
Objective calls request only the value; a subsequent gradient request reuses
that center without repeating the random-effect Newton solve.

| Version | Quadra optimization seconds | Change from baseline |
|---|---:|---:|
| Before native profiled-center cache | 0.557 | — |
| After native profiled-center cache | 0.445 | -20.1% |
| TMB reference | 0.267 | — |

The remaining split measured approximately 8 milliseconds per changed profiled
objective and 5 milliseconds per exact-gradient calculation before this
optimization. Further work should cache immutable SPDE/projection structures
and reuse the terminal sparse factorization, with each change guarded by the
existing TMB objective, gradient, field-variance, and prediction-variance
parity fixtures.

The terminal factorization is now also reused for Laplace log-determinants,
Takahashi selected inverses, exact-gradient trace contractions, and post-fit
implicit derivatives. The same complete `getsd = TRUE` workload took 0.562
seconds, compared with 0.736 seconds originally—a 23.6% reduction. Post-fit
inference now tapes only `H_u_theta` and solves all `du/dtheta` columns through
the cached terminal LDLT; it no longer reruns random-effect optimization or
refactorizes `H_uu` to obtain marginal field variances.

## 2026-07-24: selected-inverse prediction uncertainty

Prediction uncertainty now computes each local random-effect variance
contribution from the terminal Takahashi selected inverse whenever all required
entries are available. Rows outside that sparse selected-inverse pattern retain
an exact fallback, solved in bounded blocks of 256 right-hand sides. Fixed-effect
and implicit random-effect parameter sensitivities are unchanged.

The focused benchmark used 5,000 repeated Pacific cod prediction rows with
`se_fit = TRUE`. Times are the median of three calls after fitting the model.

| Model | Previous seconds | Selected-inverse seconds | Reduction |
|---|---:|---:|---:|
| Spatial Gaussian | 0.014 | 0.006 | 57.1% |
| Spatial + IID spatiotemporal Gaussian | 0.305 | 0.209 | 31.5% |

The complete Quadra backend test suite passes after this change, including
prediction standard-error parity fixtures against TMB.

## 2026-07-24: Quadra native L-BFGS

The benchmark harness now includes `quadra-lbfgs`, which runs Quadra's native
limited-memory BFGS two-loop recursion and Armijo backtracking line search
directly against the persistent profiled model state. This is distinct from the
default Quadra configuration, which exposes the same native objective and exact
gradient to R's `nlminb`.

Each row ran in a fresh process and includes fitting, fixed and random-effect
uncertainty, a 5,000-row prediction warm-up, and three measured marginal
prediction calls. Peak RSS was sampled every 5 milliseconds. The benchmark
output also reports the maximum absolute fixed-effect gradient so convergence
quality can be compared alongside runtime.

### Spatial model

| Backend / optimizer | Fit seconds | Prediction median seconds | Peak RSS (MiB) | Max. \|gradient\| |
|---|---:|---:|---:|---:|
| Quadra + `nlminb` | 0.056 | 0.005 | 304.9 | 3.12e-4 |
| Quadra + native L-BFGS | 0.080 | 0.005 | 300.0 | 1.06e-5 |
| TMB + `nlminb` | 0.131 | 1.356 | 1,000.5 | 7.11e-4 |

The native L-BFGS fit converged in 30 iterations to objective
2355.1547171330.

### Spatial + IID spatiotemporal model

| Backend / optimizer | Fit seconds | Prediction median seconds | Peak RSS (MiB) | Max. \|gradient\| |
|---|---:|---:|---:|---:|
| Quadra + `nlminb` | 0.371 | 0.202 | 352.0 | 2.69e-4 |
| Quadra + native L-BFGS | 0.487 | 0.203 | 348.3 | 6.38e-5 |
| TMB + `nlminb` | 0.320 | 4.168 | 1,204.3 | 4.41e-5 |

The native L-BFGS fit converged in 41 iterations to objective
2347.2627939846.

The native optimizer uses value-only profiled evaluations during backtracking
and requests one exact Laplace gradient at each accepted point. Its
steepest-descent initialization is normalized when no curvature history is
available; the unscaled initial gradients had norms of approximately 247 and
313 and previously caused a long sequence of rejected profiles.

Gaussian fits now also use the quadratic structure of the conditional
random-effect objective. A full Newton step is exact, so Quadra updates the
objective and gradient algebraically instead of replaying the random-effect
Hessian and running an Armijo line search at the accepted point. Report-free
bridge models skip a redundant ordinary-double model evaluation, Hessian
matrices reuse their compressed sparse pattern, and SPDE precision
factorizations reuse symbolic analysis. Together these changes reduced the
complete spatial fit from approximately 0.095 to 0.056 seconds and the IID
spatiotemporal fit from approximately 0.603 to 0.371 seconds. Quadra is now
approximately 16% slower than TMB on the complete IID spatiotemporal fit,
instead of approximately 65% slower in the earlier measurement.

Reproduce the complete comparison with:

```sh
Rscript benchmarks/benchmark-suite.R 5000
```

The worker also accepts the native optimizer directly:

```sh
Rscript benchmarks/backend-worker.R quadra-lbfgs spatial 5000
Rscript benchmarks/backend-worker.R quadra-lbfgs spatiotemporal 5000
```
