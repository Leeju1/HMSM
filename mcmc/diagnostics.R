# Diagnose saved chains without loading input data or recompiling the sampler.
library(posterior)

# Sampler arrays store iterations on their last axis. Scalar draws are handled
# explicitly in extract_chain_matrix, including Armadillo column-vector output.
flatten_draws <- function(x, prefix, keep = NULL) {
  d <- dim(x)
  parameter_dim <- head(d, -1)
  parameter_ids <- seq_len(prod(parameter_dim))
  m <- matrix(x, nrow = prod(parameter_dim), ncol = tail(d, 1))
  if (!is.null(keep)) {
    parameter_ids <- keep
    m <- m[keep, , drop = FALSE]
  }
  index <- arrayInd(parameter_ids, parameter_dim)
  out <- t(m)
  colnames(out) <- apply(index, 1, function(i) paste0(prefix, "[", paste(i, collapse = ","), "]"))
  out
}

extract_chain_matrix <- function(result, transition, tau_ids) {
  active <- which(transition == 1 & row(transition) != col(transition))
  mats <- list(
    flatten_draws(result$kappa0_save, "kappa0", active),
    flatten_draws(result$kappa1_save, "kappa1", active),
    flatten_draws(result$mu_tau_save, "mu_tau"),
    flatten_draws(result$sigma2_tau_save, "sigma2_tau"),
    flatten_draws(result$alpha_save, "alpha"),
    flatten_draws(result$mu_alpha_save, "mu_alpha"),
    flatten_draws(result$sigma2_alpha_p_save, "sigma2_alpha_p"),
    flatten_draws(result$beta_key0_save, "beta_key0"),
    flatten_draws(result$beta_key1_save, "beta_key1"),
    flatten_draws(result$mu_beta_cr_save, "mu_beta_cr"),
    flatten_draws(result$sigma2_beta_cr_save, "sigma2_beta_cr"),
    matrix(as.numeric(result$sigma2_mu_tau_save), ncol = 1,
           dimnames = list(NULL, "sigma2_mu_tau")),
    flatten_draws(result$tau_save, "tau", tau_ids)
  )
  do.call(cbind, mats)
}

# Trace layout and scales follow mcmc_single_chain_diagnostics_ver7_hierB_grouped_pdf.R.
plot_trace_page <- function(values, parameter, iterations, log_scale = FALSE) {
  label <- parameter
  if (log_scale) {
    values[!is.finite(values) | values <= 0] <- NA_real_
    values <- log(values)
    label <- paste0("log(", parameter, ")")
  }
  values[!is.finite(values)] <- NA_real_
  par(mar = c(4.5, 5, 3.2, 1.2))
  if (any(is.finite(values))) {
    plot(iterations, values, type = "l", lwd = 0.45,
         xlab = "MCMC iteration", ylab = label,
         main = paste0("Trace plot: ", label), cex.main = 0.95, xaxs = "i")
    grid()
  } else {
    plot.new()
    title(main = paste0("Trace plot: ", label))
    text(0.5, 0.5, "No finite values to plot")
  }
}

# One multipage PDF per parameter family, one scalar parameter per page.
# All retained chain 1 draws and all tau values are included. No R-hat selection.
export_traceplots <- function(run_dir) {
  metadata <- readRDS(file.path(run_dir, "run_metadata.rds"))
  e <- new.env()
  load(file.path(run_dir, metadata$chain_files[1]), envir = e)
  result <- e$result
  stopifnot(result$chain_id == 1)
  blocks <- list(
    kappa0 = result$kappa0_save, kappa1 = result$kappa1_save,
    tau = result$tau_save,
    mu_tau = result$mu_tau_save, sigma2_tau = result$sigma2_tau_save,
    sigma2_mu_tau = matrix(as.numeric(result$sigma2_mu_tau_save), nrow = 1),
    alpha = result$alpha_save, mu_alpha_p = result$mu_alpha_save,
    sigma2_alpha_p = result$sigma2_alpha_p_save,
    beta_key0 = result$beta_key0_save, beta_key1 = result$beta_key1_save,
    mu_beta_cr = result$mu_beta_cr_save, sigma2_beta_cr = result$sigma2_beta_cr_save
  )
  folders <- c("kappa0", "kappa1", rep("tau", 4), rep("alpha", 3), rep("beta", 4))
  names(folders) <- names(blocks)
  log_blocks <- c("kappa0", "kappa1", "sigma2_tau", "sigma2_mu_tau",
                  "sigma2_alpha_p", "sigma2_beta_cr")
  active <- which(metadata$transition == 1 & row(metadata$transition) != col(metadata$transition))
  out_dir <- file.path(run_dir, "diagnostics", "traceplots", "chain1")
  records <- list()
  record_id <- 0L
  # Close the current PDF if plotting stops with an error.
  device_id <- NULL
  on.exit({ if (!is.null(device_id)) grDevices::dev.off(device_id) })
  for (block in names(blocks)) {
    x <- blocks[[block]]
    d <- dim(x)
    parameter_dim <- head(d, -1)
    nparameter <- prod(parameter_dim)
    ndraw <- tail(d, 1)
    ids <- seq_len(nparameter)
    # Structural zeros on diagonal/disallowed kappa edges are not parameters.
    if (block %in% c("kappa0", "kappa1")) ids <- active
    iterations <- seq(metadata$nburn + 1, by = metadata$nthin, length.out = ndraw)
    dir.create(file.path(out_dir, folders[[block]]), recursive = TRUE, showWarnings = FALSE)
    relative_file <- paste(folders[[block]], paste0(block, "_traceplot.pdf"), sep = "/")
    grDevices::pdf(file.path(out_dir, relative_file), width = 10, height = 6,
                   onefile = TRUE, useDingbats = FALSE)
    device_id <- grDevices::dev.cur()
    for (page in seq_along(ids)) {
      id <- ids[page]
      index <- as.integer(arrayInd(id, parameter_dim))
      parameter <- paste0(block, "[", paste(index, collapse = ","), "]")
      if (block == "sigma2_mu_tau") parameter <- block
      # Extract one parameter at a time, avoiding a full draws-by-parameters copy.
      values <- as.numeric(x[id + (seq_len(ndraw) - 1) * nparameter])
      plot_trace_page(values, parameter, iterations, log_scale = block %in% log_blocks)
      record_id <- record_id + 1L
      records[[record_id]] <- data.frame(
        parameter_group = block, parameter = parameter,
        scale = if (block %in% log_blocks) "log" else "raw",
        file = relative_file, page = page, chain_id = result$chain_id,
        source_chain = metadata$chain_files[1], mcmc_seed = result$mcmc_seed,
        n_draws = ndraw, first_iteration = iterations[1], last_iteration = tail(iterations, 1),
        n_nonfinite = sum(!is.finite(values)),
        n_nonpositive = if (block %in% log_blocks) sum(is.finite(values) & values <= 0) else NA_integer_
      )
    }
    grDevices::dev.off(device_id)
    device_id <- NULL
    message("Trace PDF saved: ", block, " (", length(ids), " pages)")
  }
  index <- do.call(rbind, records)
  write.csv(index, file.path(out_dir, "index.csv"), row.names = FALSE)
  message("Chain 1 trace PDFs saved to ", out_dir)
  invisible(index)
}

# Diagnostic tables continue to use all saved chains; the tau subset applies
# only to these tables, never to export_traceplots().
diagnose_mcmc <- function(run_dir, tau_subset_size = 200, tau_subset_seed = 777) {
  metadata <- readRDS(file.path(run_dir, "run_metadata.rds"))
  stopifnot(length(metadata$chain_files) >= 2, tau_subset_size >= 1)
  set.seed(tau_subset_seed)
  tau_ids <- sort(sample(seq_len(metadata$N), min(metadata$N, tau_subset_size)))
  chains <- lapply(metadata$chain_files, function(file) {
    # A local environment prevents one chain from overwriting another.
    e <- new.env()
    load(file.path(run_dir, file), envir = e)
    extract_chain_matrix(e$result, metadata$transition, tau_ids)
  })
  stopifnot(all(vapply(chains, function(x) {
    identical(dim(x), dim(chains[[1]])) && identical(colnames(x), colnames(chains[[1]]))
  }, logical(1))))

  # Keep every monitored parameter in one table. "computed" means that the
  # diagnostics were calculated, not that convergence or ESS is satisfactory.
  status <- vapply(seq_len(ncol(chains[[1]])), function(j) {
    if (any(vapply(chains, function(x) any(!is.finite(x[, j])), logical(1))))
      return("non_finite_draws")
    if (nrow(chains[[1]]) < 4) return("insufficient_draws")
    if (any(vapply(chains, function(x) sd(x[, j]) == 0, logical(1))))
      return("constant_within_chain")
    "computed"
  }, character(1))
  summary <- data.frame(parameter = colnames(chains[[1]]), mean = NA_real_,
                        sd = NA_real_, rhat = NA_real_, ess_bulk = NA_real_,
                        ess_tail = NA_real_, status = status)
  # Finite draws still have descriptive means/SDs even if diagnostics cannot
  # be calculated. Never discard non-finite draws to manufacture a summary.
  for (j in which(status %in% c("constant_within_chain", "insufficient_draws"))) {
    values <- unlist(lapply(chains, function(x) x[, j]), use.names = FALSE)
    summary$mean[j] <- mean(values)
    summary$sd[j] <- sd(values)
  }
  keep <- status == "computed"
  if (any(keep)) {
    draws <- array(NA_real_, c(nrow(chains[[1]]), length(chains), sum(keep)),
                   dimnames = list(NULL, paste0("chain", seq_along(chains)),
                                   colnames(chains[[1]])[keep]))
    for (ch in seq_along(chains)) draws[, ch, ] <- chains[[ch]][, keep, drop = FALSE]
    calculated <- as.data.frame(summarise_draws(as_draws_array(draws),
                                 "mean", "sd", "rhat", "ess_bulk", "ess_tail"))
    statistics <- c("mean", "sd", "rhat", "ess_bulk", "ess_tail")
    summary[keep, statistics] <- calculated[match(summary$parameter[keep], calculated$variable), statistics]
    unavailable <- keep & (!is.finite(summary$rhat) | !is.finite(summary$ess_bulk) |
                             !is.finite(summary$ess_tail))
    summary$status[unavailable] <- "diagnostic_unavailable"
  }
  out_dir <- file.path(run_dir, "diagnostics")
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  write.csv(summary, file.path(out_dir, "parameter_diagnostics.csv"), row.names = FALSE)
  saveRDS(list(summary = summary, tau_ids = tau_ids,
               tau_subset_seed = tau_subset_seed,
               chain_files = metadata$chain_files,
               niter = metadata$niter, nburn = metadata$nburn, nthin = metadata$nthin,
               method = "Rank-normalized split/folded R-hat; bulk and tail ESS",
               session_info = sessionInfo()), file.path(out_dir, "diagnostics.rds"))
  message("Diagnostics saved to ", out_dir)
  invisible(summary)
}
