# Run from the repository root after both single-chain analyses.
library(amen)
library(dplyr)

refit_extra <- FALSE  # TRUE: fit seeds 456/789/1011; FALSE: read saved fits.
seeds <- c(123L, 456L, 789L, 1011L)
output_dir <- "outputs/ame_diagnostics"
comparisons <- data.frame(id = c("cdtally_AT", "cdtally_SK_GB", "lamp_GB", "lamp_SK_GB"),
  fit_dir = c("outputs/cdtally/ame/cdtally_AT", "outputs/cdtally/ame/cdtally_SK_GB",
    "outputs/lamp_return/ame/lamp_GB", "outputs/lamp_return/ame/lamp_SK_GB"))

write_table <- function(x, path) write.csv(x, path, row.names = FALSE, na = "NA")

stability_table <- function(meta, values) {
  colnames(values) <- paste0("mean_seed_", seeds)
  lower <- apply(values, 1, min)
  upper <- apply(values, 1, max)
  cbind(meta, as.data.frame(values),
    min_chain_mean = lower, max_chain_mean = upper,
    sd_chain_mean = apply(values, 1, sd),
    sign_pattern = ifelse(lower > 0, "all_positive",
      ifelse(upper < 0, "all_negative", "mixed_or_zero")))
}

run_diagnostics <- function(spec) {
  out <- file.path(output_dir, spec$id)
  dir.create(out, recursive = TRUE, showWarnings = FALSE)
  Y <- readRDS(file.path(spec$fit_dir, "ame_input.rds"))
  bundle <- readRDS(file.path(spec$fit_dir, "effects.rds"))
  node_tbl <- bundle$results$nodes
  pathway_tbl <- bundle$results$pathways
  obs <- which(!is.na(Y), arr.ind = TRUE)
  in_n <- colSums(!is.na(Y))
  primary <- readRDS(file.path(spec$fit_dir, "fit_seed123.rds"))
  settings <- c(primary$settings, primary$manifest)

  fits <- lapply(seeds, function(seed) {
    path <- file.path(spec$fit_dir, paste0("fit_seed", seed, ".rds"))
    if (seed != 123L && refit_extra) {
      message(spec$id, ": fitting diagnostic seed ", seed)
      fit <- amen::ame(Y, family = "nrm", R = 2, rvar = TRUE, cvar = TRUE,
        dcor = FALSE, symmetric = FALSE, intercept = TRUE,
        burn = settings$burn, nscan = settings$nscan, odens = settings$odens, seed = seed,
        plot = FALSE, print = FALSE, gof = TRUE)
      saveRDS(list(seed = seed, fit = fit,
        settings = list(burn = settings$burn, nscan = settings$nscan, odens = settings$odens,
          node_order = rownames(Y), input_md5 = tools::md5sum(file.path(spec$fit_dir, "ame_input.rds")),
          session = sessionInfo())), path)
      fit
    } else {
      readRDS(path)$fit
    }
  })

  # Scalar diagnostics; rho is fixed at zero.
  draws <- lapply(fits, function(f) {
    x <- cbind(f$BETA, f$VC[, c("va", "cab", "vb", "ve")])
    colnames(x) <- c("BETA_intercept", "VC_va", "VC_cab", "VC_vb", "VC_ve")
    x
  })
  scalar <- bind_rows(lapply(colnames(draws[[1]]), function(p) {
    x <- do.call(cbind, lapply(draws, function(d) d[, p]))
    data.frame(parameter = p, chains = length(seeds), draws_per_chain = nrow(x),
      mean = mean(x), sd = sd(as.vector(x)), rhat = posterior::rhat(x),
      ess_bulk = posterior::ess_bulk(x), ess_tail = posterior::ess_tail(x),
      mcse_mean = posterior::mcse_mean(x))
  }))
  scalar_draws <- bind_rows(lapply(seq_along(seeds), function(k) {
    data.frame(seed = seeds[k], retained_draw = seq_len(nrow(draws[[k]])), draws[[k]])
  }))
  pdf(file.path(out, "scalar_traces.pdf"), width = 9, height = 7)
  par(mfrow = c(2, 1), mar = c(4, 4, 2, 1))
  for (p in colnames(draws[[1]])) {
    x <- do.call(cbind, lapply(draws, function(d) d[, p]))
    matplot(x, type = "l", lty = 1, col = seq_along(seeds),
      xlab = "Retained draw", ylab = p, main = p)
    legend("topright", legend = paste("seed", seeds), col = seq_along(seeds),
      lty = 1, cex = .75, bty = "n")
  }
  dev.off()

  receiver_means <- do.call(cbind, lapply(fits, function(f) as.numeric(f$BPM)[in_n > 0]))
  pathway_means <- do.call(cbind, lapply(fits, function(f) f$UVPM[obs]))
  receivers <- stability_table(node_tbl[in_n > 0, c("node", "is_key_action")], receiver_means)
  pathways <- stability_table(pathway_tbl[, c("source", "target", "key_to_key", "paper_example")], pathway_means)

  tables <- list(scalar_diagnostics = scalar, scalar_draws = scalar_draws,
    receiver_stability = receivers, pathway_stability = pathways,
    key_receiver_stability = receivers[receivers$is_key_action, ],
    selected_pathway_stability = pathways[pathways$key_to_key | pathways$paper_example, ])
  for (name in names(tables)) write_table(tables[[name]], file.path(out, paste0(name, ".csv")))
  message(spec$id, ": diagnostics saved to ", out)
  data.frame(comparison = spec$id, chains = length(seeds), draws_per_chain = nrow(draws[[1]]),
    max_rhat = max(scalar$rhat), min_ess_bulk = min(scalar$ess_bulk), min_ess_tail = min(scalar$ess_tail))
}

diagnostics <- bind_rows(lapply(seq_len(nrow(comparisons)), function(i) run_diagnostics(comparisons[i, ])))
write_table(diagnostics, file.path(output_dir, "AME_diagnostics.csv"))
print(diagnostics)
labels <- c("CD Tally: AT correct minus incorrect", "CD Tally: incorrect SK minus GB",
  "Lamp Return: GB correct minus incorrect", "Lamp Return: correct SK minus GB")
writeLines(c("\\begin{tabular}{lrrr}", "\\toprule",
  "Contrast & Maximum $\\widehat R$ & Minimum bulk ESS & Minimum tail ESS \\\\", "\\midrule",
  sprintf("%s & %.4f & %.1f & %.1f \\\\", labels, diagnostics$max_rhat, diagnostics$min_ess_bulk, diagnostics$min_ess_tail),
  "\\bottomrule", "\\end{tabular}"), file.path(output_dir, "AME_diagnostics.tex"))
