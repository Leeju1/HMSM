# Run from the repository root: Rscript mcmc/plot_parameters.R cdtally
library(ggplot2)
library(HDInterval)

# All intervals are 95% HPD intervals of the retained draws in one chain.
posterior_summary <- function(x) {
  stopifnot(length(x) >= 2L, all(is.finite(x)))
  h <- hdi(x, credMass = 0.95)
  data.frame(PosteriorMean = mean(x), PosteriorMedian = median(x),
             HPD_Lower = unname(h[1]), HPD_Upper = unname(h[2]),
             HPD_Excludes_Zero = h[1] > 0 || h[2] < 0)
}

latex_escape <- function(x) {
  replacements <- c("\\" = "\\textbackslash{}", "&" = "\\&", "%" = "\\%",
                    "$" = "\\$", "#" = "\\#", "_" = "\\_", "{" = "\\{",
                    "}" = "\\}", "~" = "\\textasciitilde{}", "^" = "\\textasciicircum{}")
  vapply(strsplit(as.character(x), "", fixed = TRUE), function(chars) {
    matched <- chars %in% names(replacements)
    chars[matched] <- replacements[chars[matched]]
    paste0(chars, collapse = "")
  }, character(1))
}

# Standalone table fragments: use booktabs, longtable, graphicx, and xcolor.
write_parameter_tables <- function(beta, kappa, top5, countries, item, tag, out_dir) {
  interval <- function(mean, low, high, digits = 3L) {
    sprintf(paste0("%.", digits, "f [%.", digits, "f, %.", digits, "f]"), mean, low, high)
  }
  beta_rows <- vapply(countries, function(country) {
    b <- beta[beta$Country == country, ]
    cells <- interval(b$PosteriorMean, b$HPD_Lower, b$HPD_Upper)
    colors <- ifelse(b$HPD_Excludes_Zero,
                     ifelse(b$PosteriorMean > 0, "blue", "red"), "black")
    cells <- paste0("\\textcolor{", colors, "}{", cells, "}")
    paste0(paste(c(latex_escape(country), cells), collapse = " & "), " \\\\")
  }, character(1))
  writeLines(c(
    "% Requires booktabs, graphicx, xcolor.", "\\begin{table}[htbp]", "\\centering",
    paste0("\\caption{Key-action effects for ", item, ": posterior means and 95\\% HPD intervals.}"),
    paste0("\\label{tab:key-action-effects-", tag, "}"),
    "\\resizebox{\\textwidth}{!}{%", "\\begin{tabular}{lrrrrrrrrrrrr}", "\\toprule",
    " & \\multicolumn{4}{c}{Correct} & \\multicolumn{4}{c}{Incorrect} & \\multicolumn{4}{c}{Correct $-$ Incorrect} \\\\",
    paste0("Country & ", paste(rep(c("Source", "Destination", "Interaction", "Total"), 3), collapse = " & "), " \\\\") ,
    "\\midrule", beta_rows, "\\bottomrule", "\\end{tabular}}",
    "\\par\\smallskip {\\footnotesize Total = source + destination + interaction. Blue/red intervals exclude zero.}",
    "\\end{table}"), file.path(out_dir, "table_key_action_effects.tex"))

  for (name in c("top5", "all_transitions")) {
    tab <- if (name == "top5") top5 else kappa
    rows <- vapply(seq_len(nrow(tab)), function(i) {
      paste0(paste(c(tab$Group[i], latex_escape(tab$From[i]), latex_escape(tab$To[i]),
        interval(tab$KappaMean[i], tab$KappaHPD_L[i], tab$KappaHPD_U[i], 1),
        interval(tab$LogKappaMean[i], tab$LogKappaHPD_L[i], tab$LogKappaHPD_U[i], 2)),
        collapse = " & "), " \\\\")
    }, character(1))
    caption <- if (name == "top5") "Top five baseline transition hazards by response group" else
      "All group-observed baseline transition hazards"
    header <- c("\\toprule", "Group & From & To & $\\kappa$: mean [95\\% HPD] & $\\log\\kappa$: mean [95\\% HPD] \\\\", "\\midrule")
    writeLines(c("% Requires booktabs and longtable.", "\\begingroup\\footnotesize",
      "\\setlength{\\tabcolsep}{3pt}", "\\begin{longtable}{lp{2.5cm}p{2.5cm}p{3.2cm}p{3.2cm}}",
      paste0("\\caption{", caption, " for ", item, ".}\\label{tab:kappa-", name, "-", tag, "}\\\\"),
      header, "\\endfirsthead", header, "\\endhead", "\\bottomrule\\endfoot",
      rows, "\\end{longtable}",
      "Only off-diagonal transitions observed in the corresponding response group are included. Ranked by posterior mean of $\\kappa$ within each group.",
      "\\endgroup"), file.path(out_dir, paste0("table_kappa_", name, ".tex")))
  }
}

plot_parameters <- function(dataset = c("cdtally", "lamp_return"),
                            chain_file = NULL, out_dir = NULL) {
  dataset <- match.arg(dataset)
  is_cd <- dataset == "cdtally"
  tag <- if (is_cd) "cd" else "lamp"
  item <- if (is_cd) "CD Tally" else "Lamp Return"
  data_dir <- file.path("data", dataset)
  if (is.null(chain_file)) chain_file <- file.path("outputs", dataset, "mcmc", "chain01_seed1001.RData")
  if (is.null(out_dir)) out_dir <- file.path("outputs", dataset, "mcmc", "summary")

  # Separate environments prevent chain workspace variables from replacing labels.
  inputs <- new.env()
  load(file.path(data_dir, if (is_cd) "combined_cd_tally.RData" else "combined_lamp_return.RData"), inputs)
  countries <- as.character(inputs$country)
  load(file.path(data_dir, if (is_cd) "cdtally_transition_correct.RData" else "lamp_transition_correct.RData"), inputs)
  load(file.path(data_dir, if (is_cd) "cdtally_transition_incorrect.RData" else "lamp_transition_incorrect.RData"), inputs)
  states <- read.csv(file.path(data_dir, if (is_cd) "state_name_list_cdtally.csv" else "state_name_list_lamp.csv"))$state_name
  chain <- new.env()
  load(chain_file, chain)
  result <- chain$result
  covariates <- c("Gender", "Age", "Education", "IncPctRank", "Eskill")
  Q <- length(countries)
  P <- length(covariates)
  E <- length(states)
  D <- ncol(result$mu_tau_save)
  stopifnot(D >= 2L, length(unique(countries)) == Q,
    identical(dim(result$mu_tau_save), c(Q, D)),
    identical(dim(result$sigma2_tau_save), c(Q, D)),
    identical(dim(result$alpha_save), c(P, Q, D)),
    identical(dim(result$mu_alpha_save), c(P, D)),
    identical(dim(result$sigma2_alpha_p_save), c(P, D)),
    identical(dim(result$beta_key0_save), c(3L, Q, D)),
    identical(dim(result$beta_key1_save), c(3L, Q, D)),
    identical(dim(result$kappa0_save), c(E, E, D)),
    identical(dim(result$kappa1_save), c(E, E, D)),
    identical(dim(inputs$transition_correct), c(E, E)),
    identical(dim(inputs$transition_incorrect), c(E, E)))
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  save_csv <- function(tab, name) write.csv(tab, file.path(out_dir, name), row.names = FALSE)
  save_plot <- function(plot, name, width, height) {
    ggsave(file.path(out_dir, name), plot, width = width, height = height, dpi = 300, bg = "white")
  }
  sign_flag <- function(tab) factor(ifelse(tab$HPD_Excludes_Zero,
    ifelse(tab$PosteriorMean > 0, "Positive", "Negative"), "Includes zero"),
    levels = c("Includes zero", "Positive", "Negative"))
  effect_colors <- c("Includes zero" = "black", "Positive" = "#2e86c1", "Negative" = "#c0392b")
  x_label <- "Posterior mean with 95% HPD interval"

  # Speed summaries: transform every draw before computing means and intervals.
  speed <- do.call(rbind, lapply(seq_len(Q), function(q) rbind(
    cbind(Country = countries[q], Parameter = "Median speed", posterior_summary(exp(result$mu_tau_save[q, ]))),
    cbind(Country = countries[q], Parameter = "Log-SD", posterior_summary(sqrt(result$sigma2_tau_save[q, ]))))))
  speed$Country <- factor(speed$Country, levels = countries)
  speed$Parameter <- factor(speed$Parameter, levels = c("Median speed", "Log-SD"))
  references <- aggregate(PosteriorMean ~ Parameter, speed, mean)
  p_speed <- ggplot(speed, aes(PosteriorMean, Country)) +
    geom_errorbar(aes(xmin = HPD_Lower, xmax = HPD_Upper), width = 0, orientation = "y", linewidth = 0.7) +
    geom_point(size = 2) +
    geom_vline(data = references, aes(xintercept = PosteriorMean), linetype = 3, linewidth = 0.5) +
    facet_wrap(~Parameter, scales = "free_x", ncol = 2) +
    labs(x = x_label, y = NULL) + theme_bw(12) + theme(strip.text = element_text(face = "bold"))
  save_csv(speed, "country_speed_hyperparameters_summary.csv")
  save_plot(p_speed, paste0("fig_speed_", tag, ".png"), 10, max(5, 0.35 * Q + 1.5))

  # Country covariate effects and their hierarchical pooled effects/SDs.
  alpha <- do.call(rbind, lapply(seq_len(P), function(p) do.call(rbind, lapply(seq_len(Q), function(q) {
    cbind(Covariate = covariates[p], Country = countries[q], posterior_summary(result$alpha_save[p, q, ]))
  }))))
  hyper <- do.call(rbind, lapply(seq_len(P), function(p) rbind(
    cbind(Covariate = covariates[p], Parameter = "mu_alpha", Interpretation = "Hierarchical pooled effect",
          posterior_summary(result$mu_alpha_save[p, ])),
    cbind(Covariate = covariates[p], Parameter = "sigma_alpha", Interpretation = "Between-country SD",
          posterior_summary(sqrt(result$sigma2_alpha_p_save[p, ]))))))
  hyper$HPD_Excludes_Zero[hyper$Parameter == "sigma_alpha"] <- NA
  save_csv(alpha, "table_alpha_country_summary.csv")
  save_csv(hyper, "table_alpha_hyper_summary.csv")
  alpha$Covariate <- factor(alpha$Covariate, levels = covariates)
  alpha$Country <- factor(alpha$Country, levels = countries)
  alpha$Sign <- sign_flag(alpha)
  hyper$Covariate <- factor(hyper$Covariate, levels = covariates)
  pooled <- hyper[hyper$Parameter == "mu_alpha", ]
  p_alpha <- ggplot(alpha, aes(PosteriorMean, Country, color = Sign)) +
    geom_vline(xintercept = 0, linetype = 3, color = "gray45", linewidth = 0.45) +
    geom_vline(data = pooled, aes(xintercept = PosteriorMean), linetype = 2, color = "gray25", linewidth = 0.6) +
    geom_errorbar(aes(xmin = HPD_Lower, xmax = HPD_Upper), width = 0, orientation = "y", linewidth = 0.7) +
    geom_point(size = 2) + facet_wrap(~Covariate, scales = "free_x", ncol = 3) +
    scale_color_manual(values = effect_colors, guide = "none") + labs(x = x_label, y = NULL) +
    theme_bw(11) + theme(strip.text = element_text(face = "bold"))
  save_plot(p_alpha, paste0("fig_alpha_", tag, ".png"), 11, max(5.5, 0.30 * Q + 1.5))
  hyper$Panel <- factor(ifelse(hyper$Parameter == "mu_alpha", "Pooled effect", "Between-country SD"),
                        levels = c("Pooled effect", "Between-country SD"))
  hyper$Sign <- sign_flag(transform(hyper, HPD_Excludes_Zero = !is.na(HPD_Excludes_Zero) & HPD_Excludes_Zero))
  p_hyper <- ggplot(hyper, aes(PosteriorMean, Covariate, color = Sign)) +
    geom_vline(data = hyper[hyper$Parameter == "mu_alpha", ], aes(xintercept = 0),
               linetype = 3, color = "gray45", linewidth = 0.45) +
    geom_errorbar(aes(xmin = HPD_Lower, xmax = HPD_Upper), width = 0, orientation = "y", linewidth = 0.8) +
    geom_point(size = 2.2) + facet_wrap(~Panel, scales = "free_x", ncol = 2) +
    scale_color_manual(values = effect_colors, guide = "none") + labs(x = x_label, y = NULL) +
    theme_bw(11) + theme(strip.text = element_text(face = "bold"))
  save_plot(p_hyper, paste0("fig_alpha_hyper_", tag, ".png"), 10, 5)

  # Beta totals and response-group differences are computed draw by draw.
  groups <- c("Correct", "Incorrect", "Correct - Incorrect")
  effects <- c("Source is key", "Destination is key", "Source \u00d7 Destination", "Key \u2192 Key total")
  beta <- do.call(rbind, lapply(seq_len(Q), function(q) {
    b1 <- result$beta_key1_save[, q, ]
    b0 <- result$beta_key0_save[, q, ]
    draws <- list(b1, b0, b1 - b0)
    do.call(rbind, lapply(seq_along(groups), function(g) {
      b <- rbind(draws[[g]], colSums(draws[[g]]))
      do.call(rbind, lapply(seq_along(effects), function(e) {
        cbind(Country = countries[q], Group = groups[g], Effect = effects[e], posterior_summary(b[e, ]))
      }))
    }))
  }))
  save_csv(beta, "table_key_action_effects_summary.csv")
  beta$Country <- factor(beta$Country, levels = countries)
  beta$Cell <- factor(paste(beta$Group, beta$Effect, sep = " \u00b7 "),
                     levels = as.vector(t(outer(groups, effects, paste, sep = " \u00b7 "))))
  p_beta <- ggplot(beta, aes(Cell, Country, fill = PosteriorMean)) +
    geom_tile(color = "white", linewidth = 0.3) +
    geom_point(data = beta[beta$HPD_Excludes_Zero, ], color = "black", shape = 16, size = 1.6) +
    scale_fill_gradient2(low = "#c0392b", mid = "white", high = "#2e86c1", midpoint = 0, name = "Posterior mean") +
    labs(x = NULL, y = NULL, title = "Key action effects by country",
         subtitle = "Tiles: posterior mean; dots: 95% HPD excludes 0; total = source + destination + interaction") +
    theme_bw(10) + theme(axis.text.x = element_text(angle = 35, hjust = 1, vjust = 1), panel.grid = element_blank())
  save_plot(p_beta, "fig_key_action_effects_heatmap.png", 15, max(4.5, 0.32 * Q + 1.5))

  # Match the original tables: group-specific observed edges, excluding self-loops.
  # These masks differ from the pooled transition mask used by the sampler.
  kappa_group <- function(draws, mask, group) {
    diag(mask) <- 0
    edges <- which(mask > 0, arr.ind = TRUE)
    tab <- do.call(rbind, lapply(seq_len(nrow(edges)), function(i) {
      m <- edges[i, 1]
      l <- edges[i, 2]
      x <- draws[m, l, ]
      stopifnot(all(x > 0))
      raw <- posterior_summary(x)
      logged <- posterior_summary(log(x))
      data.frame(Group = group, m = m, l = l, From = states[m], To = states[l],
        KappaMean = raw$PosteriorMean, KappaHPD_L = raw$HPD_Lower, KappaHPD_U = raw$HPD_Upper,
        LogKappaMean = logged$PosteriorMean, LogKappaHPD_L = logged$HPD_Lower, LogKappaHPD_U = logged$HPD_Upper)
    }))
    tab <- tab[order(-tab$KappaMean, tab$m, tab$l), ]
    tab$Rank <- seq_len(nrow(tab))
    tab
  }
  kappa <- rbind(kappa_group(result$kappa1_save, inputs$transition_correct, "Correct"),
                 kappa_group(result$kappa0_save, inputs$transition_incorrect, "Incorrect"))
  top5 <- kappa[kappa$Rank <= 5, ]
  save_csv(kappa, "table_kappa_all_transitions.csv")
  save_csv(top5, "table_kappa_top5.csv")
  write_parameter_tables(beta, kappa, top5, countries, item, tag, out_dir)
  message(item, ": saved 4 PNG figures, 6 CSV summaries, and 3 LaTeX tables to ", out_dir)
  invisible(list(speed = speed, alpha = alpha, hyper = hyper, beta = beta, kappa = kappa))
}

if (sys.nframe() == 0L) {
  args <- commandArgs(trailingOnly = TRUE)
  plot_parameters(if (length(args)) args[1] else "cdtally")
}
