# Run from the repository root.
library(amen)
library(dplyr)

# Settings
input_dir <- "outputs/cdtally/graphs/posterior_mean"
output_dir <- "outputs/cdtally/ame"
refit <- TRUE  # TRUE: fit and save; FALSE: read the saved seed-123 fit.
burn <- 50000L
nscan <- 200000L
odens <- 10L
seed <- 123L

comparisons <- data.frame(
  id = c("cdtally_AT", "cdtally_SK_GB"),
  A = c("prob1_AT", "prob0_SK"), B = c("prob0_AT", "prob0_GB"),
  label_A = c("Correct AT", "Incorrect SK"), label_B = c("Incorrect AT", "Incorrect GB"))
key_action <- c("so_1_3", "so_2_asc", "so_ok", "so", "ss_data", "so_2_desc", "ss_so")
paper_edges <- list(
  cdtally_AT = rbind(c("ss_so", "so_1_3"), c("so_ok", "wb"), c("so_ok", "ss_so")),
  cdtally_SK_GB = rbind(c("ss_so", "so_1_3"), c("so_ok", "ss_so")))

read_network <- function(path) {
  M <- as.matrix(read.csv(path, row.names = 1, check.names = FALSE))
  M[rownames(M), rownames(M), drop = FALSE]
}

edge_key <- function(source, target) paste(source, target, sep = "\r")

write_table <- function(x, path) {
  write.csv(x, path, row.names = FALSE, na = "NA")
}

run_comparison <- function(spec) {
  out <- file.path(output_dir, spec$id)
  dir.create(out, recursive = TRUE, showWarnings = FALSE)

  # Differential network
  input_files <- file.path(input_dir, c(spec$A, spec$B), "posterior_mean.csv")
  raw <- lapply(input_files, read_network)
  nodes <- unique(c(rownames(raw[[1]]), rownames(raw[[2]])))
  pad <- function(M) {
    ans <- matrix(0, length(nodes), length(nodes), dimnames = list(nodes, nodes))
    ans[rownames(M), colnames(M)] <- M
    ans
  }
  A <- pad(raw[[1]])
  B <- pad(raw[[2]])
  mask_A <- A > 0
  mask_B <- B > 0
  source_A <- rowSums(mask_A) > 0
  source_B <- rowSums(mask_B) > 0
  common <- source_A & source_B
  Y <- A - B
  Y[!common, ] <- NA_real_
  Y[!(mask_A | mask_B)] <- NA_real_
  diag(Y) <- NA_real_
  active <- rowSums(!is.na(Y)) > 0 | colSums(!is.na(Y)) > 0
  Y <- Y[active, active, drop = FALSE]
  obs <- which(!is.na(Y), arr.ind = TRUE)
  fit_nodes <- rownames(Y)

  # Single-chain fit
  path <- file.path(out, paste0("fit_seed", seed, ".rds"))
  if (refit) {
    message(spec$id, ": fitting seed ", seed)
    fit <- amen::ame(Y, family = "nrm", R = 2, rvar = TRUE, cvar = TRUE,
      dcor = FALSE, symmetric = FALSE, intercept = TRUE,
      burn = burn, nscan = nscan, odens = odens, seed = seed,
      plot = FALSE, print = FALSE, gof = TRUE)
    saveRDS(list(seed = seed, fit = fit,
      settings = list(burn = burn, nscan = nscan, odens = odens,
        input_md5 = tools::md5sum(input_files), node_order = fit_nodes,
        session = sessionInfo())), path)
  } else {
    fit <- readRDS(path)$fit
  }
  saveRDS(Y, file.path(out, "ame_input.rds"))

  # Seed-123 effects
  a <- setNames(as.numeric(fit$APM), fit_nodes)
  b <- setNames(as.numeric(fit$BPM), fit_nodes)
  out_n <- rowSums(!is.na(Y))
  in_n <- colSums(!is.na(Y))
  node_tbl <- data.frame(node = fit_nodes, is_key_action = fit_nodes %in% key_action,
    observed_out_dyads = out_n, observed_in_dyads = in_n,
    sender_effect = ifelse(out_n > 0, a, NA_real_),
    receiver_effect = ifelse(in_n > 0, b, NA_real_),
    D_out = ifelse(out_n > 0, .5 * rowSums(abs(Y), na.rm = TRUE), NA_real_),
    row.names = NULL)
  node_tbl$receiver_exposure <- in_n
  node_tbl$abs_receiver_effect <- abs(node_tbl$receiver_effect)
  source <- fit_nodes[obs[, 1]]
  target <- fit_nodes[obs[, 2]]
  aa <- A[fit_nodes, fit_nodes, drop = FALSE][obs]
  bb <- B[fit_nodes, fit_nodes, drop = FALSE][obs]
  wanted <- paper_edges[[spec$id]]
  pathway_tbl <- data.frame(source, target, A = aa, B = bb, diff = Y[obs],
    edge_in_A = aa > 0, edge_in_B = bb > 0,
    edge_type = ifelse(aa > 0 & bb > 0, "both_observed",
      ifelse(aa > 0, "A_only_edge", "B_only_edge")),
    source_is_key_action = source %in% key_action,
    target_is_key_action = target %in% key_action,
    key_to_key = source %in% key_action & target %in% key_action,
    paper_example = edge_key(source, target) %in% edge_key(wanted[, 1], wanted[, 2]),
    intercept = mean(fit$BETA[, 1]), sender_effect = unname(a[source]),
    receiver_effect = unname(b[target]), mult_inner = fit$UVPM[obs])
  pathway_tbl$fitted <- with(pathway_tbl, intercept + sender_effect + receiver_effect + mult_inner)
  pathway_tbl$residual <- pathway_tbl$diff - pathway_tbl$fitted

  # Exclusive-source outgoing profiles
  exclusive <- data.frame(source = character(), target = character(), probability = double(),
    side = character(), source_is_key_action = logical(), target_is_key_action = logical())
  for (side in c("A", "B")) {
    M <- list(A = A, B = B)[[side]]
    sources <- list(A = nodes[source_A & !source_B], B = nodes[source_B & !source_A])[[side]]
    for (src in sources) {
      probs <- M[src, ]
      probs <- sort(probs[probs > 0], decreasing = TRUE)
      exclusive <- rbind(exclusive, data.frame(source = src, target = names(probs),
        probability = unname(probs), side = side, source_is_key_action = src %in% key_action,
        target_is_key_action = names(probs) %in% key_action))
    }
  }
  exclusive_summary <- exclusive %>%
    group_by(source, side, source_is_key_action) %>%
    summarise(out_support_size = n(), max_out_prob = max(probability),
      entropy = -sum(probability * log(probability)),
      effective_targets = exp(entropy), .groups = "drop")

  # Results
  tables <- list(ame_node_effects = node_tbl, ame_pathway = pathway_tbl,
    ame_source_Dout = node_tbl[out_n > 0, ],
    main_text_pathways = pathway_tbl[pathway_tbl$paper_example, ],
    exclusive_source_profiles = exclusive, exclusive_source_summary = exclusive_summary)
  for (name in names(tables)) write_table(tables[[name]], file.path(out, paste0(name, ".csv")))
  saveRDS(list(comparison = spec, keys = key_action, reporting_seed = seed,
    results = list(nodes = node_tbl, pathways = pathway_tbl,
      exclusive_profiles = exclusive, exclusive_summary = exclusive_summary)), file.path(out, "effects.rds"))
  message(spec$id, ": results saved to ", out)
  invisible(NULL)
}

for (i in seq_len(nrow(comparisons))) run_comparison(comparisons[i, ])
