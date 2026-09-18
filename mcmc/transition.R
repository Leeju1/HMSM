transition_support <- function(data, n_states, countries, threshold) {
  adjacent <- which(head(data$pid, -1) == tail(data$pid, -1))
  edges <- data[adjacent, c("state_new", "country", "res")]
  edges$to <- data$state_new[adjacent + 1L]
  support <- list()
  for (q in seq_along(countries)) {
    for (res in 0:1) {
      e <- edges[edges$country == q & edges$res == res, ]
      counts <- matrix(tabulate(e$state_new + (e$to - 1L) * n_states,
                                nbins = n_states^2), n_states, n_states)
      mask <- counts > 0
      # The threshold includes observed self-transitions; the graph excludes them.
      mask[rowSums(counts) < threshold[q, res + 1L], ] <- FALSE
      diag(mask) <- FALSE
      support[[paste0("prob", res, "_", countries[q])]] <- mask
    }
  }
  support
}

transition_probability <- function(kappa, beta, tau, alpha_x, key_design, support) {
  n <- nrow(kappa)
  rates <- matrix(0, n, n)
  rates[support] <- kappa[support] * tau *
    exp(alpha_x + as.vector(key_design[support, , drop = FALSE] %*% beta))
  totals <- rowSums(rates)
  if (any(!is.finite(totals)) || any(rates < 0)) {
    stop("Non-finite or negative transition rates.")
  }
  rates / ifelse(totals > 0, totals, 1)
}

new_graph_directory <- function(path) {
  if (file.exists(path) && (!dir.exists(path) ||
      length(list.files(path, all.files = TRUE, no.. = TRUE)) > 0L)) {
    stop("Output directory must be new or empty: ", path)
  }
  if (!dir.exists(path) && !dir.create(path, recursive = TRUE)) {
    stop("Cannot create output directory: ", path)
  }
}

save_mean_graph <- function(mean, path) {
  dir.create(path, recursive = TRUE, showWarnings = FALSE)
  write.csv(mean, file.path(path, "posterior_mean.csv"), fileEncoding = "UTF-8")
  saveRDS(mean, file.path(path, "posterior_mean.rds"))
}

build_transition_graphs <- function(data_file, chain_file, state_file, out_dir,
                                    threshold = 3, x = c(1, 5, 2, 4, 1),
                                    draw_indices = NULL) {
  input <- new.env()
  chain <- new.env()
  load(data_file, envir = input)
  load(chain_file, envir = chain)
  data <- input$combined_data
  countries <- input$country
  result <- chain$result
  states <- read.csv(state_file, stringsAsFactors = FALSE, check.names = FALSE)
  n <- nrow(states)
  q <- length(countries)
  n_draws <- dim(result$kappa0_save)[3]
  key <- as.vector(result$state_key_indicator)
  same_id <- head(data$pid, -1) == tail(data$pid, -1)

  stopifnot(
    is.character(countries), q > 0L, !anyNA(countries), !anyDuplicated(countries),
    identical(as.numeric(states$num), as.numeric(seq_len(n))),
    !anyNA(states$state_name), !anyDuplicated(states$state_name),
    all(nzchar(states$state_name)), nrow(data) > 1L,
    all(is.finite(as.matrix(data[, c("pid", "state_new", "timestamp", "res", "country")]))),
    all(data$state_new %in% seq_len(n)), max(data$state_new) == n,
    all(data$country %in% seq_len(q)), all(data$res %in% 0:1),
    !anyDuplicated(rle(data$pid)$values),
    all(diff(data$timestamp)[same_id] >= 0),
    all(diff(data$country)[same_id] == 0), all(diff(data$res)[same_id] == 0),
    length(key) == n, all(key %in% 0:1),
    length(n_draws) == 1L, n_draws > 0L, all(is.finite(x))
  )
  shapes <- list(kappa0_save = c(n, n, n_draws), kappa1_save = c(n, n, n_draws),
    beta_key0_save = c(3, q, n_draws), beta_key1_save = c(3, q, n_draws),
    alpha_save = c(length(x), q, n_draws), mu_tau_save = c(q, n_draws),
    sigma2_tau_save = c(q, n_draws))
  for (name in names(shapes)) {
    if (!identical(dim(result[[name]]), as.integer(shapes[[name]]))) {
      stop("Unexpected MCMC dimensions: ", name)
    }
  }
  if (length(threshold) == 1L) threshold <- matrix(threshold, q, 2)
  stopifnot(identical(dim(threshold), c(q, 2L)),
            all(is.finite(threshold)), all(threshold >= 0))
  dimnames(threshold) <- list(countries, c("prob0", "prob1"))
  if (is.null(draw_indices)) draw_indices <- seq_len(n_draws)
  stopifnot(length(draw_indices) > 0L, !anyDuplicated(draw_indices),
            all(draw_indices %in% seq_len(n_draws)), !is.unsorted(draw_indices))

  support <- transition_support(data, n, countries, threshold)
  source_key <- rep(key, times = n)
  target_key <- rep(key, each = n)
  key_design <- cbind(source_key, target_key, source_key * target_key)
  new_graph_directory(out_dir)
  for (name in names(support)) {
    dir.create(file.path(out_dir, "draws", name), recursive = TRUE)
  }
  means <- lapply(support, function(mask) matrix(0, n, n,
    dimnames = list(states$state_name, states$state_name)))

  for (k in seq_along(draw_indices)) {
    draw <- draw_indices[k]
    tau <- exp(result$mu_tau_save[, draw] + 0.5 * result$sigma2_tau_save[, draw])
    for (country_id in seq_along(countries)) {
      alpha_x <- as.numeric(result$alpha_save[, country_id, draw] %*% x)
      for (res in 0:1) {
        name <- paste0("prob", res, "_", countries[country_id])
        probability <- transition_probability(
          result[[paste0("kappa", res, "_save")]][, , draw],
          result[[paste0("beta_key", res, "_save")]][, country_id, draw],
          tau[country_id], alpha_x, key_design, support[[name]])
        dimnames(probability) <- dimnames(means[[name]])
        means[[name]] <- means[[name]] + (probability - means[[name]]) / k
        write.csv(probability, file.path(out_dir, "draws", name,
          sprintf("sample_%04d.csv", draw - 1L)), fileEncoding = "UTF-8")
      }
    }
    if (k %% 100L == 0L) message("Graphs: ", k, "/", length(draw_indices), " draws")
  }
  for (name in names(means)) {
    save_mean_graph(means[[name]], file.path(out_dir, "posterior_mean", name))
  }
  files <- c(data = data_file, chain = chain_file, states = state_file)
  metadata <- list(input_files = normalizePath(files, winslash = "/"),
    input_md5 = tools::md5sum(files), countries = countries,
    state_names = states$state_name, state_key_indicator = key,
    threshold = threshold, x = x, draw_indices = draw_indices,
    sample_files = sprintf("sample_%04d.csv", draw_indices - 1L),
    session_info = sessionInfo(), completed = TRUE)
  saveRDS(metadata, file.path(out_dir, "run_metadata.rds"))
  message("Saved ", length(draw_indices) * length(support), " graphs and ",
          length(means), " posterior means to ", out_dir)
  invisible(metadata)
}

posterior_mean_graphs <- function(draws_dir, out_dir) {
  graph_names <- list.dirs(draws_dir, full.names = FALSE, recursive = FALSE)
  stopifnot(length(graph_names) > 0L, all(grepl("^prob[01]_[A-Z]{2}$", graph_names)))
  files <- lapply(file.path(draws_dir, graph_names), list.files,
                  pattern = "^sample_[0-9]+\\.csv$")
  stopifnot(length(files[[1]]) > 0L,
            all(vapply(files, identical, logical(1), files[[1]])))
  new_graph_directory(out_dir)
  labels <- NULL
  for (name in graph_names) {
    mean <- NULL
    for (k in seq_along(files[[1]])) {
      path <- file.path(draws_dir, name, files[[1]][k])
      m <- as.matrix(read.csv(path, row.names = 1, check.names = FALSE))
      if (is.null(labels)) labels <- rownames(m)
      stopifnot(is.numeric(m), identical(rownames(m), labels),
                identical(colnames(m), labels), !anyDuplicated(labels),
                all(is.finite(m)), all(m >= 0), all(diag(m) == 0),
                all(rowSums(m) == 0 | abs(rowSums(m) - 1) < 1e-6))
      if (is.null(mean)) mean <- m * 0
      mean <- mean + (m - mean) / k
    }
    save_mean_graph(mean, file.path(out_dir, name))
  }
  saveRDS(list(draws_dir = normalizePath(draws_dir, winslash = "/"),
    graph_names = graph_names, sample_files = files[[1]], session_info = sessionInfo()),
    file.path(out_dir, "run_metadata.rds"))
  invisible(out_dir)
}
