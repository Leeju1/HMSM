# Shared configuration, validation and IO for the distance analysis scripts.
# Run entry scripts from the repository root; see README.md.

read_analysis_config <- function(args) {
  require_packages("jsonlite")
  if (length(args) %% 2L != 0L) stop("Use --dataset cdtally|lamp_return [--config file.json] [--distance-file file.csv] [--out-dir path].")
  opts <- list()
  if (length(args)) {
    for (i in seq.int(1L, length(args), by = 2L)) {
      key <- args[[i]]
      if (!key %in% c("--dataset", "--config", "--distance-file", "--out-dir")) stop("Unknown option: ", key)
      if (!is.null(opts[[key]])) stop("Repeated option: ", key)
      opts[[key]] <- args[[i + 1L]]
    }
  }
  dataset <- opts[["--dataset"]]
  if (is.null(dataset) || !dataset %in% c("cdtally", "lamp_return")) stop("Supply --dataset cdtally or --dataset lamp_return.")
  files <- c("config/distance/common.json", paste0("config/distance/", dataset, ".json"))
  if (!is.null(opts[["--config"]])) files <- c(files, opts[["--config"]])
  cfg <- list()
  for (f in files) cfg <- utils::modifyList(cfg, jsonlite::fromJSON(f))
  if (!identical(cfg$dataset, dataset)) stop("Config dataset differs from --dataset.")
  if (!is.null(opts[["--distance-file"]])) cfg$distance_file <- opts[["--distance-file"]]
  if (!is.null(opts[["--out-dir"]])) cfg$analysis_dir <- opts[["--out-dir"]]
  for (key in c("seed", "nmds_try", "nmds_trymax", "nmds_maxit", "n_perm_unrestricted")) {
    value <- cfg[[key]]
    if (length(value) != 1L || !is.finite(value) || value < 1 || value != as.integer(value)) stop("Invalid integer setting: ", key)
  }
  if (!is.logical(cfg$country_blocks) || length(cfg$country_blocks) != 1L || is.na(cfg$country_blocks)) stop("country_blocks must be true or false.")
  if (!is.logical(cfg$diagnostic_plots) || length(cfg$diagnostic_plots) != 1L || is.na(cfg$diagnostic_plots)) stop("diagnostic_plots must be true or false.")
  if (!is.finite(cfg$atol) || cfg$atol <= 0) stop("atol must be positive.")
  cfg$config_files <- files
  cfg
}

validate_analysis_labels <- function(obj, cfg) {
  expected <- sort(as.vector(outer(c("prob0", "prob1"), cfg$countries, paste, sep = "_")))
  if (!identical(sort(rownames(obj$matrix)), expected) || nrow(obj$matrix) != cfg$expected_graphs) stop("Distance graph labels differ from the configured country/response combinations.")
  invisible(TRUE)
}

save_analysis_metadata <- function(cfg, analysis, details = list()) {
  files <- c(cfg$config_files, cfg$distance_file, "distance/utils.R", paste0("distance/", analysis, ".R"))
  saveRDS(list(settings = cfg, input_files_md5 = tools::md5sum(files), session_info = sessionInfo(), details = details),
          file.path(cfg$analysis_dir, paste0(analysis, "_metadata.rds")))
}

require_packages <- function(pkgs) {
  missing <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)]
  if (length(missing) > 0L) {
    stop(
      "Missing R package(s): ", paste(missing, collapse = ", "),
      "\nInstall once with: install.packages(c(",
      paste(sprintf('"%s"', missing), collapse = ", "), "))"
    )
  }
}

ensure_outdir <- function(path) {
  if (!dir.exists(path)) {
    dir.create(path, recursive = TRUE, showWarnings = FALSE)
  }
  normalizePath(path, mustWork = FALSE)
}

validate_distance_matrix <- function(mat, atol = 1e-8) {
  if (!is.matrix(mat)) mat <- as.matrix(mat)
  storage.mode(mat) <- "double"

  if (nrow(mat) != ncol(mat)) {
    stop("Distance matrix must be square. Got ", nrow(mat), " x ", ncol(mat), ".")
  }
  if (is.null(rownames(mat)) || is.null(colnames(mat))) {
    stop("Distance matrix must have row and column labels.")
  }
  if (anyDuplicated(rownames(mat))) stop("Distance matrix row labels are not unique.")
  if (anyDuplicated(colnames(mat))) stop("Distance matrix column labels are not unique.")

  if (!identical(rownames(mat), colnames(mat))) {
    if (setequal(rownames(mat), colnames(mat))) {
      message("[INFO] Reordering columns to match row-label order.")
      mat <- mat[, rownames(mat), drop = FALSE]
    } else {
      stop("Distance matrix row/column labels do not match.")
    }
  }

  if (any(!is.finite(mat))) stop("Distance matrix contains NA/NaN/Inf values.")

  if (min(mat) < -atol) {
    stop("Distance matrix contains a negative value below tolerance. min = ", min(mat))
  }
  mat[mat < 0] <- 0

  asym <- max(abs(mat - t(mat)))
  if (asym > atol) {
    stop(
      "Distance matrix is not symmetric within atol = ", atol,
      ". max|D-D'| = ", signif(asym, 6)
    )
  }

  max_diag <- max(abs(diag(mat)))
  if (max_diag > atol) {
    stop(
      "Distance diagonal is not zero within tolerance. max|diag| = ",
      signif(max_diag, 6)
    )
  }
  diag(mat) <- 0

  mat
}

parse_prob_labels <- function(labels) {
  m <- regexec("^(prob[01])_(.+)$", labels)
  parts <- regmatches(labels, m)
  ok <- lengths(parts) == 3L

  if (!all(ok)) {
    stop(
      "Labels must follow prob0_<country> or prob1_<country>. Invalid: ",
      paste(labels[!ok], collapse = ", ")
    )
  }

  prob <- vapply(parts, `[[`, character(1), 2L)
  country <- vapply(parts, `[[`, character(1), 3L)
  group <- ifelse(prob == "prob1", "Correct", "Incorrect")

  data.frame(
    label = labels,
    country = factor(country),
    group = factor(group, levels = c("Correct", "Incorrect")),
    stringsAsFactors = FALSE
  )
}

read_distance_data <- function(path, atol = 1e-8) {
  if (!file.exists(path)) stop("Distance file not found: ", path)

  df <- read.csv(
    path,
    row.names = 1,
    check.names = FALSE,
    stringsAsFactors = FALSE
  )

  mat <- validate_distance_matrix(as.matrix(df), atol = atol)
  meta <- parse_prob_labels(rownames(mat))

  counts <- table(meta$group)
  if (any(counts < 2L)) {
    stop(
      "Each response group must contain at least two graphs. counts = ",
      paste(names(counts), counts, sep = ":", collapse = ", ")
    )
  }

  d <- as.dist(mat)
  attr(d, "Labels") <- rownames(mat)

  list(matrix = mat, dist = d, meta = meta)
}

validate_country_pairs <- function(meta) {
  tab <- table(meta$country, meta$group)

  if (!all(c("Correct", "Incorrect") %in% colnames(tab))) {
    stop("Both Correct and Incorrect graphs are required in every country block.")
  }

  bad <- rownames(tab)[tab[, "Correct"] != 1L | tab[, "Incorrect"] != 1L]

  if (length(bad) > 0L) {
    stop(
      "Country-blocked permutations require exactly one Correct and one Incorrect graph per country. ",
      "Invalid countries: ", paste(bad, collapse = ", ")
    )
  }

  invisible(TRUE)
}

summarize_numeric <- function(values, name) {
  values <- as.numeric(values)
  q <- unname(quantile(values, probs = c(0.25, 0.75), names = FALSE, type = 7))

  data.frame(
    distance_set = name,
    n_pairs = length(values),
    mean = mean(values),
    sd = if (length(values) > 1L) sd(values) else NA_real_,
    median = median(values),
    q1 = q[[1L]],
    q3 = q[[2L]],
    iqr = q[[2L]] - q[[1L]],
    variance = if (length(values) > 1L) var(values) else NA_real_,
    min = min(values),
    max = max(values),
    row.names = NULL
  )
}

write_csv <- function(x, path, row.names = FALSE) {
  write.csv(x, path, row.names = row.names, na = "NA")
  message("Saved: ", path)
}
