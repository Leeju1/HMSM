# Run from the repository root; see README.md.
source("distance/utils.R")
cfg <- read_analysis_config(commandArgs(trailingOnly = TRUE))
dist_file <- cfg$distance_file
outdir <- cfg$analysis_dir
seed <- as.integer(cfg$seed)
atol <- cfg$atol
n_try <- as.integer(cfg$nmds_try)
trymax <- as.integer(cfg$nmds_trymax)
maxit <- as.integer(cfg$nmds_maxit)
require_packages("vegan")

outdir <- ensure_outdir(outdir)
obj <- read_distance_data(dist_file, atol = atol)
validate_analysis_labels(obj, cfg)

d <- obj$dist
meta <- obj$meta

run_nmds <- function(k, seed_value) {
  set.seed(seed_value)

  vegan::metaMDS(
    d,
    k = k,
    try = n_try,
    trymax = trymax,
    maxit = maxit,
    autotransform = FALSE,
    noshare = FALSE,
    wascores = FALSE,
    halfchange = FALSE,
    trace = FALSE
  )
}

# ---- NMDS --------------------------------------------------------------------
cat("Running 2D NMDS with vegan::metaMDS ...\n")
mds2 <- run_nmds(k = 2L, seed_value = seed)

cat("Running 3D NMDS with vegan::metaMDS ...\n")
mds3 <- run_nmds(k = 3L, seed_value = seed)

coords2 <- as.data.frame(
  vegan::scores(mds2, display = "sites", choices = 1:2)
)
colnames(coords2) <- c("NMDS1", "NMDS2")
coords2$label <- rownames(coords2)

coords2 <- merge(meta, coords2, by = "label", sort = FALSE)
coords2 <- coords2[match(meta$label, coords2$label), ]

coords3 <- as.data.frame(
  vegan::scores(mds3, display = "sites", choices = 1:3)
)
colnames(coords3) <- c("NMDS1", "NMDS2", "NMDS3")
coords3$label <- rownames(coords3)

coords3 <- merge(meta, coords3, by = "label", sort = FALSE)
coords3 <- coords3[match(meta$label, coords3$label), ]

# Both solutions share one ordered graph-label table.
stopifnot(identical(coords2$label, coords3$label))
coordinates <- coords2[, c("label", "country", "group")]
coordinates$NMDS2D_1 <- coords2$NMDS1
coordinates$NMDS2D_2 <- coords2$NMDS2
coordinates$NMDS3D_1 <- coords3$NMDS1
coordinates$NMDS3D_2 <- coords3$NMDS2
coordinates$NMDS3D_3 <- coords3$NMDS3
summary_df <- data.frame(dimensions = c(2L, 3L), stress = c(mds2$stress, mds3$stress))
write_csv(coordinates, file.path(outdir, "02_nmds_coordinates.csv"))
write_csv(summary_df, file.path(outdir, "02_nmds_summary.csv"))

# ---- 2D NMDS scatter ---------------------------------------------------------
color_correct <- "#2196F3"
color_incorrect <- "#F44336"

cols <- ifelse(
  coords2$group == "Correct",
  color_correct,
  color_incorrect
)

pchs <- ifelse(
  coords2$group == "Correct",
  21,
  24
)

png(
  file.path(outdir, "02_nmds_2d.png"),
  width = 1400,
  height = 1100,
  res = 170
)

par(mar = c(4.5, 4.5, 3.5, 1.5))

x_rng <- range(coords2$NMDS1)
y_rng <- range(coords2$NMDS2)

x_pad <- diff(x_rng) * 0.08
y_pad <- diff(y_rng) * 0.10

plot(
  coords2$NMDS1,
  coords2$NMDS2,
  type = "n",
  xlim = c(x_rng[1] - x_pad, x_rng[2] + x_pad),
  ylim = c(y_rng[1] - y_pad, y_rng[2] + y_pad),
  xlab = "MDS Dim 1",
  ylab = "MDS Dim 2",
  main = ""
)

grid(col = "grey90")

points(
  coords2$NMDS1,
  coords2$NMDS2,
  pch = pchs,
  bg = cols,
  col = "white",
  cex = 1.6,
  lwd = 0.7
)

text(
  coords2$NMDS1,
  coords2$NMDS2,
  labels = as.character(coords2$country),
  pos = 3,
  cex = 0.72
)

legend(
  "topleft",
  legend = c("Correct", "Incorrect"),
  pch = c(21, 24),
  pt.bg = c(color_correct, color_incorrect),
  col = "white",
  pt.cex = 1.5,
  bty = "n"
)

dev.off()

# ---- Shepard/stress plots ----------------------------------------------------
if (cfg$diagnostic_plots) {
png(
  file.path(outdir, "02_nmds_shepard.png"),
  width = 1800,
  height = 850,
  res = 160
)

par(
  mfrow = c(1, 2),
  mar = c(4.5, 4.5, 3.5, 1.5)
)

vegan::stressplot(
  mds2,
  main = sprintf(
    "Shepard plot: 2D (stress = %.3f)",
    mds2$stress
  )
)

vegan::stressplot(
  mds3,
  main = sprintf(
    "Shepard plot: 3D (stress = %.3f)",
    mds3$stress
  )
)

dev.off()
}

cat("\n=== NMDS summary ===\n")
cat(sprintf("2D stress: %.6f\n", mds2$stress))
cat(sprintf("3D stress: %.6f\n", mds3$stress))
cat(
  "PERMANOVA/PERMDISP use the original distance matrix, ",
  "not these NMDS coordinates.\n",
  sep = ""
)


save_analysis_metadata(cfg, "nmds", details = list(engine = "monoMDS", stress = summary_df))
