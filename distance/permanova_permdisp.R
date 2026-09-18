
source("distance/utils.R")
cfg <- read_analysis_config(commandArgs(trailingOnly = TRUE))
dist_file <- cfg$distance_file
outdir <- cfg$analysis_dir
seed <- as.integer(cfg$seed)
atol <- cfg$atol
country_blocks <- cfg$country_blocks
n_perm_unrestricted <- as.integer(cfg$n_perm_unrestricted)
require_packages(c("vegan", "permute", "ggplot2"))

outdir <- ensure_outdir(outdir)
obj <- read_distance_data(dist_file, atol = atol)
validate_analysis_labels(obj, cfg)

d <- obj$dist
meta <- obj$meta

# Swap responses only within country; enumerate 2^14 - 1 nonobserved arrangements.
if (country_blocks) {

  validate_country_pairs(meta)

  perm_design <- permute::how(
    within = permute::Within(type = "free"),
    blocks = meta$country,
    complete = TRUE,
    maxperm = 20000,
    observed = FALSE
  )

  n_possible <- permute::numPerms(
    nrow(meta),
    control = perm_design
  )

  perm_matrix <- as.matrix(
    permute::allPerms(
      nrow(meta),
      control = perm_design
    )
  )

  permutation_scheme <- "country-blocked exhaustive"

} else {

  perm_design <- permute::how(
    within = permute::Within(type = "free"),
    nperm = n_perm_unrestricted,
    observed = FALSE
  )

  n_possible <- permute::numPerms(
    nrow(meta),
    control = perm_design
  )

  set.seed(seed)

  perm_matrix <- permute::shuffleSet(
    n = nrow(meta),
    nset = n_perm_unrestricted,
    control = perm_design,
    quietly = TRUE
  )

  perm_matrix <- as.matrix(perm_matrix)

  permutation_scheme <- "unrestricted Monte Carlo"
}

n_perm_used <- nrow(perm_matrix)

cat("=== Permutation design ===\n")
cat("scheme       : ", permutation_scheme, "\n", sep = "")
cat("generated    : ", n_perm_used, "\n", sep = "")
cat("possible     : ", format(n_possible, scientific = FALSE), "\n", sep = "")
cat("seed         : ", seed, "\n\n", sep = "")

# Use original distances; country defines permutation blocks, not a covariate.
permanova <- vegan::adonis2(
  d ~ group,
  data = meta,
  permutations = perm_matrix,
  by = NULL,
  sqrt.dist = FALSE,
  add = FALSE,
  parallel = 1
)

permanova_df <- data.frame(
  term = rownames(permanova),
  as.data.frame(permanova),
  row.names = NULL
)

# Dispersion uses the spatial median without distance or bias corrections.
bd <- vegan::betadisper(
  d,
  group = meta$group,
  type = "median",
  bias.adjust = FALSE,
  sqrt.dist = FALSE,
  add = FALSE
)

permdisp <- vegan::permutest(
  bd,
  permutations = perm_matrix,
  pairwise = FALSE,
  parallel = 1
)

permdisp_df <- data.frame(
  term = rownames(permdisp$tab),
  as.data.frame(permdisp$tab),
  row.names = NULL
)

dispersion_by_graph <- transform(
  meta,
  distance_to_group_spatial_median = as.numeric(bd$distances)
)

write_csv(
  dispersion_by_graph,
  file.path(outdir, "03_permdisp_by_graph.csv")
)

disp_split <- split(
  dispersion_by_graph$distance_to_group_spatial_median,
  dispersion_by_graph$group
)

dispersion_by_group <- data.frame(
  group = names(disp_split),
  n = vapply(disp_split, length, integer(1)),
  mean = vapply(disp_split, mean, numeric(1)),
  sd = vapply(disp_split, sd, numeric(1)),
  median = vapply(disp_split, median, numeric(1)),
  row.names = NULL
)

write_csv(
  dispersion_by_group,
  file.path(outdir, "03_permdisp_group_summary.csv")
)

eig <- as.numeric(bd$eig)

pos_sum <- sum(eig[eig > 0])
neg_abs_sum <- abs(sum(eig[eig < 0]))

neg_ratio <- if ((pos_sum + neg_abs_sum) > 0) {
  neg_abs_sum / (pos_sum + neg_abs_sum)
} else {
  0
}

eigen_summary <- data.frame(
  metric = c(
    "n_positive_eigenvalues",
    "n_negative_eigenvalues",
    "positive_eigenvalue_sum",
    "absolute_negative_eigenvalue_sum",
    "negative_eigenvalue_ratio_percent"
  ),
  value = c(
    sum(eig > 0),
    sum(eig < 0),
    pos_sum,
    neg_abs_sum,
    100 * neg_ratio
  )
)

perm_model_row <- which(rownames(permanova) == "Model")
if (length(perm_model_row) != 1L) perm_model_row <- 1L

pd_group_row <- which(rownames(permdisp$tab) == "Groups")
if (length(pd_group_row) != 1L) pd_group_row <- 1L

tests_summary <- data.frame(
  test = c("PERMANOVA", "PERMDISP"),
  df1 = c(permanova[perm_model_row, "Df"], permdisp$tab[pd_group_row, "Df"]),
  df2 = c(permanova["Residual", "Df"], permdisp$tab["Residuals", "Df"]),
  F = c(permanova[perm_model_row, "F"], permdisp$tab[pd_group_row, "F"]),
  R2 = c(permanova[perm_model_row, "R2"], NA_real_),
  p_value = c(permanova[perm_model_row, "Pr(>F)"], permdisp$tab[pd_group_row, "Pr(>F)"]),
  n_permutations = n_perm_used,
  permutation_scheme = permutation_scheme,
  center = c(NA_character_, "spatial median"),
  row.names = NULL
)
write_csv(tests_summary, file.path(outdir, "03_tests_summary.csv"))

library(ggplot2)

color_correct   <- "#2196F3"
color_incorrect <- "#F44336"

group_colors <- c(
  "Correct"   = color_correct,
  "Incorrect" = color_incorrect
)

permdisp_p <- as.numeric(
  permdisp$tab[pd_group_row, "Pr(>F)"]
)

fmt_p <- function(p) {
  if (is.na(p)) {
    "NA"
  } else if (p < 0.001) {
    "< 0.001"
  } else {
    sprintf("= %.3f", p)
  }
}

plot_df <- dispersion_by_graph

plot_df$group <- factor(
  plot_df$group,
  levels = c("Correct", "Incorrect")
)

pA <- ggplot(
  plot_df,
  aes(
    x = group,
    y = distance_to_group_spatial_median,
    fill = group,
    color = group
  )
) +
  
  geom_boxplot(
    width = 0.52,
    alpha = 0.28,
    linewidth = 0.8,
    outlier.shape = NA
  ) +
  
  geom_point(
    position = position_jitter(width = 0.10, height = 0, seed = seed),
    size = 2.8,
    alpha = 0.80
  ) +
  
  scale_fill_manual(
    values = group_colors
  ) +
  
  scale_color_manual(
    values = group_colors
  ) +
  
  labs(
    x = NULL,
    y = "Distance to group spatial median"
  ) +
  
  annotate(
    "label",
    x = 0.55,
    y = Inf,
    label = paste0("PERMDISP: p ", fmt_p(permdisp_p)),
    hjust = 0,
    vjust = 1.4,
    size = 4.8,
    fill = "white"
  ) +
  
  theme_bw(base_size = 15) +
  
  theme(
    legend.position = "none",
    
    axis.title.y = element_text(size = 14),
    axis.text.x  = element_text(size = 13),
    axis.text.y  = element_text(size = 12),

    panel.border = element_rect(
      colour = "black",
      fill = NA,
      linewidth = 0.8
    ),

    panel.grid.major.y = element_line(
      colour = "grey85",
      linewidth = 0.45
    ),
    panel.grid.minor.y = element_blank(),
    panel.grid.major.x = element_blank(),
    panel.grid.minor.x = element_blank()
  )

ggsave(
  filename = file.path(
    outdir,
    "03_permdisp_group.png"
  ),
  plot = pA,
  width = 6.0,
  height = 5.5,
  dpi = 300,
  bg = "white"
)

if (cfg$diagnostic_plots) {
bar_df <- dispersion_by_graph

bar_df$display_label <- paste0(
  as.character(bar_df$country),
  " (",
  as.character(bar_df$group),
  ")"
)

bar_df <- bar_df[
  order(
    factor(
      bar_df$group,
      levels = c("Incorrect", "Correct")
    ),
    -bar_df$distance_to_group_spatial_median
  ),
]

bar_df$display_label <- factor(
  bar_df$display_label,
  levels = rev(bar_df$display_label)
)

group_means <- aggregate(
  distance_to_group_spatial_median ~ group,
  data = dispersion_by_graph,
  FUN = mean
)

mean_correct <- group_means$distance_to_group_spatial_median[
  group_means$group == "Correct"
]

mean_incorrect <- group_means$distance_to_group_spatial_median[
  group_means$group == "Incorrect"
]

pB <- ggplot(
  bar_df,
  aes(
    x = distance_to_group_spatial_median,
    y = display_label,
    fill = group
  )
) +
  
  geom_col(
    width = 0.72,
    alpha = 0.78
  ) +
  
  geom_vline(
    xintercept = mean_correct,
    color = color_correct,
    linetype = "dashed",
    linewidth = 0.8
  ) +
  
  geom_vline(
    xintercept = mean_incorrect,
    color = color_incorrect,
    linetype = "dashed",
    linewidth = 0.8
  ) +
  
  scale_fill_manual(
    values = group_colors
  ) +
  
  labs(
    x = "Distance to group spatial median",
    y = NULL
  ) +
  
  theme_bw(base_size = 15) +
  
  theme(
    legend.position = "none",
    
    axis.title.x = element_text(size = 14),
    axis.text.x  = element_text(size = 12),
    axis.text.y  = element_text(size = 10.5),

    panel.border = element_rect(
      colour = "black",
      fill = NA,
      linewidth = 0.8
    ),

    panel.grid.major.x = element_line(
      colour = "grey90",
      linewidth = 0.4
    ),
    panel.grid.minor.x = element_blank(),
    panel.grid.major.y = element_blank(),
    panel.grid.minor.y = element_blank()
  )

ggsave(
  filename = file.path(
    outdir,
    "03_permdisp_by_graph.png"
  ),
  plot = pB,
  width = 7.0,
  height = 7.0,
  dpi = 300,
  bg = "white"
)
}

message(
  "Saved: ",
  file.path(outdir, "03_permdisp_group.png")
)

if (cfg$diagnostic_plots) {
message(
  "Saved: ",
  file.path(outdir, "03_permdisp_by_graph.png")
)
}

if (cfg$diagnostic_plots) {
png(
  file.path(outdir, "03_permdisp_pcoa.png"),
  width = 1200,
  height = 1000,
  res = 160
)

plot(
  bd,
  hull = FALSE,
  ellipse = FALSE,
  label = TRUE,
  main = "PERMDISP principal-coordinate representation"
)

dev.off()
}

cat("\n=== PERMANOVA (vegan::adonis2) ===\n")
print(permanova)

cat("\n=== PERMDISP (vegan::betadisper + permutest) ===\n")
print(permdisp)

cat("\n=== Mean distance to group spatial median ===\n")
print(dispersion_by_group, row.names = FALSE)

save_analysis_metadata(cfg, "permanova_permdisp", details = list(
  permanova = permanova_df,
  permdisp = permdisp_df,
  pcoa_eigenvalues = eig,
  eigen_summary = eigen_summary,
  n_possible_permutations = n_possible,
  n_permutations_used = n_perm_used,
  permutation_scheme = permutation_scheme
))
