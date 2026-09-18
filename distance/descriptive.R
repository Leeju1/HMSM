# Run from the repository root; see README.md.
source("distance/utils.R")
cfg <- read_analysis_config(commandArgs(trailingOnly = TRUE))
dist_file <- cfg$distance_file
outdir <- cfg$analysis_dir
atol <- cfg$atol

outdir <- ensure_outdir(outdir)
obj <- read_distance_data(dist_file, atol = atol)
validate_analysis_labels(obj, cfg)

D <- obj$matrix
meta <- obj$meta

idx_c <- which(meta$group == "Correct")
idx_i <- which(meta$group == "Incorrect")

pw_c <- D[idx_c, idx_c, drop = FALSE]
pw_c <- pw_c[upper.tri(pw_c)]

pw_i <- D[idx_i, idx_i, drop = FALSE]
pw_i <- pw_i[upper.tri(pw_i)]

pw_between <- as.vector(D[idx_c, idx_i, drop = FALSE])

summary_df <- rbind(
  summarize_numeric(pw_c, "Correct intra-group"),
  summarize_numeric(pw_i, "Incorrect intra-group"),
  summarize_numeric(pw_between, "Between groups")
)

peer_mean <- vapply(seq_len(nrow(meta)), function(i) {
  peers <- which(
    meta$group == meta$group[[i]] &
      seq_len(nrow(meta)) != i
  )
  mean(D[i, peers])
}, numeric(1))

by_graph <- transform(
  meta,
  mean_distance_to_same_group_peers = peer_mean
)

by_graph <- by_graph[
  order(by_graph$group, by_graph$mean_distance_to_same_group_peers),
]

cat("=== Pairwise Wasserstein distance descriptive statistics ===\n")
print(summary_df, row.names = FALSE, digits = 5)

write_csv(
  summary_df,
  file.path(outdir, "01_distance_summary.csv")
)

write_csv(
  by_graph,
  file.path(outdir, "01_within_group_by_graph.csv")
)

save_analysis_metadata(cfg, "descriptive")
