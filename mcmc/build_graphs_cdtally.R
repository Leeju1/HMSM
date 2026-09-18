# Run from the repository root after MCMC sampling.
source("mcmc/transition.R")

data_file <- "data/cdtally/combined_cd_tally.RData"
chain_file <- "outputs/cdtally/mcmc/chain01_seed1001.RData"
state_file <- "data/cdtally/state_name_list_cdtally.csv"
out_dir <- "outputs/cdtally/graphs"

threshold <- 3
x <- c(1, 5, 2, 4, 1)
draw_indices <- NULL  # All retained draws; use 1:2 for a pilot.

build_transition_graphs(data_file, chain_file, state_file, out_dir,
                        threshold, x, draw_indices)
