# Run from the repository root after MCMC sampling.
source("mcmc/transition.R")

data_file <- "data/lamp_return/combined_lamp_return.RData"
chain_file <- "outputs/lamp_return/mcmc/chain01_seed1001.RData"
state_file <- "data/lamp_return/state_name_list_lamp.csv"
out_dir <- "outputs/lamp_return/graphs"

threshold <- 3
x <- c(1, 5, 2, 4, 1)
draw_indices <- NULL  # All retained draws; use 1:2 for a pilot.

build_transition_graphs(data_file, chain_file, state_file, out_dir,
                        threshold, x, draw_indices)
