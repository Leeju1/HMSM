# CD Tally: run from the repository root.
library(Rcpp)
library(RcppArmadillo)

# 1. Input files. Keep the original row, state, country, and covariate ordering.
data_file <- "data/cdtally/combined_cd_tally.RData"
tuning_dir <- "config/mcmc/tuning/cdtally"
out_dir <- "outputs/cdtally/mcmc"
load(data_file)
load(file.path(tuning_dir, "jump_kappa_cdtally.RData"))
load(file.path(tuning_dir, "jump_alpha_cdtally.RData"))
load(file.path(tuning_dir, "jump_beta_cdtally.RData"))

# 2. Prepare the model input and item-level key-action states.
key_states <- c(5, 15, 6, 18, 17, 19, 4)
data <- as.matrix(combined_data)
storage.mode(data) <- "double"
pid <- data[, 1]
state <- data[, 2]
country <- data[, 7]
N <- max(pid)
E <- max(state)
Q <- max(country)
P <- ncol(data) - 7L
same_id <- head(pid, -1) == tail(pid, -1)

# These are indexing/sequence requirements of the C++ likelihood.
stopifnot(
  nrow(data) > 1L, P > 0L, all(is.finite(data)),
  identical(sort(unique(pid)), as.numeric(seq_len(N))),
  identical(sort(unique(country)), as.numeric(seq_len(Q))),
  all(state == as.integer(state)), all(state >= 1),
  all(key_states %in% seq_len(E)), all(data[, 4] %in% 0:1),
  !anyDuplicated(rle(pid)$values),
  all(diff(data[, 3])[same_id] >= 0),
  all(diff(country)[same_id] == 0),
  all(diff(data[, 4])[same_id] == 0)
)
transition <- matrix(0, E, E)
transition[cbind(head(state, -1)[same_id], tail(state, -1)[same_id])] <- 1
state_key_indicator <- integer(E)
state_key_indicator[key_states] <- 1L
q_by_id <- integer(N)
q_by_id[pid] <- country

# 3. Proposal scales. The downloaded beta array has source/destination scales (2 x 2 x Q).
# Match the original run: copy the destination scale to the interaction.
stopifnot(identical(dim(jump_beta), c(2L, 2L, as.integer(Q))))
beta_scales <- array(0, c(2, 3, Q))
beta_scales[, 1:2, ] <- jump_beta
beta_scales[, 3, ] <- matrix(jump_beta[, 2, , drop = FALSE], 2, Q)
proposal <- list(kappa = jump_kappa, tau = rep(0.6, N), alpha = jump_alpha,
                 beta = beta_scales, mu_tau = rep(0.04, Q))

# 4. Priors (inverse-gamma shape/rate) and sampling settings.
prior <- list(
  mu_kappa = 0, sigma2_kappa = 1, sigma2_alpha = 1, sigma2_mu_beta = 1,
  a_tau = 0.001, b_tau = 0.001, c_tau = 0.001, d_tau = 0.001,
  a_alpha = 0.001, b_alpha = 0.001, a_beta = 0.001, b_beta = 0.001
)

# Four sequential chains. A single seed runs one chain for a pilot.
niter <- 300000
nburn <- 100000
nthin <- 20
nprint <- 1000
chain_seeds <- c(1001, 2001, 3001, 4001)

sourceCpp("mcmc/sampler.cpp")

# 5. Check dimensions and record the run settings.
active <- which(transition == 1 & row(transition) != col(transition),
                arr.ind = TRUE)
active_scales <- c(proposal$kappa[cbind(1, active)], proposal$kappa[cbind(2, active)],
                   proposal$tau, proposal$alpha, proposal$beta, proposal$mu_tau)
stopifnot(
  niter > nburn, nburn >= 0, nthin >= 1, nprint >= 1,
  all(c(niter, nburn, nthin, nprint) %% 1 == 0),
  length(chain_seeds) >= 1, !anyDuplicated(chain_seeds),
  all(is.finite(chain_seeds)), all(chain_seeds %% 1 == 0),
  all(dim(proposal$kappa) == c(2, E, E)), length(dim(proposal$kappa)) == 3,
  length(proposal$tau) == N,
  all(dim(proposal$alpha) == c(P, Q)), length(dim(proposal$alpha)) == 2,
  all(dim(proposal$beta) == c(2, 3, Q)), length(dim(proposal$beta)) == 3,
  length(proposal$mu_tau) == Q, all(is.finite(active_scales)), all(active_scales > 0)
)
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
chain_files <- sprintf("chain%02d_seed%d.RData", seq_along(chain_seeds), chain_seeds)
metadata <- list(
  chain_files = chain_files, chain_seeds = chain_seeds,
  mcmc_seeds = chain_seeds + 1000000L,
  niter = niter, nburn = nburn, nthin = nthin, nprint = nprint,
  N = N, E = E, P = P, Q = Q, transition = transition,
  state_key_indicator = state_key_indicator,
  proposal = proposal, prior = prior, session_info = sessionInfo()
)
saveRDS(metadata, file.path(out_dir, "run_metadata.rds"))

# 6. Initialize, sample, and save each chain.
for (chain_id in seq_along(chain_seeds)) {
  seed <- chain_seeds[chain_id]
  # Initialize this chain using its own seed.
  set.seed(seed)
  clip <- function(x, lo, hi) pmin(pmax(x, lo), hi)
  bounded_exp <- function(x) exp(clip(x, -2, 2))
  allowed <- which(transition == 1 & row(transition) != col(transition),
                   arr.ind = TRUE)
  kappa <- array(0, c(2, E, E))
  for (cc in 1:2) {
    log_kappa <- rnorm(nrow(allowed), prior$mu_kappa, sqrt(prior$sigma2_kappa))
    kappa[cbind(rep(cc, nrow(allowed)), allowed[, 1], allowed[, 2])] <-
      exp(clip(log_kappa, -4, 4))
  }
  mu_tau <- rnorm(Q, 0, 1)
  sigma2_tau <- bounded_exp(rnorm(Q, 0, 0.5))
  tau <- exp(clip(rnorm(N, mu_tau[q_by_id], sqrt(sigma2_tau[q_by_id])), -4, 4))
  alpha <- matrix(rnorm(P * Q, 0, 0.25), P, Q)
  beta_key <- array(rnorm(2 * 3 * Q, 0, 0.25), c(2, 3, Q))
  sigma2_mu_tau <- bounded_exp(rnorm(1, 0, 0.5))
  mu_alpha_p <- rnorm(P, 0, 0.5)
  sigma2_alpha_p <- bounded_exp(rnorm(P, 0, 0.5))

  # Preserve the original IG initialization stream: the old script consumed
  # P auxiliary draws here. They have no role in the submitted model.
  invisible(rnorm(P, 0, 0.5))
  mu_beta_cr <- matrix(rnorm(6, 0, 0.25), 2, 3)
  sigma2_beta_cr <- matrix(bounded_exp(rnorm(6, 0, 0.5)), 2, 3)
  init <- list(kappa = kappa, tau = tau, alpha = alpha, beta_key = beta_key,
       mu_tau = mu_tau, sigma2_tau = sigma2_tau, sigma2_mu_tau = sigma2_mu_tau,
       mu_alpha_p = mu_alpha_p, sigma2_alpha_p = sigma2_alpha_p,
       mu_beta_cr = mu_beta_cr, sigma2_beta_cr = sigma2_beta_cr)
  mcmc_seed <- seed + 1000000L
  set.seed(mcmc_seed)
  message("Chain ", chain_id, ": initialization seed = ", seed,
          ", MCMC seed = ", mcmc_seed)
  result <- sample_hmsm(
    data = data, niter = niter, nburn = nburn, nthin = nthin, nprint = nprint,
    transition = transition, state_key_indicator = state_key_indicator,
    jump_kappa = proposal$kappa, mu_kappa = prior$mu_kappa, sigma2_kappa = prior$sigma2_kappa,
    jump_tau = proposal$tau, mu_tau = init$mu_tau, sigma2_tau = init$sigma2_tau,
    jump_mu_tau = proposal$mu_tau, sigma2_mu_tau = init$sigma2_mu_tau,
    a_tau = prior$a_tau, b_tau = prior$b_tau, c_tau = prior$c_tau, d_tau = prior$d_tau,
    jump_alpha = proposal$alpha, mu_alpha_p = init$mu_alpha_p,
    sigma2_alpha_p = init$sigma2_alpha_p, sigma2_alpha = prior$sigma2_alpha,
    a_alpha = prior$a_alpha, b_alpha = prior$b_alpha,
    jump_beta = proposal$beta, mu_beta_cr = init$mu_beta_cr,
    sigma2_beta_cr = init$sigma2_beta_cr, sigma2_mu_beta = prior$sigma2_mu_beta,
    a_beta = prior$a_beta, b_beta = prior$b_beta,
    init_kappa = init$kappa, init_tau = init$tau, init_alpha = init$alpha,
    init_beta_key = init$beta_key
  )
  result$chain_id <- chain_id
  result$seed <- seed
  result$mcmc_seed <- mcmc_seed
  result$initial_values <- init
  save(result, file = file.path(out_dir, chain_files[chain_id]))
}

# 7. Identify the exact input files used for this run.
input_files <- c(data_file, file.path(tuning_dir, c(
  "jump_kappa_cdtally.RData", "jump_alpha_cdtally.RData",
  "jump_beta_cdtally.RData")))
write.csv(data.frame(file = input_files, md5 = unname(tools::md5sum(input_files))),
          file.path(out_dir, "input_files.csv"), row.names = FALSE)
