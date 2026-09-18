// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(cpp11)]]

#include <RcppArmadillo.h>
#include <chrono>
#include <vector>
#include <iomanip>
#include <cmath>
using namespace Rcpp;

// Hierarchical multi-state survival model with inverse-gamma variance priors.
// beta[c, r, q]: source-key, destination-key, and interaction effects (r = 0,1,2).
// Input columns: pid, state, time, response, key1, key2, country, covariates.
// IDs are 1-based; response is 0/1. Rows are grouped by pid in time order.
// jump_* are proposal standard deviations (log scale for kappa and tau).
// [[Rcpp::export]]
List sample_hmsm(
    const arma::mat& data, const int niter, const int nburn, const int nthin, const int nprint,
    const arma::mat& transition,
    arma::ivec state_key_indicator,
    arma::cube jump_kappa, double mu_kappa, double sigma2_kappa,
    arma::vec jump_tau, arma::vec mu_tau, arma::vec sigma2_tau, arma::vec jump_mu_tau,
    double sigma2_mu_tau, double a_tau, double b_tau, double c_tau, double d_tau,
    arma::mat jump_alpha, arma::vec mu_alpha_p, arma::vec sigma2_alpha_p,
    double sigma2_alpha, double a_alpha, double b_alpha,
    arma::cube jump_beta, arma::mat mu_beta_cr, arma::mat sigma2_beta_cr,
    double sigma2_mu_beta, double a_beta, double b_beta,
    const arma::cube& init_kappa, const arma::vec& init_tau,
    const arma::mat& init_alpha, const arma::cube& init_beta_key
){
    if (data.n_rows < 2 || data.n_cols < 8 || niter <= nburn || nburn < 0 || nthin < 1 || nprint < 1) {
        Rcpp::stop("Invalid data shape or iteration settings.");
    }
    int nrow = data.n_rows;
    int P = data.n_cols - 7;
    int N = arma::max(data.col(0));
    int E = transition.n_rows;
    int Q = mu_tau.n_elem;
    int ndraw = (niter - nburn + nthin - 1) / nthin;
    int draw = 0;
    double n_trans = 0.0;
    for (int nt_m = 0; nt_m < E; ++nt_m) {
        for (int nt_l = 0; nt_l < E; ++nt_l) {
            if (nt_l != nt_m && transition(nt_m, nt_l)) n_trans += 1.0;
        }
    }

    int iter, c, m, l, k, p, i, q, j, r, n_q, cc, mm, d;
    int id, this_key, country_idx;

    double old_kappa, new_kappa, old_tau, new_tau, old_alpha, new_alpha, old_beta, new_beta;
    double log_prior_old, log_prior_new;
    double diff_event, diff_surv, diff_prior, hastings,log_ratio;
    double lp, s_i, w_i, k_all, k_minus, base, lp_cov, other_eff;
    double expo_cm, expo_cml, acc, prec, sum_mu, diff_scalar_alpha;
    double delta_beta, old_row_sum, new_row_sum;

    double old_mu, new_mu, sum_old, sum_new, log_tau, diff_like;
    double sum_sq, shape, rate, post_var, post_mean;
    double sum_b, sum_mu2, shape_mu, rate_mu, ss, diff_beta;
    double lp_fixed, lp_no_p, this_key_val ;

    std::chrono::steady_clock::time_point t_start;
    std::chrono::steady_clock::time_point t_now;
    double elapsed, frac, eta;

    arma::ivec res_i(N), q_i(N);
    arma::vec all_betas;
    arma::rowvec diff;
    arma::uvec idx;

    arma::cube kappa = init_kappa;
    arma::cube weighted_row_sum(2, Q, E, arma::fill::zeros);
    arma::cube expo_base_cqm(2, Q, E, arma::fill::zeros);
    arma::vec tau = init_tau;
    arma::mat alpha = init_alpha;
    arma::cube beta_key = init_beta_key;

    if ((int)kappa.n_rows != 2 || (int)kappa.n_cols != E || (int)kappa.n_slices != E) {
        Rcpp::stop("init_kappa must have dimension 2 x E x E.");
    }
    if ((int)tau.n_elem != N) {
        Rcpp::stop("init_tau must have length N, where N = max(data[, 1]).");
    }
    if ((int)alpha.n_rows != P || (int)alpha.n_cols != Q) {
        Rcpp::stop("init_alpha must have dimension P x Q.");
    }
    if ((int)mu_alpha_p.n_elem != P || (int)sigma2_alpha_p.n_elem != P) {
        Rcpp::stop("mu_alpha_p and sigma2_alpha_p must each have length P.");
    }
    for (p = 0; p < P; ++p) {
        if (!std::isfinite(mu_alpha_p(p))) {
            Rcpp::stop("Every element of mu_alpha_p must be finite.");
        }
        if (!std::isfinite(sigma2_alpha_p(p)) || sigma2_alpha_p(p) <= 0.0) {
            Rcpp::stop("Every element of sigma2_alpha_p must be positive and finite.");
        }
    }
    if ((a_alpha <= 0.0 || b_alpha <= 0.0)) {
        Rcpp::stop("a_alpha and b_alpha must be positive.");
    }
    if ((int)beta_key.n_rows != 2 || (int)beta_key.n_cols != 3 || (int)beta_key.n_slices != Q) {
        Rcpp::stop("init_beta_key must have dimension 2 x 3 x Q: beta1=source key, beta2=destination key, beta3=source-destination key interaction.");
    }
    if ((int)jump_beta.n_rows != 2 || (int)jump_beta.n_cols != 3 || (int)jump_beta.n_slices != Q) {
        Rcpp::stop("jump_beta must have dimension 2 x 3 x Q.");
    }
    if ((int)mu_beta_cr.n_rows != 2 || (int)mu_beta_cr.n_cols != 3) {
        Rcpp::stop("mu_beta_cr must have dimension 2 x 3 (response group c x effect type r).");
    }
    if ((int)sigma2_beta_cr.n_rows != 2 || (int)sigma2_beta_cr.n_cols != 3) {
        Rcpp::stop("sigma2_beta_cr must have dimension 2 x 3.");
    }
    for (c = 0; c < 2; ++c) {
        for (r = 0; r < 3; ++r) {
            if (!std::isfinite(mu_beta_cr(c, r))) {
                Rcpp::stop("Every element of mu_beta_cr must be finite.");
            }
            if (!std::isfinite(sigma2_beta_cr(c, r)) || sigma2_beta_cr(c, r) <= 0.0) {
                Rcpp::stop("Every element of sigma2_beta_cr must be positive and finite.");
            }
        }
    }
    if ((a_beta <= 0.0 || b_beta <= 0.0)) {
        Rcpp::stop("a_beta and b_beta must be positive.");
    }
    for (i = 0; i < N; ++i) {
        if (!std::isfinite(tau(i)) || tau(i) <= 0.0) {
            Rcpp::stop("Every element of init_tau must be positive and finite.");
        }
    }
    for (c = 0; c < 2; ++c) {
        for (m = 0; m < E; ++m) {
            for (l = 0; l < E; ++l) {
                if (l == m || transition(m, l) == 0) {

                    kappa(c, m, l) = 0.0;
                } else if (!std::isfinite(kappa(c, m, l)) || kappa(c, m, l) <= 0.0) {
                    Rcpp::stop("init_kappa must be positive and finite for every allowed off-diagonal transition.");
                }
            }
        }
    }
    for (p = 0; p < P; ++p) {
        for (q = 0; q < Q; ++q) {
            if (!std::isfinite(alpha(p, q))) {
                Rcpp::stop("Every element of init_alpha must be finite.");
            }
        }
    }
    for (c = 0; c < 2; ++c) {
        for (r = 0; r < 3; ++r) {
            for (q = 0; q < Q; ++q) {
                if (!std::isfinite(beta_key(c, r, q))) {
                    Rcpp::stop("Every element of init_beta_key must be finite.");
                }
            }
        }
    }

    arma::cube accept_kappa(2,E,E, arma::fill::zeros);
    arma::vec accept_tau(N, arma::fill::zeros);
    arma::mat accept_alpha(P,Q, arma::fill::zeros);
    arma::cube accept_beta_key(2,3,Q, arma::fill::zeros);
    arma::vec accept_mu_tau(Q, arma::fill::zeros);

    arma::cube kappa0_save(E, E, ndraw, arma::fill::zeros);
    arma::cube kappa1_save(E, E, ndraw, arma::fill::zeros);
    arma::mat tau_save(N, ndraw, arma::fill::zeros);
    arma::mat mu_tau_save(Q, ndraw, arma::fill::zeros);
    arma::mat sigma2_tau_save(Q, ndraw, arma::fill::zeros);
    arma::vec sigma2_mu_tau_save(ndraw, arma::fill::zeros);
    arma::cube alpha_save(P, Q, ndraw, arma::fill::zeros);
    arma::mat mu_alpha_save(P, ndraw, arma::fill::zeros);
    arma::mat sigma2_alpha_p_save(P, ndraw, arma::fill::zeros);
    arma::cube beta_key0_save(3, Q, ndraw, arma::fill::zeros);
    arma::cube beta_key1_save(3, Q, ndraw, arma::fill::zeros);
    arma::cube mu_beta_cr_save(2, 3, ndraw, arma::fill::zeros);
    arma::cube sigma2_beta_cr_save(2, 3, ndraw, arma::fill::zeros);

    // Consecutive observations within a respondent define time-at-risk segments.
    struct Seg { int id, from, to, res, country, key1, key2, key3; double dt; arma::rowvec covs;};
    std::vector<Seg> segments;
    segments.reserve(nrow-1);

    if ((int)state_key_indicator.n_elem != E) {
        Rcpp::stop("state_key_indicator must have length E (number of action states).");
    }
    // Both event and survival terms use this fixed item-level key-action set.
    arma::ivec state_is_key(E);
    for (m = 0; m < E; ++m) {
        int v = state_key_indicator(m);
        if (v != 0 && v != 1) {
            Rcpp::stop("state_key_indicator must contain only 0/1 values.");
        }
        state_is_key(m) = v;
    }

    long n_key1_col_mismatch = 0;
    long n_key2_col_mismatch = 0;

    for (k = 0; k < nrow - 1; ++k){
        if (data(k,0) == data(k+1,0)) {
            Seg s;
            s.id = data(k,0) - 1;
            s.from = data(k,1) - 1;
            s.to = data(k+1,1) - 1;
            s.res = data(k,3);
            s.country = data(k,6) - 1;

            s.key1 = state_is_key(s.from);
            s.key2 = state_is_key(s.to);
            s.key3 = s.key1 * s.key2;

            if ((int)data(k,4) != s.key1) n_key1_col_mismatch += 1;
            if ((int)data(k,5) != s.key2) n_key2_col_mismatch += 1;

            s.dt = data(k+1,2) - data(k,2);
            s.covs = data.row(k).cols(7,6+P);
            if (transition(s.from, s.to) == 1) segments.push_back(s);
        }
    }

    // Index segments once; update cached predictors during sampling.
    std::vector<std::vector<int>> segs_by_id(N);
    std::vector<std::vector<int>> ids_by_country(Q);
    std::vector<std::vector<int>> segs_by_country(Q);

    std::vector<std::vector<std::vector<int>>> segs_by_res_from(2, std::vector<std::vector<int>>(E));
    std::vector<std::vector<std::vector<int>>> segs_by_res_country(2, std::vector<std::vector<int>>(Q));

    arma::cube event_count(2, E, E, arma::fill::zeros);

    for (int k = 0; k < (int)segments.size(); ++k) {
        const auto& s = segments[k];

        segs_by_id[s.id].push_back(k);
        segs_by_country[s.country].push_back(k);
        segs_by_res_from[s.res][s.from].push_back(k);
        segs_by_res_country[s.res][s.country].push_back(k);

        event_count(s.res, s.from, s.to) += 1.0;
    }

    for(k = 0; k < nrow; ++k){
        id = data(k,0) - 1;
        res_i(id) = data(k,3);
        country_idx = data(k,6) - 1;
        q_i(id) = country_idx;
    }
    for (int i = 0; i < N; ++i) {
        ids_by_country[q_i(i)].push_back(i);
    }

    // Source-common log hazard; destination effects enter weighted row sums.
    arma::vec LP_common_cache(segments.size(), arma::fill::zeros);
    for (int k = 0; k < (int)segments.size(); ++k) {
        const auto& s = segments[k];
        LP_common_cache(k) = arma::dot(s.covs, alpha.col(s.country))
                           + beta_key(s.res, 0, s.country) * s.key1;
    }

    auto compute_weighted_row_sum = [&](int cc, int qq, int mm,
                                        double beta2_value, double beta3_value) -> double {
        double total = 0.0;
        int source_key = state_is_key(mm);
        for (int ll = 0; ll < E; ++ll) {
            if (ll == mm) continue;
            if (!transition(mm, ll)) continue;
            int dest_key = state_is_key(ll);
            total += kappa(cc, mm, ll) *
                     std::exp(beta2_value * dest_key + beta3_value * source_key * dest_key);
        }
        return total;
    };

    auto recompute_all_weighted_row_sums = [&]() {
        weighted_row_sum.zeros();
        for (int cc2 = 0; cc2 < 2; ++cc2) {
            for (int qq2 = 0; qq2 < Q; ++qq2) {
                for (int mm2 = 0; mm2 < E; ++mm2) {
                    weighted_row_sum(cc2, qq2, mm2) = compute_weighted_row_sum(
                        cc2, qq2, mm2, beta_key(cc2, 1, qq2), beta_key(cc2, 2, qq2)
                    );
                }
            }
        }
    };

    auto recompute_weighted_row_sums_one_cq = [&](int cc2, int qq2) {
        for (int mm2 = 0; mm2 < E; ++mm2) {
            weighted_row_sum(cc2, qq2, mm2) = compute_weighted_row_sum(
                cc2, qq2, mm2, beta_key(cc2, 1, qq2), beta_key(cc2, 2, qq2)
            );
        }
    };

    auto recompute_expo_base_cqm = [&]() {
        expo_base_cqm.zeros();
        for (int kk2 = 0; kk2 < (int)segments.size(); ++kk2) {
            const auto& ss2 = segments[kk2];
            expo_base_cqm(ss2.res, ss2.country, ss2.from) +=
                tau(ss2.id) * std::exp(LP_common_cache(kk2)) * ss2.dt;
        }
    };

    recompute_all_weighted_row_sums();

    t_start = std::chrono::steady_clock::now();

    for (iter = 0; iter < niter; iter++){

        // Kappa: log-scale random walk with destination-specific survival exposure.
        recompute_expo_base_cqm();
        for (c = 0; c < 2; ++c) {
            for (m = 0; m < E; ++m) {
                int source_key = state_is_key(m);

                for (l = 0; l < E; ++l) {
                    if (l == m || !transition(m, l)) continue;
                    int dest_key = state_is_key(l);

                    expo_cml = 0.0;
                    for (q = 0; q < Q; ++q) {
                        expo_cml += expo_base_cqm(c, q, m) *
                                    std::exp(beta_key(c, 1, q) * dest_key +
                                             beta_key(c, 2, q) * source_key * dest_key);
                    }

                    old_kappa = kappa(c,m,l);
                    new_kappa = R::rlnorm(std::log(old_kappa), jump_kappa(c,m,l));

                    diff_event = event_count(c, m, l) * (std::log(new_kappa) - std::log(old_kappa));
                    diff_surv  = -(new_kappa - old_kappa) * expo_cml;

                    log_prior_old = R::dlnorm(old_kappa, mu_kappa, std::sqrt(sigma2_kappa), 1);
                    log_prior_new = R::dlnorm(new_kappa, mu_kappa, std::sqrt(sigma2_kappa), 1);
                    hastings = std::log(new_kappa) - std::log(old_kappa);

                    log_ratio = diff_event + diff_surv + (log_prior_new - log_prior_old) + hastings;

                    if (std::log(R::runif(0,1)) < log_ratio) {
                        kappa(c,m,l) = new_kappa;
                        accept_kappa(c,m,l) += 1.0;
                    }
                }
            }
        }

        recompute_all_weighted_row_sums();

        for (i = 0; i < N; ++i) {
            old_tau = tau(i);
            // Tau: log-scale random walk for individual speed.
            new_tau = R::rlnorm(std::log(old_tau), jump_tau(i));
            s_i = (double)segs_by_id[i].size();
            w_i = 0.0;
            q = q_i(i);

            for (int idx_k : segs_by_id[i]) {
                const auto& s = segments[idx_k];
                w_i += weighted_row_sum(s.res, s.country, s.from) * std::exp(LP_common_cache(idx_k)) * s.dt;
            }

            diff_event = s_i * (std::log(new_tau) - std::log(old_tau));
            diff_surv = -(new_tau - old_tau) * w_i;

            log_prior_old = R::dlnorm(old_tau, mu_tau(q), std::sqrt(sigma2_tau(q)), 1);
            log_prior_new = R::dlnorm(new_tau, mu_tau(q), std::sqrt(sigma2_tau(q)), 1);
            diff_prior = log_prior_new - log_prior_old;

            hastings = std::log(new_tau) - std::log(old_tau);

            log_ratio = diff_event + diff_surv + diff_prior + hastings;
            if (std::log(R::runif(0,1)) < log_ratio){
                tau(i) = new_tau;
                accept_tau(i) += 1.0;
            }
        }

        for (q = 0; q < Q; ++q){
            const std::vector<int>& current_ids = ids_by_country[q];
            n_q = current_ids.size();
            if (n_q == 0) continue;

            old_mu = mu_tau(q);
            // Country-level mean log speed: normal random walk.
            new_mu = R::rnorm(old_mu, jump_mu_tau(q));

            sum_old = 0.0;
            sum_new = 0.0;
            for (int target_id : current_ids){
                log_tau = std::log(tau(target_id));
                sum_old += (log_tau - old_mu)*(log_tau - old_mu);
                sum_new += (log_tau - new_mu)*(log_tau - new_mu);
            }
            diff_like = -0.5/sigma2_tau(q) * (sum_new - sum_old);

            log_prior_old = R::dnorm(old_mu, 0.0, std::sqrt(sigma2_mu_tau),1);
            log_prior_new = R::dnorm(new_mu, 0.0, std::sqrt(sigma2_mu_tau),1);
            diff_prior = log_prior_new - log_prior_old;

            hastings = 0.0;

            log_ratio = diff_like + diff_prior + hastings;
            if (std::log(R::runif(0,1)) < log_ratio){
                mu_tau(q) = new_mu;
                accept_mu_tau(q) += 1.0;
            }
        }

        for (q = 0; q < Q; ++q){
            const std::vector<int>& current_ids = ids_by_country[q];
            n_q = (int)current_ids.size();
            if (n_q == 0) continue;

            sum_sq = 0.0;
            for (int target_id : current_ids){
                log_tau = std::log(tau(target_id));
                sum_sq += (log_tau - mu_tau(q))*(log_tau - mu_tau(q));
            }
            shape = a_tau + 0.5 * n_q;
            rate = b_tau + 0.5 * sum_sq;
            sigma2_tau(q) = 1.0 / R::rgamma(shape, 1.0/rate);
        }

        sum_mu2 = arma::dot(mu_tau, mu_tau);
        shape_mu = c_tau + 0.5 * Q;
        rate_mu = d_tau + 0.5 * sum_mu2;
        sigma2_mu_tau = 1.0 / R::rgamma(shape_mu, 1.0/rate_mu);

        for (q = 0; q < Q; ++q) {
            const std::vector<int>& target_segs = segs_by_country[q];
            if (target_segs.empty()) continue;

            for (p = 0; p < P; ++p) {
                old_alpha = alpha(p,q);
                new_alpha = R::rnorm(old_alpha, jump_alpha(p,q));

                diff_event = 0.0;
                diff_surv = 0.0;

                for (int idx_k : target_segs) {
                    const auto &s = segments[idx_k];

                    diff_event += s.covs(p) * (new_alpha - old_alpha);

                    lp_fixed = LP_common_cache(idx_k);
                    lp_no_p = lp_fixed - old_alpha * s.covs(p);

                    base = weighted_row_sum(s.res, s.country, s.from) * tau(s.id) * s.dt;
                    diff_surv -= base * (std::exp(new_alpha * s.covs(p) + lp_no_p) - std::exp(lp_fixed));
                }

                log_prior_new = R::dnorm(new_alpha, mu_alpha_p(p), std::sqrt(sigma2_alpha_p(p)), 1);
                log_prior_old = R::dnorm(old_alpha, mu_alpha_p(p), std::sqrt(sigma2_alpha_p(p)), 1);

                if (std::log(R::runif(0,1)) < (diff_event + diff_surv + log_prior_new - log_prior_old)) {
                    alpha(p,q) = new_alpha;
                    accept_alpha(p,q) += 1.0;

                    for (int idx_upd : target_segs) {
                        LP_common_cache(idx_upd) += (new_alpha - old_alpha) * segments[idx_upd].covs(p);
                    }

                }
            }
        }

        for (p=0;p<P;++p){
            sum_mu = 0.0;
            for (q=0;q<Q;++q){
                sum_mu += alpha(p,q);
            }
            prec   = (double)Q / sigma2_alpha_p(p) + 1.0 / sigma2_alpha;
            post_var  = 1.0 / prec;
            post_mean = post_var * ( sum_mu / sigma2_alpha_p(p) /* + 0/sigma2_alpha */ );
            mu_alpha_p(p) = R::rnorm(post_mean, std::sqrt(post_var));
        }

        for (p=0;p<P;++p){
            ss = 0.0;
            for (q=0;q<Q;++q){
                diff_scalar_alpha = alpha(p,q) - mu_alpha_p(p);
                ss += diff_scalar_alpha * diff_scalar_alpha;
            }
            // Covariate-level between-country variance: inverse-gamma Gibbs step.
            shape = a_alpha + 0.5 * (double)Q;
            rate  = b_alpha + 0.5 * ss;
            sigma2_alpha_p(p) = 1.0 / R::rgamma(shape, 1.0/rate);
        }

        for (c = 0; c < 2; c++ ){
            for (q = 0; q < Q; q++){
                const std::vector<int>& target_segs = segs_by_res_country[c][q];
                if (target_segs.empty()) continue;

                for (r = 0; r < 3; r++){
                    old_beta = beta_key(c,r,q);
                    // Response/country-specific source, destination, or interaction effect.
                    new_beta = R::rnorm(old_beta, jump_beta(c,r,q));
                    delta_beta = new_beta - old_beta;

                    diff_event = 0.0;
                    diff_surv = 0.0;

                    if (r == 0) {

                        for (int idx_k : target_segs) {
                            const auto &s = segments[idx_k];
                            this_key = s.key1;

                            diff_event += this_key * delta_beta;

                            lp_fixed = LP_common_cache(idx_k);
                            base = weighted_row_sum(c, q, s.from) * tau(s.id) * s.dt;
                            diff_surv -= base * (std::exp(lp_fixed + delta_beta * this_key) - std::exp(lp_fixed));
                        }
                    } else {

                        arma::vec old_rows(E, arma::fill::zeros);
                        arma::vec new_rows(E, arma::fill::zeros);

                        for (mm = 0; mm < E; ++mm) {
                            old_rows(mm) = weighted_row_sum(c, q, mm);
                            if (r == 1) {
                                new_rows(mm) = compute_weighted_row_sum(c, q, mm, new_beta, beta_key(c, 2, q));
                            } else {
                                new_rows(mm) = compute_weighted_row_sum(c, q, mm, beta_key(c, 1, q), new_beta);
                            }
                        }

                        for (int idx_k : target_segs) {
                            const auto &s = segments[idx_k];
                            this_key = (r == 1 ? s.key2 : s.key3);

                            diff_event += this_key * delta_beta;

                            base = tau(s.id) * std::exp(LP_common_cache(idx_k)) * s.dt;
                            diff_surv -= base * (new_rows(s.from) - old_rows(s.from));
                        }
                    }

                    log_prior_new = R::dnorm(new_beta, mu_beta_cr(c,r), std::sqrt(sigma2_beta_cr(c,r)), 1);
                    log_prior_old = R::dnorm(old_beta, mu_beta_cr(c,r), std::sqrt(sigma2_beta_cr(c,r)), 1);

                    if (std::log(R::runif(0,1)) < (diff_event + diff_surv + log_prior_new - log_prior_old)) {
                        beta_key(c,r,q) = new_beta;
                        accept_beta_key(c,r,q) += 1;

                        if (r == 0) {
                            for (int idx_upd : target_segs) {
                                LP_common_cache(idx_upd) += delta_beta * segments[idx_upd].key1;
                            }
                        } else {

                            recompute_weighted_row_sums_one_cq(c, q);
                        }
                    }
                }
            }
        }

        for ( c=0;c<2;++c){
            for ( r=0;r<3;++r){
                sum_b = 0.0;
                for ( q=0;q<Q;++q){
                    sum_b += beta_key(c,r,q);
                }
                prec  = (double)Q / sigma2_beta_cr(c,r) + 1.0 / sigma2_mu_beta;
                post_var  = 1.0 / prec;
                post_mean = post_var * ( sum_b / sigma2_beta_cr(c,r) /* + 0/sigma2_mu_beta */ );
                mu_beta_cr(c,r) = R::rnorm(post_mean, std::sqrt(post_var));
            }
        }

        for ( c=0;c<2;++c){
            for ( r=0;r<3;++r){
                ss = 0.0;
                for ( q=0;q<Q;++q){
                    diff_beta = beta_key(c,r,q) - mu_beta_cr(c,r);
                    ss += diff_beta*diff_beta;
                }
                // Effect-level between-country variance: inverse-gamma Gibbs step.
                shape = a_beta + 0.5 * (double)Q;
                rate  = b_beta + 0.5 * ss;
                sigma2_beta_cr(c,r) = 1.0 / R::rgamma(shape, 1.0/rate);
            }
        }

        // Retain draws at the same iteration indices as the original IG sampler.
        if (iter >= nburn && (iter - nburn) % nthin == 0){

            d = draw++;

            arma::mat kappa0(E,E), kappa1(E,E);
            for (m = 0; m < E; ++m){
                for (l = 0; l < E; ++l){
                    kappa0(m,l) = kappa(0,m,l);
                    kappa1(m,l) = kappa(1,m,l);
                }
            }
            kappa0_save.slice(d) = kappa0;
            kappa1_save.slice(d) = kappa1;

            tau_save.col(d) = tau;
            mu_tau_save.col(d) = mu_tau;
            sigma2_tau_save.col(d) = sigma2_tau;
            sigma2_mu_tau_save(d) = sigma2_mu_tau;

            alpha_save.slice(d) = alpha;
            mu_alpha_save.col(d) = mu_alpha_p;
            sigma2_alpha_p_save.col(d) = sigma2_alpha_p;

            arma::mat beta_key0(3,Q), beta_key1(3,Q);
            for (r = 0; r < 3; ++r) {
                for (q = 0; q < Q; ++q){
                    beta_key0(r,q) = beta_key(0,r,q);
                    beta_key1(r,q) = beta_key(1,r,q);
                }
            }
            beta_key0_save.slice(d) = beta_key0;
            beta_key1_save.slice(d) = beta_key1;

            mu_beta_cr_save.slice(d)      = mu_beta_cr;
            sigma2_beta_cr_save.slice(d)  = sigma2_beta_cr;
        }

        if ((iter+1) % nprint == 0) {

            t_now = std::chrono::steady_clock::now();
            elapsed = std::chrono::duration_cast<std::chrono::seconds>(t_now-t_start).count();

            frac = double(iter+1) / double(niter);
            eta = (elapsed / frac) - elapsed;

            Rcpp::Rcout
                << "Iter " << (iter + 1) << "/" << niter
                << "    elapsed" << std::fixed << std::setprecision(0) << elapsed << "s"
                << "    ETA : " << std::fixed << std::setprecision(0) << eta << "s"

                << "    acc_kappa : " << std::fixed << std::setprecision(2) << (arma::accu(accept_kappa) / (2.0 * n_trans * (iter+1)))
                << "    acc_tau : " << std::fixed << std::setprecision(2) << (arma::mean(accept_tau) / (iter+1) )
                << "    acc_mu_tau : " << std::fixed << std::setprecision(2) << (arma::mean(accept_mu_tau) / (iter+1))
                << "    acc_alpha : " << std::fixed << std::setprecision(2) << (arma::mean(arma::vectorise(accept_alpha)) / (iter+1))
                << "    acc_beta : " << std::fixed << std::setprecision(2) << arma::accu(accept_beta_key) / (2*3*Q*(iter+1))
                << "\n";
        }

        if ((iter+1)%500 == 0) Rcpp::checkUserInterrupt();
    }

    // Keep draw and acceptance-count names compatible with downstream scripts.
    List out;
    out["kappa0_save"] = kappa0_save;
    out["kappa1_save"] = kappa1_save;
    out["accept_kappa"] = accept_kappa;
    out["tau_save"] = tau_save;
    out["mu_tau_save"] = mu_tau_save;
    out["sigma2_tau_save"] = sigma2_tau_save;
    out["sigma2_mu_tau_save"] = sigma2_mu_tau_save;
    out["accept_tau"] = accept_tau;
    out["accept_mu_tau"] = accept_mu_tau;
    out["alpha_save"] = alpha_save;
    out["mu_alpha_save"] = mu_alpha_save;
    out["sigma2_alpha_p_save"] = sigma2_alpha_p_save;
    out["accept_alpha"] = accept_alpha;
    out["beta_key0_save"] = beta_key0_save;
    out["beta_key1_save"] = beta_key1_save;
    out["mu_beta_cr_save"] = mu_beta_cr_save;
    out["sigma2_beta_cr_save"] = sigma2_beta_cr_save;
    out["state_key_indicator"] = state_key_indicator;
    out["n_key1_col_mismatch"] = (double)n_key1_col_mismatch;
    out["n_key2_col_mismatch"] = (double)n_key2_col_mismatch;
    out["accept_beta_key"] = accept_beta_key;
    return out;
}
