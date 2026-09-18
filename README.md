# HMSM

Code for the CD Tally and Lamp Return analyses. Run all commands from the directory containing this README.

## Repository structure

```text
.
├── data/              # Processed input data
├── config/            # Analysis settings and MCMC proposal scales
├── mcmc/              # MCMC sampling and transition graph construction
├── embedding/         # Embedding training, generation, and validation
├── distance/          # Distance calculation and statistical analysis
├── ame/               # AME analysis, diagnostics, and figures
├── requirements.txt   # Python dependencies
└── environment_R.txt  # R environment
```

## Workflow

```text
Processed data → MCMC → Transition graphs
                         ├─ Posterior means → Train embedding model
                         │                    └─ Embed graph draws → Distances → NMDS / PERMANOVA / PERMDISP
                         └─ Posterior means → AME → Effect tables and figures
```

## Setup

Use Python 3.12+, R, and a C++11 compiler (matching Rtools on Windows).

```sh
python -m pip install -r requirements.txt
```

In R:

```r
install.packages(c("Rcpp", "RcppArmadillo", "posterior", "jsonlite",
                   "vegan", "permute", "ggplot2", "amen", "dplyr",
                   "ggtext", "patchwork", "HDInterval"))
```

Python package versions are pinned in `requirements.txt`. R and R package versions are listed in `environment_R.txt`.

## Inputs and settings

The workflow starts from the processed files in `data/`:

| Dataset | Data | State labels |
| --- | --- | --- |
| `cdtally` | `data/cdtally/combined_cd_tally.RData` | `data/cdtally/state_name_list_cdtally.csv` |
| `lamp_return` | `data/lamp_return/combined_lamp_return.RData` | `data/lamp_return/state_name_list_lamp.csv` |

MCMC proposal files are in `config/mcmc/tuning/<dataset>/`. Preserve the supplied state, country, and covariate order; Lamp Return data already exclude state 105.

Edit MCMC, graph, and AME settings at the top of their R scripts. Embedding and distance settings are in `config/embedding/` and `config/distance/`: `common.json`, then `<dataset>.json`, then optional `--config file.json` overrides. Paths are relative to this directory.

## 1. MCMC and transition graphs

```sh
Rscript mcmc/run_cdtally.R
Rscript mcmc/run_lamp_return.R
Rscript mcmc/build_graphs_cdtally.R
Rscript mcmc/build_graphs_lamp_return.R
```

Defaults: four chains, 300,000 iterations, 100,000 burn-in, and thinning by 20. Graph construction uses chain 1, all 10,000 retained draws, and threshold 3.

For MCMC diagnostics and trace plots, run in R after sampling:

```r
source("mcmc/diagnostics.R")
for (dataset in c("cdtally", "lamp_return")) {
  run_dir <- file.path("outputs", dataset, "mcmc")
  diagnose_mcmc(run_dir)
  export_traceplots(run_dir)
}
```

For parameter summaries and figures from the saved first chain:

```sh
Rscript mcmc/plot_parameters.R cdtally
Rscript mcmc/plot_parameters.R lamp_return
```

Each command reads `outputs/<dataset>/mcmc/chain01_seed1001.RData` and writes four PNG figures (speed, covariate effects, their hyperparameters, and key-action effects), six CSV summaries, and three LaTeX tables to `outputs/<dataset>/mcmc/summary/`. Intervals are 95% HPD intervals. The κ tables include off-diagonal transitions observed in each response group, matching the original analysis. LaTeX tables require `booktabs`, `longtable`, `graphicx`, and `xcolor`.

To use another saved chain or output directory:

```r
source("mcmc/plot_parameters.R")
plot_parameters("cdtally", chain_file = "path/to/chain.RData", out_dir = "path/to/summary")
```

## 2. Embedding and distance analysis

Run the following for each dataset; replace `cdtally` with `lamp_return` for Lamp Return.

```sh
python embedding/train.py --dataset cdtally
python embedding/embed.py --dataset cdtally
python embedding/validate.py --dataset cdtally
python distance/compute_wasserstein.py --dataset cdtally
Rscript distance/descriptive.R --dataset cdtally
Rscript distance/nmds.R --dataset cdtally
Rscript distance/permanova_permdisp.R --dataset cdtally
```

Training uses the supplied settings: 210 epochs for CD Tally and 145 for Lamp Return, on CPU by default. `validate.py` exports reconstruction diagnostics.

Optional tuning, before training:

```sh
python embedding/tune.py --dataset cdtally --stage grid
python embedding/tune.py --dataset cdtally --stage epochs
```

The CD Tally grid search uses 1,000 epochs for d = 8, 600 for d = 16, and 800 for d = 32. Tuning results do not update the configuration automatically. Apply selected settings before the next stage or training.

## 3. AME analysis

AME reads the posterior-mean graphs from step 1 in `outputs/<dataset>/graphs/posterior_mean/`. Both analysis scripts default to `refit <- TRUE` to fit and save the seed-123 model.

```sh
Rscript ame/ame_cdtally.R
Rscript ame/ame_lamp_return.R
Rscript ame/plot_ame.R
```

Set `refit <- FALSE` to reuse saved fits with the same inputs. For multi-chain diagnostics, set `refit_extra <- TRUE` in `ame/diagnose_ame.R` to generate the three additional chains, then run:

```sh
Rscript ame/diagnose_ame.R
```

Use `refit_extra <- FALSE` when those chains already exist for the same inputs and sampling settings.

## Outputs

All generated results go to `outputs/` (excluded from Git).

| Directory | Contents |
| --- | --- |
| `outputs/<dataset>/mcmc/` | Chain files and diagnostics |
| `outputs/<dataset>/mcmc/summary/` | Parameter figures, CSV summaries, and LaTeX tables |
| `outputs/<dataset>/graphs/` | `draws/` and `posterior_mean/` graphs |
| `outputs/<dataset>/embedding/` | Trained model, draw embeddings, validation, and optional tuning |
| `outputs/<dataset>/distance/` | Draw distances, posterior mean and interval CSVs |
| `outputs/<dataset>/distance/analysis/` | Distance summaries, NMDS figures, and test results |
| `outputs/<dataset>/ame/` | AME fits and effect tables |
| `outputs/figures_tables/ame/` | AME figures and plotted data |
| `outputs/ame_diagnostics/` | AME diagnostic tables and traces |

Graph generation and Python stages require new or empty output directories. MCMC and R analyses can overwrite existing results. For a new run, use separate output paths and update downstream input paths accordingly.
