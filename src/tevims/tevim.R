#!/usr/bin/env Rscript
source("tevims/R/algorithms.R")
require(tidyr)
require(tidyverse)
require(ggplot2)
require(glue)
require(SuperLearner)

k_folds <- 4

args = commandArgs(trailingOnly=TRUE)

if (length(args)!=4) {
  stop("Requires three arguments.n", call.=FALSE)
} else {
  infile <- args[1]
  outfile <- args[2]
  combos <- args[3]
  onesided <- as.logical(args[4])  # valid values: T, F, TRUE, FALSE
}

# Parse subgroup masks from combos, e.g. "1,2+3,4" -> [[1, 2], [3, 4]]
# Combos are variables to exclude as TEVIM tests difference in CATE(X) and CATE(X_-s)
combos <- strsplit(combos, "\\+")[[1]]
combos <- lapply(combos, function(x) as.integer(strsplit(x, ",")[[1]]))  # input is already 1-indexed

df <- read.csv(infile)
print(df[1:10,])

cols <- colnames(df)
num_p <- ncol(df)-2  # number of covariates after subtracting the treatment and outcome

covs <- cols[1:num_p]
print(covs)

x <- df[covs]
a <- df[[cols[num_p+1]]]
y <- df[[cols[num_p+2]]]
n <- nrow(x)

# set.seed(1234567)

folds <- sample(rep(1:k_folds, length.out = n), size = n)

# Column names from combos
covariate_groups <- c()
for (com in combos) {
  name <- paste0(covs[com], collapse=" ")
  covariate_groups[[name]] <- covs[com]
}

# Fitting functions
propensity_score_constant <- mean(a)


fn_constant <- function(y_train, ...) {
  list(pred = rep_len(propensity_score_constant, length(y_train)))
}


ranger_learners <- create.Learner("SL.ranger",
  params = list(num.trees = 400, max.depth = 4),
  name_prefix = "RANGER"
)

# let's make a version that we can use for low dimensional training
small_ranger_learners <- create.Learner("SL.ranger",
  params = list(num.trees = 400, max.depth = 4),
  name_prefix = "RANGER"
)

gam_learners <- create.Learner("SL.gam",
  tune = list(deg.gam = c(2, 3, 4)),
  name_prefix = "GAM"
)

xgb_learners <- create.Learner("SL.xgboost",
  params = list(minobspernode = 10, ntrees = 400, shrinkage = 0.01),
  tune = list(max_depth = c(2, 3)),
  name_prefix = "XGB"
)

sl_library <- c(
  # "SL.glmnet", 
  # "SL.glm"
  # ranger_learners$names,
  # gam_learners$names,
  xgb_learners$names
)

small_sl_library <- c(
  # "SL.glmnet",
  # "SL.glm"
  small_ranger_learners$names
  # gam_learners$names
  # xgb_learners$names
)

fit_sl <- function(y_train, x_train, x_new) {
  sl_lib <- sl_library

  low_dimensional <- ncol(x_train) < 4
  if (low_dimensional) {
    sl_lib <- small_sl_library
  }

  sl <- SuperLearner(
    Y = y_train,
    X = x_train,
    newX = x_new,
    family = gaussian(),
    cvControl = list(V = 3),
    SL.library = sl_lib
  )

  winning_algorithm <- which.min(sl$cvRisk)
  list(pred = sl$library.predict[, winning_algorithm])
}


fitfunc_outcome <- function(y_train, x_train, x_new) {
  t_learner(y_train, x_train, x_new, "A", fit_sl)
}
fitfunc_cate <- fit_sl
fit_func_ps <- fn_constant

# res_0 <- algorithm_0(
#     y, a, x,
#     fitfunc_outcome,
#     fit_func_ps,
#     fitfunc_cate,
#     covariate_groups
# ) # TODO: pass onesided

res_2 <- algorithm_2(
    y, a, x, folds,
    fitfunc_outcome,
    fit_func_ps,
    fitfunc_cate,
    covariate_groups
) # TODO: pass onesided

estimates <- do.call(
    rbind,
    list(
      # res_0$algorithm_0T$estiamtes,
      # res_0$algorithm_0D$estiamtes,
      res_2$estiamtes
    )
)

# algorithm_suffixes <- c("_0T", "_0D")
algorithm_suffixes <- c("_CF")
rownames(estimates) <- purrr::map2_chr(
    rownames(estimates),
    rep(algorithm_suffixes, each = 6),
    paste0
)

results <- list(
    estimates = estimates
)
res <- results$estimates %>%
  c() %>%
  unlist() %>%
  matrix(ncol = ncol(results$estimates)) %>%
  t()
colnames(res) <- rownames(results$estimates)
res <- as_tibble(res) %>%
  mutate(estimand = colnames(results$estimates)) %>%
  pivot_longer(
    cols = c(starts_with("tevim"), starts_with("std_err"), starts_with("pval")),
    names_to = c(".value", "algorithm"),
    names_pattern = "(.{1,20})_(.{2})$"
  ) %>%
  rename(tevim_u = tevim, std_err_u = std_err, pval_u = pval) %>%
  pivot_longer(
    cols = c(starts_with("tevim"), starts_with("std_err"), starts_with("pval")),
    names_to = c(".value", "scale"),
    names_pattern = "(.{1,20})_(.{1})$"
  )

print(outfile)
write.csv(res, outfile, row.names=FALSE)
