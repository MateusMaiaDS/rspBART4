# This file is to compare first trial of predictive performance of spBART and BART
# rm(list=ls())
library(dbarts)
library(mlbench)
library(purrr)
library(MOTRbart)
library(doParallel)
source("R/sim_functions.R")
source("R/main_function.R")
set.seed(42)

n_ <- 250
sd_ <- 1
n_rep_ <- 5
cv_ <- vector("list", n_rep_)

# Generating CV_ object
for( i in 1:n_rep_){

    train <- mlbench.friedman1.nointeraction.noise(n = n_,sd = sd_) %>% as.data.frame()
    test <- mlbench.friedman1.nointeraction.noise(n = n_,sd = sd_) %>% as.data.frame()

    # train <- mlbench.friedman1(n = n_,sd = sd_) %>% as.data.frame() %>% .[,c(1:5,11)]
    # test <- mlbench.friedman1(n = n_,sd = sd_) %>% as.data.frame() %>% .[,c(1:5,11)]

    # train <- mlbench.friedman1(n = n_,sd = sd_) %>% as.data.frame()
    # test <- mlbench.friedman1(n = n_,sd = sd_) %>% as.data.frame()

    # train <- mlbench.d1.break(n = n_,sd = sd_)  |> as.data.frame()
    # test <- mlbench.d1.break(n = n_,sd = sd_) |> as.data.frame()

    cv_[[i]]$train <- train
    cv_[[i]]$test <- test
}


# Setting up the parallel simulation
number_cores <- n_rep_
cl <- parallel::makeCluster(number_cores)
doParallel::registerDoParallel(cl)


# Testing the simple n_tree
result <- foreach(i = 1:n_rep_, .packages = c("dbarts","SoftBart","MOTRbart","dplyr")) %dopar%{

  source("/localusers/researchers/mmarques/spline_bart_lab/rspBART4/R/sim_functions.R")
  source("/localusers/researchers/mmarques/spline_bart_lab/rspBART4/R/main_function.R")
  source("/localusers/researchers/mmarques/spline_bart_lab/rspBART4/R/cv_functions.R")
  aux <- all_bart(cv_element = cv_[[i]],nIknots_ = 2)
  aux
}


#
stopCluster(cl)

# Summarising all the metrics and results
result_ <- result[[1]]

wrapping_comparison <- function(result_){



  comparison_metrics <- data.frame(metric = NULL, value = NULL, model = NULL)

  for(j in 1:length(result_)){


      n_burn_ <- 500
      n_mcmc_ <- result_[[j]]$spBART$mcmc$n_mcmc

      # Calculating metrics for splinesBART
      comparison_metrics <- rbind(comparison_metrics,data.frame(metric = "rmse_train",
                 value = rmse(x = colMeans(result_[[j]]$spBART$y_train_hat[(n_burn_+1):n_mcmc_,,drop = FALSE]),
                              y = cv_[[j]]$train$y),
                 model = "spBART"))

      comparison_metrics <- rbind(comparison_metrics,data.frame(metric = "rmse_test",
                                                                value = rmse(x = colMeans(result_[[j]]$spBART$y_test_hat[(n_burn_+1):n_mcmc_,,drop = FALSE]),
                                                                             y = cv_[[j]]$test$y),
                                                                model = "spBART"))



  }

}
