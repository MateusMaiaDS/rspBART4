# This file is to compare first trial of predictive performance of spBART and BART
rm(list=ls())
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
n_rep_ <- 10
nIknots_ <- 4
ntree_ <- 1
use_bs_ <- FALSE
seed_ <- 42

print(paste0("N: ",n_," SD: ", sd_, " nIknots: ", nIknots_, " Ntree: ",ntree_, " Seed: ",seed_))
cv_ <- vector("list", n_rep_)

# Generating CV_ object
for( i in 1:n_rep_){

    train <- mlbench.friedman1.nointeraction(n = n_,sd = sd_) %>% as.data.frame()
    test <- mlbench.friedman1.nointeraction(n = n_,sd = sd_) %>% as.data.frame()

    # train <- mlbench.friedman1.nointeraction.noise(n = n_,sd = sd_) %>% as.data.frame()
    # test <- mlbench.friedman1.nointeraction.noise(n = n_,sd = sd_) %>% as.data.frame()

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
  if(ntree_<=50) {
    aux <- all_bart(cv_element = cv_[[i]],
                    nIknots_ = nIknots_,ntree_ = ntree_,seed_ = seed_,
                    use_bs_ = use_bs_)
  } else {
    aux <- all_bart_lite(cv_element = cv_[[i]],
                         nIknots_ = nIknots_,ntree_ = ntree_,seed_ = seed_,
                         use_bs_ = use_bs_,
                         j = i)
  }

  aux
}


#
stopCluster(cl)



# Saving all
saveRDS(object = result,file = paste0("/localusers/researchers/mmarques/spline_bart_lab/preliminar_results/friedman_n_",n_,
               "_sd_",sd_,"_nIknots_",nIknots_,"_ntree_",ntree_,"_bs_",use_bs_,".Rds"))
