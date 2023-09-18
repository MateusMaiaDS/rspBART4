all_bart <- function(cv_element,
                     nIknots_,
                     ntree_,
                     seed_){

  # To replicate the results
  set.seed(seed_)
  train <- cv_element$train
  test <- cv_element$test

  # Getting the training elements
  x_train <- train %>% dplyr::select(dplyr::starts_with("x"))
  x_test <- test %>% dplyr::select(dplyr::starts_with("x"))
  y_train <- train %>% dplyr::pull("y")

  # Running the model
  spBART <- rspBART(x_train = x_train,
                    x_test = x_test,y_train = y_train,
                    n_mcmc = 2500,node_min_size = 5,
                    n_burn = 0,nIknots = nIknots_,n_tree = ntree_,
                    use_bs = TRUE,
                    dif_order = 0,motrbart_bool = FALSE)

  bartmod <- dbarts::bart(x.train = x_train,y.train = y_train,x.test = x_test)
  softbartmod <- SoftBart::softbart(X = x_train,Y = y_train,X_test =  x_test)

  motr_bart_mod <- motr_bart(x = x_train,y = y_train)
  motrbart_pred <- predict_motr_bart(object = motr_bart_mod,newdata = x_test,type = "all")


  return(list(spBART = spBART,
              bartmod = bartmod,
              softbartmod = softbartmod,
              motrbartmod = motr_bart_mod,
              motrbart_pred  = motrbart_pred,
              cv = cv_element))

}


# Summarising all the metrics and results
wrapping_comparison <- function(result_){

  # Initialising df
  comparison_metrics <- data.frame(metric = NULL, value = NULL, model = NULL,fold = NULL)

  for(j in 1:length(result_)){


    n_burn_ <- 500
    n_mcmc_ <- result_[[j]]$spBART$mcmc$n_mcmc

    # Calculating metrics for splinesBART
    comparison_metrics <- rbind(comparison_metrics,data.frame(metric = "rmse_train",
                                                              value = rmse(x = colMeans(result_[[j]]$spBART$y_train_hat[(n_burn_+1):n_mcmc_,,drop = FALSE]),
                                                                           y = cv_[[j]]$train$y),
                                                              model = "spBART",fold = j))

    comparison_metrics <- rbind(comparison_metrics,data.frame(metric = "rmse_test",
                                                              value = rmse(x = colMeans(result_[[j]]$spBART$y_test_hat[(n_burn_+1):n_mcmc_,,drop = FALSE]),
                                                                           y = cv_[[j]]$test$y),
                                                              model = "spBART",fold = j))

    # Calculating the CRPS as well
    comparison_metrics <- rbind(comparison_metrics,data.frame(metric = "crps_train",
                                                              value = crps(y = cv_[[j]]$train$y ,
                                                                           means = colMeans(result_[[j]]$spBART$y_train_hat[(n_burn_+1):n_mcmc_,,drop = FALSE]),
                                                                           sds = rep(mean(result_[[j]]$spBART$all_tau[(n_burn_+1):n_mcmc_])^(-1/2), length(cv_[[i]]$train$y)))$CRPS,
                                                              model = "spBART",fold = j))

    comparison_metrics <- rbind(comparison_metrics,data.frame(metric = "crps_test",
                                                              value = crps(y = cv_[[j]]$test$y ,
                                                                           means = colMeans(result_[[j]]$spBART$y_test_hat[(n_burn_+1):n_mcmc_,,drop = FALSE]),
                                                                           sds = rep(mean(result_[[j]]$spBART$all_tau[(n_burn_+1):n_mcmc_])^(-1/2), length(cv_[[i]]$test$y)))$CRPS,
                                                              model = "spBART",fold = j))

    # ============================
    # Calculating metrics for BART
    # ============================

    comparison_metrics <- rbind(comparison_metrics,data.frame(metric = "rmse_train",
                                                              value = rmse(x = result_[[j]]$bartmod$yhat.train.mean,
                                                                           y = cv_[[j]]$train$y),
                                                              model = "BART",fold = j))

    comparison_metrics <- rbind(comparison_metrics,data.frame(metric = "rmse_test",
                                                              value = rmse(x = result_[[j]]$bartmod$yhat.test.mean,
                                                                           y = cv_[[j]]$test$y),
                                                              model = "BART",fold = j))

    # Calculating the CRPS as well
    comparison_metrics <- rbind(comparison_metrics,data.frame(metric = "crps_train",
                                                              value = crps(y = cv_[[j]]$train$y ,
                                                                           means = result_[[j]]$bartmod$yhat.train.mean,
                                                                           sds = rep(mean(result_[[j]]$bartmod$sigma), length(cv_[[j]]$train$y) ))$CRPS,
                                                              model = "BART",fold = j))

    comparison_metrics <- rbind(comparison_metrics,data.frame(metric = "crps_test",
                                                              value = crps(y = cv_[[j]]$test$y ,
                                                                           means = result_[[j]]$bartmod$yhat.test.mean,
                                                                           sds = rep(mean(result_[[j]]$bartmod$sigma), length(cv_[[j]]$test$y) ))$CRPS,
                                                              model = "BART",fold = j))


    # ============================
    # Calculating metrics for softBART
    # ============================

    comparison_metrics <- rbind(comparison_metrics,data.frame(metric = "rmse_train",
                                                              value = rmse(x = result_[[j]]$softbartmod$y_hat_train_mean,
                                                                           y = cv_[[j]]$train$y),
                                                              model = "softBART",fold = j))

    comparison_metrics <- rbind(comparison_metrics,data.frame(metric = "rmse_test",
                                                              value = rmse(x = result_[[j]]$softbartmod$y_hat_test_mean,
                                                                           y = cv_[[j]]$test$y),
                                                              model = "softBART",fold = j))

    # Calculating the CRPS as well
    comparison_metrics <- rbind(comparison_metrics,data.frame(metric = "crps_train",
                                                              value = crps(y = cv_[[j]]$train$y ,
                                                                           means = result_[[j]]$softbartmod$y_hat_train_mean,
                                                                           sds = rep(mean(result_[[j]]$softbartmod$sigma), length(cv_[[j]]$train$y) ))$CRPS,
                                                              model = "softBART",fold = j))

    comparison_metrics <- rbind(comparison_metrics,data.frame(metric = "crps_test",
                                                              value = crps(y = cv_[[j]]$test$y ,
                                                                           means = result_[[j]]$softbart$y_hat_test_mean,
                                                                           sds = rep(mean(result_[[j]]$softbartmod$sigma), length(cv_[[j]]$test$y) ))$CRPS,
                                                              model = "softBART",fold = j))
    # ============================
    # Calculating metrics for MOTRBART
    # ============================

    comparison_metrics <- rbind(comparison_metrics,data.frame(metric = "rmse_train",
                                                              value = rmse(x = colMeans(result_[[j]]$motrbartmod$y_hat),
                                                                           y = cv_[[j]]$train$y),
                                                              model = "motrBART",fold = j))

    comparison_metrics <- rbind(comparison_metrics,data.frame(metric = "rmse_test",
                                                              value = rmse(x = colMeans(result_[[j]]$motrbart_pred),
                                                                           y = cv_[[j]]$test$y),
                                                              model = "motrBART",fold = j))

    # Calculating the CRPS as well
    comparison_metrics <- rbind(comparison_metrics,data.frame(metric = "crps_train",
                                                              value = crps(y = cv_[[j]]$train$y ,
                                                                           means = colMeans(result_[[j]]$motrbartmod$y_hat),
                                                                           sds = rep(mean(sqrt(result_[[j]]$motrbartmod$sigma2)), length(cv_[[j]]$train$y) ))$CRPS,
                                                              model = "motrBART",fold = j))

    comparison_metrics <- rbind(comparison_metrics,data.frame(metric = "crps_test",
                                                              value = crps(y = cv_[[j]]$test$y ,
                                                                           means = colMeans(result_[[j]]$motrbart_pred),
                                                                           sds = rep(mean(sqrt(result_[[j]]$motrbartmod$sigma2)), length(cv_[[j]]$test$y) ))$CRPS,
                                                              model = "motrBART",fold = j))



  }

  return(comparison_metrics)

}


# Getting a model to evaluate variable importance
var_importance_counter <- function(result_,rep_){

  # Getting a counter for times that a variable is used in within a tree
  p_counter <- numeric(ncol(result_[[rep_]]$cv$train)-1)
  spBART <- result_[[rep_]]$spBART

  for(i in 501:spBART$mcmc$n_mcmc){

    for(t in 1:spBART$prior$n_tree){
      curr_tree <- spBART$mcmc$all_trees[[i]][[t]]
      terminals <- get_terminals(curr_tree)
      for(ell in 1:length(terminals)){
        p_counter[unique(curr_tree[[terminals[ell]]]$ancestors)] <- p_counter[unique(curr_tree[[terminals[ell]]]$ancestors)] + 1
      }
    }
  }

  return(round(p_counter/sum(p_counter),digits = 5))

}

