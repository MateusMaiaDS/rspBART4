all_bart <- function(cv_element,
                     nIknots_,
                     ntree_,
                     use_bs_,
                     seed_,
                     motr_bart_){

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
                    use_bs = use_bs_,scale_bool = TRUE,plot_preview = FALSE,
                    dif_order = 0,motrbart_bool = motr_bart_)

  bartmod <- dbarts::bart(x.train = x_train,y.train = y_train,x.test = x_test)

  # Special case when I would have univariate regression
  if(ncol(x_train)>=2){
    softbartmod <- SoftBart::softbart(X = x_train,Y = y_train,X_test =  x_test)
    motr_bart_mod <- motr_bart(x = x_train,y = y_train,ntrees = ntree_)
    motrbart_pred <- predict_motr_bart(object = motr_bart_mod,newdata = x_test,type = "all")
  } else {
    x_train_new <- cbind(x_train,x_train)
    x_test_new <- cbind(x_test,x_test)
    colnames(x_test_new) <- colnames(x_train_new) <- paste0("x.",1:2)

    softbartmod <- SoftBart::softbart(X = x_train_new,Y = y_train,X_test =  x_test_new)
    motr_bart_mod <- motr_bart(x = x_train_new,y = y_train,ntrees = ntree_)
    # motr_bart_mod$y_hat
    if(ntree_>1){
      motrbart_pred <- predict_motr_bart(object = motr_bart_mod,newdata = x_test_new,type = "all")
    } else {
      motrbart_pred <- NULL
    }
  }



  return(list(spBART = spBART,
              bartmod = bartmod,
              softbartmod = softbartmod,
              motrbartmod = motr_bart_mod,
              motrbart_pred  = motrbart_pred,
              cv = cv_element))

}


all_bart_lite <- function(cv_element,
                     nIknots_,
                     ntree_,
                     seed_,
                     use_bs_,
                     j,
                     motr_bart_){


  # Doing a warming for the case whichI don't have
  if(ntree_<=50){
    stop("Use the all_bart() function instead.")
  }

  # To replicate the results
  set.seed(seed_)
  train <- cv_element$train
  test <- cv_element$test

  # Getting the training elements
  x_train <- train %>% dplyr::select(dplyr::starts_with("x"))
  x_test <- test %>% dplyr::select(dplyr::starts_with("x"))
  y_train <- train %>% dplyr::pull("y")
  y_test <- test %>% dplyr::pull("y")

  # Initialising df
  comparison_metrics <- data.frame(metric = NULL, value = NULL, model = NULL,fold = NULL)

  # Running the model
  spBART <- rspBART(x_train = x_train,
                    x_test = x_test,y_train = y_train,
                    n_mcmc = 2500,node_min_size = 5,
                    n_burn = 0,nIknots = nIknots_,n_tree = ntree_,
                    use_bs = use_bs_,
                    dif_order = 0,motrbart_bool = motr_bart_)


  n_burn_ <- 500
  n_mcmc_ <- spBART$mcmc$n_mcmc

  # Calculating metrics for splinesBART
  comparison_metrics <- rbind(comparison_metrics,data.frame(metric = "rmse_train",
                                                            value = rmse(x = colMeans(spBART$y_train_hat[(n_burn_+1):n_mcmc_,,drop = FALSE]),
                                                                         y = train$y),
                                                            model = "spBART",fold = j))

  comparison_metrics <- rbind(comparison_metrics,data.frame(metric = "rmse_test",
                                                            value = rmse(x = colMeans(spBART$y_test_hat[(n_burn_+1):n_mcmc_,,drop = FALSE]),
                                                                         y = test$y),
                                                            model = "spBART",fold = j))

  # Calculating the CRPS as well
  comparison_metrics <- rbind(comparison_metrics,data.frame(metric = "crps_train",
                                                            value = crps(y = train$y ,
                                                                         means = colMeans(spBART$y_train_hat[(n_burn_+1):n_mcmc_,,drop = FALSE]),
                                                                         sds = rep(mean(spBART$all_tau[(n_burn_+1):n_mcmc_])^(-1/2), length(train$y)))$CRPS,
                                                            model = "spBART",fold = j))

  comparison_metrics <- rbind(comparison_metrics,data.frame(metric = "crps_test",
                                                            value = crps(y = test$y ,
                                                                         means = colMeans(spBART$y_test_hat[(n_burn_+1):n_mcmc_,,drop = FALSE]),
                                                                         sds = rep(mean(spBART$all_tau[(n_burn_+1):n_mcmc_])^(-1/2), length(test$y)))$CRPS,
                                                            model = "spBART",fold = j))

  # Removing the model
  rm(spBART)

  # Initializing the modelling for the BART model.
  bartmod <- dbarts::bart(x.train = x_train,y.train = y_train,x.test = x_test)

  comparison_metrics <- rbind(comparison_metrics,data.frame(metric = "rmse_train",
                                                            value = rmse(x = bartmod$yhat.train.mean,
                                                                         y = train$y),
                                                            model = "BART",fold = j))

  comparison_metrics <- rbind(comparison_metrics,data.frame(metric = "rmse_test",
                                                            value = rmse(x = bartmod$yhat.test.mean,
                                                                         y = test$y),
                                                            model = "BART",fold = j))

  # Calculating the CRPS as well
  comparison_metrics <- rbind(comparison_metrics,data.frame(metric = "crps_train",
                                                            value = crps(y = train$y ,
                                                                         means = bartmod$yhat.train.mean,
                                                                         sds = rep(mean(bartmod$sigma), length(train$y) ))$CRPS,
                                                            model = "BART",fold = j))

  comparison_metrics <- rbind(comparison_metrics,data.frame(metric = "crps_test",
                                                            value = crps(y = test$y ,
                                                                         means = bartmod$yhat.test.mean,
                                                                         sds = rep(mean(bartmod$sigma), length(test$y) ))$CRPS,
                                                            model = "BART",fold = j))
  rm(bartmod)


  # Since SOFTBART and MOTR-BART dont do it properly for 1-d
  if(ncol(x_train)>1){
      # Doing for SoftBART
      softbartmod <- SoftBart::softbart(X = x_train,Y = y_train,X_test =  x_test)

      comparison_metrics <- rbind(comparison_metrics,data.frame(metric = "rmse_train",
                                                                value = rmse(x = softbartmod$y_hat_train_mean,
                                                                             y = train$y),
                                                                model = "softBART",fold = j))

      comparison_metrics <- rbind(comparison_metrics,data.frame(metric = "rmse_test",
                                                                value = rmse(x = softbartmod$y_hat_test_mean,
                                                                             y = test$y),
                                                                model = "softBART",fold = j))

      # Calculating the CRPS as well
      comparison_metrics <- rbind(comparison_metrics,data.frame(metric = "crps_train",
                                                                value = crps(y = train$y ,
                                                                             means = softbartmod$y_hat_train_mean,
                                                                             sds = rep(mean(softbartmod$sigma), length(train$y) ))$CRPS,
                                                                model = "softBART",fold = j))

      comparison_metrics <- rbind(comparison_metrics,data.frame(metric = "crps_test",
                                                                value = crps(y = test$y ,
                                                                             means = softbartmod$y_hat_test_mean,
                                                                             sds = rep(mean(softbartmod$sigma), length(test$y) ))$CRPS,
                                                                model = "softBART",fold = j))

      rm(softbartmod)

      # Doing for MOTR-BART
      motrbartmod <- motr_bart(x = x_train,y = y_train,ancestors = TRUE,ntrees = 1)
      motrbart_pred <- predict_motr_bart(object = motrbartmod,newdata = x_test,type = "all")

      comparison_metrics <- rbind(comparison_metrics,data.frame(metric = "rmse_train",
                                                                value = rmse(x = colMeans(motrbartmod$y_hat),
                                                                             y = train$y),
                                                                model = "motrBART",fold = j))

      comparison_metrics <- rbind(comparison_metrics,data.frame(metric = "rmse_test",
                                                                value = rmse(x = colMeans(motrbart_pred),
                                                                             y = test$y),
                                                                model = "motrBART",fold = j))

      # Calculating the CRPS as well
      comparison_metrics <- rbind(comparison_metrics,data.frame(metric = "crps_train",
                                                                value = crps(y = train$y ,
                                                                             means = colMeans(motrbartmod$y_hat),
                                                                             sds = rep(mean(sqrt(motrbartmod$sigma2)), length(train$y) ))$CRPS,
                                                                model = "motrBART",fold = j))

      comparison_metrics <- rbind(comparison_metrics,data.frame(metric = "crps_test",
                                                                value = crps(y = test$y ,
                                                                             means = colMeans(motrbart_pred),
                                                                             sds = rep(mean(sqrt(motrbartmod$sigma2)), length(test$y) ))$CRPS,
                                                                model = "motrBART",fold = j))

      rm(motrbartmod)

  } else {

    # Doing for SoftBART
    x_train_new <- cbind(x_train,x_train)
    x_test_new <- cbind(x_test,x_test)
    colnames(x_train_new) <- colnames(x_test_new) <- c("x.1","x.2")

    softbartmod <- SoftBart::softbart(X = x_train_new,Y = y_train,X_test =  x_test_new)

    comparison_metrics <- rbind(comparison_metrics,data.frame(metric = "rmse_train",
                                                              value = rmse(x = softbartmod$y_hat_train_mean,
                                                                           y = train$y),
                                                              model = "softBART",fold = j))

    comparison_metrics <- rbind(comparison_metrics,data.frame(metric = "rmse_test",
                                                              value = rmse(x = softbartmod$y_hat_test_mean,
                                                                           y = test$y),
                                                              model = "softBART",fold = j))

    # Calculating the CRPS as well
    comparison_metrics <- rbind(comparison_metrics,data.frame(metric = "crps_train",
                                                              value = crps(y = train$y ,
                                                                           means = softbartmod$y_hat_train_mean,
                                                                           sds = rep(mean(softbartmod$sigma), length(train$y) ))$CRPS,
                                                              model = "softBART",fold = j))

    comparison_metrics <- rbind(comparison_metrics,data.frame(metric = "crps_test",
                                                              value = crps(y = test$y ,
                                                                           means = softbartmod$y_hat_test_mean,
                                                                           sds = rep(mean(softbartmod$sigma), length(test$y) ))$CRPS,
                                                              model = "softBART",fold = j))

    rm(softbartmod)

    # Doing for MOTR-BART
    motrbartmod <- motr_bart(x = x_train_new,y = y_train,ancestors = TRUE,ntrees = 1)

    comparison_metrics <- rbind(comparison_metrics,data.frame(metric = "rmse_train",
                                                              value = rmse(x = colMeans(motrbartmod$y_hat),
                                                                           y = train$y),
                                                              model = "motrBART",fold = j))

    # Calculating the CRPS as well
    comparison_metrics <- rbind(comparison_metrics,data.frame(metric = "crps_train",
                                                              value = crps(y = train$y ,
                                                                           means = colMeans(motrbartmod$y_hat),
                                                                           sds = rep(mean(sqrt(motrbartmod$sigma2)), length(train$y) ))$CRPS,
                                                              model = "motrBART",fold = j))

    rm(motrbartmod)

  }

  return(comparison_metrics)

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
                                                                           y = result_[[j]]$cv$train$y),
                                                              model = "spBART",fold = j))

    comparison_metrics <- rbind(comparison_metrics,data.frame(metric = "rmse_test",
                                                              value = rmse(x = colMeans(result_[[j]]$spBART$y_test_hat[(n_burn_+1):n_mcmc_,,drop = FALSE]),
                                                                           y = result_[[j]]$cv$test$y),
                                                              model = "spBART",fold = j))

    # Calculating the CRPS as well
    comparison_metrics <- rbind(comparison_metrics,data.frame(metric = "crps_train",
                                                              value = crps(y = result_[[j]]$cv$train$y ,
                                                                           means = colMeans(result_[[j]]$spBART$y_train_hat[(n_burn_+1):n_mcmc_,,drop = FALSE]),
                                                                           sds = rep(mean(result_[[j]]$spBART$all_tau[(n_burn_+1):n_mcmc_])^(-1/2), length(result_[[j]]$cv$train$y)))$CRPS,
                                                              model = "spBART",fold = j))

    comparison_metrics <- rbind(comparison_metrics,data.frame(metric = "crps_test",
                                                              value = crps(y = result_[[j]]$cv$test$y ,
                                                                           means = colMeans(result_[[j]]$spBART$y_test_hat[(n_burn_+1):n_mcmc_,,drop = FALSE]),
                                                                           sds = rep(mean(result_[[j]]$spBART$all_tau[(n_burn_+1):n_mcmc_])^(-1/2), length(result_[[j]]$cv$test$y)))$CRPS,
                                                              model = "spBART",fold = j))

    # ============================
    # Calculating metrics for BART
    # ============================

    comparison_metrics <- rbind(comparison_metrics,data.frame(metric = "rmse_train",
                                                              value = rmse(x = result_[[j]]$bartmod$yhat.train.mean,
                                                                           y = result_[[j]]$cv$train$y),
                                                              model = "BART",fold = j))

    comparison_metrics <- rbind(comparison_metrics,data.frame(metric = "rmse_test",
                                                              value = rmse(x = result_[[j]]$bartmod$yhat.test.mean,
                                                                           y = result_[[j]]$cv$test$y),
                                                              model = "BART",fold = j))

    # Calculating the CRPS as well
    comparison_metrics <- rbind(comparison_metrics,data.frame(metric = "crps_train",
                                                              value = crps(y = result_[[j]]$cv$train$y ,
                                                                           means = result_[[j]]$bartmod$yhat.train.mean,
                                                                           sds = rep(mean(result_[[j]]$bartmod$sigma), length(result_[[j]]$cv$train$y) ))$CRPS,
                                                              model = "BART",fold = j))

    comparison_metrics <- rbind(comparison_metrics,data.frame(metric = "crps_test",
                                                              value = crps(y = result_[[j]]$cv$test$y ,
                                                                           means = result_[[j]]$bartmod$yhat.test.mean,
                                                                           sds = rep(mean(result_[[j]]$bartmod$sigma), length(result_[[j]]$cv$test$y) ))$CRPS,
                                                              model = "BART",fold = j))


    # ============================
    # Calculating metrics for softBART
    # ============================

    comparison_metrics <- rbind(comparison_metrics,data.frame(metric = "rmse_train",
                                                              value = rmse(x = result_[[j]]$softbartmod$y_hat_train_mean,
                                                                           y = result_[[j]]$cv$train$y),
                                                              model = "softBART",fold = j))

    comparison_metrics <- rbind(comparison_metrics,data.frame(metric = "rmse_test",
                                                              value = rmse(x = result_[[j]]$softbartmod$y_hat_test_mean,
                                                                           y = result_[[j]]$cv$test$y),
                                                              model = "softBART",fold = j))

    # Calculating the CRPS as well
    comparison_metrics <- rbind(comparison_metrics,data.frame(metric = "crps_train",
                                                              value = crps(y = result_[[j]]$cv$train$y ,
                                                                           means = result_[[j]]$softbartmod$y_hat_train_mean,
                                                                           sds = rep(mean(result_[[j]]$softbartmod$sigma), length(result_[[j]]$cv$train$y) ))$CRPS,
                                                              model = "softBART",fold = j))

    comparison_metrics <- rbind(comparison_metrics,data.frame(metric = "crps_test",
                                                              value = crps(y = result_[[j]]$cv$test$y ,
                                                                           means = result_[[j]]$softbart$y_hat_test_mean,
                                                                           sds = rep(mean(result_[[j]]$softbartmod$sigma), length(result_[[j]]$cv$test$y) ))$CRPS,
                                                              model = "softBART",fold = j))
    # ============================
    # Calculating metrics for MOTRBART
    # ============================

    comparison_metrics <- rbind(comparison_metrics,data.frame(metric = "rmse_train",
                                                              value = rmse(x = colMeans(result_[[j]]$motrbartmod$y_hat),
                                                                           y = result_[[j]]$cv$train$y),
                                                              model = "motrBART",fold = j))

    comparison_metrics <- rbind(comparison_metrics,data.frame(metric = "rmse_test",
                                                              value = rmse(x = colMeans(result_[[j]]$motrbart_pred),
                                                                           y = result_[[j]]$cv$test$y),
                                                              model = "motrBART",fold = j))

    # Calculating the CRPS as well
    comparison_metrics <- rbind(comparison_metrics,data.frame(metric = "crps_train",
                                                              value = crps(y = result_[[j]]$cv$train$y ,
                                                                           means = colMeans(result_[[j]]$motrbartmod$y_hat),
                                                                           sds = rep(mean(sqrt(result_[[j]]$motrbartmod$sigma2)), length(result_[[j]]$cv$train$y) ))$CRPS,
                                                              model = "motrBART",fold = j))

    comparison_metrics <- rbind(comparison_metrics,data.frame(metric = "crps_test",
                                                              value = crps(y = result_[[j]]$cv$test$y ,
                                                                           means = colMeans(result_[[j]]$motrbart_pred),
                                                                           sds = rep(mean(sqrt(result_[[j]]$motrbartmod$sigma2)), length(result_[[j]]$cv$test$y) ))$CRPS,
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

