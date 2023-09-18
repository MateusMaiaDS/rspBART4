all_bart <- function(cv_element,
                     nIknots_){

  train <- cv_element$train
  test <- cv_element$test

  # Getting the training elements
  x_train <- train %>% dplyr::select(dplyr::starts_with("x"))
  x_test <- test %>% dplyr::select(dplyr::starts_with("x"))
  y_train <- train %>% dplyr::pull("y")

  # Running the model
  spBART <- rspBART(x_train = x_train,
                    x_test = x_test,y_train = y_train,
                    n_mcmc = 2000,node_min_size = 5,
                    n_burn = 0,nIknots = nIknots_,n_tree = 50,
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
