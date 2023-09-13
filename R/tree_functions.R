# Creating a stump for a tree
stump <- function(data){

  # Creating the base node
  node <- list()
  node[["node0"]] <- list(
    # Creating the node number
    node_number = 0,
    isRoot = TRUE,
    # Creating a vector with the tranining index
    train_index = 1:nrow(data$x_train),
    test_index = 1:nrow(data$x_test),
    depth_node = 0,
    node_var = NA,
    node_cutpoint_index = NA,
    left = NA,
    right = NA,
    parent_node = NA,
    ancestors = NULL,
    terminal = TRUE,
    gamma = 0,
    betas_vec = NULL
   )

  # Returning the node
  return(node)

}

# Get all the terminal nodes
get_terminals <- function(tree){

  # Return the name of the termianl nodes
  return(names(tree)[unlist(lapply(tree, function(x){x$terminal}),use.names =  TRUE)])
}

# Get nog terminal nodes
get_nogs <- function(tree){

  # Return the name of the termianl nodes
  non_terminal <- names(tree)[!unlist(lapply(tree, function(x){x$terminal}),use.names =  TRUE)]

  # In case there are non nonterminal nondes
  if(length(non_terminal)==0){
    return(non_terminal)
  }

  bool_nog <- vector("logical",length = length(non_terminal))
  for(i in 1:length(bool_nog)){
    # Checking if both children are terminal
    if( tree[[tree[[non_terminal[i]]]$left]]$terminal & tree[[tree[[non_terminal[i]]]$right]]$terminal) {
      bool_nog[i] <- TRUE
    }
  }

  return(  non_terminal[bool_nog])
}

# Getting the maximum node index number
get_max_node <- function(tree){

  # Return the name of the termianl nodes
  return(max(unlist(lapply(tree, function(x){x$node_number}),use.names =  TRUE)))
}




# A function to calculate the loglikelihood
nodeLogLike <- function(curr_part_res,
                        ancestors,
                        index_node,
                        data){

  # Subsetting the residuals
  curr_part_res_leaf <- curr_part_res[index_node]

  # Getting the number of observationsin the terminal node
  n_leaf <- length(index_node)
  d_basis <- length(ancestors)
  ones <- matrix(1,nrow = n_leaf)
  # Getting the index of all covariates for each basis from each ancestors
  D_subset_index <- unlist(data$basis_subindex[ancestors])

  # Calculating the nodeLogLikelihood for a stump
  if(length(D_subset_index)==0){

    # Using the Andrew's approach I would have
    mean_aux <- rep(0,length(curr_part_res_leaf))
    cov_aux <- diag(x = (data$tau^(-1)),nrow = n_leaf) + (data$tau_gamma^(-1))*matrix(1,nrow = n_leaf,ncol = n_leaf)
    result <- mvnfast::dmvn(X = curr_part_res_leaf,mu = mean_aux,sigma = cov_aux,log = TRUE)
  } else {
    # Getting the p_{tell} i.e: number of betas of the current terminal node
    d_basis <- length(D_subset_index)
    D_leaf <- data$D_train[index_node,D_subset_index, drop = FALSE]

    # Using the Andrew's approach I would have
    mean_aux <- rep(0,length(curr_part_res_leaf))
    cov_aux <- diag(x = (data$tau^(-1)),nrow = n_leaf) + (data$tau_gamma^(-1))*matrix(1,nrow = n_leaf,ncol = n_leaf)  + (data$tau_beta^(-1))*tcrossprod(D_leaf)
    result <- mvnfast::dmvn(X = curr_part_res_leaf,mu = mean_aux,sigma = cov_aux,log = TRUE)

  }

  return(c(result))

}

# Grow a tree
grow <- function(tree,
                 curr_part_res,
                 data){

  # Getting the maximum index number
  max_index <- get_max_node(tree)

  # Sampling a terminal node
  terminal_nodes <- get_terminals(tree)
  n_t_nodes <- length(terminal_nodes)
  nog_nodes <- get_nogs(tree)
  n_nog_nodes <- length(nog_nodes)
  g_node_name <- sample(terminal_nodes,size = 1)
  g_node <- tree[[g_node_name]]


  valid_terminal_node <- TRUE
  valid_count <- 0
  while(valid_terminal_node){
      # Convinience while to avoid terminal nodes of 2
      # Sample a split var
      p_var <- sample(1:ncol(data$x_train),size = 1)

      # Selecting an available cutpoint from this terminal node
      valid_range_grow <- range(data$x_train[g_node$train_index,p_var])

      # Case of invalid range
      if(length(valid_range_grow)==0){
        return(tree)
      }

      # Subsetting the indexes of
      valid_cutpoint <- which(data$xcut_m[,p_var]>valid_range_grow[1] & data$xcut_m[,p_var]<valid_range_grow[2])

      # When there's no valid cutpoint on the sampled terminal node
      if(length(valid_cutpoint)==0){
        return(tree)
      }

      # Getting which cutpoints are valid and sample onde index
      sample_cutpoint <- sample(valid_cutpoint,
             size = 1)

      # Getting the left & right index
      left_index  <- data$all_var_splits[[p_var]][[sample_cutpoint]]$left_train[data$all_var_splits[[p_var]][[sample_cutpoint]]$left_train %in% g_node$train_index]
      right_index  <- data$all_var_splits[[p_var]][[sample_cutpoint]]$right_train[data$all_var_splits[[p_var]][[sample_cutpoint]]$right_train %in% g_node$train_index]

      left_test_index  <- data$all_var_splits[[p_var]][[sample_cutpoint]]$left_test[data$all_var_splits[[p_var]][[sample_cutpoint]]$left_test %in% g_node$test_index]
      right_test_index  <- data$all_var_splits[[p_var]][[sample_cutpoint]]$right_test[data$all_var_splits[[p_var]][[sample_cutpoint]]$right_test %in% g_node$test_index]



      # Verifying that the correct number was used
      if((length(left_index)+length(right_index))!=length(g_node$train_index)){
        stop("Something went wrong here --- train grown index doest match")
      }

      if((length(left_test_index)+length(right_test_index))!=length(g_node$test_index)){
        stop("Something went wrong here --- test grown index doest match")
      }

      if( (length(left_index) > data$node_min_size) & (length(right_index)>data$node_min_size)){
          # Getting out of the while
          break
      } else {

          # Adding one to the counter
          valid_count = valid_count + 1

          # Stop trying to search for a valid cutpoint
          if(valid_count > 2) {
            valid_terminal_node = FALSE
            return(tree)
          }
      }
  }

  # For convinience we are going to avoid terminal nodes less than 2
  if( (length(left_index)<2) || (length(right_index) < 2)) {
    print("AARARARARRA")
    stop("Error of invalid terminal node")
  }
  # Calculating loglikelihood for the grown node, the left and the right node

  g_loglike <- nodeLogLike(curr_part_res = curr_part_res,
                           ancestors = unique(g_node$ancestors),
                           index_node = g_node$train_index,
                           data = data)


  left_loglike <-  nodeLogLike(curr_part_res = curr_part_res,
                               index_node = left_index,
                               ancestors = unique(c(g_node$ancestors,p_var)),
                               data = data)

  right_loglike <-  nodeLogLike(curr_part_res = curr_part_res,
                               index_node = right_index,
                               ancestors = unique(c(g_node$ancestors,p_var)),
                               data = data)

  # Calculating the prior
  prior_loglike <- log(data$alpha*(1+g_node$depth_node)^(-data$beta)) + # Prior of the grown node becoming nonterminal
                   2*log(1-data$alpha*(1+g_node$depth_node+1)^(-data$beta)) - # plus the prior of the two following nodes being terminal
                   log(1-data$alpha*(1+g_node$depth_node)^(-data$beta)) # minus the probability of the grown node being terminal

  # Transition prob
  log_trasition_prob  = log(0.3/(n_nog_nodes+1))-log(0.3/n_t_nodes)

  # Calculating the acceptance probability
  acceptance <- exp(-g_loglike+left_loglike+right_loglike+prior_loglike+log_trasition_prob)

  # Getting the training the left and the right index for the the grown node
  if(stats::runif(n = 1)<acceptance){
    left_node <- list(node_number = max_index+1,
                      isRoot = FALSE,
                      train_index = left_index,
                      test_index = left_test_index,
                      depth_node = g_node$depth_node+1,
                      node_var = p_var,
                      node_cutpoint_index = sample_cutpoint,
                      left = NA,
                      right = NA,
                      parent_node = g_node_name,
                      ancestors = c(g_node$ancestors,p_var),
                      terminal = TRUE,
                      gamma = 0,
                      betas_vec = rep(0,ncol(data$D_train)))

    # names(tree[[1]])

    right_node <- list(node_number = max_index+2,
                      isRoot = FALSE,
                      train_index = right_index,
                      test_index = right_test_index,
                      depth_node = g_node$depth_node+1,
                      node_var = p_var,
                      node_cutpoint_index = sample_cutpoint,
                      left = NA,
                      right = NA,
                      parent_node = g_node_name,
                      ancestors = c(g_node$ancestors,p_var),
                      terminal = TRUE,
                      gamma = 0,
                      betas_vec = rep(0,ncol(data$D_train)))

    # Modifying the current node
    tree[[g_node_name]]$left = paste0("node",max_index+1)
    tree[[g_node_name]]$right = paste0("node",max_index+2)
    tree[[g_node_name]]$terminal = FALSE

    tree[[paste0("node",max_index+1)]] <- left_node
    tree[[paste0("node",max_index+2)]] <- right_node


  } else {

    # Do nothing

  }

  # Return the new tree
  return(tree)
}


# Pruning a tree
prune <- function(tree,
                 curr_part_res,
                 data){


  # Getting the maximum index number
  max_index <- get_max_node(tree)

  # Sampling a terminal node
  terminal_nodes <- get_terminals(tree)
  n_t_nodes <- length(terminal_nodes)
  nog_nodes <- get_nogs(tree)
  n_nog_nodes <- length(nog_nodes)

  # Just in case to avoid errors
  if(n_nog_nodes==0){
    return(tree)
  }

  # Selecting a node to be pruned
  p_node_name <- sample(nog_nodes,size = 1)
  p_node <- tree[[p_node_name]]

  # Getting the indexes from the left and right children from the pruned node
  children_left_index <- tree[[p_node$left]]$train_index
  children_right_index <- tree[[p_node$right]]$train_index
  children_left_ancestors <- tree[[p_node$left]]$ancestors
  children_right_ancestors <- tree[[p_node$right]]$ancestors

  # Calculating loglikelihood for the grown node, the left and the right node

  p_loglike <- nodeLogLike(curr_part_res = curr_part_res,
                           index_node = p_node$train_index,
                           data = data,
                           ancestors = unique(p_node$ancestors))


  p_left_loglike <-  nodeLogLike(curr_part_res = curr_part_res,
                               index_node = children_left_index,
                               ancestors = unique(children_left_ancestors),
                               data = data)

  p_right_loglike <-  nodeLogLike(curr_part_res = curr_part_res,
                                index_node = children_right_index,
                                ancestors = unique(children_right_index),
                                data = data)

  # Calculating the prior
  prior_loglike <- log(1-data$alpha*(1+p_node$depth_node)^(-data$beta)) - # Prior of the new terminal node
    log(data$alpha*(1+p_node$depth_node)^(-data$beta)) - # Prior of the grown node becoming nonterminal
    2*log(1-data$alpha*(1+p_node$depth_node+1)^(-data$beta))  # plus the prior of the two following nodes being terminal
    # minus the probability of the grown node being terminal

  # Transition prob
  log_trasition_prob  = log(0.3/(n_t_nodes))-log(0.3/n_nog_nodes)

  # Calculating the acceptance probability
  acceptance <- exp(p_loglike-p_left_loglike-p_right_loglike+prior_loglike+log_trasition_prob)

  # Getting the training the left and the right index for the the grown node
  if(stats::runif(n = 1)<acceptance){

    # Erasing the terminal nodes
    tree[[p_node$left]] <- NULL
    tree[[p_node$right]] <- NULL

    # Modifying back the pruned node
    tree[[p_node_name]]$left <- NA
    tree[[p_node_name]]$right <- NA
    tree[[p_node_name]]$terminal <- TRUE

  } else {
    # Do nothing
  }

  # Return the new tree
  return(tree)

}


# Change a tree
change <- function(tree,
                 curr_part_res,
                 data){

  # Sampling a terminal node
  nog_nodes <- get_nogs(tree)
  n_nog_nodes <- length(nog_nodes)
  c_node_name <- sample(nog_nodes,size = 1)
  c_node <- tree[[c_node_name]]


  valid_terminal_node <- TRUE
  valid_count <- 0


  while(valid_terminal_node){
    # Convinience while to avoid terminal nodes of 2
    # Sample a split var
    p_var <- sample(1:ncol(data$x_train),size = 1)

    # Selecting an available cutpoint from this terminal node
    valid_range_grow <- range(data$x_train[c_node$train_index,p_var])

    # Subsetting the indexes of
    valid_cutpoint <- which(data$xcut_m[,p_var]>valid_range_grow[1] & data$xcut_m[,p_var]<valid_range_grow[2])

    # When there's no valid cutpoint on the sampled terminal node
    if(length(valid_cutpoint)==0){
      return(tree)
    }

    # Getting which cutpoints are valid and sample onde index
    sample_cutpoint <- sample(valid_cutpoint,
                              size = 1)

    # Getting the left & right index
    left_index  <- data$all_var_splits[[p_var]][[sample_cutpoint]]$left_train[data$all_var_splits[[p_var]][[sample_cutpoint]]$left_train %in% c_node$train_index]
    right_index  <- data$all_var_splits[[p_var]][[sample_cutpoint]]$right_train[data$all_var_splits[[p_var]][[sample_cutpoint]]$right_train %in% c_node$train_index]

    left_test_index  <- data$all_var_splits[[p_var]][[sample_cutpoint]]$left_test[data$all_var_splits[[p_var]][[sample_cutpoint]]$left_test %in% c_node$test_index]
    right_test_index  <- data$all_var_splits[[p_var]][[sample_cutpoint]]$right_test[data$all_var_splits[[p_var]][[sample_cutpoint]]$right_test %in% c_node$test_index]



    # Verifying that the correct number was used
    if((length(left_index)+length(right_index))!=length(c_node$train_index)){
      stop("Something went wrong here --- train grown index doest match")
    }

    if((length(left_test_index)+length(right_test_index))!=length(c_node$test_index)){
      stop("Something went wrong here --- test grown index doest match")
    }

    # Avoiding having terminal nodes with just one observation
    if( (length(left_index) > data$node_min_size) & (length(right_index)>data$node_min_size)){
      # Getting out of the while
      break
    } else {

      # Adding one to the counter
      valid_count = valid_count + 1

      # Stop trying to search for a valid cutpoint
      if(valid_count > 2) {
        valid_terminal_node = FALSE
        return(tree)
      }
    }
  }

  # For convinience we are going to avoid terminal nodes less than 2
  if( (length(left_index)<2) || (length(right_index) < 2)) {
    stop("Error of invalid terminal node")
  }
  # Calculating loglikelihood for the new changed nodes and the old ones

  c_loglike_left <- nodeLogLike(curr_part_res = curr_part_res,
                           index_node = tree[[c_node$left]]$train_index,
                           ancestors = unique(tree[[c_node$left]]$ancestors),
                           data = data)


  c_loglike_right <-  nodeLogLike(curr_part_res = curr_part_res,
                               index_node = tree[[c_node$right]]$train_index,
                               ancestors = unique(tree[[c_node$right]]$ancestors),
                               data = data)

  # Calculating a new ancestors left and right
  old_p_var <- tree[[c_node$left]]$node_var

  # Storing new left and right ancestors
  new_left_ancestors <- tree[[c_node$left]]$ancestors
  new_left_ancestors[length(new_left_ancestors)] <- p_var

  new_right_ancestors <- tree[[c_node$right]]$ancestors
  new_right_ancestors[length(new_right_ancestors)] <- p_var

  new_c_loglike_left <-  nodeLogLike(curr_part_res = curr_part_res,
                                index_node = left_index,
                                ancestors = unique(new_left_ancestors),
                                data = data)

  new_c_loglike_right <-  nodeLogLike(curr_part_res = curr_part_res,
                                     index_node = right_index,
                                     ancestors = unique(new_right_ancestors),
                                     data = data)


  # Calculating the acceptance probability
  acceptance <- exp(new_c_loglike_left+new_c_loglike_right-c_loglike_left-c_loglike_right)

  # Getting the training the left and the right index for the the grown node
  if(stats::runif(n = 1,min = 0,max = 1)<acceptance){

    # Updating the left and the right node
    # === Left =====
    tree[[c_node$left]]$node_var <- p_var
    tree[[c_node$left]]$node_cutpoint_index <- sample_cutpoint
    tree[[c_node$left]]$train_index <- left_index
    tree[[c_node$left]]$test_index <- left_test_index
    tree[[c_node$left]]$ancestors <- new_left_ancestors

    #==== Right ====
    tree[[c_node$right]]$node_var <- p_var
    tree[[c_node$right]]$node_cutpoint_index <- sample_cutpoint
    tree[[c_node$right]]$train_index <- right_index
    tree[[c_node$right]]$test_index <- right_test_index
    tree[[c_node$right]]$ancestors <- new_right_ancestors

  } else {

    # Do nothing

  }

  # Return the new tree
  return(tree)

}


# ==========================
# Updating the gamma
# ==========================
updateGamma <- function(tree,
                        curr_part_res,
                        data){

  # Getting the terminals
  t_nodes_names <- get_terminals(tree)


  for(i in 1:length(t_nodes_names)){

    cu_t <- tree[[t_nodes_names[i]]]
    # Selecting the actually parameters subsetting
    leaf_basis_subindex <- unlist(data$basis_subindex[unique(cu_t$ancestors)]) # Recall to the "unique function here

    D_dim <- length(leaf_basis_subindex)

    # Subpartitioning the leaf elements
    D_leaf <- data$D_train[cu_t$train_index,leaf_basis_subindex,drop = FALSE]
    r_leaf <- matrix(curr_part_res[cu_t$train_index],ncol = 1)
    n_leaf <- length(cu_t$train_index)
    diag_ <- diag(nrow = n_leaf)
    diag_basis <- diag(nrow = ncol(D_leaf))

    if(length(leaf_basis_subindex)==0){
    # Computing mean and sd from the intercep
      one_s_gamma_one <- data$tau*diag_
      gamma_post_var <- 1/(one_s_gamma_one+data$tau_gamma)
      gamma_post_mean <- gamma_post_var*sum(r_leaf)
    } else {
      s_gamma_inv <- data$tau*diag_-(data$tau^2)*D_leaf%*%solve((data$tau_beta*diag_basis+data$tau*crossprod(D_leaf)),t(D_leaf))
      one_s_gamma_one <- sum(s_gamma_inv)
      gamma_post_var <- 1/(one_s_gamma_one+data$tau_gamma)
      gamma_post_mean <- gamma_post_var*crossprod(colSums(s_gamma_inv),r_leaf)
    }


    tree[[t_nodes_names[i]]]$gamma = stats::rnorm(n = 1,mean = gamma_post_mean,sd = sqrt(gamma_post_var))

  }

  return(tree)
}

# ============
# Update Betas
# ============
updateBetas <- function(tree,
                        curr_part_res,
                        data){


  # Getting the terminals
  t_nodes_names <- get_terminals(tree)


  for(i in 1:length(t_nodes_names)){


    # Select the current terminal node
    cu_t <- tree[[t_nodes_names[i]]]


    if(length(cu_t$ancestors)==0){
      # There's nothing to update if there's no ancestors
      next
    }

    res_leaf <- matrix(curr_part_res[cu_t$train_index], ncol=1)

    # Creatinga  vector of zeros for betas_vec
    tree[[t_nodes_names[[i]]]]$betas_vec <- rep(0,ncol(data$D_train))

    # Selecting the actually parameters subsetting
    leaf_basis_subindex <- unlist(data$basis_subindex[unique(cu_t$ancestors)]) # Recall to the unique() here too
    basis_dim <- length(leaf_basis_subindex)
    D_leaf <- data$D_train[cu_t$train_index,leaf_basis_subindex, drop = FALSE]
    n_leaf <- length(cu_t$train_index)
    diag_leaf <- diag(nrow = n_leaf)
    diag_basis <- diag(nrow = basis_dim)

    s_beta_inv <- data$tau*diag_leaf-(data$tau^2)/(data$tau_gamma+n_leaf*data$tau)*matrix(1,nrow = n_leaf,ncol = n_leaf)
    beta_post_var_chol <- (chol(   ( crossprod(D_leaf,s_beta_inv)%*%D_leaf + data$tau*diag_basis ) ))
    beta_post_var <- chol2inv(beta_post_var_chol)

    # Calculating the mean to be sampled
    beta_mean <- beta_post_var%*%crossprod(D_leaf,s_beta_inv)%*%res_leaf
    # print(sum_aux)
    # Check this line again if there's any bug on the cholesky decomposition
    tree[[t_nodes_names[i]]]$betas_vec[leaf_basis_subindex] <- c(mvnfast::rmvn(n = 1,mu = beta_mean,
                                                            sigma = beta_post_var,isChol = TRUE))
  }

  # Returning the tree
  return(tree)

}

# =================
# Update \tau_betas
# =================
update_tau_betas <- function(forest,
                             data){



  # Setting some default hyperparameters
  a_tau_beta <- d_tau_beta <- 1
  tau_b_shape <- 0.0
  tau_b_rate <- 0.0


  # Iterating over all trees
  for(i in 1:length(forest)){

    # Getting terminal nodes
    t_nodes_names <- get_terminals(forest[[i]])
    n_t_nodes <- length(t_nodes_names)

    # Iterating over the terminal nodes
    for(j in 1:length(t_nodes_names)){

      cu_t <- forest[[i]][[t_nodes_names[j]]]
      leaf_basis_subindex <- unlist(data$basis_subindex[unique(cu_t$ancestors)]) # Recall to the unique() function here

      if(!is.null(cu_t$betas_vec)){
        tau_b_shape <- tau_b_shape + length(leaf_basis_subindex)
        tau_b_rate <- tau_b_rate + c(crossprod(cu_t$betas_vec[leaf_basis_subindex]))
      }

    }


    tau_beta_vec_aux <- rgamma(n = 1,
                                   shape = 0.5*tau_b_shape + a_tau_beta,
                                   rate = 0.5*tau_b_rate + d_tau_beta)
  }

  return(tau_beta_vec_aux)

}


# =================
# Update \tau_gammas
# =================
update_tau_gamma <- function(forest,
                             data){


  # Same default as the paper;
  a_tau_gamma <- d_tau_gamma <- 1

  tau_gamma_shape <- 0
  tau_gamma_rate <- 0

  # Iterating over all trees
  for(i in 1:length(forest)){

    # Getting terminal nodes
    t_nodes_names <- get_terminals(forest[[i]])
    n_t_nodes <- length(t_nodes_names)
    tau_gamma_shape <- tau_gamma_shape + n_t_nodes

    # Iterating over the terminal nodes
    for(j in 1:length(t_nodes_names)){
      tau_gamma_rate <- tau_gamma_rate + forest[[i]][[t_nodes_names[j]]]$gamma^2

    }


  }

  tau_gamma_aux <- stats::rgamma(n = 1,
                                shape = 0.5*tau_gamma_shape + a_tau_gamma,
                                rate = 0.5*tau_gamma_rate + d_tau_gamma)


  return(tau_gamma_aux)

}

# ===================
# Updating the \delta
# ===================

update_delta <- function(data){

  delta_aux <- numeric(dim(data$B_train_arr)[3])
  nu <- 2
  a_delta <- d_delta <- 0.001

  for(i in 1:dim(data$B_train_arr)[3]){

    delta_aux[i] <- stats::rgamma(n = 1,shape =  0.5*nu + a_delta,
                               rate = 0.5*nu*data$tau_beta_vec[i]+d_delta)
  }


  # Returning the delta sampled vector
  return(delta_aux)
}

# A function to get predictions
getPredictions <- function(tree,
                           data){

  # Creating the vector to hold the values of the prediction
  y_hat <- matrix(0, nrow = nrow(data$x_train), ncol = ncol(data$x_train)+1)
  y_hat_test <- matrix(0,nrow(data$x_test), ncol = ncol(data$x_test)+ 1 )

  # Getting terminal nodes
  t_nodes <- get_terminals(tree = tree)
  n_t_nodes <- length(t_nodes)

  for(i in 1:n_t_nodes){

    leaf_train_index <- tree[[t_nodes[i]]]$train_index
    leaf_test_index <- tree[[t_nodes[i]]]$test_index
    leaf_intercept <- tree[[t_nodes[i]]]$gamma
    leaf_ancestors <- unique(tree[[t_nodes[[i]]]]$ancestors) # recall the unique() argument here
    leaf_basis_subindex <- data$basis_subindex[leaf_ancestors]

    # Test unit
    if(length(leaf_ancestors)!=length(leaf_basis_subindex)){
      stop("Error on the getPredictions function")
    }
    # Only add the marginal effects if the variables are within that terminal node
    if(length(leaf_basis_subindex)!=0){
      for(k in 1:length(leaf_basis_subindex)){

          y_hat[leaf_train_index,leaf_ancestors[k]] <- y_hat[leaf_train_index,leaf_ancestors[k]] + data$D_train[leaf_train_index,leaf_basis_subindex[[k]], drop = FALSE]%*%tree[[t_nodes[i]]]$betas_vec[leaf_basis_subindex[[k]]]
          y_hat_test[leaf_test_index,leaf_ancestors[k]] <- y_hat_test[leaf_test_index,leaf_ancestors[k]] + data$D_test[leaf_test_index,leaf_basis_subindex[[k]], drop = FALSE]%*%tree[[t_nodes[i]]]$betas_vec[leaf_basis_subindex[[k]]]

      }
    }
    y_hat[leaf_train_index,ncol(data$x_train)+1] <- y_hat[leaf_train_index,ncol(data$x_train)+1] +  leaf_intercept
    y_hat_test[leaf_test_index,ncol(data$x_test)+1] <- y_hat_test[leaf_test_index,ncol(data$x_test)+1] + leaf_intercept

  }

  # Returning both training and test set predictions
  return(list(y_train_hat = y_hat,
              y_hat_test = y_hat_test))

}

# Updating tau
update_tau <- function(y_train_hat,
                       data){

  # Sampling a tau value
  n_ <- nrow(data$x_train)
  tau_sample <- stats::rgamma(n = 1,shape = 0.5*n_+data$a_tau,rate = 0.5*crossprod((data$y_train-y_train_hat))+data$d_tau)

  return(tau_sample)

}


