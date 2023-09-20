library(mlbench)
n_ <- 101
sd_ <- 1
sim_train <- mlbench.friedman1.nointeraction(n = n_,sd = sd_)  |> as.data.frame()
sim_test <- mlbench.friedman1.nointeraction(n = n_,sd = sd_)  |> as.data.frame()

# sim_train <- mlbench.d1.break(n = n_,sd = 0.5)  |> as.data.frame()
# sim_test <- mlbench.d1.break(n = n_,sd = 0.5) |> as.data.frame()
x_train <- sim_train |> dplyr::select(dplyr::starts_with("x"))
x_test <-  sim_test|> dplyr::select(dplyr::starts_with("x"))
y_train <- sim_train$y

# x_train <- x_train[,1:5]
# x_test <- x_test[,1:5]
n_tree <- 10
node_min_size = 5
n_mcmc = 3000
n_burn = 0
alpha = 0.95
beta = 2
df = 3
sigquant = 0.9
kappa = 2
tau = 100
scale_bool = TRUE
stump = FALSE
no_rotation_bool = FALSE
numcut = 100L # Defining the grid of split rules
usequants = TRUE
delta <- 1

# Splines parameters
nIknots = 5
dif_order = 0
motrbart_bool <- FALSE
use_bs <- FALSE
