# Just running the default values so I can go through the function and debug all
#the things

mlbench.friedman1.nointeraction <- function (n, sd = 1)
{
  x <- matrix(runif(4 * n), ncol = 4)
  y <- 10 * sin(pi * x[, 1])
  y <- y + 20 * (x[, 2] - 0.5)^2 + 10 * x[, 3] + 5 * x[, 4]
  if (sd > 0) {
    y <- y + rnorm(n, sd = sd)
  }
  list(x = x, y = y)
}

mlbench.friedman1.nointeraction.noise <- function (n, sd = 1)
{
  x <- matrix(runif(8 * n), ncol = 8)
  y <- 10 * sin(pi * x[, 1])
  y <- y + 20 * (x[, 2] - 0.5)^2 + 10 * x[, 3] + 5 * x[, 4]
  if (sd > 0) {
    y <- y + rnorm(n, sd = sd)
  }
  list(x = x, y = y)
}

mlbench.d1 <- function(n, sd = 1) {
  x <- matrix(runif(n,min = -pi,max = pi),ncol = 1)
  y <- sin(2*x)
  if (sd > 0) {
    y <- y + rnorm(n, sd = sd)
  }
  list(x = x, y = y)
}

mlbench.d1.break <- function(n, sd = 1) {
  x <- matrix(runif(n,min = -pi,max = pi),ncol = 1)
  y <- sin(2*x)
  y[x<0] <- y[x<0] + 5
  y[x>=0] <- y[x>=0] - 5

  if (sd > 0) {
    y <- y + rnorm(n, sd = sd)
  }
  list(x = x, y = y)
}
