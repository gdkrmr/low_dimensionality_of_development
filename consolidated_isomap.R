## library(tidyverse)
## library(dimRed)
## library(igraph)
## library(data.table)
## library(abind)
## library(Matrix)
## library(rgl)
## library(RANN)





## sr1 <- dimRed::loadDataSet("Swiss Roll")

## plot(sr1)

## ## Isomap works just the same, when duplicating all columns
## sr <- cbind(sr1 %>% getData,
##             sr1 %>% getData)
## pairs(sr)
## sr_iso <- embed(sr, "Isomap", ndim = 6, knn = 15)

## ## Disregard all but Iso1 and Iso2:
## plot(sr_iso)

## ## Now try the same with some noise added
## sr2 <- sr1 %>% getData %>% { . + matrix(rnorm(n = 6000, sd = 0.05), 2000, 3)}

## sr <- cbind(sr1 %>% getData, sr2)
## pairs(sr)
## sr_iso <- embed(sr, "Isomap", ndim = 6, knn = 15)

## ## Disregard all but Iso1 and Iso2:
## plot(sr_iso)

## ## Now duplicate only one columns:
## sr <- cbind(sr1 %>% getData,
##             sr1 %>% getData %>% {.[, 1, drop = FALSE]})

## sr_iso <- embed(sr, "Isomap", ndim = 6, knn = 10)

## ## Disregard all but Iso1 and Iso2:
## ## This distorts everything quite a bit with less neighbors
## plot(sr_iso)



## ## Do Isomap, taking the mean of the squared geodesic distances

## sr1_knn <- dimRed:::makeKNNgraph(sr1 %>% getData, k = 15)
## ## sr2_knn <- dimRed:::makeKNNgraph(sr2, k = 15)
## ## Do the same, but with only the dimensions that define the roll
## sr2_knn <- dimRed:::makeKNNgraph(sr2[ ,2:3], k = 15)

## sr1_geo <- igraph::distances(sr1_knn) ^ 2
## sr2_geo <- igraph::distances(sr2_knn) ^ 2
## sr_geo <- (sr1_geo * 3 + sr2_geo * 2) / 5

## dc <- diag(rep(1, times = 2000)) - matrix(1/2000, 2000, 2000)

## sr1_dc <- -1/2 * dc %*% sr1_geo %*% dc
## sr2_dc <- -1/2 * dc %*% sr2_geo %*% dc
## sr_dc <-  -1/2 * dc %*% sr_geo %*% dc

## sr1_eig <- eigen(sr1_dc, symmetric = TRUE)
## sr2_eig <- eigen(sr2_dc, symmetric = TRUE)
## sr_eig <- eigen(sr_dc, symmetric = TRUE)

## sr1_iso <- sweep(sr1_eig$vectors[, 1:6], 2, sqrt(sr1_eig$values[1:6]), "*")
## sr2_iso <- sweep(sr2_eig$vectors[, 1:6], 2, sqrt(sr2_eig$values[1:6]), "*")
## sr_iso <- sweep(sr_eig$vectors[, 1:6], 2, sqrt(sr_eig$values[1:6]), "*")

## ## Again disregard everything but Iso 1 and 2:
## pairs(sr1_iso)
## pairs(sr2_iso)
## pairs(sr_iso)                           # Dimension 3 has a weird artifact



## ## Piecewise manifold

## bb <- 0.1
## sr1 <- expand.grid(x = seq(4.7, 11, by = bb) %>% round(1),
##                    y = seq(0, 1, by = bb)) %>% as.data.table
## sr2 <- expand.grid(x = seq(7.8, 14, by = bb) %>% round(1),
##                    y = seq(0, 1, by = bb)) %>% as.data.table

## sr1[, c("s1", "s2", "s3") := as.data.table(dimRed:::swissRollMapping(x, y))]
## sr2[, c("r1", "r2", "r3") := as.data.table(dimRed:::swissRollMapping(x, y))]

## mfrow3d(1, 3, sharedMouse = TRUE)
## plot3d(sr1[,c("s1", "s2", "s3")], col = "green")
## next3d()
## plot3d(sr2[,c("r1", "r2", "r3")], col = "blue")
## next3d()
## plot3d(sr1[,c("s1", "s2", "s3")], col = "green")
## points3d(sr2[,c("r1", "r2", "r3")], col = "blue")

## sr <- merge(sr1, sr2, by = c("x", "y"), all = TRUE)
## image(is.na(sr))
## sb <- copy(sr)
## sr[is.na(sr$s1), c("s1", "s2", "s3") := .(r1, r2, r3)]

## n1 <- nrow(sr1)
## n2 <- nrow(sr2)
## n <- nrow(sr)

## n1 + n2 - n
## image(is.na(sr))
## image(is.na(sb))

## setkey(sr1, "x", "y")
## setkey(sr2, "x", "y")

## sr1
## sr2

## sr1_knn <- sr1[, c("s1", "s2", "s3")] %>% as.matrix %>% dimRed:::makeKNNgraph(k = 25)
## sr2_knn <- sr2[, c("r1", "r2", "r3")] %>% as.matrix %>% dimRed:::makeKNNgraph(k = 25)
## sr_knn <- sr[, c("s1", "s2", "s3")] %>% as.matrix %>% dimRed:::makeKNNgraph(k = 25)

## sr1_geo <- igraph::distances(sr1_knn, algorithm = "dijkstra")
## sr2_geo <- igraph::distances(sr2_knn, algorithm = "dijkstra")
## sb_geo <- igraph::distances(sr_knn, algorithm = "dijkstra")

## sr_geo <- list()
## sr_geo[[1]] <- matrix(NA_real_, n, n)
## sr_geo[[1]][!is.na(sb$s1), !is.na(sb$s1)] <- sr1_geo
## sr_geo[[2]] <- matrix(NA_real_, n, n)
## sr_geo[[2]][!is.na(sb$r1), !is.na(sb$r1)] <- sr2_geo
## sr_geo <- abind(sr_geo[[1]], sr_geo[[2]], along = 3)
## sr_geo <- rowMeans(sr_geo, na.rm = TRUE, dims = 2)

## image(sr_geo, useRaster = TRUE)
## image(sb_geo, useRaster = TRUE)

## sr_geo[is.na(sr_geo)] <- sb_geo[is.na(sr_geo)]
## image(sr_geo, useRaster = TRUE)

## sr1_geo %>% cmdscale(k = 3) %>% pairs
## sr2_geo %>% cmdscale(k = 3) %>% pairs
## sr_geo %>% cmdscale(k = 3) %>% pairs


#' construct a piecewise Isomap
#'
#' TODO: THIS VERSION DOES NOT DO WHAT IT IS SUPPOSED TO DO, I am not sure, why
#'
#' @param data a list of data.frames or data.tables
#' @param id_cols a character vector with the column names to merge the data
#'   frames.
#' @param knn the number of nearest neighbors for the Isomap
#'
#' @import data.table
#' @import dimRed
#' @import igraph
#' @import RANN
#'
#' @section TODOS
#'
#' TODO: is it better to make all the knn graphs before merging or after
#'   merging?
#'
#' Computationally it is better to do it before, when there are still the
#' orignal coorinates available
#'
pw_isomap_step_across <- function(data, id_cols, knn) {

  ## It is important to keep se sorting ALL the time

  lapply(data, data.table::setDT)
  lapply(data, data.table::setkeyv, cols = id_cols)

  dummy_data <- lapply(seq_along(data), function (i) {
    x <- data[[i]][, id_cols, with = FALSE]
    x[[paste0("p", i)]] <- TRUE
    data.table::setkeyv(x, paste0("p", i))
    x
  })

  occurrence_table <- Reduce(function(x, y) {
    data.table:::merge.data.table(x, y, by = id_cols, all = TRUE)
  }, dummy_data)

  nvars <- sapply(data, ncol) - length(id_cols)
  nobs <- nrow(occurrence_table)
  nid_cols <- length(id_cols)

  adj <- array(NA, dim = c(nobs, nobs, length(data)))
  for (i in seq_along(data)) {
    gd <- as.matrix(dist(data[[i]][, -id_cols, with = FALSE]))
    ## scale by number of observations
    gd <- sqrt((gd ^ 2) * sum(nvars) / nvars[i])
    adj[which(occurrence_table[[nid_cols + i]]),
        which(occurrence_table[[nid_cols + i]]), i] <- gd
  }
  adj <- rowMeans(adj, na.rm = TRUE, dims = 2)
  adj[is.nan(adj)] <- Inf

  ## Step across
  dgeo <- as.matrix(vegan::isomapdist(adj, knn))

  cmdscale(dgeo, k = 6)
}

#' construct a piecewise Isomap
#'
#' This version pieces together the kNN graphs
#'
#' @param data a list of data.frames or data.tables
#' @param id_cols a character vector with the column names to merge the data
#'   frames.
#' @param knn the number of nearest neighbors for the Isomap
#'
#' @import data.table
#' @import dimRed
#' @import igraph
#' @import RANN
#'
#' @section TODOS
#'
#' TODO: is it better to make all the knn graphs before merging or after
#'   merging?
#'
#' Computationally it is better to do it before, when there are still the
#' original coorinates available
#'
pw_isomap_graph <- function(data, id_cols, knn) {

  ## It is important to keep se sorting ALL the time

  lapply(data, data.table::setDT)
  lapply(data, data.table::setkeyv, id_cols)

  dummy_data <- lapply(seq_along(data), function (i) {
    x <- data[[i]][, id_cols, with = FALSE]
    x[[paste0("p", i)]] <- TRUE
    data.table::setkeyv(x, paste0("p", i))
    x
  })

  occurrence_table <- Reduce(function(x, y) {
      data.table:::merge.data.table(x, y, by = id_cols, all = TRUE)
    },
    dummy_data)

  nvars <- sapply(data, ncol) - length(id_cols)
  nobs <- nrow(occurrence_table)
  nid_cols <- length(id_cols)

  gs <- lapply(data, function (x) {
    x[, -id_cols, with = FALSE] %>%
      as.matrix %>%
      dimRed:::makeKNNgraph(k = knn)
  })

  a <- array(NA, dim = c(nobs, nobs, length(data)))
  for (i in seq_along(data)) {
    a[cbind(which(occurrence_table[[nid_cols + i]])[as_edgelist(gs[[i]])[, 1]],
            which(occurrence_table[[nid_cols + i]])[as_edgelist(gs[[i]])[, 2]],
            i)] <- sqrt((edge_attr(gs[[i]], "weight") ^ 2) / nvars[i])
    a[, , i] <- pmin(a[, , i], t(a[, , i]), na.rm = TRUE)
  }
  dgeo <- rowMeans(a, na.rm = TRUE, dims = 2)
  dgeo[is.nan(dgeo)] <- 0

  g <- igraph::graph_from_adjacency_matrix(
                 dgeo, mode = "upper", diag = FALSE, weighted = "weight")
  dgeo <- igraph::distances(g, algorithm = "dijkstra")

  cmdscale(dgeo, k = 6)
}


#' workhorse function for pw_isomap_holes
pw_geod_holes <- function(data, knn, ciso = FALSE, is_dist = FALSE) {

  ## dist knows how to handle missing values and rows are scaled accordingly!
  message(Sys.time(), ": Computing distance matrix")
  if (!is_dist)
    dx <- as.matrix(dist(data))
  else
    dx <- data
  if (!is.matrix(dx))
    stop("data has to be a matrix")

  if (anyNA(dx)) warning("NAs in the distance matrix")

  ## Keep only the knn nearest neighbors in distance matrix, ignore points with
  ## distance zero, later we set these distances to a small value so that
  ## igraph::graph_from_adjacency_matrix will not ignore them
  diag(dx) <- NA
  off_diag_zeros <- dx == 0
  dx[dx == 0] <- NA
  message("Sum off diagonal zeros dx: ", sum(off_diag_zeros, na.rm = TRUE))
  message(Sys.time(), ": Computing knn adjacency matrix")
  for (i in 1:ncol(dx)) {
    ri <- rank(dx[, i], na.last = TRUE, ties.method = "first")
    dx[ri > knn, i] <- NA
  }

  if (ciso) {
    message(Sys.time(), ": CIso, calculate scaling")
    sqrt_mean_knn <- sqrt(colMeans(dx, na.rm = TRUE))

    message("min(sqrt_mean_knn) = ", min(sqrt_mean_knn, na.rm = TRUE))

    if (anyNA(sqrt_mean_knn))
      stop("There are NAs in the kNN means.")

    dx <- dx / (sqrt_mean_knn %o% sqrt_mean_knn)
    message("min(dx) = ", min(dx, na.rm = TRUE))
    if (anyNA(dx))
      warning("There are NAs in the scaled distance matrix.")
  }

  ## igraph::graph_from_adjacency_matrix ignores zero weights
  dx[is.na(dx)] <- 0
  dx[off_diag_zeros] <- 1e-5

  message(Sys.time(), ": Creating graph")
  gx <- graph_from_adjacency_matrix(
    dx, weighted = TRUE,
    mode = "undirected", diag = FALSE)
  if (!igraph::is_connected(gx))
    stop("kNN Graph is not connected, increase knn")

  message(Sys.time(), ": Dijkstra")
  dgx <- igraph::distances(gx, algorithm = "dijkstra")
  if (anyNA(dgx))
    stop("There are NAs in the final distance matrix, ",
         "this should not have happened, something went wrong.")

  return(dgx)
}

#' construct a piecewise Isomap
#'
#' This version pieces together the kNN graphs from a single data frame which
#' contains missing values.
#'
#' @param data a data.frames or data.tables
#' @param id_cols a character vector with the column names, these columns are
#'   ignored for the calculation of the distance matrix frames.
#' @param knn the number of nearest neighbors for the Isomap
#'
#' @import igraph
#' @import RSpectra
#'
#' @section TODOS
#'
pw_isomap_holes <- function(data, knn, out_dim = 2, id_cols = character(0),
                            ciso = FALSE, regularize = 0, is_dist = FALSE) {

  dgx <- pw_geod_holes(
    if (is_dist) data else data[, which(!colnames(data) %in% id_cols)],
    knn = knn, ciso = ciso, is_dist = is_dist)

  if (regularize > 0) {
    message(Sys.time(), ": Regularizing, ", regularize)
    dgx <- dgx + regularize
    diag(dgx) <- 0
  }

  message(Sys.time(), ": classical scaling")
  dgx2 <- dgx ^ 2
  ## This does everything except multiplying with -0.5
  dgx2 <- .Call(stats:::C_DoubleCentre, dgx2)
  dgx2 <- - dgx2 / 2
  dgx2 <- (dgx2 + t(dgx2)) / 2
  message("Symmetic: ", isSymmetric(dgx2))

  ## There is an issue with the last couple of eigenvalues, see
  ## https://github.com/yixuan/RSpectra/issues/7#issuecomment-368921563
  ## therefore we calculate more eigenvalues and discard the last ones.
  e <- RSpectra::eigs_sym(dgx2, out_dim, which = "LA",
                          opts = list(tol = 1e-18, retvec = TRUE))

  print(e$values)
  if(any(e$values < 0))
    stop("Eigenvalues < 0, need to calculate more Eigenvalues")

  y <- e$vectors * rep(sqrt(e$values), each = nrow(e$vectors))

  colnames(y) <- paste0("iso", seq_len(ncol(y)))

  message(Sys.time(), ": DONE")

  return(list(y = y, d_geo = dgx))
}

## length(E(gx))


## dat <- loadDataSet("Swiss") %>% getData

## dat1 <- dat[-1:-666,]
## dat2 <- dat[-1333:-2000,]

## dat1 <- dat2 <- dat
## dat1[1:666, ] <- NA
## dat2[1333:2000, ] <- NA
## dat3 <- cbind(dat1, dat2)
## image(dat3, useRaster = TRUE)
## d3 <- as.matrix(dist(dat3))
## image(d3, useRaster = TRUE)


## y3 <- pw_isomap_holes(dat3, knn = 8, out_dim = 6, ciso = FALSE)
## pairs(dat3)
## pairs(y3$y)

## y4 <- dimred_pw_iso(dat3, knn = 8, ciso = FALSE)
## quality_pw_iso(d_geo = y4$d_geo, y = y4$dim_red)

## 1 - (cor(as.dist(y3$d_geo), dist(y3$y[,1:2]), use = "pairwise.complete.obs") ^ 2)

## data <- dat3
## id_cols <- character(0)
## knn <- 10


## x <- matrix(1:9, 3, 3)
## x[1, 1] <- NA
## x
## dist(x) %>% as.matrix

## dim(dat3)


## dat1 <- as.data.frame(dat1)
## dat1$id <- (1:2000)[-1:-666]
## dat2 <- as.data.frame(dat2)
## dat2$id <- (1:2000)[-1333:-2000]

## res <- pw_isomap_step_across(list(dat1, dat2), "id", 10)
## res <- pw_isomap_2(list(dat1, dat2), "id", 5)

## data <- list(dat1, dat2)
## id_cols <- "id"
## knn <- 10
## res <- cmdscale(dgeo, k = 6)

## pairs(res)
## cmdscale(vegan::isomapdist(dist(cbind(dat, dat)), k = 10), k = 6) %>% pairs

## image(adj)
## image(dgeo)







## data <- lapply(1:5, function(i) {
##   is <- iris[sample(1:150, 40), ]
##   is$id <- rownames(is)
##   is
## })
## id_cols <- c("Species", "id")
## knn <- 15
## pw_isomap_1(data, id_cols, 5) %>% plot
## pw_isomap_2(data, id_cols, 15) %>% plot
