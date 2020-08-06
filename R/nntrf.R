#' @title{nntrf: a supervised transformation function based on neural networks (Neural Net based Transformations)}
#'
#' @description{This function transforms a dataset into the activations of the neurons of the hidden layer of a neural network. 
#' This is done by training a neural network and then computing the activations of the neural network for each input pattern. 
#' It uses the \strong{nnet} package under the hood.
#' }
#' @param keep_nn (default TRUE) Keep NN model. In most cases, the actual NN and associated results obtained by nnet is not required
#' @param repetitions (default 1) Repeat nnet several times with different random seeds and select the best run using nnet's minimum \emph{value}. This is useful because some random initialization of weights may lead to local minima.
#' @param random_seed (default NULL)
#' @param ... See \code{\link{nnet}} params. Most important: \itemize{
#' \item formula 
#' \item data :dataframe with data. The last column \strong{must} be the dependent variable. The dependent variable \strong{must} be a factor for classification problems and numeric for regression problems. The input/independent variables \strong{must} contain numeric values only.
#' \item size :number of units in the hidden layer.  
#' \item maxit : the number of iterations of the net.
#' }
#' @return list of: \itemize{
#'   \item trf: a function that transforms the input dataset using the weights of the hidden layer. This function has three arguments:
#'     \itemize{
#'       \item x: the input numeric \strong{matrix} or \strong{data.frame} to be transformed. Only numeric values.
#'       \item use_sigmoid (default TRUE): Whether the sigmoid function should be used for the transformation. nnet uses the sigmoid in the hidden layer, but in some cases better results could be obtained with use_sigmoid=FALSE.
#'       \item norm (default FALSE): If TRUE, this function's output is normalized (scaled) to range 0-1.
#'     }
#'   \item mod: values returned by nnet (this includes the neural network trained by nnet). If keep_nn=FALSE, then NULL is returned here. 
#'   \item matrix1: weights of hidden layer 
#'   \item matrix2: weights of output layer 
#'   }
#' @seealso \code{\link{nnet}}
#' @importFrom nnet nnet
#' @importFrom NeuralNetTools neuralweights
#' @importFrom pracma sigmoid
#' @importFrom stats runif
#' @export
#' @examples
#  Check also nntrf_doughnut function for examples
#  For multi-class classification, check nntrf_iris function
#' data("doughnutRandRotated")
#' rd <- doughnutRandRotated
#' 
#' # Make sure it is a classification problem
#' rd[,ncol(rd)] <- as.factor(rd[,ncol(rd)])
#' n <- nrow(rd)
#' 
#' # Split data into training and test
#' set.seed(0)
#' training_index <- sample(1:n, round(0.6*n))
#' train <- rd[training_index,]
#' test <- rd[-training_index,]
#' x_train <- train[,-ncol(train)]
#' y_train <- train[,ncol(train)]
#' x_test <- test[,-ncol(test)]
#' y_test <- test[,ncol(test)]
#' 
#' # Train nntrf transformation
#' set.seed(0)
#' nnpo <- nntrf(formula=V11~.,
#'               data=train,
#'               size=4, maxit=50, trace=TRUE)
#'               
#' # Apply nntrf transformation to the train and test splits               
#' trf_x_train <- nnpo$trf(x=x_train, use_sigmoid=FALSE)
#' trf_x_test <- nnpo$trf(x=x_test, use_sigmoid=FALSE)      
#' 
#' # Compute the success rate of KNN on the transformed feature space                   
#' outputs <- FNN::knn(trf_x_train, trf_x_test, y_train)
#' success <- mean(outputs == y_test)
#' print(success)
nntrf <- function(keep_nn=TRUE, repetitions=1, random_seed=NULL, ...){

  normalize <- function(x) { 
    minAttr=apply(x, 2, min)
    maxAttr=apply(x, 2, max)
    x <- sweep(x, 2, minAttr, FUN="-") 
    diffe <-  maxAttr-minAttr
    diffe[diffe==0] <- 1
    x=sweep(x, 2,  maxAttr-minAttr, "/") 
    return (x)
  } 
  
#  dots <- list(...)
#  inputs <- ncol(dots$data)-1
#  outputs <- length(unique())
  
#  MaxNWts	<- 
  # stopifnot(is.matrix(list(...)$x))
  if(!is.null(random_seed)){set.seed(random_seed)}

  inputs <- list(...)
  data <- inputs$data
  size <- inputs$size
  
  rang <- inputs$rang
  if(is.null(rang)) rang <- 1
  

  # Note: Only classification, no complete support for regression yet.  
  
  if(is.factor(data[,ncol(data)])){
    # Classification
    num_classes <- length(unique(data[,ncol(data)]))
  } else {
    # Regression
    # warning("Use of nntrf for regression is still experimental.")
    num_classes <- 1
  }
  num_inputs <- (dim(data)[2]-1)
  nweigths_hidden <- (num_inputs+1)*size
  
  if(num_classes <= 2){
    num_outputs <- 1
  } else {
    num_outputs <- num_classes
  }
  
  MaxNWts <- nweigths_hidden +(size+1)*num_outputs
  
  mod <- if(MaxNWts>1000){
    nnet::nnet.formula(...,MaxNWts = MaxNWts)    
  } else {
    nnet::nnet.formula(...)
  }
  
  
  wts <- NeuralNetTools::neuralweights(mod)
  struct <- wts$struct
  value <- mod$value
  
  repetitions <- repetitions - 1
  
  while(repetitions>0){
      mod_rep <- if(MaxNWts>1000){
          nnet::nnet.formula(...,MaxNWts = MaxNWts)    
      } else {
          nnet::nnet.formula(...)
      }
      
      if(mod_rep$value < value){
        mod <- mod_rep
        wts <- NeuralNetTools::neuralweights(mod)
        struct <- wts$struct
        value <- mod$value
      }
      repetitions <- repetitions - 1
  }

  matrix1 <- as.matrix(as.data.frame(wts$wts[1:struct[2]]))
  matrix2 <- as.matrix(as.data.frame(wts$wts[(1+struct[2]):length(wts$wts)]))

  #computed_output <- sigmoid(cbind(1,sigmoid(cbind(1,x) %*% matrix1)) %*% matrix2)
  #net_output <- compute(mod, x)

  output <- list(trf = function(x,use_sigmoid=TRUE, norm=FALSE) {
                         x <- as.matrix(x)
                         result <- cbind(1,x) %*% matrix1
                         #result <- x %*% matrix1[-1,]
                         if(use_sigmoid) {result <- pracma::sigmoid(result)}
                         if(norm) {result <- normalize(result)}
                         return(result)
                       },
                 mod = if(keep_nn) {mod} else {NULL},
                 #mod = mod,
                 matrix1 = matrix1,
                 matrix2 = matrix2)

  return( output )
}


#' Doughnut shaped dataset
#'
#' Doughnut-shaped two-class 2-dimensinoal classification problem
#' V1 and V2 are the input features, normalized to 0-1
#' V3 is the output (TRUE / FALSE)
#' @docType data
#' @usage data(doughnut)
#' @keywords datasets
#' @examples 
#' data(doughnut)
#' plot(doughnut$V1, doughnut$V2, col=doughnut$V3)
"doughnut"

#' Doughnut shaped dataset with 8 extra random attributes
#'
#' V1 to V8 are the extra random features (in the range 0-1)
#' V9 and V10 are the original input features, normalized to 0-1
#' V11 is the output (TRUE / FALSE)
#' @docType data
#' @usage data(doughnutRand)
#' @keywords datasets
#' @examples 
#' data(doughnutRand)
#' plot(doughnutRand$V9, doughnutRand$V10, col=doughnutRand$V11)
"doughnutRand"

#' Doughnut shaped dataset with 8 extra random attributes and rotated
#'
#' doughnutRandRotated randomly rotated
#' @docType data
#' @usage data(doughnutRandRotated)
#' @keywords datasets
"doughnutRandRotated"

#' @title{nntrf_doughnut: a benchmark of three doughnut datasets}
#'
#' @description{This function compares KNN with data untransformed vs. data transformed with nntrf.
#' In order to see what it does, check the code: nntrf::nntrf_doughnut
#' }
#' @param verbose (default TRUE) Print results to the console.
#' @return NULL
#' @return list of: \itemize{
#'   \item Accuracies of KNN on the Doughnut Dataset: with no nntrf, with nntrf and no sigmoid, and with nntrf and no sigmoid and 5 repetitions
#'   \item Accuracies of KNN on the DoughnutRand Dataset: with no nntrf, with nntrf and no sigmoid, and with nntrf and no sigmoid and 5 repetitions
#'   \item Accuracies of KNN on the DoughnutRandRotated Dataset: with no nntrf, with nntrf and no sigmoid, and with nntrf and no sigmoid and 5 repetitions
#' }
#' @importFrom utils data
#' @export
nntrf_doughnut <- function(verbose=TRUE){

  nntrf_train_test <- function(formula, rd, training_index){
    # This function compares KNN with data untransformed vs. data transformed with nntrf.
    # param rd data matrix
    # param training_index indices to be used for training
    # importFrom FNN knn
    
    train <- rd[training_index,]
    test <- rd[-training_index,]
    x_train <- train[,-ncol(train)]
    y_train <- train[,ncol(train)]
    x_test <- test[,-ncol(test)]
    y_test <- test[,ncol(test)] 
    
    x_train <- as.matrix(x_train)
    x_test <- as.matrix(x_test)
    
    set.seed(0)
    outputs <- FNN::knn(x_train, x_test, y_train)
    success <- mean(outputs == y_test)
    if(verbose) {cat(paste0("Without nntrf ", success, "\n"))}
    no_nntrf <- success
    
    set.seed(0)

    nnpo <- nntrf::nntrf(formula=formula, data=train,
                         size=5, maxit=100, trace=FALSE)
    outputs <- FNN::knn(nnpo$trf(x=x_train,use_sigmoid=FALSE), nnpo$trf(x=x_test,use_sigmoid=FALSE), y_train)
    success <- mean(outputs == y_test)
    nntrf_nosigmoid = success
    
    if(verbose){cat(paste0("With nntrf (sigmoid=FALSE) ", success, "\n"))}
    
    nnpo <- nntrf::nntrf(formula=formula, data=train,
                         size=5, maxit=100, trace=FALSE, repetitions=5)
    outputs <- FNN::knn(nnpo$trf(x=x_train,use_sigmoid=FALSE), nnpo$trf(x=x_test,use_sigmoid=FALSE), y_train)
    success <- mean(outputs == y_test)
    nntrf_nosigmoid_5reps = success
    
    if(verbose){cat(paste0("With nntrf (sigmoid=FALSE) and 5 repetitions ", success, "\n"))}
    
    return(c(no_nntrf=no_nntrf, nntrf_nosigmoid=nntrf_nosigmoid, nntrf_nosigmoid_5reps=nntrf_nosigmoid_5reps))
  }

  doughnut <- NULL
  doughnutRand <- NULL
  doughnutRandRotated <- NULL
  
  data("doughnut", envir = environment())
  rd <- doughnut
  rd$V3 <- as.factor(rd$V3)
  n <- nrow(rd)
  
  # Use 60% of data for training 40% for testing
  set.seed(0)
  training_index <- sample(1:n, round(0.6*n))
  if(verbose){cat(paste0("Dataset Doughnut \n"))}
  results_doughnut <- nntrf_train_test(V3~., rd, training_index)

  data("doughnutRand", envir = environment())
  rd <- doughnutRand  
  rd$V11 <- as.factor(rd$V11)
  # Use the same training instances as for doughnut
  if(verbose){cat(paste0("Dataset DoughnutRand \n"))}
  results_doughnut_rand <- nntrf_train_test(V11~., rd, training_index)
  
  data("doughnutRandRotated", envir = environment())
  rd <- doughnutRandRotated
  rd$V11 <- as.factor(rd$V11)
  # Use the same training instances as for doughnut
  if(verbose){cat(paste0("Dataset DoughnutRandRotated \n"))}
  results_doughnut_rand_rotated <- nntrf_train_test(V11~., rd, training_index)
  
  return(list(doughnut=results_doughnut, 
              doughnutRand=results_doughnut_rand, 
              doughnutRandRotated=results_doughnut_rand_rotated))
    
}

#' @title{nntrf_iris: a benchmark for the iris dataset}
#'
#' @description{This function compares KNN with data untransformed vs. data transformed with nntrf.
#' In order to see what it does, check the code: nntrf::nntrf_iris
#' }
#' @param verbose (default TRUE) Print results to the console.
#' @return Accuracies of KNN on the Iris Dataset: with no nntrf, and with nntrf and no sigmoid
#' @export
nntrf_iris <- function(verbose=TRUE){
  iris <- NULL
  data("iris", envir = environment())
  rd <- iris
  n <- nrow(rd)
  
  set.seed(0)
  training_index <- sample(1:n, round(0.6*n))
  
  train <- rd[training_index,]
  test <- rd[-training_index,]
  x_train <- as.matrix(train[,-ncol(train)])
  y_train <- train[,ncol(train)]
  x_test <- as.matrix(test[,-ncol(test)])
  y_test <- test[,ncol(test)] 
  
  set.seed(0)
  outputs <- FNN::knn(x_train, x_test, train$Species)
  success <- mean(outputs == test$Species)
  no_nntrf = success
  
  if(verbose){cat(paste0("Success rate of KNN (K=1) with iris ", success, "\n"))}
  
  set.seed(0)
  nnpo <- nntrf::nntrf(formula=Species ~. ,
                data=train,
                size=2, maxit=140, trace=verbose)
  
  trf_x_train <- nnpo$trf(x=x_train,use_sigmoid=FALSE)
  trf_x_test <- nnpo$trf(x=x_test,use_sigmoid=FALSE)
  
  outputs <- FNN::knn(trf_x_train, trf_x_test, train$Species)
  success <- mean(outputs == test$Species)
  nntrf_nosigmoid <- success
  
  if(verbose){cat(paste0("Success rate of KNN (K=1) with iris transformed by nntrf with 2 hidden neurons ", success, "\n"))}
  return(c(no_nntrf=no_nntrf, nntrf_nosigmoid=nntrf_nosigmoid))
}



### OLD CODE

# nntrf <- function(keep_nn=TRUE, repetitions=1, random_seed=NULL, xavier_ini=FALSE, orthog_ini=FALSE, ...){
#   normalize <- function(x) { 
#     minAttr=apply(x, 2, min)
#     maxAttr=apply(x, 2, max)
#     x <- sweep(x, 2, minAttr, FUN="-") 
#     diffe <-  maxAttr-minAttr
#     diffe[diffe==0] <- 1
#     x=sweep(x, 2,  maxAttr-minAttr, "/") 
#     return (x)
#   } 
#   
#   #  dots <- list(...)
#   #  inputs <- ncol(dots$data)-1
#   #  outputs <- length(unique())
#   
#   #  MaxNWts	<- 
#   # stopifnot(is.matrix(list(...)$x))
#   if(!is.null(random_seed)){set.seed(random_seed)}
#   
#   inputs <- list(...)
#   data <- inputs$data
#   size <- inputs$size
#   
#   rang <- inputs$rang
#   if(is.null(rang)) rang <- 1
#   
#   
#   # Note: Only classification, no complete support for regression yet.  
#   
#   if(is.factor(data[,ncol(data)])){
#     # Classification
#     num_classes <- length(unique(data[,ncol(data)]))
#   } else {
#     # Regression
#     warning("Use of nntrf for regression is still experimental.")
#     num_classes <- 1
#   }
#   num_inputs <- (dim(data)[2]-1)
#   nweigths_hidden <- (num_inputs+1)*size
#   
#   if(num_classes <= 2){
#     num_outputs <- 1
#   } else {
#     num_outputs <- num_classes
#   }
#   
#   MaxNWts <- nweigths_hidden +(size+1)*num_outputs
#   
#   
#   # XAVIER INITIALIZATION
#   if(xavier_ini | orthog_ini){
#     a <- rang*4*sqrt(6/(num_inputs+size))
#     capa1 <- runif(nweigths_hidden, -a, +a)
#     
#     if(orthog_ini){
#       capa1 <- matrix(capa1, nrow = size, byrow = TRUE)
#       mat_ortho <- pracma::randortho(num_inputs, type = c("orthonormal"))
#       capa1[,2:ncol(capa1)] <- mat_ortho[1:size,]
#       capa1 <-  as.vector(t(capa1))
#       # Maximum value to be a
#       multiplier <- runif(1,min=0,max=a)
#       if(multiplier==0) multiplier <- 1
#       capa1 <- capa1/max(abs(capa1))*multiplier
#     }
#     
#     a <- rang*4*sqrt(6/(size+num_outputs))
#     capa2 <- runif(MaxNWts-nweigths_hidden, -a, +a)
#     wts_b <- c(capa1, capa2)
#   }
#   
#   mod <- if(MaxNWts>1000){
#     if(xavier_ini | orthog_ini){
#       nnet::nnet.formula(...,MaxNWts = MaxNWts, Wts=wts_b)
#     } else {
#       nnet::nnet.formula(...,MaxNWts = MaxNWts)    
#     }
#   } else {
#     if(xavier_ini | orthog_ini){
#       nnet::nnet.formula(..., Wts=wts_b)
#     } else {
#       nnet::nnet.formula(...)
#     }  
#   }
#   
#   
#   wts <- NeuralNetTools::neuralweights(mod)
#   struct <- wts$struct
#   value <- mod$value
#   
#   repetitions <- repetitions - 1
#   
#   while(repetitions>0){
#     # XAVIER INITIALIZATION
#     if(xavier_ini | orthog_ini){
#       a <- rang*4*sqrt(6/(num_inputs+size))
#       capa1 <- runif(nweigths_hidden, -a, +a)
#       
#       if(orthog_ini){
#         capa1 <- matrix(capa1, nrow = size, byrow = TRUE)
#         mat_ortho <- pracma::randortho(num_inputs, type = c("orthonormal"))
#         capa1[,2:ncol(capa1)] <- mat_ortho[1:size,]
#         capa1 <-  as.vector(t(capa1))
#         # Maximum value to be a
#         multiplier <- runif(1,min=0,max=a)
#         if(multiplier==0) multiplier <- 1
#         capa1 <- capa1/max(abs(capa1))*multiplier
#       }
#       
#       a <- rang*4*sqrt(6/(size+num_outputs))
#       capa2 <- runif(MaxNWts-nweigths_hidden, -a, +a)
#       wts_b <- c(capa1, capa2)
#     }
#     
#     
#     mod_rep <- if(MaxNWts>1000){
#       if(xavier_ini | orthog_ini){
#         nnet::nnet.formula(...,MaxNWts = MaxNWts, Wts=wts_b)
#       } else {
#         nnet::nnet.formula(...,MaxNWts = MaxNWts)    
#       }
#     } else {
#       if(xavier_ini | orthog_ini){
#         nnet::nnet.formula(..., Wts=wts_b)
#       } else {
#         nnet::nnet.formula(...)
#       }  
#     }
#     
#     if(mod_rep$value < value){
#       mod <- mod_rep
#       wts <- NeuralNetTools::neuralweights(mod)
#       struct <- wts$struct
#       value <- mod$value
#     }
#     repetitions <- repetitions - 1
#   }
#   
#   matrix1 <- as.matrix(as.data.frame(wts$wts[1:struct[2]]))
#   matrix2 <- as.matrix(as.data.frame(wts$wts[(1+struct[2]):length(wts$wts)]))
#   
#   #computed_output <- sigmoid(cbind(1,sigmoid(cbind(1,x) %*% matrix1)) %*% matrix2)
#   #net_output <- compute(mod, x)
#   
#   output <- list(trf = function(x,use_sigmoid=TRUE, norm=FALSE) {
#     x <- as.matrix(x)
#     result <- cbind(1,x) %*% matrix1
#     #result <- x %*% matrix1[-1,]
#     if(use_sigmoid) {result <- pracma::sigmoid(result)}
#     if(norm) {result <- normalize(result)}
#     return(result)
#   },
#   #mod = if(keep_nn) {mod} else {NULL},
#   mod = mod,
#   matrix1 = matrix1,
#   matrix2 = matrix2)
#   
#   return( output )
# }
# 
