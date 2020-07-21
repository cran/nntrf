# Project Title

*nntrf*: Neural Network based Transformations: Supervised Data Transformation by Means of Neural Network Hidden Layers

## Getting Started

The goal of *nntrf* is to transform datasets from its original space to the space defined by the activations of the hidden layer of a 3-layer Multi-layer Perceptron. This is done by training a neural network and then computing the activations of the neural network for each input pattern. It is a supervised transformation because it results from solving a supervised problem, as opposed to Principal Component Analysis.

The following example shows how to transform the iris dataset, from the original 4-dimension space, into a 2-dimension space by means of *nntrf*.

```
  iris <- NULL
  data("iris", envir = environment())
  rd <- iris
  n <- nrow(rd)
  
  set.seed(1)
  training_index <- sample(1:n, round(0.6*n))
  
  # Get training and test data
  train <- rd[training_index,]
  test <- rd[-training_index,]
  x_train <- as.matrix(train[,-ncol(train)])
  y_train <- train[,ncol(train)]
  x_test <- as.matrix(test[,-ncol(test)])
  y_test <- test[,ncol(test)] 
  
  # Now, use standard KNN to classify the iris dataset on the original data space
  set.seed(1)
  outputs <- FNN::knn(x_train, x_test, train$Species)
  success <- mean(outputs == test$Species)
  cat(paste0("Success rate of KNN (K=1) with iris ", success, "\n"))
  
  # Now, use nntrf to transform the 4-dim space into a 2-dim space (size=2)
  # First, we train the neural network with 2 hidden neurons
  set.seed(1)
  nnpo <- nntrf(formula=Species ~. ,
                data=train,
                size=2, maxit=140, trace=TRUE)
  
  # Second, we transform the dataset using the weights of the hidden layer
  trf_x_train <- nnpo$trf(x=x_train,use_sigmoid=FALSE)
  trf_x_test <- nnpo$trf(x=x_test,use_sigmoid=FALSE)
  
  # Third, KNN is used to classify the dataset on the transformed space
  outputs <- FNN::knn(trf_x_train, trf_x_test, train$Species)
  success <- mean(outputs == test$Species)
  cat(paste0("Success rate of KNN (K=1) with iris transformed by nntrf ", success, "\n"))
```



