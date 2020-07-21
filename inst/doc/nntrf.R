## ----setup, include = FALSE----------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>"
)

## ------------------------------------------------------------------------
library(dplyr)
library(tidyr)
library(ggplot2)
library(ggridges)
library(nntrf)

## ------------------------------------------------------------------------
data("doughnut")
plot(doughnut$V1, doughnut$V2, col=2+doughnut$V3)

## ------------------------------------------------------------------------
head(doughnutRandRotated,5)

## ------------------------------------------------------------------------
data("doughnutRandRotated")

rd <- doughnutRandRotated
rd$V11 <- as.factor(rd$V11)
n <- nrow(rd)

set.seed(0)
training_index <- sample(1:n, round(0.6*n))
  
train <- rd[training_index,]
test <- rd[-training_index,]
x_train <- train[,-ncol(train)]
y_train <- train[,ncol(train)]
x_test <- test[,-ncol(test)]
y_test <- test[,ncol(test)] 

set.seed(1)
outputs <- FNN::knn(x_train, x_test, factor(y_train))
success <- mean(outputs == y_test)
cat(paste0("Success rate of KNN (K=1) with doughnutRandRotated ", success, "\n"))

set.seed(1)
nnpo <- nntrf(formula=V11~.,
              data=train,
              size=4, maxit=100, trace=FALSE)

trf_x_train <- nnpo$trf(x=x_train,use_sigmoid=FALSE)
trf_x_test <- nnpo$trf(x=x_test,use_sigmoid=FALSE)

outputs <- FNN::knn(trf_x_train, trf_x_test, factor(y_train))
success <- mean(outputs == y_test)
cat(paste0("Success rate of KNN (K=1) with doughnutRandRotated transformed by nntrf ", success, "\n"))

## ------------------------------------------------------------------------
plot(trf_x_train[,1], trf_x_train[,4], col=y_train)

## ------------------------------------------------------------------------
set.seed(0)
nnpo <- nntrf(repetitions=5,
              formula=V11~.,
              data=train,
              size=4, maxit=100, trace=FALSE)

trf_x_train <- nnpo$trf(x=x_train,use_sigmoid=FALSE)
trf_x_test <- nnpo$trf(x=x_test,use_sigmoid=FALSE)

outputs <- FNN::knn(trf_x_train, trf_x_test, factor(y_train))
success <- mean(outputs == y_test)
cat(paste0("Success rate of KNN (K=1) with doughnutRandRotated transformed by nntrf ", success, "\n"))

## ------------------------------------------------------------------------
rd <- iris
n <- nrow(rd)

set.seed(1)
training_index <- sample(1:n, round(0.6*n))
  
train <- rd[training_index,]
test <- rd[-training_index,]
x_train <- as.matrix(train[,-ncol(train)])
y_train <- train[,ncol(train)]
x_test <- as.matrix(test[,-ncol(test)])
y_test <- test[,ncol(test)]

set.seed(1)
outputs <- FNN::knn(x_train, x_test, train$Species)
success <- mean(outputs == test$Species)
cat(paste0("Success rate of KNN (K=1) with iris ", success, "\n"))

set.seed(1)
nnpo <- nntrf(formula = Species~.,
              data=train,
              size=2, maxit=100, trace=FALSE)

trf_x_train <- nnpo$trf(x=x_train,use_sigmoid=FALSE)
trf_x_test <- nnpo$trf(x=x_test,use_sigmoid=FALSE)

outputs <- FNN::knn(trf_x_train, trf_x_test, train$Species)
success <- mean(outputs == test$Species)
cat(paste0("Success rate of KNN (K=1) with iris transformed by nntrf ", success, "\n"))

