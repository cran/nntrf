## ----setup, include=FALSE------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>"
)

## ------------------------------------------------------------------------
library(nntrf)
library(mlr)
library(mlrCPO)
library(FNN)

## ------------------------------------------------------------------------
x <- seq(-10,10,length.out = 5000)
y <- sin(x)
extra <- matrix(runif(5000*9),nrow = 5000)
data <- cbind(x, as.data.frame(extra), y)
plot(data$x,data$y)

## ------------------------------------------------------------------------
m <- as.matrix(cbind(x, extra))
set.seed(0)
mat_ortho <- pracma::randortho(ncol(m), type = c("orthonormal"))
m_p <- m %*% mat_ortho
data <- cbind(as.data.frame(m_p), y)


## ------------------------------------------------------------------------
data <- mlr::normalizeFeatures(data, method="range")

## ------------------------------------------------------------------------
rd <- data
n <- nrow(rd)

set.seed(0)
training_index <- sample(1:n, round(0.6*n))
  
train <- rd[training_index,]
test <- rd[-training_index,]
x_train <- train[,-ncol(train)]
y_train <- train[,ncol(train)]
x_test <- test[,-ncol(test)]
y_test <- test[,ncol(test)] 

## ------------------------------------------------------------------------
set.seed(0)
outputs <- FNN::knn.reg(as.data.frame(x[training_index]), as.data.frame(x[-training_index]), y_train, k=3)
mae_error <- mean(abs(outputs$pred - y_test))
cat(paste0("MAE of KNN (K=3):", mae_error, "\n"))

## ------------------------------------------------------------------------
set.seed(0)
outputs <- FNN::knn.reg(x_train, x_test, y_train, k=3)
mae_error <- mean(abs(outputs$pred - y_test))
cat(paste0("MAE of KNN (K=3):", mae_error, "\n"))

## ------------------------------------------------------------------------
set.seed(0)
nnpo <- nntrf(formula=y~.,
              data=train,
              size=1, maxit=500, trace=FALSE, repetitions=2)

# With sigmoid

trf_x_train <- nnpo$trf(x=x_train,use_sigmoid=TRUE)
trf_x_test <- nnpo$trf(x=x_test,use_sigmoid=TRUE)

outputs <- FNN::knn.reg(trf_x_train, trf_x_test, y_train, k = 3)
mae <- mean(abs(outputs$pred - y_test))
cat(paste0("MAE of KNN (K=3) transformed by nntrf with Sigmoid: ", mae, "\n"))

# With no sigmoid
trf_x_train <- nnpo$trf(x=x_train,use_sigmoid=FALSE)
trf_x_test <- nnpo$trf(x=x_test,use_sigmoid=FALSE)

outputs <- FNN::knn.reg(trf_x_train, trf_x_test, y_train, k=3)
mae <- mean(abs(outputs$pred - y_test))
cat(paste0("MAE of KNN (K=3) transformed by nntrf with no sigmoid: ", mae, "\n"))




## ------------------------------------------------------------------------
plot(trf_x_test[,1], y_test)

## ------------------------------------------------------------------------
sin_task <- makeRegrTask(data=data, target="y")

control_grid <- makeTuneControlGrid()
inner_desc <- makeResampleDesc("CV", iter=3)
outer_desc <-  makeResampleDesc("CV", iter=3)
set.seed(0)
outer_inst <- makeResampleInstance(outer_desc, sin_task)

## ------------------------------------------------------------------------
cpo_nntrf = makeCPO("nntrfCPO",  
                       # Here, the hyper-parameters of nntrf are defined
                       pSS(repetitions = 1 : integer[1, ],
                           size: integer[1, ],
                           maxit = 100 : integer[1, ],
                           use_sigmoid = FALSE: logical),
                       dataformat = "numeric",
                       cpo.train = function(data, target, 
                                            repetitions, 
                                            size, maxit, use_sigmoid) {
                         data_and_class <- cbind(as.data.frame(data), class=target[[1]])
                         nnpo <- nntrf(repetitions=repetitions,
                                       formula=class~.,
                                       data=data_and_class,
                                       size=size, maxit=maxit, trace=FALSE)
                       },
                       cpo.retrafo = function(data, control, 
                                              repetitions, 
                                              size, maxit, use_sigmoid) {
                       
                         trf_x <- control$trf(x=data,use_sigmoid=use_sigmoid)
                         trf_x
                       })

## ------------------------------------------------------------------------
# knn is the machine learning method. The knn available in the FNN package is used
knn_lrn <- makeLearner("regr.fnn")
# Then, knn is combined with nntrf's preprocessing into a pipeline
knn_nntrf <- cpo_nntrf() %>>% knn_lrn
# Just in case, we fix the values of the hyper-parameters that we do not require to optimize
# (not necessary, because they already have default values. Just to make their values explicit)
knn_nntrf <- setHyperPars(knn_nntrf, nntrfCPO.repetitions=1, nntrfCPO.maxit=100,
                          nntrfCPO.use_sigmoid=FALSE)

# However, we are going to use 2 repetitions here, instead of 1 (the default):

knn_nntrf <- setHyperPars(knn_nntrf, nntrfCPO.repetitions=2)

## ------------------------------------------------------------------------
ps <- makeParamSet(makeDiscreteParam("k", values = 1:7),
                   makeDiscreteParam("nntrfCPO.size", values = 1:5),
                   makeDiscreteParam("nntrfCPO.maxit", values = c(50, 100, 500, 1000, 2000))
)

## ------------------------------------------------------------------------
knn_nntrf_tune <- makeTuneWrapper(knn_nntrf, resampling = inner_desc, par.set = ps, 
                                     control = control_grid, measures = list(mlr::mae), show.info = FALSE)

## ------------------------------------------------------------------------
set.seed(0)
# Please, note that in order to save time, results have been precomputed
cached <- system.file("extdata", "error_knn_nntrf_regression.rda", package = "nntrf")
if(file.exists(cached)){load(cached)} else {
  error_knn_nntrf_tune <- resample(knn_nntrf_tune, sin_task, outer_inst, 
                                   measures = list(mlr::mae), 
                                   extract = getTuneResult, show.info =  FALSE)

  # save(error_knn_nntrf_tune, file="../inst/extdata/error_knn_nntrf_regression.rda")
}

## ------------------------------------------------------------------------
print(error_knn_nntrf_tune$extract)

## ------------------------------------------------------------------------
print(error_knn_nntrf_tune$aggr)

## ------------------------------------------------------------------------
knn_pca <- cpoPca(center=TRUE, scale=TRUE, export=c("rank")) %>>% makeLearner("regr.fnn")

ps_pca <- makeParamSet(makeDiscreteParam("k", values = 1:7),
                       makeDiscreteParam("pca.rank", values = 1:5)
)

knn_pca_tune <- makeTuneWrapper(knn_pca, resampling = inner_desc, par.set = ps_pca, 
                                     control = control_grid, measures = list(mlr::mae), show.info = FALSE)

## ------------------------------------------------------------------------
set.seed(0)
# Please, note that in order to save time, results have been precomputed

cached <- system.file("extdata", "error_knn_pca_tune_regression.rda", package = "nntrf")
if(file.exists(cached)){load(cached)} else {
error_knn_pca_tune <- resample(knn_pca_tune, sin_task, outer_inst, 
                               measures = list(mlr::mae), 
                               extract = getTuneResult, show.info =  FALSE)
#save(error_knn_pca_tune, file="../inst/extdata/error_knn_pca_tune_regression.rda")
}


## ------------------------------------------------------------------------
print(error_knn_pca_tune$extract)

## ------------------------------------------------------------------------
print(error_knn_pca_tune$aggr)

## ------------------------------------------------------------------------
ps_knn <- makeParamSet(makeDiscreteParam("k", values = 1:7))

knn_lrn <- makeLearner("regr.fnn")
knn_tune <- makeTuneWrapper(knn_lrn, resampling = inner_desc, par.set = ps_knn, 
                                     control = control_grid, measures = list(mlr::mae), show.info = FALSE)

set.seed(0)
# Please, note that in order to save time, results have been precomputed
cached <- system.file("extdata", "error_knn_tune_regression.rda", package = "nntrf")
if(file.exists(cached)){load(cached)} else {
error_knn_tune <- resample(knn_tune, sin_task, outer_inst, measures = list(mlr::mae), 
                           extract = getTuneResult, show.info =  FALSE)
#save(error_knn_tune, file="../inst/extdata/error_knn_tune_regression.rda")
}


## ------------------------------------------------------------------------
print(error_knn_tune$aggr)

