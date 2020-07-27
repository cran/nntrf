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
data("doughnutRandRotated")

doughnut_task <- makeClassifTask(data = doughnutRandRotated, target = "V11")
control_grid <- makeTuneControlGrid()
inner_desc <- makeResampleDesc("CV", iter=3)
outer_desc <-  makeResampleDesc("CV", iter=3)
set.seed(0)
outer_inst <- makeResampleInstance(outer_desc, doughnut_task)

## ------------------------------------------------------------------------
cpo_nntrf = makeCPO("nntrfCPO",  
                       # Here, the hyper-parameters of nntrf are defined
                       pSS(repetitions = 1 : integer[1, ],
                           xavier_ini = FALSE : logical,
                           orthog_ini = FALSE : logical,
                           size: integer[1, ],
                           maxit = 100 : integer[1, ],
                           use_sigmoid = FALSE: logical),
                       dataformat = "numeric",
                       cpo.train = function(data, target, 
                                            repetitions, xavier_ini, orthog_ini, 
                                            size, maxit, use_sigmoid) {
                         data_and_class <- cbind(as.data.frame(data), class=target[[1]])
                         nnpo <- nntrf(repetitions=repetitions,
                                       xavier_ini=xavier_ini,
                                       orthog_ini=orthog_ini,
                                       formula=class~.,
                                       data=data_and_class,
                                       size=size, maxit=maxit, trace=FALSE)
                       },
                       cpo.retrafo = function(data, control, 
                                              repetitions, xavier_ini, orthog_ini, 
                                              size, maxit, use_sigmoid) {
                       
                         trf_x <- control$trf(x=data,use_sigmoid=use_sigmoid)
                         trf_x
                       })

## ------------------------------------------------------------------------
# knn is the machine learning method. The knn available in the FNN package is used
knn_lrn <- makeLearner("classif.fnn")
# Then, knn is combined with nntrf's preprocessing into a pipeline
knn_nntrf <- cpo_nntrf() %>>% knn_lrn
# Just in case, we fix the values of the hyper-parameters that we do not require to optimize
# (not necessary, because they already have default values. Just to make their values explicit)
knn_nntrf <- setHyperPars(knn_nntrf, nntrfCPO.repetitions=1, nntrfCPO.maxit=100, 
                          nntrfCPO.xavier_ini=FALSE, nntrfCPO.orthog_ini=FALSE,
                          nntrfCPO.use_sigmoid=FALSE)

# However, we are going to use 2 repetitions here, instead of 1 (the default):

knn_nntrf <- setHyperPars(knn_nntrf, nntrfCPO.repetitions=2)

## ------------------------------------------------------------------------
ps <- makeParamSet(makeDiscreteParam("k", values = 1:7),
                   makeDiscreteParam("nntrfCPO.size", values = 1:10)
)

## ------------------------------------------------------------------------
knn_nntrf_tune <- makeTuneWrapper(knn_nntrf, resampling = inner_desc, par.set = ps, 
                                     control = control_grid, measures = list(acc), show.info = FALSE)

## ------------------------------------------------------------------------
set.seed(0)
# Please, note that in order to save time, results have been precomputed
cached <- system.file("extdata", "error_knn_nntrf_tune.rda", package = "nntrf")
if(file.exists(cached)){load(cached)} else {
error_knn_nntrf_tune <- resample(knn_nntrf_tune, doughnut_task, outer_inst, 
                                 measures = list(acc), 
                                 extract = getTuneResult, show.info =  FALSE)
save(error_knn_nntrf_tune, file="../inst/extdata/error_knn_nntrf_tune.rda")
}


## ------------------------------------------------------------------------
print(error_knn_nntrf_tune$extract)

## ------------------------------------------------------------------------
print(error_knn_nntrf_tune$aggr)

## ------------------------------------------------------------------------
library(dplyr)
results_hyper <- generateHyperParsEffectData(error_knn_nntrf_tune)
head(arrange(results_hyper$data, -acc.test.mean))

## ------------------------------------------------------------------------
knn_nntrf <- cpo_nntrf() %>>% makeLearner("classif.fnn")

knn_nntrf <- setHyperPars(knn_nntrf, nntrfCPO.repetitions=2, nntrfCPO.maxit=100,
                          nntrfCPO.xavier_ini=FALSE, nntrfCPO.orthog_ini=FALSE,
                          nntrfCPO.use_sigmoid=FALSE, k=5, nntrfCPO.size=4)

set.seed(0)
# Please, note that in order to save time, results have been precomputed
cached <- system.file("extdata", "error_knn_nntrf.rda", package = "nntrf")
if(file.exists(cached)){load(cached)} else {
  error_knn_nntrf <- resample(knn_nntrf, doughnut_task, outer_inst, measures = list(acc), 
                            show.info =  FALSE)
save(error_knn_nntrf, file="../inst/extdata/error_knn_nntrf.rda")
}



## ------------------------------------------------------------------------
# First, the three evaluations of the outer 3-fold crossvalidation, one per fold:
print(error_knn_nntrf$measures.test)
# Second, their average
print(error_knn_nntrf$aggr)

## ------------------------------------------------------------------------
knn_pca <- cpoPca(center=TRUE, scale=TRUE, export=c("rank")) %>>% knn_lrn

ps_pca <- makeParamSet(makeDiscreteParam("k", values = 1:7),
                       makeDiscreteParam("pca.rank", values = 1:10)
)

knn_pca_tune <- makeTuneWrapper(knn_pca, resampling = inner_desc, par.set = ps_pca, 
                                     control = control_grid, measures = list(acc), show.info = FALSE)

## ------------------------------------------------------------------------
set.seed(0)
# Please, note that in order to save time, results have been precomputed

cached <- system.file("extdata", "error_knn_pca_tune.rda", package = "nntrf")
if(file.exists(cached)){load(cached)} else {
error_knn_pca_tune <- resample(knn_pca_tune, doughnut_task, outer_inst, 
                               measures = list(acc), 
                               extract = getTuneResult, show.info =  FALSE)
save(error_knn_pca_tune, file="../inst/extdata/error_knn_pca_tune.rda")
}


## ------------------------------------------------------------------------
print(error_knn_pca_tune$extract)
print(error_knn_pca_tune$aggr)
results_hyper <- generateHyperParsEffectData(error_knn_pca_tune)
head(arrange(results_hyper$data, -acc.test.mean))

## ------------------------------------------------------------------------

ps_knn <- makeParamSet(makeDiscreteParam("k", values = 1:7))


knn_tune <- makeTuneWrapper(knn_lrn, resampling = inner_desc, par.set = ps_knn, 
                                     control = control_grid, measures = list(acc), show.info = FALSE)

set.seed(0)
# Please, note that in order to save time, results have been precomputed
cached <- system.file("extdata", "error_knn_tune.rda", package = "nntrf")
if(file.exists(cached)){load(cached)} else {
error_knn_tune <- resample(knn_tune, doughnut_task, outer_inst, measures = list(acc), 
                           extract = getTuneResult, show.info =  FALSE)
save(error_knn_tune, file="../inst/extdata/error_knn_tune.rda")
}


## ------------------------------------------------------------------------
print(error_knn_tune$extract)
print(error_knn_tune$aggr)
results_hyper <- generateHyperParsEffectData(error_knn_tune)
head(arrange(results_hyper$data, -acc.test.mean))

## ------------------------------------------------------------------------
# knn is the machine learning method. The knn available in the FNN package is used
knn_lrn <- makeLearner("classif.fnn")
# Then, knn is combined with nntrf's preprocessing into a pipeline
knn_nntrf <- cpo_nntrf() %>>% knn_lrn
# Just in case, we fix the values of the hyper-parameters that we do not require to optimize
# (not necessary, because they already have default values. Just to make their values explicit)
knn_nntrf <- setHyperPars(knn_nntrf, nntrfCPO.repetitions=1, nntrfCPO.maxit=100, 
                          nntrfCPO.xavier_ini=FALSE, nntrfCPO.orthog_ini=FALSE,
                          nntrfCPO.use_sigmoid=FALSE)

# However, we are going to use 2 repetitions here, instead of 1 (the default):

knn_nntrf <- setHyperPars(knn_nntrf, nntrfCPO.repetitions=2, nntrfCPO.xavier_ini=TRUE,
                          nntrfCPO.orthog_ini=FALSE)

## ------------------------------------------------------------------------
ps <- makeParamSet(makeDiscreteParam("k", values = 1:7),
                   makeDiscreteParam("nntrfCPO.size", values = 1:10)
)

## ------------------------------------------------------------------------
knn_nntrf_tune <- makeTuneWrapper(knn_nntrf, resampling = inner_desc, par.set = ps, 
                                     control = control_grid, measures = list(acc), show.info = FALSE)

## ------------------------------------------------------------------------
set.seed(0)
# Please, note that in order to save time, results have been precomputed
cached <- system.file("extdata", "error_knn_nntrf_xavier_tune.rda", package = "nntrf")
if(file.exists(cached)) {load(cached)} else {
error_knn_nntrf_xavier_tune <- resample(knn_nntrf_tune, doughnut_task, outer_inst, 
                                 measures = list(acc), 
                                 extract = getTuneResult, show.info =  FALSE)
save(error_knn_nntrf_xavier_tune, file="../inst/extdata/error_knn_nntrf_xavier_tune.rda")
}


## ------------------------------------------------------------------------
print(error_knn_nntrf_xavier_tune$extract)

## ------------------------------------------------------------------------
print(error_knn_nntrf_xavier_tune$aggr)

## ------------------------------------------------------------------------
library(dplyr)
results_hyper <- generateHyperParsEffectData(error_knn_nntrf_xavier_tune)
head(arrange(results_hyper$data, -acc.test.mean))

## ------------------------------------------------------------------------
knn_nntrf <- setHyperPars(knn_nntrf, nntrfCPO.repetitions=2, 
                          nntrfCPO.xavier_ini=TRUE,
                          nntrfCPO.orthog_ini=TRUE)
knn_nntrf_tune <- makeTuneWrapper(knn_nntrf, resampling = inner_desc, par.set = ps, 
                                     control = control_grid, measures = list(acc), show.info = FALSE)

## ------------------------------------------------------------------------
set.seed(0)
# Please, note that in order to save time, results have been precomputed
cached <- system.file("extdata", "error_knn_nntrf_xavier_orthog_tune.rda", package = "nntrf")
if(file.exists(cached)) {load(cached)} else {
error_knn_nntrf_xavier_tune <- resample(knn_nntrf_tune, doughnut_task, outer_inst, 
                                 measures = list(acc), 
                                 extract = getTuneResult, show.info =  FALSE)
save(error_knn_nntrf_xavier_tune, file="../inst/extdata/error_knn_nntrf_xavier_orthog_tune.rda")
}

## ------------------------------------------------------------------------
print(error_knn_nntrf_xavier_tune$extract)
print(error_knn_nntrf_xavier_tune$aggr)
library(dplyr)
results_hyper <- generateHyperParsEffectData(error_knn_nntrf_xavier_tune)
head(arrange(results_hyper$data, -acc.test.mean))

