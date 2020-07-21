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
set.seed(1)
outer_inst <- makeResampleInstance(outer_desc, doughnut_task)

## ------------------------------------------------------------------------
cpo_nntrf = makeCPO("nntrfCPO",  
                       # Here, the hyper-parameters of nntrf are defined
                       pSS(size: integer[1, ],
                           repetitions = 1 : integer[1, ],
                           maxit = 100 : integer[1, ],
                           use_sigmoid = FALSE: logical),
                       dataformat = "numeric",
                       cpo.train = function(data, target, size, repetitions, maxit, use_sigmoid) {
                         nnpo <- nntrf(repetitions=repetitions,
                                       formula=target[[1]]~.,
                                       data=data,
                                       size=size, maxit=maxit, trace=FALSE)
                       },
                       cpo.retrafo = function(data, control, size, repetitions, maxit, use_sigmoid) {
                       
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
knn_nntrf <- setHyperPars(knn_nntrf, nntrfCPO.repetitions=1, nntrfCPO.maxit=100, nntrfCPO.use_sigmoid=FALSE)

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
set.seed(1)
# Please, note that in order to save time, results have been precomputed
if(file.exists("../inst/error_knn_nntrf_tune.rda")){load("../inst/error_knn_nntrf_tune.rda")} else {
error_knn_nntrf_tune <- resample(knn_nntrf_tune, doughnut_task, outer_inst, 
                                 measures = list(acc), 
                                 extract = getTuneResult, show.info =  FALSE)
#save(error_knn_nntrf_tune, file="../inst/error_knn_nntrf_tune.rda")
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

knn_nntrf <- setHyperPars(knn_nntrf, nntrfCPO.repetitions=2, nntrfCPO.maxit=100, nntrfCPO.use_sigmoid=FALSE, k=7, nntrfCPO.size=4)

set.seed(1)
# Please, note that in order to save time, results have been precomputed
if(file.exists("../inst/error_knn_nntrf.rda")){load("../inst/error_knn_nntrf.rda")} else {
error_knn_nntrf <- resample(knn_nntrf, doughnut_task, outer_inst, measures = list(acc), 
                            show.info =  FALSE)
#save(error_knn_nntrf, file="../inst/error_knn_nntrf.rda")
}



## ------------------------------------------------------------------------
# First, the three evaluations of the outer 3-fold crossvalidation, one per fold:
print(error_knn_nntrf$measures.test)
# Second, their average
print(error_knn_nntrf$aggr)

## ------------------------------------------------------------------------
knn_lrn <- makeLearner("classif.fnn")
knn_nntrf <- cpo_nntrf() %>>% knn_lrn
knn_nntrf <- setHyperPars(knn_nntrf, nntrfCPO.repetitions=1, nntrfCPO.maxit=100, nntrfCPO.use_sigmoid=FALSE)
knn_nntrf <- setHyperPars(knn_nntrf, nntrfCPO.repetitions=5)

ps <- makeParamSet(makeDiscreteParam("k", values = 1:7),
                   makeDiscreteParam("nntrfCPO.size", values = 1:10)
)
knn_nntrf_tune <- makeTuneWrapper(knn_nntrf, resampling = inner_desc, par.set = ps, 
                                     control = control_grid, measures = list(acc), show.info = FALSE)

set.seed(1)
# Please, note that in order to save time, results have been precomputed
if(file.exists("../inst/error_knn_nntrf_tune5.rda")){load("../inst/error_knn_nntrf_tune5.rda")} else {
error_knn_nntrf_tune <- resample(knn_nntrf_tune, doughnut_task, outer_inst, 
                                 measures = list(acc), 
                                 extract = getTuneResult, show.info =  FALSE)
# save(error_knn_nntrf_tune, file="../inst/error_knn_nntrf_tune5.rda")
}


print(error_knn_nntrf_tune$extract)
print(error_knn_nntrf_tune$aggr)
results_hyper <- generateHyperParsEffectData(error_knn_nntrf_tune)
head(arrange(results_hyper$data, -acc.test.mean))

## ------------------------------------------------------------------------
knn_pca <- cpoPca(center=TRUE, scale=TRUE, export=c("rank")) %>>% knn_lrn

ps_pca <- makeParamSet(makeDiscreteParam("k", values = 1:7),
                       makeDiscreteParam("pca.rank", values = 1:10)
)

knn_pca_tune <- makeTuneWrapper(knn_pca, resampling = inner_desc, par.set = ps_pca, 
                                     control = control_grid, measures = list(acc), show.info = FALSE)

## ------------------------------------------------------------------------
set.seed(1)
# Please, note that in order to save time, results have been precomputed
if(file.exists("../inst/error_knn_pca_tune.rda")){load("../inst/error_knn_pca_tune.rda")} else {
error_knn_pca_tune <- resample(knn_pca_tune, doughnut_task, outer_inst, 
                               measures = list(acc), 
                               extract = getTuneResult, show.info =  FALSE)
# save(error_knn_pca_tune, file="../inst/error_knn_pca_tune.rda")
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

set.seed(1)
# Please, note that in order to save time, results have been precomputed
if(file.exists("../inst/error_knn_tune.rda")){load("../inst/error_knn_tune.rda")} else {
error_knn_tune <- resample(knn_tune, doughnut_task, outer_inst, measures = list(acc), 
                           extract = getTuneResult, show.info =  FALSE)
#save(error_knn_tune, file="../inst/error_knn_tune.rda")
}


## ------------------------------------------------------------------------
print(error_knn_tune$extract)
print(error_knn_tune$aggr)
results_hyper <- generateHyperParsEffectData(error_knn_tune)
head(arrange(results_hyper$data, -acc.test.mean))

