---
title: "nntrf hyper-parameter tuning"
author: "Ricardo Aler"
date: "2020-07-21"
output: rmarkdown::html_vignette
vignette: >
  %\VignetteIndexEntry{nntrf hyper-parameter tuning}
  %\VignetteEngine{knitr::knitr}
  %\VignetteEncoding{UTF-8}
---




```r
library(nntrf)
library(mlr)
#> Loading required package: ParamHelpers
#> 'mlr' is in maintenance mode since July 2019. Future development
#> efforts will go into its successor 'mlr3'
#> (<https://mlr3.mlr-org.com>).
library(mlrCPO)
library(FNN)
```

# nntrf Hyper-parameter Tuning

**nntrf** has several hyper-parameters which are important in order to obtain good results. Those are:

- **size:** The number of hidden neurons
- **maxit:** The number of iterations
- **repetitions:** The number of training repetitions
- **use_sigmoid:** Whether the transformation should use the sigmoid or not

Machine learning pipelines usually contain two kinds of steps: pre-processing and classifier/regressor. Both kinds of steps contain hyper-parameters and they are optimized together. **nntrf** is a preprocessing step. The classifier method that will be used after preprocessing is KNN, whose main hyper-parameter is the number of neighbors (**k**). Hyper-parameter tuning could be programmed from scratch, but it is more efficient to use the procedures already available in machine learning packages such as [mlr](https://mlr.mlr-org.com/) or Caret. In this case, **mlr** will be used. Code to do that is described below.

The next piece of code has nothing to do with **nntrf**. It just establishes that the doughnutRandRotated dataset is going to be used (with target variable "V11"), that grid search is going to be used for hyper-parameter tuning, that an external 3-fold crossvalidation is going to be used to evaluate models, while an inner 3-fold crossvalidation is going to be used for hyper-parameter tuning. 


```r
data("doughnutRandRotated")

doughnut_task <- makeClassifTask(data = doughnutRandRotated, target = "V11")
control_grid <- makeTuneControlGrid()
inner_desc <- makeResampleDesc("CV", iter=3)
outer_desc <-  makeResampleDesc("CV", iter=3)
set.seed(1)
outer_inst <- makeResampleInstance(outer_desc, doughnut_task)
```

A mlr subpakage, called mlrCPO, is going to be used to combine pre-processing and learning into a single pipeline. In order to do that, **nntrf** must be defined as a pipeline step, as follows. Basically, it defines **train** and **retrafo** methods. The former, trains the neural networks and stores the hidden layer weights, the latter applies the transformation on a datasaet. **pSS** is used to define the main **nntrf** hyper-parameters.


```r
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
```

Next, the pipeline of pre-processing + classifier method (KNN in this case) is defined.


```r
# knn is the machine learning method. The knn available in the FNN package is used
knn_lrn <- makeLearner("classif.fnn")
# Then, knn is combined with nntrf's preprocessing into a pipeline
knn_nntrf <- cpo_nntrf() %>>% knn_lrn
# Just in case, we fix the values of the hyper-parameters that we do not require to optimize
# (not necessary, because they already have default values. Just to make their values explicit)
knn_nntrf <- setHyperPars(knn_nntrf, nntrfCPO.repetitions=1, nntrfCPO.maxit=100, nntrfCPO.use_sigmoid=FALSE)

# However, we are going to use 2 repetitions here, instead of 1 (the default):

knn_nntrf <- setHyperPars(knn_nntrf, nntrfCPO.repetitions=2)
```

Next, the hyper-parameter space for the pipeline is defined. Only two hyper-parameters will be optimized: the number of KNN neighbors (k), from 1 to 7, and the number of hidden neurons (size), from 1 to 10. The remaining hyper-parameters are left to some default values.


```r
ps <- makeParamSet(makeDiscreteParam("k", values = 1:7),
                   makeDiscreteParam("nntrfCPO.size", values = 1:10)
)
```

Next, a mlr wrapper is used to give the **knn_nntrf** pipeline the ability to do hyper-parameter tuning.


```r
knn_nntrf_tune <- makeTuneWrapper(knn_nntrf, resampling = inner_desc, par.set = ps, 
                                     control = control_grid, measures = list(acc), show.info = FALSE)
```

Finally, the complete process (3-fold hyper-parameter tuning) and 3-fold outer model evaluation is run. It takes some time. 


```r
set.seed(1)
# Please, note that in order to save time, results have been precomputed
if(file.exists("../inst/error_knn_nntrf_tune.rda")){load("../inst/error_knn_nntrf_tune.rda")} else {
error_knn_nntrf_tune <- resample(knn_nntrf_tune, doughnut_task, outer_inst, 
                                 measures = list(acc), 
                                 extract = getTuneResult, show.info =  FALSE)
}
#save(error_knn_nntrf_tune, file="error_knn_nntrf_tune.rda")
```

Errors and optimal hyper-parameters are as follows (the 3-fold inner hyper-parameter tuning crossvalidation accuracy is also shown in **acc.test.mean** ). Note that 


```r
print(error_knn_nntrf_tune$extract)
#> [[1]]
#> Tune result:
#> Op. pars: k=7; nntrfCPO.size=6
#> acc.test.mean=0.9710512
#> 
#> [[2]]
#> Tune result:
#> Op. pars: k=5; nntrfCPO.size=6
#> acc.test.mean=0.9665467
#> 
#> [[3]]
#> Tune result:
#> Op. pars: k=3; nntrfCPO.size=5
#> acc.test.mean=0.9589009
```

And the final outer 3-fold crossvalition accuracy is displayed below. Please, note that this **acc.test.mean** corresponds to the outer 3-fold crossvalidation, while the **acc.test.mean** above, corresponds to the inner 3-fold crossvalidation accuracy (computed during hyper-parameter tuning).


```r
print(error_knn_nntrf_tune$aggr)
#> acc.test.mean 
#>     0.9522975
```

Although not required, mlr allows to display the results of the different hyper-parameter values, sorted by the **inner** 3-fold crossvalidation, from best to worse.


```r
library(dplyr)
#> 
#> Attaching package: 'dplyr'
#> The following objects are masked from 'package:stats':
#> 
#>     filter, lag
#> The following objects are masked from 'package:base':
#> 
#>     intersect, setdiff, setequal, union
results_hyper <- generateHyperParsEffectData(error_knn_nntrf_tune)
arrange(results_hyper$data, -acc.test.mean)
#>     k nntrfCPO.size acc.test.mean iteration exec.time nested_cv_run
#> 1   7             6     0.9710512        42     3.161             1
#> 2   5             6     0.9665467        40     3.088             2
#> 3   7             4     0.9637031        28     2.458             1
#> 4   3             5     0.9589009        31     2.897             3
#> 5   7            10     0.9581526        70     4.333             3
#> 6   4             6     0.9576958        39     3.514             2
#> 7   7             6     0.9572517        42     3.038             3
#> 8   7             8     0.9568007        56     3.802             1
#> 9   5             7     0.9566527        47     3.257             3
#> 10  4             6     0.9547045        39     3.098             3
#> 11  4             5     0.9546999        32     2.883             3
#> 12  7             8     0.9543954        56     3.413             2
#> 13  7             9     0.9540954        63     4.373             2
#> 14  5             6     0.9538012        40     3.066             3
#> 15  6             8     0.9536572        55     3.412             1
#> 16  5             6     0.9536526        40     3.399             1
#> 17  7             3     0.9533492        21     2.412             3
#> 18  4             4     0.9532015        25     2.374             3
#> 19  1             7     0.9524521        43     3.378             3
#> 20  3             6     0.9518587        38     3.267             1
#> 21  3             4     0.9517005        24     2.427             1
#> 22  7             5     0.9512509        35     2.814             3
#> 23  3             8     0.9511009        52     3.999             1
#> 24  6            10     0.9498950        69     5.034             2
#> 25  6             9     0.9492949        62     3.707             2
#> 26  6             7     0.9491512        48     3.794             1
#> 27  7            10     0.9490008        70     4.625             1
#> 28  4            10     0.9487009        67     4.751             3
#> 29  5             5     0.9485449        33     2.716             2
#> 30  3             8     0.9485449        52     3.616             2
#> 31  5             9     0.9485449        61     4.724             2
#> 32  3             7     0.9482524        45     3.658             1
#> 33  1             7     0.9475010        43     4.258             1
#> 34  5             8     0.9474997        54     3.923             3
#> 35  6             8     0.9473447        55     3.817             2
#> 36  1             8     0.9472009        50     3.764             3
#> 37  6             4     0.9472000        27     2.377             3
#> 38  6             4     0.9466025        27     2.503             1
#> 39  7             8     0.9466016        56     3.610             3
#> 40  4             7     0.9464505        46     3.640             1
#> 41  4             4     0.9464446        25     2.797             2
#> 42  7             9     0.9458510        63     4.140             3
#> 43  5             7     0.9458446        47     3.426             2
#> 44  6             8     0.9449514        55     3.857             3
#> 45  5             7     0.9443525        47     4.150             1
#> 46  2             8     0.9442014        51     3.656             3
#> 47  1             6     0.9439015        36     3.199             3
#> 48  5            10     0.9432943        68     4.534             2
#> 49  5             9     0.9424011        61     4.234             3
#> 50  7             9     0.9422530        63     4.043             1
#> 51  6             7     0.9422442        48     3.324             2
#> 52  1             5     0.9421010        29     2.871             3
#> 53  4             8     0.9413539        53     3.652             3
#> 54  5             8     0.9412051        54     3.831             1
#> 55  4             7     0.9411941        46     3.605             2
#> 56  7             6     0.9410441        42     3.013             2
#> 57  4             8     0.9403029        53     3.876             1
#> 58  2            10     0.9398553        65     4.782             3
#> 59  1             6     0.9395494        36     3.107             1
#> 60  3             9     0.9395440        59     4.315             2
#> 61  6             6     0.9394046        41     3.159             1
#> 62  7             7     0.9382039        49     3.702             1
#> 63  4            10     0.9381938        67     4.666             2
#> 64  3             7     0.9381938        45     3.437             2
#> 65  7             7     0.9380500        49     3.540             3
#> 66  6             9     0.9380489        62     4.045             1
#> 67  4             5     0.9380438        32     2.862             2
#> 68  7            10     0.9380438        70     4.061             2
#> 69  5             5     0.9374524        33     2.735             3
#> 70  6             5     0.9362548        34     2.828             1
#> 71  6             7     0.9359542        48     3.556             3
#> 72  2             5     0.9357936        30     2.701             2
#> 73  6             9     0.9356594        62     4.122             3
#> 74  1             5     0.9354972        29     2.930             1
#> 75  3            10     0.9352008        66     4.598             1
#> 76  3            10     0.9335571        66     4.800             3
#> 77  4            10     0.9332538        67     4.652             1
#> 78  1             3     0.9326433        15     2.167             2
#> 79  4             9     0.9325013        60     4.078             3
#> 80  5            10     0.9323528        68     4.607             1
#> 81  4             7     0.9320521        46     3.590             3
#> 82  3             9     0.9317513        59     4.055             3
#> 83  3             9     0.9316107        59     4.062             1
#> 84  3             6     0.9314431        38     3.012             2
#> 85  4             9     0.9311431        60     4.213             2
#> 86  2             9     0.9298068        58     4.244             3
#> 87  2             9     0.9281428        58     3.577             2
#> 88  5             8     0.9281428        54     3.940             2
#> 89  2             5     0.9275506        30     2.674             1
#> 90  6            10     0.9275504        69     3.785             1
#> 91  2             7     0.9274053        44     3.715             3
#> 92  1             7     0.9267927        43     3.221             2
#> 93  2             5     0.9257507        30     2.663             3
#> 94  2             4     0.9256004        23     2.528             1
#> 95  4             5     0.9253034        32     2.696             1
#> 96  3             7     0.9251511        45     3.189             3
#> 97  6             4     0.9249925        27     2.463             2
#> 98  3             8     0.9245547        52     3.867             3
#> 99  1             8     0.9240924        50     3.541             2
#> 100 1             6     0.9236424        36     3.257             2
#> 101 6            10     0.9233514        69     3.988             3
#> 102 6             5     0.9225923        34     2.838             2
#> 103 2             6     0.9211039        37     3.205             3
#> 104 3             6     0.9193031        38     3.102             3
#> 105 2            10     0.9188577        65     3.815             1
#> 106 1             9     0.9181033        57     3.983             1
#> 107 7             7     0.9179418        49     3.400             2
#> 108 1             8     0.9176461        50     3.814             1
#> 109 5            10     0.9175010        68     4.251             3
#> 110 1             9     0.9167417        57     4.258             2
#> 111 1            10     0.9161570        64     4.674             1
#> 112 1             9     0.9161507        57     3.667             3
#> 113 2             7     0.9143414        44     3.032             2
#> 114 4             8     0.9137414        53     3.830             2
#> 115 1            10     0.9128498        64     4.612             3
#> 116 3            10     0.9114911        66     4.620             2
#> 117 2            10     0.9113411        65     4.188             2
#> 118 2             6     0.9099910        37     2.916             2
#> 119 4             9     0.9088022        60     3.738             1
#> 120 2             7     0.9067117        44     3.409             1
#> 121 1            10     0.9030903        64     4.366             2
#> 122 2             8     0.8993399        51     3.271             2
#> 123 2             9     0.8989004        58     3.932             1
#> 124 2             6     0.8902114        37     3.004             1
#> 125 5             5     0.8868945        33     3.018             1
#> 126 4             6     0.8775986        39     3.056             1
#> 127 5             3     0.8739923        19     2.049             1
#> 128 2             8     0.8738764        51     3.643             1
#> 129 1             3     0.8720429        15     2.230             1
#> 130 5             3     0.8639423        19     2.246             3
#> 131 7             4     0.8639364        28     2.505             2
#> 132 5             4     0.8603360        26     2.348             2
#> 133 3             3     0.8564418        17     2.268             3
#> 134 7             4     0.8530384        28     2.404             3
#> 135 3             4     0.8526853        24     2.404             2
#> 136 4             4     0.8516419        25     2.661             1
#> 137 5             3     0.8504350        19     2.179             2
#> 138 3             5     0.8495856        31     2.433             1
#> 139 3             4     0.8487948        24     2.435             3
#> 140 3             5     0.8478848        31     2.782             2
#> 141 1             4     0.8458419        22     2.357             3
#> 142 7             5     0.8450345        35     2.695             2
#> 143 6             5     0.8429438        34     2.500             3
#> 144 1             2     0.8406841         8     1.777             2
#> 145 6             6     0.8405885        41     2.997             3
#> 146 1             5     0.8397840        29     2.839             2
#> 147 1             4     0.8339334        22     2.456             2
#> 148 2             4     0.8325833        23     2.316             2
#> 149 2             3     0.8279455        16     1.831             3
#> 150 2             4     0.8268924        23     2.427             3
#> 151 4             3     0.8268827        18     2.086             2
#> 152 3             3     0.8192424        17     2.025             1
#> 153 1             4     0.8171888        22     2.351             1
#> 154 4             3     0.8045576        18     2.207             3
#> 155 7             5     0.8044290        35     2.658             1
#> 156 5             9     0.8043877        61     3.259             1
#> 157 1             2     0.8011081         8     1.774             1
#> 158 2             3     0.7954354        16     2.026             1
#> 159 5             4     0.7798139        26     2.456             3
#> 160 2             2     0.7766682         9     1.869             3
#> 161 6             6     0.7748275        41     2.935             2
#> 162 6             3     0.7589259        20     2.138             2
#> 163 7             3     0.7562256        21     2.219             2
#> 164 1             3     0.7557789        15     2.100             3
#> 165 4             3     0.7550812        18     2.132             1
#> 166 2             2     0.7514799         9     1.971             1
#> 167 3             3     0.7511251        17     2.177             2
#> 168 6             3     0.7488827        20     2.109             3
#> 169 7             3     0.7339133        21     2.274             1
#> 170 6             2     0.7286754        13     1.908             1
#> 171 7             2     0.7286229        14     2.121             2
#> 172 4             2     0.7227723        11     1.820             2
#> 173 3             2     0.7104929        10     1.890             3
#> 174 6             2     0.7017702        13     1.950             2
#> 175 5             2     0.6953195        12     1.805             2
#> 176 5             4     0.6707729        26     2.661             1
#> 177 5             2     0.6704550        12     2.083             1
#> 178 6             2     0.6700091        13     2.145             3
#> 179 6             3     0.6652175        20     1.861             1
#> 180 2             3     0.6605161        16     2.341             2
#> 181 3             2     0.6572692        10     2.238             1
#> 182 7             2     0.6563673        14     1.706             1
#> 183 1             2     0.6562228         8     1.672             3
#> 184 7             1     0.6514174         7     1.474             1
#> 185 2             2     0.6501650         9     1.796             2
#> 186 3             2     0.6500150        10     1.899             2
#> 187 4             2     0.6496186        11     1.630             1
#> 188 5             2     0.6487168        12     1.908             3
#> 189 5             1     0.6439195         5     1.560             3
#> 190 7             2     0.6401687        14     1.892             3
#> 191 4             1     0.6398680         4     1.290             1
#> 192 4             2     0.6388173        11     1.953             3
#> 193 4             1     0.6383138         4     1.490             2
#> 194 5             1     0.6380138         5     1.550             2
#> 195 3             1     0.6315632         3     1.260             2
#> 196 6             1     0.6260662         6     1.478             1
#> 197 6             1     0.6244168         6     1.638             3
#> 198 2             1     0.6240624         2     1.315             2
#> 199 2             1     0.6206765         2     1.225             3
#> 200 6             1     0.6195620         6     1.444             2
#> 201 2             1     0.6178170         2     1.529             1
#> 202 7             1     0.6162616         7     1.394             2
#> 203 3             1     0.6017653         3     1.305             1
#> 204 3             1     0.6002652         3     1.595             3
#> 205 1             1     0.5991599         1     1.509             2
#> 206 1             1     0.5923300         1     1.204             3
#> 207 5             1     0.5903628         5     1.510             1
#> 208 7             1     0.5890123         7     1.408             3
#> 209 4             1     0.5816717         4     1.360             3
#> 210 1             1     0.5653125         1     1.612             1
```

We can also check directly what would happen with only 4 neurons (and 7 neighbors), as suggested by the table above.


```r
knn_nntrf <- cpo_nntrf() %>>% makeLearner("classif.fnn")

knn_nntrf <- setHyperPars(knn_nntrf, nntrfCPO.repetitions=2, nntrfCPO.maxit=100, nntrfCPO.use_sigmoid=FALSE, k=7, nntrfCPO.size=4)

set.seed(1)
# Please, note that in order to save time, results have been precomputed
if(file.exists("../inst/error_knn_nntrf.rda")){load("../inst/error_knn_nntrf.rda")} else {
error_knn_nntrf <- resample(knn_nntrf, doughnut_task, outer_inst, measures = list(acc), 
                            show.info =  FALSE)
}
#save(error_knn_nntrf, file="error_knn_nntrf.rda")
```



```r
# First, the three evaluations of the outer 3-fold crossvalidation, one per fold:
print(error_knn_nntrf$measures.test)
#>   iter       acc
#> 1    1 0.9636964
#> 2    2 0.9565087
#> 3    3 0.9477948
# Second, their average
print(error_knn_nntrf$aggr)
#> acc.test.mean 
#>     0.9559999
```
## nntrf Hyper-parameter Tuning with repetitions=5

Despite knowing that 2 neurons is enough to solve this problem, hyper-parameter tuning in the previous section always selected between 5 and 6 neurons. In order to check whether the reason is that the neural network training gets stuck in local minima, the piece of code below uses repetitions = 5. It will not be run here because it takes long, but indeed, results show that the number of neurons can be reduced to 4-5 and accuracy increases up to 0.9773002.



## Hyper-parameter tuning with PCA

In order to compare a supervised transformation method (**nntrf**) with an unsupervised one (PCA), it is very easy to do exactly the same pre-processing with PCA. In this case, the main hyper-paramaters are **k** (number of KNN neighbors) and **Pca.rank** (the number of PCA components to be used, which would be the counterpart of **size**, the number of hidden neurons used by **nntrf**).


```r
knn_pca <- cpoPca(center=TRUE, scale=TRUE, export=c("rank")) %>>% knn_lrn

ps_pca <- makeParamSet(makeDiscreteParam("k", values = 1:7),
                       makeDiscreteParam("pca.rank", values = 1:10)
)

knn_pca_tune <- makeTuneWrapper(knn_pca, resampling = inner_desc, par.set = ps_pca, 
                                     control = control_grid, measures = list(acc), show.info = FALSE)
```


```r
set.seed(1)
# Please, note that in order to save time, results have been precomputed
if(file.exists("../inst/error_knn_pca_tune.rda")){load("../inst/error_knn_pca_tune.rda")} else {
error_knn_pca_tune <- resample(knn_pca_tune, doughnut_task, outer_inst, 
                               measures = list(acc), 
                               extract = getTuneResult, show.info =  FALSE)
}
# save(error_knn_pca_tune, file="error_knn_pca_tune.rda")
```

It can be seen below that while **nntrf** is able to get an accuracy higher than 0.95, **PCA** only gets to nearly 0.65. Also the number of components required by **PCA** is the maximum allowed (pca.rank=10)


```r
print(error_knn_pca_tune$extract)
#> [[1]]
#> Tune result:
#> Op. pars: k=2; pca.rank=10
#> acc.test.mean=0.6424198
#> 
#> [[2]]
#> Tune result:
#> Op. pars: k=6; pca.rank=10
#> acc.test.mean=0.6461146
#> 
#> [[3]]
#> Tune result:
#> Op. pars: k=2; pca.rank=10
#> acc.test.mean=0.6331176
print(error_knn_pca_tune$aggr)
#> acc.test.mean 
#>     0.6495998
results_hyper <- generateHyperParsEffectData(error_knn_pca_tune)
arrange(results_hyper$data, -acc.test.mean)
#>     k pca.rank acc.test.mean iteration exec.time nested_cv_run
#> 1   6       10     0.6461146        69     1.719             2
#> 2   4       10     0.6447645        67     1.567             2
#> 3   2       10     0.6424198        65     1.432             1
#> 4   6       10     0.6404687        69     1.627             1
#> 5   7       10     0.6372637        70     1.780             2
#> 6   4       10     0.6349195        67     1.766             1
#> 7   5       10     0.6347135        68     1.678             2
#> 8   2       10     0.6331176        65     1.471             3
#> 9   2       10     0.6329133        65     1.442             2
#> 10  4        9     0.6304186        60     1.396             1
#> 11  6        9     0.6297630        62     1.692             2
#> 12  2        9     0.6292181        58     1.294             1
#> 13  6        9     0.6283196        62     1.531             3
#> 14  4       10     0.6281674        67     1.735             3
#> 15  4        9     0.6273627        60     1.491             2
#> 16  2        9     0.6269127        58     1.347             2
#> 17  7       10     0.6266690        70     1.711             1
#> 18  2        9     0.6265190        58     1.276             3
#> 19  6        9     0.6262186        62     1.493             1
#> 20  6       10     0.6256181        69     1.691             3
#> 21  4        9     0.6247200        60     1.434             3
#> 22  6        8     0.6240624        55     1.389             2
#> 23  2        7     0.6237624        44     1.133             2
#> 24  5       10     0.6227691        68     1.572             1
#> 25  4        8     0.6227123        53     1.306             2
#> 26  2        8     0.6225623        51     1.202             2
#> 27  7        9     0.6216622        63     1.574             2
#> 28  4        7     0.6212121        46     1.193             2
#> 29  3       10     0.6189619        66     1.556             2
#> 30  4        8     0.6167696        53     1.314             1
#> 31  7        9     0.6160215        63     1.569             3
#> 32  5        9     0.6155707        61     1.503             3
#> 33  5        9     0.6149693        61     1.451             1
#> 34  5        9     0.6143114        61     1.532             2
#> 35  2        7     0.6142199        44     1.139             1
#> 36  6        7     0.6140114        48     1.273             2
#> 37  7        8     0.6140114        56     1.393             2
#> 38  7        9     0.6137681        63     1.523             1
#> 39  6        7     0.6134711        48     1.214             3
#> 40  4        8     0.6125720        53     1.342             3
#> 41  3       10     0.6124196        66     1.544             1
#> 42  6        8     0.6116674        55     1.404             1
#> 43  6        8     0.6113716        55     1.362             3
#> 44  4        7     0.6113684        46     1.215             1
#> 45  2        8     0.6109207        51     1.227             1
#> 46  6        4     0.6108611        27     1.016             2
#> 47  2        6     0.6103219        37     1.050             1
#> 48  3        9     0.6100214        59     1.378             3
#> 49  2        8     0.6098714        51     1.220             3
#> 50  6        6     0.6098110        41     1.106             2
#> 51  5       10     0.6097188        68     1.683             3
#> 52  4        4     0.6086109        25     1.003             2
#> 53  4        6     0.6086109        39     1.042             2
#> 54  3       10     0.6083688        66     1.561             3
#> 55  2        6     0.6078608        37     1.045             2
#> 56  7        8     0.6076232        56     1.442             3
#> 57  1       10     0.6074107        64     1.297             2
#> 58  7       10     0.6070193        70     1.740             3
#> 59  3        9     0.6059680        59     1.375             1
#> 60  6        7     0.6058203        48     1.211             1
#> 61  4        7     0.6050716        46     1.153             3
#> 62  2        5     0.6047105        30     1.011             2
#> 63  3        9     0.6047105        59     1.459             2
#> 64  5        7     0.6044104        47     1.245             2
#> 65  5        8     0.6039604        54     1.371             2
#> 66  1       10     0.6020725        64     1.281             1
#> 67  4        3     0.6017102        18     1.000             2
#> 68  4        2     0.6014101        11     1.026             2
#> 69  2        7     0.6013218        44     1.271             3
#> 70  7        7     0.6006601        49     1.396             2
#> 71  2        3     0.6002100        16     1.001             2
#> 72  4        5     0.6000600        32     1.019             2
#> 73  2        4     0.5999100        23     1.018             2
#> 74  2        1     0.5985599         2     0.968             2
#> 75  5        8     0.5983232        54     1.370             3
#> 76  6        5     0.5981098        34     1.031             2
#> 77  6        6     0.5971213        41     1.135             1
#> 78  2        6     0.5971192        37     1.062             3
#> 79  6        3     0.5970597        20     1.045             2
#> 80  2        4     0.5969712        23     0.986             1
#> 81  4        6     0.5963709        39     1.100             1
#> 82  6        2     0.5963096        13     0.953             2
#> 83  4        4     0.5962211        25     1.009             1
#> 84  7        8     0.5962176        56     1.424             1
#> 85  2        5     0.5960719        30     0.986             1
#> 86  2        2     0.5956235         9     1.005             1
#> 87  5        8     0.5954682        54     1.347             1
#> 88  7        7     0.5953230        49     1.280             3
#> 89  1       10     0.5953201        64     1.309             3
#> 90  6        5     0.5953186        34     1.054             3
#> 91  7        6     0.5945095        42     1.119             2
#> 92  2        3     0.5944220        16     1.015             1
#> 93  2        2     0.5943594         9     0.982             2
#> 94  4        3     0.5942712        18     1.035             1
#> 95  7        7     0.5942710        49     1.230             1
#> 96  3        7     0.5940594        45     1.176             2
#> 97  1        9     0.5939724        57     1.375             3
#> 98  6        1     0.5931593         6     0.989             2
#> 99  6        4     0.5927720        27     1.035             1
#> 100 1        9     0.5927695        57     1.355             1
#> 101 2        1     0.5923210         2     0.992             3
#> 102 2        5     0.5923191        30     1.042             3
#> 103 6        3     0.5923177        20     0.954             3
#> 104 3        8     0.5919592        52     1.243             2
#> 105 6        2     0.5918721        13     1.026             1
#> 106 4        2     0.5915730        11     1.000             1
#> 107 4        6     0.5912692        39     1.071             3
#> 108 4        5     0.5906724        32     1.071             1
#> 109 6        3     0.5905204        20     1.045             1
#> 110 6        5     0.5900734        34     1.029             1
#> 111 4        5     0.5900684        32     1.023             3
#> 112 4        1     0.5895590         4     1.053             2
#> 113 3        8     0.5890225        52     1.284             3
#> 114 6        6     0.5890196        41     1.136             3
#> 115 2        3     0.5879676        16     1.019             3
#> 116 5        7     0.5876737        47     1.200             3
#> 117 2        4     0.5875195        23     1.001             3
#> 118 1        9     0.5874587        57     1.210             2
#> 119 5        7     0.5873707        47     1.355             1
#> 120 4        4     0.5872189        25     1.025             3
#> 121 6        4     0.5869202        27     1.060             3
#> 122 5        6     0.5856586        40     1.064             2
#> 123 2        2     0.5855698         9     0.987             3
#> 124 4        1     0.5849739         4     1.001             3
#> 125 3        6     0.5847585        38     1.038             2
#> 126 7        3     0.5844584        21     1.031             2
#> 127 6        1     0.5839230         6     1.041             3
#> 128 3        8     0.5834685        52     1.247             1
#> 129 7        5     0.5831083        35     1.047             2
#> 130 3        7     0.5830188        45     1.150             1
#> 131 4        3     0.5824185        18     1.280             3
#> 132 2        1     0.5822719         2     0.974             1
#> 133 1        7     0.5822082        43     1.057             2
#> 134 7        4     0.5819082        28     1.052             2
#> 135 6        1     0.5810719         6     1.016             1
#> 136 1        8     0.5807081        50     1.100             2
#> 137 3        6     0.5789731        38     1.084             1
#> 138 6        2     0.5786704        13     1.025             3
#> 139 5        4     0.5774077        26     1.010             2
#> 140 5        3     0.5772577        19     1.006             2
#> 141 7        6     0.5767229        42     1.161             1
#> 142 5        5     0.5762076        33     1.087             2
#> 143 7        5     0.5759732        35     1.053             1
#> 144 4        1     0.5758220         4     0.995             1
#> 145 1        8     0.5753683        50     1.070             1
#> 146 4        2     0.5752205        11     1.014             3
#> 147 5        6     0.5750723        40     1.105             1
#> 148 3        7     0.5744736        45     1.121             3
#> 149 7        2     0.5744074        14     0.977             2
#> 150 5        5     0.5743240        33     1.227             1
#> 151 5        6     0.5737204        40     1.105             3
#> 152 1        7     0.5732712        43     1.069             1
#> 153 7        6     0.5728205        42     1.175             3
#> 154 1        8     0.5725228        50     1.131             3
#> 155 7        1     0.5714071         7     1.035             2
#> 156 7        4     0.5713224        28     1.040             1
#> 157 1        7     0.5701239        43     1.084             3
#> 158 3        4     0.5697570        24     1.142             2
#> 159 7        2     0.5696729        14     1.023             1
#> 160 7        5     0.5695202        35     1.056             3
#> 161 1        6     0.5675738        36     1.005             1
#> 162 3        5     0.5675068        31     1.005             2
#> 163 3        5     0.5672736        31     1.100             1
#> 164 7        3     0.5672691        21     0.963             3
#> 165 5        4     0.5665227        26     1.047             1
#> 166 3        3     0.5664566        17     0.950             2
#> 167 7        3     0.5659216        21     1.159             1
#> 168 5        1     0.5651065         5     0.981             2
#> 169 5        2     0.5648065        12     1.145             2
#> 170 5        5     0.5645699        33     1.050             3
#> 171 7        1     0.5627739         7     1.027             3
#> 172 5        4     0.5624724        26     1.025             3
#> 173 5        2     0.5615754        12     1.004             1
#> 174 3        4     0.5615726        24     1.004             1
#> 175 7        4     0.5615724        28     1.058             3
#> 176 5        3     0.5614230        19     1.037             1
#> 177 5        3     0.5612693        19     0.931             3
#> 178 3        6     0.5611197        38     1.092             3
#> 179 1        5     0.5600738        29     1.020             1
#> 180 1        6     0.5591059        36     1.165             2
#> 181 1        6     0.5572229        36     1.041             3
#> 182 5        1     0.5567748         5     1.010             3
#> 183 7        1     0.5560224         7     1.019             1
#> 184 3        2     0.5541554        10     0.975             2
#> 185 3        3     0.5537751        17     1.050             1
#> 186 3        5     0.5521217        31     1.229             3
#> 187 1        5     0.5520552        29     0.986             2
#> 188 3        2     0.5518253        10     0.991             1
#> 189 1        3     0.5502550        15     0.949             2
#> 190 3        1     0.5490549         3     0.981             2
#> 191 1        4     0.5489049        22     0.992             2
#> 192 7        2     0.5488221        14     1.057             3
#> 193 3        3     0.5470205        17     1.014             3
#> 194 3        4     0.5467218        24     1.010             3
#> 195 5        1     0.5461222         5     1.012             1
#> 196 1        2     0.5458225         8     0.957             3
#> 197 3        1     0.5455244         3     1.006             3
#> 198 1        5     0.5453742        29     1.010             3
#> 199 1        4     0.5453727        22     0.957             1
#> 200 1        1     0.5436544         1     0.965             2
#> 201 5        2     0.5423724        12     1.022             3
#> 202 3        2     0.5408720        10     0.983             3
#> 203 1        3     0.5392255        15     0.993             1
#> 204 1        2     0.5371241         8     0.960             1
#> 205 1        3     0.5359227        15     0.991             3
#> 206 1        2     0.5358536         8     0.926             2
#> 207 1        4     0.5357725        22     0.916             3
#> 208 3        1     0.5350234         3     1.196             1
#> 209 1        1     0.5239238         1     1.171             3
#> 210 1        1     0.5210758         1     0.932             1
```
## Hyper-parameter tuning with just KNN

For completeness sake, below are the results with just KNN (similar to the ones with PCA):


```r

ps_knn <- makeParamSet(makeDiscreteParam("k", values = 1:7))


knn_tune <- makeTuneWrapper(knn_lrn, resampling = inner_desc, par.set = ps_knn, 
                                     control = control_grid, measures = list(acc), show.info = FALSE)

set.seed(1)
# Please, note that in order to save time, results have been precomputed
if(file.exists("../inst/error_knn_tune.rda")){load("../inst/error_knn_tune.rda")} else {
error_knn_tune <- resample(knn_tune, doughnut_task, outer_inst, measures = list(acc), 
                           extract = getTuneResult, show.info =  FALSE)
}
#save(error_knn_tune, file="error_knn_tune.rda")
```


```r
print(error_knn_tune$extract)
#> [[1]]
#> Tune result:
#> Op. pars: k=2
#> acc.test.mean=0.6403202
#> 
#> [[2]]
#> Tune result:
#> Op. pars: k=4
#> acc.test.mean=0.6401140
#> 
#> [[3]]
#> Tune result:
#> Op. pars: k=2
#> acc.test.mean=0.6316178
print(error_knn_tune$aggr)
#> acc.test.mean 
#>     0.6461999
results_hyper <- generateHyperParsEffectData(error_knn_tune)
arrange(results_hyper$data, -acc.test.mean)
#>    k acc.test.mean iteration exec.time nested_cv_run
#> 1  2     0.6403202         2     1.196             1
#> 2  4     0.6401140         4     1.361             2
#> 3  6     0.6398140         6     1.484             2
#> 4  4     0.6367190         4     1.386             1
#> 5  6     0.6356693         6     1.640             1
#> 6  7     0.6345635         7     1.522             2
#> 7  2     0.6316178         2     1.221             3
#> 8  5     0.6288629         5     1.411             2
#> 9  2     0.6285629         2     1.203             2
#> 10 4     0.6251686         4     1.396             3
#> 11 6     0.6203678         6     1.613             3
#> 12 7     0.6172198         7     1.505             1
#> 13 5     0.6151193         5     1.443             1
#> 14 3     0.6143114         3     1.311             2
#> 15 3     0.6112194         3     1.309             1
#> 16 7     0.6059692         7     1.484             3
#> 17 5     0.6041693         5     1.456             3
#> 18 3     0.6029689         3     1.315             3
#> 19 1     0.6019222         1     1.058             1
#> 20 1     0.5996100         1     1.389             2
#> 21 1     0.5942708         1     1.098             3
```
