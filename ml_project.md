
---
title: "ML project"
author: "OS"
date: "October 7, 2018"
output:
  html_document:
    keep_md: true
---
#Coursera - Practical Machine Learning Project

One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, my goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants, to predict the manner in which they did the exercise

##Set up the libraries

```r
library(caret)
```

```
## Loading required package: lattice
```

```
## Loading required package: ggplot2
```

```r
library(randomForest)
```

```
## randomForest 4.6-14
```

```
## Type rfNews() to see new features/changes/bug fixes.
```

```
## 
## Attaching package: 'randomForest'
```

```
## The following object is masked from 'package:ggplot2':
## 
##     margin
```

```r
library(rattle)
```

```
## Rattle: A free graphical interface for data science with R.
## Version 5.2.0 Copyright (c) 2006-2018 Togaware Pty Ltd.
## Type 'rattle()' to shake, rattle, and roll your data.
```

```
## 
## Attaching package: 'rattle'
```

```
## The following object is masked from 'package:randomForest':
## 
##     importance
```

```r
library(e1071) 
library(rpart)
library(gbm)
```

```
## Loaded gbm 2.1.4
```

##Load the data

```r
train_data<-read.csv("pml-training.csv", header = TRUE)
test_data<-read.csv("pml-testing.csv", header = TRUE)
```
create a new partition

```r
to_part  <- createDataPartition(train_data$classe, p=0.7, list=FALSE)
train2_data <- train_data[to_part, ]
test2_data <- train_data[-to_part, ]
```

##Explore and clean the data

There are columns that are mostly empty, I will remove them for the data. I will also remove identification only variables (columns 1 to 7)


```r
data_remove <- sapply(train2_data, function(x) mean(is.na(x))) > 0.90
train2_data <- train2_data[, data_remove==FALSE]
test2_data  <- test2_data[, data_remove==FALSE]
zero <- nearZeroVar(train2_data)
train2_data <- train2_data[, -zero]
test2_data  <- test2_data[, -zero]
train2_data <- train2_data[, -(1:5)]
test2_data <- test2_data[, -(1:5)]
```

##Prediction

###Random forest


```r
set.seed(12345)
controlRF <- trainControl(method="cv", number=3, verboseIter=FALSE)
modFitRandForest <- train(classe ~ ., data=train2_data, method="rf",
                          trControl=controlRF)
modFitRandForest$finalModel
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 27
## 
##         OOB estimate of  error rate: 0.18%
## Confusion matrix:
##      A    B    C    D    E  class.error
## A 3905    1    0    0    0 0.0002560164
## B    3 2654    1    0    0 0.0015048909
## C    0    7 2389    0    0 0.0029215359
## D    0    0    5 2246    1 0.0026642984
## E    0    1    0    6 2518 0.0027722772
```

We will use it with the test dataset


```r
predictRF <- predict(modFitRandForest, newdata=test2_data)
confMatRF <- confusionMatrix(predictRF, test2_data$classe)
confMatRF
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1674    5    0    0    0
##          B    0 1132    3    0    0
##          C    0    2 1023    2    0
##          D    0    0    0  962    6
##          E    0    0    0    0 1076
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9969          
##                  95% CI : (0.9952, 0.9982)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9961          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9939   0.9971   0.9979   0.9945
## Specificity            0.9988   0.9994   0.9992   0.9988   1.0000
## Pos Pred Value         0.9970   0.9974   0.9961   0.9938   1.0000
## Neg Pred Value         1.0000   0.9985   0.9994   0.9996   0.9988
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2845   0.1924   0.1738   0.1635   0.1828
## Detection Prevalence   0.2853   0.1929   0.1745   0.1645   0.1828
## Balanced Accuracy      0.9994   0.9966   0.9981   0.9984   0.9972
```

###Decision Trees

```r
set.seed(12345)
modFitDT <- rpart(classe ~ ., data=train2_data, method="class")
fancyRpartPlot(modFitDT)
```

![](ml_project_files/figure-html/trees-1.png)<!-- -->

We will use it with the test dataset


```r
predictDT <- predict(modFitDT, newdata=test2_data, type="class")
confMatDT <- confusionMatrix(predictDT, test2_data$classe)
confMatDT
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1519  257   37   91   52
##          B   43  652   41   23   24
##          C    8   56  818  128   64
##          D   82  112   68  630  130
##          E   22   62   62   92  812
## 
## Overall Statistics
##                                           
##                Accuracy : 0.7529          
##                  95% CI : (0.7417, 0.7639)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.6859          
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9074   0.5724   0.7973   0.6535   0.7505
## Specificity            0.8962   0.9724   0.9473   0.9203   0.9504
## Pos Pred Value         0.7766   0.8327   0.7616   0.6164   0.7733
## Neg Pred Value         0.9605   0.9045   0.9568   0.9313   0.9442
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2581   0.1108   0.1390   0.1071   0.1380
## Detection Prevalence   0.3324   0.1331   0.1825   0.1737   0.1784
## Balanced Accuracy      0.9018   0.7724   0.8723   0.7869   0.8505
```

##Generalized Boosted Model
This will be the last method to test

```r
set.seed(12345)
controlGBM <- trainControl(method = "repeatedcv", number = 5, repeats = 1)
modFitGBM  <- train(classe ~ ., data=train2_data, method = "gbm",
                    trControl = controlGBM, verbose = FALSE)
modFitGBM$finalModel
```

```
## A gradient boosted model with multinomial loss function.
## 150 iterations were performed.
## There were 53 predictors of which 43 had non-zero influence.
```

Now it is time to predict


```r
predictGBM <- predict(modFitGBM, newdata=test2_data)
confMatGBM <- confusionMatrix(predictGBM, test2_data$classe)
confMatGBM
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1669    8    0    0    1
##          B    5 1124   12    5    5
##          C    0    7 1009   10    3
##          D    0    0    4  948   14
##          E    0    0    1    1 1059
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9871          
##                  95% CI : (0.9839, 0.9898)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9837          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9970   0.9868   0.9834   0.9834   0.9787
## Specificity            0.9979   0.9943   0.9959   0.9963   0.9996
## Pos Pred Value         0.9946   0.9765   0.9806   0.9814   0.9981
## Neg Pred Value         0.9988   0.9968   0.9965   0.9967   0.9952
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2836   0.1910   0.1715   0.1611   0.1799
## Detection Prevalence   0.2851   0.1956   0.1749   0.1641   0.1803
## Balanced Accuracy      0.9974   0.9906   0.9897   0.9899   0.9892
```

##Select the model

Random Forests gave an Accuracy of  99,66%, which was more accurate that what I got from the Decision Trees (74,72%) or GBM (98,56%)


```r
predictTEST <- predict(modFitRandForest, newdata=test_data)
predictTEST
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

