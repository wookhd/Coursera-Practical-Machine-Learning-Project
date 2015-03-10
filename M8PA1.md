# Predict the manner in which they exercise using data from devices
Daniel Woo  

## Environment setup

```
## Warning: package 'caret' was built under R version 3.1.2
```

```
## Warning: package 'mlbench' was built under R version 3.1.2
```

## Executive Summary
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset). We are particularly interested in the following two questions:

* The goal of your project is to predict the manner in which they did the exercise. This is the "classe" variable in the training set. You may use any of the other variables to predict with.



## Cleansing the dataset

Remove the columns that is simply an index, timestamp or username.


```r
training<-trainData[,7:160]
test<-testData[,7:160]

dim(training)
dim(test)
```

Remove the columns that are mostly NAs. 


```r
mostly_data<-apply(!is.na(training),2,sum)>19621
training<-training[,mostly_data]
test<-test[,mostly_data]
```

## Partition the training dataset

Partitioned the training set. The training data is splitted into two sets, 70% for training and 30% for cross-validation


```r
inTrain <- createDataPartition(y=training$classe, p=0.7, list=FALSE)
training1 <- training[inTrain, ]
testing <- training[-inTrain, ]
```

The model is automatically tuned and is evaluated using 5-folds cross validation.

## Prepare training scheme

```r
control <- trainControl(allowParallel = TRUE, method="cv", number=5)
```

## Models

The three models that are constructed and tuned are Stochastic Gradient Boosting (also known as Gradient Boosted Machine or GBM), Random Forest (RF) and Recursive Partitioning (RPART).

The random number seed is set before each algorithm is trained to ensure that each algorithm get the same data partitions. This will allow valid comparison in the final results.

## Train the GBM model


```r
set.seed(7)
modGbm<-train(classe ~ ., method="gbm",data=training1,verbose=FALSE, trControl=control)
```

```
## Warning: package 'gbm' was built under R version 3.1.2
```

## Train the KNN model


```r
set.seed(7)
modKNN<-train(classe ~ .,data=training1,method="knn", trControl=control)
```

## Train the RPART model


```r
set.seed(7)
modRpart <- train(classe ~ .,method="rpart",data=training1, trControl=control)
```

Each model has 5 results (5-fold cros validation). The 5 results are compared between the models by accuracy distributions. 

## Collect the resamples


```r
results <- resamples(list(GBM=modGbm, KNN=modKNN, RPART=modRpart))
```

The distributions are summarized in terms of percentiles.
## Summarize the distributions


```r
summary(results)
```

```
## 
## Call:
## summary.resamples(object = results)
## 
## Models: GBM, KNN, RPART 
## Number of resamples: 5 
## 
## Accuracy 
##         Min. 1st Qu. Median   Mean 3rd Qu.   Max. NA's
## GBM   0.9858  0.9858 0.9862 0.9863  0.9869 0.9869    0
## KNN   0.8995  0.9036 0.9043 0.9040  0.9050 0.9075    0
## RPART 0.4889  0.5020 0.5060 0.5177  0.5417 0.5501    0
## 
## Kappa 
##         Min. 1st Qu. Median   Mean 3rd Qu.   Max. NA's
## GBM   0.9820  0.9821 0.9825 0.9827  0.9834 0.9834    0
## KNN   0.8729  0.8780 0.8788 0.8785  0.8799 0.8830    0
## RPART 0.3319  0.3495 0.3526 0.3737  0.4117 0.4227    0
```

## Select the most accurate Model

GDM model is selected as it produces the highest accuracy.

## Predicting for Test Data Set

The GDM model is applied to the validation data set and we measure the performance


```r
pred <- predict(modGbm, testing)
confusionMatrix(pred, testing$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1669    4    0    0    0
##          B    4 1119   17    4    8
##          C    0   15 1004   13    4
##          D    1    1    5  947    7
##          E    0    0    0    0 1063
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9859          
##                  95% CI : (0.9825, 0.9888)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9822          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9970   0.9824   0.9786   0.9824   0.9824
## Specificity            0.9991   0.9930   0.9934   0.9972   1.0000
## Pos Pred Value         0.9976   0.9714   0.9691   0.9854   1.0000
## Neg Pred Value         0.9988   0.9958   0.9955   0.9965   0.9961
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2836   0.1901   0.1706   0.1609   0.1806
## Detection Prevalence   0.2843   0.1958   0.1760   0.1633   0.1806
## Balanced Accuracy      0.9980   0.9877   0.9860   0.9898   0.9912
```

## Predicting for the Test Data Set

The GDM model is applied to the testing data set downloaded.


```r
submitResults <- predict(modGbm, test)
```

## Appendix: Figures

### boxplots of results

```r
bwplot(results)
```

![](M8PA1_files/figure-html/bwplotresults-1.png) 

### dot plots of results

```r
dotplot(results)
```

![](M8PA1_files/figure-html/dotplotresults-1.png) 
