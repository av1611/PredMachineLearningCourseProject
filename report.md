Practical Machine Learning Project
========================================================

# Introduction

This is the course project of Practical Machine Learning from coursera.
The goal of this project is to predict the manner in which they did the exercise. 


```
## Loading required package: lattice
## Loading required package: ggplot2
```

```
## Error: there is no package called 'kernlab'
```

```
## Error: there is no package called 'randomForest'
```

# Loading data

```r
raw_training <- read.csv('pml-training.csv')
```

```
## Warning: cannot open file 'pml-training.csv': No such file or directory
```

```
## Error: cannot open the connection
```

```r
raw_testing <- read.csv('pml-testing.csv')
```

```
## Warning: cannot open file 'pml-testing.csv': No such file or directory
```

```
## Error: cannot open the connection
```

```r
set.seed(8888)
inTrain <- createDataPartition(raw_training$classe, list=FALSE, p=.9)
```

```
## Error: object 'raw_training' not found
```

```r
training = raw_training[inTrain,]
```

```
## Error: object 'raw_training' not found
```

```r
testing = raw_training[-inTrain,]
```

```
## Error: object 'raw_training' not found
```


# Preprocessing

```r
nzv <- nearZeroVar(training)
```

```
## Error: object 'training' not found
```

```r
training <- training[-nzv]
```

```
## Error: object 'training' not found
```

```r
testing <- testing[-nzv]
```

```
## Error: object 'testing' not found
```

```r
raw_testing <- raw_testing[-nzv]
```

```
## Error: object 'raw_testing' not found
```

```r
training <- training[-5]
```

```
## Error: object 'training' not found
```

```r
testing <- testing[-5]
```

```
## Error: object 'testing' not found
```

```r
raw_testing <- raw_testing[-5]
```

```
## Error: object 'raw_testing' not found
```

```r
num_features_idx = which(lapply(training,class) %in% c('numeric')  )
```

```
## Error: object 'training' not found
```

```r
preModel <- preProcess(training[,num_features_idx], method=c('knnImpute'))
```

```
## Error: object 'training' not found
```
In some situations, the data generating mechanism can create predictors that only have a single unique value (i.e. a "zero-variance predictor"). For many models (excluding tree-based models), this may cause the model to crash or the fit to be unstable.
PreProcess can be used to impute data sets based only on information in the training set. One method of doing this is with K-nearest neighbors.

# Get preprocessed data

```r
ptraining <- cbind(training$classe, predict(preModel, training[,num_features_idx]))
```

```
## Error: object 'training' not found
```

```r
ptesting <- cbind(testing$classe, predict(preModel, testing[,num_features_idx]))
```

```
## Error: object 'testing' not found
```

```r
prtesting <- predict(preModel, raw_testing[,num_features_idx])
```

```
## Error: object 'preModel' not found
```

```r
names(ptraining)[1] <- 'classe'
```

```
## Error: object 'ptraining' not found
```

```r
names(ptesting)[1] <- 'classe'
```

```
## Error: object 'ptesting' not found
```

```r
ptraining[is.na(ptraining)] <- 0
```

```
## Error: object 'ptraining' not found
```

```r
ptesting[is.na(ptesting)] <- 0
```

```
## Error: object 'ptesting' not found
```

```r
prtesting[is.na(prtesting)] <- 0
```

```
## Error: object 'prtesting' not found
```

# Fit model and corss validation

```r
rf_model  <- randomForest(classe ~ ., ptraining)
```

```
## Error: could not find function "randomForest"
```

## In-sample accuracy

```r
training_pred <- predict(rf_model, ptraining) 
```

```
## Error: object 'rf_model' not found
```

```r
print(table(training_pred, ptraining$classe))
```

```
## Error: object 'training_pred' not found
```

```r
print(mean(training_pred == ptraining$classe))
```

```
## Error: object 'training_pred' not found
```

## Out-of-sample accuracy

```r
testing_pred <- predict(rf_model, ptesting) 
```

```
## Error: object 'rf_model' not found
```

```r
print(table(testing_pred, ptesting$classe))
```

```
## Error: object 'testing_pred' not found
```

```r
print(mean(testing_pred == ptesting$classe))
```

```
## Error: object 'testing_pred' not found
```

### Confusion Matrix: 

```r
print(confusionMatrix(testing_pred, ptesting$classe))
```

```
## Error: object 'testing_pred' not found
```

# Apply model to the test set

```r
answers <- predict(rf_model, prtesting) 
```

```
## Error: object 'rf_model' not found
```

```r
answers
```

```
## Error: object 'answers' not found
```




# Conclusion
We are able to provide very good prediction of weight lifting style as measured with accelerometers.
