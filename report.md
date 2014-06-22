Practical Machine Learning Project of Coursera
========================================================

# Introduction

This is the course project of Practical Machine Learning at Coursera.
The purpose of this project is to predict the manner in which they did the exercise. 

While performing the exercise, it is important to have the correct form and movement 
to avoid injury and maximize efficiency. Given the proper equipment, we can
diagnose cases of doing exercises correctly using existing, machine
learning libraries in R. We take a look how this is possible by using the Human 
Activity Recognition dataset.

## Prerequisite Conditions

Loading libraries: 


```r
library(caret)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
library(doMC)
```

```
## Error: there is no package called 'doMC'
```

```r
registerDoMC(cores = 8)
```

```
## Error: could not find function "registerDoMC"
```

```r
set.seed(6348)
```

## Preprocessing  Data

At first look on the data, it is obvious that a significant amount of columns are not availabl. To simplify our model, we remove any column possessing either of these two traits. Afterwards, we should notice that the first eight columns are measurements not pertaining to the sensor.
We manually remove them before running PCA on the remaining data columns.

In order to get an idea of how well our algorithm will perform on the testing data, we create 
a split to make a "cross-validated" data partition. This is an 80/20 split in order to give sufficient data to developing the models while giving a good estimate to testing performance. Later on, we'll let caret train using a 4-fold cross-validation setup.

In order to capture most of the variance of the data and to remove repeated operations,
we pre-process the data with PCA outside of caret's train function. We also center and scale the data in order to treat each column equally. The threshold for the PCA was left at the defaul 0.95 in order to capture most of the variance. Finally, we use the predict function to apply our PCA to the data.

The preprocessing is performed to both the training and testing data like so:


```r
# Read in the data
raw_train_data = read.csv("pml-training.csv", header = TRUE, 
                          na.strings=c("", "NA"))
```

```
## Warning: cannot open file 'pml-training.csv': No such file or directory
```

```
## Error: cannot open the connection
```

```r
raw_test_data = read.csv("pml-testing.csv", header = TRUE, 
                          na.strings=c("", "NA"))
```

```
## Warning: cannot open file 'pml-testing.csv': No such file or directory
```

```
## Error: cannot open the connection
```

```r
# Remove the NA/empty columns
raw_train_data = raw_train_data[, colSums(is.na(raw_train_data)) == 0]
```

```
## Error: object 'raw_train_data' not found
```

```r
raw_test_data = raw_test_data[, colSums(is.na(raw_test_data)) == 0]
```

```
## Error: object 'raw_test_data' not found
```

```r
# Remove non-sensor measurements
raw_train_data = raw_train_data[, 8:length(raw_train_data)]
```

```
## Error: object 'raw_train_data' not found
```

```r
raw_test_data = raw_test_data[, 8:length(raw_test_data)]
```

```
## Error: object 'raw_test_data' not found
```

```r
# Divide the training data into 80/20 training/CV and remove classe for PCA
train_data = raw_train_data[, 1:length(raw_train_data) - 1]
```

```
## Error: object 'raw_train_data' not found
```

```r
test_data = raw_test_data[,1:length(raw_test_data) - 1]
```

```
## Error: object 'raw_test_data' not found
```

```r
# Define the PCA operation
pre_proc = preProcess(rbind(train_data, test_data), 
                      method = c("center", "scale", "pca"))
```

```
## Error: object 'train_data' not found
```

```r
# Apply the PCA to the data
in_train = createDataPartition(y = raw_train_data$classe, p = 0.8, list = FALSE)
```

```
## Error: object 'raw_train_data' not found
```

```r
pca_data = predict(pre_proc, train_data)
```

```
## Error: object 'pre_proc' not found
```

```r
pca_train_data = pca_data[in_train,]
```

```
## Error: object 'pca_data' not found
```

```r
pca_cv_data = pca_data[-in_train,]
```

```
## Error: object 'pca_data' not found
```

```r
pca_test_data = predict(pre_proc, test_data)
```

```
## Error: object 'pre_proc' not found
```

```r
# Create classe variables for easier access
train_classe = raw_train_data[in_train, length(raw_train_data)]
```

```
## Error: object 'raw_train_data' not found
```

```r
cv_classe = raw_train_data[-in_train, length(raw_train_data)]
```

```
## Error: object 'raw_train_data' not found
```

## Data Modeling

Once all of the data has been preprocessed, we can begin to model the data to the class. In the Netflix contest, the top performers used an ensemble of models in their predictions. To model this data, I decided to adopt the same idea and use multiple models to determine whether or not someone is performing the exercise correctly. 

I decided to use the following four models: recursive partitioning and regression trees (**rpart**), random forests (**rf**), linear discriminant analysis (**lda**), and radial support vector machines (**svmRadial**). All of them were picked arbitrarily. Other models such as the generalized boosted regression models (**gbm**) were not selected on the basis of time and processing constraints. 

To ensure that we do not overfit on the training data, we perform cross-validation while training our models. Since we already separated 20% of our training data to get a performance estimate, I decided to perform a 4-fold cross-validation so that the training data and cross-validated partition within the train function is 60% and 20%, respectively, for each of the repeats. 

The model training is performed like so:


```r
# Define k-fold cross-validation training conditions
fit_control = trainControl(method = "repeatedcv", number = 4, repeats = 4)

# Train multiple models under default parameters
rpart_model = train(pca_train_data, train_classe, 
                    method = "rpart", trControl = fit_control)
```

```
## Error: object 'pca_train_data' not found
```

```r
rf_model = train(pca_train_data, train_classe,
                 method = "rf", trControl = fit_control)
```

```
## Error: object 'pca_train_data' not found
```

```r
lda_model = train(pca_train_data, train_classe, 
                  method = "lda", trControl = fit_control)
```

```
## Error: object 'pca_train_data' not found
```

```r
svm_model = train(pca_train_data, train_classe, 
                  method = "svmRadial", trControl = fit_control)
```

```
## Error: object 'pca_train_data' not found
```

After training the models, we could take a look at the in-sample error for each of the individual models. However, we can instead just combine all of them to form an ensemble. To do this, we predict on the training set for each of the models and create a new data frame containing the predictions and the classe. We then fit a random forest over the predictions to the classe, thus forming an ensemble prediction as performed here:


```r
# Predict with each of the trained models
rpart_train_pred = predict(rpart_model, pca_train_data)
```

```
## Error: object 'rpart_model' not found
```

```r
rf_train_pred = predict(rf_model, pca_train_data)
```

```
## Error: object 'rf_model' not found
```

```r
lda_train_pred = predict(lda_model, pca_train_data)
```

```
## Error: object 'lda_model' not found
```

```r
svm_train_pred = predict(svm_model, pca_train_data)
```

```
## Error: object 'svm_model' not found
```

```r
# Fit a model combining the predictions
train_df = data.frame(rpart_pred = rpart_train_pred, rf_pred = rf_train_pred, 
                      lda_pred = lda_train_pred, svm_pred = svm_train_pred, 
                      classe = train_classe)
```

```
## Error: object 'rpart_train_pred' not found
```

```r
ensemble_model = train(classe ~ ., method = "rf", data = train_df)
```

```
## Error: object 'train_df' not found
```

```r
print(ensemble_model)
```

```
## Error: object 'ensemble_model' not found
```

We can get the in-sample error for our ensemble by subtracting with our highest accuracy attained with our model. The in-sample error is:


```r
error = 1 - max(ensemble_model$results$Accuracy)
```

```
## Error: object 'ensemble_model' not found
```

```r
names(error) = "Error"
```

```
## Error: object 'error' not found
```

```r
print(error)
```

```
## Error: object 'error' not found
```

## Prediction with Trained Models

To get the out-of-sample error, we use the ensemble model on the cross-validation data partition that was not included in the training. We run the data through each of the individual models before using the ensemble model.


```r
# Generate predictions on the data
rpart_cv_pred = predict(rpart_model, pca_cv_data)
```

```
## Error: object 'rpart_model' not found
```

```r
rf_cv_pred = predict(rf_model, pca_cv_data)
```

```
## Error: object 'rf_model' not found
```

```r
lda_cv_pred = predict(lda_model, pca_cv_data)
```

```
## Error: object 'lda_model' not found
```

```r
svm_cv_pred = predict(svm_model, pca_cv_data)
```

```
## Error: object 'svm_model' not found
```

```r
# Fit a model combining the predictions
cv_df = data.frame(rpart_pred = rpart_cv_pred, rf_pred = rf_cv_pred, 
                   lda_pred = lda_cv_pred, svm_pred = svm_cv_pred, 
                   classe = cv_classe)
```

```
## Error: object 'rpart_cv_pred' not found
```

```r
cv_pred = predict(ensemble_model, cv_df)
```

```
## Error: object 'ensemble_model' not found
```

By using a confusion matrix, we can get the accuracy of ensemble model on the cross-validation data partition. The expected out-of-sample error is:


```r
cv_confusion_matrix = confusionMatrix(cv_pred, cv_classe)
```

```
## Error: object 'cv_pred' not found
```

```r
error = 1 - cv_confusion_matrix$overall["Accuracy"]
```

```
## Error: object 'cv_confusion_matrix' not found
```

```r
names(error) = "Error"
```

```
## Error: object 'error' not found
```

```r
print(error)
```

```
## Error: object 'error' not found
```

## Predicting on the Test Set

Similar to the cross-validation data partition, we can predict on the test data. Since the test data does not have the class, this section is just outputting the results submitted to Coursera.


```r
# Generate predictions on the data
rpart_test_pred = predict(rpart_model, pca_test_data)
```

```
## Error: object 'rpart_model' not found
```

```r
rf_test_pred = predict(rf_model, pca_test_data)
```

```
## Error: object 'rf_model' not found
```

```r
lda_test_pred = predict(lda_model, pca_test_data)
```

```
## Error: object 'lda_model' not found
```

```r
svm_test_pred = predict(svm_model, pca_test_data)
```

```
## Error: object 'svm_model' not found
```

```r
# Fit a model combining the predictions
test_df = data.frame(rpart_pred = rpart_test_pred, rf_pred = rf_test_pred, 
                     lda_pred = lda_test_pred, svm_pred = svm_test_pred)
```

```
## Error: object 'rpart_test_pred' not found
```

```r
test_pred = predict(ensemble_model, test_df)
```

```
## Error: object 'ensemble_model' not found
```

```r
# Write out predictions to file. Code taken from project submission page.
n = length(test_pred)
```

```
## Error: object 'test_pred' not found
```

```r
for(i in 1:n){
  filename = paste0("problem_id_", i, ".txt")
  write.table(test_pred[i], file = filename, quote = FALSE, 
              row.names = FALSE, col.names = FALSE)
}
```

```
## Error: object 'n' not found
```
