# Included in MAIN
#######################################################################################################
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(scales)) install.packages("scales", repos = "http://cran.us.r-project.org")
if(!require(ROSE)) install.packages("ROSE", repos = "http://cran.us.r-project.org")
# Load libraries
library(tidyverse)
library(caret) # Machine learning algos
library(scales) # Custom ggplot axis scales
library(ROSE) # Data balancing
# Data source: https://www.kaggle.com/fedesoriano/stroke-prediction-dataset
# Load the stroke data
healthcare_data <- read_csv('healthcare-dataset-stroke-data.csv')
# Extract Validation set
set.seed(420, sample.kind = 'Rounding')
test_index <- createDataPartition(healthcare_data$stroke, times = 1, p = 0.5, list = FALSE)
validation_set <- healthcare_data[test_index, ]
stroke <- healthcare_data[-test_index, ]
# Remove un-needed variables
rm(test_index, healthcare_data)
# Smoking status as factor, & reorder to appear better in plots
stroke <- stroke %>% mutate(smoking_status=fct_relevel(as.factor(smoking_status), 
                                                       'smokes',
                                                       'formerly smoked',
                                                       'Unknown',
                                                       'never smoked'))
# Other variables as factors will help models later on
stroke <- stroke %>% mutate(gender=as.factor(gender),
                            hypertension=as.factor(hypertension),
                            heart_disease=as.factor(heart_disease),
                            ever_married=as.factor(ever_married),
                            work_type=as.factor(work_type),
                            Residence_type=as.factor(Residence_type),
                            stroke=fct_relevel(as.factor(stroke), '1', '0'))
# Remove variables that negatively impact models
strokes <- stroke %>% select(gender, age, hypertension, heart_disease, ever_married,
                             work_type, Residence_type, avg_glucose_level, smoking_status, stroke)
#strokes <- strokes %>% select(-id)
# Get a test/train set of data
set.seed(69, sample.kind = 'Rounding')
test_index <- createDataPartition(strokes$stroke, times = 1, p = 0.1, list = FALSE)
test <- strokes[test_index, ]
train <- strokes[-test_index, ]
rm(test_index)
#######################################################################################################


# Models
#######################################################################################################
# Descision Tree
#----------------------------------------------------
train_tree <- train(stroke ~ .,
                    method='rpart',
                    data=train,
                    tuneGrid=data.frame(cp=seq(0.005, 0.03, len=25)))
ggplot(train_tree, highlight=T)
y_hat_tree <- predict(train_tree, test)

# Evaluate Tree Model
confusionMatrix(y_hat_tree, test$stroke)
confusionMatrix(y_hat_tree, test$stroke)$byClass
# Sensitivity = 0; Predicting 0 for all cases. F1=NA
roc.curve(test$stroke, y_hat_tree) #ROSE
F_meas(y_hat_tree, test$stroke)
# F1=NA
#######################################################################################################


# Change the beta value
######################################################################################################
# Tree2
#----------------------------------------------------
train_tree <- train(stroke ~ ., method='rpart', data=train,
                    tuneGrid=data.frame(cp=seq(0, 0.005, len=25)))
y_hat_tree <- predict(train_tree, test)
# Evaluate Tree Model
confusionMatrix(y_hat_tree, test$stroke)$overall["Accuracy"]
accuracy.meas(test$stroke, y_hat_tree) #ROSE
roc.curve(test$stroke, y_hat_tree) #ROSE
# GLM
#----------------------------------------------------
glm.rose <- train(stroke ~ ., method='glm', data=train)
pred.glm.rose <- predict(glm.rose, test)
roc.curve(test$stroke, pred.glm.rose, add.roc = T, col=3, lwd=2)

# KNN
#----------------------------------------------------
knn.rose <- train(stroke ~ ., method='knn', data=train)
pred.knn.rose <- predict(knn.rose, test)
roc.curve(test$stroke, pred.knn.rose, add.roc = T, col=4, lwd=2)

# Random Forest
#----------------------------------------------------
rf.rose <- train(stroke ~ ., method='rf', data=train)
pred.rf.rose <- predict(rf.rose, test)
roc.curve(test$stroke, pred.rf.rose, add.roc = T, col=5, lwd=2)

# LDA
#----------------------------------------------------
lda.rose <- train(stroke ~ ., method='lda', data=train)
pred.lda.rose <- predict(lda.rose, test)
roc.curve(test$stroke, pred.lda.rose, add.roc = T, col=6, lwd=2)

# Ensemble
#----------------------------------------------------
set.seed(69, sample.kind = "Rounding")
# Machine learning models to implement in ensemble
models <- c("glm", "lda", "naive_bayes", "svmLinear", "knn", "multinom", "lda", "rf")
# Fit all the models to the trainging data
fits <- lapply(models, function(model){ 
  print(model)
  train(stroke ~ ., method = model, data = train)
}) 
# column names as the model names
names(fits) <- models

# Generate a matrix of predictions
pred <- sapply(fits, function(object) 
  predict(object, newdata = test))
dim(pred)
# Get the average accuracy of each model
acc <- colMeans(pred == test$stroke)
# Get the mean accuracy of all models
mean(acc)
# Each model gets a vote
votes <- rowMeans(pred == "1")
# The majority of votes wins
y_hat <- ifelse(votes > 0.5, "1", "0")
mean(y_hat == test$stroke)
confusionMatrix(as.factor(y_hat), test$stroke)

legend('bottomright', c('Tree', 'GLM', 'KNN', 'Random Forest', 'LDA', 'Ensemble'), col=c(2:6, 8), lwd=2)