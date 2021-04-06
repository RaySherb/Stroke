# Load
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
#######################################################################################################



# Munge
#######################################################################################################
# Smoking status as factor, & reorder to appear better in plots
stroke <- stroke %>% mutate(smoking_status=fct_relevel(as.factor(smoking_status), 
                                                       'smokes',
                                                       'formerly smoked',
                                                       'Unknown',
                                                       'never smoked'))
# Other variables as factors will help models later on
# Reorder stroke levels for confusion matrix formatting
stroke <- stroke %>% mutate(gender=as.factor(gender),
                            hypertension=as.factor(hypertension),
                            heart_disease=as.factor(heart_disease),
                            ever_married=as.factor(ever_married),
                            work_type=as.factor(work_type),
                            Residence_type=as.factor(Residence_type),
                            stroke=fct_relevel(as.factor(stroke), '1', '0'),
                            bmi=as.numeric(bmi)) # bmi has NA's

# Replace NA values in BMI to mean
stroke$bmi[is.na(stroke$bmi)] <- mean(stroke$bmi, na.rm = TRUE)
#######################################################################################################



# Exploratory Data Visualization
#######################################################################################################
# Barplot of stroke & gender
stroke %>% group_by(stroke, gender) %>% summarise(n=n()) %>%
  ggplot(aes(x=stroke, fill=gender, y=n))+
  geom_col(position='dodge', color='black')+
  ylab('Count')+
  xlab('Stroke')+
  scale_fill_brewer(palette = "Pastel1")
# The dataset has a lot more females, and about equal amount of strokes
stroke %>% group_by(stroke, gender) %>% summarise(n=n()) %>%
  ggplot(aes(x=gender, fill=stroke, y=n))+
  geom_col(position='fill', color='black')+
  scale_y_continuous(label=scales::percent)+
  xlab('Stroke')+
  ylab('Proportion')+
  scale_fill_brewer(palette = "Dark2")

# CDF of age for people with strokes
stroke[stroke$stroke == 1, ] %>% group_by(gender) %>%
  ggplot(aes(x=age, color=gender))+
  stat_ecdf(geom = 'step')+
  scale_y_continuous(label=scales::percent)+
  ylab('Strokes Occured')
# Probability of stroke in women as a function of age is smooth exponential
# men have a steep increase later in life

stroke %>% group_by(stroke) %>%
  ggplot(aes(x=avg_glucose_level, color=stroke))+
  geom_histogram(bins=30)
# Bimodal distribution. high glucose has higher risk of stroke

# Smoking Status
stroke %>% group_by(smoking_status, stroke) %>% summarise(n=n()) %>%
  ggplot(aes(x=stroke, y=n, fill=smoking_status))+
  geom_col(position='fill', color='black')+
  scale_y_continuous(label=scales::percent)
# Seems a lot of smokers never get a stroke
# And people who quite don't necessarily save themselves
# But do they add a few years onto their life?
stroke[stroke$stroke == 1, ] %>% group_by(smoking_status) %>%
     ggplot(aes(x=age, color=smoking_status))+
     stat_ecdf(geom = 'step')+
     scale_y_continuous(label=scales::percent)+
     ylab('Strokes Occured')
# The people who quite smoking, but got a stroke anyway lasted longer than smokers

# Hypertension (high blood pressure), heart disease, & high bmi
stroke %>% group_by(stroke, hypertension) %>% summarise(n=n()) %>%
  ggplot(aes(x=hypertension, y=n, color=stroke))+
  scale_y_continuous(label=scales::percent)+
  ylab('Proportion')+
  geom_col(position='fill')

# BMI
stroke %>% 
  arrange(stroke) %>%
  ggplot(aes(x=bmi, y=avg_glucose_level, color=stroke))+
  geom_point()
#######################################################################################################



# Model 
#######################################################################################################
# Remove variables that negatively impact models
strokes <- stroke %>% select(-bmi, -id)

# Get a test/train set of data
set.seed(69, sample.kind = 'Rounding')
test_index <- createDataPartition(strokes$stroke, times = 1, p = 0.1, list = FALSE)
test <- strokes[test_index, ]
train <- strokes[-test_index, ]
rm(test_index)

# Decision Tree
#----------------------------------------------------
train_tree <- train(stroke ~ .,
                    method='rpart',
                    data=train,
                    tuneGrid=data.frame(cp=seq(0.005, 0.03, len=25)))
# ggplot(train_tree, highlight=T)
y_hat_tree <- predict(train_tree, test)

# Evaluate Tree Model
confusionMatrix(y_hat_tree, test$stroke)
confusionMatrix(y_hat_tree, test$stroke)$byClass
# Sensitivity = 0; Predicting 0 for all cases. F1=NA
roc.curve(test$stroke, y_hat_tree) #ROSE
F_meas(y_hat_tree, test$stroke)
# F1=NA
#######################################################################################################



# Data Balancing
#######################################################################################################
# Unbalanced
unbalanced.table <- table(train$stroke)

# Over-sample
train.over <- ovun.sample(stroke ~ ., data=train, method='over', N=2183*2)$data
over.table <- table(train.over$stroke)

# Under-sample
train.under <- ovun.sample(stroke ~ ., data=train, method='under', N=116*2)$data
under.table <- table(train.under$stroke)

# Over & Under
train.both <- ovun.sample(stroke ~ ., data=train, method='both', p=0.5)$data
both.table <- table(train.both$stroke)

# Inject Synthetic Data
train.rose <- ROSE(stroke ~ ., data=train, seed = 69)$data
rose.table <- table(train.rose$stroke)

# Table of data-balancing 
sample_table <- rbind(unbalanced.table, over.table, under.table, both.table, rose.table)
rownames(sample_table) <- c('Unbalanced', 'Over-Balanced', 'Under-Balanced', 'Both', 'ROSE (Data Injection)')
sample_table

# Select best data-balancing technique based on ROC of tree model
tree.over <- train(stroke ~ ., method='rpart', data=train.over, tuneGrid=data.frame(cp=seq(0, 0.005, len=25)))
tree.under <- train(stroke ~ ., method='rpart', data=train.under, tuneGrid=data.frame(cp=seq(0, 0.005, len=25)))
tree.both <- train(stroke ~ ., method='rpart', data=train.both, tuneGrid=data.frame(cp=seq(0, 0.005, len=25)))
tree.rose <- train(stroke ~ ., method='rpart', data=train.rose, tuneGrid=data.frame(cp=seq(0, 0.005, len=25)))
pred.tree.over <- predict(tree.over, test)
pred.tree.under <- predict(tree.under, test)
pred.tree.both <- predict(tree.both, test)
pred.tree.rose <- predict(tree.rose, test)
roc.curve(test$stroke, y_hat_tree, col=2, lwd=2)
roc.curve(test$stroke, pred.tree.over, add.roc = T, col=3, lwd=2)
roc.curve(test$stroke, pred.tree.under, add.roc = T, col=4, lwd=2)
roc.curve(test$stroke, pred.tree.both, add.roc = T, col=5, lwd=2)
roc.curve(test$stroke, pred.tree.rose, add.roc = T, col=6, lwd=2)
legend('bottomright', c('Not Balanced', 'Over', 'Under', 'Both', 'ROSE'), col=2:6, lwd=2)

# Remove un-needed data & Keep ROSE data
rm(train_tree, y_hat_tree, train.over, train.under, train.both, pred.tree.over, pred.tree.under, pred.tree.both)
rm(tree.over, tree.under, tree.both)
rm(unbalanced.table, over.table, under.table, both.table, rose.table)
#######################################################################################################



# Model Part 2
#######################################################################################################
# Tree
#----------------------------------------------------
pred.tree.rose <- pred.tree.rose %>% fct_relevel('1', '0')
tree.cm <- confusionMatrix(pred.tree.rose, test$stroke)$table
#confusionMatrix(pred.tree.rose, test$stroke)$byClass
tree.f <- F_meas(pred.tree.rose, test$stroke) # beta=0.5
roc.curve(test$stroke, pred.tree.rose, col=2, lwd=2)

# GLM
#----------------------------------------------------
glm.rose <- train(stroke ~ ., method='glm', data=train.rose)
pred.glm.rose <- predict(glm.rose, test) %>% fct_relevel('1', '0')
glm.cm <- confusionMatrix(pred.glm.rose, test$stroke)$table
#confusionMatrix(pred.glm.rose, test$stroke)$byClass
glm.f <- F_meas(pred.glm.rose, test$stroke)
roc.curve(test$stroke, pred.glm.rose, add.roc = T, col=3, lwd=2)

# KNN
#----------------------------------------------------
knn.rose <- train(stroke ~ ., method='knn', data=train.rose)
pred.knn.rose <- predict(knn.rose, test) %>% fct_relevel('1', '0')
knn.cm <- confusionMatrix(pred.knn.rose, test$stroke)$table
#confusionMatrix(pred.knn.rose, test$stroke)$byClass
knn.f <- F_meas(pred.knn.rose, test$stroke) # beta=0.5
roc.curve(test$stroke, pred.knn.rose, add.roc = T, col=4, lwd=2)

# Random Forest
#----------------------------------------------------
rf.rose <- train(stroke ~ ., method='rf', data=train.rose)
pred.rf.rose <- predict(rf.rose, test) %>% fct_relevel('1', '0')
rf.cm <- confusionMatrix(pred.rf.rose, test$stroke)$table
#confusionMatrix(pred.rf.rose, test$stroke)$byClass
rf.f <- F_meas(pred.rf.rose, test$stroke) # beta=0.5
roc.curve(test$stroke, pred.rf.rose, add.roc = T, col=5, lwd=2)

# LDA
#----------------------------------------------------
lda.rose <- train(stroke ~ ., method='lda', data=train.rose)
pred.lda.rose <- predict(lda.rose, test) %>% fct_relevel('1', '0')
lda.cm <- confusionMatrix(pred.lda.rose, test$stroke)$table
#confusionMatrix(pred.lda.rose, test$stroke)$byClass
lda.f <- F_meas(pred.lda.rose, test$stroke) # beta=0.5
roc.curve(test$stroke, pred.lda.rose, add.roc = T, col=6, lwd=2)

# Ensemble Try v2
#----------------------------------------------------
set.seed(69, sample.kind = "Rounding")
# Machine learning models to implement in ensemble
models <- c("glm", "lda", "rpart", "knn", "rf")
# Fit all the models to the trainging data
fits <- lapply(models, function(model){ 
  print(model)
  train(stroke ~ ., method = model, data = train.rose)
}) 
# column names as the model names
names(fits) <- models

# Generate a matrix of predictions
pred <- sapply(fits, function(object) 
  predict(object, newdata = test))

# Each model gets a vote
votes <- rowMeans(pred == "1")
# The majority of votes predicts the result
y_hat <- ifelse(votes > 0.5, "1", "0") %>% fct_relevel('1', '0')
ensemble.cm <- confusionMatrix(as.factor(y_hat), test$stroke)$table
ensemble.f <- F_meas(pred.lda.rose, test$stroke)

legend('bottomright', c('Tree', 'GLM', 'KNN', 'Random Forest', 'LDA', 'Ensemble'), col=c(2:6, 8), lwd=2)


# Table of F values for each method
tibble('Method' = c('Tree', 'GLM', 'KNN', 'Random Forest', 'LDA', 'Ensemble'),
       'F' = c(tree.f, glm.f, knn.f, rf.f, lda.f, ensemble.f))
#######################################################################################################


# Validation Set
#######################################################################################################

# Munge the validation set
validation_set <- validation_set %>% mutate(gender=as.factor(gender),
                            hypertension=as.factor(hypertension),
                            heart_disease=as.factor(heart_disease),
                            ever_married=as.factor(ever_married),
                            work_type=as.factor(work_type),
                            Residence_type=as.factor(Residence_type),
                            stroke=fct_relevel(as.factor(stroke), '1', '0')) %>%
  select(-id, -bmi)

# Data injection on complete strokes data set
final.rose <- ROSE(stroke ~ ., data=strokes, seed = 69)$data

# Train with complete strokes data set
final.model <- train(stroke ~ ., method='glm', data=final.rose)
final.pred <- predict(final.model, validation_set) %>% fct_relevel('1', '0')
final.cm <- confusionMatrix(final.pred, validation_set$stroke)$table
#confusionMatrix(pred.glm.rose, test$stroke)$byClass
final.f <- F_meas(pred.glm.rose, test$stroke)







