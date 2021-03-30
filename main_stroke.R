# Load libraries
library(tidyverse)
library(caret)
library(scales)

# Load the stroke data
healthcare_data <- read_csv('healthcare-dataset-stroke-data.csv')
# Extract Validation set
set.seed(420, sample.kind = 'Rounding')
test_index <- createDataPartition(healthcare_data$stroke, times = 1, p = 0.5, list = FALSE)
validation_set <- healthcare_data[test_index, ]
stroke <- healthcare_data[-test_index, ]
# Remove variables
rm(test_index, healthcare_data)

# Explore, Munge, Visualize
summary(stroke)

# Smoking status as factor, & reorder to appear better in plots
stroke <- stroke %>% mutate(smoking_status=fct_relevel(as.factor(smoking_status), 
                                                       'smokes',
                                                       'formerly smoked',
                                                       'never smoked',
                                                       'Unknown'))
# Other variables as factors will help models later on
stroke <- stroke %>% mutate(gender=as.factor(gender),
                            hypertension=as.factor(hypertension),
                            heart_disease=as.factor(heart_disease),
                            ever_married=as.factor(ever_married),
                            work_type=as.factor(work_type),
                            Residence_type=as.factor(Residence_type),
                            stroke=as.factor(stroke),
                            bmi=as.numeric(bmi)) # bmi has NA's


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
  ylab('Percentage of strokes that have already occured')
# Probability of stroke in men as a function of age is smooth exponential
# Women have a steep increase later in life

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
     scale_y_continuous(label=scales::percent)
# The people who quite smoking, but got a stroke anyway lasted longer than smokers

# Hypertension (high blood pressure), heart disease, & high bmi
stroke %>% group_by(stroke, hypertension) %>% summarise(n=n()) %>%
  ggplot(aes(x=hypertension, y=n, color=stroke))+
    geom_col(position='fill')


#############################################################
# Ok so we can make some more pretty plots but the data looks
# like there are a bunch of small predictors with no clear
# patterns to take advantage of when predicting
# ---------------------------------------------------------
# Time to make some models
# 
#############################################################


stroke <- stroke %>% select(-bmi)
# Get a test/train set of data
set.seed(69, sample.kind = 'Rounding')
test_index <- createDataPartition(stroke$stroke, times = 1, p = 0.1, list = FALSE)
test <- stroke[test_index, ]
train <- stroke[-test_index, ]
rm(test_index)

# Baseline will be predicting the 'healthy' for entire population
# Sensitivity = 0
# Specificity = 1
mean(train$stroke == 0)

# Tree method
train_tree <- train(stroke ~ .,
                     method='rpart',
                     data=train,
                     tuneGrid=data.frame(cp=seq(0, 0.05, len=25)))
ggplot(train_tree, highlight=T)
plot(train_tree$finalModel, margin = 0.1)
text(train_tree$finalModel)
y_hat_tree <- predict(train_tree, test)
confusionMatrix(y_hat_tree, test$stroke)$overall["Accuracy"]

# Random Forest
train_forest <- train(stroke ~ .,
                      method='rf',
                      nodeSize=1,
                      tuneGrid = data.frame(mtry = seq(50, 200, 25)),
                      data=train)
ggplot(train_forest)
confusionMatrix(predict(train_forest, test), test$stroke)
