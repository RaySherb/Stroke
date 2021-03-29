# Load libraries
library(tidyverse)
library(caret)
library(scales)
library(psych)
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

# Barplot of stroke & gender
stroke %>% group_by(stroke, gender) %>% summarise(n=n()) %>%
  ggplot(aes(x=as.factor(stroke), fill=gender, y=n))+
  geom_col(position='dodge', color='black')+
  ylab('Count')+
  xlab('Stroke')+
  scale_fill_brewer(palette = "Pastel1")
# The dataset has a lot more females, and about equal amount of strokes
stroke %>% group_by(stroke, gender) %>% summarise(n=n()) %>%
  ggplot(aes(x=gender, fill=as.factor(stroke), y=n))+
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
  ggplot(aes(x=avg_glucose_level, color=as.factor(stroke)))+
  geom_histogram(bins=30)
# Bimodal distribution. high glucose has higher risk of stroke

# Smoking Status
stroke %>% group_by(smoking_status, stroke) %>% summarise(n=n()) %>%
  ggplot(aes(x=as.factor(stroke), y=n, fill=smoking_status))+
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
  ggplot(aes(x=hypertension, y=n, color=as.factor(stroke)))+
    geom_col(position='fill', color='black')


#############################################################
# Ok so we can make some more pretty plots but the data looks
# like there are a bunch of small predictors with no clear
# patterns to take advantage of when predicting
# ---------------------------------------------------------
# Time to make some models
# 
#############################################################

# Remove the solution vector
#y <- stroke$stroke
#stroke <- stroke %>% select(-stroke)
# Get a test/train set of data
set.seed(69, sample.kind = 'Rounding')
test_index <- createDataPartition(y, times = 1, p = 0.1, list = FALSE)
test <- stroke[test_index, ]
train <- stroke[-test_index, ]
rm(test_index)

# Baseline will be predicting the 'healthy' for entire population

# Tree method
train_rpart <- train(stroke ~ .,
                     method='rpart',
                     data=train,
                     tuneGrid=data.frame(cp=seq(0, 0.05, len=25)))
ggplot(train_rpart, highlight=T)
plot(train_rpart$finalModel, margin=0.1)
text(train_rpart$finalModel, cex=0.75)