library(tidyverse)
library(tidymodels)
library(vroom)
install.packages("discrim")
library(discrim)
install.packages("themis")
library(themis)
library(embed)

#setwd("./GhostsGhoulsGoblins")
train <- vroom("train.csv") %>%
  mutate(type = as.factor(type))
test <- vroom("test.csv")

recipe <- recipe(type ~., data = train) %>%
  step_mutate_at(color, fn = factor) %>%
  step_normalize(all_numeric_predictors())


baked_data <- bake(prep(recipe), new_data = train)

####                 SUPPORT VECTOR MACHINES                  ####

svmRadial <- svm_rbf(rbf_sigma = tune(), 
                     cost = tune()) %>% # set or tune
  set_mode("classification") %>%
  set_engine("kernlab")

svm_wf <- workflow() %>%
  add_recipe(recipe) %>%
  add_model(svmRadial)

tuning_grid <- grid_regular(rbf_sigma(), #bestTune = 0.00316
                            cost(), #cost = 2.38 or 32
                            levels = 5) ## L^2 total tuning possibilities

folds <- vfold_cv(train, v = 5, repeats=1)

cv_results <- svm_wf %>%
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(accuracy))

bestTune <- cv_results %>%
  select_best(metric = "accuracy")

final_wf <- svm_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data = train)

svm_preds <- predict(final_wf, new_data = test, type="class")

kag_sub <- svm_preds %>%
  bind_cols(., test) %>% #Bind predictions with test data
  select(id, .pred_class) %>% #Just keep datetime and prediction variables
  rename(type=.pred_class) 

vroom_write(x=kag_sub, file="./SVM.csv", delim=",")

####                 NAIVE BAYES                 ####
nb_recipe <- recipe(type ~., data = train) %>%
  # update_role(id, new_role = "id") %>%
  step_mutate_at(color, fn = factor) %>%
  step_lencode_glm(all_nominal_predictors(), outcome = vars(type)) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_smote(all_outcomes(), neighbors = 4)

baked_data <- bake(prep(nb_recipe), new_data = train)

nb_model <- naive_Bayes(Laplace = tune(),
                        smoothness = tune()) %>%
  set_mode("classification") %>%
  set_engine("naivebayes") 

nb_wf <- workflow() %>%
  add_recipe(nb_recipe) %>%
  add_model(nb_model)

tuning_grid <- grid_regular(Laplace(),
                            smoothness(),
                            levels = 5) ## L^2 total tuning possibilities

folds <- vfold_cv(train, v = 5, repeats=1)

cv_results <- nb_wf %>%
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(accuracy))

bestTune <- cv_results %>%
  select_best(metric = "accuracy")

final_wf <- nb_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data = train)

nb_preds <- predict(final_wf, new_data = test, type="class")

kag_sub <- nb_preds %>%
  bind_cols(., test) %>% #Bind predictions with test data
  select(id, .pred_class) %>% #Just keep datetime and prediction variables
  rename(type=.pred_class) 

vroom_write(x=kag_sub, file="./NaiveBayes.csv", delim=",")

