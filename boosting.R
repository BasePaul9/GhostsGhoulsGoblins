library(tidyverse)
library(tidymodels)
library(vroom)
install.packages("bonsai")
library(bonsai)
install.packages("lightgbm")
library(lightgbm)

#setwd("./GhostsGhoulsGoblins")
train <- vroom("train.csv") %>%
  mutate(type = as.factor(type))
test <- vroom("test.csv")

boost_recipe <- recipe(type ~., data = train) %>%
  update_role(id, new_role = "id") %>%
  step_mutate_at(color, fn = factor) %>%
  step_dummy(color) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_range(all_numeric_predictors(), min = 0, max = 1)

boost_model <- boost_tree(tree_depth = tune(),
                          trees = tune(),
                          learn_rate = tune()) %>%
  set_engine("lightgbm") %>%
  set_mode("classification")

boost_wf <- workflow() %>%
  add_recipe(boost_recipe) %>%
  add_model(boost_model)

tuning_grid <- grid_regular(tree_depth(), # bestTune = 2000
                            trees(), # bestTune = 8
                            learn_rate(), # bestTune = 0.1
                            levels = 3)

folds <- vfold_cv(train, v = 3, repeats=1)

cv_results <- boost_wf %>%
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(accuracy))

bestTune <- cv_results %>%
  select_best(metric = "accuracy")

final_wf <- boost_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data = train)

boost_preds <- predict(final_wf, new_data = test, type="class")

kag_sub <- boost_preds %>%
  bind_cols(., test) %>% #Bind predictions with test data
  select(id, .pred_class) %>% #Just keep datetime and prediction variables
  rename(type=.pred_class) 

vroom_write(x=kag_sub, file="./Boosting.csv", delim=",")








