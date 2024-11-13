library(tidyverse)
library(tidymodels)
library(vroom)
library(keras)

#setwd("./GhostsGhoulsGoblins")
train <- vroom("train.csv") %>%
  mutate(type = as.factor(type))
test <- vroom("test.csv")

nn_recipe <- recipe(type ~., data = train) %>%
  update_role(id, new_role = "id") %>%
  step_mutate_at(color, fn = factor) %>%
  step_dummy(color) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_range(all_numeric_predictors(), min = 0, max = 1)

nn_model <- mlp(hidden_units = tune(), #BestTune = 10
                epochs = 50) %>%
  set_engine("keras") %>%
  set_mode("classification")

nn_wf <- workflow() %>%
  add_recipe(nn_recipe) %>%
  add_model(nn_model)

tuning_grid <- grid_regular(hidden_units(range=c(1, 10)),
                         levels = 5)
folds <- vfold_cv(train, v = 5, repeats=1)

cv_results <- nn_wf %>%
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(accuracy))

cv_results %>%
  collect_metrics() %>%
  filter(.metric == "accuracy") %>%
  ggplot(aes(x=hidden_units, y=mean)) + geom_line()

bestTune <- cv_results %>%
  select_best(metric = "accuracy")

final_wf <- nn_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data = train)

nn_preds <- predict(final_wf, new_data = test, type="class")

kag_sub <- nn_preds %>%
  bind_cols(., test) %>% #Bind predictions with test data
  select(id, .pred_class) %>% #Just keep datetime and prediction variables
  rename(type=.pred_class) 

vroom_write(x=kag_sub, file="./NeuralNet.csv", delim=",")

















