library(tidyverse)
library(tidymodels)
library(vroom)

setwd("./GhostsGhoulsGoblins")
train_missing <- vroom("trainWithMissingValues.csv")
train <- vroom("train.csv")
test <- vroom("test.csv")

recipe <- recipe(type ~ ., data = train_missing) %>%
  step_mutate_at(color, fn = factor) %>%
  step_impute_bag(hair_length, impute_with = imp_vars(has_soul, color), trees = 3) %>%
  step_impute_knn(rotting_flesh, impute_with = imp_vars(color, has_soul, hair_length), neighbors = 10) %>%
  step_impute_linear(bone_length, impute_with = imp_vars(color, has_soul, hair_length, rotting_flesh))

baked_data <- bake(prep(recipe), new_data = train_missing)

rmse_vec(train[is.na(train_missing)],
         baked_data[is.na(train_missing)])
