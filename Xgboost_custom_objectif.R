library(ISLR)
library(xgboost)
library(tidyverse)
library(Metrics)

# Data
df = ISLR::Hitters %>% select(Salary,AtBat,Hits,HmRun,Runs,RBI,Walks,Years,CAtBat,
                              CHits,CHmRun,CRuns,CRBI,CWalks,PutOuts,Assists,Errors)
df = df[complete.cases(df),]
train = df[1:150,]
test = df[151:nrow(df),]

# XGBoost Matrix
dtrain <- xgb.DMatrix(data=as.matrix(train[,-1]),label=as.matrix(train[,1]))
dtest <- xgb.DMatrix(data=as.matrix(test[,-1]),label=as.matrix(test[,1]))
watchlist <- list(eval = dtest)

# Custom objective function (squared error)
myobjective <- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  grad <- (preds-labels)    
  hess <- rep(1, length(labels))                
  return(list(grad = grad, hess = hess))
}

# Custom Metric
evalerror <- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  err <- (preds-labels)^2        
  return(list(metric = "MyError", value = mean(err)))   
}

# Custom Model
param1 <- list(booster = 'gbtree', learning_rate = 0.1, objective = myobjective 
               , eval_metric = evalerror, set.seed = 2020)

xgb1 <- xgb.train(params = param1, data = dtrain, nrounds = 500, watchlist, maximize = FALSE,
                  early_stopping_rounds = 5)

pred1 = predict(xgb1, dtest)
mae1 = mae(test$Salary, pred1)
mae1

## Normal Model
param2 <- list(booster = 'gbtree', learning_rate = 0.1, objective = "reg:squarederror", set.seed = 2020)

xgb2 <- xgb.train(params = param2, data = dtrain, nrounds = 500, watchlist, maximize = FALSE
                  , early_stopping_rounds = 5)

pred2 = predict(xgb2, dtest)
mae2 = mae(test$Salary, pred2)
print(list(mae1, mae2))
