library(igraph)
library(rNMF)
library(xgboost)
library(fpc)
library(cluster)
library(gbm)
library(caret)
library(plyr)
library(parallel)
library(snow)

# # read self-test dataset  _____5 fold cv

PARS_human <- read.table(file = "../data/PARS_human/ph_encode_37.csv", header = F,sep=",",skip=1)
PARS_yeast <- read.table(file = "../data/PARS_yeast/py_encode_37.csv", header = F,sep=",",skip=1)
SS_PDB <- read.table(file = "../data/SS_PDB/pdb_encode_37.csv", header = F,sep=",",skip=1)


  ###############
  #Sampling
  ###############
# Number of observations
N <- nrow(PARS_human)
# Number of desired splits
folds <- 5

# Generate indices of holdout observations
holdout <- split(sample(1:N), 1:folds)

  ###############
  #Training model
  ###############
  set.seed(666)
  print("Training the model...")

  #Fold-1
  X_train<-PARS_human[-holdout$`1`,][,c(-1,-2,-151)]
  #Y_trian=labels
  Y_train<-PARS_human[-holdout$`1`,][,151]

  Test_data<-PARS_human[holdout$`1`,][,c(-1,-2,-151)]



  gbm2=caret::train(data.frame(X_train), Y_train, method = "gbm", preProcess = NULL, 
  weights = NULL,trControl=trainControl(method="none")
  ,metric = ifelse(is.factor(Y_train), "Accuracy", "RMSE"),
  maximize = ifelse(metric %in% c("RMSE", "logLoss", "MAE"), FALSE, TRUE),tuneGrid = NULL)

  #############
  #use the model to predict
  #############
  print("Predicting scores...")
  PARS_human_result1 <- predict(gbm2, data.frame(Test_data))
  true_class<-PARS_human[holdout$`1`,][,151]
  class_0_prob<-1-PARS_human_result1
  class_1_prob<-PARS_human_result1
  

  ##########
  #get the ranking
  ##########
  print("Give the ranking...")
  globalRankingOfNegated <- which(sort.int(predictedWeightsGlobal, decreasing = T, index.return = T)$ix == negatedIndexInGlobalTesting)
  localRankingOfNegated <- which(sort.int(predictedWeightsLocal, decreasing = T, index.return = T)$ix == negatedIndexInLocalTesting)
  localRankingOfNegated2 <- which(sort.int(predictedWeightsLocal2, decreasing = T, index.return = T)$ix == negatedIndexInLocalTesting2)
  rankings <- rbind(rankings, c(globalRankingOfNegated, localRankingOfNegated, localRankingOfNegated2))






write.csv(rankings, file = "./test/1-100simi.csv", row.names = F)
write.table(rankings, file = "./test/1-100simi.txt", col.names = T,row.names = F, sep = "\t")





  # gbm1=caret::train(data.frame(X_train), Y_train, method = "gbm", preProcess = NULL, 
  # weights = NULL,trControl=trainControl(method="cv",number=5,verboseIter=TRUE,returnData=TRUE,returnResamp="all")
  # ,metric = ifelse(is.factor(Y_train), "Accuracy", "RMSE"),
  # maximize = ifelse(metric %in% c("RMSE", "logLoss", "MAE"), FALSE,
  # TRUE),tuneGrid = NULL)

