library(igraph)
library(rNMF)
library(xgboost)
library(fpc)
library(cluster)
library(gbm)
library(caret)
library(plyr)
library(parallel)
library(stringr)

# # read self-test dataset  _____5 fold cv

PARS_human <- read.table(file = "../data/PARS_human/ph_encode_37.csv", header = F,sep=",",skip=1)



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
t1<-Sys.time()
Result<-matrix(rep(0,25),5,5)
colnames(Result)<-c("AUC","Precision","Recall","F","Accuracy")
for(i in 1:5){
#Fold-1
  X_train<-PARS_human[-holdout[[i]],][,c(-1,-2,-151)]
  #Y_trian=labels
  Y_train<-PARS_human[-holdout[[i]],][,151]

 # Test_data<-PARS_human[holdout[[i]],][,c(-1,-2,-151)]
 Test_data<-PARS_human[holdout[[i]],][,c(-1,-2,-151)]


trControl<-trainControl(method = "cv",number = 3)
gbmGrid <-  expand.grid(interaction.depth = c(5,9), 
                        n.trees = c(400,1000), 
                        shrinkage = c(0.03,0.1),
                        n.minobsinnode = 10)
#test maximization
  gbm2=caret::train(data.frame(X_train), Y_train, method = "gbm", preProcess = "pca", 
  weights = NULL,trControl=trControl,metric = "RMSE",tuneGrid = gbmGrid)
#original
# trControl<-trainControl(method = "cv",number = 5)
# gbm1=caret::train(data.frame(X_train), Y_train, method = "gbm", preProcess = NULL, 
#   weights = NULL,trControl=trControl ,metric = "RMSE",
#   maximize = ifelse(metric %in% c("RMSE", "logLoss", "MSPE","MSAE","MAE"), FALSE,TRUE),tuneGrid = NULL)



  #############
  #use the model to predict
  #############
  print("Predicting scores...")
  PARS_human_result1 <- predict(gbm2, data.frame(Test_data))
  true_class<-str_glue("Class{PARS_human[holdout[[i]],][,151]}")
  class_1_prob<-PARS_human_result1
  test_set <- data.frame(obs = true_class,
                         Class1 = class_1_prob)
  test_set$Class0 <- 1 - test_set$Class1
  test_set$pred <- factor(ifelse(test_set$Class1 >= .5, "Class1", "Class0"))
  #get the scores
  ACC<-confusionMatrix(data = test_set$pred, reference = test_set$obs, mode = "prec_recall")$overall["Accuracy"]
  unitResult<-prSummary(test_set, lev = levels(test_set$obs))
  Result[i,]<-t(as.matrix(c(unitResult,ACC)))
}
t2<-Sys.time()
print("Time we need is:")
print(t2-t1)  
write.csv(Result, file = "../result/PARS-human-5fold.csv", row.names = F)




############################
#22222222222222222222222222
############################
t1<-Sys.time()
PARS_human <- read.table(file = "../data/PARS_yeast/py_encode_37.csv", header = F,sep=",",skip=1)
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

Result<-matrix(rep(0,25),5,5)
colnames(Result)<-c("AUC","Precision","Recall","F","Accuracy")
for(i in 1:5){
#Fold-1
  X_train<-PARS_human[-holdout[[i]],][,c(-1,-2,-151)]
  #Y_trian=labels
  Y_train<-PARS_human[-holdout[[i]],][,151]

 # Test_data<-PARS_human[holdout[[i]],][,c(-1,-2,-151)]
 Test_data<-PARS_human[holdout[[i]],][,c(-1,-2,-151)]


trControl<-trainControl(method = "cv",number = 3)
gbmGrid <-  expand.grid(interaction.depth = c(5,9), 
                        n.trees = c(400,1000), 
                        shrinkage = c(0.03,0.1),
                        n.minobsinnode = 10)
#test maximization
  gbm2=caret::train(data.frame(X_train), Y_train, method = "gbm", preProcess = "pca", 
  weights = NULL,trControl=trControl,metric = "RMSE",tuneGrid = gbmGrid)
#original
# trControl<-trainControl(method = "cv",number = 5)
# gbm1=caret::train(data.frame(X_train), Y_train, method = "gbm", preProcess = NULL, 
#   weights = NULL,trControl=trControl ,metric = "RMSE",
#   maximize = ifelse(metric %in% c("RMSE", "logLoss", "MSPE","MSAE","MAE"), FALSE,TRUE),tuneGrid = NULL)



  #############
  #use the model to predict
  #############
  print("Predicting scores...")
  PARS_human_result1 <- predict(gbm2, data.frame(Test_data))
  true_class<-str_glue("Class{PARS_human[holdout[[i]],][,151]}")
  class_1_prob<-PARS_human_result1
  test_set <- data.frame(obs = true_class,
                         Class1 = class_1_prob)
  test_set$Class0 <- 1 - test_set$Class1
  test_set$pred <- factor(ifelse(test_set$Class1 >= .5, "Class1", "Class0"))
  #get the scores
  ACC<-confusionMatrix(data = test_set$pred, reference = test_set$obs, mode = "prec_recall")$overall["Accuracy"]
  unitResult<-prSummary(test_set, lev = levels(test_set$obs))
  Result[i,]<-t(as.matrix(c(unitResult,ACC)))
}
t2<-Sys.time()
print("Time we need is:")
print(t2-t1)
write.csv(Result, file = "../result/PARS-yeast-5fold.csv", row.names = F)




############################
#3333333333333333333333333
############################
t1<-Sys.time()
PARS_human <- read.table(file = "../data/SS_PDB/pdb_encode_37.csv", header = F,sep=",",skip=1)
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

Result<-matrix(rep(0,25),5,5)
colnames(Result)<-c("AUC","Precision","Recall","F","Accuracy")
for(i in 1:5){
#Fold-1
  X_train<-PARS_human[-holdout[[i]],][,c(-1,-2,-151)]
  #Y_trian=labels
  Y_train<-PARS_human[-holdout[[i]],][,151]

 # Test_data<-PARS_human[holdout[[i]],][,c(-1,-2,-151)]
 Test_data<-PARS_human[holdout[[i]],][,c(-1,-2,-151)]


trControl<-trainControl(method = "cv",number = 3)
gbmGrid <-  expand.grid(interaction.depth = c(5,9), 
                        n.trees = c(400,1000), 
                        shrinkage = c(0.03,0.1),
                        n.minobsinnode = 10)
#test maximization
  gbm2=caret::train(data.frame(X_train), Y_train, method = "gbm", preProcess = "pca", 
  weights = NULL,trControl=trControl,metric = "RMSE",tuneGrid = gbmGrid)
#original
# trControl<-trainControl(method = "cv",number = 5)
# gbm1=caret::train(data.frame(X_train), Y_train, method = "gbm", preProcess = NULL, 
#   weights = NULL,trControl=trControl ,metric = "RMSE",
#   maximize = ifelse(metric %in% c("RMSE", "logLoss", "MSPE","MSAE","MAE"), FALSE,TRUE),tuneGrid = NULL)



  #############
  #use the model to predict
  #############
  print("Predicting scores...")
  PARS_human_result1 <- predict(gbm2, data.frame(Test_data))
  true_class<-str_glue("Class{PARS_human[holdout[[i]],][,151]}")
  class_1_prob<-PARS_human_result1
  test_set <- data.frame(obs = true_class,
                         Class1 = class_1_prob)
  test_set$Class0 <- 1 - test_set$Class1
  test_set$pred <- factor(ifelse(test_set$Class1 >= .5, "Class1", "Class0"))
  #get the scores
  ACC<-confusionMatrix(data = test_set$pred, reference = test_set$obs, mode = "prec_recall")$overall["Accuracy"]
  unitResult<-prSummary(test_set, lev = levels(test_set$obs))
  Result[i,]<-t(as.matrix(c(unitResult,ACC)))
}
t2<-Sys.time()
print("Time we need is:")
print(t2-t1)
write.csv(Result, file = "../result/SS_PDB-5fold.csv", row.names = F)