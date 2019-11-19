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
library(randomForest)
library(bst)
# # read self-test dataset  _____5 fold cv
#PARS_human <- read.table(file = "../data/PARS_human/ph_encode_37.csv", header = F,sep=",",skip=1)
#PARS_human <- read.table(file = "../data/PARS_yeast/py_encode_37.csv", header = F,sep=",",skip=1)
PARS_human <- read.table(file = "../data/SS_PDB/pdb_encode_37.csv", header = F,sep=",",skip=1)


###############
#Sampling
###############
# Number of observations
N <- nrow(PARS_human)
# Number of desired splits
folds <- 5
set.seed(666)
# Generate indices of holdout observations
holdout <- split(sample(1:N), 1:folds)

###############
#Training model
###############

print("Training the model...")

Result<-matrix(rep(0,25),5,5)
colnames(Result)<-c("AUC","Precision","Recall","F","Accuracy")
for(i in 1){
  #Fold-1
  ############NO-RESAMPLING##########
  # X_train<-PARS_human[-holdout[[i]],][,c(-1,-2,-151)]
  # #Y_trian=labels
  # Y_train<-PARS_human[-holdout[[i]],][,151]
  
  # # Test_data<-PARS_human[holdout[[i]],][,c(-1,-2,-151)]
  # Test_data<-PARS_human[holdout[[i]],][,c(-1,-2,-151)]
  # Y_test<-PARS_human[holdout[[i]],][,151]
  ############RESAMPLING#############
  X_train0<-PARS_human[-holdout[[i]],][,c(-1,-2,-151)]
  #Y_trian=labels
  Y_train0<-PARS_human[-holdout[[i]],][,151]
  
  # Test_data<-PARS_human[holdout[[i]],][,c(-1,-2,-151)]
  Test_data0<-PARS_human[holdout[[i]],][,c(-1,-2,-151)]
  Y_test0<-PARS_human[holdout[[i]],][,151]
  threshold= 30    # the larger threshold the less samples selected
  X_train<-X_train0[which(rowSums(X_train0)>threshold),]
  Y_train<-Y_train0[which(rowSums(X_train0)>threshold)]
  Test_data<-Test_data0[which(rowSums(Test_data0)>threshold),]
  Y_test<-Y_test0[which(rowSums(Test_data0)>threshold)]

  #####################training###################
  trControl<-trainControl(method = "cv",number =2,verboseIter=TRUE)
  #trControl<-trainControl(method="none",verboseIter=TRUE)
  # gbmGrid <-  expand.grid(interaction.depth = c(1, 5, 9), 
  #                         n.trees = c(50,100,400,800,1000,1500), 
  #                         shrinkage = c(0.03,0.1),
  #                         n.minobsinnode = c(10,20))
  gbmGrid <-  expand.grid(interaction.depth = 3, 
                          n.trees = 100, 
                          shrinkage = 0.1,verboseIter=TRUE,
                          n.minobsinnode = 10)
  
  xgbTreeGrid<- expand.grid(nrounds=500,max_depth=c(4,8),eta=c(0.1),
                            gamma=c(0.5,1),colsample_bytree=c(0.8),min_child_weight=c(0.05),
                            subsample=c(0.7))#0.81
  #xgbLinearGrid<-expand.grid(nrounds=c(800,1000,1200,1500),lambda=c(0.8,1,2,2.5),alpha=c(0.8,1,2,2.5),eta=c(0.04,0.1))
  #xgbLinearGrid<-expand.grid(nrounds=c(600),lambda=c(3),alpha=c(3),eta=c(0.08))   #best for ph
  xgbLinearGrid<-expand.grid(nrounds=c(500),lambda=c(2.5),alpha=c(2.5),
                             eta=c(0.02))   #best for ph
  xgbDARTGrid<-expand.grid(nrounds=200,max_depth=5,eta=0.04,
                           gamma=0.4,subsample=1,colsample_bytree=0.8,
                           rate_drop=0.2,skip_drop=0.1, min_child_weight=0.1)
  glmnetGrid<-expand.grid(alpha=3,lambda=3)
  pcrGrid<-expand.grid(ncomp=140)
  #test PCA BEST!
  gbm2=caret::train(data.frame(X_train), Y_train, method = "xgbLinear", preProcess = NULL, 
                    weights = NULL,trControl=trControl,metric = "RMSE",tuneGrid=xgbLinearGrid)
  
  #original
  # trControl<-trainControl(method = "cv",number = 5)
  # gbm1=caret::train(data.frame(X_train), Y_train, method = "gbm", preProcess = NULL, 
  #   weights = NULL,trControl=trControl ,metric = "RMSE",
  #   maximize = ifelse(metric %in% c("RMSE", "logLoss", "MSPE","MSAE","MAE"), FALSE,TRUE),tuneGrid = NULL)
  
  
  # #############use as classifications#############
  #  trControl<-trainControl(method = "cv",number =5)
  # gbmGrid <-  expand.grid(interaction.depth = 3, 
  #                         n.trees = 200, 
  #                         shrinkage = 0.1,
  #                         n.minobsinnode = 10)
  #  trControl<-trainControl(method = "none")
  
  # mtryGrid<- expand.grid(mtry = 100)
  # bstLmGrid<-expand.grid(nu=0.1,mstop=500)
  # lmGrid<-expand.grid(intercept=0.7) #0.595
  # cubGrid<-expand.grid(committees=5,neighbors=3)
  # #test PCA BEST!
  #   gbm2=caret::train(data.frame(X_train), Y_train, method = "rf",trControl=trControl,metric = "RMSE"
  #     ,preProcess=NULL)
  
  
  #   #############
  #   #use GBM model to predict
  #   #############
  print("Predicting scores...")
  
  PARS_human_result1 <- predict(gbm2, data.frame(Test_data))
  PARS_human_result1 <- (PARS_human_result1-min(PARS_human_result1))/(max(PARS_human_result1)-min(PARS_human_result1))
 # true_class<-str_glue("Class{PARS_human[holdout[[i]],][,151]}")
  true_class<-str_glue("Class{Y_test}")
  class_1_prob<-PARS_human_result1
  test_set <- data.frame(obs = true_class,
                         Class1 = class_1_prob)
  test_set$Class0 <- 1 - test_set$Class1
  test_set$pred <- factor(ifelse(test_set$Class1 >= .5, "Class1", "Class0"))
  #get the scores
  Confu_matrix<-confusionMatrix(data = test_set$pred, reference = test_set$obs, mode = "prec_recall")
  ACC<- Confu_matrix$overall["Accuracy"]
  unitResult<-prSummary(test_set, lev = levels(test_set$obs))
  Result[i,]<-t(as.matrix(c(unitResult,ACC)))
  
  
  ######################USE XGBOOST##########################
  
  # parameters <- list(eta = 1, maxDepth = 8, lambda = 1, gamma = 0)
  # xgboostLoocv <- xgboost(data = as.matrix(X_train), booster = "gbtree", 
  #                         label = Y_train, params = parameters, nthread = 2, nrounds = 4, 
  #                         objective = "binary:logitraw")
  
  # PARS_human_result1 <- predict(xgboostLoocv, as.matrix(Test_data))
  # PARS_human_result1 <- (PARS_human_result1-min(PARS_human_result1))/(max(PARS_human_result1)-min(PARS_human_result1))
  
  # print("Predicting scores...")
  # # PARS_human_result1<-predict(xgboostLoocv, as.matrix(Test_data))
  # true_class<-str_glue("Class{PARS_human[holdout[[i]],][,151]}")
  # class_1_prob<-PARS_human_result1
  # test_set <- data.frame(obs = true_class,
  #                        Class1 = class_1_prob)
  # test_set$Class0 <- 1 - test_set$Class1
  # test_set$pred <- factor(ifelse(test_set$Class1 >= .5, "Class1", "Class0"))
  # #get the scores
  # Confu_matrix<-confusionMatrix(data = test_set$pred, reference = test_set$obs, mode = "prec_recall")
  # ACC<- Confu_matrix$overall["Accuracy"]
  # unitResult<-prSummary(test_set, lev = levels(test_set$obs))
  # Result[i,]<-t(as.matrix(c(unitResult,ACC)))
  ###################use RF#################
  # Test_data_label<-PARS_human[holdout[[i]],][,c(151)]
  # RFmodel<-randomForest(X_train,y=Y_train,xtest=Test_data,ytest=Test_data_label,
  #   ntree=50,mtry=50)
  
  
  
  
  
}






#write()



# gbm1=caret::train(data.frame(X_train), Y_train, method = "gbm", preProcess = NULL, 
# weights = NULL,trControl=trainControl(method="cv",number=5,verboseIter=TRUE,returnData=TRUE,returnResamp="all")
# ,metric = ifelse(is.factor(Y_train), "Accuracy", "RMSE"),
# maximize = ifelse(metric %in% c("RMSE", "logLoss", "MAE"), FALSE,
# TRUE),tuneGrid = NULL)

