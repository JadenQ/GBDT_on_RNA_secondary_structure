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
for(k in 1){
#Fold-1
  X_train<-PARS_human[-holdout[[k]],][,c(-1,-2,-151)]
  #Y_trian=labels
  Y_train<-PARS_human[-holdout[[k]],][,151]

 # Test_data<-PARS_human[holdout[[i]],][,c(-1,-2,-151)]
 Test_data<-PARS_human[holdout[[k]],][,c(-1,-2,-151)]
 Y_test<-PARS_human[holdout[[k]],][,151]
#####################training presettings###################
 trControl<-trainControl(method = "cv",number =3,verboseIter=TRUE)
 #trControl<-trainControl(method="none",verboseIter=TRUE)
# gbmGrid <-  expand.grid(interaction.depth = c(1, 5, 9), 
#                         n.trees = c(50,100,400,800,1000,1500), 
#                         shrinkage = c(0.03,0.1),
#                         n.minobsinnode = c(10,20))
gbmGrid <-  expand.grid(n.trees = 100, 
                        interaction.depth = 3, 
                        shrinkage = 0.1,
                        n.minobsinnode = 10)

xgbTreeGrid<- expand.grid(nrounds=500,max_depth=c(4,5),eta=c(0.1),
                          gamma=c(0.5,1),colsample_bytree=c(0.8),min_child_weight=c(0.05),
                          subsample=c(0.7))#0.81
#xgbLinearGrid<-expand.grid(nrounds=c(800,1000,1200,1500),lambda=c(0.8,1,2,2.5),alpha=c(0.8,1,2,2.5),eta=c(0.04,0.1))
#xgbLinearGrid<-expand.grid(nrounds=c(600),lambda=c(3),alpha=c(3),eta=c(0.08))   #best for ph

xgbDARTGrid<-expand.grid(nrounds=500,max_depth=5,eta=0.04,
                      gamma=3,subsample=0.7,colsample_bytree=0.8,
                      rate_drop=0.3,skip_drop=0.5,min_child_weight=2)
glmnetGrid<-expand.grid(alpha=3,lambda=3)
avNNetGrid<-expand.grid(size=2,decay=0.7,bag=TRUE)
pcrGrid<-expand.grid(ncomp=130)

   parameters <- list(eta = 0.04, maxDepth = 2, lambda = 0.5, gamma = 0.3
          ,subsample=0.7,verbosity=2) 
  # parametersL<-list(lambda=5,alpha=5)
  #  xgboostModel <- xgboost(data = as.matrix(X_train), booster = "gbtree", 
  #                         label = as.matrix(Y_train), params = parameters, nthread = 2, nrounds =5, 
  #                         objective = "binary:logitraw",verbose=2)
#################end of training presetting########################

############Feature selection!##############
# xgboostModel[["feature_names"]]<-c(1:148)
# for(i in 1:length(xgboostModel[["feature_names"]])){xgboostModel[["feature_names"]][i]<-str_glue("V{i}")}
X_train_M<-as.matrix(X_train)
Test_data_M<-as.matrix(Test_data)
colnames(X_train_M)<-c(1:148)
colnames(Test_data_M)<-c(1:148)
for(i in 1:ncol(X_train_M)){colnames(X_train_M)[i]<-str_glue("V{i}")}
for(i in 1:ncol(Test_data_M)){colnames(Test_data_M)[i]<-str_glue("V{i}")}

#pre-settings
xgboostRround<-20
xgbL1Rround<-200
xgbL2Rround<-400

############old feature###########
dtrain <- xgb.DMatrix(data = X_train_M, label = Y_train)
dtest <- xgb.DMatrix(data = Test_data_M, label = Y_test)
bst1 = xgb.train(params = parameters, data = dtrain, nrounds = xgboostRround, nthread = 2,verbose=2)

###########new feature#############
new.features.train <- xgb.create.features(model = bst1, X_train_M)
new.features.test <- xgb.create.features(model = bst1, Test_data_M)

for(i in 1:dim(new.features.train)[2]){new.features.train@Dimnames[[2]][i]<-str_glue("V{i}")}
for(i in 1:dim(new.features.test)[2]){new.features.test@Dimnames[[2]][i]<-str_glue("V{i}")}

new.dtrain <- xgb.DMatrix(data = new.features.train, label = Y_train)
new.dtest <- xgb.DMatrix(data = new.features.test, label = Y_test)

###########use new feature by XGBOOST#########
# parameters2 <- list(eta = 0.05, maxDepth = 5, lambda = 0.5, gamma = 0.3
#            ,subsample=0.7,verbosity=2) 
# bst <- xgb.train(params = parameters2, data = new.dtrain, nrounds = xgboostRround, nthread = 2,verbose=2)
# bst[["feature_names"]]<-new.features.test@Dimnames[[2]]
# PARS_human_result1<-predict(bst, new.dtest)
#PARS_human_result1<-predict(bst1, Test_data_M)
# bst1[["feature_names"]]<-new.features.test@Dimnames[[2]]
# PARS_human_result1<-predict(bst1, new.dtest)

###########use new feature by XGBLinear for feature selection############
######not sparse matrix#####
#  TrainNewf<-as.matrix(new.features.train)
#  TestNewf<-as.matrix(new.features.test)
#  xgbLinearGrid<-expand.grid(nrounds=xgbL1Rround,lambda=c(5),alpha=c(5),
#                            eta=c(0.03))   #best for ph
#  gbm2=caret::train(TrainNewf, Y_train, method = "xgbLinear", preProcess = NULL, 
# weights = NULL,trControl=trControl,metric = "RMSE",tuneGrid=xgbLinearGrid)
#  #select features importance larger than 0.0003____overall:0.9
#  gbm2Imp<-varImp(gbm2,scale=F)
#  GoodFeatures<-rownames(gbm2Imp$importance)[which(gbm2Imp$importance>0.0003)]  #0.0004:97% 0.0003:98%

####$$$$#####sparse matrix#######$$$$#######

#$$$$$$$$$$use xgboost

 # new.dtrain<-xgb.DMatrix(data=new.features.train,label=Y_train)
 # new.dtest<-xgb.DMatrix(data=new.features.test,label=Y_test)
 # parametersT<-list(eta = 0.04, maxDepth = 2, lambda = 0.5, gamma = 0.3
 #          ,subsample=0.7,verbosity=2) 
 # bstTree<-xgb.train(params=parametersT,data=new.dtrain,nrounds=xgbL1Rround,nthred=2,verbose=2)

#$$$$$$$$$$$use caret: problematic in allocating memories

xgbLinearGrid<-expand.grid(nrounds=xgbL1Rround,lambda=c(5),alpha=c(5),
                           eta=c(0.03))   #best for ph
gbm2=caret::train(new.features.train, Y_train, method = "xgbLinear", preProcess = NULL, 
weights = NULL,trControl=trControl,metric = "RMSE",tuneGrid=xgbLinearGrid)
 #select features importance larger than 0.0003____overall:0.9
 gbm2Imp<-varImp(gbm2,scale=F)
 GoodFeatures<-rownames(gbm2Imp$importance)[which(gbm2Imp$importance>0.0003)]  #0.0004:97% 0.0003:98%
 
 TrainNewf<-as.matrix(new.features.train)
 TestNewf<-as.matrix(new.features.test)

#USE selected NEW FEATURE AND NEW gbm (good result but not fabulous)
          ##select features importance larger than 0.0003____overall:0.9

 xgbLinearGrid2<-expand.grid(nrounds=xgbL2Rround,lambda=c(1),alpha=c(1),
                           eta=c(0.03))   #best for ph
 gbm2=caret::train(TrainNewf[,GoodFeatures], Y_train, method = "xgbLinear", preProcess = NULL, 
weights = NULL,trControl=trControl,metric = "RMSE",tuneGrid=xgbLinearGrid2)

PARS_human_result1<-predict(gbm2,TestNewf[,GoodFeatures])

# USE OLD FEATURE AND XGBOOST(not good of course) 
# PARS_human_result1<-predict(bst1,Test_data_M)
#######################USE Logistic regression ? CLASSIFICATION###########################

  # glmmodel2<-glm(Y_train~TrainNewf[,GoodFeatures],method = "glm.fit")
  # PARS_human_result1<-predict.glm(glmmodel2,newdata=as.data.frame(TestNewf[,GoodFeatures]))
 

                      #################select features importance larger than 0.002____overall:0.9#####################
                      # names <- dimnames(data.matrix(new.features.test@Dimnames[[2]][c(1:148)]))[[2]]
                      # originalNames<-colnames(X_train_M)

                      # importance_matrix <- xgb.importance(originalNames,model=xgboostModel) # importance of predictors
                      # xgb.ggplot.importance(importance_matrix[,])
                      # GoodFeatures<-importance$Feature[which(importance$Gain>0.02)]
                      ######################################################################################


# }
                      ###########use the feature on XGBOOST###############
                      # parameters2 <- list(eta = 0.05, maxDepth = 5, lambda = 0.5, gamma = 0.3
                      #           ,subsample=0.7,verbosity=2) 
                      #   bst <- xgb.train(params = parameters2, data = new.dtrain, nrounds = 300, nthread = 2,verbose=2)
                      #   bst[["feature_names"]]<-c(1:dim(new.features.train)[2])
                      #   for(i in 1:length(bst[["feature_names"]])){bst[["feature_names"]][i]<-str_glue("V{i}")}
                      #   PARS_human_result1<- predict(bst, new.dtest)

                      ##################use gblinear on reduced features###################
                        
                        # gbm2=caret::train(X_train_M[,GoodFeatures], Y_train, method = "xgbLinear", preProcess = NULL, 
                        # weights = NULL,trControl=trControl,metric = "RMSE",tuneGrid=xgbLinearGrid)
                        # PARS_human_result1<-predict(gbm2,data.frame(Test_data_M[,GoodFeatures]))


 #PARS_human_result1<-predict(bst,new.dtest)     #use xgboost again

# data1<-data.frame(Y_train,X_train)
# X_train_f<-data.frame(matrix(unlist(X_train),ncol = 148))
  # gbm_test=gbm(formula = Y_train~X_train_f, distribution = "bernoulli",
  # data = data1, var.monotone = NULL, n.trees = 500,
  # interaction.depth = 3, n.minobsinnode = 10, shrinkage = 0.1,
  # bag.fraction = 0.5, train.fraction = 1, cv.folds = 0,
  # keep.data = TRUE, verbose = TRUE, class.stratify.cv = NULL,
  # n.cores = 2)

  
  #  gbm2=caret::train(data.frame(X_train), Y_train, method = "xgbLinear", preProcess = NULL, 
  # weights = NULL,trControl=trControl,metric = "RMSE",tuneGrid=xgbLinearGrid)




#   #############
#   #use GBM model to predict
#   #############
  print("Predicting scores...")

 # PARS_human_result1 <- predict(gbm2, data.frame(Test_data))
  PARS_human_result1 <- (PARS_human_result1-min(PARS_human_result1))/(max(PARS_human_result1)-min(PARS_human_result1))
  true_class<-str_glue("Class{PARS_human[holdout[[k]],][,151]}")
  class_1_prob<-PARS_human_result1
  test_set <- data.frame(obs = true_class,
                         Class1 = class_1_prob)
  test_set$Class0 <- 1 - test_set$Class1
  test_set$pred <- factor(ifelse(test_set$Class1 >= .5, "Class1", "Class0"))
  #get the scores
  Confu_matrix<-confusionMatrix(data = test_set$pred, reference = test_set$obs, mode = "prec_recall")
  ACC<- Confu_matrix$overall["Accuracy"]
  unitResult<-prSummary(test_set, lev = levels(test_set$obs))
  Result[k,]<-t(as.matrix(c(unitResult,ACC)))


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
# Test_data_label<-PARS_human[holdout[[k]],][,c(151)]
# RFmodel<-randomForest(X_train,y=Y_train,xtest=Test_data,ytest=Test_data_label,
#   ntree=50,mtry=50)





}
  





#write()



  # gbm1=caret::train(data.frame(X_train), Y_train, method = "gbm", preProcess = NULL, 
  # weights = NULL,trControl=trainControl(method="cv",number=5,verboseIter=TRUE,returnData=TRUE,returnResamp="all")
  # ,metric = ifelse(is.factor(Y_train), "Accuracy", "RMSE"),
  # maximize = ifelse(metric %in% c("RMSE", "logLoss", "MAE"), FALSE,
  # TRUE),tuneGrid = NULL)

